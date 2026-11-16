"""GitHub helpers via the `gh` CLI."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from super_orchestrator.state import IssueInfo

PrState = Literal["OPEN", "CLOSED", "MERGED"]

# Upper bound for issue / PR listing. Large enough that real queues are not truncated.
_LIST_LIMIT = "1000"


class GitHubError(RuntimeError):
    """Raised when a `gh` invocation fails or returns unexpected data."""


class GhRunner(Protocol):
    """Abstraction over `gh` command execution for unit tests."""

    def run_json(self, args: list[str], *, cwd: Path | None = None) -> object:
        """Run `gh` with args and parse stdout as JSON."""

    def run_text(self, args: list[str], *, cwd: Path | None = None) -> str:
        """Run `gh` with args and return stdout as text."""


class GitHubGateway(Protocol):
    """Subset of GitHub operations required by the LangGraph control loop."""

    def list_open_issues(self) -> list[IssueInfo]:
        """Return open, eligible issues ordered for processing."""

    def find_open_orchestrator_pr(self, issue_number: int) -> str | None:
        """Return an open orchestrator PR URL for the issue, if any."""

    def get_pr_state(self, pr_url: str) -> PrState:
        """Return OPEN, CLOSED, or MERGED for the PR."""

    def close_issue(self, issue_number: int, *, comment: str) -> None:
        """Close the issue with a comment."""

    def mark_issue_failed(self, issue_number: int, *, label: str, comment: str) -> None:
        """Label and comment on an issue the orchestrator gave up on."""


@dataclass(frozen=True)
class SubprocessGhRunner:
    """Production GhRunner that shells out to the `gh` binary."""

    gh_bin: str = "gh"
    timeout_seconds: float = 120.0

    def run_json(self, args: list[str], *, cwd: Path | None = None) -> object:
        """Run `gh` and parse JSON stdout."""
        stdout = self.run_text(args, cwd=cwd)
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"gh returned invalid JSON: {exc}: {stdout[:500]}") from exc

    def run_text(self, args: list[str], *, cwd: Path | None = None) -> str:
        """Run `gh` and return stripped stdout; raise GitHubError on failure or timeout."""
        if shutil.which(self.gh_bin) is None:
            raise GitHubError(f"GitHub CLI binary {self.gh_bin!r} was not found on PATH")
        command = [self.gh_bin, *args]
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd is not None else None,
                check=False,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubError(
                f"gh {' '.join(args)} timed out after {self.timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise GitHubError(f"Failed to start gh: {exc}") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise GitHubError(
                f"gh {' '.join(args)} failed ({completed.returncode}): {stderr}"
            )
        return completed.stdout.strip()


# Branch naming convention used by the ship-github-issue skill.
_ORCHESTRATOR_BRANCH_RE = re.compile(r"^issue-(\d+)(?:-|$)", re.IGNORECASE)


def _label_names(item: dict[str, object]) -> set[str]:
    """Return the label names on a `gh` issue JSON object."""
    labels = item.get("labels")
    names: set[str] = set()
    if not isinstance(labels, list):
        return names
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str):
                names.add(name)
        elif isinstance(label, str):
            names.add(label)
    return names


def _author_login(item: dict[str, object]) -> str | None:
    """Return the lowercase author login from a `gh` issue/PR JSON object."""
    author = item.get("author")
    if isinstance(author, dict):
        login = author.get("login")
        if isinstance(login, str) and login.strip() != "":
            return login.strip().lower()
    return None


class GitHubClient:
    """
    Thin wrapper around `gh` for the orchestrator control loop.

    Issue listing, PR discovery, merge-state checks, and issue close live here.
    Merge and request-changes themselves are performed by the OpenCode review agent.
    """

    def __init__(
        self,
        *,
        repo: str,
        workdir: Path,
        runner: GhRunner | None = None,
        skip_label: str = "orchestrator:skip",
        allowed_authors: Iterable[str] | None = None,
        require_label: str | None = None,
    ) -> None:
        """
        Args:
            repo: owner/name slug.
            workdir: Local checkout used as cwd for `gh`.
            runner: Injectable `gh` runner (tests).
            skip_label: Issues carrying this label are never processed.
            allowed_authors: When set, only issues opened by these logins are
                processed. None means any author (unsafe for public repos).
            require_label: When set, only issues carrying this label are processed.
        """
        if not isinstance(repo, str) or "/" not in repo:
            raise ValueError("repo must look like 'owner/name'")
        if not isinstance(workdir, Path):
            raise TypeError("workdir must be a Path")
        if not isinstance(skip_label, str) or skip_label.strip() == "":
            raise ValueError("skip_label must be a non-empty str")
        if require_label is not None and (
            not isinstance(require_label, str) or require_label.strip() == ""
        ):
            raise ValueError("require_label must be a non-empty str when provided")
        normalized_authors: frozenset[str] | None = None
        if allowed_authors is not None:
            authors: set[str] = set()
            for login in allowed_authors:
                if not isinstance(login, str) or login.strip() == "":
                    raise ValueError("allowed_authors entries must be non-empty str")
                authors.add(login.strip().lower())
            if len(authors) == 0:
                raise ValueError("allowed_authors must not be empty when provided")
            normalized_authors = frozenset(authors)

        self._repo = repo
        self._workdir = workdir
        self._runner: GhRunner = runner if runner is not None else SubprocessGhRunner()
        self._skip_label = skip_label
        self._allowed_authors = normalized_authors
        self._require_label = require_label
        self._viewer_login: str | None = None

    @property
    def repo(self) -> str:
        """Return the configured owner/name repository slug."""
        return self._repo

    @property
    def skip_label(self) -> str:
        """Return the label that causes an issue to be skipped."""
        return self._skip_label

    @property
    def allowed_authors(self) -> frozenset[str] | None:
        """Return the normalized author allowlist, or None for any author."""
        return self._allowed_authors

    def viewer_login(self) -> str:
        """
        Return the login `gh` is authenticated as (cached).

        The issue-worker opens PRs with the same credentials, so this is also
        the expected author of orchestrator PRs.
        """
        if self._viewer_login is not None:
            return self._viewer_login
        login = self._runner.run_text(["api", "user", "--jq", ".login"], cwd=self._workdir)
        login = login.strip()
        if login == "":
            raise GitHubError("gh api user returned an empty login")
        self._viewer_login = login
        return login

    def list_open_issues(self) -> list[IssueInfo]:
        """
        List eligible open issues ordered by ascending number.

        Excludes skip-label issues, issues missing the required label, and
        issues from authors outside the allowlist. Pull requests are excluded
        because `gh issue list` returns issues only.
        """
        raw = self._runner.run_json(
            [
                "issue",
                "list",
                "--repo",
                self._repo,
                "--state",
                "open",
                "--limit",
                _LIST_LIMIT,
                "--json",
                "number,title,body,url,labels,author",
            ],
            cwd=self._workdir,
        )
        if not isinstance(raw, list):
            raise GitHubError("gh issue list returned a non-list JSON value")

        issues: list[IssueInfo] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GitHubError("gh issue list item was not an object")
            if not self._is_eligible(item):
                continue
            number = item.get("number")
            title = item.get("title")
            body = item.get("body")
            url = item.get("url")
            if not isinstance(number, int):
                raise GitHubError(f"issue number missing or invalid: {item!r}")
            if not isinstance(title, str):
                raise GitHubError(f"issue {number} title missing or invalid")
            if body is None:
                body_text = ""
            elif isinstance(body, str):
                body_text = body
            else:
                raise GitHubError(f"issue {number} body must be a string or null")
            if not isinstance(url, str) or url.strip() == "":
                raise GitHubError(f"issue {number} url missing or invalid")
            issues.append(IssueInfo(number=number, title=title, body=body_text, url=url))

        issues.sort(key=lambda issue: issue["number"])
        return issues

    def find_open_orchestrator_pr(self, issue_number: int) -> str | None:
        """
        Find an open orchestrator PR for the issue.

        A PR qualifies when its head branch matches `issue-<number>-...`, it is
        not from a fork, and it was opened by the authenticated `gh` user (the
        identity the issue-worker uses). This ignores look-alike human branches.

        Returns:
            The PR HTML URL, or None when no matching open PR exists.
        """
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("issue_number must be a positive int")

        viewer = self.viewer_login().lower()
        raw = self._runner.run_json(
            [
                "pr",
                "list",
                "--repo",
                self._repo,
                "--state",
                "open",
                "--limit",
                _LIST_LIMIT,
                "--json",
                "url,headRefName,number,title,author,isCrossRepository",
            ],
            cwd=self._workdir,
        )
        if not isinstance(raw, list):
            raise GitHubError("gh pr list returned a non-list JSON value")

        for item in raw:
            if not isinstance(item, dict):
                continue
            head = item.get("headRefName")
            url = item.get("url")
            if not isinstance(head, str) or not isinstance(url, str):
                continue
            if item.get("isCrossRepository") is True:
                continue
            if _author_login(item) != viewer:
                continue
            match = _ORCHESTRATOR_BRANCH_RE.match(head)
            if match is not None and int(match.group(1)) == issue_number:
                return url
        return None

    def get_pr_state(self, pr_url: str) -> PrState:
        """
        Return the PR state via `gh pr view --json state,mergedAt`.

        A non-empty mergedAt is treated as MERGED even if state lags behind.
        """
        if not isinstance(pr_url, str) or pr_url.strip() == "":
            raise ValueError("pr_url must be a non-empty str")

        raw = self._runner.run_json(
            ["pr", "view", pr_url, "--json", "state,mergedAt"],
            cwd=self._workdir,
        )
        if not isinstance(raw, dict):
            raise GitHubError("gh pr view returned a non-object JSON value")

        state = raw.get("state")
        merged_at = raw.get("mergedAt")
        if isinstance(merged_at, str) and merged_at.strip() != "":
            return "MERGED"
        if not isinstance(state, str):
            raise GitHubError(f"gh pr view returned no state for {pr_url}")
        normalized = state.upper()
        if normalized == "MERGED":
            return "MERGED"
        if normalized == "CLOSED":
            return "CLOSED"
        if normalized == "OPEN":
            return "OPEN"
        raise GitHubError(f"unexpected PR state {state!r} for {pr_url}")

    def is_pr_merged(self, pr_url: str) -> bool:
        """Return True when the PR at pr_url is merged."""
        return self.get_pr_state(pr_url) == "MERGED"

    def close_issue(self, issue_number: int, *, comment: str) -> None:
        """
        Close an issue with a comment pointing at the merged work.

        Args:
            issue_number: Issue to close.
            comment: Closing comment body.
        """
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("issue_number must be a positive int")
        if not isinstance(comment, str) or comment.strip() == "":
            raise ValueError("comment must be a non-empty str")

        self._runner.run_text(
            [
                "issue",
                "close",
                str(issue_number),
                "--repo",
                self._repo,
                "--comment",
                comment,
            ],
            cwd=self._workdir,
        )

    def ensure_label(self, label: str, *, description: str) -> None:
        """Create (or update) a repository label so it can be applied to issues."""
        if not isinstance(label, str) or label.strip() == "":
            raise ValueError("label must be a non-empty str")
        self._runner.run_text(
            [
                "label",
                "create",
                label,
                "--repo",
                self._repo,
                "--description",
                description,
                "--force",
            ],
            cwd=self._workdir,
        )

    def mark_issue_failed(self, issue_number: int, *, label: str, comment: str) -> None:
        """
        Label an issue the orchestrator gave up on and explain why.

        Applying the skip label keeps later runs from retrying it until a human
        removes the label.
        """
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("issue_number must be a positive int")
        if not isinstance(label, str) or label.strip() == "":
            raise ValueError("label must be a non-empty str")
        if not isinstance(comment, str) or comment.strip() == "":
            raise ValueError("comment must be a non-empty str")
        self._runner.run_text(
            ["issue", "edit", str(issue_number), "--repo", self._repo, "--add-label", label],
            cwd=self._workdir,
        )
        self._runner.run_text(
            ["issue", "comment", str(issue_number), "--repo", self._repo, "--body", comment],
            cwd=self._workdir,
        )

    def resolve_remote_slug(self) -> str:
        """
        Resolve the `origin` remote to an owner/name slug for preflight checks.

        Raises:
            GitHubError: When the remote cannot be resolved.
        """
        raw = self._runner.run_json(
            ["repo", "view", "--json", "nameWithOwner"],
            cwd=self._workdir,
        )
        if not isinstance(raw, dict):
            raise GitHubError("gh repo view returned a non-object JSON value")
        name = raw.get("nameWithOwner")
        if not isinstance(name, str) or "/" not in name:
            raise GitHubError(f"unexpected nameWithOwner value: {name!r}")
        return name

    def _is_eligible(self, item: dict[str, object]) -> bool:
        """Apply skip-label, required-label, and author-allowlist filters."""
        labels = _label_names(item)
        if self._skip_label in labels:
            return False
        if self._require_label is not None and self._require_label not in labels:
            return False
        if self._allowed_authors is not None:
            login = _author_login(item)
            if login is None or login not in self._allowed_authors:
                return False
        return True
