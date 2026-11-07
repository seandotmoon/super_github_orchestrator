"""GitHub helpers via the `gh` CLI."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from super_orchestrator.state import IssueInfo


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
        """Return open issues ordered for processing."""

    def find_open_orchestrator_pr(self, issue_number: int) -> str | None:
        """Return an open orchestrator PR URL for the issue, if any."""

    def is_pr_merged(self, pr_url: str) -> bool:
        """Return True when the PR is merged."""

    def close_issue(self, issue_number: int, *, comment: str) -> None:
        """Close the issue with a comment."""


@dataclass(frozen=True)
class SubprocessGhRunner:
    """Production GhRunner that shells out to the `gh` binary."""

    gh_bin: str = "gh"

    def run_json(self, args: list[str], *, cwd: Path | None = None) -> object:
        """Run `gh` and parse JSON stdout."""
        stdout = self.run_text(args, cwd=cwd)
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"gh returned invalid JSON: {exc}: {stdout[:500]}") from exc

    def run_text(self, args: list[str], *, cwd: Path | None = None) -> str:
        """Run `gh` and return stripped stdout; raise GitHubError on failure."""
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
            )
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
    ) -> None:
        if not isinstance(repo, str) or "/" not in repo:
            raise ValueError("repo must look like 'owner/name'")
        if not isinstance(workdir, Path):
            raise TypeError("workdir must be a Path")
        if not isinstance(skip_label, str) or skip_label.strip() == "":
            raise ValueError("skip_label must be a non-empty str")
        self._repo = repo
        self._workdir = workdir
        self._runner: GhRunner = runner if runner is not None else SubprocessGhRunner()
        self._skip_label = skip_label

    @property
    def repo(self) -> str:
        """Return the configured owner/name repository slug."""
        return self._repo

    @property
    def skip_label(self) -> str:
        """Return the label that causes an issue to be skipped."""
        return self._skip_label

    def list_open_issues(self) -> list[IssueInfo]:
        """
        List open issues ordered by ascending number, excluding skip-label issues.

        Pull requests are excluded via `gh issue list` (issues only).
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
                "1000",
                "--json",
                "number,title,body,url,labels",
            ],
            cwd=self._workdir,
        )
        if not isinstance(raw, list):
            raise GitHubError("gh issue list returned a non-list JSON value")

        issues: list[IssueInfo] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GitHubError("gh issue list item was not an object")
            if self._has_skip_label(item):
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
            issues.append(
                IssueInfo(
                    number=number,
                    title=title,
                    body=body_text,
                    url=url,
                )
            )

        issues.sort(key=lambda issue: issue["number"])
        return issues

    def find_open_orchestrator_pr(self, issue_number: int) -> str | None:
        """
        Find an open PR whose head branch matches `issue-<number>-...`.

        Returns:
            The PR HTML URL, or None when no matching open PR exists.
        """
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("issue_number must be a positive int")

        raw = self._runner.run_json(
            [
                "pr",
                "list",
                "--repo",
                self._repo,
                "--state",
                "open",
                "--limit",
                "100",
                "--json",
                "url,headRefName,number,title",
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
            match = _ORCHESTRATOR_BRANCH_RE.match(head)
            if match is not None and int(match.group(1)) == issue_number:
                return url
        return None

    def is_pr_merged(self, pr_url: str) -> bool:
        """
        Return True when the PR at pr_url is merged.

        Uses `gh pr view --json state,mergedAt`.
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
        if isinstance(state, str) and state.upper() == "MERGED":
            return True
        if isinstance(merged_at, str) and merged_at.strip() != "":
            return True
        return False

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

    def _has_skip_label(self, item: dict[str, object]) -> bool:
        """Return True when the issue carries the configured skip label."""
        labels = item.get("labels")
        if not isinstance(labels, list):
            return False
        for label in labels:
            if isinstance(label, dict):
                name = label.get("name")
                if isinstance(name, str) and name == self._skip_label:
                    return True
            elif isinstance(label, str) and label == self._skip_label:
                return True
        return False
