"""Unit tests for GitHub issue filtering, PR discovery, and merge state."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from super_orchestrator.github import GitHubClient, GitHubError, SubprocessGhRunner


class FakeGhRunner:
    """Scripted GhRunner keyed by a simplified command signature."""

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    def _key(self, args: list[str]) -> str:
        return " ".join(args[:3])

    def run_json(self, args: list[str], *, cwd: Path | None = None) -> object:
        del cwd
        self.calls.append(list(args))
        key = self._key(args)
        if key not in self.responses:
            raise GitHubError(f"no fake response for {key!r}")
        return self.responses[key]

    def run_text(self, args: list[str], *, cwd: Path | None = None) -> str:
        del cwd
        self.calls.append(list(args))
        key = self._key(args)
        value = self.responses.get(key, "")
        if isinstance(value, str):
            return value
        return ""


def _issue_json(
    number: int, *, labels: list[str] | None = None, author: str = "alice"
) -> dict[str, object]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "body": f"b{number}",
        "url": f"https://github.com/acme/app/issues/{number}",
        "labels": [{"name": name} for name in (labels or [])],
        "author": {"login": author},
    }


def test_list_open_issues_orders_by_number_and_skips_label(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "issue list --repo": [
                _issue_json(5),
                _issue_json(2, labels=["orchestrator:skip"]),
                _issue_json(3, labels=["bug"]),
            ]
        }
    )
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    issues = client.list_open_issues()
    assert [issue["number"] for issue in issues] == [3, 5]
    assert issues[0]["title"] == "Issue 3"
    assert "author" in runner.calls[0][-1]


def test_list_open_issues_filters_by_author_allowlist(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "issue list --repo": [
                _issue_json(1, author="Alice"),
                _issue_json(2, author="mallory"),
                {**_issue_json(3), "author": None},
            ]
        }
    )
    client = GitHubClient(
        repo="acme/app", workdir=tmp_path, runner=runner, allowed_authors=["alice"]
    )
    assert [issue["number"] for issue in client.list_open_issues()] == [1]


def test_list_open_issues_requires_label(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "issue list --repo": [
                _issue_json(1, labels=["orchestrate"]),
                _issue_json(2, labels=["bug"]),
            ]
        }
    )
    client = GitHubClient(
        repo="acme/app", workdir=tmp_path, runner=runner, require_label="orchestrate"
    )
    assert [issue["number"] for issue in client.list_open_issues()] == [1]


def test_allowed_authors_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        GitHubClient(repo="acme/app", workdir=tmp_path, allowed_authors=[])
    with pytest.raises(ValueError):
        GitHubClient(repo="acme/app", workdir=tmp_path, allowed_authors=[" "])


def test_find_open_orchestrator_pr_requires_own_same_repo_pr(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "api user --jq": "bot-user",
            "pr list --repo": [
                {
                    "url": "https://github.com/acme/app/pull/9",
                    "headRefName": "feature/unrelated",
                    "author": {"login": "bot-user"},
                    "isCrossRepository": False,
                },
                {
                    "url": "https://github.com/acme/app/pull/11",
                    "headRefName": "issue-3-human-branch",
                    "author": {"login": "someone-else"},
                    "isCrossRepository": False,
                },
                {
                    "url": "https://github.com/acme/app/pull/12",
                    "headRefName": "issue-3-from-fork",
                    "author": {"login": "bot-user"},
                    "isCrossRepository": True,
                },
                {
                    "url": "https://github.com/acme/app/pull/10",
                    "headRefName": "issue-3-add-login",
                    "author": {"login": "Bot-User"},
                    "isCrossRepository": False,
                },
            ],
        }
    )
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    assert client.find_open_orchestrator_pr(3) == "https://github.com/acme/app/pull/10"
    assert client.find_open_orchestrator_pr(99) is None
    pr_list_call = next(call for call in runner.calls if call[:2] == ["pr", "list"])
    assert pr_list_call[pr_list_call.index("--limit") + 1] == "1000"
    # viewer login is cached across calls
    assert sum(1 for call in runner.calls if call[:2] == ["api", "user"]) == 1


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"state": "MERGED", "mergedAt": "2026-01-01T00:00:00Z"}, "MERGED"),
        ({"state": "OPEN", "mergedAt": "2026-01-01T00:00:00Z"}, "MERGED"),
        ({"state": "OPEN", "mergedAt": None}, "OPEN"),
        ({"state": "CLOSED", "mergedAt": None}, "CLOSED"),
    ],
)
def test_get_pr_state(tmp_path: Path, payload: dict[str, object], expected: str) -> None:
    runner = FakeGhRunner({"pr view https://github.com/acme/app/pull/10": payload})
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    assert client.get_pr_state("https://github.com/acme/app/pull/10") == expected
    assert client.is_pr_merged("https://github.com/acme/app/pull/10") is (expected == "MERGED")


def test_close_issue_invokes_gh(tmp_path: Path) -> None:
    runner = FakeGhRunner({"issue close 3": ""})
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    client.close_issue(3, comment="Closed by super-orchestrator after merge.")
    assert runner.calls[0][:2] == ["issue", "close"]
    assert "3" in runner.calls[0]
    assert "--comment" in runner.calls[0]


def test_mark_issue_failed_labels_and_comments(tmp_path: Path) -> None:
    runner = FakeGhRunner({})
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    client.mark_issue_failed(4, label="orchestrator:skip", comment="gave up")
    assert runner.calls[0][:3] == ["issue", "edit", "4"]
    assert runner.calls[0][-2:] == ["--add-label", "orchestrator:skip"]
    assert runner.calls[1][:3] == ["issue", "comment", "4"]
    assert runner.calls[1][-1] == "gave up"


def test_repo_validation() -> None:
    with pytest.raises(ValueError):
        GitHubClient(repo="invalid", workdir=Path("."))


def test_subprocess_gh_runner_times_out(tmp_path: Path) -> None:
    script = tmp_path / "slow-gh"
    script.write_text(
        "\n".join([f"#!{sys.executable}", "import time", "time.sleep(30)", ""]),
        encoding="utf-8",
    )
    script.chmod(0o755)
    runner = SubprocessGhRunner(gh_bin=str(script), timeout_seconds=0.5)
    with pytest.raises(GitHubError, match="timed out"):
        runner.run_text(["issue", "list"])
