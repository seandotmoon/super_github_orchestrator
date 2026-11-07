"""Unit tests for GitHub issue ordering and skip-label filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_orchestrator.github import GitHubClient, GitHubError


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


def test_list_open_issues_orders_by_number_and_skips_label(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "issue list --repo": [
                {
                    "number": 5,
                    "title": "Later",
                    "body": "b5",
                    "url": "https://github.com/acme/app/issues/5",
                    "labels": [],
                },
                {
                    "number": 2,
                    "title": "Skip me",
                    "body": "b2",
                    "url": "https://github.com/acme/app/issues/2",
                    "labels": [{"name": "orchestrator:skip"}],
                },
                {
                    "number": 3,
                    "title": "First real",
                    "body": "b3",
                    "url": "https://github.com/acme/app/issues/3",
                    "labels": [{"name": "bug"}],
                },
            ]
        }
    )
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    issues = client.list_open_issues()
    assert [issue["number"] for issue in issues] == [3, 5]
    assert issues[0]["title"] == "First real"


def test_find_open_orchestrator_pr(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "pr list --repo": [
                {
                    "number": 9,
                    "title": "Other",
                    "url": "https://github.com/acme/app/pull/9",
                    "headRefName": "feature/unrelated",
                },
                {
                    "number": 10,
                    "title": "Issue 3",
                    "url": "https://github.com/acme/app/pull/10",
                    "headRefName": "issue-3-add-login",
                },
            ]
        }
    )
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    assert client.find_open_orchestrator_pr(3) == "https://github.com/acme/app/pull/10"
    assert client.find_open_orchestrator_pr(99) is None


def test_is_pr_merged_true_when_merged_at_set(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "pr view https://github.com/acme/app/pull/10": {
                "state": "MERGED",
                "mergedAt": "2026-01-01T00:00:00Z",
            }
        }
    )
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    assert client.is_pr_merged("https://github.com/acme/app/pull/10") is True


def test_is_pr_merged_false_when_open(tmp_path: Path) -> None:
    runner = FakeGhRunner(
        {
            "pr view https://github.com/acme/app/pull/10": {
                "state": "OPEN",
                "mergedAt": None,
            }
        }
    )
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    assert client.is_pr_merged("https://github.com/acme/app/pull/10") is False


def test_close_issue_invokes_gh(tmp_path: Path) -> None:
    runner = FakeGhRunner({"issue close 3": ""})
    client = GitHubClient(repo="acme/app", workdir=tmp_path, runner=runner)
    client.close_issue(3, comment="Closed by super-orchestrator after merge.")
    assert runner.calls[0][:2] == ["issue", "close"]
    assert "3" in runner.calls[0]
    assert "--comment" in runner.calls[0]


def test_repo_validation() -> None:
    with pytest.raises(ValueError):
        GitHubClient(repo="invalid", workdir=Path("."))
