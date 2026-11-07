"""Unit tests for LangGraph routing with fake GitHub and OpenCode gateways."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from super_orchestrator.github import GitHubClient
from super_orchestrator.graph import (
    OrchestratorHalt,
    create_orchestrator_graph,
    initial_state,
    run_orchestrator,
)
from super_orchestrator.opencode import ImplementResult, OpenCodeClient, ReviewResult
from super_orchestrator.state import IssueInfo


@dataclass
class FakeGitHub:
    """In-memory stand-in for GitHubGateway used by graph nodes."""

    issues: list[IssueInfo]
    open_prs: dict[int, str] = field(default_factory=dict)
    merged_prs: set[str] = field(default_factory=set)
    closed: list[int] = field(default_factory=list)
    repo: str = "acme/app"
    skip_label: str = "orchestrator:skip"

    def list_open_issues(self) -> list[IssueInfo]:
        return list(self.issues)

    def find_open_orchestrator_pr(self, issue_number: int) -> str | None:
        return self.open_prs.get(issue_number)

    def is_pr_merged(self, pr_url: str) -> bool:
        return pr_url in self.merged_prs

    def close_issue(self, issue_number: int, *, comment: str) -> None:
        del comment
        self.closed.append(issue_number)
        self.issues = [issue for issue in self.issues if issue["number"] != issue_number]


@dataclass
class ScriptedOpenCode:
    """
    Fake OpenCodeGateway that returns scripted implement/review results in order.

    Each call pops the next scripted result from the corresponding list.
    """

    implement_results: list[ImplementResult]
    review_results: list[ReviewResult]
    implement_calls: list[dict[str, object]] = field(default_factory=list)
    review_calls: list[dict[str, object]] = field(default_factory=list)
    workdir: Path = field(default_factory=Path)

    def run_implement(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "issue-worker",
    ) -> ImplementResult:
        self.implement_calls.append(
            {
                "issue_number": issue_number,
                "attempt": attempt,
                "message": message,
                "session_id": session_id,
                "agent": agent,
            }
        )
        if len(self.implement_results) == 0:
            raise AssertionError("unexpected implement call")
        return self.implement_results.pop(0)

    def run_review(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "pr-reviewer",
    ) -> ReviewResult:
        self.review_calls.append(
            {
                "issue_number": issue_number,
                "attempt": attempt,
                "message": message,
                "session_id": session_id,
                "agent": agent,
            }
        )
        if len(self.review_results) == 0:
            raise AssertionError("unexpected review call")
        return self.review_results.pop(0)


def _issue(number: int, title: str = "t") -> IssueInfo:
    return IssueInfo(
        number=number,
        title=title,
        body=f"body-{number}",
        url=f"https://github.com/acme/app/issues/{number}",
    )


def _implement_ok(pr_url: str, *, session_id: str = "ses_1") -> ImplementResult:
    return ImplementResult(
        exit_code=0,
        session_id=session_id,
        pr_url=pr_url,
        status="PR_READY",
        assistant_text=f"ISSUE_STATUS: PR_READY\nPR_URL: {pr_url}",
        log_path=Path("/tmp/implement.log"),
    )


def _review_approve() -> ReviewResult:
    return ReviewResult(
        exit_code=0,
        session_id="ses_r",
        verdict="APPROVE",
        feedback="merged",
        assistant_text="REVIEW_VERDICT: APPROVE\nmerged",
        log_path=Path("/tmp/review.log"),
    )


def _review_changes(feedback: str = "fix tests") -> ReviewResult:
    return ReviewResult(
        exit_code=0,
        session_id="ses_r",
        verdict="REQUEST_CHANGES",
        feedback=feedback,
        assistant_text=f"REVIEW_VERDICT: REQUEST_CHANGES\n{feedback}",
        log_path=Path("/tmp/review.log"),
    )


def test_happy_path_two_issues(tmp_path: Path) -> None:
    pr1 = "https://github.com/acme/app/pull/1"
    pr2 = "https://github.com/acme/app/pull/2"
    github = FakeGitHub(issues=[_issue(1), _issue(2)], merged_prs={pr1, pr2})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr1), _implement_ok(pr2, session_id="ses_2")],
        review_results=[_review_approve(), _review_approve()],
        workdir=tmp_path,
    )

    final_state = run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    assert final_state["status"] == "done"
    assert github.closed == [1, 2]
    assert len(opencode.implement_calls) == 2
    assert len(opencode.review_calls) == 2


def test_request_changes_then_approve(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/5"
    github = FakeGitHub(issues=[_issue(5)], merged_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[
            _implement_ok(pr_url, session_id="ses_a"),
            _implement_ok(pr_url, session_id="ses_a"),
        ],
        review_results=[_review_changes("add tests"), _review_approve()],
        workdir=tmp_path,
    )

    final_state = run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        max_rounds=3,
        use_memory_checkpointer=True,
    )
    assert final_state["status"] == "done"
    assert github.closed == [5]
    assert len(opencode.implement_calls) == 2
    assert opencode.implement_calls[1]["attempt"] == 2
    assert opencode.implement_calls[1]["session_id"] == "ses_a"
    assert "add tests" in str(opencode.implement_calls[1]["message"])


def test_max_rounds_halts(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/8"
    github = FakeGitHub(issues=[_issue(8)])
    opencode = ScriptedOpenCode(
        implement_results=[
            _implement_ok(pr_url),
            _implement_ok(pr_url),
        ],
        review_results=[
            _review_changes("round1"),
            _review_changes("round2"),
        ],
        workdir=tmp_path,
    )

    with pytest.raises(OrchestratorHalt, match="max review rounds"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            max_rounds=2,
            use_memory_checkpointer=True,
        )
    assert github.closed == []


def test_existing_open_pr_skips_implement(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/4"
    github = FakeGitHub(
        issues=[_issue(4)],
        open_prs={4: pr_url},
        merged_prs={pr_url},
    )
    opencode = ScriptedOpenCode(
        implement_results=[],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )

    final_state = run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    assert final_state["status"] == "done"
    assert opencode.implement_calls == []
    assert len(opencode.review_calls) == 1
    assert github.closed == [4]


def test_implement_missing_sentinel_halts(tmp_path: Path) -> None:
    github = FakeGitHub(issues=[_issue(9)])
    opencode = ScriptedOpenCode(
        implement_results=[
            ImplementResult(
                exit_code=0,
                session_id="ses",
                pr_url=None,
                status=None,
                assistant_text="I forgot the sentinel",
                log_path=Path("/tmp/x.log"),
            )
        ],
        review_results=[],
        workdir=tmp_path,
    )

    with pytest.raises(OrchestratorHalt, match="ISSUE_STATUS"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            use_memory_checkpointer=True,
        )


def test_empty_issue_queue_is_done(tmp_path: Path) -> None:
    github = FakeGitHub(issues=[])
    opencode = ScriptedOpenCode(
        implement_results=[],
        review_results=[],
        workdir=tmp_path,
    )
    final_state = run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    assert final_state["status"] == "done"
    assert final_state["issues"] == []


def test_initial_state_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        initial_state(
            repo="acme/app",
            workdir=tmp_path,
            merge_method="squash",
            max_rounds=0,
            skip_label="orchestrator:skip",
        )


def test_create_graph_compiles(tmp_path: Path) -> None:
    """Smoke-test that the graph compiles against real client types."""

    class UnusedGh:
        def run_json(self, args: list[str], *, cwd: Path | None = None) -> object:
            del args, cwd
            raise AssertionError("not used")

        def run_text(self, args: list[str], *, cwd: Path | None = None) -> str:
            del args, cwd
            raise AssertionError("not used")

    class NoopRunner:
        def run(
            self,
            command: object,
            *,
            cwd: Path,
            env: object,
            log_path: Path,
            on_stdout_line: object = None,
        ) -> int:
            del command, cwd, env, log_path, on_stdout_line
            return 0

    github = GitHubClient(repo="acme/app", workdir=tmp_path, runner=UnusedGh())
    opencode = OpenCodeClient(workdir=tmp_path, runner=NoopRunner())
    compiled = create_orchestrator_graph(github=github, opencode=opencode)
    assert compiled is not None
