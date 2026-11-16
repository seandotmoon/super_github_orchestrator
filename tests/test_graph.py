"""Unit tests for LangGraph routing with fake GitHub and OpenCode gateways."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from super_orchestrator.github import GitHubClient, GitHubError, PrState
from super_orchestrator.graph import (
    OrchestratorHalt,
    create_orchestrator_graph,
    initial_state,
    run_orchestrator,
)
from super_orchestrator.opencode import ImplementResult, LiveOutput, OpenCodeClient, ReviewResult
from super_orchestrator.state import IssueInfo


@dataclass
class FakeGitHub:
    """In-memory stand-in for GitHubGateway used by graph nodes."""

    issues: list[IssueInfo]
    open_prs: dict[int, str] = field(default_factory=dict)
    merged_prs: set[str] = field(default_factory=set)
    closed_prs: set[str] = field(default_factory=set)
    # PR url -> number of get_pr_state polls before it reports MERGED.
    merge_after_polls: dict[str, int] = field(default_factory=dict)
    closed: list[int] = field(default_factory=list)
    failed: list[tuple[int, str, str]] = field(default_factory=list)
    list_calls: int = 0
    # Issues that appear in the queue after the Nth list call.
    late_issues: dict[int, list[IssueInfo]] = field(default_factory=dict)
    fail_labeling: bool = False
    repo: str = "acme/app"
    skip_label: str = "orchestrator:skip"

    def list_open_issues(self) -> list[IssueInfo]:
        self.list_calls += 1
        self.issues.extend(self.late_issues.pop(self.list_calls, []))
        return sorted(self.issues, key=lambda issue: issue["number"])

    def find_open_orchestrator_pr(self, issue_number: int) -> str | None:
        return self.open_prs.get(issue_number)

    def get_pr_state(self, pr_url: str) -> PrState:
        remaining = self.merge_after_polls.get(pr_url)
        if remaining is not None:
            if remaining <= 0:
                return "MERGED"
            self.merge_after_polls[pr_url] = remaining - 1
            return "OPEN"
        if pr_url in self.merged_prs:
            return "MERGED"
        if pr_url in self.closed_prs:
            return "CLOSED"
        return "OPEN"

    def close_issue(self, issue_number: int, *, comment: str) -> None:
        del comment
        self.closed.append(issue_number)
        self.issues = [issue for issue in self.issues if issue["number"] != issue_number]

    def mark_issue_failed(self, issue_number: int, *, label: str, comment: str) -> None:
        if self.fail_labeling:
            raise GitHubError("label permission denied")
        self.failed.append((issue_number, label, comment))
        self.issues = [issue for issue in self.issues if issue["number"] != issue_number]


@dataclass
class ScriptedOpenCode:
    """
    Fake OpenCodeGateway that returns scripted implement/review results in order.

    Each call pops the next scripted result; an Exception entry is raised instead.
    """

    implement_results: list[ImplementResult | Exception]
    review_results: list[ReviewResult]
    implement_calls: list[dict[str, object]] = field(default_factory=list)
    review_calls: list[dict[str, object]] = field(default_factory=list)
    workdir: Path = field(default_factory=Path)
    # Side effect run on every implement call (simulates work the agent did).
    on_implement: Callable[[], object] | None = None

    def run_implement(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "issue-worker",
        slice_number: int = 1,
    ) -> ImplementResult:
        self.implement_calls.append(
            {
                "issue_number": issue_number,
                "attempt": attempt,
                "message": message,
                "session_id": session_id,
                "agent": agent,
                "slice_number": slice_number,
            }
        )
        if self.on_implement is not None:
            self.on_implement()
        if len(self.implement_results) == 0:
            raise AssertionError("unexpected implement call")
        result = self.implement_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

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


@dataclass
class FakeClock:
    """Deterministic clock whose sleep advances time instead of blocking."""

    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


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


def _implement_missing_sentinel() -> ImplementResult:
    return ImplementResult(
        exit_code=0,
        session_id="ses",
        pr_url=None,
        status=None,
        assistant_text="I forgot the sentinel",
        log_path=Path("/tmp/x.log"),
    )


def _implement_timed_out(*, session_id: str | None = "ses_1", text: str = "") -> ImplementResult:
    return ImplementResult(
        exit_code=-9,
        session_id=session_id,
        pr_url=None,
        status=None,
        assistant_text=text,
        log_path=Path("/tmp/implement-timeout.log"),
        timed_out=True,
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
    assert final_state["handled"] == [1, 2]
    assert len(opencode.implement_calls) == 2
    assert len(opencode.review_calls) == 2


def test_progress_messages_describe_each_phase(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/5"
    github = FakeGitHub(issues=[_issue(5, title="Add login")], merged_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr_url), _implement_ok(pr_url)],
        review_results=[_review_changes("1. add tests\n2. rename var"), _review_approve()],
        workdir=tmp_path,
    )
    messages: list[str] = []
    run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
        notify=messages.append,
    )
    text = "\n".join(messages)
    assert "Queue: 1 eligible issue(s): #5" in text
    assert "Issue #5: Add login" in text
    assert "#5 implement attempt 1/3 → agent issue-worker" in text
    assert f"PR ready: {pr_url}" in text
    assert "#5 review attempt 1/3 → agent pr-reviewer" in text
    assert "→ REQUEST_CHANGES" in text
    assert "blockers: 1. add tests" in text
    assert "#5 implement attempt 2/3 → agent issue-worker, resuming session ses_1" in text
    assert "→ APPROVE" in text
    assert "PR merged" in text
    assert "✓ Closed issue #5 (1 closed, 0 skipped this run)" in text
    assert messages[-1] == "Queue empty: no eligible open issues left."


def test_many_issues_do_not_hit_recursion_limit(tmp_path: Path) -> None:
    count = 40
    prs = [f"https://github.com/acme/app/pull/{n}" for n in range(1, count + 1)]
    github = FakeGitHub(issues=[_issue(n) for n in range(1, count + 1)], merged_prs=set(prs))
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr) for pr in prs],
        review_results=[_review_approve() for _ in prs],
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
    assert len(github.closed) == count


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
    second_message = str(opencode.implement_calls[1]["message"])
    assert "add tests" in second_message
    assert pr_url in second_message
    assert "do NOT open another pull request" in second_message


def test_existing_pr_change_round_passes_pr_url(tmp_path: Path) -> None:
    """Without a resumable session, the worker must still be told which PR to update."""
    pr_url = "https://github.com/acme/app/pull/4"
    github = FakeGitHub(issues=[_issue(4)], open_prs={4: pr_url}, merged_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr_url)],
        review_results=[_review_changes("rename var"), _review_approve()],
        workdir=tmp_path,
    )
    run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    assert opencode.implement_calls[0]["session_id"] is None
    assert pr_url in str(opencode.implement_calls[0]["message"])


def test_max_rounds_halts(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/8"
    github = FakeGitHub(issues=[_issue(8)])
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr_url), _implement_ok(pr_url)],
        review_results=[_review_changes("round1"), _review_changes("round2")],
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
    github = FakeGitHub(issues=[_issue(4)], open_prs={4: pr_url}, merged_prs={pr_url})
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
        implement_results=[_implement_missing_sentinel()],
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


def test_timed_out_run_continues_in_the_same_session(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/17"
    github = FakeGitHub(issues=[_issue(17, title="Dashboard")], merged_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_timed_out(session_id="ses_w"), _implement_ok(pr_url, session_id="ses_w")],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    messages: list[str] = []
    final_state = run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
        notify=messages.append,
    )
    assert final_state["status"] == "done"
    assert github.closed == [17]
    first, second = opencode.implement_calls
    assert (first["slice_number"], first["session_id"]) == (1, None)
    assert (second["slice_number"], second["session_id"]) == (2, "ses_w")
    assert second["attempt"] == 1
    continue_message = str(second["message"])
    assert continue_message.startswith("Continue GitHub issue #17: Dashboard")
    assert "CONTINUATION: this is run 2 of at most 4" in continue_message
    # The resumed session already has the issue body; it is not re-sent.
    assert "UNTRUSTED ISSUE BODY" not in continue_message
    assert "ISSUE_STATUS: PR_READY" in continue_message
    text = "\n".join(messages)
    assert "stopped at the time limit" in text
    assert "(run 1/4); continuing in run 2" in text
    assert "#17 implement attempt 1/3, run 2/4 → agent issue-worker, resuming session ses_w" in text


def test_timed_out_run_without_session_starts_fresh_with_handoff(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/8"
    github = FakeGitHub(issues=[_issue(8)], merged_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_timed_out(session_id=None), _implement_ok(pr_url)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    second = opencode.implement_calls[1]
    assert second["session_id"] is None
    message = str(second["message"])
    assert "Implement GitHub issue #8" in message
    assert "body-8" in message
    assert "CONTINUATION: this is run 2 of at most 4" in message


def test_continuation_picks_up_pr_opened_by_the_stopped_run(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/30"
    github = FakeGitHub(issues=[_issue(3)], merged_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_timed_out(), _implement_ok(pr_url)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    # The stopped run got as far as `gh pr create` before the kill.
    opencode.on_implement = lambda: github.open_prs.setdefault(3, pr_url)
    run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    message = str(opencode.implement_calls[1]["message"])
    assert f"An open pull request already exists for this issue: {pr_url}" in message


def test_every_slice_timing_out_halts(tmp_path: Path) -> None:
    github = FakeGitHub(issues=[_issue(5)])
    opencode = ScriptedOpenCode(
        implement_results=[_implement_timed_out() for _ in range(3)],
        review_results=[],
        workdir=tmp_path,
    )
    with pytest.raises(OrchestratorHalt, match=r"time limit in all 3 run\(s\) of attempt 1"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            use_memory_checkpointer=True,
            max_continuations=2,
        )
    assert [call["slice_number"] for call in opencode.implement_calls] == [1, 2, 3]


def test_zero_continuations_halts_on_first_timeout(tmp_path: Path) -> None:
    github = FakeGitHub(issues=[_issue(5)])
    opencode = ScriptedOpenCode(
        implement_results=[_implement_timed_out()],
        review_results=[],
        workdir=tmp_path,
    )
    with pytest.raises(OrchestratorHalt, match=r"time limit in all 1 run\(s\)"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            use_memory_checkpointer=True,
            max_continuations=0,
        )


def test_timeout_after_the_sentinel_still_counts_as_done(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/6"
    github = FakeGitHub(issues=[_issue(6)], merged_prs={pr_url})
    late_kill = ImplementResult(
        exit_code=-9,
        session_id="ses_1",
        pr_url=pr_url,
        status="PR_READY",
        assistant_text=f"ISSUE_STATUS: PR_READY\nPR_URL: {pr_url}",
        log_path=Path("/tmp/implement.log"),
        timed_out=True,
    )
    opencode = ScriptedOpenCode(
        implement_results=[late_kill],
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
    assert len(opencode.implement_calls) == 1


def test_change_round_after_continuation_restarts_at_slice_one(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/2"
    github = FakeGitHub(issues=[_issue(2)], merged_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_timed_out(), _implement_ok(pr_url), _implement_ok(pr_url)],
        review_results=[_review_changes(), _review_approve()],
        workdir=tmp_path,
    )
    run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    calls = [(call["attempt"], call["slice_number"]) for call in opencode.implement_calls]
    assert calls == [(1, 1), (1, 2), (2, 1)]


def test_skip_policy_labels_failed_issue_and_continues(tmp_path: Path) -> None:
    pr2 = "https://github.com/acme/app/pull/2"
    github = FakeGitHub(issues=[_issue(1), _issue(2)], merged_prs={pr2})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_missing_sentinel(), _implement_ok(pr2)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    final_state = run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        on_halt="skip",
        use_memory_checkpointer=True,
    )
    assert final_state["status"] == "done"
    assert github.closed == [2]
    assert [entry[0] for entry in github.failed] == [1]
    assert github.failed[0][1] == "orchestrator:skip"
    assert "ISSUE_STATUS" in github.failed[0][2]
    assert [failure["number"] for failure in final_state["failures"]] == [1]


def test_skip_policy_halts_when_labeling_fails(tmp_path: Path) -> None:
    github = FakeGitHub(issues=[_issue(1), _issue(2)], fail_labeling=True)
    opencode = ScriptedOpenCode(
        implement_results=[_implement_missing_sentinel()],
        review_results=[],
        workdir=tmp_path,
    )
    with pytest.raises(OrchestratorHalt, match="failed to label"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            on_halt="skip",
            use_memory_checkpointer=True,
        )


def test_queue_is_reloaded_after_each_issue(tmp_path: Path) -> None:
    pr1 = "https://github.com/acme/app/pull/1"
    pr7 = "https://github.com/acme/app/pull/7"
    github = FakeGitHub(
        issues=[_issue(1)],
        merged_prs={pr1, pr7},
        late_issues={2: [_issue(7)]},
    )
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr1), _implement_ok(pr7)],
        review_results=[_review_approve(), _review_approve()],
        workdir=tmp_path,
    )
    run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        use_memory_checkpointer=True,
    )
    assert github.closed == [1, 7]


def test_handled_issue_not_retried_if_still_open(tmp_path: Path) -> None:
    """An issue that stays open after close (e.g. eventual consistency) is not redone."""
    pr1 = "https://github.com/acme/app/pull/1"

    class StickyGitHub(FakeGitHub):
        def close_issue(self, issue_number: int, *, comment: str) -> None:
            del comment
            self.closed.append(issue_number)

    github = StickyGitHub(issues=[_issue(1)], merged_prs={pr1})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr1)],
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
    assert github.closed == [1]


def test_approve_waits_for_auto_merge(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/3"
    github = FakeGitHub(issues=[_issue(3)], merge_after_polls={pr_url: 2})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr_url)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    clock = FakeClock()
    final_state = run_orchestrator(
        github=github,
        opencode=opencode,
        repo="acme/app",
        workdir=tmp_path,
        merge_wait_seconds=60,
        use_memory_checkpointer=True,
        sleep=clock.sleep,
        clock=clock.time,
        merge_poll_interval_seconds=10,
    )
    assert final_state["status"] == "done"
    assert clock.sleeps == [10, 10]
    assert github.closed == [3]


def test_approve_halts_when_merge_never_lands(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/3"
    github = FakeGitHub(issues=[_issue(3)])
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr_url)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    clock = FakeClock()
    with pytest.raises(OrchestratorHalt, match="not merged after 25s"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            merge_wait_seconds=25,
            use_memory_checkpointer=True,
            sleep=clock.sleep,
            clock=clock.time,
            merge_poll_interval_seconds=10,
        )
    assert clock.sleeps == [10, 10, 5]


def test_approve_halts_immediately_when_pr_closed(tmp_path: Path) -> None:
    pr_url = "https://github.com/acme/app/pull/3"
    github = FakeGitHub(issues=[_issue(3)], closed_prs={pr_url})
    opencode = ScriptedOpenCode(
        implement_results=[_implement_ok(pr_url)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    clock = FakeClock()
    with pytest.raises(OrchestratorHalt, match="closed without merging"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            use_memory_checkpointer=True,
            sleep=clock.sleep,
            clock=clock.time,
        )
    assert clock.sleeps == []


def test_github_error_in_prepare_halts_cleanly(tmp_path: Path) -> None:
    class BrokenPrLookup(FakeGitHub):
        def find_open_orchestrator_pr(self, issue_number: int) -> str | None:
            raise GitHubError(f"boom {issue_number}")

    github = BrokenPrLookup(issues=[_issue(1)])
    opencode = ScriptedOpenCode(implement_results=[], review_results=[], workdir=tmp_path)
    with pytest.raises(OrchestratorHalt, match="failed to look up open PRs"):
        run_orchestrator(
            github=github,
            opencode=opencode,
            repo="acme/app",
            workdir=tmp_path,
            use_memory_checkpointer=True,
        )


def test_interrupted_run_resumes_from_sqlite_checkpoint(tmp_path: Path) -> None:
    pr1 = "https://github.com/acme/app/pull/1"
    github = FakeGitHub(issues=[_issue(1)], merged_prs={pr1})
    crashing = ScriptedOpenCode(
        implement_results=[RuntimeError("container killed")],
        review_results=[],
        workdir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="container killed"):
        run_orchestrator(github=github, opencode=crashing, repo="acme/app", workdir=tmp_path)
    assert github.list_calls == 1

    messages: list[str] = []
    healthy = ScriptedOpenCode(
        implement_results=[_implement_ok(pr1)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    final_state = run_orchestrator(
        github=github,
        opencode=healthy,
        repo="acme/app",
        workdir=tmp_path,
        notify=messages.append,
    )
    assert final_state["status"] == "done"
    assert github.closed == [1]
    # Resumed at implement: the queue was not reloaded before implementing.
    assert healthy.implement_calls[0]["attempt"] == 1
    assert any(message.startswith("Resuming interrupted run") for message in messages)
    assert (tmp_path / ".orchestrator" / "state.sqlite").exists()


def test_fresh_flag_ignores_interrupted_checkpoint(tmp_path: Path) -> None:
    pr1 = "https://github.com/acme/app/pull/1"
    github = FakeGitHub(issues=[_issue(1)], merged_prs={pr1})
    crashing = ScriptedOpenCode(
        implement_results=[RuntimeError("container killed")],
        review_results=[],
        workdir=tmp_path,
    )
    with pytest.raises(RuntimeError):
        run_orchestrator(github=github, opencode=crashing, repo="acme/app", workdir=tmp_path)

    messages: list[str] = []
    healthy = ScriptedOpenCode(
        implement_results=[_implement_ok(pr1)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    run_orchestrator(
        github=github,
        opencode=healthy,
        repo="acme/app",
        workdir=tmp_path,
        resume=False,
        notify=messages.append,
    )
    assert not any(message.startswith("Resuming interrupted run") for message in messages)
    assert github.list_calls >= 2
    assert github.closed == [1]


def test_completed_run_starts_fresh_next_time(tmp_path: Path) -> None:
    pr1 = "https://github.com/acme/app/pull/1"
    github = FakeGitHub(issues=[_issue(1)], merged_prs={pr1})
    first = ScriptedOpenCode(
        implement_results=[_implement_ok(pr1)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    run_orchestrator(github=github, opencode=first, repo="acme/app", workdir=tmp_path)

    pr2 = "https://github.com/acme/app/pull/2"
    github.issues.append(_issue(2))
    github.merged_prs.add(pr2)
    second = ScriptedOpenCode(
        implement_results=[_implement_ok(pr2)],
        review_results=[_review_approve()],
        workdir=tmp_path,
    )
    final_state = run_orchestrator(github=github, opencode=second, repo="acme/app", workdir=tmp_path)
    assert final_state["handled"] == [2]
    assert github.closed == [1, 2]


def test_empty_issue_queue_is_done(tmp_path: Path) -> None:
    github = FakeGitHub(issues=[])
    opencode = ScriptedOpenCode(implement_results=[], review_results=[], workdir=tmp_path)
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
    with pytest.raises(ValueError):
        initial_state(
            repo="acme/app",
            workdir=tmp_path,
            merge_method="squash",
            max_rounds=1,
            skip_label="orchestrator:skip",
            merge_wait_seconds=-1,
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
            command: Sequence[str],
            *,
            cwd: Path,
            env: Mapping[str, str] | None,
            log_path: Path,
            on_stdout_line: Callable[[str], None] | None = None,
            timeout_seconds: float | None = None,
            live: LiveOutput | None = None,
        ) -> int:
            del command, cwd, env, log_path, on_stdout_line, timeout_seconds, live
            return 0

    github = GitHubClient(repo="acme/app", workdir=tmp_path, runner=UnusedGh())
    opencode = OpenCodeClient(workdir=tmp_path, runner=NoopRunner())
    compiled = create_orchestrator_graph(github=github, opencode=opencode)
    assert compiled is not None
