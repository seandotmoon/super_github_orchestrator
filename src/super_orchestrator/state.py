"""LangGraph state types for the serial issue orchestrator."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

MergeMethod = Literal["squash", "merge", "rebase"]
HaltPolicy = Literal["stop", "skip"]
ReviewVerdict = Literal["APPROVE", "REQUEST_CHANGES"]
RunStatus = Literal[
    "starting",
    "implementing",
    "reviewing",
    "closing",
    "done",
    "halted",
]


class IssueInfo(TypedDict):
    """Minimal GitHub issue fields needed by the control loop."""

    number: int
    title: str
    body: str
    url: str


class IssueFailure(TypedDict):
    """An issue that halted and was skipped under the `skip` halt policy."""

    number: int
    reason: str


class OrchestratorState(TypedDict):
    """
    Persistent graph state for one orchestrator run.

    Fields are updated by graph nodes; the SQLite checkpointer stores them after
    every step so an interrupted run (crash, kill, timeout) resumes at the node
    that was executing instead of starting over.
    """

    repo: str
    workdir: str
    merge_method: MergeMethod
    max_rounds: int
    skip_label: str
    on_halt: HaltPolicy
    merge_wait_seconds: int
    # Extra issue-worker runs allowed after a run hits the implement time budget,
    # per implement attempt. Read with .get(): checkpoints from older versions lack it.
    max_continuations: int
    # Latest snapshot of the queue; the current issue is always issues[issue_index].
    issues: list[IssueInfo]
    issue_index: int
    # Issue numbers already closed or skipped in this run; excluded on reload.
    handled: list[int]
    failures: list[IssueFailure]
    attempt: int
    # 1-based time slice within the current implement attempt (also read with .get()).
    implement_slice: int
    session_id: str | None
    pr_url: str | None
    review_verdict: ReviewVerdict | None
    review_feedback: str | None
    halt_reason: str | None
    status: RunStatus
    # Optional bookkeeping for logs / resume diagnostics.
    last_phase: NotRequired[str]
