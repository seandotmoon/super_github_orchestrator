"""LangGraph state types for the serial issue orchestrator."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class IssueInfo(TypedDict):
    """Minimal GitHub issue fields needed by the control loop."""

    number: int
    title: str
    body: str
    url: str


class OrchestratorState(TypedDict):
    """
    Persistent graph state for one orchestrator run.

    Fields are updated by graph nodes; the SQLite checkpointer restores them
    across process restarts so a halted issue can be resumed.
    """

    repo: str
    workdir: str
    merge_method: Literal["squash", "merge", "rebase"]
    max_rounds: int
    skip_label: str
    issues: list[IssueInfo]
    issue_index: int
    attempt: int
    session_id: str | None
    pr_url: str | None
    review_verdict: Literal["APPROVE", "REQUEST_CHANGES"] | None
    review_feedback: str | None
    halt_reason: str | None
    status: Literal[
        "starting",
        "implementing",
        "reviewing",
        "closing",
        "done",
        "halted",
    ]
    # Optional bookkeeping for logs / resume diagnostics.
    last_phase: NotRequired[str]
