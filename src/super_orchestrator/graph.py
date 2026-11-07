"""LangGraph control loop for serial GitHub issue orchestration."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from super_orchestrator.github import GitHubError, GitHubGateway
from super_orchestrator.opencode import OpenCodeError, OpenCodeGateway
from super_orchestrator.prompts import build_implement_message, build_review_message
from super_orchestrator.state import IssueInfo, OrchestratorState


class OrchestratorHalt(RuntimeError):
    """Raised by the CLI when the graph ends in a halted state."""


def _current_issue(state: OrchestratorState) -> IssueInfo | None:
    """Return the issue at issue_index, or None when the queue is exhausted."""
    issues = state["issues"]
    index = state["issue_index"]
    if index < 0 or index >= len(issues):
        return None
    return issues[index]


def build_orchestrator_graph(
    *,
    github: GitHubGateway,
    opencode: OpenCodeGateway,
) -> StateGraph:
    """
    Build an uncompiled StateGraph for the serial orchestrator.

    Nodes mutate OrchestratorState; conditional edges implement
    implement → review → close / retry / halt.
    """

    def load_issues(state: OrchestratorState) -> dict[str, object]:
        """Fetch open issues ordered by number and reset per-issue fields."""
        issues = github.list_open_issues()
        return {
            "issues": issues,
            "issue_index": 0,
            "attempt": 0,
            "session_id": None,
            "pr_url": None,
            "review_verdict": None,
            "review_feedback": None,
            "halt_reason": None,
            "status": "starting",
            "last_phase": "load_issues",
        }

    def prepare_issue(state: OrchestratorState) -> dict[str, object]:
        """
        Prepare the next issue: reuse an existing orchestrator PR when present.

        When an open PR already exists for this issue, skip implementation and
        jump straight to review by seeding pr_url and attempt=1.
        """
        issue = _current_issue(state)
        if issue is None:
            return {
                "status": "done",
                "halt_reason": None,
                "last_phase": "prepare_issue",
            }

        existing_pr = github.find_open_orchestrator_pr(issue["number"])
        if existing_pr is not None:
            return {
                "pr_url": existing_pr,
                "attempt": 1,
                "session_id": None,
                "review_verdict": None,
                "review_feedback": None,
                "halt_reason": None,
                "status": "reviewing",
                "last_phase": "prepare_issue_existing_pr",
            }

        return {
            "pr_url": None,
            "attempt": 1,
            "session_id": None,
            "review_verdict": None,
            "review_feedback": None,
            "halt_reason": None,
            "status": "implementing",
            "last_phase": "prepare_issue",
        }

    def implement(state: OrchestratorState) -> dict[str, object]:
        """Block on issue-worker until the process exits; parse PR sentinels."""
        issue = _current_issue(state)
        if issue is None:
            return {
                "status": "halted",
                "halt_reason": "implement called with no current issue",
                "last_phase": "implement",
            }

        attempt = state["attempt"]
        message = build_implement_message(
            issue,
            repo=state["repo"],
            attempt=attempt,
            review_feedback=state.get("review_feedback"),
        )
        # Continue the same OpenCode session across change rounds when available.
        resume_session = state.get("session_id") if attempt > 1 else None

        try:
            result = opencode.run_implement(
                issue_number=issue["number"],
                attempt=attempt,
                message=message,
                session_id=resume_session,
            )
        except OpenCodeError as exc:
            return {
                "status": "halted",
                "halt_reason": f"OpenCode implement failed: {exc}",
                "last_phase": "implement",
            }

        if result.exit_code != 0:
            return {
                "session_id": result.session_id or state.get("session_id"),
                "status": "halted",
                "halt_reason": (
                    f"issue-worker exited {result.exit_code} for issue "
                    f"#{issue['number']} (log: {result.log_path})"
                ),
                "last_phase": "implement",
            }

        if result.status != "PR_READY" or result.pr_url is None:
            return {
                "session_id": result.session_id or state.get("session_id"),
                "status": "halted",
                "halt_reason": (
                    f"issue-worker finished without ISSUE_STATUS: PR_READY / PR_URL "
                    f"for issue #{issue['number']} (log: {result.log_path})"
                ),
                "last_phase": "implement",
            }

        return {
            "session_id": result.session_id or state.get("session_id"),
            "pr_url": result.pr_url,
            "review_verdict": None,
            "status": "reviewing",
            "halt_reason": None,
            "last_phase": "implement",
        }

    def review(state: OrchestratorState) -> dict[str, object]:
        """Block on pr-reviewer; verify merge when the verdict is APPROVE."""
        issue = _current_issue(state)
        pr_url = state.get("pr_url")
        if issue is None or pr_url is None:
            return {
                "status": "halted",
                "halt_reason": "review called without issue or pr_url",
                "last_phase": "review",
            }

        attempt = state["attempt"]
        message = build_review_message(
            issue,
            repo=state["repo"],
            pr_url=pr_url,
            merge_method=state["merge_method"],
            attempt=attempt,
        )

        try:
            result = opencode.run_review(
                issue_number=issue["number"],
                attempt=attempt,
                message=message,
                session_id=None,
            )
        except OpenCodeError as exc:
            return {
                "status": "halted",
                "halt_reason": f"OpenCode review failed: {exc}",
                "last_phase": "review",
            }

        if result.exit_code != 0:
            return {
                "status": "halted",
                "halt_reason": (
                    f"pr-reviewer exited {result.exit_code} for issue "
                    f"#{issue['number']} (log: {result.log_path})"
                ),
                "last_phase": "review",
            }

        if result.verdict is None:
            return {
                "status": "halted",
                "halt_reason": (
                    f"pr-reviewer finished without REVIEW_VERDICT for issue "
                    f"#{issue['number']} (log: {result.log_path})"
                ),
                "last_phase": "review",
            }

        if result.verdict == "APPROVE":
            try:
                merged = github.is_pr_merged(pr_url)
            except GitHubError as exc:
                return {
                    "review_verdict": "APPROVE",
                    "review_feedback": result.feedback,
                    "status": "halted",
                    "halt_reason": f"failed to verify PR merge state: {exc}",
                    "last_phase": "review",
                }
            if not merged:
                return {
                    "review_verdict": "APPROVE",
                    "review_feedback": result.feedback,
                    "status": "halted",
                    "halt_reason": (
                        f"REVIEW_VERDICT: APPROVE but PR is not merged: {pr_url}"
                    ),
                    "last_phase": "review",
                }
            return {
                "review_verdict": "APPROVE",
                "review_feedback": result.feedback,
                "status": "closing",
                "halt_reason": None,
                "last_phase": "review",
            }

        # REQUEST_CHANGES path — bump attempt and re-enter implement when under max.
        next_attempt = attempt + 1
        if next_attempt > state["max_rounds"]:
            return {
                "review_verdict": "REQUEST_CHANGES",
                "review_feedback": result.feedback,
                "status": "halted",
                "halt_reason": (
                    f"max review rounds ({state['max_rounds']}) exhausted for "
                    f"issue #{issue['number']}"
                ),
                "last_phase": "review",
            }

        return {
            "review_verdict": "REQUEST_CHANGES",
            "review_feedback": result.feedback,
            "attempt": next_attempt,
            "status": "implementing",
            "halt_reason": None,
            "last_phase": "review",
        }

    def close_issue(state: OrchestratorState) -> dict[str, object]:
        """Close the GitHub issue after a verified merge and advance the queue."""
        issue = _current_issue(state)
        pr_url = state.get("pr_url")
        if issue is None:
            return {
                "status": "halted",
                "halt_reason": "close_issue called with no current issue",
                "last_phase": "close_issue",
            }

        comment_parts = ["Closed by super-orchestrator after merge."]
        if isinstance(pr_url, str) and pr_url.strip() != "":
            comment_parts.append(f"Pull request: {pr_url}")
        comment = "\n".join(comment_parts)

        try:
            github.close_issue(issue["number"], comment=comment)
        except GitHubError as exc:
            return {
                "status": "halted",
                "halt_reason": f"failed to close issue #{issue['number']}: {exc}",
                "last_phase": "close_issue",
            }

        next_index = state["issue_index"] + 1
        if next_index >= len(state["issues"]):
            return {
                "issue_index": next_index,
                "session_id": None,
                "pr_url": None,
                "review_verdict": None,
                "review_feedback": None,
                "attempt": 0,
                "status": "done",
                "halt_reason": None,
                "last_phase": "close_issue",
            }

        return {
            "issue_index": next_index,
            "session_id": None,
            "pr_url": None,
            "review_verdict": None,
            "review_feedback": None,
            "attempt": 0,
            "status": "starting",
            "halt_reason": None,
            "last_phase": "close_issue",
        }

    def route_after_load(state: OrchestratorState) -> str:
        """After loading, stop immediately when there are no open issues."""
        if len(state["issues"]) == 0:
            return END
        return "prepare_issue"

    def route_after_prepare(state: OrchestratorState) -> str:
        """Route to implement, review (existing PR), or end (done)."""
        status = state["status"]
        if status == "done":
            return END
        if status == "reviewing" and state.get("pr_url") is not None:
            return "review"
        return "implement"

    def route_after_implement(state: OrchestratorState) -> str:
        """Continue to review on success; end when halted."""
        if state["status"] == "halted":
            return END
        return "review"

    def route_after_review(state: OrchestratorState) -> str:
        """Close on approve, re-implement on changes, or end when halted."""
        status = state["status"]
        if status == "halted":
            return END
        if status == "closing":
            return "close_issue"
        if status == "implementing":
            return "implement"
        return END

    def route_after_close(state: OrchestratorState) -> str:
        """Advance to the next issue or finish the run."""
        if state["status"] in {"halted", "done"}:
            return END
        return "prepare_issue"

    graph: StateGraph = StateGraph(OrchestratorState)
    graph.add_node("load_issues", load_issues)
    graph.add_node("prepare_issue", prepare_issue)
    graph.add_node("implement", implement)
    graph.add_node("review", review)
    graph.add_node("close_issue", close_issue)

    graph.set_entry_point("load_issues")
    graph.add_conditional_edges("load_issues", route_after_load)
    graph.add_conditional_edges("prepare_issue", route_after_prepare)
    graph.add_conditional_edges("implement", route_after_implement)
    graph.add_conditional_edges("review", route_after_review)
    graph.add_conditional_edges("close_issue", route_after_close)
    return graph


def create_orchestrator_graph(
    *,
    github: GitHubGateway,
    opencode: OpenCodeGateway,
    checkpointer: MemorySaver | SqliteSaver | None = None,
) -> CompiledStateGraph:
    """Compile the orchestrator graph, optionally with a checkpointer."""
    graph = build_orchestrator_graph(github=github, opencode=opencode)
    if checkpointer is None:
        return graph.compile()
    return graph.compile(checkpointer=checkpointer)


@contextmanager
def open_sqlite_checkpointer(workdir: Path) -> Iterator[SqliteSaver]:
    """
    Open a SQLite checkpointer under `.orchestrator/state.sqlite`.

    Keeps the connection alive for the duration of the orchestrator run.
    """
    if not isinstance(workdir, Path):
        raise TypeError("workdir must be a Path")
    state_dir = workdir / ".orchestrator"
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "state.sqlite"
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        saver.setup()
        yield saver
    finally:
        connection.close()


def initial_state(
    *,
    repo: str,
    workdir: Path,
    merge_method: Literal["squash", "merge", "rebase"],
    max_rounds: int,
    skip_label: str,
) -> OrchestratorState:
    """Construct the starting OrchestratorState for a new (or resumed) run."""
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be a positive int")
    if merge_method not in {"squash", "merge", "rebase"}:
        raise ValueError("merge_method must be squash, merge, or rebase")
    return OrchestratorState(
        repo=repo,
        workdir=str(workdir),
        merge_method=merge_method,
        max_rounds=max_rounds,
        skip_label=skip_label,
        issues=[],
        issue_index=0,
        attempt=0,
        session_id=None,
        pr_url=None,
        review_verdict=None,
        review_feedback=None,
        halt_reason=None,
        status="starting",
        last_phase="init",
    )


def run_orchestrator(
    *,
    github: GitHubGateway,
    opencode: OpenCodeGateway,
    repo: str,
    workdir: Path,
    merge_method: Literal["squash", "merge", "rebase"] = "squash",
    max_rounds: int = 3,
    skip_label: str = "orchestrator:skip",
    thread_id: str = "super-orchestrator",
    use_memory_checkpointer: bool = False,
) -> OrchestratorState:
    """
    Compile the graph with a checkpointer and invoke until completion or halt.

    Returns:
        The final OrchestratorState.

    Raises:
        OrchestratorHalt: When status is halted.
    """
    state = initial_state(
        repo=repo,
        workdir=workdir,
        merge_method=merge_method,
        max_rounds=max_rounds,
        skip_label=skip_label,
    )
    config: dict[str, object] = {"configurable": {"thread_id": thread_id}}

    if use_memory_checkpointer:
        compiled = create_orchestrator_graph(
            github=github,
            opencode=opencode,
            checkpointer=MemorySaver(),
        )
        final_state = compiled.invoke(state, config=config)
        return _finalize(final_state)

    with open_sqlite_checkpointer(workdir) as checkpointer:
        compiled = create_orchestrator_graph(
            github=github,
            opencode=opencode,
            checkpointer=checkpointer,
        )
        final_state = compiled.invoke(state, config=config)
        return _finalize(final_state)


def _finalize(final_state: object) -> OrchestratorState:
    """Validate invoke output and raise OrchestratorHalt when status is halted."""
    if not isinstance(final_state, dict):
        raise RuntimeError("graph.invoke returned a non-dict state")
    result = _coerce_state(final_state)
    if result["status"] == "halted":
        reason = result["halt_reason"] or "unknown halt reason"
        raise OrchestratorHalt(str(reason))
    # Mark empty-queue success as done.
    if result["status"] == "starting" and len(result["issues"]) == 0:
        result["status"] = "done"
    return result


def _coerce_state(raw: dict[str, object]) -> OrchestratorState:
    """
    Rebuild OrchestratorState from a LangGraph invoke dict with type checks.

    Avoids unchecked casts while remaining tolerant of extra LangGraph keys.
    """
    repo = raw.get("repo")
    workdir = raw.get("workdir")
    merge_method = raw.get("merge_method")
    max_rounds = raw.get("max_rounds")
    skip_label = raw.get("skip_label")
    issues = raw.get("issues")
    issue_index = raw.get("issue_index")
    attempt = raw.get("attempt")
    session_id = raw.get("session_id")
    pr_url = raw.get("pr_url")
    review_verdict = raw.get("review_verdict")
    review_feedback = raw.get("review_feedback")
    halt_reason = raw.get("halt_reason")
    status = raw.get("status")
    last_phase = raw.get("last_phase")

    if not isinstance(repo, str):
        raise RuntimeError("state.repo must be a str")
    if not isinstance(workdir, str):
        raise RuntimeError("state.workdir must be a str")
    typed_merge: Literal["squash", "merge", "rebase"]
    if merge_method == "squash":
        typed_merge = "squash"
    elif merge_method == "merge":
        typed_merge = "merge"
    elif merge_method == "rebase":
        typed_merge = "rebase"
    else:
        raise RuntimeError("state.merge_method invalid")
    if not isinstance(max_rounds, int):
        raise RuntimeError("state.max_rounds must be an int")
    if not isinstance(skip_label, str):
        raise RuntimeError("state.skip_label must be a str")
    if not isinstance(issues, list):
        raise RuntimeError("state.issues must be a list")
    if not isinstance(issue_index, int):
        raise RuntimeError("state.issue_index must be an int")
    if not isinstance(attempt, int):
        raise RuntimeError("state.attempt must be an int")
    typed_session_id: str | None
    if session_id is None:
        typed_session_id = None
    elif isinstance(session_id, str):
        typed_session_id = session_id
    else:
        raise RuntimeError("state.session_id must be a str or None")
    typed_pr_url: str | None
    if pr_url is None:
        typed_pr_url = None
    elif isinstance(pr_url, str):
        typed_pr_url = pr_url
    else:
        raise RuntimeError("state.pr_url must be a str or None")
    typed_verdict: Literal["APPROVE", "REQUEST_CHANGES"] | None
    if review_verdict is None:
        typed_verdict = None
    elif review_verdict == "APPROVE":
        typed_verdict = "APPROVE"
    elif review_verdict == "REQUEST_CHANGES":
        typed_verdict = "REQUEST_CHANGES"
    else:
        raise RuntimeError("state.review_verdict invalid")
    typed_feedback: str | None
    if review_feedback is None:
        typed_feedback = None
    elif isinstance(review_feedback, str):
        typed_feedback = review_feedback
    else:
        raise RuntimeError("state.review_feedback must be a str or None")
    typed_halt: str | None
    if halt_reason is None:
        typed_halt = None
    elif isinstance(halt_reason, str):
        typed_halt = halt_reason
    else:
        raise RuntimeError("state.halt_reason must be a str or None")
    typed_status: Literal[
        "starting",
        "implementing",
        "reviewing",
        "closing",
        "done",
        "halted",
    ]
    if status == "starting":
        typed_status = "starting"
    elif status == "implementing":
        typed_status = "implementing"
    elif status == "reviewing":
        typed_status = "reviewing"
    elif status == "closing":
        typed_status = "closing"
    elif status == "done":
        typed_status = "done"
    elif status == "halted":
        typed_status = "halted"
    else:
        raise RuntimeError("state.status invalid")

    typed_issues: list[IssueInfo] = []
    for item in issues:
        if not isinstance(item, dict):
            raise RuntimeError("issue entry must be a dict")
        number = item.get("number")
        title = item.get("title")
        body = item.get("body")
        url = item.get("url")
        if not isinstance(number, int) or not isinstance(title, str):
            raise RuntimeError("invalid issue entry")
        if not isinstance(body, str) or not isinstance(url, str):
            raise RuntimeError("invalid issue entry")
        typed_issues.append(
            IssueInfo(number=number, title=title, body=body, url=url)
        )

    state: OrchestratorState = {
        "repo": repo,
        "workdir": workdir,
        "merge_method": typed_merge,
        "max_rounds": max_rounds,
        "skip_label": skip_label,
        "issues": typed_issues,
        "issue_index": issue_index,
        "attempt": attempt,
        "session_id": typed_session_id,
        "pr_url": typed_pr_url,
        "review_verdict": typed_verdict,
        "review_feedback": typed_feedback,
        "halt_reason": typed_halt,
        "status": typed_status,
    }
    if isinstance(last_phase, str):
        state["last_phase"] = last_phase
    return state
