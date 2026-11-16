"""LangGraph control loop for serial GitHub issue orchestration."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from super_orchestrator.activity import format_duration
from super_orchestrator.github import GitHubError, GitHubGateway
from super_orchestrator.opencode import OpenCodeError, OpenCodeGateway
from super_orchestrator.prompts import (
    build_continue_message,
    build_implement_message,
    build_review_message,
)
from super_orchestrator.state import (
    HaltPolicy,
    IssueFailure,
    IssueInfo,
    MergeMethod,
    OrchestratorState,
    ReviewVerdict,
    RunStatus,
)

# Each issue costs ~5 graph steps plus 2 per change round, and the queue is
# reloaded after every issue, so the step count is unbounded in principle.
# Set the limit explicitly rather than relying on LangGraph's version-dependent default.
RECURSION_LIMIT = 1_000_000

DEFAULT_MERGE_POLL_INTERVAL_SECONDS = 15.0
# Extra issue-worker runs after one hits the implement time budget (per attempt).
DEFAULT_MAX_CONTINUATIONS = 3

NodeUpdate = dict[str, object]
OrchestratorGraph = StateGraph[OrchestratorState]
CompiledOrchestratorGraph = CompiledStateGraph[OrchestratorState]


class OrchestratorHalt(RuntimeError):
    """Raised by the CLI when the graph ends in a halted state."""


def _current_issue(state: OrchestratorState) -> IssueInfo | None:
    """Return the issue at issue_index, or None when the queue is exhausted."""
    issues = state["issues"]
    index = state["issue_index"]
    if index < 0 or index >= len(issues):
        return None
    return issues[index]


def _halt(phase: str, reason: str, **extra: object) -> NodeUpdate:
    """Build a halted node update with a reason and the phase that halted."""
    update: NodeUpdate = {"status": "halted", "halt_reason": reason, "last_phase": phase}
    update.update(extra)
    return update


def _implement_slice(state: OrchestratorState) -> int:
    """Return the current 1-based implement slice (1 for checkpoints that predate slices)."""
    value = state.get("implement_slice", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"state.implement_slice invalid: {value!r}")
    return value


def _max_slices(state: OrchestratorState) -> int:
    """Return how many implement runs one attempt may use (first run + continuations)."""
    value = state.get("max_continuations", DEFAULT_MAX_CONTINUATIONS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"state.max_continuations invalid: {value!r}")
    return 1 + value


def _format_errors(errors: tuple[str, ...]) -> str:
    """Render OpenCode error events as a halt-reason suffix."""
    if len(errors) == 0:
        return ""
    return f"; OpenCode errors: {' | '.join(errors)}"


def build_orchestrator_graph(
    *,
    github: GitHubGateway,
    opencode: OpenCodeGateway,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    merge_poll_interval_seconds: float = DEFAULT_MERGE_POLL_INTERVAL_SECONDS,
    notify: Callable[[str], None] | None = None,
) -> OrchestratorGraph:
    """
    Build an uncompiled StateGraph for the serial orchestrator.

    Flow: load_issues → prepare_issue → implement ⇄ review → close_issue →
    load_issues (reloaded so new issues are picked up). Under the `skip` halt
    policy a halted issue goes to fail_issue (labelled, then the loop continues).

    Args:
        github: GitHub gateway.
        opencode: OpenCode gateway.
        sleep: Injectable sleep used while waiting for a merge (tests).
        clock: Injectable monotonic clock used for the merge deadline (tests).
        merge_poll_interval_seconds: Delay between PR state polls.
        notify: Receives human-readable progress messages.
    """
    if merge_poll_interval_seconds <= 0:
        raise ValueError("merge_poll_interval_seconds must be positive")

    def say(message: str) -> None:
        """Emit a progress message when a listener is attached."""
        if notify is not None:
            notify(message)

    def first_line(text: str | None) -> str:
        """Return the first non-empty line of text, truncated for the console."""
        if text is None:
            return ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped != "":
                return stripped if len(stripped) <= 160 else f"{stripped[:159]}…"
        return ""

    def load_issues(state: OrchestratorState) -> NodeUpdate:
        """Fetch eligible open issues, drop ones handled this run, reset per-issue fields."""
        try:
            fetched = github.list_open_issues()
        except GitHubError as exc:
            return _halt("load_issues", f"failed to list issues: {exc}", issues=[], issue_index=0)
        handled = set(state["handled"])
        remaining = [issue for issue in fetched if issue["number"] not in handled]
        status: RunStatus = "starting" if len(remaining) > 0 else "done"
        if len(remaining) == 0:
            say("Queue empty: no eligible open issues left.")
        else:
            numbers = ", ".join(f"#{issue['number']}" for issue in remaining[:15])
            more = f" (+{len(remaining) - 15} more)" if len(remaining) > 15 else ""
            say(f"Queue: {len(remaining)} eligible issue(s): {numbers}{more}")
        return {
            "issues": remaining,
            "issue_index": 0,
            "attempt": 0,
            "implement_slice": 1,
            "session_id": None,
            "pr_url": None,
            "review_verdict": None,
            "review_feedback": None,
            "halt_reason": None,
            "status": status,
            "last_phase": "load_issues",
        }

    def prepare_issue(state: OrchestratorState) -> NodeUpdate:
        """
        Prepare the current issue: reuse an existing orchestrator PR when present.

        When an open PR already exists for this issue, skip implementation and
        jump straight to review by seeding pr_url and attempt=1.
        """
        issue = _current_issue(state)
        if issue is None:
            return _halt("prepare_issue", "prepare_issue called with no current issue")

        try:
            existing_pr = github.find_open_orchestrator_pr(issue["number"])
        except GitHubError as exc:
            return _halt(
                "prepare_issue",
                f"failed to look up open PRs for issue #{issue['number']}: {exc}",
            )

        say(f"━━━ Issue #{issue['number']}: {issue['title']} ━━━")
        say(f"    {issue['url']}")
        if existing_pr is not None:
            say(f"#{issue['number']} found existing orchestrator PR {existing_pr}; skipping to review")
        status: RunStatus = "reviewing" if existing_pr is not None else "implementing"
        return {
            "pr_url": existing_pr,
            "attempt": 1,
            "implement_slice": 1,
            "session_id": None,
            "review_verdict": None,
            "review_feedback": None,
            "halt_reason": None,
            "status": status,
            "last_phase": "prepare_issue_existing_pr" if existing_pr is not None else "prepare_issue",
        }

    def find_pr_after_stop(issue_number: int) -> str | None:
        """Look up a PR the stopped run may have opened; a lookup failure is not fatal."""
        try:
            return github.find_open_orchestrator_pr(issue_number)
        except GitHubError as exc:
            say(f"#{issue_number} could not check for a PR from the stopped run: {exc}")
            return None

    def implement(state: OrchestratorState) -> NodeUpdate:
        """
        Block on issue-worker until the process exits; parse PR sentinels.

        A run killed at the implement time budget is one "slice". Unless every
        slice for this attempt is used, the node returns status `implementing`
        with the next slice number and routes back to itself. That slice resumes
        the same OpenCode session when its id is known (fresh session with a
        handoff note otherwise); uncommitted edits stay in the shared checkout.
        """
        issue = _current_issue(state)
        if issue is None:
            return _halt("implement", "implement called with no current issue")

        attempt = state["attempt"]
        try:
            slice_number = _implement_slice(state)
            max_slices = _max_slices(state)
        except RuntimeError as exc:
            return _halt("implement", str(exc))
        if slice_number > max_slices:
            return _halt(
                "implement",
                f"implement slice {slice_number} exceeds the limit of {max_slices}",
            )

        pr_url = state.get("pr_url")
        if slice_number > 1 and pr_url is None:
            pr_url = find_pr_after_stop(issue["number"])

        # Change rounds and continuations reuse the OpenCode session when available.
        resume_session = state.get("session_id") if attempt > 1 or slice_number > 1 else None
        if slice_number > 1 and resume_session is not None:
            message = build_continue_message(
                issue,
                repo=state["repo"],
                slice_number=slice_number,
                max_slices=max_slices,
                pr_url=pr_url,
            )
        else:
            message = build_implement_message(
                issue,
                repo=state["repo"],
                attempt=attempt,
                review_feedback=state.get("review_feedback"),
                pr_url=pr_url,
                slice_number=slice_number,
                max_slices=max_slices,
            )
        slice_text = f", run {slice_number}/{max_slices}" if slice_number > 1 else ""
        session_text = f", resuming session {resume_session}" if resume_session else ""
        pr_text = f", updating {pr_url}" if pr_url else ""
        say(
            f"#{issue['number']} implement attempt {attempt}/{state['max_rounds']}{slice_text} → "
            f"agent issue-worker{session_text}{pr_text}"
        )
        started = clock()

        try:
            result = opencode.run_implement(
                issue_number=issue["number"],
                attempt=attempt,
                message=message,
                session_id=resume_session,
                slice_number=slice_number,
            )
        except OpenCodeError as exc:
            return _halt("implement", f"OpenCode implement failed: {exc}", pr_url=pr_url)

        took = format_duration(clock() - started)
        session_id = result.session_id or state.get("session_id")
        has_pr = result.status == "PR_READY" and result.pr_url is not None
        # The sentinel is the worker's last word, so a run killed right after it still counts.
        finished = has_pr and (result.exit_code == 0 or result.timed_out)

        if result.timed_out and not finished:
            if slice_number >= max_slices:
                say(
                    f"#{issue['number']} issue-worker stopped at the time limit after {took}; "
                    f"no runs left ({max_slices} used)"
                )
                return _halt(
                    "implement",
                    (
                        f"issue-worker hit the time limit in all {max_slices} run(s) of "
                        f"attempt {attempt} for issue #{issue['number']} "
                        f"(log: {result.log_path}){_format_errors(result.errors)}"
                    ),
                    session_id=session_id,
                    pr_url=pr_url,
                )
            say(
                f"#{issue['number']} issue-worker stopped at the time limit after {took} "
                f"(run {slice_number}/{max_slices}); continuing in run {slice_number + 1}"
            )
            return {
                "session_id": session_id,
                "pr_url": pr_url,
                "implement_slice": slice_number + 1,
                "status": "implementing",
                "halt_reason": None,
                "last_phase": "implement_timeout",
            }

        if finished:
            say(f"#{issue['number']} issue-worker finished in {took} → PR ready: {result.pr_url}")
        else:
            say(f"#{issue['number']} issue-worker finished in {took} (exit {result.exit_code}) without a PR")
        if result.exit_code != 0 and not finished:
            return _halt(
                "implement",
                (
                    f"issue-worker exited {result.exit_code} for issue "
                    f"#{issue['number']} (log: {result.log_path})"
                    f"{_format_errors(result.errors)}"
                ),
                session_id=session_id,
            )

        if result.status != "PR_READY" or result.pr_url is None:
            return _halt(
                "implement",
                (
                    f"issue-worker finished without ISSUE_STATUS: PR_READY / PR_URL "
                    f"for issue #{issue['number']} (log: {result.log_path})"
                    f"{_format_errors(result.errors)}"
                ),
                session_id=session_id,
            )

        return {
            "session_id": session_id,
            "pr_url": result.pr_url,
            "implement_slice": 1,
            "review_verdict": None,
            "status": "reviewing",
            "halt_reason": None,
            "last_phase": "implement",
        }

    def wait_for_merge(pr_url: str, wait_seconds: int) -> str | None:
        """
        Poll the PR until it is merged, closed, or the wait budget runs out.

        Covers auto-merge and merge queues, where `gh pr merge` returns before
        the merge lands.

        Returns:
            None when merged, otherwise a halt reason.
        """
        deadline = clock() + wait_seconds
        announced = False
        while True:
            pr_state = github.get_pr_state(pr_url)
            if pr_state == "MERGED":
                say(f"PR merged: {pr_url}")
                return None
            if pr_state == "CLOSED":
                return f"REVIEW_VERDICT: APPROVE but PR was closed without merging: {pr_url}"
            remaining = deadline - clock()
            if remaining <= 0:
                return (
                    f"REVIEW_VERDICT: APPROVE but PR is not merged after "
                    f"{wait_seconds}s: {pr_url}"
                )
            if not announced:
                say(
                    f"Waiting for merge (auto-merge / merge queue), polling every "
                    f"{format_duration(merge_poll_interval_seconds)} for up to "
                    f"{format_duration(wait_seconds)}"
                )
                announced = True
            sleep(min(merge_poll_interval_seconds, remaining))

    def review(state: OrchestratorState) -> NodeUpdate:
        """Block on pr-reviewer; verify merge when the verdict is APPROVE."""
        issue = _current_issue(state)
        pr_url = state.get("pr_url")
        if issue is None or pr_url is None:
            return _halt("review", "review called without issue or pr_url")

        attempt = state["attempt"]
        message = build_review_message(
            issue,
            repo=state["repo"],
            pr_url=pr_url,
            merge_method=state["merge_method"],
            attempt=attempt,
        )

        say(
            f"#{issue['number']} review attempt {attempt}/{state['max_rounds']} → "
            f"agent pr-reviewer on {pr_url}"
        )
        started = clock()
        try:
            result = opencode.run_review(
                issue_number=issue["number"],
                attempt=attempt,
                message=message,
                session_id=None,
            )
        except OpenCodeError as exc:
            return _halt("review", f"OpenCode review failed: {exc}")

        took = format_duration(clock() - started)
        verdict_text = result.verdict if result.verdict is not None else "no verdict"
        say(f"#{issue['number']} pr-reviewer finished in {took} → {verdict_text}")
        if result.verdict == "REQUEST_CHANGES":
            say(f"#{issue['number']} blockers: {first_line(result.feedback)}")

        if result.exit_code != 0:
            return _halt(
                "review",
                (
                    f"pr-reviewer exited {result.exit_code} for issue "
                    f"#{issue['number']} (log: {result.log_path})"
                    f"{_format_errors(result.errors)}"
                ),
            )

        if result.verdict is None:
            return _halt(
                "review",
                (
                    f"pr-reviewer finished without REVIEW_VERDICT for issue "
                    f"#{issue['number']} (log: {result.log_path})"
                    f"{_format_errors(result.errors)}"
                ),
            )

        if result.verdict == "APPROVE":
            try:
                merge_problem = wait_for_merge(pr_url, state["merge_wait_seconds"])
            except GitHubError as exc:
                merge_problem = f"failed to verify PR merge state: {exc}"
            if merge_problem is not None:
                return _halt(
                    "review",
                    merge_problem,
                    review_verdict="APPROVE",
                    review_feedback=result.feedback,
                )
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
            return _halt(
                "review",
                f"max review rounds ({state['max_rounds']}) exhausted for issue #{issue['number']}",
                review_verdict="REQUEST_CHANGES",
                review_feedback=result.feedback,
            )

        say(f"#{issue['number']} sending feedback back to issue-worker (round {next_attempt})")
        return {
            "review_verdict": "REQUEST_CHANGES",
            "review_feedback": result.feedback,
            "attempt": next_attempt,
            "implement_slice": 1,
            "status": "implementing",
            "halt_reason": None,
            "last_phase": "review",
        }

    def close_issue(state: OrchestratorState) -> NodeUpdate:
        """Close the GitHub issue after a verified merge, then reload the queue."""
        issue = _current_issue(state)
        pr_url = state.get("pr_url")
        if issue is None:
            return _halt("close_issue", "close_issue called with no current issue")

        comment_parts = ["Closed by super-orchestrator after merge."]
        if isinstance(pr_url, str) and pr_url.strip() != "":
            comment_parts.append(f"Pull request: {pr_url}")
        comment = "\n".join(comment_parts)

        try:
            github.close_issue(issue["number"], comment=comment)
        except GitHubError as exc:
            return _halt("close_issue", f"failed to close issue #{issue['number']}: {exc}")

        done = len(state["handled"]) + 1 - len(state["failures"])
        say(
            f"✓ Closed issue #{issue['number']} "
            f"({done} closed, {len(state['failures'])} skipped this run)"
        )
        return {
            "handled": [*state["handled"], issue["number"]],
            "status": "starting",
            "halt_reason": None,
            "last_phase": "close_issue",
        }

    def fail_issue(state: OrchestratorState) -> NodeUpdate:
        """
        Skip a halted issue (halt policy `skip`): label it, comment, continue.

        Labelling with the skip label keeps future runs from retrying it until a
        human removes the label. If labelling itself fails, the run halts.
        """
        issue = _current_issue(state)
        reason = state.get("halt_reason") or "unknown halt reason"
        if issue is None:
            return _halt("fail_issue", reason)

        label = state["skip_label"]
        comment = "\n".join(
            [
                f"super-orchestrator stopped working on this issue and applied `{label}`.",
                "",
                f"Reason: {reason}",
                "",
                f"Remove the `{label}` label to let the orchestrator retry it.",
            ]
        )
        try:
            github.mark_issue_failed(issue["number"], label=label, comment=comment)
        except GitHubError as exc:
            return _halt(
                "fail_issue",
                f"{reason}; additionally failed to label issue #{issue['number']}: {exc}",
            )

        say(f"✗ Skipped issue #{issue['number']} (labelled {label}): {reason}")
        failure = IssueFailure(number=issue["number"], reason=reason)
        return {
            "handled": [*state["handled"], issue["number"]],
            "failures": [*state["failures"], failure],
            "status": "starting",
            "halt_reason": None,
            "last_phase": "fail_issue",
        }

    def halt_route(state: OrchestratorState) -> str:
        """Send a halted issue to fail_issue under the skip policy, else stop."""
        if state["on_halt"] == "skip" and _current_issue(state) is not None:
            return "fail_issue"
        return END

    def route_after_load(state: OrchestratorState) -> str:
        """Stop when the queue is empty or listing failed."""
        if state["status"] in {"done", "halted"}:
            return END
        return "prepare_issue"

    def route_after_prepare(state: OrchestratorState) -> str:
        """Route to implement or review (existing PR); halts follow the halt policy."""
        status = state["status"]
        if status == "halted":
            return halt_route(state)
        if status == "reviewing" and state.get("pr_url") is not None:
            return "review"
        return "implement"

    def route_after_implement(state: OrchestratorState) -> str:
        """Review on success, run the next slice after a time-limit stop; halts follow the policy."""
        status = state["status"]
        if status == "halted":
            return halt_route(state)
        if status == "implementing":
            return "implement"
        return "review"

    def route_after_review(state: OrchestratorState) -> str:
        """Close on approve, re-implement on changes; halts follow the halt policy."""
        status = state["status"]
        if status == "halted":
            return halt_route(state)
        if status == "closing":
            return "close_issue"
        if status == "implementing":
            return "implement"
        return END

    def route_after_close(state: OrchestratorState) -> str:
        """Reload the queue after a close; halts follow the halt policy."""
        if state["status"] == "halted":
            return halt_route(state)
        return "load_issues"

    def route_after_fail(state: OrchestratorState) -> str:
        """Reload the queue after skipping, or stop if labelling failed."""
        if state["status"] == "halted":
            return END
        return "load_issues"

    graph: OrchestratorGraph = StateGraph(OrchestratorState)
    graph.add_node("load_issues", load_issues)
    graph.add_node("prepare_issue", prepare_issue)
    graph.add_node("implement", implement)
    graph.add_node("review", review)
    graph.add_node("close_issue", close_issue)
    graph.add_node("fail_issue", fail_issue)

    graph.set_entry_point("load_issues")
    graph.add_conditional_edges("load_issues", route_after_load)
    graph.add_conditional_edges("prepare_issue", route_after_prepare)
    graph.add_conditional_edges("implement", route_after_implement)
    graph.add_conditional_edges("review", route_after_review)
    graph.add_conditional_edges("close_issue", route_after_close)
    graph.add_conditional_edges("fail_issue", route_after_fail)
    return graph


def create_orchestrator_graph(
    *,
    github: GitHubGateway,
    opencode: OpenCodeGateway,
    checkpointer: MemorySaver | SqliteSaver | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    merge_poll_interval_seconds: float = DEFAULT_MERGE_POLL_INTERVAL_SECONDS,
    notify: Callable[[str], None] | None = None,
) -> CompiledOrchestratorGraph:
    """Compile the orchestrator graph, optionally with a checkpointer."""
    graph = build_orchestrator_graph(
        github=github,
        opencode=opencode,
        sleep=sleep,
        clock=clock,
        merge_poll_interval_seconds=merge_poll_interval_seconds,
        notify=notify,
    )
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
    merge_method: MergeMethod,
    max_rounds: int,
    skip_label: str,
    on_halt: HaltPolicy = "stop",
    merge_wait_seconds: int = 600,
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
) -> OrchestratorState:
    """Construct the starting OrchestratorState for a new run."""
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be a positive int")
    if (
        isinstance(max_continuations, bool)
        or not isinstance(max_continuations, int)
        or max_continuations < 0
    ):
        raise ValueError("max_continuations must be a non-negative int")
    if merge_method not in {"squash", "merge", "rebase"}:
        raise ValueError("merge_method must be squash, merge, or rebase")
    if on_halt not in {"stop", "skip"}:
        raise ValueError("on_halt must be stop or skip")
    if not isinstance(merge_wait_seconds, int) or merge_wait_seconds < 0:
        raise ValueError("merge_wait_seconds must be a non-negative int")
    return OrchestratorState(
        repo=repo,
        workdir=str(workdir),
        merge_method=merge_method,
        max_rounds=max_rounds,
        skip_label=skip_label,
        on_halt=on_halt,
        merge_wait_seconds=merge_wait_seconds,
        max_continuations=max_continuations,
        issues=[],
        issue_index=0,
        handled=[],
        failures=[],
        attempt=0,
        implement_slice=1,
        session_id=None,
        pr_url=None,
        review_verdict=None,
        review_feedback=None,
        halt_reason=None,
        status="starting",
        last_phase="init",
    )


def _invoke(
    compiled: CompiledOrchestratorGraph,
    state: OrchestratorState,
    config: RunnableConfig,
    *,
    resume: bool,
    notify: Callable[[str], None] | None,
) -> object:
    """
    Resume an interrupted run for this thread, or start a new one.

    A checkpoint with pending nodes means the previous process died mid-run
    (crash, kill, timeout); invoking with None continues at that node with the
    saved state. Finished runs (done or halted) have no pending nodes, so a new
    run starts and reloads the issue queue from GitHub.
    """
    if resume:
        snapshot = compiled.get_state(config)
        values = snapshot.values
        same_target = (
            isinstance(values, Mapping)
            and values.get("repo") == state["repo"]
            and values.get("workdir") == state["workdir"]
        )
        if len(snapshot.next) > 0 and same_target:
            if notify is not None:
                notify(f"Resuming interrupted run at node(s): {', '.join(snapshot.next)}")
            return compiled.invoke(None, config=config)
    return compiled.invoke(state, config=config)


def run_orchestrator(
    *,
    github: GitHubGateway,
    opencode: OpenCodeGateway,
    repo: str,
    workdir: Path,
    merge_method: MergeMethod = "squash",
    max_rounds: int = 3,
    skip_label: str = "orchestrator:skip",
    on_halt: HaltPolicy = "stop",
    merge_wait_seconds: int = 600,
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
    thread_id: str = "super-orchestrator",
    use_memory_checkpointer: bool = False,
    resume: bool = True,
    notify: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    merge_poll_interval_seconds: float = DEFAULT_MERGE_POLL_INTERVAL_SECONDS,
) -> OrchestratorState:
    """
    Compile the graph with a checkpointer and invoke until completion or halt.

    Args:
        resume: Continue an interrupted run on the same thread_id when one
            exists. Settings saved with that run take precedence over the
            arguments; pass resume=False to start over with new settings.
        notify: Optional callback for human-readable progress messages.

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
        on_halt=on_halt,
        merge_wait_seconds=merge_wait_seconds,
        max_continuations=max_continuations,
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    def compile_with(checkpointer: MemorySaver | SqliteSaver) -> CompiledOrchestratorGraph:
        return create_orchestrator_graph(
            github=github,
            opencode=opencode,
            checkpointer=checkpointer,
            sleep=sleep,
            clock=clock,
            merge_poll_interval_seconds=merge_poll_interval_seconds,
            notify=notify,
        )

    if use_memory_checkpointer:
        compiled = compile_with(MemorySaver())
        return _finalize(_invoke(compiled, state, config, resume=resume, notify=notify))

    with open_sqlite_checkpointer(workdir) as checkpointer:
        compiled = compile_with(checkpointer)
        return _finalize(_invoke(compiled, state, config, resume=resume, notify=notify))


def _finalize(final_state: object) -> OrchestratorState:
    """Validate invoke output and raise OrchestratorHalt when status is halted."""
    if not isinstance(final_state, dict):
        raise RuntimeError("graph.invoke returned a non-dict state")
    result = _coerce_state(final_state)
    if result["status"] == "halted":
        reason = result["halt_reason"] or "unknown halt reason"
        raise OrchestratorHalt(str(reason))
    return result


_T = TypeVar("_T", bound=str)

_MERGE_METHODS: dict[str, MergeMethod] = {"squash": "squash", "merge": "merge", "rebase": "rebase"}
_HALT_POLICIES: dict[str, HaltPolicy] = {"stop": "stop", "skip": "skip"}
_VERDICTS: dict[str, ReviewVerdict] = {"APPROVE": "APPROVE", "REQUEST_CHANGES": "REQUEST_CHANGES"}
_STATUSES: dict[str, RunStatus] = {
    "starting": "starting",
    "implementing": "implementing",
    "reviewing": "reviewing",
    "closing": "closing",
    "done": "done",
    "halted": "halted",
}


def _lookup(raw: Mapping[str, object], key: str, table: Mapping[str, _T]) -> _T:
    """Return the Literal value for raw[key] via a lookup table, or raise."""
    value = raw.get(key)
    if isinstance(value, str) and value in table:
        return table[value]
    raise RuntimeError(f"state.{key} invalid: {value!r}")


def _optional_lookup(
    raw: Mapping[str, object], key: str, table: Mapping[str, _T]
) -> _T | None:
    """Like _lookup but allows None."""
    if raw.get(key) is None:
        return None
    return _lookup(raw, key, table)


def _req_str(raw: Mapping[str, object], key: str) -> str:
    """Return raw[key] as str or raise."""
    value = raw.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"state.{key} must be a str")
    return value


def _opt_str(raw: Mapping[str, object], key: str) -> str | None:
    """Return raw[key] as str or None, or raise."""
    value = raw.get(key)
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(f"state.{key} must be a str or None")


def _req_int(raw: Mapping[str, object], key: str) -> int:
    """Return raw[key] as int (bool excluded) or raise."""
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"state.{key} must be an int")
    return value


def _int_or_default(raw: Mapping[str, object], key: str, default: int) -> int:
    """Return raw[key] as int, or default when the key is missing (older checkpoints)."""
    if raw.get(key) is None:
        return default
    return _req_int(raw, key)


def _req_list(raw: Mapping[str, object], key: str) -> list[object]:
    """Return raw[key] as a list or raise."""
    value = raw.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"state.{key} must be a list")
    return list(value)


def _coerce_state(raw: dict[str, object]) -> OrchestratorState:
    """
    Rebuild OrchestratorState from a LangGraph invoke dict with type checks.

    Avoids unchecked casts while remaining tolerant of extra LangGraph keys.
    """
    typed_issues: list[IssueInfo] = []
    for item in _req_list(raw, "issues"):
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
        typed_issues.append(IssueInfo(number=number, title=title, body=body, url=url))

    handled: list[int] = []
    for number in _req_list(raw, "handled"):
        if not isinstance(number, int):
            raise RuntimeError("state.handled entries must be int")
        handled.append(number)

    failures: list[IssueFailure] = []
    for item in _req_list(raw, "failures"):
        if not isinstance(item, dict):
            raise RuntimeError("state.failures entries must be dicts")
        number = item.get("number")
        reason = item.get("reason")
        if not isinstance(number, int) or not isinstance(reason, str):
            raise RuntimeError("invalid failure entry")
        failures.append(IssueFailure(number=number, reason=reason))

    state: OrchestratorState = {
        "repo": _req_str(raw, "repo"),
        "workdir": _req_str(raw, "workdir"),
        "merge_method": _lookup(raw, "merge_method", _MERGE_METHODS),
        "max_rounds": _req_int(raw, "max_rounds"),
        "skip_label": _req_str(raw, "skip_label"),
        "on_halt": _lookup(raw, "on_halt", _HALT_POLICIES),
        "merge_wait_seconds": _req_int(raw, "merge_wait_seconds"),
        "max_continuations": _int_or_default(raw, "max_continuations", DEFAULT_MAX_CONTINUATIONS),
        "issues": typed_issues,
        "issue_index": _req_int(raw, "issue_index"),
        "handled": handled,
        "failures": failures,
        "attempt": _req_int(raw, "attempt"),
        "implement_slice": _int_or_default(raw, "implement_slice", 1),
        "session_id": _opt_str(raw, "session_id"),
        "pr_url": _opt_str(raw, "pr_url"),
        "review_verdict": _optional_lookup(raw, "review_verdict", _VERDICTS),
        "review_feedback": _opt_str(raw, "review_feedback"),
        "halt_reason": _opt_str(raw, "halt_reason"),
        "status": _lookup(raw, "status", _STATUSES),
    }
    last_phase = raw.get("last_phase")
    if isinstance(last_phase, str):
        state["last_phase"] = last_phase
    return state
