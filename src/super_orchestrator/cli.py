"""Command-line entry point for the serial GitHub issue orchestrator."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from super_orchestrator.github import GitHubClient, GitHubError
from super_orchestrator.graph import DEFAULT_MAX_CONTINUATIONS, OrchestratorHalt, run_orchestrator
from super_orchestrator.opencode import (
    DEFAULT_COMPACT_TOKENS,
    DEFAULT_IMPLEMENT_TIMEOUT_SECONDS,
    DEFAULT_REVIEW_TIMEOUT_SECONDS,
    OpenCodeClient,
)
from super_orchestrator.preflight import PreflightError, run_preflight
from super_orchestrator.state import HaltPolicy, MergeMethod


def _positive_int(text: str) -> int:
    """argparse type: integer >= 1."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    """argparse type: integer >= 0."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return value


def _timeout_seconds(text: str) -> int | None:
    """argparse type: positive seconds, or 0 to disable the timeout."""
    value = _non_negative_int(text)
    return None if value == 0 else value


def log(message: str, *, error: bool = False) -> None:
    """Print a timestamped console line, flushed immediately (Docker buffers stdout)."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr if error else sys.stdout, flush=True)


def log_error(message: str) -> None:
    """Timestamped line on stderr."""
    log(message, error=True)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="super-orchestrator",
        description=(
            "Walk open GitHub issues in number order, blocking on "
            "`opencode run --agent` for implementation and PR review/merge."
        ),
    )
    parser.add_argument("--repo", required=True, help="GitHub repository as owner/name")
    parser.add_argument(
        "--workdir",
        required=True,
        type=Path,
        help="Local checkout of the target repository (where OpenCode runs)",
    )
    parser.add_argument(
        "--merge-method",
        choices=("squash", "merge", "rebase"),
        default="squash",
        help="Merge method the pr-reviewer agent should prefer (default: squash)",
    )
    parser.add_argument(
        "--max-rounds",
        type=_positive_int,
        default=3,
        help="Maximum implement/review rounds per issue before halt (default: 3)",
    )
    parser.add_argument(
        "--skip-label",
        default="orchestrator:skip",
        help="Issues with this label are ignored (default: orchestrator:skip)",
    )
    parser.add_argument(
        "--on-halt",
        choices=("stop", "skip"),
        default="stop",
        help=(
            "What to do when an issue halts: stop the run (default), or apply the "
            "skip label, comment, and continue with the next issue"
        ),
    )
    authors = parser.add_mutually_exclusive_group()
    authors.add_argument(
        "--allow-author",
        action="append",
        default=None,
        metavar="LOGIN",
        help=(
            "Only process issues opened by this GitHub login (repeatable). "
            "Default: only the authenticated gh user"
        ),
    )
    authors.add_argument(
        "--allow-any-author",
        action="store_true",
        help="Process issues from any author (unsafe on public repos: prompt injection)",
    )
    parser.add_argument(
        "--require-label",
        default=None,
        metavar="LABEL",
        help="Only process issues carrying this label",
    )
    parser.add_argument(
        "--implement-timeout",
        type=_timeout_seconds,
        default=DEFAULT_IMPLEMENT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "Stop one issue-worker run after this many seconds and continue in a new run "
            f"(see --max-continuations); 0 disables (default: {DEFAULT_IMPLEMENT_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-continuations",
        type=_non_negative_int,
        default=DEFAULT_MAX_CONTINUATIONS,
        metavar="N",
        help=(
            "Extra issue-worker runs allowed per implement attempt after a run hits "
            f"--implement-timeout; 0 halts on the first timeout (default: {DEFAULT_MAX_CONTINUATIONS})"
        ),
    )
    parser.add_argument(
        "--review-timeout",
        type=_timeout_seconds,
        default=DEFAULT_REVIEW_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"Kill pr-reviewer after this many seconds; 0 disables (default: {DEFAULT_REVIEW_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--compact-tokens",
        type=_non_negative_int,
        default=DEFAULT_COMPACT_TOKENS,
        metavar="TOKENS",
        help=f"Compact an agent session when its context reaches this many tokens; 0 disables (default: {DEFAULT_COMPACT_TOKENS})",
    )
    parser.add_argument(
        "--merge-wait",
        type=_non_negative_int,
        default=600,
        metavar="SECONDS",
        help=(
            "After APPROVE, wait up to this long for the PR to show as merged "
            "(auto-merge / merge queues) (default: 600)"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print orchestrator progress, not live agent activity (tool calls, subagents)",
    )
    parser.add_argument(
        "--heartbeat",
        type=_positive_int,
        default=120,
        metavar="SECONDS",
        help="Print a 'still running' line after this much agent silence (default: 120)",
    )
    parser.add_argument(
        "--thread-id",
        default="super-orchestrator",
        help="LangGraph checkpoint thread id (default: super-orchestrator)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore an interrupted checkpoint and start a new run with these settings",
    )
    parser.add_argument(
        "--skip-preflight-agents",
        action="store_true",
        help="Skip the check that issue-worker and pr-reviewer agents exist",
    )
    parser.add_argument(
        "--memory-checkpointer",
        action="store_true",
        help="Use an in-memory checkpointer instead of SQLite (testing only)",
    )
    return parser


def _merge_method(value: str) -> MergeMethod:
    """Narrow an argparse choice to the MergeMethod literal."""
    if value == "squash":
        return "squash"
    if value == "merge":
        return "merge"
    if value == "rebase":
        return "rebase"
    raise ValueError(f"invalid merge method {value!r}")


def _halt_policy(value: str) -> HaltPolicy:
    """Narrow an argparse choice to the HaltPolicy literal."""
    if value == "stop":
        return "stop"
    if value == "skip":
        return "skip"
    raise ValueError(f"invalid halt policy {value!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse CLI args, run preflight, then drive the orchestrator loop.

    Returns:
        Process exit code (0 success, 1 halt/error or skipped issues, 2 usage).
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    workdir: Path = args.workdir.expanduser().resolve()
    repo: str = args.repo
    merge_method = _merge_method(args.merge_method)
    on_halt = _halt_policy(args.on_halt)
    skip_label: str = args.skip_label
    require_label: str | None = args.require_label
    if require_label is not None and require_label.strip() == "":
        parser.error("--require-label must not be empty")

    compact_tokens: int | None = args.compact_tokens if args.compact_tokens > 0 else None
    opencode = OpenCodeClient(
        workdir=workdir,
        implement_timeout_seconds=args.implement_timeout,
        review_timeout_seconds=args.review_timeout,
        activity=None if args.quiet else log,
        status=log,
        heartbeat_seconds=args.heartbeat,
        compact_tokens=compact_tokens,
    )

    try:
        allowed_authors: list[str] | None
        if args.allow_any_author:
            allowed_authors = None
            log_error(
                "warning: --allow-any-author set; issue text from anyone will drive "
                "agents that run with --auto and can merge"
            )
        elif args.allow_author is not None:
            allowed_authors = list(args.allow_author)
        else:
            # Safe default: only issues the operator opened themselves.
            probe = GitHubClient(repo=repo, workdir=workdir, skip_label=skip_label)
            allowed_authors = [probe.viewer_login()]
        github = GitHubClient(
            repo=repo,
            workdir=workdir,
            skip_label=skip_label,
            allowed_authors=allowed_authors,
            require_label=require_label,
        )
    except (GitHubError, ValueError) as exc:
        log_error(f"preflight failed: {exc}")
        return 1

    log(f"Preflight: checking gh, opencode, {workdir} and agents…")
    try:
        run_preflight(
            repo=repo,
            workdir=workdir,
            github=github,
            opencode=opencode,
            require_agents=not args.skip_preflight_agents,
            on_halt=on_halt,
        )
    except PreflightError as exc:
        log_error(f"preflight failed: {exc}")
        return 1

    author_text = "any" if allowed_authors is None else ", ".join(sorted(allowed_authors))
    log(f"Starting orchestrator for {repo} in {workdir}")
    log(
        " ".join(
            [
                f"merge_method={merge_method}",
                f"max_rounds={args.max_rounds}",
                f"implement_timeout={args.implement_timeout}",
                f"max_continuations={args.max_continuations}",
                f"on_halt={on_halt}",
                f"authors={author_text}",
            ]
        )
    )

    try:
        final_state = run_orchestrator(
            github=github,
            opencode=opencode,
            repo=repo,
            workdir=workdir,
            merge_method=merge_method,
            max_rounds=args.max_rounds,
            skip_label=skip_label,
            on_halt=on_halt,
            merge_wait_seconds=args.merge_wait,
            max_continuations=args.max_continuations,
            thread_id=args.thread_id,
            use_memory_checkpointer=args.memory_checkpointer,
            resume=not args.fresh,
            notify=log,
        )
    except OrchestratorHalt as exc:
        log_error(f"orchestrator halted: {exc}")
        return 1
    except KeyboardInterrupt:
        log_error("interrupted; rerun the same command to resume from the last checkpoint")
        return 130
    except Exception as exc:  # noqa: BLE001 — top-level CLI boundary
        log_error(f"orchestrator error: {exc}")
        return 1

    failures = final_state["failures"]
    completed = len(final_state["handled"]) - len(failures)
    log(
        f"orchestrator finished status={final_state['status']} "
        f"closed={completed} skipped={len(failures)}"
    )
    for failure in failures:
        log_error(f"  skipped #{failure['number']}: {failure['reason']}")
    return 1 if len(failures) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
