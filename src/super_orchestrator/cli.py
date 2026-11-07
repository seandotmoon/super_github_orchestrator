"""Command-line entry point for the serial GitHub issue orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, Sequence

from super_orchestrator.github import GitHubClient
from super_orchestrator.graph import OrchestratorHalt, run_orchestrator
from super_orchestrator.opencode import OpenCodeClient
from super_orchestrator.preflight import PreflightError, run_preflight


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="super-orchestrator",
        description=(
            "Walk open GitHub issues in number order, blocking on "
            "`opencode run --agent` for implementation and PR review/merge."
        ),
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository as owner/name",
    )
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
        type=int,
        default=3,
        help="Maximum implement/review rounds per issue before halt (default: 3)",
    )
    parser.add_argument(
        "--skip-label",
        default="orchestrator:skip",
        help="Issues with this label are ignored (default: orchestrator:skip)",
    )
    parser.add_argument(
        "--thread-id",
        default="super-orchestrator",
        help="LangGraph checkpoint thread id (default: super-orchestrator)",
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


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse CLI args, run preflight, then drive the orchestrator loop.

    Returns:
        Process exit code (0 success, 1 halt/error, 2 usage).
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not isinstance(args.max_rounds, int) or args.max_rounds < 1:
        parser.error("--max-rounds must be a positive integer")

    workdir: Path = args.workdir.expanduser().resolve()
    repo: str = args.repo
    merge_method: Literal["squash", "merge", "rebase"] = args.merge_method
    skip_label: str = args.skip_label

    github = GitHubClient(repo=repo, workdir=workdir, skip_label=skip_label)
    opencode = OpenCodeClient(workdir=workdir)

    try:
        run_preflight(
            repo=repo,
            workdir=workdir,
            github=github,
            opencode=opencode,
            require_agents=not args.skip_preflight_agents,
        )
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 1

    print(f"Starting orchestrator for {repo} in {workdir}")
    print(f"merge_method={merge_method} max_rounds={args.max_rounds}")

    try:
        final_state = run_orchestrator(
            github=github,
            opencode=opencode,
            repo=repo,
            workdir=workdir,
            merge_method=merge_method,
            max_rounds=args.max_rounds,
            skip_label=skip_label,
            thread_id=args.thread_id,
            use_memory_checkpointer=args.memory_checkpointer,
        )
    except OrchestratorHalt as exc:
        print(f"orchestrator halted: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — top-level CLI boundary
        print(f"orchestrator error: {exc}", file=sys.stderr)
        return 1

    status = final_state.get("status")
    issue_count = len(final_state.get("issues", []))
    print(f"orchestrator finished status={status} issues_seen={issue_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
