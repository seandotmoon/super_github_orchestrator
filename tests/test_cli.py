"""CLI default checks."""

from __future__ import annotations

from super_orchestrator.cli import build_parser
from super_orchestrator.graph import DEFAULT_MAX_CONTINUATIONS
from super_orchestrator.opencode import (
    DEFAULT_COMPACT_TOKENS,
    DEFAULT_IMPLEMENT_TIMEOUT_SECONDS,
    DEFAULT_REVIEW_TIMEOUT_SECONDS,
)


def test_agent_timeouts_default_to_one_hour_implement_slices() -> None:
    parser = build_parser()
    args = parser.parse_args(["--repo", "acme/app", "--workdir", "/tmp/app"])
    assert args.implement_timeout == DEFAULT_IMPLEMENT_TIMEOUT_SECONDS == 3_600
    assert args.max_continuations == DEFAULT_MAX_CONTINUATIONS == 3
    assert args.review_timeout == DEFAULT_REVIEW_TIMEOUT_SECONDS == 7_200
    assert args.compact_tokens == DEFAULT_COMPACT_TOKENS == 150_000


def test_zero_disables_timeouts_continuations_and_compaction() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--repo",
            "acme/app",
            "--workdir",
            "/tmp/app",
            "--implement-timeout",
            "0",
            "--review-timeout",
            "0",
            "--compact-tokens",
            "0",
            "--max-continuations",
            "0",
        ]
    )
    assert args.implement_timeout is None
    assert args.review_timeout is None
    assert args.compact_tokens == 0
    assert args.max_continuations == 0
