"""Unit tests for OpenCode user-message builders."""

from __future__ import annotations

from super_orchestrator.prompts import build_implement_message, build_review_message
from super_orchestrator.state import IssueInfo


def _sample_issue() -> IssueInfo:
    return IssueInfo(
        number=12,
        title="Add health check",
        body="Return 200 from /health",
        url="https://github.com/acme/app/issues/12",
    )


def test_build_implement_message_includes_issue_and_sentinel_instructions() -> None:
    message = build_implement_message(_sample_issue(), repo="acme/app", attempt=1)
    assert "https://github.com/acme/app" in message
    assert "issue #12" in message.lower() or "#12" in message
    assert "ISSUE_STATUS: PR_READY" in message
    assert "PR_URL:" in message
    assert "Return 200 from /health" in message


def test_build_implement_message_includes_review_feedback() -> None:
    message = build_implement_message(
        _sample_issue(),
        repo="acme/app",
        attempt=2,
        review_feedback="Add a unit test for /health",
    )
    assert "Add a unit test for /health" in message
    assert "Attempt: 2" in message


def test_build_review_message_includes_merge_method_and_verdict() -> None:
    message = build_review_message(
        _sample_issue(),
        repo="acme/app",
        pr_url="https://github.com/acme/app/pull/4",
        merge_method="squash",
        attempt=1,
    )
    assert "https://github.com/acme/app/pull/4" in message
    assert "Preferred merge method: squash" in message
    assert "REVIEW_VERDICT: APPROVE" in message
    assert "REVIEW_VERDICT: REQUEST_CHANGES" in message
