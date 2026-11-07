"""User-message builders for OpenCode agent invocations."""

from __future__ import annotations

from typing import Literal

from super_orchestrator.state import IssueInfo


def build_implement_message(
    issue: IssueInfo,
    *,
    repo: str,
    attempt: int,
    review_feedback: str | None = None,
) -> str:
    """
    Build the user message for the issue-worker primary agent.

    The agent markdown supplies the system prompt; this message carries the
    concrete issue URL, body, and optional previous review feedback.
    """
    if not isinstance(repo, str) or repo.strip() == "":
        raise ValueError("repo must be a non-empty str")
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive int")
    _validate_issue(issue)

    lines: list[str] = [
        f"You are an agentic orchestrator working on https://github.com/{repo}.",
        "",
        f"Implement GitHub issue #{issue['number']}: {issue['title']}",
        f"Issue URL: {issue['url']}",
        f"Attempt: {attempt}",
        "",
        "Issue body:",
        issue["body"] if issue["body"].strip() != "" else "(empty)",
        "",
        "Follow the ship-github-issue skill.",
        "Use existing project subagents (impl-backend, impl-frontend, test, review) as needed.",
        "Only you may commit, push, and open the pull request.",
        "When the PR is ready, end with exactly:",
        "ISSUE_STATUS: PR_READY",
        "PR_URL: <https url of the pull request>",
    ]
    if review_feedback is not None and review_feedback.strip() != "":
        lines.extend(
            [
                "",
                "The previous PR review requested changes. Address ALL of the following:",
                review_feedback.strip(),
            ]
        )
    return "\n".join(lines)


def build_review_message(
    issue: IssueInfo,
    *,
    repo: str,
    pr_url: str,
    merge_method: Literal["squash", "merge", "rebase"],
    attempt: int,
) -> str:
    """
    Build the user message for the pr-reviewer primary agent.

    Merge vs request-changes is performed by the agent; this message only
    supplies context and the required sentinel format.
    """
    if not isinstance(repo, str) or repo.strip() == "":
        raise ValueError("repo must be a non-empty str")
    if not isinstance(pr_url, str) or pr_url.strip() == "":
        raise ValueError("pr_url must be a non-empty str")
    if merge_method not in {"squash", "merge", "rebase"}:
        raise ValueError("merge_method must be squash, merge, or rebase")
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive int")
    _validate_issue(issue)

    lines: list[str] = [
        f"You are reviewing a pull request for https://github.com/{repo}.",
        "",
        f"Issue #{issue['number']}: {issue['title']}",
        f"Issue URL: {issue['url']}",
        f"Pull request: {pr_url}",
        f"Review attempt: {attempt}",
        f"Preferred merge method: {merge_method}",
        "",
        "Issue body:",
        issue["body"] if issue["body"].strip() != "" else "(empty)",
        "",
        "Follow the review-and-merge-pr skill.",
        "You may use the existing review subagent for a read-only code review.",
        "If the PR fully satisfies the issue, merge it with the preferred method.",
        "If it does not, request changes with concrete actionable feedback.",
        "Do not edit repository files.",
        "End with exactly one of:",
        "REVIEW_VERDICT: APPROVE",
        "or",
        "REVIEW_VERDICT: REQUEST_CHANGES",
        "After the verdict line, include a short summary (and blockers if requesting changes).",
    ]
    return "\n".join(lines)


def _validate_issue(issue: IssueInfo) -> None:
    """Type-check the required IssueInfo fields."""
    if not isinstance(issue, dict):
        raise TypeError("issue must be a dict-like IssueInfo")
    number = issue.get("number")
    title = issue.get("title")
    body = issue.get("body")
    url = issue.get("url")
    if not isinstance(number, int) or number < 1:
        raise ValueError("issue.number must be a positive int")
    if not isinstance(title, str):
        raise ValueError("issue.title must be a str")
    if not isinstance(body, str):
        raise ValueError("issue.body must be a str")
    if not isinstance(url, str) or url.strip() == "":
        raise ValueError("issue.url must be a non-empty str")
