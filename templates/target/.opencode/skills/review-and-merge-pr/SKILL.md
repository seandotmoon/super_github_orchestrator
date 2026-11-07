---
name: review-and-merge-pr
description: Review an open pull request against its GitHub issue, then merge or request changes. Use when acting as pr-reviewer for super-orchestrator.
---

# Review and merge (or request changes)

## Preconditions

- The user message provides the issue URL/body, the PR URL, the review attempt number, and the preferred merge method (`squash`, `merge`, or `rebase`).
- You must not edit repository files.

## Workflow

1. Read the issue acceptance criteria.
2. Inspect the PR with `gh pr view <url>` and `gh pr diff <url>` (and checks if available via `gh pr checks`).
3. Optionally launch the existing `review` subagent with the issue criteria and ask for APPROVE vs REQUEST_CHANGES style findings.
4. Decide:
   - **Approve and merge** only when the PR fully satisfies the issue, tests/checks look acceptable, and there are no real blockers.
   - **Request changes** when anything material is wrong, missing, or unverified.

## Approve path

1. Merge with the preferred method, for example:
   - `gh pr merge <url> --squash --delete-branch`
   - or `--merge` / `--rebase` when requested.
2. Confirm the PR is merged (`gh pr view <url> --json state,mergedAt`).
3. End with:

```
REVIEW_VERDICT: APPROVE
<summary of why it was safe to merge>
```

## Request-changes path

1. Post review feedback:
   - `gh pr review <url> --request-changes --body "..."` with concrete blockers (file references when possible).
2. Do not merge.
3. End with:

```
REVIEW_VERDICT: REQUEST_CHANGES
<numbered blockers the issue-worker must fix>
```

## Do not

- Edit files or push commits.
- Approve when checks are failing for reasons related to this PR.
- Emit `REVIEW_VERDICT: APPROVE` unless the merge already succeeded.
