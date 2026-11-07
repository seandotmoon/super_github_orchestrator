---
name: ship-github-issue
description: Implement one GitHub issue, verify it, commit, push a branch named issue-<number>-<slug>, and open a pull request. Use when acting as issue-worker for super-orchestrator.
---

# Ship one GitHub issue

## Preconditions

- The user message names a single issue URL, number, title, and body.
- `gh` and `git` are available in this working directory.
- Existing project subagents (impl-backend, impl-frontend, test, review) may be used for coding and verification.

## Workflow

1. Read the issue body carefully. Extract acceptance criteria.
2. Sync the default branch (`git fetch origin` and check out the repo default branch; pull latest).
3. Create a branch named `issue-<number>-<short-slug>` where `<short-slug>` is a lowercase kebab summary of the title (max ~40 chars).
4. Implement the change:
   - Prefer launching the matching impl subagent (backend vs frontend) with a precise task brief.
   - Keep the diff minimal and consistent with existing patterns.
5. Verify:
   - Prefer launching the `test` subagent (or run the project’s documented lint/typecheck/test commands).
   - Iterate until verification is green, or stop with a clear failure explanation (no PR).
6. Optionally launch the `review` subagent for a read-only self-check; fix blockers before committing.
7. Commit with a concise message that references the issue (for example `fix #12: short summary`).
8. Push the branch to `origin` with upstream tracking.
9. Open a PR with `gh pr create`:
   - Title: concise summary
   - Body: must include `Closes #<number>` (or `Fixes #<number>`) and a short description of what changed and how it was verified.
10. Print the final sentinel block and stop:

```
ISSUE_STATUS: PR_READY
PR_URL: https://github.com/<owner>/<repo>/pull/<n>
```

## Change rounds

When the user message includes previous review feedback, address every blocker, push new commits to the same branch (resume the same session when possible), and re-emit the sentinel with the same PR URL.

## Do not

- Commit or push from a subagent session.
- Open PRs for unrelated work.
- Emit `ISSUE_STATUS: PR_READY` unless the PR URL is real and open.
