# super-orchestrator

Serial GitHub issue orchestrator. It walks open issues in ascending number order, blocks on `opencode run --agent` for implementation and for PR review/merge, closes each issue after merge, then continues until the queue is empty.

OpenCode does the model work. This package is a LangGraph control loop only (no second chat model).

## Assumptions

- Runs **inside a container** where `opencode` and `gh` are already on `PATH`.
- The **target repository** (the app under change) is a separate git checkout passed as `--workdir`.
- That checkout already has project subagents under `.opencode/agent/` (or `.opencode/agents/`), such as `impl-backend`, `impl-frontend`, `test`, and `review`.
- You have copied the primary agents and skills from [`templates/target/.opencode/`](templates/target/.opencode/) into the target repo.

## Install

```bash
pip install -e ".[dev]"
```

## One-time target setup

From the target repository root:

```bash
cp -R /path/to/github_do_all_issues_opencode/templates/target/.opencode/* .opencode/
# or merge agents/ and skills/ into an existing .opencode tree
```

Required primary agents (for `opencode run --agent`):

| Agent | Role |
| --- | --- |
| `issue-worker` | Implement, verify, commit, push, open PR |
| `pr-reviewer` | Review PR, merge or request changes |

Required skills:

| Skill | Used by |
| --- | --- |
| `ship-github-issue` | `issue-worker` |
| `review-and-merge-pr` | `pr-reviewer` |

Existing `mode: subagent` workers stay as-is. They must not commit or open PRs; only `issue-worker` does.

## Run

```bash
super-orchestrator \
  --repo owner/name \
  --workdir /path/to/target/checkout \
  --merge-method squash \
  --max-rounds 3
```

Or:

```bash
python -m super_orchestrator --repo owner/name --workdir /path/to/target/checkout
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--repo` | required | `owner/name` |
| `--workdir` | required | Target git checkout |
| `--merge-method` | `squash` | Told to the review agent |
| `--max-rounds` | `3` | Implement/review cycles per issue before halt |
| `--skip-label` | `orchestrator:skip` | Issues with this label are ignored |
| `--thread-id` | `super-orchestrator` | LangGraph checkpoint thread id |
| `--skip-preflight-agents` | off | Skip agent presence check |
| `--memory-checkpointer` | off | In-memory checkpoint (tests) |

## Loop

1. List open issues (`gh issue list`), lowest number first; skip `orchestrator:skip`.
2. If an open PR already exists on branch `issue-<n>-…`, skip implement and go to review.
3. Otherwise run `opencode run --agent issue-worker --auto --format json …` and wait.
4. On `ISSUE_STATUS: PR_READY` + `PR_URL`, run `opencode run --agent pr-reviewer …`.
5. On `REVIEW_VERDICT: APPROVE`, verify the PR is merged, close the issue, next issue.
6. On `REVIEW_VERDICT: REQUEST_CHANGES`, resume implement (same session when possible) until `--max-rounds` or success.
7. Halt (leave the issue open) on nonzero OpenCode exit, missing sentinels, or exhausted rounds.

Progress and resume state live under the target workdir:

```text
.orchestrator/
  state.sqlite
  logs/<issue>-implement-<n>.log
  logs/<issue>-review-<n>.log
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use fake `gh` / OpenCode runners. They do not call live OpenCode.
