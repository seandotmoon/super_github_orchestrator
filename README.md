# super-orchestrator

Serial GitHub issue orchestrator. It walks open issues in ascending number order, blocks on `opencode run --agent` for implementation and for PR review/merge, closes each issue after merge, then continues until the queue is empty.

OpenCode does the model work. This package is a LangGraph control loop only (no second chat model).

## Assumptions

- Runs **inside a container** where `opencode`, `gh`, and `git` are already on `PATH`.
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

`pr-reviewer` denies file edits, common write/push shell commands, and shell redirection via `permission`. Because the orchestrator runs OpenCode with `--auto` (approve anything not denied), keep that deny list in place and extend it for your repo's tooling. Pattern matching is best-effort (an interpreter one-liner can still write files), so rely on GitHub branch protection and a scoped `GH_TOKEN` as the real guardrails.

Instead of copying per repo, you can install the agents and skills globally under `~/.config/opencode/agents/` and `~/.config/opencode/skills/`; the Docker image below does this.

## Docker

The agentic image installs `super-orchestrator` from this directory through a named build context, so only `pyproject.toml`, `README.md`, `src/`, and `templates/` are sent to the builder. From the directory that contains the Dockerfile:

```bash
docker build -f Dockerfile-agentic.agent \
  --build-context orchestrator=./github_do_all_issues_opencode \
  -t smart-agent-agentic .
```

With the `agent` shell launcher, `agent agentic /path/to/repo` starts this image with `GH_TOKEN` and the git identity already set; then run `super-orchestrator --repo owner/name --workdir /workspace` inside it. Otherwise:

Run it with a GitHub token and a git identity for the worker's commits:

```bash
docker run --rm -it \
  -e GH_TOKEN=ghp_... \
  -e GIT_AUTHOR_NAME="Orchestrator Bot" -e GIT_AUTHOR_EMAIL=bot@example.com \
  -e GIT_COMMITTER_NAME="Orchestrator Bot" -e GIT_COMMITTER_EMAIL=bot@example.com \
  -v "$PWD/my-app:/workspace/my-app" \
  smart-agent-agentic \
  super-orchestrator --repo owner/my-app --workdir /workspace/my-app
```

The image configures git to authenticate HTTPS pushes through `gh auth git-credential`, so `GH_TOKEN` covers both `gh` and `git push`.

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
| `--on-halt` | `stop` | `stop` ends the run on a halt; `skip` labels the issue with `--skip-label`, comments the reason, and continues |
| `--allow-author` | authenticated `gh` user | Only process issues opened by these logins (repeatable) |
| `--allow-any-author` | off | Process issues from anyone (see Security) |
| `--require-label` | none | Only process issues carrying this label |
| `--implement-timeout` | `3600` | Seconds one `issue-worker` run may take before it is stopped and continued (`0` disables) |
| `--max-continuations` | `3` | Extra `issue-worker` runs per implement attempt after a timeout; `0` halts on the first timeout |
| `--review-timeout` | `7200` | Seconds before `pr-reviewer` is killed (`0` disables) |
| `--compact-tokens` | `150000` | Compact a session when its context reaches this many tokens (`0` disables) |
| `--merge-wait` | `600` | Seconds to wait after `APPROVE` for the PR to show as merged (auto-merge / merge queue) |
| `--thread-id` | `super-orchestrator` | LangGraph checkpoint thread id |
| `--fresh` | off | Ignore an interrupted checkpoint and start over with the given flags |
| `--skip-preflight-agents` | off | Skip agent presence check |
| `--memory-checkpointer` | off | In-memory checkpoint (tests) |
| `--quiet` | off | Print only orchestrator progress, not live agent activity |
| `--heartbeat` | `120` | Seconds of agent silence before a "still running" line is printed |

### Time slices

`opencode run` blocks while a subagent works, and subagent events are not in the run's JSON stream. A subagent stuck in a loop (for example a local model repeating invalid tool calls) can hold the run for hours with no visible activity. To bound that, each `issue-worker` run is a slice of at most `--implement-timeout` seconds:

1. When the time is up, the orchestrator kills the run (and any subagent inside it). Edits stay in the shared checkout.
2. It checks GitHub for a PR the stopped run may have opened.
3. It starts the next slice. That slice resumes the same OpenCode session with a short `CONTINUATION` message when the session id is known. Otherwise it starts a fresh session with the full issue plus the handoff. Either way the worker is told to inspect `git status`/`git diff`, keep finished work, and delegate only what is left, as smaller tasks.
4. After `1 + --max-continuations` slices without `ISSUE_STATUS: PR_READY`, the issue halts (see `--on-halt`).

Slices only bound a stuck subagent from the outside. Also cap each implementation subagent with OpenCode's `steps` frontmatter (for example `steps: 80` in `impl-backend.md`). On its last step OpenCode removes the tools and makes the agent summarize, so a loop returns "done X, remaining Y" to `issue-worker` within minutes. The `ship-github-issue` skill tells the worker to delegate only the remaining part.

A run killed right after it printed the PR sentinels counts as finished. The slice counter resets for each review change round. Slice logs are named `<issue>-implement-<attempt>-slice<n>.log` from the second slice on. Reviews are not sliced: `--review-timeout` still halts.

### Context compaction

OpenCode already summarizes a session when it nears the model's context limit. A long issue often gets slow well before that. While `issue-worker` or `pr-reviewer` is running, each `step_finish` token count in that run's JSON stream is checked. That covers the primary session and any other session OpenCode includes in the stream (a subagent, when its events are forwarded). At `--compact-tokens` the orchestrator runs:

```bash
opencode api post /api/session/<id>/compact --data '{}'
```

The command only admits the request. OpenCode applies the summary at the next step boundary, so the agent keeps working. The same session is left alone until a later step reports context under the threshold. A failed request is retried after 60 seconds. Sessions that never appear in the run log stay on OpenCode's own auto-compact. OpenCode builds without `opencode api` answer with their help text. The orchestrator then prints one `compaction disabled` line and stops trying for the rest of the run.

The count is `tokens.total` when OpenCode sends it. Otherwise it is input + output + reasoning + cache read + cache write.

### Console output

Every line is timestamped and flushed immediately. Orchestrator progress shows the queue, the issue being worked, which agent runs which attempt, how long it took, the reviewer verdict, merge waiting, and the closed/skipped totals. Live agent activity (tool calls, subagent launches, todo progress, and the agent's own status lines) is tailed from the log file while the agent runs:

```text
[14:02:11] Queue: 3 eligible issue(s): #9, #12, #14
[14:02:12] ━━━ Issue #9: Admin user management ━━━
[14:02:12] #9 implement attempt 1/3 → agent issue-worker
[14:02:12] #9 implement 1 · issue-worker │ log: /workspace/.orchestrator/logs/9-implement-1.log
[14:02:40] #9 implement 1 · issue-worker │ skill ship-github-issue
[14:03:05] #9 implement 1 · issue-worker │ subagent impl-backend: Add role endpoints
[14:09:51] #9 implement 1 · issue-worker │ bash npm test
[14:12:02] #9 implement 1 · issue-worker │ … still running (9m50s elapsed, 41 actions so far)
[14:15:30] #9 issue-worker finished in 13m18s → PR ready: https://github.com/OWNER/REPO/pull/31
[14:15:30] #9 review attempt 1/3 → agent pr-reviewer on https://github.com/OWNER/REPO/pull/31
```

The full raw NDJSON stays in `.orchestrator/logs/`.

Exit code is `0` when every issue was closed, `1` on a halt, an error, or when any issue was skipped under `--on-halt skip`.

## Loop

1. List eligible open issues (`gh issue list`), lowest number first; skip `--skip-label`, unlisted authors, and (if set) issues without `--require-label`.
2. If an open PR authored by the `gh` user already exists on branch `issue-<n>-…` in the same repo, skip implement and go to review.
3. Otherwise run `opencode run --agent issue-worker --auto --format json …` and wait.
4. On `ISSUE_STATUS: PR_READY` + `PR_URL`, run `opencode run --agent pr-reviewer …`.
5. On `REVIEW_VERDICT: APPROVE`, wait (up to `--merge-wait`) for the PR to be merged, close the issue, then **reload the queue** so newly opened issues are picked up.
6. On `REVIEW_VERDICT: REQUEST_CHANGES`, re-run implement with the feedback and the existing PR URL (same session when possible) until `--max-rounds` or success. After pushing the fixes, `issue-worker` posts a PR comment listing each blocker and how it was addressed, plus the verification it ran.
7. When an implement run hits `--implement-timeout`, continue in a new slice (see Time slices) up to `--max-continuations` times.
8. Halt on nonzero OpenCode exit, review timeout, every implement slice timing out, missing sentinels, exhausted rounds, or GitHub errors. With `--on-halt stop` the run ends and the issue stays open; with `--on-halt skip` the issue is labelled and the loop continues. Remove the label to retry it later.

## Resume

State is checkpointed after every step. If the process dies mid-run (container killed, crash), the next invocation with the same `--thread-id` resumes at the node that was running, using the settings saved with that run. Pass `--fresh` to discard it. A run that finished (done or halted) always starts fresh: the queue is reloaded from GitHub, and an issue with an existing orchestrator PR goes straight to review.

Progress and resume state live under the target workdir. Preflight adds `/.orchestrator/` to the checkout's local `.git/info/exclude` so agents never commit it:

```text
.orchestrator/
  state.sqlite
  logs/<issue>-implement-<n>.log
  logs/<issue>-implement-<n>-slice<k>.log
  logs/<issue>-review-<n>.log
```

## Security

Issue bodies are fed to agents that run with `--auto` and can merge to your default branch, which makes them a prompt-injection vector. By default only issues opened by the authenticated `gh` user are processed. Widen this deliberately with `--allow-author` or `--require-label` (a label only maintainers can apply). Use `--allow-any-author` only on private repositories you trust. Prompts also fence issue bodies as untrusted content, but treat that as defense in depth, not a guarantee.

The issue-worker and pr-reviewer use the same `gh` identity, and GitHub does not allow that identity to formally request changes on its own PR. The review skill falls back to a PR comment in that case.

## Tests

```bash
pip install -e ".[dev]"
pytest
mypy
```

Tests use fake `gh` / OpenCode runners and real OpenCode NDJSON event shapes. They do not call live OpenCode.
