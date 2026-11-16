"""Unit tests for OpenCode NDJSON and sentinel parsing."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from super_orchestrator.opencode import (
    COMPACT_REQUEST_TIMEOUT_SECONDS,
    TIMEOUT_EXIT_CODE,
    CompactionWatch,
    NdjsonParseState,
    OpenCodeClient,
    OpenCodeTimeout,
    LiveOutput,
    ProcessRunner,
    SubprocessRunner,
    ingest_ndjson_line,
    parse_agent_ids,
    parse_context_usages,
    parse_implement_sentinels,
    parse_review_sentinel,
)


class RecordingRunner:
    """Fake ProcessRunner that records commands and writes scripted stdout."""

    def __init__(self, lines: list[str], exit_code: int = 0) -> None:
        self.lines = lines
        self.exit_code = exit_code
        self.commands: list[list[str]] = []
        self.timeouts: list[float | None] = []
        self.api_exit_code = 0
        self.api_output: list[str] = []
        # When set, the agent command writes its lines and then times out.
        self.time_out = False

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        log_path: Path,
        on_stdout_line: Callable[[str], None] | None = None,
        timeout_seconds: float | None = None,
        live: LiveOutput | None = None,
    ) -> int:
        del cwd, env
        self.commands.append(list(command))
        self.timeouts.append(timeout_seconds)
        # Compaction is a side command issued from the live tail. Replaying the
        # agent script here would recurse back into this runner.
        if len(command) > 1 and command[1] == "api":
            if on_stdout_line is not None:
                for line in self.api_output:
                    on_stdout_line(line)
            return self.api_exit_code
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            for line in self.lines:
                handle.write(f"{line}\n")
                if on_stdout_line is not None:
                    on_stdout_line(line)
                if live is not None:
                    live.on_line(line)
        if self.time_out:
            raise OpenCodeTimeout(f"{command[0]} exceeded {timeout_seconds}s and was killed")
        return self.exit_code


def _event(event_type: str, part: dict[str, object], session: str = "ses_abc") -> str:
    """Render one event in the shape emitted by `opencode run --format json`."""
    return json.dumps(
        {"type": event_type, "timestamp": 1, "sessionID": session, "part": part}
    )


def _text(text: str, part_id: str = "prt_1") -> str:
    """Render a `text` event delta for a part."""
    return _event(
        "text",
        {"id": part_id, "sessionID": "ses_abc", "messageID": "msg_1", "type": "text", "text": text},
    )


def test_parse_implement_sentinels_happy_path() -> None:
    text = "\n".join(
        [
            "Opened the PR.",
            "ISSUE_STATUS: PR_READY",
            "PR_URL: https://github.com/acme/app/pull/7",
        ]
    )
    status, url = parse_implement_sentinels(text)
    assert status == "PR_READY"
    assert url == "https://github.com/acme/app/pull/7"


def test_parse_implement_sentinels_missing() -> None:
    status, url = parse_implement_sentinels("still working")
    assert status is None
    assert url is None


@pytest.mark.parametrize(
    "url_line",
    [
        "PR_URL: <https://github.com/acme/app/pull/7>",
        "PR_URL: `https://github.com/acme/app/pull/7`",
        "**PR_URL:** https://github.com/acme/app/pull/7",
        "  PR_URL: https://github.com/acme/app/pull/7  ",
    ],
)
def test_parse_implement_sentinels_tolerates_formatting(url_line: str) -> None:
    status, url = parse_implement_sentinels(f"**ISSUE_STATUS: PR_READY**\n{url_line}")
    assert status == "PR_READY"
    assert url == "https://github.com/acme/app/pull/7"


def test_parse_implement_sentinels_uses_last_url() -> None:
    text = "\n".join(
        [
            "PR_URL: https://github.com/acme/app/pull/1",
            "That PR was closed; opened a new one.",
            "ISSUE_STATUS: PR_READY",
            "PR_URL: https://github.com/acme/app/pull/2",
        ]
    )
    assert parse_implement_sentinels(text)[1] == "https://github.com/acme/app/pull/2"


def test_parse_review_sentinel_approve() -> None:
    assert parse_review_sentinel("REVIEW_VERDICT: APPROVE\nok") == "APPROVE"


def test_parse_review_sentinel_request_changes() -> None:
    assert (
        parse_review_sentinel("notes\nREVIEW_VERDICT: REQUEST_CHANGES\n- fix tests")
        == "REQUEST_CHANGES"
    )


def test_parse_review_sentinel_uses_final_verdict() -> None:
    text = "\n".join(
        [
            "Plan: if everything passes I will print",
            "REVIEW_VERDICT: APPROVE",
            "but tests fail, so:",
            "REVIEW_VERDICT: REQUEST_CHANGES",
            "1. fix tests",
        ]
    )
    assert parse_review_sentinel(text) == "REQUEST_CHANGES"


def test_ingest_real_opencode_text_event() -> None:
    state = NdjsonParseState()
    for line in [
        _event("step_start", {"id": "prt_s", "type": "step-start"}),
        _event(
            "tool_use",
            {"id": "prt_t", "type": "tool", "tool": "shell", "state": {"status": "completed"}},
        ),
        _text("Done.\nISSUE_STATUS: PR_READY\nPR_URL: https://github.com/a/b/pull/1"),
        _event("step_finish", {"id": "prt_f", "type": "step-finish"}),
    ]:
        ingest_ndjson_line(state, line)
    assert state.session_id == "ses_abc"
    assert parse_implement_sentinels(state.assistant_text) == (
        "PR_READY",
        "https://github.com/a/b/pull/1",
    )


def test_ingest_concatenates_deltas_for_same_part() -> None:
    """Deltas split mid-line must not be joined with newlines."""
    state = NdjsonParseState()
    for chunk in ["Done.\nISSUE_STA", "TUS: PR_READY\nPR_URL: https://github", ".com/a/b/pull/9"]:
        ingest_ndjson_line(state, _text(chunk, part_id="prt_x"))
    ingest_ndjson_line(state, _text("separate part", part_id="prt_y"))
    assert state.assistant_chunks[0].endswith("PR_URL: https://github.com/a/b/pull/9")
    assert state.assistant_chunks[1] == "separate part"
    assert parse_implement_sentinels(state.assistant_text)[1] == "https://github.com/a/b/pull/9"


def test_ingest_ignores_reasoning_and_records_errors() -> None:
    state = NdjsonParseState()
    ingest_ndjson_line(
        state,
        _event("reasoning", {"id": "prt_r", "type": "reasoning", "text": "REVIEW_VERDICT: APPROVE"}),
    )
    ingest_ndjson_line(
        state,
        json.dumps({"type": "error", "sessionID": "ses_abc", "error": {"message": "rate limited"}}),
    )
    assert state.assistant_text == ""
    assert state.errors == ["rate limited"]


def test_ingest_legacy_shapes_still_supported() -> None:
    state = NdjsonParseState()
    ingest_ndjson_line(
        state,
        json.dumps({"type": "session", "sessionID": "ses_abc", "text": "hello"}),
    )
    ingest_ndjson_line(
        state,
        json.dumps(
            {
                "type": "message",
                "properties": {"part": {"type": "text", "text": "world"}},
            }
        ),
    )
    assert state.session_id == "ses_abc"
    assert "hello" in state.assistant_text
    assert "world" in state.assistant_text


def test_parse_agent_ids_from_json() -> None:
    output = json.dumps([{"id": "build"}, {"id": "issue-worker"}, {"id": "pr-reviewer-old"}])
    ids = parse_agent_ids(output)
    assert "issue-worker" in ids
    assert "pr-reviewer" not in ids


def test_parse_agent_ids_from_agent_list_output() -> None:
    """Shape printed by `opencode agent list` in released builds (1.19.x)."""
    output = "\n".join(
        [
            "build (primary)",
            '  [{"permission": "external_directory",',
            '    "pattern": "/root/.config/opencode/skills/pr-reviewer/*"}]',
            "explore (subagent)",
            "issue-worker (primary)",
            "pr-reviewer-old (primary)",
        ]
    )
    assert parse_agent_ids(output) == {"build", "explore", "issue-worker", "pr-reviewer-old"}


def test_list_agents_falls_back_to_debug_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class SequencedRunner(RecordingRunner):
        def run(
            self,
            command: Sequence[str],
            *,
            cwd: Path,
            env: Mapping[str, str] | None,
            log_path: Path,
            on_stdout_line: Callable[[str], None] | None = None,
            timeout_seconds: float | None = None,
            live: LiveOutput | None = None,
        ) -> int:
            self.commands.append(list(command))
            if list(command[1:]) == ["agent", "list"]:
                return 1
            if on_stdout_line is not None:
                on_stdout_line(json.dumps([{"id": "issue-worker"}, {"id": "pr-reviewer"}]))
            return 0

    monkeypatch.setattr("super_orchestrator.opencode.shutil.which", lambda name: f"/bin/{name}")
    runner = SequencedRunner(lines=[])
    client = OpenCodeClient(workdir=tmp_path, runner=runner)
    assert client.list_agent_ids() == {"issue-worker", "pr-reviewer"}
    assert [command[1:] for command in runner.commands] == [["agent", "list"], ["debug", "agents"]]


def test_run_implement_builds_command_and_parses(tmp_path: Path) -> None:
    lines = [
        _text("working", part_id="prt_a"),
        _text("ISSUE_STATUS: PR_READY\nPR_URL: https://github.com/acme/app/pull/3", part_id="prt_b"),
    ]
    runner = RecordingRunner(lines=lines, exit_code=0)
    client = OpenCodeClient(workdir=tmp_path, runner=runner, implement_timeout_seconds=60)
    result = client.run_implement(issue_number=3, attempt=1, message="Implement issue 3")
    assert result.exit_code == 0
    assert result.session_id == "ses_abc"
    assert result.status == "PR_READY"
    assert result.pr_url == "https://github.com/acme/app/pull/3"
    command = runner.commands[0]
    assert command[:7] == ["opencode", "run", "--agent", "issue-worker", "--auto", "--format", "json"]
    # Released OpenCode (1.19.x) rejects unknown flags such as --archived.
    assert "--archived" not in command
    assert command[-1] == "Implement issue 3"
    assert runner.timeouts == [60]


def test_run_implement_resumes_session(tmp_path: Path) -> None:
    runner = RecordingRunner(
        lines=["ISSUE_STATUS: PR_READY", "PR_URL: https://github.com/acme/app/pull/3"],
        exit_code=0,
    )
    client = OpenCodeClient(workdir=tmp_path, runner=runner)
    client.run_implement(
        issue_number=3,
        attempt=2,
        message="Address review feedback",
        session_id="ses_prior",
    )
    command = runner.commands[0]
    assert "--session" in command
    assert command[command.index("--session") + 1] == "ses_prior"


def test_run_review_parses_verdict(tmp_path: Path) -> None:
    runner = RecordingRunner(
        lines=[
            _text(
                "Looks good overall.\nREVIEW_VERDICT: REQUEST_CHANGES\n1. Add a null check in src/a.ts"
            )
        ],
        exit_code=0,
    )
    client = OpenCodeClient(workdir=tmp_path, runner=runner, review_timeout_seconds=30)
    result = client.run_review(issue_number=3, attempt=1, message="Review PR")
    assert result.verdict == "REQUEST_CHANGES"
    assert "null check" in result.feedback
    assert runner.timeouts == [30]


def test_subprocess_runner_streams_and_returns_exit_code(tmp_path: Path) -> None:
    seen: list[str] = []
    code = SubprocessRunner().run(
        [sys.executable, "-c", "print('a'); print('b'); raise SystemExit(3)"],
        cwd=tmp_path,
        env=None,
        log_path=tmp_path / "out.log",
        on_stdout_line=seen.append,
        timeout_seconds=30,
    )
    assert code == 3
    assert seen == ["a", "b"]
    assert "[exit_code=3]" in (tmp_path / "out.log").read_text(encoding="utf-8")


def test_subprocess_runner_keeps_output_when_child_exits_abruptly(tmp_path: Path) -> None:
    """Large output followed by an immediate exit must be replayed in full."""
    script = "\n".join(
        [
            "import os, sys",
            "for i in range(20000):",
            "    sys.stdout.write(f'line-{i}-' + 'x' * 40 + '\\n')",
            "sys.stdout.write('ISSUE_STATUS: PR_READY\\n')",
            "sys.stdout.flush()",
            "os._exit(0)",
        ]
    )
    seen: list[str] = []
    code = SubprocessRunner().run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=None,
        log_path=tmp_path / "big.log",
        on_stdout_line=seen.append,
        timeout_seconds=60,
    )
    assert code == 0
    assert len(seen) == 20001
    assert seen[-1] == "ISSUE_STATUS: PR_READY"


def _tokens(
    *,
    total: int | None = None,
    input_tokens: int = 0,
    output: int = 0,
    reasoning: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, object]:
    """Build an OpenCode token object, with `total` only when the caller sets it."""
    payload: dict[str, object] = {
        "input": input_tokens,
        "output": output,
        "reasoning": reasoning,
        "cache": {"read": cache_read, "write": cache_write},
    }
    if total is not None:
        payload["total"] = total
    return payload


def _step_finish(session: str, tokens: dict[str, object]) -> str:
    """Render a step_finish event for one session."""
    return _event("step_finish", {"type": "step-finish", "tokens": tokens}, session=session)


def test_parse_context_usages_reads_step_finish_tokens() -> None:
    with_total = _step_finish("ses_abc", _tokens(total=151_000, input_tokens=1))
    assert parse_context_usages(with_total) == [("ses_abc", 151_000)]
    summed = _step_finish(
        "ses_child",
        _tokens(input_tokens=1_000, output=10, cache_read=160_000, cache_write=5),
    )
    assert parse_context_usages(summed) == [("ses_child", 161_015)]
    assert parse_context_usages("not json") == []
    assert parse_context_usages(_text("hello")) == []
    unsafe = json.dumps(
        {
            "type": "step_finish",
            "sessionID": "../etc",
            "part": {"type": "step-finish", "tokens": _tokens(total=200_000)},
        }
    )
    assert parse_context_usages(unsafe) == []


def test_compaction_watch_latches_until_context_shrinks() -> None:
    requested: list[str] = []
    clock = {"now": 0.0}

    def request(session_id: str) -> bool:
        requested.append(session_id)
        return True

    watch = CompactionWatch(
        threshold=150_000,
        request=request,
        announce=lambda _message: None,
        now=lambda: clock["now"],
        retry_seconds=60,
    )
    watch.observe("ses_a", 149_999)
    watch.observe("ses_a", 150_000)
    watch.observe("ses_a", 180_000)
    watch.observe("ses_child", 150_000)
    clock["now"] = 10.0
    watch.observe("ses_a", 140_000)
    watch.observe("ses_a", 151_000)
    assert requested == ["ses_a", "ses_child", "ses_a"]


def test_compaction_watch_retries_a_rejected_request_after_the_backoff() -> None:
    requested: list[str] = []
    clock = {"now": 0.0}

    def request(session_id: str) -> bool:
        requested.append(session_id)
        return False

    watch = CompactionWatch(
        threshold=150_000,
        request=request,
        announce=lambda _message: None,
        now=lambda: clock["now"],
        retry_seconds=60,
    )
    watch.observe("ses_a", 150_000)
    clock["now"] = 30.0
    watch.observe("ses_a", 160_000)
    clock["now"] = 60.0
    watch.observe("ses_a", 160_000)
    assert requested == ["ses_a", "ses_a"]


def test_run_compacts_each_session_over_threshold_once(tmp_path: Path) -> None:
    under = _tokens(input_tokens=1_000, output=10)
    over = _tokens(input_tokens=140_000, output=1_000, cache_read=20_000)
    lines = [
        _step_finish("ses_parent", under),
        _step_finish("ses_parent", over),
        _step_finish("ses_parent", over),
        _step_finish("ses_child", over),
        _step_finish("ses_parent", under),
        _step_finish("ses_parent", over),
    ]
    runner = RecordingRunner(lines=lines)
    notes: list[str] = []
    client = OpenCodeClient(
        workdir=tmp_path,
        runner=runner,
        status=notes.append,
        compact_tokens=150_000,
        implement_timeout_seconds=60,
    )
    client.run_implement(issue_number=13, attempt=1, message="Implement issue 13")
    api = [command for command in runner.commands if len(command) > 1 and command[1] == "api"]
    assert [command[3] for command in api] == [
        "/api/session/ses_parent/compact",
        "/api/session/ses_child/compact",
        "/api/session/ses_parent/compact",
    ]
    assert all(command[4:6] == ["--data", "{}"] for command in api)
    assert all(timeout == COMPACT_REQUEST_TIMEOUT_SECONDS for timeout in runner.timeouts[1:])
    assert any("compacting ses_parent at 161000 tokens" in note for note in notes)
    assert any("compacting ses_child at 161000 tokens" in note for note in notes)


def test_run_review_compacts_and_a_failure_is_not_retried_immediately(tmp_path: Path) -> None:
    runner = RecordingRunner(
        lines=[
            _step_finish("ses_review", _tokens(total=150_000)),
            _step_finish("ses_review", _tokens(total=160_000)),
        ],
    )
    runner.api_exit_code = 1
    notes: list[str] = []
    client = OpenCodeClient(
        workdir=tmp_path,
        runner=runner,
        status=notes.append,
        compact_tokens=150_000,
        review_timeout_seconds=30,
    )
    client.run_review(issue_number=13, attempt=1, message="Review the PR")
    api = [command for command in runner.commands if len(command) > 1 and command[1] == "api"]
    assert api == [
        ["opencode", "api", "post", "/api/session/ses_review/compact", "--data", "{}"]
    ]
    assert any("compact failed for ses_review" in note for note in notes)


def test_compaction_can_be_disabled(tmp_path: Path) -> None:
    runner = RecordingRunner(lines=[_step_finish("ses_abc", _tokens(total=200_000))])
    client = OpenCodeClient(
        workdir=tmp_path,
        runner=runner,
        compact_tokens=None,
        implement_timeout_seconds=60,
    )
    client.run_implement(issue_number=3, attempt=1, message="Implement issue 3")
    assert [command[1] for command in runner.commands] == ["run"]


def test_run_implement_reports_timeout_as_an_unfinished_slice(tmp_path: Path) -> None:
    runner = RecordingRunner(lines=[_text("Dispatching the backend subagent")])
    runner.time_out = True
    client = OpenCodeClient(workdir=tmp_path, runner=runner, implement_timeout_seconds=60)
    result = client.run_implement(
        issue_number=17, attempt=1, message="Continue issue 17", slice_number=2
    )
    assert result.timed_out is True
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert result.session_id == "ses_abc"
    assert result.status is None
    assert result.log_path == tmp_path / ".orchestrator" / "logs" / "17-implement-1-slice2.log"


def test_run_implement_rejects_bad_slice_number(tmp_path: Path) -> None:
    client = OpenCodeClient(workdir=tmp_path, runner=RecordingRunner(lines=[]))
    with pytest.raises(ValueError, match="slice_number"):
        client.run_implement(issue_number=1, attempt=1, message="m", slice_number=0)


def test_run_review_timeout_still_raises(tmp_path: Path) -> None:
    runner = RecordingRunner(lines=[_text("reviewing")])
    runner.time_out = True
    client = OpenCodeClient(workdir=tmp_path, runner=runner, review_timeout_seconds=30)
    with pytest.raises(OpenCodeTimeout, match="pr-reviewer exceeded 30s"):
        client.run_review(issue_number=3, attempt=1, message="Review PR")


def test_compaction_turns_off_when_opencode_has_no_api_command(tmp_path: Path) -> None:
    over = _step_finish("ses_parent", _tokens(total=200_000))
    under = _step_finish("ses_parent", _tokens(total=1_000))
    runner = RecordingRunner(lines=[over, under, over])
    runner.api_exit_code = 1
    runner.api_output = [
        "Commands:",
        "  opencode run [message..]     run opencode with a message",
        "Options:",
    ]
    notes: list[str] = []
    client = OpenCodeClient(
        workdir=tmp_path,
        runner=runner,
        status=notes.append,
        compact_tokens=150_000,
        implement_timeout_seconds=60,
    )
    client.run_implement(issue_number=13, attempt=1, message="Implement issue 13")
    client.run_implement(issue_number=14, attempt=1, message="Implement issue 14")
    api = [command for command in runner.commands if len(command) > 1 and command[1] == "api"]
    assert len(api) == 1
    disabled = [note for note in notes if "compaction disabled" in note]
    assert len(disabled) == 1
    assert not any("compact failed" in note for note in notes)


def test_subprocess_runner_kills_on_timeout(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(OpenCodeTimeout):
        SubprocessRunner().run(
            [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"],
            cwd=tmp_path,
            env=None,
            log_path=tmp_path / "slow.log",
            timeout_seconds=0.5,
        )
    assert time.monotonic() - started < 10
    assert "timed out" in (tmp_path / "slow.log").read_text(encoding="utf-8")


# Satisfy type checkers that RecordingRunner matches ProcessRunner structurally.
def _assert_runner_protocol(runner: ProcessRunner) -> None:
    del runner


_assert_runner_protocol(RecordingRunner(lines=[]))
