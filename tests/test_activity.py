"""Unit tests for live console activity: event summaries, tailing, heartbeats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from super_orchestrator.activity import describe_event, format_duration
from super_orchestrator.opencode import LiveOutput, SubprocessRunner


def _tool(tool: str, tool_input: dict[str, object], status: str = "completed") -> str:
    state: dict[str, object] = {"status": status, "input": tool_input}
    if status == "error":
        state["error"] = "permission denied by rule"
    return json.dumps(
        {"type": "tool_use", "sessionID": "ses", "part": {"type": "tool", "tool": tool, "state": state}}
    )


def test_describe_bash_command() -> None:
    assert describe_event(_tool("bash", {"command": "npm test -- --runInBand"})) == (
        "bash npm test -- --runInBand"
    )


def test_describe_subagent_launch() -> None:
    line = _tool("task", {"subagent_type": "impl-backend", "description": "Add role endpoint"})
    assert describe_event(line) == "subagent impl-backend: Add role endpoint"


def test_describe_failed_tool() -> None:
    summary = describe_event(_tool("bash", {"command": "git push"}, status="error"))
    assert summary is not None
    assert summary.startswith("✗ bash git push")
    assert "permission denied" in summary


def test_describe_file_edit_and_todos() -> None:
    assert describe_event(_tool("edit", {"filePath": "src/api/users.ts"})) == "edit src/api/users.ts"
    todos = [{"status": "completed"}, {"status": "pending"}, {"status": "completed"}]
    assert describe_event(_tool("todowrite", {"todos": todos})) == "todo list updated (2/3 done)"


def test_describe_text_shows_first_line_only() -> None:
    line = json.dumps(
        {"type": "text", "part": {"type": "text", "text": "\nAll tests pass.\nISSUE_STATUS: PR_READY"}}
    )
    assert describe_event(line) == "says: All tests pass."


def test_describe_ignores_noise() -> None:
    assert describe_event(json.dumps({"type": "step_start", "part": {}})) is None
    assert describe_event(json.dumps({"type": "reasoning", "part": {"text": "hmm"}})) is None
    assert describe_event("plain stderr line") is None
    assert describe_event("") is None


def test_describe_truncates_long_details() -> None:
    summary = describe_event(_tool("bash", {"command": "x" * 500}))
    assert summary is not None
    assert len(summary) < 160
    assert summary.endswith("…")


def test_format_duration() -> None:
    assert format_duration(5) == "5s"
    assert format_duration(125) == "2m05s"
    assert format_duration(3 * 3600 + 120) == "3h02m"


def test_live_output_streams_lines_before_exit_and_heartbeats(tmp_path: Path) -> None:
    """Lines arrive while the child is still running, and silence triggers heartbeats."""
    script = "\n".join(
        [
            "import sys, time",
            "print('first', flush=True)",
            "time.sleep(1.5)",
            "print('second', flush=True)",
        ]
    )
    events: list[str] = []
    replayed: list[str] = []

    def on_line(line: str) -> None:
        events.append(f"line:{line}")

    def on_idle(elapsed: float) -> None:
        events.append("idle")

    code = SubprocessRunner().run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=None,
        log_path=tmp_path / "live.log",
        on_stdout_line=replayed.append,
        timeout_seconds=30,
        live=LiveOutput(
            on_line=on_line,
            on_idle=on_idle,
            idle_interval_seconds=0.5,
            poll_interval_seconds=0.1,
        ),
    )
    assert code == 0
    assert replayed == ["first", "second"]
    assert events[0] == "line:first"
    assert events[-1] == "line:second"
    assert "idle" in events[1:-1]


def test_live_callback_errors_do_not_break_run(tmp_path: Path) -> None:
    def explode(line: str) -> None:
        raise RuntimeError(f"display failed on {line}")

    replayed: list[str] = []
    code = SubprocessRunner().run(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env=None,
        log_path=tmp_path / "boom.log",
        on_stdout_line=replayed.append,
        timeout_seconds=30,
        live=LiveOutput(on_line=explode, poll_interval_seconds=0.05),
    )
    assert code == 0
    assert replayed == ["ok"]
