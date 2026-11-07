"""Unit tests for OpenCode NDJSON and sentinel parsing."""

from __future__ import annotations

import json
from pathlib import Path

from super_orchestrator.opencode import (
    NdjsonParseState,
    OpenCodeClient,
    ProcessRunner,
    ingest_ndjson_line,
    parse_implement_sentinels,
    parse_review_sentinel,
)


class RecordingRunner:
    """Fake ProcessRunner that records commands and writes scripted stdout."""

    def __init__(self, lines: list[str], exit_code: int = 0) -> None:
        self.lines = lines
        self.exit_code = exit_code
        self.commands: list[list[str]] = []

    def run(
        self,
        command: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        env: object,
        log_path: Path,
        on_stdout_line: object = None,
    ) -> int:
        self.commands.append(list(command))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            for line in self.lines:
                handle.write(line + "\n")
                if callable(on_stdout_line):
                    on_stdout_line(line)
        return self.exit_code


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


def test_parse_review_sentinel_approve() -> None:
    assert parse_review_sentinel("REVIEW_VERDICT: APPROVE\nok") == "APPROVE"


def test_parse_review_sentinel_request_changes() -> None:
    assert (
        parse_review_sentinel("notes\nREVIEW_VERDICT: REQUEST_CHANGES\n- fix tests")
        == "REQUEST_CHANGES"
    )


def test_ingest_ndjson_session_and_text() -> None:
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
    assert "hello" in "\n".join(state.assistant_chunks)
    assert "world" in "\n".join(state.assistant_chunks)


def test_run_implement_builds_command_and_parses(
    tmp_path: Path,
) -> None:
    lines = [
        json.dumps({"sessionID": "ses_1", "text": "working"}),
        "ISSUE_STATUS: PR_READY",
        "PR_URL: https://github.com/acme/app/pull/3",
    ]
    runner = RecordingRunner(lines=lines, exit_code=0)
    client = OpenCodeClient(workdir=tmp_path, runner=runner)
    result = client.run_implement(
        issue_number=3,
        attempt=1,
        message="Implement issue 3",
    )
    assert result.exit_code == 0
    assert result.session_id == "ses_1"
    assert result.status == "PR_READY"
    assert result.pr_url == "https://github.com/acme/app/pull/3"
    command = runner.commands[0]
    assert command[:7] == [
        "opencode",
        "run",
        "--agent",
        "issue-worker",
        "--auto",
        "--format",
        "json",
    ]
    assert "--archived" in command
    assert command[-1] == "Implement issue 3"


def test_run_implement_resumes_session(tmp_path: Path) -> None:
    runner = RecordingRunner(
        lines=[
            "ISSUE_STATUS: PR_READY",
            "PR_URL: https://github.com/acme/app/pull/3",
        ],
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
            "Looks good overall.",
            "REVIEW_VERDICT: REQUEST_CHANGES",
            "1. Add a null check in src/a.ts",
        ],
        exit_code=0,
    )
    client = OpenCodeClient(workdir=tmp_path, runner=runner)
    result = client.run_review(
        issue_number=3,
        attempt=1,
        message="Review PR",
    )
    assert result.verdict == "REQUEST_CHANGES"
    assert "null check" in result.feedback


# Satisfy type checkers that RecordingRunner matches ProcessRunner structurally.
def _assert_runner_protocol(runner: ProcessRunner) -> None:
    del runner


_assert_runner_protocol(RecordingRunner(lines=[]))
