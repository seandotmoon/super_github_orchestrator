"""Blocking OpenCode subprocess runner and sentinel parsers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol


class OpenCodeError(RuntimeError):
    """Raised when an OpenCode invocation fails or returns an incomplete result."""


@dataclass(frozen=True)
class ImplementResult:
    """Parsed outcome of an issue-worker OpenCode run."""

    exit_code: int
    session_id: str | None
    pr_url: str | None
    status: Literal["PR_READY"] | None
    assistant_text: str
    log_path: Path


@dataclass(frozen=True)
class ReviewResult:
    """Parsed outcome of a pr-reviewer OpenCode run."""

    exit_code: int
    session_id: str | None
    verdict: Literal["APPROVE", "REQUEST_CHANGES"] | None
    feedback: str
    assistant_text: str
    log_path: Path


@dataclass
class NdjsonParseState:
    """Mutable accumulator while reading OpenCode `--format json` NDJSON events."""

    session_id: str | None = None
    assistant_chunks: list[str] = field(default_factory=list)


class ProcessRunner(Protocol):
    """
    Abstraction over process execution so unit tests can inject a fake runner.

    Implementations must stream stdout and stderr into the provided log path and
    return the process exit code.
    """

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        log_path: Path,
        on_stdout_line: Callable[[str], None] | None = None,
    ) -> int:
        """Run a command, append output to log_path, and return the exit code."""


class OpenCodeGateway(Protocol):
    """Subset of OpenCode operations required by the LangGraph control loop."""

    def run_implement(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "issue-worker",
    ) -> ImplementResult:
        """Block on the issue-worker agent and return a parsed result."""

    def run_review(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "pr-reviewer",
    ) -> ReviewResult:
        """Block on the pr-reviewer agent and return a parsed result."""


class SubprocessRunner:
    """Production ProcessRunner that shells out with subprocess.Popen."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        log_path: Path,
        on_stdout_line: Callable[[str], None] | None = None,
    ) -> int:
        """
        Launch the command, stream stdout/stderr into log_path, and return exit code.

        Raises:
            OpenCodeError: If the binary cannot be started.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=dict(env) if env is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise OpenCodeError(f"Failed to start process {command[0]!r}: {exc}") from exc

        stdout = process.stdout
        if stdout is None:
            process.kill()
            raise OpenCodeError("OpenCode process started without a stdout pipe")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(command)}\n")
            log_file.flush()
            for line in stdout:
                log_file.write(line)
                log_file.flush()
                if on_stdout_line is not None:
                    on_stdout_line(line.rstrip("\n"))
            return_code = process.wait()
            log_file.write(f"\n[exit_code={return_code}]\n")
        return return_code


# Sentinel patterns expected in the final assistant text.
_ISSUE_STATUS_RE = re.compile(r"^ISSUE_STATUS:\s*(PR_READY)\s*$", re.MULTILINE)
_PR_URL_RE = re.compile(r"^PR_URL:\s*(https://\S+)\s*$", re.MULTILINE)
_REVIEW_VERDICT_RE = re.compile(
    r"^REVIEW_VERDICT:\s*(APPROVE|REQUEST_CHANGES)\s*$",
    re.MULTILINE,
)


def parse_implement_sentinels(text: str) -> tuple[Literal["PR_READY"] | None, str | None]:
    """
    Extract ISSUE_STATUS and PR_URL sentinels from assistant text.

    Returns:
        A pair of (status, pr_url). Either element may be None when missing.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    status_match = _ISSUE_STATUS_RE.search(text)
    url_match = _PR_URL_RE.search(text)
    status: Literal["PR_READY"] | None = None
    if status_match is not None:
        status = "PR_READY"
    pr_url = url_match.group(1) if url_match is not None else None
    return status, pr_url


def parse_review_sentinel(text: str) -> Literal["APPROVE", "REQUEST_CHANGES"] | None:
    """
    Extract REVIEW_VERDICT from assistant text.

    Returns:
        The verdict literal, or None when the sentinel is absent.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    match = _REVIEW_VERDICT_RE.search(text)
    if match is None:
        return None
    verdict = match.group(1)
    if verdict == "APPROVE":
        return "APPROVE"
    if verdict == "REQUEST_CHANGES":
        return "REQUEST_CHANGES"
    return None


def ingest_ndjson_line(state: NdjsonParseState, line: str) -> None:
    """
    Update parse state from one OpenCode NDJSON event line.

    Tolerates non-JSON lines (plain text) by appending them as assistant text.
    Session id is taken from the first event that exposes a recognizable field.
    """
    if not isinstance(line, str):
        raise TypeError("line must be a str")
    stripped = line.strip()
    if stripped == "":
        return

    try:
        payload: object = json.loads(stripped)
    except json.JSONDecodeError:
        state.assistant_chunks.append(stripped)
        return

    if not isinstance(payload, dict):
        return

    session_candidate = _extract_session_id(payload)
    if session_candidate is not None and state.session_id is None:
        state.session_id = session_candidate

    text_chunk = _extract_assistant_text(payload)
    if text_chunk is not None and text_chunk.strip() != "":
        state.assistant_chunks.append(text_chunk)


def _extract_session_id(payload: Mapping[str, object]) -> str | None:
    """Pull a session id from common OpenCode event shapes."""
    for key in ("sessionID", "sessionId", "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() != "":
            return value.strip()

    nested = payload.get("session")
    if isinstance(nested, Mapping):
        nested_id = nested.get("id")
        if isinstance(nested_id, str) and nested_id.strip() != "":
            return nested_id.strip()

    properties = payload.get("properties")
    if isinstance(properties, Mapping):
        for key in ("sessionID", "sessionId", "session_id", "id"):
            value = properties.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
        info = properties.get("info")
        if isinstance(info, Mapping):
            info_id = info.get("id")
            if isinstance(info_id, str) and info_id.strip() != "":
                return info_id.strip()
    return None


def _extract_assistant_text(payload: Mapping[str, object]) -> str | None:
    """
    Pull assistant-visible text from common OpenCode event shapes.

    Prefers explicit text / part / message content over dumping the whole event.
    """
    event_type = payload.get("type")
    if isinstance(event_type, str):
        # Skip noisy non-content events when possible.
        if event_type in {"tool_call", "tool_result", "step_start", "step_finish"}:
            return None

    for key in ("text", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() != "":
            return value
        if isinstance(value, Mapping):
            nested_text = value.get("content")
            if isinstance(nested_text, str) and nested_text.strip() != "":
                return nested_text
            nested_parts = value.get("parts")
            joined = _join_parts(nested_parts)
            if joined is not None:
                return joined

    properties = payload.get("properties")
    if isinstance(properties, Mapping):
        part = properties.get("part")
        if isinstance(part, Mapping):
            part_type = part.get("type")
            part_text = part.get("text")
            if part_type == "text" and isinstance(part_text, str):
                return part_text
        props_text = properties.get("text")
        if isinstance(props_text, str) and props_text.strip() != "":
            return props_text

    parts = payload.get("parts")
    return _join_parts(parts)


def _join_parts(parts: object) -> str | None:
    """Concatenate text parts from an OpenCode message parts array."""
    if not isinstance(parts, list):
        return None
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, Mapping):
            part_type = part.get("type")
            text = part.get("text")
            if part_type == "text" and isinstance(text, str) and text.strip() != "":
                chunks.append(text)
            elif isinstance(text, str) and text.strip() != "":
                chunks.append(text)
        elif isinstance(part, str) and part.strip() != "":
            chunks.append(part)
    if len(chunks) == 0:
        return None
    return "\n".join(chunks)


class OpenCodeClient:
    """
    Invokes `opencode run --agent ...` as a blocking subprocess.

    Completion is defined as exit code 0 plus the expected sentinel in the
    accumulated assistant text. Session ids are captured from NDJSON events so
    a later changes round can resume with `--session`.
    """

    def __init__(
        self,
        *,
        workdir: Path,
        runner: ProcessRunner | None = None,
        opencode_bin: str = "opencode",
    ) -> None:
        if not isinstance(workdir, Path):
            raise TypeError("workdir must be a Path")
        if not isinstance(opencode_bin, str) or opencode_bin.strip() == "":
            raise ValueError("opencode_bin must be a non-empty str")
        self._workdir = workdir
        self._runner: ProcessRunner = runner if runner is not None else SubprocessRunner()
        self._opencode_bin = opencode_bin

    @property
    def workdir(self) -> Path:
        """Return the repository working directory used for OpenCode runs."""
        return self._workdir

    def logs_dir(self) -> Path:
        """Return (and create) the orchestrator log directory under workdir."""
        path = self._workdir / ".orchestrator" / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_agents(self) -> str:
        """
        Run `opencode debug agents` and return combined output.

        Raises:
            OpenCodeError: When the command fails or the binary is missing.
        """
        if shutil.which(self._opencode_bin) is None:
            raise OpenCodeError(
                f"OpenCode binary {self._opencode_bin!r} was not found on PATH"
            )
        log_path = self.logs_dir() / "debug-agents.log"
        captured: list[str] = []

        def capture(line: str) -> None:
            captured.append(line)

        exit_code = self._runner.run(
            [self._opencode_bin, "debug", "agents"],
            cwd=self._workdir,
            env=None,
            log_path=log_path,
            on_stdout_line=capture,
        )
        output = "\n".join(captured)
        if exit_code != 0:
            raise OpenCodeError(
                f"`opencode debug agents` failed with exit {exit_code}: {output[-2000:]}"
            )
        return output

    def run_implement(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "issue-worker",
    ) -> ImplementResult:
        """
        Block on the issue-worker agent until the process exits, then parse sentinels.

        Args:
            issue_number: GitHub issue number (for log naming).
            attempt: 1-based review/implement attempt counter.
            message: User message passed to OpenCode.
            session_id: Optional existing session to continue.
            agent: Primary agent name (default issue-worker).
        """
        self._validate_message(message)
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("issue_number must be a positive int")
        if not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive int")

        log_path = self.logs_dir() / f"{issue_number}-implement-{attempt}.log"
        parse_state = NdjsonParseState(session_id=session_id)
        command = self._build_command(agent=agent, message=message, session_id=session_id)
        exit_code = self._runner.run(
            command,
            cwd=self._workdir,
            env=None,
            log_path=log_path,
            on_stdout_line=lambda line: ingest_ndjson_line(parse_state, line),
        )
        assistant_text = "\n".join(parse_state.assistant_chunks)
        status, pr_url = parse_implement_sentinels(assistant_text)
        return ImplementResult(
            exit_code=exit_code,
            session_id=parse_state.session_id,
            pr_url=pr_url,
            status=status,
            assistant_text=assistant_text,
            log_path=log_path,
        )

    def run_review(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "pr-reviewer",
    ) -> ReviewResult:
        """
        Block on the pr-reviewer agent until the process exits, then parse the verdict.

        Args:
            issue_number: GitHub issue number (for log naming).
            attempt: 1-based review attempt counter.
            message: User message passed to OpenCode.
            session_id: Optional existing review session to continue.
            agent: Primary agent name (default pr-reviewer).
        """
        self._validate_message(message)
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("issue_number must be a positive int")
        if not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive int")

        log_path = self.logs_dir() / f"{issue_number}-review-{attempt}.log"
        parse_state = NdjsonParseState(session_id=session_id)
        command = self._build_command(agent=agent, message=message, session_id=session_id)
        exit_code = self._runner.run(
            command,
            cwd=self._workdir,
            env=None,
            log_path=log_path,
            on_stdout_line=lambda line: ingest_ndjson_line(parse_state, line),
        )
        assistant_text = "\n".join(parse_state.assistant_chunks)
        verdict = parse_review_sentinel(assistant_text)
        feedback = self._extract_review_feedback(assistant_text)
        return ReviewResult(
            exit_code=exit_code,
            session_id=parse_state.session_id,
            verdict=verdict,
            feedback=feedback,
            assistant_text=assistant_text,
            log_path=log_path,
        )

    def _build_command(
        self,
        *,
        agent: str,
        message: str,
        session_id: str | None,
    ) -> list[str]:
        """Assemble the `opencode run` argv list."""
        if not isinstance(agent, str) or agent.strip() == "":
            raise ValueError("agent must be a non-empty str")
        command: list[str] = [
            self._opencode_bin,
            "run",
            "--agent",
            agent,
            "--auto",
            "--format",
            "json",
            "--archived",
        ]
        if session_id is not None:
            if not isinstance(session_id, str) or session_id.strip() == "":
                raise ValueError("session_id must be a non-empty str when provided")
            command.extend(["--session", session_id])
        command.append(message)
        return command

    @staticmethod
    def _validate_message(message: str) -> None:
        """Reject empty or non-string user messages."""
        if not isinstance(message, str):
            raise TypeError("message must be a str")
        if message.strip() == "":
            raise ValueError("message must be non-empty")

    @staticmethod
    def _extract_review_feedback(assistant_text: str) -> str:
        """
        Prefer text after the verdict sentinel as feedback for the next implement round.

        Falls back to the full assistant text when no sentinel is present.
        """
        match = _REVIEW_VERDICT_RE.search(assistant_text)
        if match is None:
            return assistant_text.strip()
        after = assistant_text[match.end() :].strip()
        if after != "":
            return after
        before = assistant_text[: match.start()].strip()
        return before if before != "" else assistant_text.strip()
