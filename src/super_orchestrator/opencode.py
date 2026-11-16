"""Blocking OpenCode subprocess runner and sentinel parsers."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from super_orchestrator.activity import describe_event, format_duration
from super_orchestrator.state import ReviewVerdict


class OpenCodeError(RuntimeError):
    """Raised when an OpenCode invocation fails or returns an incomplete result."""


class OpenCodeTimeout(OpenCodeError):
    """Raised when an OpenCode process exceeds its time budget and is killed."""


# Wall-clock budgets for one agent invocation. An implement run that hits its
# budget is a "slice": the orchestrator stops it and continues in a new slice,
# so a stuck subagent costs at most one slice instead of the whole issue.
DEFAULT_IMPLEMENT_TIMEOUT_SECONDS = 3_600
DEFAULT_REVIEW_TIMEOUT_SECONDS = 7_200
# Exit code reported for a run the orchestrator killed at its time budget (SIGKILL).
TIMEOUT_EXIT_CODE = -9
# Ask OpenCode to compact once a session's reported context reaches this size.
DEFAULT_COMPACT_TOKENS = 150_000
# `opencode api …/compact` only admits the request; the summary runs in-session.
COMPACT_REQUEST_TIMEOUT_SECONDS = 30.0
# A failed admit is retried on a later step, but not on every step.
COMPACT_RETRY_SECONDS = 60.0

# OpenCode session ids are `ses_…`. Refuse anything that could change the URL path.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Events that carry a fresh context-token count for one session.
_USAGE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "step_finish",
        "step-finish",
        "session.step.ended",
        "session.usage.updated",
    }
)


@dataclass(frozen=True)
class ImplementResult:
    """Parsed outcome of an issue-worker OpenCode run."""

    exit_code: int
    session_id: str | None
    pr_url: str | None
    status: Literal["PR_READY"] | None
    assistant_text: str
    log_path: Path
    errors: tuple[str, ...] = ()
    # True when the run was killed at its time budget (the slice ended unfinished).
    timed_out: bool = False


@dataclass(frozen=True)
class ReviewResult:
    """Parsed outcome of a pr-reviewer OpenCode run."""

    exit_code: int
    session_id: str | None
    verdict: ReviewVerdict | None
    feedback: str
    assistant_text: str
    log_path: Path
    errors: tuple[str, ...] = ()


@dataclass
class NdjsonParseState:
    """
    Mutable accumulator while reading OpenCode `--format json` NDJSON events.

    OpenCode streams assistant text as `text` events whose `part.text` is a
    delta of the part identified by `part.id`. Deltas for the same part are
    concatenated verbatim; distinct parts (and plain-text lines) become
    separate segments joined by newlines.
    """

    session_id: str | None = None
    segment_order: list[str] = field(default_factory=list)
    segments: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    _anonymous_count: int = 0

    def append(self, segment_id: str | None, text: str) -> None:
        """Append text to a keyed part, or as a new anonymous segment when unkeyed."""
        if segment_id is None:
            self._anonymous_count += 1
            segment_id = f"__anon_{self._anonymous_count}"
        if segment_id not in self.segments:
            self.segment_order.append(segment_id)
            self.segments[segment_id] = text
            return
        self.segments[segment_id] = f"{self.segments[segment_id]}{text}"

    @property
    def assistant_chunks(self) -> list[str]:
        """Return non-empty segments in arrival order."""
        chunks = [self.segments[key] for key in self.segment_order]
        return [chunk for chunk in chunks if chunk.strip() != ""]

    @property
    def assistant_text(self) -> str:
        """Return all assistant-visible text as one newline-joined string."""
        return "\n".join(self.assistant_chunks)


@dataclass(frozen=True)
class LiveOutput:
    """
    Callbacks for watching a process while it runs (console progress only).

    on_line receives each output line shortly after it is written; on_idle
    receives total elapsed seconds after every idle_interval_seconds of silence.
    Exceptions raised by these callbacks are swallowed so display problems
    never break a run.
    """

    on_line: Callable[[str], None]
    on_idle: Callable[[float], None] | None = None
    idle_interval_seconds: float = 120.0
    poll_interval_seconds: float = 0.5


class ProcessRunner(Protocol):
    """
    Abstraction over process execution so unit tests can inject a fake runner.

    Implementations must stream stdout and stderr into the provided log path and
    return the process exit code, raising OpenCodeTimeout when timeout_seconds
    elapses before the process exits.
    """

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
        slice_number: int = 1,
    ) -> ImplementResult:
        """
        Block on the issue-worker agent and return a parsed result.

        A run killed at the implement time budget returns `timed_out=True`
        instead of raising, so the caller can continue in another slice.
        """

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


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill the child and everything in its process group (OpenCode spawns helpers)."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        # Already exited, or the group is gone; nothing left to kill.
        pass


def _safe_call(callback: Callable[[str], None], value: str) -> None:
    """Invoke a display callback, ignoring any error it raises."""
    try:
        callback(value)
    except Exception:  # noqa: BLE001 — display must never break a run
        pass


class _LogTail:
    """Incrementally read complete lines appended to a log file after an offset."""

    def __init__(self, log_path: Path, offset: int, on_line: Callable[[str], None]) -> None:
        self._reader = log_path.open("rb")
        self._reader.seek(offset)
        self._buffer = b""
        self._on_line = on_line

    def pump(self, *, final: bool = False) -> bool:
        """Emit newly completed lines; with final=True also emit a trailing partial line."""
        chunk = self._reader.read()
        emitted = False
        if chunk:
            self._buffer = b"".join([self._buffer, chunk])
        while b"\n" in self._buffer:
            raw, self._buffer = self._buffer.split(b"\n", 1)
            _safe_call(self._on_line, raw.decode("utf-8", errors="replace"))
            emitted = True
        if final and self._buffer != b"":
            _safe_call(self._on_line, self._buffer.decode("utf-8", errors="replace"))
            self._buffer = b""
            emitted = True
        return emitted

    def close(self) -> None:
        """Close the underlying reader."""
        self._reader.close()


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
        timeout_seconds: float | None = None,
        live: LiveOutput | None = None,
    ) -> int:
        """
        Launch the command with stdout/stderr going straight into log_path.

        OpenCode (Bun) can exit before flushing a pipe, silently truncating large
        or final output; writing to a regular file does not lose data. The new
        portion of the log is replayed through on_stdout_line after the process
        exits; `live` (optional) tails the file while the process runs. The child
        runs in its own session so a timeout (or Ctrl-C in the orchestrator)
        kills the whole process group rather than orphaning it.

        Raises:
            OpenCodeError: If the binary cannot be started.
            OpenCodeTimeout: If the process outlives timeout_seconds.
        """
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        timed_out = False
        with log_path.open("ab") as log_file:
            log_file.write(f"$ {' '.join(command)}\n".encode("utf-8"))
            log_file.flush()
            output_start = log_file.tell()
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=str(cwd),
                    env=dict(env) if env is not None else None,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=os.name == "posix",
                )
            except OSError as exc:
                raise OpenCodeError(f"Failed to start process {command[0]!r}: {exc}") from exc

            tail = _LogTail(log_path, output_start, live.on_line) if live is not None else None
            started = time.monotonic()
            deadline = None if timeout_seconds is None else started + timeout_seconds
            poll = live.poll_interval_seconds if live is not None else 1.0
            last_signal = started
            try:
                while True:
                    try:
                        return_code = process.wait(timeout=poll)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    now = time.monotonic()
                    if tail is not None and live is not None:
                        if tail.pump():
                            last_signal = now
                        elif (
                            live.on_idle is not None
                            and now - last_signal >= live.idle_interval_seconds
                        ):
                            elapsed = now - started
                            try:
                                live.on_idle(elapsed)
                            except Exception:  # noqa: BLE001 — display only
                                pass
                            last_signal = now
                    if deadline is not None and now >= deadline:
                        timed_out = True
                        _kill_process_tree(process)
                        return_code = process.wait()
                        break
            except BaseException:
                # Covers KeyboardInterrupt: never leave the agent running.
                _kill_process_tree(process)
                process.wait()
                raise
            finally:
                if tail is not None:
                    tail.pump(final=True)
                    tail.close()
            output_end = log_file.seek(0, os.SEEK_END)

        if on_stdout_line is not None:
            with log_path.open("rb") as reader:
                reader.seek(output_start)
                output = reader.read(output_end - output_start).decode("utf-8", errors="replace")
            for line in output.splitlines():
                on_stdout_line(line)

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[exit_code={return_code}]\n")
            if timed_out:
                log_file.write(f"[timed out after {timeout_seconds}s]\n")

        if timed_out:
            raise OpenCodeTimeout(
                f"{command[0]} exceeded {timeout_seconds}s and was killed (log: {log_path})"
            )
        return return_code


# Sentinel patterns expected in the final assistant text. They tolerate common
# model formatting (leading whitespace, backticks, bold, angle-bracketed URLs).
_DECOR = r"[`*_]*"
_ISSUE_STATUS_RE = re.compile(
    rf"^\s*{_DECOR}ISSUE_STATUS:{_DECOR}\s*{_DECOR}(PR_READY){_DECOR}\s*$",
    re.MULTILINE,
)
_PR_URL_RE = re.compile(
    rf"^\s*{_DECOR}PR_URL:{_DECOR}\s*[<`]?(https://[^\s<>`]+?)[>`]?{_DECOR}\s*$",
    re.MULTILINE,
)
_REVIEW_VERDICT_RE = re.compile(
    rf"^\s*{_DECOR}REVIEW_VERDICT:{_DECOR}\s*{_DECOR}(APPROVE|REQUEST_CHANGES){_DECOR}\s*$",
    re.MULTILINE,
)


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Return the final match so the closing sentinel wins over earlier mentions."""
    last: re.Match[str] | None = None
    for match in pattern.finditer(text):
        last = match
    return last


def parse_implement_sentinels(text: str) -> tuple[Literal["PR_READY"] | None, str | None]:
    """
    Extract ISSUE_STATUS and PR_URL sentinels from assistant text.

    Returns:
        A pair of (status, pr_url). Either element may be None when missing.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    status_match = _last_match(_ISSUE_STATUS_RE, text)
    url_match = _last_match(_PR_URL_RE, text)
    status: Literal["PR_READY"] | None = None
    if status_match is not None:
        status = "PR_READY"
    pr_url = url_match.group(1) if url_match is not None else None
    return status, pr_url


def parse_review_sentinel(text: str) -> ReviewVerdict | None:
    """
    Extract the final REVIEW_VERDICT from assistant text.

    Returns:
        The verdict literal, or None when the sentinel is absent.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    match = _last_match(_REVIEW_VERDICT_RE, text)
    if match is None:
        return None
    verdict = match.group(1)
    if verdict == "APPROVE":
        return "APPROVE"
    if verdict == "REQUEST_CHANGES":
        return "REQUEST_CHANGES"
    return None


# OpenCode `run --format json` event types that never carry assistant answer text.
_NON_TEXT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "reasoning",
        "tool_use",
        "step_start",
        "step_finish",
        "background_wait",
        "error",
        # Older / alternative names seen in other OpenCode builds.
        "tool_call",
        "tool_result",
    }
)


def ingest_ndjson_line(state: NdjsonParseState, line: str) -> None:
    """
    Update parse state from one OpenCode NDJSON event line.

    Handles the current OpenCode shape
    `{"type": "text", "sessionID": ..., "part": {"id": ..., "type": "text", "text": ...}}`
    plus older event shapes. Non-JSON lines (plain text) are kept as their own
    segments. Session id is taken from the first event that exposes one.
    """
    if not isinstance(line, str):
        raise TypeError("line must be a str")
    stripped = line.strip()
    if stripped == "":
        return

    try:
        payload: object = json.loads(stripped)
    except json.JSONDecodeError:
        state.append(None, stripped)
        return

    if not isinstance(payload, dict):
        return

    session_candidate = _extract_session_id(payload)
    if session_candidate is not None and state.session_id is None:
        state.session_id = session_candidate

    event_type = payload.get("type")
    if event_type == "error":
        state.errors.append(_extract_error_message(payload))
        return
    if isinstance(event_type, str) and event_type in _NON_TEXT_EVENT_TYPES:
        return

    part = payload.get("part")
    if isinstance(part, Mapping):
        part_type = part.get("type")
        part_text = part.get("text")
        if part_type == "text" and isinstance(part_text, str):
            part_id = part.get("id")
            state.append(part_id if isinstance(part_id, str) else None, part_text)
        # Any other part (reasoning, tool, step) is not answer text.
        return

    text_chunk = _extract_legacy_assistant_text(payload)
    if text_chunk is not None and text_chunk.strip() != "":
        state.append(None, text_chunk)


def _extract_error_message(payload: Mapping[str, object]) -> str:
    """Render an OpenCode `error` event into a short human-readable string."""
    error = payload.get("error")
    if isinstance(error, str) and error.strip() != "":
        return error.strip()
    if isinstance(error, Mapping):
        for key in ("message", "name"):
            value = error.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
        data = error.get("data")
        if isinstance(data, Mapping):
            message = data.get("message")
            if isinstance(message, str) and message.strip() != "":
                return message.strip()
    return "unknown OpenCode error event"


def parse_context_usages(line: str) -> list[tuple[str, int]]:
    """
    Read context size from one OpenCode JSON event.

    Recognizes `step_finish` (and the same token payload under a few newer
    event names). Returns `(session_id, tokens)` for that session, or an empty
    list when the line is not a usage report. `tokens.total` wins when OpenCode
    sends it; otherwise the count is input + output + reasoning + cache read +
    cache write, which is the sum OpenCode uses for its own overflow check.
    """
    if not isinstance(line, str):
        raise TypeError("line must be a str")
    stripped = line.strip()
    if stripped == "" or not stripped.startswith("{"):
        return []
    try:
        payload: object = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict) or not _is_usage_event(payload):
        return []
    session_id = _extract_session_id(payload)
    tokens = _extract_token_count(payload)
    if session_id is None or tokens is None:
        return []
    if _SESSION_ID_RE.fullmatch(session_id) is None:
        return []
    return [(session_id, tokens)]


def _is_usage_event(payload: Mapping[str, object]) -> bool:
    """True when this event is a step-finish or a session usage update."""
    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type in _USAGE_EVENT_TYPES:
        return True
    part = payload.get("part")
    if isinstance(part, Mapping):
        part_type = part.get("type")
        if part_type in {"step-finish", "step_finish"}:
            return True
    return False


def _extract_token_count(payload: Mapping[str, object]) -> int | None:
    """Pull a token object from the shapes OpenCode has used for step usage."""
    part = payload.get("part")
    if isinstance(part, Mapping):
        count = _token_count(part.get("tokens"))
        if count is not None:
            return count
    count = _token_count(payload.get("tokens"))
    if count is not None:
        return count
    properties = payload.get("properties")
    if not isinstance(properties, Mapping):
        return None
    count = _token_count(properties.get("tokens"))
    if count is not None:
        return count
    info = properties.get("info")
    if isinstance(info, Mapping):
        count = _token_count(info.get("tokens"))
        if count is not None:
            return count
    nested = properties.get("part")
    if isinstance(nested, Mapping):
        return _token_count(nested.get("tokens"))
    return None


def _token_count(value: object) -> int | None:
    """
    Reduce an OpenCode token object to one context size.

    Prefer `total` when it is present and positive. Otherwise require `input`
    and add the other non-negative fields. Returns None when the object is not
    a usable usage report.
    """
    if not isinstance(value, Mapping):
        return None
    total = _non_negative_number(value.get("total"))
    if total is not None and total > 0:
        return total
    input_tokens = _non_negative_number(value.get("input"))
    if input_tokens is None:
        return None
    output_tokens = _non_negative_number(value.get("output"))
    reasoning_tokens = _non_negative_number(value.get("reasoning"))
    cache_read = 0
    cache_write = 0
    cache = value.get("cache")
    if isinstance(cache, Mapping):
        read = _non_negative_number(cache.get("read"))
        write = _non_negative_number(cache.get("write"))
        cache_read = 0 if read is None else read
        cache_write = 0 if write is None else write
    elif cache is not None:
        return None
    output = 0 if output_tokens is None else output_tokens
    reasoning = 0 if reasoning_tokens is None else reasoning_tokens
    return input_tokens + output + reasoning + cache_read + cache_write


def _non_negative_number(value: object) -> int | None:
    """Return an int for a non-negative number, or None when the value is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(value)


class CompactionWatch:
    """
    Request compaction once per session when context crosses a token threshold.

    The latch clears when a later report falls back under the threshold, so a
    session that grows again is compacted again. A rejected request is retried
    after `retry_seconds` rather than on every following step.
    """

    def __init__(
        self,
        *,
        threshold: int,
        request: Callable[[str], bool],
        announce: Callable[[str], None],
        now: Callable[[], float] = time.monotonic,
        retry_seconds: float = COMPACT_RETRY_SECONDS,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ValueError("threshold must be a positive int")
        if not isinstance(retry_seconds, (int, float)) or isinstance(retry_seconds, bool):
            raise TypeError("retry_seconds must be a number")
        if retry_seconds < 0:
            raise ValueError("retry_seconds must be >= 0")
        self._threshold = threshold
        self._request = request
        self._announce = announce
        self._now = now
        self._retry_seconds = float(retry_seconds)
        self._latched: set[str] = set()
        self._attempted_at: dict[str, float] = {}

    def observe(self, session_id: str, tokens: int) -> None:
        """Compact `session_id` when `tokens` is at or above the threshold."""
        if not isinstance(session_id, str) or session_id.strip() == "":
            raise ValueError("session_id must be a non-empty str")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a non-negative int")
        if tokens < self._threshold:
            self._latched.discard(session_id)
            self._attempted_at.pop(session_id, None)
            return
        if session_id in self._latched:
            return
        attempted = self._attempted_at.get(session_id)
        if attempted is not None and self._now() - attempted < self._retry_seconds:
            return
        self._attempted_at[session_id] = self._now()
        try:
            accepted = self._request(session_id)
        except Exception as exc:  # noqa: BLE001 — compaction must not kill the agent
            self._announce(f"compact failed for {session_id}: {exc}")
            return
        if not accepted:
            return
        self._latched.add(session_id)
        self._announce(
            f"compacting {session_id} at {tokens} tokens (threshold {self._threshold})"
        )


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

    part = payload.get("part")
    if isinstance(part, Mapping):
        part_session = part.get("sessionID")
        if isinstance(part_session, str) and part_session.strip() != "":
            return part_session.strip()

    properties = payload.get("properties")
    if isinstance(properties, Mapping):
        for key in ("sessionID", "sessionId", "session_id"):
            value = properties.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
    return None


def _extract_legacy_assistant_text(payload: Mapping[str, object]) -> str | None:
    """Pull assistant-visible text from older OpenCode event shapes."""
    for key in ("text", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() != "":
            return value
        if isinstance(value, Mapping):
            nested_text = value.get("content")
            if isinstance(nested_text, str) and nested_text.strip() != "":
                return nested_text
            joined = _join_parts(value.get("parts"))
            if joined is not None:
                return joined

    properties = payload.get("properties")
    if isinstance(properties, Mapping):
        part = properties.get("part")
        if isinstance(part, Mapping):
            part_text = part.get("text")
            if part.get("type") == "text" and isinstance(part_text, str):
                return part_text
        props_text = properties.get("text")
        if isinstance(props_text, str) and props_text.strip() != "":
            return props_text

    return _join_parts(payload.get("parts"))


def _join_parts(parts: object) -> str | None:
    """Concatenate text parts from an OpenCode message parts array."""
    if not isinstance(parts, list):
        return None
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, Mapping):
            part_type = part.get("type")
            text = part.get("text")
            if part_type in {None, "text"} and isinstance(text, str) and text.strip() != "":
                chunks.append(text)
        elif isinstance(part, str) and part.strip() != "":
            chunks.append(part)
    if len(chunks) == 0:
        return None
    return "\n".join(chunks)


def _looks_like_cli_help(output: str) -> bool:
    """
    True when output is OpenCode's top-level usage text.

    OpenCode answers an unknown subcommand with its command list and exit 1,
    so a `Commands:` section next to `opencode run` marks the subcommand as
    missing in this build.
    """
    if not isinstance(output, str):
        raise TypeError("output must be a str")
    return "Commands:" in output and "opencode run" in output


# `opencode agent list` header line, e.g. "issue-worker (primary)".
_AGENT_LIST_LINE_RE = re.compile(r"^([A-Za-z0-9_.:-]+) \((?:primary|subagent|all)\)\s*$", re.MULTILINE)


def parse_agent_ids(output: str) -> set[str]:
    """
    Extract agent identifiers from OpenCode agent-listing output.

    Supports both `opencode agent list` (released builds: one
    `name (mode)` header per agent followed by its permission JSON) and
    `opencode debug agents` (newer builds: a JSON array of agent objects).
    """
    if not isinstance(output, str):
        raise TypeError("output must be a str")
    header_ids = set(_AGENT_LIST_LINE_RE.findall(output))
    if len(header_ids) > 0:
        return header_ids
    start = output.find("[")
    if start == -1:
        return set()
    try:
        payload: object = json.loads(output[start:])
    except json.JSONDecodeError:
        return set()
    ids: set[str] = set()
    if not isinstance(payload, list):
        return ids
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        for key in ("id", "name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip() != "":
                ids.add(value.strip())
    return ids


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
        implement_timeout_seconds: float | None = None,
        review_timeout_seconds: float | None = None,
        utility_timeout_seconds: float = 120.0,
        activity: Callable[[str], None] | None = None,
        status: Callable[[str], None] | None = None,
        heartbeat_seconds: float = 120.0,
        compact_tokens: int | None = DEFAULT_COMPACT_TOKENS,
    ) -> None:
        """
        Args:
            workdir: Target checkout where OpenCode runs.
            runner: Injectable process runner (tests).
            opencode_bin: OpenCode executable name or path.
            implement_timeout_seconds: Kill issue-worker after this long (None = never).
            review_timeout_seconds: Kill pr-reviewer after this long (None = never).
            utility_timeout_seconds: Limit for short commands such as agent listing.
            activity: Receives one-line summaries of agent activity while a run
                is in progress (tool calls, subagents, messages, heartbeats).
            status: Receives operational lines such as compaction. Falls back to
                `activity` when omitted. Printed even when activity is quiet.
            heartbeat_seconds: Emit a "still running" line after this much silence.
            compact_tokens: Compact a session at this context size. None disables.
        """
        if not isinstance(workdir, Path):
            raise TypeError("workdir must be a Path")
        if not isinstance(opencode_bin, str) or opencode_bin.strip() == "":
            raise ValueError("opencode_bin must be a non-empty str")
        for label, value in (
            ("implement_timeout_seconds", implement_timeout_seconds),
            ("review_timeout_seconds", review_timeout_seconds),
            ("utility_timeout_seconds", utility_timeout_seconds),
        ):
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"{label} must be a positive number or None")
        if not isinstance(heartbeat_seconds, (int, float)) or isinstance(heartbeat_seconds, bool):
            raise TypeError("heartbeat_seconds must be a number")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be a positive number")
        if compact_tokens is not None and (
            isinstance(compact_tokens, bool)
            or not isinstance(compact_tokens, int)
            or compact_tokens < 1
        ):
            raise ValueError("compact_tokens must be a positive int or None")
        self._workdir = workdir
        self._runner: ProcessRunner = runner if runner is not None else SubprocessRunner()
        self._opencode_bin = opencode_bin
        self._implement_timeout = implement_timeout_seconds
        self._review_timeout = review_timeout_seconds
        self._utility_timeout = utility_timeout_seconds
        self._activity = activity
        self._status = status if status is not None else activity
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._compact_tokens = compact_tokens
        # Set once OpenCode shows it has no `api` subcommand; compaction then stays off.
        self._compact_unsupported = False

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
        List agents and return the raw output.

        Tries `opencode agent list` (released builds) first, then
        `opencode debug agents` (newer builds).

        Raises:
            OpenCodeError: When every listing command fails or the binary is missing.
        """
        if shutil.which(self._opencode_bin) is None:
            raise OpenCodeError(
                f"OpenCode binary {self._opencode_bin!r} was not found on PATH"
            )
        log_path = self.logs_dir() / "agent-list.log"
        failures: list[str] = []
        for subcommand in (["agent", "list"], ["debug", "agents"]):
            captured: list[str] = []
            exit_code = self._runner.run(
                [self._opencode_bin, *subcommand],
                cwd=self._workdir,
                env=None,
                log_path=log_path,
                on_stdout_line=captured.append,
                timeout_seconds=self._utility_timeout,
            )
            output = "\n".join(captured)
            # Some builds print help and exit 0 for unknown subcommands; require parsed agents.
            if exit_code == 0 and len(parse_agent_ids(output)) > 0:
                return output
            failures.append(
                f"`opencode {' '.join(subcommand)}` exited {exit_code}: {output[-1000:]}"
            )
        raise OpenCodeError("; ".join(failures))

    def list_agent_ids(self) -> set[str]:
        """Return the set of agent ids OpenCode resolves for the workdir."""
        return parse_agent_ids(self.list_agents())

    def run_implement(
        self,
        *,
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None = None,
        agent: str = "issue-worker",
        slice_number: int = 1,
    ) -> ImplementResult:
        """
        Block on the issue-worker agent until the process exits, then parse sentinels.

        A run killed at `implement_timeout_seconds` is returned with
        `timed_out=True` (plus whatever it streamed before the kill) rather
        than raised, so the caller can continue the work in another slice.

        Args:
            issue_number: GitHub issue number (for log naming).
            attempt: 1-based review/implement attempt counter.
            message: User message passed to OpenCode.
            session_id: Optional existing session to continue.
            agent: Primary agent name (default issue-worker).
            slice_number: 1-based time slice within this attempt (for log naming).
        """
        if isinstance(slice_number, bool) or not isinstance(slice_number, int) or slice_number < 1:
            raise ValueError("slice_number must be a positive int")
        log_path, parse_state, exit_code, timed_out = self._run_agent(
            kind="implement",
            issue_number=issue_number,
            attempt=attempt,
            message=message,
            session_id=session_id,
            agent=agent,
            timeout_seconds=self._implement_timeout,
            slice_number=slice_number,
        )
        assistant_text = parse_state.assistant_text
        status, pr_url = parse_implement_sentinels(assistant_text)
        return ImplementResult(
            exit_code=exit_code,
            session_id=parse_state.session_id,
            pr_url=pr_url,
            status=status,
            assistant_text=assistant_text,
            log_path=log_path,
            errors=tuple(parse_state.errors),
            timed_out=timed_out,
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
        log_path, parse_state, exit_code, timed_out = self._run_agent(
            kind="review",
            issue_number=issue_number,
            attempt=attempt,
            message=message,
            session_id=session_id,
            agent=agent,
            timeout_seconds=self._review_timeout,
        )
        if timed_out:
            # Reviews are not sliced: a review that runs out of time halts the issue.
            raise OpenCodeTimeout(
                f"{agent} exceeded {self._review_timeout}s and was killed (log: {log_path})"
            )
        assistant_text = parse_state.assistant_text
        return ReviewResult(
            exit_code=exit_code,
            session_id=parse_state.session_id,
            verdict=parse_review_sentinel(assistant_text),
            feedback=self._extract_review_feedback(assistant_text),
            assistant_text=assistant_text,
            log_path=log_path,
            errors=tuple(parse_state.errors),
        )

    def _run_agent(
        self,
        *,
        kind: Literal["implement", "review"],
        issue_number: int,
        attempt: int,
        message: str,
        session_id: str | None,
        agent: str,
        timeout_seconds: float | None,
        slice_number: int = 1,
    ) -> tuple[Path, NdjsonParseState, int, bool]:
        """
        Validate inputs, run one `opencode run`, and return (log, parsed, exit, timed_out).

        A timeout is reported through the flag rather than raised; the runner
        replays the output it captured before the kill, so `parsed` still holds
        the session id and any text the agent streamed.
        """
        self._validate_message(message)
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("issue_number must be a positive int")
        if not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive int")

        # Slice 1 keeps the historical name so existing log paths stay stable.
        slice_suffix = "" if slice_number == 1 else f"-slice{slice_number}"
        log_path = self.logs_dir() / f"{issue_number}-{kind}-{attempt}{slice_suffix}.log"
        parse_state = NdjsonParseState(session_id=session_id)
        command = self._build_command(agent=agent, message=message, session_id=session_id)
        slice_label = "" if slice_number == 1 else f".{slice_number}"
        label = f"#{issue_number} {kind} {attempt}{slice_label} · {agent}"
        # Tail while the process runs. Compaction has to be requested before the
        # next step; the post-exit replay is only for sentinel parsing.
        try:
            exit_code = self._runner.run(
                command,
                cwd=self._workdir,
                env=None,
                log_path=log_path,
                on_stdout_line=lambda line: ingest_ndjson_line(parse_state, line),
                timeout_seconds=timeout_seconds,
                live=self._live_output(label, log_path, self._make_watch(label)),
            )
        except OpenCodeTimeout:
            return log_path, parse_state, TIMEOUT_EXIT_CODE, True
        return log_path, parse_state, exit_code, False

    def _make_watch(self, label: str) -> CompactionWatch | None:
        """Build a per-run compaction watch, or None when compaction is disabled."""
        threshold = self._compact_tokens
        if threshold is None or self._compact_unsupported:
            return None

        def request(session_id: str) -> bool:
            return self._request_compact(label, session_id)

        def announce(message: str) -> None:
            self._status_line(f"{label} │ {message}")

        return CompactionWatch(threshold=threshold, request=request, announce=announce)

    def _request_compact(self, label: str, session_id: str) -> bool:
        """
        Admit a compaction request for one session.

        Uses the shared OpenCode server (`opencode api`), which is the same
        server `opencode run` is already talking to. The call returns when the
        request is accepted; OpenCode applies the summary at the next step.

        OpenCode builds without `opencode api` print the top-level help and
        exit 1. That is reported once and compaction is switched off for the
        rest of the run instead of failing every minute.
        """
        if self._compact_unsupported:
            return False
        if _SESSION_ID_RE.fullmatch(session_id) is None:
            self._status_line(f"{label} │ compact skipped; unsafe session id")
            return False
        command = [
            self._opencode_bin,
            "api",
            "post",
            f"/api/session/{session_id}/compact",
            "--data",
            "{}",
        ]
        log_path = self.logs_dir() / "compact.log"
        captured: list[str] = []
        try:
            exit_code = self._runner.run(
                command,
                cwd=self._workdir,
                env=None,
                log_path=log_path,
                on_stdout_line=captured.append,
                timeout_seconds=COMPACT_REQUEST_TIMEOUT_SECONDS,
            )
        except OpenCodeError as exc:
            self._status_line(f"{label} │ compact failed for {session_id}: {exc}")
            return False
        if exit_code != 0 and _looks_like_cli_help("\n".join(captured)):
            self._compact_unsupported = True
            self._status_line(
                f"{label} │ compaction disabled: this OpenCode build has no "
                f"`opencode api` command (log: {log_path}). OpenCode's own "
                "auto-compaction still applies."
            )
            return False
        if exit_code != 0:
            self._status_line(
                f"{label} │ compact failed for {session_id} (exit {exit_code}; log: {log_path})"
            )
            return False
        return True

    def _status_line(self, message: str) -> None:
        """Print an operational line, ignoring display errors."""
        status = self._status
        if status is None:
            return
        _safe_call(status, message)

    def _live_output(
        self,
        label: str,
        log_path: Path,
        watch: CompactionWatch | None,
    ) -> LiveOutput | None:
        """
        Build callbacks that run while the agent process is still alive.

        Activity lines are optional (`--quiet`). The tail still runs when
        compaction is on, so a quiet run can compact too.
        """
        activity = self._activity
        if activity is None and watch is None:
            return None
        if activity is not None:
            activity(f"{label} │ log: {log_path}")
        tool_calls = 0

        def on_line(line: str) -> None:
            nonlocal tool_calls
            if watch is not None:
                for session_id, tokens in parse_context_usages(line):
                    watch.observe(session_id, tokens)
            if activity is None:
                return
            summary = describe_event(line)
            if summary is None:
                return
            if not summary.startswith("says:"):
                tool_calls += 1
            activity(f"{label} │ {summary}")

        def on_idle(elapsed: float) -> None:
            if activity is None:
                return
            activity(
                f"{label} │ … still running ({format_duration(elapsed)} elapsed, "
                f"{tool_calls} actions so far)"
            )

        return LiveOutput(
            on_line=on_line,
            on_idle=on_idle if activity is not None else None,
            idle_interval_seconds=self._heartbeat_seconds,
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
        Prefer text after the final verdict sentinel as feedback for the next round.

        Falls back to the full assistant text when no sentinel is present.
        """
        match = _last_match(_REVIEW_VERDICT_RE, assistant_text)
        if match is None:
            return assistant_text.strip()
        after = assistant_text[match.end() :].strip()
        if after != "":
            return after
        before = assistant_text[: match.start()].strip()
        return before if before != "" else assistant_text.strip()
