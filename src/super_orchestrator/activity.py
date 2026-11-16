"""Human-readable summaries of OpenCode NDJSON events for live console output."""

from __future__ import annotations

import json
from collections.abc import Mapping

_MAX_DETAIL_CHARS = 140

# Tool input keys worth showing, in priority order, when a tool has no title.
_DETAIL_KEYS: tuple[str, ...] = (
    "command",
    "filePath",
    "path",
    "pattern",
    "url",
    "query",
    "name",
    "description",
)


def format_duration(seconds: float) -> str:
    """Render seconds as `45s`, `12m05s`, or `1h02m`."""
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _one_line(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    """Collapse whitespace and truncate for single-line console output."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 1]}…"


def _describe_tool(part: Mapping[str, object]) -> str | None:
    """Summarize a completed or failed tool part."""
    tool = part.get("tool")
    if not isinstance(tool, str):
        return None
    state = part.get("state")
    if not isinstance(state, Mapping):
        return f"tool {tool}"
    status = state.get("status")
    raw_input = state.get("input")
    tool_input: Mapping[str, object] = raw_input if isinstance(raw_input, Mapping) else {}

    if tool == "task":
        subagent = tool_input.get("subagent_type")
        description = tool_input.get("description")
        name = subagent if isinstance(subagent, str) else "subagent"
        detail = f": {_one_line(description)}" if isinstance(description, str) else ""
        prefix = "subagent failed" if status == "error" else "subagent"
        return f"{prefix} {name}{detail}"

    if tool == "todowrite":
        todos = tool_input.get("todos")
        if isinstance(todos, list):
            done = sum(
                1
                for todo in todos
                if isinstance(todo, Mapping) and todo.get("status") == "completed"
            )
            return f"todo list updated ({done}/{len(todos)} done)"

    detail = ""
    for key in _DETAIL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip() != "":
            detail = _one_line(value)
            break
    if detail == "":
        title = state.get("title")
        if isinstance(title, str) and title.strip() != "":
            detail = _one_line(title)

    if status == "error":
        error = state.get("error")
        reason = f" ({_one_line(error, 80)})" if isinstance(error, str) else ""
        return f"✗ {tool} {detail}{reason}".rstrip()
    return f"{tool} {detail}".rstrip()


def describe_event(line: str) -> str | None:
    """
    Turn one OpenCode `--format json` line into a short status message.

    Returns None for events not worth showing (step boundaries, reasoning,
    blank or non-JSON noise).
    """
    if not isinstance(line, str):
        raise TypeError("line must be a str")
    stripped = line.strip()
    if stripped == "" or not stripped.startswith("{"):
        return None
    try:
        payload: object = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None

    event_type = payload.get("type")
    part_raw = payload.get("part")
    part: Mapping[str, object] = part_raw if isinstance(part_raw, Mapping) else {}

    if event_type == "tool_use":
        return _describe_tool(part)
    if event_type == "text":
        text = part.get("text")
        if isinstance(text, str) and text.strip() != "":
            first_line = next(line for line in text.splitlines() if line.strip() != "")
            return f"says: {_one_line(first_line)}"
        return None
    if event_type == "error":
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = error.get("message") or error.get("name")
            if isinstance(message, str):
                return f"✗ error: {_one_line(message)}"
        if isinstance(error, str):
            return f"✗ error: {_one_line(error)}"
        return "✗ error"
    return None
