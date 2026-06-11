"""Minimal Server-Sent Events parser for streaming HTTP responses."""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def iter_sse_events(lines: Iterable[str | bytes]) -> Iterator[dict[str, str]]:
    """Yield ``{"event": ..., "data": ...}`` dictionaries from SSE lines."""
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")

        if not line:
            if data_lines:
                yield {"event": event_name, "data": "\n".join(data_lines)}
            event_name = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    if data_lines:
        yield {"event": event_name, "data": "\n".join(data_lines)}
