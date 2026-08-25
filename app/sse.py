# This file parses Server-Sent Events (SSE), which are used for streaming responses. It reads incoming data line by line, converts it into structured events containing the event type and message data, and yields them one at a time. This enables real-time token-by-token responses instead of waiting for the full output.

"""Minimal Server-Sent Events parser for streaming HTTP responses."""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def iter_sse_events(lines: Iterable[str | bytes]) -> Iterator[dict[str, str]]:  #Receives streamed data.
    """Yield ``{"event": ..., "data": ...}`` dictionaries from SSE lines."""
    event_name = "message" #Default event.
    data_lines: list[str] = []  #Stores received text.

    for raw_line in lines: # read stream  line by line 
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line # if in bytes convert to string 
        line = line.rstrip("\r\n") # remove new lines 

        if not line: # when empty line is encounter it means the event is completed 
            if data_lines:
                yield {"event": event_name, "data": "\n".join(data_lines)}
            event_name = "message" #reset
            data_lines = [] # clear old data 
            continue

        if line.startswith(":"):
            continue # ignore comments

        field, separator, value = line.partition(":") # splits the line 
        if separator and value.startswith(" "): # removes space after colon
            value = value[1:]

        if field == "event":
            event_name = value 
        elif field == "data":
            data_lines.append(value)

    if data_lines:# Still remaining data? return the final event 
        yield {"event": event_name, "data": "\n".join(data_lines)}
