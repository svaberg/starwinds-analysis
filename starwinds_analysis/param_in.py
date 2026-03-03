"""THIS FILE contains a small reader for SWMF-style `PARAM.in` files.

It flattens `#INCLUDE` statements when the child file exists, preserves sessions,
components, and duplicate commands, and exposes simple typed access helpers.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


def _strip_lines(path) -> list[str]:
    """Read a text file and return stripped lines."""
    with open(path, encoding="utf-8") as stream:
        return [line.strip() for line in stream]


def flatten_includes(file_path) -> list[str]:
    """Flatten resolvable `#INCLUDE` directives into one flat line stream."""
    path = Path(file_path)
    content = _strip_lines(path)
    flat_lines: list[str] = []
    line_id = 0

    while line_id < len(content):
        line = content[line_id]
        if line.startswith("#INCLUDE") and line_id + 1 < len(content):
            child_name = content[line_id + 1].split()[0]
            child_path = path.parent / child_name
            if child_path.exists():
                flat_lines.extend(flatten_includes(child_path))
                line_id += 2
                continue
        flat_lines.append(line)
        line_id += 1

    return flat_lines


def _new_session():
    """Create one parsed session container."""
    return OrderedDict()


def _ensure_component(session, component_name: str):
    """Ensure a component entry exists inside a session."""
    session.setdefault(component_name, OrderedDict())
    return session[component_name]


def parse_sessions(flat_lines) -> list[OrderedDict]:
    """Parse flat config lines into sessions, components, commands, and blocks."""
    sessions = [_new_session()]
    session = sessions[-1]
    component_name = "root"
    current_command = None

    for line in flat_lines:
        if not line:
            continue
        if line.startswith("!"):
            continue
        if line.lower().startswith("begin session:"):
            continue

        if line.startswith("#BEGIN_COMP"):
            tokens = line.split()
            component_name = tokens[1] if len(tokens) > 1 else "root"
            current_command = None
            _ensure_component(session, component_name)
            continue

        if line.startswith("#END_COMP"):
            component_name = "root"
            current_command = None
            continue

        if line.startswith("#RUN") or line.startswith("#END"):
            if session:
                session = _new_session()
                sessions.append(session)
            component_name = "root"
            current_command = None
            continue

        if line.startswith("#"):
            current_command = line.split()[0]
            component = _ensure_component(session, component_name)
            component.setdefault(current_command, []).append([])
            continue

        if current_command is None:
            continue

        component = _ensure_component(session, component_name)
        component[current_command][-1].append(line)

    if sessions and not sessions[-1]:
        sessions.pop()
    return sessions


def _parse_scalar_token(text):
    """Parse one first-token scalar using SWMF-style `T/F/int/float/string` rules."""
    token = str(text).split()[0]
    upper = token.upper()
    if upper == "T":
        return True
    if upper == "F":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _parse_value_text(text):
    """Parse a full value field without discarding embedded spaces."""
    value_text = str(text).strip()
    upper = value_text.upper()
    if upper == "T":
        return True
    if upper == "F":
        return False
    try:
        return int(value_text)
    except ValueError:
        pass
    try:
        return float(value_text)
    except ValueError:
        return value_text


def _split_value_and_label(line: str) -> tuple[str, str]:
    """Split one SWMF parameter line into a value field and a trailing label."""
    text = str(line).strip()
    for index in range(len(text) - 1):
        if text[index].isspace() and text[index + 1].isspace():
            value = text[:index].strip()
            label = text[index + 1 :].strip()
            while label and label[0].isspace():
                label = label[1:]
            return value, label
    tokens = text.split()
    if len(tokens) <= 1:
        return text, ""
    return tokens[0], " ".join(tokens[1:])


class ParamIn:
    """Read and query one `PARAM.in` file as sessions, components, and commands."""

    def __init__(self, file_path):
        """Parse a `PARAM.in` file immediately."""
        self.path = Path(file_path)
        self.flat_lines = flatten_includes(self.path)
        self.sessions = parse_sessions(self.flat_lines)

    @classmethod
    def from_file(cls, file_path):
        """Construct a parsed config from disk."""
        return cls(file_path)

    def num_sessions(self) -> int:
        """Return the number of parsed sessions."""
        return len(self.sessions)

    def get_commands(self, command, *, component="root", session=None) -> list[list[str]]:
        """Return all blocks for one command in one session/component."""
        if session is None:
            blocks: list[list[str]] = []
            for session_data in self.sessions:
                component_data = session_data.get(component, {})
                blocks.extend(component_data.get(command, ()))
            return blocks
        session_data = self.sessions[int(session)]
        component_data = session_data.get(component, {})
        return list(component_data.get(command, ()))

    def get_command(self, command, *, component="root", session=None, occurrence=-1) -> list[str] | None:
        """Return one command block, defaulting to the most recent occurrence."""
        blocks = self.get_commands(command, component=component, session=session)
        if not blocks:
            return None
        return blocks[occurrence]

    def get_param_line(
        self,
        command,
        index,
        *,
        component="root",
        session=None,
        occurrence=-1,
    ) -> str | None:
        """Return one raw parameter line from a command block."""
        block = self.get_command(command, component=component, session=session, occurrence=occurrence)
        if block is None:
            return None
        if index < 0 or index >= len(block):
            return None
        return block[index]

    def get_param(
        self,
        command,
        index,
        *,
        component="root",
        session=None,
        occurrence=-1,
    ):
        """Return one parsed parameter value from the first token on the line."""
        line = self.get_param_line(command, index, component=component, session=session, occurrence=occurrence)
        if line is None:
            return None
        return _parse_scalar_token(line)

    def get_named_params(self, command, *, component="root", session=None, occurrence=-1) -> OrderedDict:
        """Return an ordered mapping from trailing labels to parsed values."""
        block = self.get_command(command, component=component, session=session, occurrence=occurrence)
        out = OrderedDict()
        if block is None:
            return out
        for line in block:
            value_text, label = _split_value_and_label(line)
            if not value_text:
                continue
            key = label or f"param_{len(out)}"
            out[key] = _parse_value_text(value_text)
        return out

    def __str__(self) -> str:
        """Summarize the parsed config briefly."""
        component_count = sum(len(session) for session in self.sessions)
        command_count = 0
        for session in self.sessions:
            for component in session.values():
                command_count += sum(len(blocks) for blocks in component.values())
        return (
            f"ParamIn(path={self.path.name!r}, sessions={self.num_sessions()}, "
            f"components={component_count}, command_blocks={command_count})"
        )
