"""THIS FILE contains the generic ambient result sink for pipeline runs."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Callable

_result_sink: ContextVar[Callable[[str, Any], None] | None] = ContextVar("sw_pipe_result_sink", default=None)


def set_result_sink(sink: Callable[[str, Any], None]) -> Token:
    """
    Set the active per-context pipeline result sink.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return _result_sink.set(sink)


def reset_result_sink(token: Token) -> None:
    """
    Reset the active pipeline result sink to a previous context value.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    _result_sink.reset(token)


def emit_result(key: str, value: Any) -> None:
    """
    Emit a key/value result to the active sink when one is configured.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`
    """
    sink = _result_sink.get()
    if sink is None:
        return
    sink(key, value)
