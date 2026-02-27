"""THIS FILE contains the standalone dummy per-file pipeline step for `sw-pipe`."""

from __future__ import annotations

from contextvars import ContextVar, Token
import logging
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)
_result_sink: ContextVar[Callable[[str, Any], None] | None] = ContextVar("sw_pipe_result_sink", default=None)


def name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    vowels = sum(ch.lower() in {"a", "e", "i", "o", "u"} for ch in str(name) if ch.isalpha())
    consonants = sum(ch.lower() not in {"a", "e", "i", "o", "u"} for ch in str(name) if ch.isalpha())
    return vowels, consonants


def set_result_sink(sink: Callable[[str, Any], None]) -> Token:
    """
    Set the current `sw-pipe` result sink for the active context.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return _result_sink.set(sink)


def reset_result_sink(token: Token) -> None:
    """
    Reset the current `sw-pipe` result sink to the previous context value.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    _result_sink.reset(token)


def emit_result(key: str, value: Any) -> None:
    """
    Emit a pipeline result to the active sink when one is configured.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`
    """
    sink = _result_sink.get()
    if sink is None:
        return
    sink(key, value)


def process_plt_file(file_path: str | Path) -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    vowels, consonants = name_letter_counts(path.stem)
    log.info("%s vowels=%d consonants=%d", path.name, vowels, consonants)
    emit_result("letter_counts", {"vowels": vowels, "consonants": consonants})
