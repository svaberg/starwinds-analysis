"""THIS FILE defines the `sw_emit` payload schema used by pipeline logging."""

from __future__ import annotations

from logging import LogRecord
from typing import TypedDict


class SwEmitPayload(TypedDict):
    """
    Structured payload stored on a log record for pipeline result capture.
    Used by: `starwinds_analysis/pipelines/emit_payload.py`
    """

    key: str
    value: object


def sw_emit_extra(key: str, value: object) -> dict[str, SwEmitPayload]:
    """
    Build `logging.extra` payload for one emitted pipeline quantity.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`
    """
    return {"sw_emit": {"key": key, "value": value}}


def read_sw_emit(record: LogRecord) -> SwEmitPayload | None:
    """
    Read and validate `sw_emit` payload from a log record.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    payload = getattr(record, "sw_emit", None)
    if not isinstance(payload, dict):
        return None
    key = payload.get("key")
    if not isinstance(key, str):
        return None
    return {"key": key, "value": payload.get("value")}
