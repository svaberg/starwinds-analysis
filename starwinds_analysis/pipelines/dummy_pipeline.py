"""THIS FILE contains the standalone dummy per-file pipeline step for `sw-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path

from starwinds_analysis.pipelines.result_context import emit_result

log = logging.getLogger(__name__)


def name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string, then emit the payload.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    vowels = sum(ch.lower() in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    consonants = sum(ch.lower() not in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    emit_result("letter_counts", {"vowels": vowels, "consonants": consonants})
    log.info("%s vowels=%d consonants=%d", name, vowels, consonants)
    return vowels, consonants


def process_plt_file(file_path: str | Path) -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    name_letter_counts(path.stem)
