"""THIS FILE contains the standalone dummy per-file pipeline step for `sw-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path

from starwinds_analysis.pipelines.result_context import emit_result

log = logging.getLogger(__name__)


def name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    vowels = sum(ch.lower() in {"a", "e", "i", "o", "u"} for ch in str(name) if ch.isalpha())
    consonants = sum(ch.lower() not in {"a", "e", "i", "o", "u"} for ch in str(name) if ch.isalpha())
    return vowels, consonants


def compute_and_emit_letter_counts(name: str) -> tuple[int, int]:
    """
    Compute and then emit letter-count payload from a name string.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`
    """
    vowels, consonants = name_letter_counts(name)
    emit_result("letter_counts", {"vowels": vowels, "consonants": consonants})
    return vowels, consonants


def process_plt_file(file_path: str | Path) -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    vowels, consonants = compute_and_emit_letter_counts(path.stem)
    log.info("%s vowels=%d consonants=%d", path.name, vowels, consonants)
