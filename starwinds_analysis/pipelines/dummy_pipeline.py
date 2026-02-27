"""THIS FILE contains the standalone dummy per-file pipeline step for `sw-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starwinds_analysis.pipelines.sw_pipe import SwPipeResults

log = logging.getLogger(__name__)


def name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    vowels = sum(ch.lower() in {"a", "e", "i", "o", "u"} for ch in str(name) if ch.isalpha())
    consonants = sum(ch.lower() not in {"a", "e", "i", "o", "u"} for ch in str(name) if ch.isalpha())
    return vowels, consonants


def process_plt_file(file_path: str | Path, results: "SwPipeResults") -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    vowels, consonants = name_letter_counts(path.stem)
    log.info("%s vowels=%d consonants=%d", path.name, vowels, consonants)
    results.add_processed_file(path)
    file_key = path.resolve().relative_to(results.directory.resolve()).as_posix()
    results.add_computed_result(file_key, vowels=vowels, consonants=consonants)
