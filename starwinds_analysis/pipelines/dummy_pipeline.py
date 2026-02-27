"""THIS FILE contains the standalone dummy per-file pipeline step for `sw-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path

from starwinds_analysis.pipelines.emit_payload import sw_emit_extra

log = logging.getLogger(__name__)
emit_log = logging.getLogger("starwinds_analysis.pipelines.emit.dummy_pipeline")


def name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string, then emit the payload.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    vowels = sum(ch.lower() in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    consonants = sum(ch.lower() not in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    emit_log.debug("emit letter_counts", extra=sw_emit_extra("letter_counts", {"vowels": vowels, "consonants": consonants}))
    log.info("%s vowels=%d consonants=%d", name, vowels, consonants)
    return vowels, consonants


def name_profile_payload(name: str) -> tuple[float, str, list[int]]:
    """
    Compute and emit additional filename payload values (float/string/array).
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    letters = [ch.lower() for ch in name if ch.isalpha()]
    letter_count = len(letters)
    vowel_count = sum(ch in {"a", "e", "i", "o", "u"} for ch in letters)
    consonant_count = letter_count - vowel_count
    vowel_fraction = float(vowel_count / letter_count) if letter_count else 0.0
    dominance = "vowel-rich" if vowel_count >= consonant_count else "consonant-rich"
    shape = [letter_count, vowel_count, consonant_count]
    emit_log.debug("emit name_vowel_fraction", extra=sw_emit_extra("name_vowel_fraction", vowel_fraction))
    emit_log.debug("emit name_dominance", extra=sw_emit_extra("name_dominance", dominance))
    emit_log.debug("emit name_shape", extra=sw_emit_extra("name_shape", shape))
    return vowel_fraction, dominance, shape


def process_plt_file(file_path: str | Path) -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    name = path.stem
    name_letter_counts(name)
    name_profile_payload(name)
