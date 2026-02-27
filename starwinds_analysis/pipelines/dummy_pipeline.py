"""THIS FILE contains the standalone dummy per-file pipeline step for `sw-pipe`."""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)
emit_log = logging.getLogger(f"emit.{__name__}")


def name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string, then emit the payload.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    vowels = sum(ch.lower() in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    consonants = sum(ch.lower() not in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    emit_log.debug("letter_counts %r", {"vowels": vowels, "consonants": consonants})
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
    emit_log.debug("name_vowel_fraction %r", vowel_fraction)
    emit_log.debug("name_dominance %r", dominance)
    emit_log.debug("name_shape %r", shape)
    return vowel_fraction, dominance, shape


def name_codepoints_payload(name: str) -> np.ndarray:
    """
    Compute and emit filename code points as a NumPy array.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    codepoints = np.array([ord(ch) for ch in name], dtype=np.int32)
    emit_log.debug("name_codepoints %r", codepoints)
    return codepoints


def name_waveform_payload(name: str) -> np.ndarray:
    """
    Compute and emit a large NumPy waveform payload to exercise sidecar output.
    Used by: `starwinds_analysis/pipelines/dummy_pipeline.py`, `test/test_sw_pipe.py`
    """
    phase = float(len(name))
    waveform = np.linspace(0.0, 2.0 * np.pi, 131_072, dtype=np.float64) + phase
    emit_log.debug("name_waveform %r", waveform)
    return waveform


def process_plt_file(file_path: str | Path) -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    log.info("%s", path.name)
    name = path.stem
    name_letter_counts(name)
    name_profile_payload(name)
    name_codepoints_payload(name)
    name_waveform_payload(name)
