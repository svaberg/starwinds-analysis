"""The standalone dummy per-file pipeline step for `batwind-pipe`."""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug


def name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string, then record the payload.
    Used by: `batwind/pipelines/dummy_pipeline.py`, `test/test_batwind_pipe.py`
    """
    vowels = sum(ch.lower() in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    consonants = sum(ch.lower() not in {"a", "e", "i", "o", "u"} for ch in name if ch.isalpha())
    add_record("letter_counts %r", {"vowels": vowels, "consonants": consonants})
    return vowels, consonants


def name_profile_payload(name: str) -> tuple[float, str, list[int]]:
    """
    Compute and record additional filename payload values (float/string/array).
    Used by: `batwind/pipelines/dummy_pipeline.py`, `test/test_batwind_pipe.py`
    """
    letters = [ch.lower() for ch in name if ch.isalpha()]
    letter_count = len(letters)
    vowel_count = sum(ch in {"a", "e", "i", "o", "u"} for ch in letters)
    consonant_count = letter_count - vowel_count
    vowel_fraction = float(vowel_count / letter_count) if letter_count else 0.0
    dominance = "vowel-rich" if vowel_count >= consonant_count else "consonant-rich"
    shape = [letter_count, vowel_count, consonant_count]
    add_record("name_vowel_fraction %r", vowel_fraction)
    add_record("name_dominance %r", dominance)
    add_record("name_shape %r", shape)
    return vowel_fraction, dominance, shape


def name_codepoints_payload(name: str) -> np.ndarray:
    """
    Compute and record filename code points as a NumPy array.
    Used by: `batwind/pipelines/dummy_pipeline.py`, `test/test_batwind_pipe.py`
    """
    codepoints = np.array([ord(ch) for ch in name], dtype=np.int32)
    add_record("name_codepoints %r", codepoints)
    return codepoints


def name_waveform_payload(name: str) -> np.ndarray:
    """
    Compute and record a large NumPy waveform payload to exercise sidecar output.
    Used by: `batwind/pipelines/dummy_pipeline.py`, `test/test_batwind_pipe.py`
    """
    phase = float(len(name))
    waveform = np.linspace(0.0, 2.0 * np.pi, 131_072, dtype=np.float64) + phase
    add_record("name_waveform %r", waveform)
    return waveform


def process_plt_file(file_path: str | Path) -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `batwind/pipelines/batwind_pipe.py`, `test/test_batwind_pipe.py`
    """
    path = Path(file_path)
    log.info("%s", path.name)
    name = path.stem
    name_letter_counts(name)
    name_profile_payload(name)
    name_codepoints_payload(name)
    name_waveform_payload(name)
