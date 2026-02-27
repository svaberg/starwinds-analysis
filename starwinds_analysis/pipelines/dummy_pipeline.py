"""THIS FILE contains the standalone dummy per-file pipeline step for `sw-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def process_plt_file(file_path: str | Path) -> None:
    """
    Demo pipeline step for `.plt` files, separate from orchestration.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    log.info("%s", path.name)
