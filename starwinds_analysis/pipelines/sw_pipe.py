"""THIS FILE contains the generic `sw-pipe` orchestration CLI.

It discovers `.plt` files in a working directory and runs a per-file pipeline
handler. The current handler is intentionally a placeholder.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class SwPipeResults:
    """
    Minimal results container for one `sw-pipe` run.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    directory: Path
    recursive: bool
    discovered_files: list[Path] = field(default_factory=list)
    processed_files: list[Path] = field(default_factory=list)

    def add_processed_file(self, file_path: str | Path) -> None:
        """
        Record a processed file path in the run results.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        self.processed_files.append(Path(file_path))


def discover_plt_files(directory: str | Path = ".", *, recursive: bool = False) -> list[Path]:
    """
    Discover `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    base = Path(directory)
    paths = base.rglob("*") if recursive else base.iterdir()
    files = [path for path in paths if path.is_file() and path.suffix.lower() == ".plt"]
    return sorted(files)


def process_plt_file(file_path: str | Path, results: SwPipeResults) -> None:
    """
    Placeholder per-file pipeline step; logs only the file name.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    log.info("%s", path.name)
    results.add_processed_file(path)


def run_sw_pipe(
    directory: str | Path = ".",
    *,
    recursive: bool = False,
    process_file: Callable[[Path, SwPipeResults], None] = process_plt_file,
) -> SwPipeResults:
    """
    Run `sw-pipe` over all discovered `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    files = discover_plt_files(directory, recursive=recursive)
    results = SwPipeResults(
        directory=Path(directory),
        recursive=recursive,
        discovered_files=list(files),
    )
    log.debug(
        "sw-pipe discovered %d .plt files in %s (recursive=%s)",
        len(files),
        Path(directory),
        recursive,
    )
    for file_path in files:
        process_file(file_path, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    """
    Build the `sw-pipe` CLI argument parser.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    parser = argparse.ArgumentParser(description="Run the starwinds generic .plt pipeline.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan for .plt files (default: current directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for .plt files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for `sw-pipe`.
    Used by: CLI (`sw-pipe`), `test/test_sw_pipe.py`
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(message)s",
        force=True,
    )
    run_sw_pipe(args.directory, recursive=bool(args.recursive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
