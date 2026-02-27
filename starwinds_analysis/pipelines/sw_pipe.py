"""THIS FILE contains the generic `sw-pipe` orchestration CLI.

It discovers `.plt` files in a working directory and runs a per-file pipeline
handler. The current handler is intentionally a placeholder.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
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
    noclobber: bool
    discovered_files: list[Path] = field(default_factory=list)
    processed_files: list[Path] = field(default_factory=list)
    skipped_files: list[Path] = field(default_factory=list)
    computed_results: dict[str, dict[str, int]] = field(default_factory=dict)
    state_file: Path | None = None

    def add_processed_file(self, file_path: str | Path) -> None:
        """
        Record a processed file path in the run results.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        self.processed_files.append(Path(file_path))

    def add_skipped_file(self, file_path: str | Path) -> None:
        """
        Record a skipped file path in the run results.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        self.skipped_files.append(Path(file_path))

    def add_computed_result(self, file_key: str, *, vowels: int, consonants: int) -> None:
        """
        Store computed per-file demo results.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        self.computed_results[str(file_key)] = {
            "vowels": int(vowels),
            "consonants": int(consonants),
        }


def _state_file_path(directory: str | Path) -> Path:
    """
    Default state-file path for processed `.plt` tracking.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return Path(directory) / ".sw-pipe.processed.json"


def _relative_file_key(file_path: str | Path, *, base_dir: str | Path) -> str:
    """
    Stable relative key for a processed file inside the tracked directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return Path(file_path).resolve().relative_to(Path(base_dir).resolve()).as_posix()


def _name_letter_counts(name: str) -> tuple[int, int]:
    """
    Count vowels and consonants in a name string (alphabetic characters only).
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    vowels = 0
    consonants = 0
    for ch in str(name).lower():
        if not ch.isalpha():
            continue
        if ch in {"a", "e", "i", "o", "u"}:
            vowels += 1
        else:
            consonants += 1
    return vowels, consonants


def _load_state(state_file: str | Path) -> tuple[set[str], dict[str, dict[str, int]]]:
    """
    Load processed-file keys and computed results from state file.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    path = Path(state_file)
    if not path.exists():
        return set(), {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return set(), {}
    files = payload.get("processed_files", [])
    processed_keys = {str(item) for item in files} if isinstance(files, list) else set()
    computed = payload.get("computed_results", {})
    if not isinstance(computed, dict):
        computed = {}
    out_computed: dict[str, dict[str, int]] = {}
    for key, value in computed.items():
        if not isinstance(value, dict):
            continue
        vowels = value.get("vowels")
        consonants = value.get("consonants")
        if isinstance(vowels, int) and isinstance(consonants, int):
            out_computed[str(key)] = {"vowels": vowels, "consonants": consonants}
    return processed_keys, out_computed


def _save_state(
    state_file: str | Path,
    *,
    processed_keys: set[str],
    computed_results: dict[str, dict[str, int]],
) -> None:
    """
    Save processed keys and computed results to state file.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_files": sorted(processed_keys),
        "computed_results": computed_results,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


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
    Placeholder per-file pipeline step with demo computed filename metrics.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    path = Path(file_path)
    vowels, consonants = _name_letter_counts(path.stem)
    log.info("%s vowels=%d consonants=%d", path.name, vowels, consonants)
    results.add_processed_file(path)
    file_key = _relative_file_key(path, base_dir=results.directory)
    results.add_computed_result(file_key, vowels=vowels, consonants=consonants)


def run_sw_pipe(
    directory: str | Path = ".",
    *,
    recursive: bool = False,
    noclobber: bool = False,
    process_file: Callable[[Path, SwPipeResults], None] = process_plt_file,
) -> SwPipeResults:
    """
    Run `sw-pipe` over all discovered `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    state_file = _state_file_path(directory)
    known_processed, known_computed = _load_state(state_file)
    files = discover_plt_files(directory, recursive=recursive)
    results = SwPipeResults(
        directory=Path(directory),
        recursive=recursive,
        noclobber=noclobber,
        discovered_files=list(files),
        computed_results=dict(known_computed),
        state_file=state_file,
    )
    log.debug(
        "sw-pipe discovered %d .plt files in %s (recursive=%s, noclobber=%s)",
        len(files),
        Path(directory),
        recursive,
        noclobber,
    )
    processed_keys = set(known_processed)
    for file_path in files:
        file_key = _relative_file_key(file_path, base_dir=directory)
        if noclobber and file_key in known_processed:
            results.add_skipped_file(file_path)
            log.debug("sw-pipe skipping already processed file %s", file_path.name)
            continue
        process_file(file_path, results)
        processed_keys.add(file_key)
    _save_state(
        state_file,
        processed_keys=processed_keys,
        computed_results=results.computed_results,
    )
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
    parser.add_argument(
        "--noclobber",
        action="store_true",
        help="Skip files already listed in the sw-pipe state file.",
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
    run_sw_pipe(
        args.directory,
        recursive=bool(args.recursive),
        noclobber=bool(args.noclobber),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
