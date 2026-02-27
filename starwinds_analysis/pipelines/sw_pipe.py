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
import re
from typing import Callable

log = logging.getLogger(__name__)
_EMIT_PATTERN = re.compile(r"^([A-Za-z0-9_]+)\b")


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
    computed_results: dict[str, dict[str, object]] = field(default_factory=dict)
    state_file: Path | None = None


class _SwEmitHandler(logging.Handler):
    """
    Capture `sw_emit` payloads from log records into a per-file results mapping.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """

    def __init__(self, target: dict[str, object]):
        """
        Initialize a handler that writes captured emit payloads into `target`.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        super().__init__(level=logging.NOTSET)
        self.target = target

    def emit(self, record: logging.LogRecord) -> None:
        """
        Pull emit payloads from logger template/args and store them by key.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        parsed = _parse_emit_record(record)
        if parsed is None:
            return
        key, value = parsed
        self.target[key] = value


def _parse_emit_record(record: logging.LogRecord) -> tuple[str, object] | None:
    """
    Parse `<key> ...` logger template and args into a key/value payload.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    if not isinstance(record.msg, str):
        return None
    match = _EMIT_PATTERN.match(record.msg)
    if match is None:
        return None
    key = match.group(1)
    args = record.args
    if isinstance(args, tuple):
        if len(args) == 0:
            return key, None
        if len(args) == 1:
            return key, args[0]
        return key, list(args)
    if isinstance(args, dict):
        return key, dict(args)
    if args is None:
        return key, None
    return key, args


class _PipelineLogFormatter(logging.Formatter):
    """
    Format pipeline logs as `[level] source message`.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Render one pipeline log record with lowercase level and short source name.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        source = record.name.rsplit(".", 1)[-1]
        message = record.getMessage()
        out = f"[{record.levelname.lower()}] {source} {message}"
        if record.exc_info:
            out = f"{out}\n{self.formatException(record.exc_info)}"
        return out


def configure_emit_logger(level_name: str = "WARNING") -> None:
    """
    Configure the dedicated emit logger with its own stream handler and level.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    emit_logger = logging.getLogger("starwinds_analysis.pipelines.emit")
    emit_logger.setLevel(logging.DEBUG)
    emit_logger.handlers.clear()
    emit_handler = logging.StreamHandler()
    emit_handler.setLevel(getattr(logging, str(level_name).upper()))
    emit_handler.setFormatter(_PipelineLogFormatter())
    emit_logger.addHandler(emit_handler)
    emit_logger.propagate = False


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


def _load_state(state_file: str | Path) -> tuple[set[str], dict[str, dict[str, object]]]:
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
    if isinstance(computed, dict):
        return processed_keys, {str(key): value for key, value in computed.items() if isinstance(value, dict)}
    return processed_keys, {}


def _save_state(
    state_file: str | Path,
    *,
    processed_keys: set[str],
    computed_results: dict[str, dict[str, object]],
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


def run_sw_pipe(
    directory: str | Path = ".",
    *,
    recursive: bool = False,
    noclobber: bool = False,
    process_file: Callable[[Path], None] | None = None,
) -> SwPipeResults:
    """
    Run `sw-pipe` over all discovered `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    if process_file is None:
        from starwinds_analysis.pipelines.dummy_pipeline import process_plt_file as process_file

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
    emit_logger = logging.getLogger("starwinds_analysis.pipelines.emit")
    for file_path in files:
        file_key = _relative_file_key(file_path, base_dir=directory)
        if noclobber and file_key in known_processed:
            results.skipped_files.append(file_path)
            log.debug("sw-pipe skipping already processed file %s", file_path.name)
            continue
        file_results: dict[str, object] = {}
        emit_handler = _SwEmitHandler(file_results)
        emit_logger.addHandler(emit_handler)
        try:
            process_file(file_path)
        finally:
            emit_logger.removeHandler(emit_handler)
            emit_handler.close()
        if file_results:
            results.computed_results[file_key] = file_results
        else:
            results.computed_results.pop(file_key, None)
        results.processed_files.append(file_path)
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
        "--emit-log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Emit logger level for stdout/stderr stream output (default: WARNING).",
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
        force=True,
    )
    formatter = _PipelineLogFormatter()
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    configure_emit_logger(str(args.emit_log_level))
    run_sw_pipe(
        args.directory,
        recursive=bool(args.recursive),
        noclobber=bool(args.noclobber),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
