"""THIS FILE contains the generic `sw-pipe` orchestration CLI.

It discovers `.plt` files in a working directory and runs a per-file pipeline
handler. Built-in handlers are `dummy` and `quicklook2d`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, field
import hashlib
import json
import logging
import numpy as np
from pathlib import Path
import re
from typing import Callable

log = logging.getLogger(__name__)
_RECORD_PATTERN = re.compile(r"^([A-Za-z0-9_]+)\b")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_DEFAULT_ARRAY_OFFLOAD_MIN_BYTES = 1_000_000
_DEFAULT_JSON_WARN_BYTES = 10_000_000


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


class _SwRecordHandler(logging.Handler):
    """
    Capture `sw_record` payloads from log records into a per-file results mapping.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """

    def __init__(
        self,
        target: dict[str, object],
        *,
        file_key: str,
        artifacts_root: str | Path,
        array_offload_min_bytes: int = _DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
    ):
        """
        Initialize a handler that writes captured record payloads into `target`.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        super().__init__(level=logging.NOTSET)
        self.target = target
        self.file_key = file_key
        self.artifacts_root = Path(artifacts_root)
        self.array_offload_min_bytes = int(array_offload_min_bytes)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Pull record payloads from logger template/args and store them by key.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        parsed = _parse_record_payload(record)
        if parsed is None:
            return
        key, value = parsed
        module_name = record.name[len("recorder.") :] if record.name.startswith("recorder.") else record.name
        normalized = _normalize_recorded_value(
            value,
            file_key=self.file_key,
            field_key=key,
            artifacts_root=self.artifacts_root,
            array_offload_min_bytes=self.array_offload_min_bytes,
        )
        self.target[key] = {
            "value": normalized,
            "source": {
                "module": module_name,
                "function": record.funcName,
                "line": int(record.lineno),
            },
        }


def _parse_record_payload(record: logging.LogRecord) -> tuple[str, object] | None:
    """
    Parse `<key> ...` logger template and args into a key/value payload.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    if not isinstance(record.msg, str):
        return None
    match = _RECORD_PATTERN.match(record.msg)
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


def _utc_now_iso() -> str:
    """
    Return the current UTC time in ISO-8601 format with `Z`.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_name(text: str) -> str:
    """
    Convert arbitrary text to a filesystem-safe token.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return _SAFE_NAME_PATTERN.sub("_", str(text)).strip("._") or "value"


def _array_artifact_relpath(file_key: str, field_key: str) -> str:
    """
    Relative artifact path for one recorded array payload.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    file_token = _safe_name(str(file_key).replace("/", "__"))
    field_token = _safe_name(field_key)
    return f"sw-pipe.artifacts/{file_token}__{field_token}.npy"


def _normalize_recorded_value(
    value: object,
    *,
    file_key: str,
    field_key: str,
    artifacts_root: str | Path,
    array_offload_min_bytes: int,
) -> object:
    """
    Convert recorded values into JSON-safe payloads with array offloading support.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    if isinstance(value, np.ndarray):
        if value.nbytes >= int(array_offload_min_bytes):
            rel_path = _array_artifact_relpath(file_key, field_key)
            out_path = Path(artifacts_root).parent / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, value)
            return {"path": rel_path}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [
            _normalize_recorded_value(
                item,
                file_key=file_key,
                field_key=field_key,
                artifacts_root=artifacts_root,
                array_offload_min_bytes=array_offload_min_bytes,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _normalize_recorded_value(
                item,
                file_key=file_key,
                field_key=field_key,
                artifacts_root=artifacts_root,
                array_offload_min_bytes=array_offload_min_bytes,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(k): _normalize_recorded_value(
                v,
                file_key=file_key,
                field_key=field_key,
                artifacts_root=artifacts_root,
                array_offload_min_bytes=array_offload_min_bytes,
            )
            for k, v in value.items()
        }
    return value


def _sha256_file(path: str | Path) -> str:
    """
    Compute SHA-256 for a file.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
        if record.name.startswith("recorder."):
            origin = f"{record.name}.{record.funcName}:{record.lineno}"
        else:
            origin = source
        out = f"[{record.levelname.lower()}] {origin} {message}"
        if record.exc_info:
            out = f"{out}\n{self.formatException(record.exc_info)}"
        return out


def configure_recorder_logger(level_name: str = "WARNING") -> None:
    """
    Configure the dedicated recorder logger with its own stream handler and level.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    recorder_logger = logging.getLogger("recorder")
    recorder_logger.setLevel(logging.DEBUG)
    recorder_logger.handlers.clear()
    recorder_handler = logging.StreamHandler()
    recorder_handler.setLevel(getattr(logging, str(level_name).upper()))
    recorder_handler.setFormatter(_PipelineLogFormatter())
    recorder_logger.addHandler(recorder_handler)
    recorder_logger.propagate = False


def _state_file_path(directory: str | Path) -> Path:
    """
    Default state-file path for processed `.plt` tracking.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return Path(directory) / "sw-pipe.processed.json"


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
    json_warn_bytes: int = _DEFAULT_JSON_WARN_BYTES,
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
    payload_text = json.dumps(payload, indent=2)
    payload_size_bytes = len(payload_text.encode("utf-8"))
    if int(json_warn_bytes) > 0 and payload_size_bytes >= int(json_warn_bytes):
        log.warning(
            "sw-pipe state file is large: %d bytes (threshold=%d bytes) at %s",
            payload_size_bytes,
            int(json_warn_bytes),
            path,
        )
    path.write_text(payload_text)


def discover_plt_files(directory: str | Path = ".", *, recursive: bool = False) -> list[Path]:
    """
    Discover `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    base = Path(directory)
    paths = base.rglob("*") if recursive else base.iterdir()
    files = [path for path in paths if path.is_file() and path.suffix.lower() == ".plt"]
    return sorted(files)


def _resolve_pipeline_process_file(pipeline: str) -> Callable[[Path], None]:
    """
    Resolve a named built-in pipeline to its per-file process function.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    key = str(pipeline).strip().lower()
    if key == "dummy":
        from starwinds_analysis.pipelines.dummy_pipeline import process_plt_file

        return process_plt_file
    if key == "quicklook2d":
        from starwinds_analysis.pipelines.quicklook2d import process_plt_file

        return process_plt_file
    raise KeyError(f"Unknown pipeline '{pipeline}'")


def run_sw_pipe(
    directory: str | Path = ".",
    *,
    pipeline: str = "dummy",
    recursive: bool = False,
    noclobber: bool = False,
    include_file_hash: bool = False,
    array_offload_min_bytes: int = _DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
    json_warn_bytes: int = _DEFAULT_JSON_WARN_BYTES,
    process_file: Callable[[Path], None] | None = None,
) -> SwPipeResults:
    """
    Run `sw-pipe` over all discovered `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    if process_file is None:
        process_file = _resolve_pipeline_process_file(pipeline)
        process_label = str(pipeline)
    else:
        process_label = f"{process_file.__module__}.{process_file.__name__}"

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
    recorder_logger = logging.getLogger("recorder")
    recorder_logger.setLevel(logging.DEBUG)
    artifacts_root = Path(directory) / "sw-pipe.artifacts"
    for file_path in files:
        file_key = _relative_file_key(file_path, base_dir=directory)
        if noclobber and file_key in known_processed:
            results.skipped_files.append(file_path)
            log.debug("sw-pipe skipping already processed file %s", file_path.name)
            continue
        file_results: dict[str, object] = {
            "meta": {
                "input_file": str(file_path.resolve()),
                "pipeline": process_label,
                "start_time_utc": _utc_now_iso(),
            }
        }
        if include_file_hash:
            file_results["meta"]["file_hash_sha256"] = _sha256_file(file_path)
        recorder_handler = _SwRecordHandler(
            file_results,
            file_key=file_key,
            artifacts_root=artifacts_root,
            array_offload_min_bytes=array_offload_min_bytes,
        )
        recorder_logger.addHandler(recorder_handler)
        try:
            process_file(file_path)
        finally:
            meta = file_results.get("meta")
            if isinstance(meta, dict):
                meta["end_time_utc"] = _utc_now_iso()
            recorder_logger.removeHandler(recorder_handler)
            recorder_handler.close()
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
        json_warn_bytes=int(json_warn_bytes),
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
        "--pipeline",
        default="dummy",
        choices=("dummy", "quicklook2d"),
        help="Built-in per-file pipeline to run (default: dummy).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--record-log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Recorder logger level for stdout/stderr stream output (default: WARNING).",
    )
    parser.add_argument(
        "--noclobber",
        action="store_true",
        help="Skip files already listed in the sw-pipe state file.",
    )
    parser.add_argument(
        "--file-hash",
        action="store_true",
        help="Include SHA-256 hash of each input file in per-file metadata.",
    )
    parser.add_argument(
        "--array-offload-min-bytes",
        type=int,
        default=_DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
        help="Offload recorded NumPy arrays at or above this byte size to .npy artifacts.",
    )
    parser.add_argument(
        "--json-warn-bytes",
        type=int,
        default=_DEFAULT_JSON_WARN_BYTES,
        help="Warn if sw-pipe.processed.json is at or above this byte size (0 disables).",
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
    configure_recorder_logger(str(args.record_log_level))
    run_sw_pipe(
        args.directory,
        pipeline=str(args.pipeline),
        recursive=bool(args.recursive),
        noclobber=bool(args.noclobber),
        include_file_hash=bool(args.file_hash),
        array_offload_min_bytes=int(args.array_offload_min_bytes),
        json_warn_bytes=int(args.json_warn_bytes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
