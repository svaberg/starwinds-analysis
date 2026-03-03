"""THIS FILE contains the generic `sw-pipe` orchestration CLI.

It discovers `.plt` files in a working directory and runs a per-file pipeline
handler. Built-in handlers are `dummy`, `slice`, `shell`, and `volume`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, field
import hashlib
import json
from json.encoder import INFINITY, _make_iterencode, encode_basestring, encode_basestring_ascii
import logging
import math
import numpy as np
from pathlib import Path
import re
import sys
from typing import Callable

from starwinds_analysis.pipelines.orchestration_helpers import log_pipeline_event

log = logging.getLogger(__name__)
_RECORD_PATTERN = re.compile(r"^([A-Za-z0-9_]+)\b")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_DEFAULT_ARRAY_OFFLOAD_MIN_BYTES = 1_000_000
_DEFAULT_JSON_WARN_BYTES = 10_000_000


class _ScientificFloatEncoder(json.JSONEncoder):
    """
    JSON encoder that writes finite floats in scientific notation.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """

    def iterencode(self, o, _one_shot=False):
        """
        Encode JSON while formatting finite floats as scientific literals.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        markers = {} if self.check_circular else None
        _encoder = encode_basestring_ascii if self.ensure_ascii else encode_basestring

        def floatstr(
            value,
            allow_nan=self.allow_nan,
            _inf=INFINITY,
            _neginf=-INFINITY,
        ):
            if math.isnan(value):
                text = "NaN"
            elif value == _inf:
                text = "Infinity"
            elif value == _neginf:
                text = "-Infinity"
            else:
                return format(value, ".16e")
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: "
                    + repr(value)
                )
            return text

        _iterencode = _make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, 0)


def _dumps_json_scientific(payload: object, *, indent: int = 2) -> str:
    """
    Dump JSON text with scientific float formatting.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return json.dumps(payload, indent=indent, cls=_ScientificFloatEncoder)


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
    failed_files: list[Path] = field(default_factory=list)
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
        Render one pipeline log record with standard level and short source name.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        source = record.name.rsplit(".", 1)[-1]
        message = record.getMessage()
        if record.name.startswith("recorder."):
            origin = f"{record.name}.{record.funcName}:{record.lineno}"
        else:
            origin = source
        out = f"[{record.levelname}] {origin} {message}"
        if record.exc_info:
            out = f"{out}\n{self.formatException(record.exc_info)}"
        return out


def _stdout_pipeline_formatter(stream) -> logging.Formatter:
    """
    Build the stdout formatter for human pipeline logs, using color when available.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    use_color = bool(getattr(stream, "isatty", lambda: False)())
    if use_color:
        for module_name in ("colorlog", "coloredlogs"):
            try:
                module = __import__(module_name, fromlist=["ColoredFormatter"])
                formatter_class = getattr(module, "ColoredFormatter")
                return formatter_class(
                    "%(log_color)s[%(levelname)s]%(reset)s %(name_last)s %(message)s",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "bold_red",
                    },
                    secondary_log_colors={},
                    reset=True,
                    style="%",
                )
            except Exception:
                continue
    return _PipelineLogFormatter()


def _enrich_pipeline_record(record: logging.LogRecord) -> bool:
    """
    Add helper fields used by pipeline log formatters.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    record.name_last = record.name.rsplit(".", 1)[-1]
    return True


def _configure_stdout_logger(level_name: str) -> None:
    """
    Configure the root logger for human-readable pipeline logs on stdout.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, str(level_name).upper()))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, str(level_name).upper()))
    handler.addFilter(_enrich_pipeline_record)
    handler.setFormatter(_stdout_pipeline_formatter(sys.stdout))
    root_logger.addHandler(handler)


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


def _state_file_path(directory: str | Path, *, pipeline_name: str) -> Path:
    """
    Default per-pipeline state-file path for processed `.plt` tracking.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    return Path(directory) / f"sw-pipe.{_safe_name(pipeline_name)}.processed.json"


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
    payload_text = _dumps_json_scientific(payload, indent=2)
    payload_size_bytes = len(payload_text.encode("utf-8"))
    if int(json_warn_bytes) > 0 and payload_size_bytes >= int(json_warn_bytes):
        log.warning(
            "sw-pipe state file is large: %d bytes (threshold=%d bytes) at %s",
            payload_size_bytes,
            int(json_warn_bytes),
            path,
        )
    path.write_text(payload_text)


def discover_input_files(directory: str | Path = ".", *, recursive: bool = False) -> list[Path]:
    """
    Discover supported input files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    base = Path(directory)
    paths = base.rglob("*") if recursive else base.iterdir()
    files = [path for path in paths if path.is_file() and path.suffix.lower() in {".plt", ".dat"}]
    return sorted(files)


def _resolve_pipeline_process_file(
    pipeline: str,
) -> Callable[[Path], None]:
    """
    Resolve a named built-in pipeline to its per-file process function.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    key = str(pipeline).strip().lower()
    if key == "dummy":
        from starwinds_analysis.pipelines.dummy_pipeline import process_plt_file

        return process_plt_file
    if key == "slice":
        from starwinds_analysis.pipelines.slice import process_plt_file

        return process_plt_file
    if key == "shell":
        from starwinds_analysis.pipelines.shell import process_plt_file

        return process_plt_file
    if key == "volume":
        from starwinds_analysis.pipelines.volume import process_plt_file

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
    fail_fast: bool = False,
    process_file: Callable[[Path], None] | None = None,
) -> SwPipeResults:
    """
    Run `sw-pipe` over all discovered `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    if process_file is None:
        process_file = _resolve_pipeline_process_file(pipeline)
        process_label = str(pipeline)
        state_pipeline_name = str(pipeline)
    else:
        process_label = f"{process_file.__module__}.{process_file.__name__}"
        state_pipeline_name = "custom"

    state_file = _state_file_path(directory, pipeline_name=state_pipeline_name)
    known_processed, known_computed = _load_state(state_file)
    files = discover_input_files(directory, recursive=recursive)
    results = SwPipeResults(
        directory=Path(directory),
        recursive=recursive,
        noclobber=noclobber,
        discovered_files=list(files),
        computed_results=dict(known_computed),
        state_file=state_file,
    )
    log_pipeline_event(
        log,
        "sw_pipe.discovered",
        count=len(files),
        directory=Path(directory),
        recursive=recursive,
        noclobber=noclobber,
    )
    processed_keys = set(known_processed)
    recorder_logger = logging.getLogger("recorder")
    recorder_logger.setLevel(logging.DEBUG)
    artifacts_root = Path(directory) / "sw-pipe.artifacts"
    for file_path in files:
        file_key = _relative_file_key(file_path, base_dir=directory)
        if noclobber and file_key in known_processed:
            results.skipped_files.append(file_path)
            log_pipeline_event(log, "sw_pipe.skip_processed", file=file_path.name)
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
        except Exception as exc:
            results.failed_files.append(file_path)
            log.exception("sw-pipe file failed: %s", file_path.name)
            file_results["meta"]["status"] = "failed"
            file_results["meta"]["error"] = str(exc)
            if fail_fast:
                raise
        else:
            file_results["meta"]["status"] = "processed"
            results.processed_files.append(file_path)
            processed_keys.add(file_key)
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
        choices=("dummy", "slice", "shell", "volume"),
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
        help="Warn if the per-pipeline sw-pipe state JSON is at or above this byte size (0 disables).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the run on the first per-file pipeline failure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for `sw-pipe`.
    Used by: CLI (`sw-pipe`), `test/test_sw_pipe.py`
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure the logger that writes to stdout (for human consumption).
    _configure_stdout_logger(str(args.log_level))

    # Configure the recorder logger that writes machine-readable data.
    configure_recorder_logger(str(args.record_log_level))
    run_sw_pipe(
        args.directory,
        pipeline=str(args.pipeline),
        recursive=bool(args.recursive),
        noclobber=bool(args.noclobber),
        include_file_hash=bool(args.file_hash),
        array_offload_min_bytes=int(args.array_offload_min_bytes),
        json_warn_bytes=int(args.json_warn_bytes),
        fail_fast=bool(args.fail_fast),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
