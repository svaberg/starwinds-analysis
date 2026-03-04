"""THIS FILE contains the generic `sw-pipe` orchestration CLI.

It discovers `.plt` files in a working directory and runs a per-file pipeline
handler. Built-in handlers are `dummy`, `slice`, `shell`, and `volume`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Callable

from starwinds_analysis.pipelines.orchestration_helpers import log_pipeline_event
from starwinds_analysis.pipelines.recorder import DEFAULT_ARRAY_OFFLOAD_MIN_BYTES
from starwinds_analysis.pipelines.recorder import DEFAULT_JSON_WARN_BYTES
from starwinds_analysis.pipelines.recorder import SwPipeResults
from starwinds_analysis.pipelines.recorder import SwRecordHandler
from starwinds_analysis.pipelines.recorder import load_state
from starwinds_analysis.pipelines.recorder import relative_file_key
from starwinds_analysis.pipelines.recorder import save_state
from starwinds_analysis.pipelines.recorder import sha256_file
from starwinds_analysis.pipelines.recorder import state_file_path

log = logging.getLogger(__name__)
PIPELINE_LOG_FORMAT = "[%(levelname)s] %(pipeline_source)s %(message)s"
PIPELINE_COLOR_LOG_FORMAT = "%(log_color)s[%(levelname)s]%(reset)s %(pipeline_source)s %(message)s"


# Human/recorder logging setup
class PipelineSourceFilter(logging.Filter):
    """
    Add the shared `pipeline_source` field used by pipeline log formatters.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Populate the shared pipeline-source field for one log record.
        Used by: `starwinds_analysis/pipelines/sw_pipe.py`
        """
        if record.name.startswith("recorder."):
            record.pipeline_source = f"{record.name}.{record.funcName}:{record.lineno}"
        else:
            record.pipeline_source = record.name.rsplit(".", 1)[-1]
        return True


def configure_logger(level_name: str) -> None:
    """
    Configure the root logger for human-readable pipeline logs on stdout.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(level_name).upper()))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, str(level_name).upper()))
    handler.addFilter(PipelineSourceFilter())
    color_backend = None
    color_fallback_notes: list[tuple[int, str]] = []
    if bool(getattr(sys.stdout, "isatty", lambda: False)()):
        for module_name in ("colorlog", "coloredlogs"):
            try:
                module = __import__(module_name, fromlist=["ColoredFormatter"])
            except ImportError:
                color_fallback_notes.append((logging.INFO, f"Pipeline color logging backend not available: {module_name}"))
                continue
            formatter_class = getattr(module, "ColoredFormatter", None)
            if formatter_class is None:
                color_fallback_notes.append((logging.WARNING, f"Pipeline color logging backend missing ColoredFormatter: {module_name}"))
                continue
            handler.setFormatter(
                formatter_class(
                    PIPELINE_COLOR_LOG_FORMAT,
                    secondary_log_colors={},
                    reset=True,
                    style="%",
                )
            )
            color_backend = module_name
            break
    if handler.formatter is None:
        handler.setFormatter(logging.Formatter(PIPELINE_LOG_FORMAT))
    logger.addHandler(handler)
    if color_backend is not None:
        log.info("Using colored pipeline logging via %s", color_backend)
    elif color_fallback_notes:
        for level, message in color_fallback_notes:
            log.log(level, "%s", message)
        log.info("Using plain pipeline logging (no color backend available)")


def configure_recorder(level_name: str = "WARNING") -> None:
    """
    Configure the dedicated recorder logger with its own stream handler and level.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    recorder = logging.getLogger("recorder")
    recorder.setLevel(logging.DEBUG)
    recorder.handlers.clear()
    recorder_handler = logging.StreamHandler()
    recorder_handler.setLevel(getattr(logging, str(level_name).upper()))
    recorder_handler.addFilter(PipelineSourceFilter())
    recorder_handler.setFormatter(logging.Formatter(PIPELINE_LOG_FORMAT))
    recorder.addHandler(recorder_handler)
    recorder.propagate = False


# File discovery and pipeline routing
def discover_input_files(directory: str | Path = ".", *, recursive: bool = False) -> list[Path]:
    """
    Discover supported input files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    base = Path(directory)
    paths = base.rglob("*") if recursive else base.iterdir()
    files = [path for path in paths if path.is_file() and path.suffix.lower() in {".plt", ".dat"}]
    return sorted(files)


def pipeline_name_for_file(file_path: str | Path) -> str | None:
    """
    Infer the built-in pipeline from the input filename prefix.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    file_name = Path(file_path).name.lower()
    if file_name.startswith("3d"):
        return "volume"
    if file_name.startswith("shl"):
        return "shell"
    if file_name.startswith("x=0") or file_name.startswith("y=0") or file_name.startswith("z=0"):
        return "slice"
    return None


def process_file_for_pipeline(pipeline_name: str) -> Callable[[Path], None]:
    """
    Return the built-in per-file process function for one pipeline name.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    if pipeline_name == "dummy":
        from starwinds_analysis.pipelines.dummy_pipeline import process_plt_file

        return process_plt_file
    if pipeline_name == "slice":
        from starwinds_analysis.pipelines.slice import process_plt_file

        return process_plt_file
    if pipeline_name == "shell":
        from starwinds_analysis.pipelines.shell import process_plt_file

        return process_plt_file
    if pipeline_name == "volume":
        from starwinds_analysis.pipelines.volume import process_plt_file

        return process_plt_file
    raise KeyError(f"Unknown pipeline '{pipeline_name}'")


def select_pipeline_work(
    files: list[Path],
    *,
    directory: str | Path,
    pipeline: str | None,
    process_file: Callable[[Path], None] | None,
) -> tuple[
    dict[str, Path],
    dict[str, set[str]],
    dict[str, dict[str, dict[str, object]]],
    list[tuple[Path, str, Callable[[Path], None], str]],
]:
    """
    Build the per-file work list together with per-pipeline state snapshots.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    state_files: dict[str, Path] = {}
    known_processed_by_pipeline: dict[str, set[str]] = {}
    known_computed_by_pipeline: dict[str, dict[str, dict[str, object]]] = {}
    process_functions: dict[str, Callable[[Path], None]] = {}
    selected: list[tuple[Path, str, Callable[[Path], None], str]] = []

    if process_file is None:
        if pipeline == "dummy":
            process_functions["dummy"] = process_file_for_pipeline("dummy")
            state_files["dummy"] = state_file_path(directory, pipeline_name="dummy")
            known_processed, known_computed = load_state(state_files["dummy"])
            known_processed_by_pipeline["dummy"] = known_processed
            known_computed_by_pipeline["dummy"] = known_computed
            for file_path in files:
                selected.append((file_path, "dummy", process_functions["dummy"], "dummy"))
        elif pipeline is not None:
            pipeline_name = str(pipeline)
            state_files[pipeline_name] = state_file_path(directory, pipeline_name=pipeline_name)
            known_processed, known_computed = load_state(state_files[pipeline_name])
            known_processed_by_pipeline[pipeline_name] = known_processed
            known_computed_by_pipeline[pipeline_name] = known_computed
            process_functions[pipeline_name] = process_file_for_pipeline(pipeline_name)
            for file_path in files:
                if pipeline_name_for_file(file_path) != pipeline_name:
                    continue
                selected.append((file_path, pipeline_name, process_functions[pipeline_name], pipeline_name))
        else:
            for file_path in files:
                resolved_pipeline = pipeline_name_for_file(file_path)
                if resolved_pipeline is None:
                    continue
                if resolved_pipeline not in process_functions:
                    process_functions[resolved_pipeline] = process_file_for_pipeline(resolved_pipeline)
                if resolved_pipeline not in state_files:
                    state_files[resolved_pipeline] = state_file_path(directory, pipeline_name=resolved_pipeline)
                    known_processed, known_computed = load_state(state_files[resolved_pipeline])
                    known_processed_by_pipeline[resolved_pipeline] = known_processed
                    known_computed_by_pipeline[resolved_pipeline] = known_computed
                selected.append((file_path, resolved_pipeline, process_functions[resolved_pipeline], resolved_pipeline))
    else:
        process_label = f"{process_file.__module__}.{process_file.__name__}"
        state_files["custom"] = state_file_path(directory, pipeline_name="custom")
        known_processed, known_computed = load_state(state_files["custom"])
        known_processed_by_pipeline["custom"] = known_processed
        known_computed_by_pipeline["custom"] = known_computed
        for file_path in files:
            selected.append((file_path, process_label, process_file, "custom"))

    return state_files, known_processed_by_pipeline, known_computed_by_pipeline, selected


def run_sw_pipe(
    directory: str | Path = ".",
    *,
    pipeline: str | None = None,
    recursive: bool = False,
    noclobber: bool = False,
    include_file_hash: bool = False,
    array_offload_min_bytes: int = DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
    json_warn_bytes: int = DEFAULT_JSON_WARN_BYTES,
    fail_fast: bool = False,
    process_file: Callable[[Path], None] | None = None,
) -> SwPipeResults:
    """
    Run `sw-pipe` over all discovered `.plt` files in a directory.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_sw_pipe.py`
    """
    files = discover_input_files(directory, recursive=recursive)
    pipeline_label = "auto" if pipeline is None else str(pipeline)
    state_files, known_processed_by_pipeline, known_computed_by_pipeline, selected = select_pipeline_work(
        files,
        directory=directory,
        pipeline=pipeline,
        process_file=process_file,
    )

    results = SwPipeResults(
        directory=Path(directory),
        recursive=recursive,
        noclobber=noclobber,
        discovered_files=[item[0] for item in selected],
        computed_results={},
        state_file=None if len(state_files) != 1 else next(iter(state_files.values())),
    )
    for pipeline_name, pipeline_results in known_computed_by_pipeline.items():
        if pipeline_name == "custom":
            results.computed_results.update(pipeline_results)
            continue
        for file_key, payload in pipeline_results.items():
            results.computed_results[file_key] = payload
    log_pipeline_event(
        log,
        "sw_pipe.discovered",
        count=len(selected),
        directory=Path(directory),
        recursive=recursive,
        noclobber=noclobber,
        pipeline=pipeline_label,
    )
    if not selected:
        for state_pipeline_name, state_file in state_files.items():
            save_state(
                state_file,
                processed_keys=known_processed_by_pipeline[state_pipeline_name],
                computed_results=known_computed_by_pipeline[state_pipeline_name],
                json_warn_bytes=int(json_warn_bytes),
            )
        return results
    recorder = logging.getLogger("recorder")
    recorder.setLevel(logging.DEBUG)
    artifacts_root = Path(directory) / "sw-pipe.artifacts"

    def run_one_file(file_path: Path, process_label: str, active_process_file: Callable[[Path], None], state_pipeline_name: str) -> None:
        """
        Run one pipeline step for one file and update in-memory results/state.
        """
        file_key = relative_file_key(file_path, base_dir=directory)
        processed_keys = known_processed_by_pipeline[state_pipeline_name]
        if noclobber and file_key in processed_keys:
            results.skipped_files.append(file_path)
            log_pipeline_event(log, "sw_pipe.skip_processed", file=file_path.name)
            return

        file_results: dict[str, object] = {
            "meta": {
                "input_file": str(file_path.resolve()),
                "pipeline": process_label,
                "start_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        }
        if include_file_hash:
            file_results["meta"]["file_hash_sha256"] = sha256_file(file_path)

        recorder_handler = SwRecordHandler(
            file_results,
            file_key=file_key,
            artifacts_root=artifacts_root,
            array_offload_min_bytes=array_offload_min_bytes,
        )
        recorder.addHandler(recorder_handler)
        failure: Exception | None = None
        failure_tb = None
        try:
            active_process_file(file_path)
        except Exception as exc:
            failure = exc
            failure_tb = exc.__traceback__
            results.failed_files.append(file_path)
            file_results["meta"]["status"] = "failed"
            file_results["meta"]["error"] = str(exc)
            if not fail_fast:
                log.error("sw-pipe file failed: %s (%s)", file_path.name, exc)
        else:
            file_results["meta"]["status"] = "processed"
            results.processed_files.append(file_path)
            processed_keys.add(file_key)
        finally:
            meta = file_results.get("meta")
            if isinstance(meta, dict):
                meta["end_time_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            recorder.removeHandler(recorder_handler)
            recorder_handler.close()
            results.computed_results[file_key] = file_results
            known_computed_by_pipeline[state_pipeline_name][file_key] = file_results
            save_state(
                state_files[state_pipeline_name],
                processed_keys=processed_keys,
                computed_results=known_computed_by_pipeline[state_pipeline_name],
                json_warn_bytes=int(json_warn_bytes),
            )
        if failure is not None and fail_fast:
            raise failure.with_traceback(failure_tb)

    for file_path, process_label, active_process_file, state_pipeline_name in selected:
        run_one_file(file_path, process_label, active_process_file, state_pipeline_name)
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
        default=None,
        choices=("dummy", "slice", "shell", "volume"),
        help="Built-in per-file pipeline to run (default: auto by filename prefix).",
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
        default=DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
        help="Offload recorded NumPy arrays at or above this byte size to .npy artifacts.",
    )
    parser.add_argument(
        "--json-warn-bytes",
        type=int,
        default=DEFAULT_JSON_WARN_BYTES,
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
    configure_logger(str(args.log_level))

    # Configure the recorder logger that writes machine-readable data.
    configure_recorder(str(args.record_log_level))
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
