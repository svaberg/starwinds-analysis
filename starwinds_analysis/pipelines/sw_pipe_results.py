"""CLI parser for per-pipeline `sw-pipe.<pipeline>.processed.json` results."""

from __future__ import annotations

import argparse
import json
import logging

from starwinds_analysis.pipelines.recorder import load_state_payload

log = logging.getLogger(__name__)


def _computed_results(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """
    Return the computed-results mapping from a state payload.
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    computed = payload.get("computed_results")
    if isinstance(computed, dict):
        return {str(key): value for key, value in computed.items() if isinstance(value, dict)}
    log.warning("_computed_results: missing/invalid computed_results mapping")
    return {}


def _iter_file_keys(payload: dict[str, object], computed: dict[str, dict[str, object]]) -> list[str]:
    """
    List file keys in recorded order, using `processed_files` first and `computed_results` as fallback.
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    ordered: list[str] = []
    seen: set[str] = set()
    processed = payload.get("processed_files")
    if isinstance(processed, list):
        for item in processed:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    for key in computed:
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _iter_fields(computed: dict[str, dict[str, object]]) -> list[str]:
    """
    List field names in first-seen order (excluding per-file `meta`).
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in computed.values():
        for field in entry:
            if field == "meta" or field in seen:
                continue
            seen.add(field)
            ordered.append(field)
    return ordered


def _resolve_field_name(requested: str, available: list[str]) -> str:
    """
    Resolve a requested field name, supporting singular/plural convenience.
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    if requested in available:
        return requested
    if f"{requested}s" in available:
        return f"{requested}s"
    if requested.endswith("s") and requested[:-1] in available:
        return requested[:-1]
    log.error("_resolve_field_name failed for '%s'", requested)
    raise KeyError(requested)


def _extract_field_value(field_payload: object, *, with_source: bool) -> object:
    """
    Extract field payload as value-only or full payload (`value`+`source`).
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    if with_source:
        return field_payload
    if isinstance(field_payload, dict) and "value" in field_payload:
        return field_payload["value"]
    return field_payload


def build_parser() -> argparse.ArgumentParser:
    """
    Build the `sw-pipe-results` CLI parser.
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    parser = argparse.ArgumentParser(description="Inspect sw-pipe.<pipeline>.processed.json results.")
    parser.add_argument(
        "--state",
        default="sw-pipe.dummy.processed.json",
        help="Path to sw-pipe state JSON (default: ./sw-pipe.dummy.processed.json).",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List processed file keys.",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List available field names.",
    )
    parser.add_argument(
        "--field",
        help="Field name to query (supports singular/plural convenience).",
    )
    parser.add_argument(
        "--file",
        dest="file_key",
        help="Restrict output to one processed file key.",
    )
    parser.add_argument(
        "--with-source",
        action="store_true",
        help="Include full field payload with source metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for `sw-pipe-results`.
    Used by: CLI (`sw-pipe-results`), `test/test_sw_pipe_results.py`
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    log.info("sw-pipe-results start state=%s", args.state)

    payload = load_state_payload(args.state)
    computed = _computed_results(payload)
    available_files = _iter_file_keys(payload, computed)
    available_fields = _iter_fields(computed)

    if args.list_files:
        log.info("sw-pipe-results listing files")
        for file_key in available_files:
            print(file_key)
        return 0

    if args.list_fields:
        log.info("sw-pipe-results listing fields")
        for field in available_fields:
            print(field)
        return 0

    if args.field:
        try:
            resolved = _resolve_field_name(str(args.field), available_fields)
        except KeyError:
            parser.error("field '%s' not found" % args.field)
            return 2

        if args.file_key:
            file_entry = computed.get(str(args.file_key))
            if file_entry is None:
                parser.error("file key '%s' not found" % args.file_key)
                return 2
            if resolved not in file_entry:
                parser.error("field '%s' not found in file '%s'" % (resolved, args.file_key))
                return 2
            print(json.dumps(_extract_field_value(file_entry[resolved], with_source=bool(args.with_source)), indent=2))
            log.info("sw-pipe-results field query done file=%s field=%s", args.file_key, resolved)
            return 0

        selected: dict[str, object] = {}
        for file_key, entry in computed.items():
            if resolved in entry:
                selected[file_key] = _extract_field_value(entry[resolved], with_source=bool(args.with_source))
        print(json.dumps(selected, indent=2))
        log.info("sw-pipe-results field query done field=%s matches=%d", resolved, len(selected))
        return 0

    if args.file_key:
        file_entry = computed.get(str(args.file_key))
        if file_entry is None:
            parser.error("file key '%s' not found" % args.file_key)
            return 2
        print(json.dumps(file_entry, indent=2))
        log.info("sw-pipe-results dumped file=%s", args.file_key)
        return 0

    print(json.dumps(payload, indent=2))
    log.info("sw-pipe-results dumped full payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
