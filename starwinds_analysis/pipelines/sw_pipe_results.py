"""CLI parser for `sw-pipe.processed.json` results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_state(path: str | Path) -> dict[str, object]:
    """
    Load a sw-pipe state JSON payload from disk.
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    return json.loads(Path(path).read_text())


def _computed_results(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """
    Return the computed-results mapping from a state payload.
    Used by: `starwinds_analysis/pipelines/sw_pipe_results.py`
    """
    computed = payload.get("computed_results")
    if isinstance(computed, dict):
        return {str(key): value for key, value in computed.items() if isinstance(value, dict)}
    return {}


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
    parser = argparse.ArgumentParser(description="Inspect sw-pipe.processed.json results.")
    parser.add_argument(
        "--state",
        default="sw-pipe.processed.json",
        help="Path to sw-pipe state JSON (default: ./sw-pipe.processed.json).",
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

    payload = _load_state(args.state)
    computed = _computed_results(payload)
    available_fields = _iter_fields(computed)

    if args.list_fields:
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
            return 0

        selected: dict[str, object] = {}
        for file_key, entry in computed.items():
            if resolved in entry:
                selected[file_key] = _extract_field_value(entry[resolved], with_source=bool(args.with_source))
        print(json.dumps(selected, indent=2))
        return 0

    if args.file_key:
        file_entry = computed.get(str(args.file_key))
        if file_entry is None:
            parser.error("file key '%s' not found" % args.file_key)
            return 2
        print(json.dumps(file_entry, indent=2))
        return 0

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
