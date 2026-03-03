# Goal Audit

Date: 2026-02-27
Branch: `dev`

## Goals Checked

1. `quicklook2d` can run as a real `sw-pipe` pipeline.
2. Recorder API is used via `add_record` where pipeline payloads are emitted.
3. Dict payloads that were created but not consumed are removed, and creation-time logging/recording is used instead.
4. Pipeline output remains traceable (module/function/line) and machine-ingestable.

## Status

### 1) `quicklook2d` pipeline wiring

Status: DONE

- `sw-pipe` supports `--pipeline quicklook2d`.
- Resolver routes to `starwinds_analysis.pipelines.quicklook2d.process_plt_file`.
- Per-file metadata records selected pipeline.

Files:
- `starwinds_analysis/pipelines/sw_pipe.py`
- `starwinds_analysis/pipelines/quicklook2d.py`

### 2) `add_record` usage

Status: DONE

- `quicklook2d` per-file step emits structured fields via `add_record`.
- Emitted fields include output dir, summary file path, and output counts.

Files:
- `starwinds_analysis/pipelines/quicklook2d.py`

### 3) Unused dict-return cleanup

Status: DONE for confirmed unused cases found in current scan

- `register_spherical_geometry_fields(...)` no longer returns an unused dict.
- `auto_register_vector_spherical_components(...)` no longer returns an unused dict.
- Both now log creation summaries at the point of creation.

Files:
- `starwinds_analysis/recipes/spherical.py`

### 4) Traceability and ingest format

Status: DONE

- `sw-pipe.<pipeline>.processed.json` entries keep `value` and `source` (`module`, `function`, `line`).
- Quicklook pipeline records were confirmed present in state JSON with valid source metadata.

Files:
- `starwinds_analysis/pipelines/sw_pipe.py`

## Validation Run

- `test/test_sw_pipe.py` (pipeline selection + recorder wiring)
- `test/test_quicklook2d.py` (quicklook per-file process behavior)
- `test/test_smart_ds.py`, `test/test_shell_analysis.py`, `test/test_shell_resample_smartds_spec.py`
- `test/test_sw_pipe_results.py`
- `test/test_code_rules.py`

All passed in this audit pass.

## Remaining Work (Not Claimed Complete Here)

- Broader technical debt items in `docs/technical-debt.md` and `docs/technical-debt-remediation-plan.md` remain open (especially quantity-specific wrappers and deeper SmartDs/griblet migration).
