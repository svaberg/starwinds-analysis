# Goal Audit

Date: 2026-03-07  
Branch: `dev`

## Current Status Snapshot

### 1) Pipeline layout

Status: DONE (current shape)

- `sw-pipe` routes built-in pipelines by filename prefix:
  - `3d* -> volume`
  - `shl* -> shell`
  - `x=0*`, `y=0*`, `z=0* -> slice`
- Built-in per-file pipelines are:
  - `slice`
  - `shell`
  - `volume`
  - `dummy`

Files:

- `starwinds_analysis/pipelines/sw_pipe.py`
- `starwinds_analysis/pipelines/slice.py`
- `starwinds_analysis/pipelines/shell.py`
- `starwinds_analysis/pipelines/volume.py`

### 2) Recorder-backed pipeline output

Status: DONE (current model)

- Pipelines emit result records through `add_record(...)` at compute time.
- `sw-pipe` captures records and writes per-pipeline state files:
  - `sw-pipe.<pipeline>.processed.json`
- Recorded fields are machine-ingestable and traceable with:
  - `value`
  - `source.module`
  - `source.function`
  - `source.line`

Files:

- `starwinds_analysis/pipelines/recorder.py`
- `starwinds_analysis/pipelines/sw_pipe.py`
- `starwinds_analysis/pipelines/sw_pipe_results.py`

### 3) SmartDs and recipe surface

Status: PARTIAL, usable

- `SmartDs.prepare(...)` is the normal setup path.
- BATSRUS + spherical graph fragments are attached by default in `prepare(...)`.
- Active spherical names are:
  - `R [R]`
  - `polar [rad]`
  - `azimuth [rad]`
  - vector suffixes `_r`, `_p`, `_a`
- Graph introspection is available through `SmartDs.explain(...)`.

Files:

- `starwinds_analysis/smart_ds.py`
- `starwinds_analysis/recipes/spherical.py`
- `starwinds_analysis/recipes/batsrus.py`

### 4) Nearby config and stellar parameters

Status: DONE (first pass)

- `PARAM.in` parsing is available via `ParamIn`.
- `SmartDs.from_file(...)` searches nearby `PARAM.in` / `param.in`.
- Parsed stellar values are exposed in graph-accessible SI fields:
  - `star_radius [m]`
  - `star_mass [kg]`
  - `star_rotational_period [s]`
  - `star_rotation_rate [rad/s]`

Files:

- `starwinds_analysis/param_in.py`
- `starwinds_analysis/smart_ds.py`
- `starwinds_analysis/recipes/batsrus.py`

## Validation Baseline

Focused checks that should stay green:

- `test/test_sw_pipe.py`
- `test/test_sw_pipe_results.py`
- `test/test_param_in.py`
- `test/test_smart_ds.py`
- `test/test_alfven_radius.py`

Last verified: 2026-03-07  
Environment: `starwinds-analysis` conda env

```bash
conda run -n starwinds-analysis python -m pytest -q \
  test/test_sw_pipe.py \
  test/test_sw_pipe_results.py \
  test/test_param_in.py \
  test/test_smart_ds.py \
  test/test_alfven_radius.py
```

Result: `50 passed in 0.96s`

## Boundary Note

- This file is snapshot-only (state + verification).
- Execution sequencing and next implementation steps live in:
  - `docs/technical-debt-remediation-plan.md`
