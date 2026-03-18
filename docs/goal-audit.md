# Goal Audit

Date: 2026-03-07  
Branch: `dev`

## Current Status Snapshot

### 1) Pipeline layout

Status: DONE (current shape)

- `batwind-pipe` routes built-in pipelines by filename prefix:
  - `3d* -> volume`
  - `shl* -> shell`
  - `x=0*`, `y=0*`, `z=0* -> slice`
- Built-in per-file pipelines are:
  - `slice`
  - `shell`
  - `volume`
  - `dummy`

Files:

- `batwind/pipelines/batwind_pipe.py`
- `batwind/pipelines/slice.py`
- `batwind/pipelines/shell.py`
- `batwind/pipelines/volume.py`

### 2) Recorder-backed pipeline output

Status: DONE (current model)

- Pipelines emit result records through `add_record(...)` at compute time.
- `batwind-pipe` captures records and writes per-pipeline state files:
  - `batwind-pipe.<pipeline>.processed.json`
- Recorded fields are machine-ingestable and traceable with:
  - `value`
  - `source.module`
  - `source.function`
  - `source.line`

Files:

- `batwind/pipelines/recorder.py`
- `batwind/pipelines/batwind_pipe.py`
- `batwind/pipelines/batwind_pipe_results.py`

### 3) SmartDs and recipe surface

Status: PARTIAL, usable

- BATSRUS + spherical graph fragments are attached explicitly through:
  - `SmartDs.add_batsrus_graph(...)`
  - `SmartDs.add_spherical_graph(...)`
- Active spherical names are:
  - `R [R]`
  - `polar [rad]`
  - `azimuth [rad]`
  - vector suffixes `_r`, `_p`, `_a`

Files:

- `batwind/smart_ds.py`
- `batwind/recipes/spherical.py`
- `batwind/recipes/batsrus.py`

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

- `batwind/param_in.py`
- `batwind/smart_ds.py`
- `batwind/recipes/batsrus.py`

## Validation Baseline

Focused checks that should stay green:

- `test/test_batwind_pipe.py`
- `test/test_batwind_pipe_results.py`
- `test/test_param_in.py`
- `test/test_smart_ds.py`
- `test/test_alfven_radius.py`

Last verified: 2026-03-07  
Environment: `batwind` conda env

```bash
conda run -n batwind python -m pytest -q \
  test/test_batwind_pipe.py \
  test/test_batwind_pipe_results.py \
  test/test_param_in.py \
  test/test_smart_ds.py \
  test/test_alfven_radius.py
```

Result: `50 passed in 0.96s`

## Boundary Note

- This file is snapshot-only (state + verification).
- Execution sequencing and next implementation steps live in:
  - `docs/technical-debt-remediation-plan.md`
