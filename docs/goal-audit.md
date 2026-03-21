# Goal Audit

Date: 2026-03-21  
Branch: `main`

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

- BATSRUS + spherical graph fragments are commonly attached through:
  - `SmartDs.from_file(..., batsrus=True, spherical=True, body_radius_m=...)`
  - or explicitly on `sds.computation_graph`
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

Status: PARTIAL

- `PARAM.in` parsing is available via `ParamIn`.
- `SmartDs.from_file(...)` searches nearby `PARAM.in` / `param.in`.
- Nearby stellar aux is merged into the raw dataset aux.
- `Star_radius_m` can seed `body_radius_m`, which then exposes:
  - `RBODY [m]`
- Direct graph exposure of the broader stellar `Star_*` values is not yet the final cleaned-up shape.

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

Last verified: 2026-03-21  
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
Result: `85 passed, 2 warnings`

## Boundary Note

- This file is snapshot-only (state + verification).
- Execution sequencing and next implementation steps live in:
  - `docs/technical-debt-remediation-plan.md`
