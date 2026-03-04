# Goal Audit

Date: 2026-03-04
Branch: `dev`

## Current Status Snapshot

### 1) Pipeline layout

Status: DONE (current shape)

- `sw-pipe` routes built-in pipelines by filename prefix:
  - `3d* -> volume`
  - `shl* -> shell`
  - `x=0*`, `y=0*`, `z=0* -> slice`
- The old `quicklook2d` pipeline layer is gone.
- Current built-in per-file pipelines are:
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

- Pipelines emit results through `add_record(...)` as values are produced.
- `sw-pipe` captures those records and writes:
  - `sw-pipe.<pipeline>.processed.json`
- Recorded values remain machine-ingestable and traceable via:
  - `value`
  - `source.module`
  - `source.function`
  - `source.line`

Files:
- `starwinds_analysis/pipelines/recorder.py`
- `starwinds_analysis/pipelines/sw_pipe.py`

### 3) SmartDs and recipe surface

Status: PARTIAL, usable

- `SmartDs.prepare(...)` is the normal workflow setup method.
- Local spherical fields are explicit (`add_spherical_fields(...)`), not auto-attached in `__init__()`.
- Active spherical field names now use:
  - `R [R]`
  - `polar [rad]`
  - `azimuth [rad]`
  - `_r`, `_p`, `_a`
- `theta` / `phi` and `*_theta` / `*_phi` aliases are removed from the active path.

Remaining debt:
- `SmartDs.resolve(...)` still exists and still violates the project rule against `resolve*` API names.

Files:
- `starwinds_analysis/smart_ds.py`
- `starwinds_analysis/recipes/spherical.py`
- `starwinds_analysis/recipes/batsrus.py`

### 4) Nearby config and stellar parameters

Status: DONE (first pass)

- `PARAM.in` parsing is available through `ParamIn`.
- `SmartDs.from_file(...)` looks for nearby `PARAM.in` / `param.in`.
- Parsed stellar parameters are exposed through graph-backed scalar fields:
  - `star_radius [m]`
  - `star_mass [kg]`
  - `star_rotational_period [s]`
  - `star_rotation_rate [rad/s]`

Files:
- `starwinds_analysis/param_in.py`
- `starwinds_analysis/smart_ds.py`
- `starwinds_analysis/recipes/batsrus.py`

## Validation Baseline

Current focused checks that should stay healthy:

- `test/test_sw_pipe.py`
- `test/test_sw_pipe_results.py`
- `test/test_param_in.py`
- `test/test_smart_ds.py`

## Main Remaining Work

- Continue shrinking real debt in `docs/technical-debt.md`, especially:
  - `SmartDs.resolve(...)`
  - quantity-specific `*_vs_radius` wrappers in `physics/`
  - keeping `shell.py` from growing into another logic blob
