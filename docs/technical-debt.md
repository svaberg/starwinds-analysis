# Technical Debt Ledger (Current)

Last reviewed: 2026-03-07 (`dev`)

## Document Role (Source of Truth)

- This file is the canonical debt ledger.
- It lists what is wrong now and where.
- It does not define implementation order.
- Ordering and execution live in `docs/technical-debt-remediation-plan.md`.
- Project rules live in `docs/bad-practices.md`.

## Open Debt Items

### TD-01 (P0) Pipeline complexity concentration

Area:

- `starwinds_analysis/pipelines/shell.py`
- `starwinds_analysis/pipelines/sw_pipe.py`
- `starwinds_analysis/pipelines/recorder.py`

Debt:

- pipeline logic surface is still too large in single modules
- responsibilities remain mixed (routing, execution policy, persistence, reporting)

### TD-02 (P0) Workflow-heavy physics modules

Area:

- `starwinds_analysis/physics/orbit_surface.py`
- `starwinds_analysis/physics/curve.py`
- `starwinds_analysis/physics/torque.py`

Debt:

- modules still mix primitives with workflow assembly
- several functions still return broad dict payloads

### TD-03 (P1) Recipe and graph hygiene

Area:

- `starwinds_analysis/recipes/batsrus.py`
- `starwinds_analysis/smart_ds.py`

Debt:

- avoidable internal complexity remains in graph registration/access paths
- strict fail-fast and fragment-based graph composition still needs cleanup passes

### TD-04 (P1) Linear resampling performance

Area:

- `starwinds_analysis/_smart_ds_resample.py`

Debt:

- linear interpolation remains slow for larger 3D -> curve/surface workflows

### TD-05 (P1) Analysis/physics API hygiene

Area:

- `starwinds_analysis/analysis/`
- `starwinds_analysis/physics/`

Debt:

- still carrying some low-value wrappers and thin APIs
- some functions still have no non-test/non-example callers

### TD-06 (P2) Documentation drift risk

Area:

- `docs/`

Debt:

- docs can drift quickly after architecture moves unless refreshed as part of each batch

## Debt Baseline Rule

When code changes architecture or API boundaries, update these in the same batch:

1. `docs/function-audit-notes.md`
2. `docs/shim-sized-function-audit.md`
3. `docs/technical-debt.md`
4. `docs/technical-debt-remediation-plan.md`
5. `docs/goal-audit.md`
