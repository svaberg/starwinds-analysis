# Technical Debt Remediation Plan

Last reviewed: 2026-03-21 (`main`)

## Document Role

- This file is execution order only.
- Debt definitions live in `docs/technical-debt.md`.
- Project rules live in `docs/bad-practices.md`.
- Status snapshots and test baselines live in `docs/goal-audit.md`.

## Ordered Plan

### Step 1: TD-01 (Pipelines)

Targets:

- `batwind/pipelines/shell.py`
- `batwind/pipelines/batwind_pipe.py`
- `batwind/pipelines/recorder.py`

Execution focus:

- reduce mixed responsibilities
- keep pipelines flat and sequential (compute -> plot -> record)
- trim argument/plumbing surface

### Step 2: TD-02 (Physics workflow separation)

Targets:

- `batwind/physics/orbit_surface.py`
- `batwind/physics/curve.py`
- `batwind/physics/torque.py`

Execution focus:

- keep only reusable primitives in physics
- reduce broad dict payloads
- move pointwise quantities to recipes where feasible

### Step 3: TD-03 (Recipe/SmartDs hygiene)

Targets:

- `batwind/recipes/batsrus.py`
- `batwind/smart_ds.py`

Execution focus:

- simplify registration/access paths
- keep strict fail-fast resolution
- keep graph composition explicit and deterministic

### Step 4: TD-05 (API hygiene in analysis/physics)

Targets:

- `batwind/analysis/`
- `batwind/physics/`

Execution focus:

- remove low-value wrappers/shims
- collapse unnecessary tiny APIs
- keep only boundaries with real reuse

### Step 5: TD-04 (Structured/2D linear resampling performance)

Targets:

- `batwind/_smart_ds_resample.py`

Execution focus:

- optimize shared-geometry linear workflows where `linear` is still the right method
- do not treat 3D octree-backed workflows as part of this problem

### Step 6: TD-06 (Docs sync pass)

Targets:

- `docs/function-audit-notes.md`
- `docs/shim-sized-function-audit.md`
- `docs/technical-debt.md`
- `docs/technical-debt-remediation-plan.md`
- `docs/goal-audit.md`

Execution focus:

- refresh docs in same PR/batch as architecture/API changes

### Step 7: TD-07 (`PARAM.in` stellar naming cleanup)

Targets:

- `batwind/param_in.py`
- `batwind/pipelines/volume.py`
- `batwind/pipelines/shell.py`

Execution focus:

- have `ParamIn.stellar_params()` expose what the file actually provides instead of renamed bridge keys
- align pipeline/body-radius consumers to that direct `PARAM.in` output path

## Batch Validation Rule

For each completed step:

1. run targeted tests for touched modules
2. run:

```bash
conda run -n batwind python -m pytest -q test/test_code_rules.py
```

3. run full suite when step scope is complete:

```bash
conda run -n batwind python -m pytest -q -ra
```
