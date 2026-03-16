# Shim-Sized Function Audit

Last reviewed: 2026-03-07 (`dev`)

Generated metadata:

- Generation method: repo-local AST scan of function/method statement counts.
- Scope: `batwind/**/*.py`.
- Heuristic threshold: `stmt_count <= 4` (review list only, not auto-delete).

Short-function scan for `batwind/` to find potential shim/wrapper code.

## Heuristic

- Candidate threshold: `stmt_count <= 4`
- Scope: top-level functions and methods in `batwind/`
- This is a review list, not an auto-delete list.

Current scan result:

- `53` short functions/methods matched the heuristic.
- Most are API surface methods, tiny math/geometry transforms, or stable path helpers.

## Keep (Intentional Short Functions)

These are compact by design and should stay short:

- `SmartDs` API passthrough/property methods in `batwind/smart_ds.py`
- coordinate/geometry transform helpers in `batwind/algorithms/spherical.py`
- grid-edge/center accessors in `batwind/algorithms/sphere_sampling.py`
- recorder path/key helpers in `batwind/pipelines/recorder.py`
- small parser accessors in `batwind/param_in.py`

## Review Candidates (Current)

These are short and currently worth explicit review for placement/value.

1. `batwind/physics/curve.py`
- `mass_loss_from_curve`
- `torque_from_curve`
- Both are small, quantity-specific wrappers. Keep only if they remain useful call boundaries.

2. `batwind/physics/torque.py`
- `spherical_wind_torque_density_terms`
- `radial_surface_normals`
- Compact and valid, but should stay only if reused as independent primitives.

3. `batwind/pipelines/utils.py`
- `slug_key`
- `output_prefix_from_input_file`
- Keep if shared by multiple pipelines (currently true). Revisit if they become single-use.

4. `batwind/data/samples.py`
- `data_dir`
- `data_file`
- Intentionally small. Keep as stable sample-data lookup boundary.

## Removed/Stale Entries Purged

This audit no longer tracks removed symbols/modules from earlier snapshots.

## Next Cleanup Pass

If we continue trimming short wrappers, do it in this order:

1. review short quantity-specific wrappers in `physics/`
2. keep `SmartDs` API passthroughs intact unless API surface is intentionally reduced
3. avoid replacing one tiny helper with another tiny helper in a different file
