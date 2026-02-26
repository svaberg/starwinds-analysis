# Technical Debt Remediation Plan

This plan is derived from `/Users/dagfev/Documents/starwinds/starwinds-analysis/docs/technical-debt.md` and the project rules in `/Users/dagfev/Documents/starwinds/starwinds-analysis/docs/bad-practices.md`.

## Principles (Hard Constraints)

- Do not reduce notebook clutter by moving one-off code into the library.
- Deep layers must have smaller API surfaces and higher quality.
- Same algorithm + different physical quantity is not a new function/module.
- `SmartDs` + griblet should provide SI quantities; new `resolve_*` usage is not acceptable.
- `analysis` should not import from `physics` (target direction), and circular imports must be removed by changing boundaries, not by lazy imports alone.

## Phase 0 (Completed Low-Hanging Fruit)

Completed in this pass:

- Moved named planetary orbit presets/helpers out of `analysis` and into `physics`.
- Added deep Kepler primitives in `physics.orbits` (`orbital_period`, `orbital_velocity`) and rewired callers.
- Centralized `MU0` in `physics.constants` and removed local redefinitions.
- Removed `open_wind_magnetisation_from_profiles` from `physics` (profile-bundle helper moved to local quicklook logic).

## Phase 1 (Low-Hanging, Next)

Goal: remove clear layer violations and obvious API-surface bloat without changing behavior.

1. DONE: Shrink `physics.__init__` and `analysis.__init__` re-export surfaces.
- Both package `__init__.py` files are now minimal (`__all__ = []`).
- Callers import from owning modules directly.

2. Continue removing stale debt markers by fixing or deleting the underlying patterns.
- DONE: `analysis/__init__.py` (`analysis` re-exporting `physics`) removed.
- PARTIAL: `analysis/fluxes.py` moved to `physics/fluxes.py`; quantity-specific wrapper debt remains and top-level TODOs still apply there.
- DONE (intermediate): `analysis/surface_torque.py` removed; wrappers currently live in `physics/surface_torque.py`.
- NEXT: split `physics/surface_torque.py` back down so only local torque terms remain in the deep layer.

3. Move any remaining deep primitives out of mixed modules when the split is file-clean.
- DONE (partial): `analysis/orbits.py` now contains geometry/sampling primitives only; local mass-loss/torque orbit workflows were moved out.
- NEXT: reduce workflow debt in `physics/orbit_local.py` (still quantity-specific and `resolve_*`-based).

## Phase 2 (SmartDs / griblet Migration)

Goal: stop computing physical quantities outside SmartDs/griblet.

1. Add/extend griblet recipes for common SI quantities currently recomputed in code.
- `B_r [T]`, `U_r [m/s]`
- `magnetic_pressure [Pa]`, `ram_pressure [Pa]`
- `mass_flux [kg/m^2/s]`, `energy_flux [W/m^2]`
- torque-density terms where feasible (with explicit geometry inputs)

2. Replace `resolve_*` usage in callers with direct SmartDs requests.
- Start with `analysis/fluxes.py`
- Then `physics/mass_loss.py`, `physics/shell_torque.py`
- Then orbit workflows (`physics/orbit_pressure.py`, `physics/orbit_surface.py`)

3. Rename/clarify internal SmartDs graph resolve naming if needed.
- Keep graph-path resolution distinct from user quantity requests.

## Phase 3 (Shell / Surface Primitive Consolidation)

Goal: one primitive pipeline + parameterized quantities, not `*_vs_radius` duplication.

1. Define generic shell reduction primitives (sampling + reduction) in `analysis/shells.py`.
- Inputs: sampled shell arrays and reducer/integrand callbacks or quantity arrays.
- Outputs: generic profile arrays/metadata.

2. Move quantity definitions into `physics` (pointwise/local formulas only).
- `mass_loss`, `energy flux`, `torque`, `open flux` should reuse the same shell reduction primitive.

3. Eliminate quantity-specific shell profile wrappers where they only differ by integrand.
- `mass_loss_vs_radius`, `torque_vs_radius`, `open_magnetic_flux_vs_radius`, `energy_flux_vs_radius`
- Keep only truly distinct algorithms (for example axisymmetric reductions requiring grid structure).

4. Apply the same split to explicit-surface torque.
- Keep local `T1..T4` physics in `physics.surface_torque`
- Keep generic explicit-surface integration/reduction in `analysis`
- Remove quantity-specific `*_vs_radius` wrappers as separate APIs when possible

## Phase 4 (Custom Containers and Structured SmartDs)

Goal: stop inventing per-function containers and use shared abstractions.

1. Replace `SphericalShellSamples` compatibility usage gradually with structured `SmartDs` shell resampling.
- Preserve explicit metadata fields (`R`, `theta`, `phi`, `area`) on resampled SmartDs.
- Keep temporary compatibility bridge only while migrating callers.

2. Remove `ShellMassFluxMap` (custom workflow container).
- Use structured resampled `SmartDs` + explicit arrays/metadata instead.

3. Add geometry-measure support (`area`, `volume`) only where geometry basis is explicit.
- Structured grids/shell grids first.
- Native mesh (`corners`) later with documented assumptions.

## Phase 5 (Plotting and Quicklook API Reduction)

Goal: stop hardening quantity-specific plotting/orchestration wrappers into library APIs.

1. Reduce `physics.plotting` to genuinely reusable primitives only.
- Remove quantity-specific `plot_*` wrappers as notebooks/quicklook migrate to direct Matplotlib.

2. Shrink `quicklook2d.py` or split by responsibility (without adding wrappers for notebooks).
- Keep it only if it remains a real reusable orchestration layer.
- Otherwise move one-off composition into examples/scripts.

3. Keep `visualisation/` for real plotting primitives only.
- No quantity-specific APIs unless algorithmically distinct and reused.

## Phase 6 (Import Direction Cleanup)

Goal: enforce one-way layer direction and eliminate circular import pressure.

1. Remove `analysis -> physics` imports where the split is wrong.
- Move shared primitives deeper (for example `physics.constants`, `physics.orbits`) or sideways into a true primitive module.

2. Remove `physics -> analysis` imports in workflow-heavy modules by either:
- moving workflows out of `physics`, or
- extracting the needed primitive out of `analysis`.

3. Only after boundaries are correct, remove lazy-import cycle workarounds.

## Execution Order (Practical)

Recommended next implementation batches:

1. `physics/fluxes.py` + `physics/mass_loss.py` + `physics/shell_torque.py`
- DONE (partial): `resolve_*` field-resolution helpers removed from these files; SI fields are now requested through SmartDs/griblet.
- NEXT: Introduce shared shell reduction primitive usage and remove local `B_r`/`U_r`/flux-density recomputation where possible.

2. `physics/orbit_pressure.py` + `physics/orbit_surface.py`
- Separate orbit geometry/sampling from quantity assembly
- Remove remaining `resolve_*` calls in favor of SmartDs requests

3. `physics.plotting.py` + `quicklook2d.py`
- Delete quantity-specific plotting wrappers as notebooks/examples use direct Matplotlib
- Keep only genuinely reused plotting primitives (if any)

## Testing Strategy (Required For Each Batch)

- Run targeted tests for touched areas first.
- Then run the full suite in `starwinds-analysis` before finishing a batch:

```bash
conda run -n starwinds-analysis python -m pytest -q -ra
```

- If a change is comment-only, run `py_compile` on touched files instead.
