# Technical Debt Review (Python Files)

This file tracks a full file-by-file pass against `/Users/dagfev/Documents/starwinds/starwinds-analysis/docs/bad-practices.md`.

- Files reviewed: **57** (`*.py`, entire repo)
- Scope of this pass: identify debt and mark code with `TODO` comments where bad practices are present
- Note: tests/examples were reviewed too, but production architecture rules are applied primarily to library code

Legend:
- `Debt`: bad-practice hit found in this pass (code TODO added or already present)
- `Reviewed`: no additional debt marker added in this pass

## Architecture Recommendations (Current)

The target is a DAG-shaped library with a small folder tree. The goal is not to
create more packages; the goal is to keep dependency direction clean.

Recommended practical structure (using the folders that already mostly exist):

- `recipes/`: field semantics, unit conversion, coordinate transforms, pointwise derived quantities
- `analysis/`: generic sampling, geometry, interpolation, and numerical reductions
- `physics/`: domain-specific diagnostics (`mass_loss`, `torque`, `pressure`, orbit diagnostics)
- `visualisation/`: plotting primitives only
- `pipelines/`: thin per-file workflows and CLI-facing glue
- top-level modules (`smart_ds.py`, `utils.py`, `constants.py`-style modules): shared access/utilities

Recommended dependency direction:

- `recipes` should not depend on `analysis`, `physics`, `visualisation`, or `pipelines`
- `analysis` should stay generic and should not depend on `physics`
- `smart_ds` can depend on `recipes` and low-level helpers, but should not own domain workflows
- `physics` can depend on `smart_ds`, `analysis`, and `recipes`
- `visualisation` can depend on `smart_ds` and plotting inputs, but should not compute physics
- `pipelines` can depend on `physics`, `visualisation`, and small pipeline-specific helpers only
- `sw_pipe` should stay execution/state/recording glue and should not know science

Current practical fit:

- The existing folders already mostly match this structure.
- No major folder reorg is required right now.
- The main remaining structural risks are:
  - `sw_pipe.py` and `recorder.py` still carrying more orchestration/persistence surface than ideal
  - `shell.py` becoming a second monolith if shell-specific logic keeps accumulating there
  - quantity-specific workflow wrappers living too deep in `physics/`
  - the empty `sampling/` package and similar side folders becoming architectural clutter if they stay around without a clear boundary

Immediate recommendation:

- keep the current folders
- enforce the dependency direction above
- prefer moving or deleting leaky helpers over creating new folders

Pipeline follow-up:

- The 2D `slice` pipeline should add an Alfvén-surface-style plot, matching the
  demonstrated approach in `examples/smartds_2d_xy_points.ipynb`.
- The 2D `slice` pipeline should add a wind-pressure plot in the stellar frame.

Notebook follow-up:

- Example notebook filenames should not be prefixed with `smartds_`.

Coordinate/vector naming note:

- In recipes and SmartDs/griblet-facing field names, use:
  - `R [R]`, `polar [rad]`, `azimuth [rad]`
- For vector components, use compact suffixes:
  - `_r`, `_p`, `_a`
  - for example `U_r [m/s]`, `U_p [m/s]`, `U_a [m/s]`
- Do not reintroduce `theta`/`phi` aliases in active field names; use `polar`/`azimuth`.

## examples

- `examples/earth-xuv-neutrals/earth-xuv-neutrals.py` — **Reviewed**. Example script (one-off workflow code is allowed here).
- `examples/planet.py` — **Reviewed**. Example/legacy script; one-off code is allowed in examples.

## starwinds_analysis

- `starwinds_analysis/_smart_ds_graph.py` — **Debt**. Internal `resolve_field(...)` naming still collides conceptually with forbidden `resolve_*` pattern; keep graph-path resolution distinct from user field/unit requests. Code TODO: existing TODO.
- `starwinds_analysis/_smart_ds_resample.py` — **Reviewed**. Core resampling internals; no new bad-practice hit beyond existing documented NaN handling for interpolation edge cases.
- `starwinds_analysis/algorithms/sphere_sampling.py` — **Reviewed**. Geometry/sampling primitive module (good layer fit).
- `starwinds_analysis/analysis/__init__.py` — **Reviewed**. Re-export facade removed; package boundary is now minimal (`__all__ = []`).
- `starwinds_analysis/analysis/shell_summary.py` — **Reviewed**. Reducer/summary helpers; finite filtering appears intentional for shell-band summaries.
- `starwinds_analysis/analysis/shells.py` — **Debt**. Both shell samplers now return structured `SmartDs`; temporary shell-compat attributes and the `SphericalShellSamples` custom container were removed. Remaining debt in this module is mainly `infer_body_radius_m(...)` permissiveness and shell-primitive API cleanup. Code TODO: existing TODOs.
- `starwinds_analysis/analysis/slices.py` — **Reviewed**. Structured slice resampling/topology helpers; no clear rule violation found in this pass.
- `starwinds_analysis/analysis/stats.py` — **Reviewed**. Generic weighted stats primitives; now also owns the reusable `summarize_samples(...)` helper.
- `starwinds_analysis/data/samples.py` — **Reviewed**. Sample-data path helper; no bad-practice hit found.
- `starwinds_analysis/physics/__init__.py` — **Reviewed**. Deep-layer package boundary is now minimal (`__all__ = []`).
- `starwinds_analysis/physics/fluxes.py` — **Debt**. Quantity-specific shell flux profile wrappers remain (`*_vs_radius`); SI field requests now go through SmartDs/griblet (including spherical components like `B_r`/`U_r` and pointwise `energy_flux [W/m^2]`) and the shell `SmartDs` is read directly (no compat `.fields/.x/.area` usage), but local diagnostics like `B·n` and axisymmetric reductions are still computed in code. Code TODO: existing TODO(debt) + TODO(griblet).
- `starwinds_analysis/physics/constants.py` — **Reviewed**. Shared constants module (good deep-layer home for physical constants like `MU0`).
- `starwinds_analysis/physics/local_estimates.py` — **Debt**. Local physics formulas remain outside SmartDs/griblet (intentional TODOs), but summary/reporting helper was removed and the `analysis.stats` import was eliminated. Code TODO: existing TODO(griblet).
- `starwinds_analysis/physics/magnetic.py` — **Reviewed**. Reduced to magnetic display-unit scaling only; shell magnetic components are now requested via SmartDs/griblet in callers/tests.
- `starwinds_analysis/physics/mass_loss.py` — **Debt**. Quantity-specific shell profile wrapper (`mass_loss_vs_radius`) and `analysis.shells` dependency remain. The custom container (`ShellMassFluxMap`) and one-off shell-map wrapper were removed; tests/examples now use shell primitives directly. SI field requests now go through SmartDs/griblet (including pointwise `mass_flux [kg/m^2/s]`) and shell values are read directly from the shell `SmartDs` (no `resolve_*`, no compat `.fields/.x/.area` usage). Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/physics/orbit_pressure.py` — **Debt**. Orbit workflow/pipeline in `physics` (sampling + field resolution + summaries); imports `analysis.orbits` sampling primitives and remains workflow-heavy. SI field requests now go through SmartDs/griblet (no `resolve_*` in this file), including base pointwise pressure quantities (`U`, `B`, magnetic/ram pressure). Remaining local work is mainly relative-velocity and standoff assembly. Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/physics/orbit_local.py` — **Debt**. Quantity-specific local orbit workflow wrappers (`local_mass_loss_*`, `local_torque_*`) now live in `physics`; they depend on `analysis.orbits` sampling primitives but remain workflow-heavy. SI field requests now go through SmartDs/griblet (including direct sampled `U_r/U_a/B_r/B_a`; no `resolve_*` in this file). Code TODO: added TODO.
- `starwinds_analysis/physics/orbit_surface.py` — **Debt**. Large orbit-surface workflow/pipeline in `physics`; imports `analysis.orbits` sampling primitives but still couples geometry/sampling with quantity assembly. SI field requests now go through SmartDs/griblet (no `resolve_*` in this file). Code TODO: added TODO.
- `starwinds_analysis/physics/orbits.py` — **Reviewed**. Kepler orbit kinematics primitives moved into `physics` (deeper/shared layer).
- `starwinds_analysis/physics/planetary_orbits.py` — **Reviewed**. Named orbit presets/helpers now depend on deep `physics.orbits` primitives instead of `analysis.orbits`.
- `starwinds_analysis/visualisation/profile_plots.py` — **Reviewed**. Generic shell-profile plotting helpers moved out of `physics` into the visualisation layer.
- `starwinds_analysis/physics/pressure.py` — **Debt**. `magnetic_pressure [Pa]` and `ram_pressure [Pa]` are now available through SmartDs/griblet, but this module still computes pressure/standoff quantities directly (including the component bundle) instead of requesting them from the graph. Code TODO: existing TODO(griblet).
- `starwinds_analysis/physics/torque.py` — **Debt**. Torque code is now consolidated in one file (local spherical torque density, explicit-surface torque terms, and temporary shell/surface wrappers), but it still carries quantity-specific shell/radius wrapper functions (`torque_vs_radius`, `surface_torque_vs_radius`) in a deep layer. SI field requests go through SmartDs/griblet (including `U_r/U_a/B_r/B_a` and pointwise shell torque densities), and shell values are read directly from shell `SmartDs`. Code TODO: existing TODO(griblet) + TODO(debt).
- `starwinds_analysis/physics/torque.py` — **Debt**. Local spherical torque-density terms are physical quantities computed outside SmartDs/griblet. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/wind_scaling.py` — **Reviewed**. Local wind-scaling formulas only; profile-bundle helper removed and `MU0` now comes from `physics.constants`.
- `starwinds_analysis/pipelines/__init__.py` — **Reviewed**. Boundary package only; intentionally minimal.
- `starwinds_analysis/analysis/orbits.py` — **Reviewed**. Orbit geometry and 1D-curve sampling primitives live in `analysis`; removed the one-module `sampling/` package.
- `starwinds_analysis/pipelines/sw_pipe.py` — **Debt**. Still a large mixed-responsibility CLI module (dispatch, recorder schema, state persistence, stdout logging, fail-fast policy) and remains the main monolith in `pipelines/`.
- `starwinds_analysis/pipelines/recorder.py` — **Debt**. Recorder capture + JSON normalization + persistence are now split out cleanly, but the file is still large and schema-heavy; keep it from becoming a second monolith.
- `starwinds_analysis/pipelines/shell.py` — **Debt**. The shell pipeline is readable, but it is still the largest pipeline and still contains significant shell-specific compute logic; keep pushing pointwise parts down into recipes/physics and avoid further local growth.
- `starwinds_analysis/param_in.py` — **Debt**. Still uses `_ensure_component(...)`; this is the same `_ensure*` helper pattern already called out as a bad smell and should be replaced with direct structure population.
- `starwinds_analysis/recipes/__init__.py` — **Reviewed**. Recipe exports; no bad-practice hit found in this pass.
- `starwinds_analysis/recipes/batsrus.py` — **Reviewed**. griblet recipe definitions (preferred place for derived quantity paths).
- `starwinds_analysis/recipes/spherical.py` — **Reviewed**. griblet/local spherical quantity recipes (preferred place for coordinate transforms/components).
- `starwinds_analysis/smart_ds.py` — **Debt**. Still carries `resolve` naming ambiguity and incomplete unit/centering-aware quantity request path; multiple TODOs already track this. Code TODO: existing TODOs.
- `starwinds_analysis/utils.py` — **Reviewed**. General small helpers; no clear current bad-practice hit recorded in this pass.
- `starwinds_analysis/visualisation/histograms.py` — **Reviewed**. Visualisation layer; plotting functions belong here more than in analysis/physics. Some quantity defaults exist but no code TODO added in this pass.
- `starwinds_analysis/vtk_utils.py` — **Reviewed**. Optional 3D visualisation bridge (separate integration layer); no additional debt marker added in this pass.

## test

- `test/test_installation.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_integrals.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_isosurface.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_orbit_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_orbit_pressure.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_orbit_surface_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_planetary_orbits.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_profile_plotting.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_read_plt.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_sample_data_helpers.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_shell_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_shell_magnetic_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_shell_resample_smartds_spec.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_slices_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_smart_ds.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_surface_torque_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_volumetric.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
