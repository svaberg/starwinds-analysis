# Technical Debt Review (Python Files)

This file tracks a full file-by-file pass against `/Users/dagfev/Documents/starwinds/starwinds-analysis/docs/bad-practices.md`.

- Files reviewed: **55** (`*.py`, entire repo)
- Scope of this pass: identify debt and mark code with `TODO` comments where bad practices are present
- Note: tests/examples were reviewed too, but production architecture rules are applied primarily to library code

Legend:
- `Debt`: bad-practice hit found in this pass (code TODO added or already present)
- `Reviewed`: no additional debt marker added in this pass

## examples

- `examples/earth-xuv-neutrals/earth-xuv-neutrals.py` — **Reviewed**. Example script (one-off workflow code is allowed here).
- `examples/planet.py` — **Reviewed**. Example/legacy script; one-off code is allowed in examples.

## starwinds_analysis

- `starwinds_analysis/_smart_ds_graph.py` — **Debt**. Internal `resolve_field(...)` naming still collides conceptually with forbidden `resolve_*` pattern; keep graph-path resolution distinct from user field/unit requests. Code TODO: existing TODO.
- `starwinds_analysis/_smart_ds_resample.py` — **Reviewed**. Core resampling internals; no new bad-practice hit beyond existing documented NaN handling for interpolation edge cases.
- `starwinds_analysis/algorithms/sphere_sampling.py` — **Reviewed**. Geometry/sampling primitive module (good layer fit).
- `starwinds_analysis/analysis/__init__.py` — **Debt**. `analysis` re-exports many `physics` symbols (reversed layer inclusion path) and exposes a very broad API surface. Code TODO: added TODO.
- `starwinds_analysis/analysis/fluxes.py` — **Debt**. Quantity-specific `analysis` module (`fluxes`) with `*_vs_radius` wrappers, `resolve_*` usage, and local quantity recomputation (`B_r`, `U_r`, `E*U_r`). Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/analysis/orbits.py` — **Debt**. Mixed generic orbit geometry/sampling with quantity-specific comparison workflows (`local_mass_loss_*`, `local_torque_*`); imports from `physics`. Code TODO: added TODO.
- `starwinds_analysis/analysis/shell_summary.py` — **Reviewed**. Reducer/summary helpers; finite filtering appears intentional for shell-band summaries.
- `starwinds_analysis/analysis/shells.py` — **Debt**. Contains `resolve_*` helpers and compatibility custom container (`SphericalShellSamples`) alongside core shell primitives. Code TODO: added TODO + existing TODOs.
- `starwinds_analysis/analysis/slices.py` — **Reviewed**. Structured slice resampling/topology helpers; no clear rule violation found in this pass.
- `starwinds_analysis/analysis/stats.py` — **Reviewed**. Generic weighted stats primitives; no clear rule violation found in this pass.
- `starwinds_analysis/analysis/surface_torque.py` — **Debt**. Quantity-specific `surface_torque` analysis wrappers (`*_vs_radius`) and `resolve_*` usage; imports from `physics`. Code TODO: added TODO.
- `starwinds_analysis/data/samples.py` — **Reviewed**. Sample-data path helper; no bad-practice hit found.
- `starwinds_analysis/physics/__init__.py` — **Debt**. Deep-layer re-export surface includes non-local/profile-derived helper exports, widening API surface. Code TODO: added TODO.
- `starwinds_analysis/physics/flux_density.py` — **Debt**. Local physical quantity (`q * U_r`) is computed outside SmartDs/griblet instead of requested as an SI quantity. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/local_estimates.py` — **Debt**. Mixes local physics formulas with summary/reporting helper and imports `analysis.stats` (reversed layer direction). Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/physics/magnetic.py` — **Debt**. Magnetic spherical components (`B_r`, `B_theta`, `B_phi`) are recomputed locally instead of requested via SmartDs/griblet. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/mass_loss.py` — **Debt**. Quantity-specific shell pipeline wrappers (`sample_shell_mass_flux_map`, `mass_loss_vs_radius`), custom container (`ShellMassFluxMap`), `resolve_*`, and `analysis.shells` dependency. Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/physics/orbit_pressure.py` — **Debt**. Orbit workflow/pipeline in `physics` (sampling + field resolution + summaries), imports `analysis`, and uses `resolve_*`. Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/physics/orbit_surface.py` — **Debt**. Large orbit-surface workflow/pipeline in `physics`, imports `analysis`, and couples geometry/sampling with quantity assembly. Code TODO: added TODO.
- `starwinds_analysis/physics/planetary_orbits.py` — **Debt**. Constants/helper module still imports Kepler primitive (`orbital_period`) from `analysis.orbits` (reversed layer direction). Code TODO: added TODO.
- `starwinds_analysis/physics/plotting.py` — **Debt**. Quantity-specific plotting wrappers in deep `physics` layer; plotting API surface is larger than desired and not purely generic. Code TODO: added TODO.
- `starwinds_analysis/physics/pressure.py` — **Debt**. Pressure and standoff quantities (`magnetic_pressure`, `ram_pressure`, component bundle) still computed outside SmartDs/griblet. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/shell_torque.py` — **Debt**. Quantity-specific shell torque profile wrapper (`torque_vs_radius`) depends on `analysis.shells` + `resolve_*`. Code TODO: added TODO.
- `starwinds_analysis/physics/surface_torque.py` — **Debt**. Local torque terms (`T1..T4`) still computed outside SmartDs/griblet; heavy finite-mask/NaN handling should remain intentional/documented. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/torque.py` — **Debt**. Local spherical torque-density terms are physical quantities computed outside SmartDs/griblet. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/wind_scaling.py` — **Debt**. Mixes local formulas with profile-bundle helper (`open_wind_magnetisation_from_profiles`) and redefines `MU0` locally. Code TODO: added TODO.
- `starwinds_analysis/pipelines/__init__.py` — **Reviewed**. Boundary package only; intentionally minimal.
- `starwinds_analysis/quicklook2d.py` — **Debt**. High-level orchestration/convenience wrappers in library (large API surface, quantity-specific presets/workflows) vs library-purity guideline. Code TODO: added TODO.
- `starwinds_analysis/recipes/__init__.py` — **Reviewed**. Recipe exports; no bad-practice hit found in this pass.
- `starwinds_analysis/recipes/batsrus.py` — **Reviewed**. griblet recipe definitions (preferred place for derived quantity paths).
- `starwinds_analysis/recipes/spherical.py` — **Reviewed**. griblet/local spherical quantity recipes (preferred place for coordinate transforms/components).
- `starwinds_analysis/smart_ds.py` — **Debt**. Still carries `resolve` naming ambiguity and incomplete unit/centering-aware quantity request path; multiple TODOs already track this. Code TODO: existing TODOs.
- `starwinds_analysis/utils.py` — **Reviewed**. General small helpers; no clear current bad-practice hit recorded in this pass.
- `starwinds_analysis/visualisation/histograms.py` — **Reviewed**. Visualisation layer; plotting functions belong here more than in analysis/physics. Some quantity defaults exist but no code TODO added in this pass.
- `starwinds_analysis/vtk_utils.py` — **Reviewed**. Optional VTK/PyVista bridge (separate integration layer); no additional debt marker added in this pass.

## test

- `test/test_installation.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_integrals.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_isosurface.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_orbit_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_orbit_pressure.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_orbit_surface_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_planetary_orbits.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_profile_plotting.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_quicklook2d.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_read_plt.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_sample_data_helpers.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_shell_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_shell_magnetic_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_shell_resample_smartds_spec.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_slices_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_smart_ds.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_surface_torque_analysis.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.
- `test/test_volumetric.py` — **Reviewed**. Test module reviewed; production-layer bad-practice rules are not directly enforced here.

