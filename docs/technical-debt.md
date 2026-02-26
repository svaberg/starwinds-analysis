# Technical Debt Review (Python Files)

This file tracks a full file-by-file pass against `/Users/dagfev/Documents/starwinds/starwinds-analysis/docs/bad-practices.md`.

- Files reviewed: **57** (`*.py`, entire repo)
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
- `starwinds_analysis/analysis/__init__.py` — **Reviewed**. Re-export facade removed; package boundary is now minimal (`__all__ = []`).
- `starwinds_analysis/analysis/orbits.py` — **Reviewed**. Now limited to orbit geometry/sampling primitives (local mass-loss/torque workflows moved out).
- `starwinds_analysis/analysis/shell_summary.py` — **Reviewed**. Reducer/summary helpers; finite filtering appears intentional for shell-band summaries.
- `starwinds_analysis/analysis/shells.py` — **Debt**. Contains `resolve_*` helpers and compatibility custom container (`SphericalShellSamples`) alongside core shell primitives. Code TODO: added TODO + existing TODOs.
- `starwinds_analysis/analysis/slices.py` — **Reviewed**. Structured slice resampling/topology helpers; no clear rule violation found in this pass.
- `starwinds_analysis/analysis/stats.py` — **Reviewed**. Generic weighted stats primitives; now also owns the reusable `summarize_samples(...)` helper.
- `starwinds_analysis/data/samples.py` — **Reviewed**. Sample-data path helper; no bad-practice hit found.
- `starwinds_analysis/physics/__init__.py` — **Reviewed**. Deep-layer package boundary is now minimal (`__all__ = []`).
- `starwinds_analysis/physics/fluxes.py` — **Debt**. Quantity-specific shell flux profile wrappers remain (`*_vs_radius`); SI field requests now go through SmartDs/griblet, but local quantity recomputation (`B_r`, `U_r`, `E*U_r`) is still done in code. Code TODO: existing TODO(debt) + TODO(griblet).
- `starwinds_analysis/physics/constants.py` — **Reviewed**. Shared constants module (good deep-layer home for physical constants like `MU0`).
- `starwinds_analysis/physics/flux_density.py` — **Debt**. Local physical quantity (`q * U_r`) is computed outside SmartDs/griblet instead of requested as an SI quantity. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/local_estimates.py` — **Debt**. Local physics formulas remain outside SmartDs/griblet (intentional TODOs), but summary/reporting helper was removed and the `analysis.stats` import was eliminated. Code TODO: existing TODO(griblet).
- `starwinds_analysis/physics/magnetic.py` — **Debt**. Magnetic spherical components (`B_r`, `B_theta`, `B_phi`) are recomputed locally instead of requested via SmartDs/griblet. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/mass_loss.py` — **Debt**. Quantity-specific shell pipeline wrappers (`sample_shell_mass_flux_map`, `mass_loss_vs_radius`), custom container (`ShellMassFluxMap`), and `analysis.shells` dependency remain. SI field requests now go through SmartDs/griblet (no `resolve_*` in this file). Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/physics/orbit_pressure.py` — **Debt**. Orbit workflow/pipeline in `physics` (sampling + field resolution + summaries) and imports `analysis`; SI field requests now go through SmartDs/griblet (no `resolve_*` in this file). Code TODO: added TODO + existing TODO(griblet).
- `starwinds_analysis/physics/orbit_local.py` — **Debt**. Quantity-specific local orbit workflow wrappers (`local_mass_loss_*`, `local_torque_*`) now live in `physics`; still depend on `analysis` sampling, but SI field requests now go through SmartDs/griblet (no `resolve_*` in this file). Code TODO: added TODO.
- `starwinds_analysis/physics/orbit_surface.py` — **Debt**. Large orbit-surface workflow/pipeline in `physics`, imports `analysis`, and couples geometry/sampling with quantity assembly. SI field requests now go through SmartDs/griblet (no `resolve_*` in this file). Code TODO: added TODO.
- `starwinds_analysis/physics/orbits.py` — **Reviewed**. Kepler orbit kinematics primitives moved into `physics` (deeper/shared layer).
- `starwinds_analysis/physics/planetary_orbits.py` — **Reviewed**. Named orbit presets/helpers now depend on deep `physics.orbits` primitives instead of `analysis.orbits`.
- `starwinds_analysis/physics/plotting.py` — **Debt**. Quantity-specific shell-map plotting wrapper was removed; remaining plotting helpers are generic shell-profile plotting, but deep-layer plotting API surface should stay small. Code TODO: added TODO.
- `starwinds_analysis/physics/pressure.py` — **Debt**. Pressure and standoff quantities (`magnetic_pressure`, `ram_pressure`, component bundle) still computed outside SmartDs/griblet. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/shell_torque.py` — **Debt**. Quantity-specific shell torque profile wrapper (`torque_vs_radius`) depends on `analysis.shells`; SI field requests now go through SmartDs/griblet (no `resolve_*` in this file). Code TODO: added TODO.
- `starwinds_analysis/physics/surface_torque.py` — **Debt**. Local torque terms (`T1..T4`) still computed outside SmartDs/griblet, and the file now also carries temporary shell/radius wrapper functions (`surface_torque_vs_radius`, etc.); SI field requests now go through SmartDs/griblet (no `resolve_*` in this file). Code TODO: existing TODO(griblet) + TODO(debt).
- `starwinds_analysis/physics/torque.py` — **Debt**. Local spherical torque-density terms are physical quantities computed outside SmartDs/griblet. Code TODO: existing TODO(griblet) added.
- `starwinds_analysis/physics/wind_scaling.py` — **Reviewed**. Local wind-scaling formulas only; profile-bundle helper removed and `MU0` now comes from `physics.constants`.
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
