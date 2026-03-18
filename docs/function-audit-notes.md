# Function Audit Notes (Current Snapshot)

Date: 2026-03-07  
Branch: `dev`

Generated metadata:

- Generation method: repo-local static scan + manual caller review.
- Scope: `batwind/**/*.py` plus non-test callers in `examples/`.
- Refresh rule: rerun after API moves/renames in `analysis/`, `physics/`, `pipelines/`, `recipes/`, or `smart_ds.py`.

This file is a fresh snapshot of functions/classes currently present in
`batwind/`.

Conventions used here:

- "Used by" lists non-test, non-self callers in `batwind/` and `examples/`.
- If no such caller is found, it is marked as "No non-test call sites found".
- This replaces the previous stale snapshot and tracks only the current tree.

## `batwind/_smart_ds_graph.py`

- `graph_field_names`: Lists available fields from the active runtime graph.  
  Used by: `batwind/smart_ds.py`
- `graph_path`: Resolves the selected dependency path for a field.  
  Used by: `explain_field`
- `explain_field`: Returns human-readable graph path text (or tree).  
  Used by: `batwind/smart_ds.py`
- `compute_via_graph`: Resolves + evaluates a graph field.  
  Used by: `batwind/smart_ds.py`
- `build_runtime_graph`: Merges loader graph + attached graph recipes.  
  Used by: `compute_via_graph`, `graph_path`
- `build_loader_graph`: Exposes raw dataset variables and aux entries as zero-dependency recipes.  
  Used by: `build_runtime_graph`
- `evaluate_resolved_tree`: Evaluates one resolved dependency tree.  
  Used by: `compute_via_graph`

## `batwind/_smart_ds_resample.py`

- `resample_smart_ds`: Interpolates requested fields onto target points and returns a new `SmartDs`.  
  Used by: `batwind/smart_ds.py`

## `batwind/algorithms/sphere_sampling.py`

- `fibonacci_sphere`: Generates approximately uniform unit-sphere points.  
  Used by: `batwind/analysis/shells.py`
- `PolarAzimuthalGrid`: Structured polar/azimuth edge+center grid with area/cartesian helpers.  
  Used by: `batwind/analysis/shells.py`

## `batwind/algorithms/spherical.py`

- `cartesian_to_spherical_coordinates`: Cartesian -> `(R, polar, azimuth)`.  
  Used by: `batwind/recipes/spherical.py`
- `spherical_to_cartesian_coordinates`: `(R, polar, azimuth)` -> Cartesian.  
  Used by: `batwind/recipes/spherical.py`
- `polar_azimuth_to_latitude_longitude`: `(polar, azimuth)` -> `(latitude, longitude)`.  
  Used by: `batwind/recipes/spherical.py`
- `latitude_longitude_to_polar_azimuth`: `(latitude, longitude)` -> `(polar, azimuth)`.  
  Used by: `batwind/recipes/spherical.py`
- `cartesian_vector_to_spherical_components`: Cartesian vector -> `(r, p, a)` components.  
  Used by: `batwind/recipes/spherical.py`
- `spherical_vector_to_cartesian_components`: `(r, p, a)` vector -> Cartesian components.  
  Used by: No non-test call sites found

## `batwind/analysis/shell_summary.py`

- `boxcar_shell_weights`: Boxcar weighting over shell radii.  
  Used by: No non-test call sites found
- `summarize_shell_series`: Weighted mean/std/quantiles for one shell series.  
  Used by: No non-test call sites found
- `summarize_shell_diagnostics_band`: Band summary over shell diagnostics dicts.  
  Used by: No non-test call sites found

## `batwind/analysis/shells.py`

- `infer_cartesian_axis_radii`: Infers shell radii from axis-aligned samples.  
  Used by: No non-test call sites found
- `sample_spherical_shells`: Resamples to structured polar/azimuth shell grids.  
  Used by: notebooks/examples
- `sample_spherical_shells_fibonacci`: Resamples to equal-area Fibonacci shell points.  
  Used by: `batwind/pipelines/volume.py`, notebooks/examples
- `integrate_shell_scalar`: Area-integrates shell scalar fields and returns integral + coverage.  
  Used by: `batwind/physics/torque.py`, `batwind/pipelines/volume.py`, notebooks/examples

## `batwind/analysis/slices.py`

- `structured_quad_corners`: Builds quad connectivity for structured `(nz, nx)` grids.  
  Used by: No non-test call sites found
- `infer_range`: Infers plotting/resampling ranges with optional symmetry/padding.  
  Used by: No non-test call sites found
- `resample_structured_xz_slice`: Resamples 3D data to a structured XZ slice `SmartDs`.  
  Used by: No non-test call sites found

## `batwind/analysis/stats.py`

- `weighted_mean_std`: Weighted mean and standard deviation.  
  Used by: `batwind/analysis/shell_summary.py`
- `weighted_quantile`: Weighted quantiles for 1D samples.  
  Used by: `batwind/analysis/shell_summary.py`
- `summarize_samples`: Combined weighted summary helper (quantiles + mean/std).  
  Used by: `batwind/physics/orbit_surface.py`

## `batwind/analysis/trajectories.py`

- `trajectory_velocity`: Finite-difference trajectory velocity from points + time.  
  Used by: `examples/orbit_surface_revolution.ipynb`
- `circular_orbit_points`: Circular XY orbit points.  
  Used by: `examples/orbit_surface_revolution.ipynb`
- `sample_curve`: Resamples fields onto an explicit 1D curve.  
  Used by: notebooks/examples workflow
- `sample_trajectory`: Resamples fields onto trajectory points and appends `t`/optional `V_*`.  
  Used by: No non-test call sites found

## `batwind/data/field_names.py`

- `unit_from_brackets`: Extracts unit token from bracketed field names (`X [R]` -> `R`).  
  Used by: `batwind/analysis/shells.py`

## `batwind/data/samples.py`

- `data_dir`: Returns repository `sample_data` path.  
  Used by: No non-test call sites found
- `data_file`: Returns one `sample_data/<name>` path.  
  Used by: No non-test call sites found

## `batwind/param_in.py`

- `flatten_includes`: Flattens `#INCLUDE` trees into one line stream.  
  Used by: `ParamIn.from_file`
- `find_param_in`: Finds nearest `PARAM.in`/`param.in` up the folder chain.  
  Used by: `stellar_aux_from_nearby_param_in`
- `parse_sessions`: Parses command sessions/components from flat config lines.  
  Used by: `ParamIn.from_file`
- `ParamIn`: Parsed parameter-file object with command/parameter accessors (`get_*`, `stellar_params`).  
  Used by: `stellar_aux_from_nearby_param_in`
- `stellar_aux_from_nearby_param_in`: Reads nearby `PARAM.in` stellar values into aux-compatible keys.  
  Used by: `batwind/smart_ds.py`

## `batwind/physics/alfven_radius.py`

- `alfven_radius_map`: Computes first outward `M_A` crossing radius map.  
  Used by: `examples/alfven_radius_shell.ipynb`
- `projected_solid_angle_weights`: Computes projected `dOmega` weights from shell `dA` and `R`.  
  Used by: `examples/alfven_radius_shell.ipynb`
- `summarize_alfven_radius`: Summarizes map min/max/mean/cyl-mean/coverage.  
  Used by: `examples/alfven_radius_shell.ipynb`

## `batwind/physics/curve.py`

- `mass_loss_from_curve`: Local mass-loss estimate from sampled curve fields.  
  Used by: No non-test call sites found
- `torque_from_curve`: Local magnetic/dynamic/total torque estimates from sampled curve fields.  
  Used by: No non-test call sites found
- `relative_ram_pressure_from_trajectory`: Trajectory-frame ram pressure + standoff estimate.  
  Used by: No non-test call sites found

## `batwind/physics/orbit_surface.py`

- `surface_of_revolution_from_trajectory`: Builds a 2D surface grid from trajectory revolution.  
  Used by: `sample_surface_revolution`
- `surface_sample_weights`: Builds phase/longitude integration weights.  
  Used by: `pressure_components_on_surface`, `torque_components_on_surface`
- `phase_quantile_rows`: Quantiles by phase row over a 2D sampled surface.  
  Used by: `pressure_components_on_surface`, `torque_components_on_surface`
- `surface_point_normals_and_areas`: Centered-difference normals + local area estimate.  
  Used by: `torque_components_on_surface`
- `phase_line_integrals`: Integrates 2D surface density along longitude per phase.  
  Used by: `torque_components_on_surface`
- `sample_surface_revolution`: Resamples `SmartDs` fields on a trajectory-derived revolution surface.  
  Used by: `examples/orbit_surface_revolution.ipynb`
- `pressure_components_on_surface`: Computes pressure diagnostics on sampled surfaces.  
  Used by: `examples/orbit_surface_revolution.ipynb`
- `torque_components_on_surface`: Computes explicit-surface torque diagnostics (`T1..T4`, totals, summaries).  
  Used by: `examples/orbit_surface_revolution.ipynb`

## `batwind/physics/orbits.py`

- `PlanetOrbitElements`: Named orbital element container + built-in solar-system table.  
  Used by: No non-test call sites found
- `orbital_period`: Kepler period from semi-major axis + stellar mass.  
  Used by: No non-test call sites found
- `orbital_velocity`: Vis-viva orbital speed.  
  Used by: No non-test call sites found

## `batwind/physics/pressure.py`

- `ram_pressure`: `rho * V^2`.  
  Used by: `batwind/physics/curve.py`, `batwind/physics/orbit_surface.py`
- `magnetospheric_standoff_distance`: Stand-off proxy from pressure balance.  
  Used by: `batwind/physics/curve.py`, `batwind/physics/orbit_surface.py`

## `batwind/physics/torque.py`

- `spherical_wind_torque_density_terms`: Pointwise shell-style magnetic/dynamic torque densities.  
  Used by: No non-test call sites found
- `rotational_frame_velocity`: Converts inertial velocity to rotating-frame velocity.  
  Used by: `surface_torque_density_terms`
- `normalize_surface_normals`: Normalizes explicit surface normals.  
  Used by: `surface_torque_density_terms`, `radial_surface_normals`
- `radial_surface_normals`: Builds radial normals from position vectors.  
  Used by: `surface_torque_terms_on_shell_samples`
- `surface_torque_density_terms`: Computes `T1..T4` torque density terms and total.  
  Used by: `batwind/physics/orbit_surface.py`, `surface_torque_terms_on_shell_samples`
- `integrate_surface_torque_terms`: Integrates explicit-surface torque density terms over area.  
  Used by: `batwind/physics/orbit_surface.py`
- `surface_torque_terms_on_shell_samples`: Convenience wrapper for torque terms on shell sampled fields.  
  Used by: No non-test call sites found

## `batwind/physics/wind_scaling.py`

- `surface_escape_speed`: Escape speed from stellar mass/radius.  
  Used by: `open_wind_magnetisation`
- `open_wind_magnetisation`: Reville-style open wind magnetisation (`Upsilon_open`).  
  Used by: No non-test call sites found

## `batwind/pipelines/dummy_pipeline.py`

- `name_letter_counts`: Emits vowel/consonant counts for file stem.
- `name_profile_payload`: Emits float/string/array payload for file stem.
- `name_codepoints_payload`: Emits codepoint array payload.
- `name_waveform_payload`: Emits large waveform array payload (sidecar test path).
- `process_plt_file`: Dummy per-file pipeline entrypoint.

Used by: `batwind/pipelines/batwind_pipe.py`

## `batwind/pipelines/recorder.py`

- `ScientificFloatEncoder`: JSON encoder with scientific float formatting.
- `BatwindPipeResults`: In-memory run result container.
- `safe_name`: Safe filesystem tokenization helper.
- `relative_file_key`: Relative, stable file keys for state payload.
- `state_file_path`: Per-pipeline state filename builder.
- `sha256_file`: Input file hashing helper.
- `parse_record_payload`: Parses logger message payload into record key/value.
- `normalize_recorded_value`: JSON-normalizes payloads and offloads large arrays.
- `BatwindRecordHandler`: Logging handler that captures and stores recorded payloads.
- `load_state`: Loads processed/computed state.
- `load_state_payload`: Loads raw state payload for inspector CLI.
- `save_state`: Writes per-pipeline state JSON.

Used by: `batwind/pipelines/batwind_pipe.py`, `batwind/pipelines/batwind_pipe_results.py`

## `batwind/pipelines/shell.py`

- `shell_cell_values`: Converts nodal values to cell-centered shell map values.
- `load_shell_grid`: Loads shell radii/grid nodes/masks/areas from `SmartDs`.
- `shell_map_and_profile`: Computes outer shell map + radial integral profile.
- `process_plt_file`: Shell pipeline entrypoint.

Used by: `batwind/pipelines/batwind_pipe.py`

## `batwind/pipelines/slice.py`

- `process_plt_file`: Slice pipeline entrypoint (`rho`, `U`, `B`, `B_r` figures + records).

Used by: `batwind/pipelines/batwind_pipe.py`

## `batwind/pipelines/batwind_pipe.py`

- `PipelineSourceFilter`: Injects `pipeline_source` for unified logging format.
- `configure_logger`: Human stdout logger setup (colorlog when available).
- `configure_recorder`: Recorder logger setup.
- `discover_input_files`: Finds `.plt`/`.dat` files.
- `pipeline_name_for_file`: Filename-prefix pipeline selection (`3d`, `shl`, `x=0/y=0/z=0`).
- `process_file_for_pipeline`: Resolves built-in pipeline process function.
- `run_batwind_pipe`: Core orchestrator over selected files.
- `build_parser`: CLI parser for `batwind-pipe`.
- `main`: CLI entrypoint.

Used by: CLI

## `batwind/pipelines/batwind_pipe_results.py`

- `_computed_results`: Extracts the computed-results mapping.
- `_iter_file_keys`: Lists file keys in display order.
- `_iter_fields`: Lists available recorded field names.
- `_resolve_field_name`: Singular/plural convenience field resolution.
- `_extract_field_value`: Returns value-only or full payload with source metadata.
- `build_parser`: CLI parser for results inspection.
- `main`: CLI entrypoint for querying/listing state payload contents.

Used by: CLI

## `batwind/pipelines/utils.py`

- `slug_key`: Slugifies text for output naming.
- `output_prefix_from_input_file`: Output prefix builder from input filename.

Used by: `batwind/pipelines/slice.py`, `batwind/pipelines/shell.py`, `batwind/pipelines/volume.py`

## `batwind/pipelines/volume.py`

- `process_plt_file`: Volume pipeline entrypoint (shell sampling + mass/torque/open-flux/energy plots and records).

Used by: `batwind/pipelines/batwind_pipe.py`

## `batwind/recipes/batsrus.py`

- `build_griblet_batsrus_graph`: Top-level BATSRUS recipe graph builder (normalization + derived).
- `build_griblet_unit_normalization_graph`: Raw-unit -> SI conversions + scalar aux parsing recipes.
- `build_griblet_common_derived_graph`: Derived SI recipes (Mach, pressure, fluxes, torque densities, helpers).
- `build_griblet_vector_cartesian_graph`: `prefix_xyz` and vector magnitudes from Cartesian components.
- `body_radius_from_inputs`: Resolves body radius from explicit input or aux keys.

Used by: `batwind/smart_ds.py`, `batwind/analysis/trajectories.py`

## `batwind/recipes/spherical.py`

- `_vector_triplets`: Finds Cartesian vector triplets (`*_x/_y/_z`) by prefix/unit.
- `build_griblet_spherical_geometry_graph`: Adds coordinate recipes (`XYZ <-> R/polar/azimuth`, `lat/lon`).
- `build_griblet_vector_spherical_components_graph`: Adds vector component recipes (`xyz -> r/p/a`, stacked `*_rpa`).

Used by: `batwind/smart_ds.py`, `batwind/recipes/batsrus.py`

## `batwind/smart_ds.py`

- `SmartDs`: Main wrapper around `batread.Dataset`.
- `SmartDs.from_file`: File loader + nearby `PARAM.in` stellar aux ingestion.
- `SmartDs.prepare`: Attaches BATSRUS + spherical recipe graphs.
- `SmartDs.__getitem__`: Main field accessor (`raw -> graph`, with cache).
- `SmartDs.explain`: Dependency-path explanation for a requested field.
- `SmartDs.base_fields_for_resample`: Expands requested fields to raw interpolation dependencies.
- `SmartDs.resample`: Generic resampling entrypoint returning a new `SmartDs`.
- `SmartDs.append_fields`: Appends structured extra fields into a new dataset/wrapper.
- `SmartDs.has_field` / `keys`: Field availability and discoverability helpers.
- `SmartDs.clear_cache`: Clears field cache + resample spatial caches.

Used by: all pipelines, recipes, analysis primitives, and examples/notebooks.

## `batwind/visualisation/histograms.py`

- `plot_cumulative_hists`: CDF histogram helper.
- `plot_vs_radius`: Scatter-vs-radius helper.
- `plot_binned_vs_radius`: Binned radial summary helper.
- `plot_radial_hist2d`: 2D radial histogram helper.

Used by: `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`

## `batwind/visualisation/profile_plots.py`

- `shell_profile_height`: Extracts/derives shell height from profile payload.
- `plot_shell_height_series`: Generic shell profile line plotting primitive.

Used by: No non-test call sites found

## `batwind/visualisation/slice.py`

- `auto_coords`: Detects two varying coordinates in nominal 2D datasets.
- `triangles`: Builds triangulation from quad cell connectivity.
- `default_slice_field`: Chooses a default field for slice plotting.
- `plot_xz_slice_tripcolor_with_marginals`: Tripcolor + marginal panels.
- `plot_xz_slice_tripcolor_with_cross_quantiles`: Tripcolor + cross-quantile panels.
- `plot_xz_slice_with_marginal_points`: Tripcolor + point marginals.
- `plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords`: Tripcolor + unique-coordinate quantiles.

Used by: `batwind/pipelines/slice.py`, `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`, plain-data examples

## `batwind/vtk_utils.py`

- `read`: Reads `.plt` file and returns converted VTK/PyVista grid, optionally SI-normalized.
- `convert`: Converts raw dataset into VTK/PyVista unstructured grid.
- `convert_to_base_si`: Renames/scales common BATSRUS fields to SI on VTK grid.

Used by: No non-test call sites found
