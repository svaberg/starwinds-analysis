# Logging Plan

This file defines how logging should be done in every production `*.py` file in `batwind/`, excluding package `__init__.py` files and tests.

The standard is:

- `INFO` for meaningful stage starts and user-visible progress
- `DEBUG` for algorithm steps, branch choices, inferred values, cache behavior, skipped work, and completion
- `WARNING` for recoverable but suspicious runtime situations
- `ERROR` only when the code is about to fail or return a degraded result that should be obvious

The goal is not "log everywhere." The goal is:

- no silent nontrivial workflow code
- no `INFO` spam for tiny internal actions
- `DEBUG` logs that explain what the algorithm is doing

## General Rules

### `INFO`

Use `INFO` when a person running the code would care that a meaningful step has started.

Examples:

- loading a dataset
- building a graph
- starting resampling
- starting a shell or slice analysis
- starting a costly physical calculation
- writing a real output artifact

### `DEBUG`

Use `DEBUG` as executable documentation.

Examples:

- inferred coordinate fields
- graph layers attached
- detected vector triplets
- chosen interpolation branch
- cache hit vs cache build
- skipped optional field
- output shape
- reduction sizes
- completion with elapsed time

### Completion Pattern

Where a step is expensive enough to matter, use:

- `INFO Doing xyz...`
- `DEBUG Doing xyz complete in %.2f s.`

### Avoid

- duplicate `INFO` start and `INFO` complete lines
- logs that simply repeat the function name
- defensive noise
- success chatter for trivial helpers

## File-By-File Plan

### Root Modules

#### `batwind/_smart_ds_resample.py`

Role:

- core resampling engine for `nearest`, `linear`, and `octree`

Logging plan:

- keep `INFO` at resample entry
- add `DEBUG` for:
  - sample shape
  - coordinate field tuple
  - method
  - source coordinate shape
  - finite coordinate count
  - output variable count
  - cache creation vs reuse
  - shared-tree/shared-triangulation vs field-local path
  - octree interpolator creation vs reuse
  - dataset rebuild
  - total interpolation elapsed time

#### `batwind/constants.py`

Role:

- pure constants

Logging plan:

- no runtime logging

#### `batwind/param_in.py`

Role:

- nearby `PARAM.in` discovery and parsing

Logging plan:

- keep `INFO` when a nearby `PARAM.in` is found and used
- use `DEBUG` for:
  - search path resolution
  - missing file case
  - parsed block presence
  - parsed stellar fields
- avoid extra chatter for tiny parsing helpers

#### `batwind/smart_ds.py`

Role:

- graph-backed dataset wrapper and main user-facing data access layer

Logging plan:

- `INFO` for:
  - `from_file(...)`
  - `resample(...)`
- `DEBUG` for:
  - nearby `PARAM.in` aux merge
  - graph layers attached in `from_file(...)`
  - `body_radius_m` source
  - `__getitem__` raw path vs graph path
  - graph resolution cost
  - unresolved field before raising
  - cache hit and cache store
  - graph reset and graph merge counts
  - source-field expansion for resampling
  - inferred coordinate fields
  - append-fields behavior
  - cache clearing behavior
  - completion timing where a step is large enough

#### `batwind/utils.py`

Role:

- small shared helpers

Logging plan:

- only use `DEBUG` where helpers make a real choice:
  - coordinate inference
  - triangulation setup
  - geometric branch decisions
- no `INFO` here unless a helper actually starts a meaningful workflow stage

### Algorithms

#### `batwind/algorithms/sphere_sampling.py`

Role:

- sphere point/grid generation

Logging plan:

- `DEBUG` for:
  - sampling strategy
  - point count
  - angular resolution
  - output shape
  - completion

#### `batwind/algorithms/spherical.py`

Role:

- spherical geometry math

Logging plan:

- mostly `DEBUG` only
- use `DEBUG` for:
  - input/output shape narration where useful
  - branch choices for coordinate interpretation
- keep warnings/errors for invalid geometry states

### Analysis

#### `batwind/analysis/shell_summary.py`

Role:

- summaries and reporting over shell diagnostics

Logging plan:

- `INFO` for major summary build starts
- `DEBUG` for:
  - band counts
  - included metrics
  - reduction sizes
  - completion

#### `batwind/analysis/shells.py`

Role:

- shell sampling and shell-based integrations

Logging plan:

- `INFO` for shell sampling/integration entry points
- `DEBUG` for:
  - shell radii count
  - per-shell point count
  - resample method
  - requested fields
  - source fields
  - output shape
  - integration choices
  - completion with elapsed time

#### `batwind/analysis/slices.py`

Role:

- structured slice dataset generation

Logging plan:

- `INFO` for starting slice generation
- `DEBUG` for:
  - inferred plane
  - bounds/ranges
  - resolution
  - coordinate fields
  - resample method
  - output shape
  - completion with elapsed time

#### `batwind/analysis/stats.py`

Role:

- weighted reductions and distribution summaries

Logging plan:

- `DEBUG` for:
  - data size
  - weight shape/broadcast choices
  - quantile requests
  - reduction mode
- no `INFO` unless a larger summary workflow is added

#### `batwind/analysis/trajectories.py`

Role:

- trajectory sampling and curve construction

Logging plan:

- `INFO` for trajectory/curve sampling starts
- `DEBUG` for:
  - path shape
  - field set
  - attached vector graph details
  - source-field expansion
  - output dataset shape
  - completion with elapsed time

### Data

#### `batwind/data/field_names.py`

Role:

- field-name parsing and canonical-name helpers

Logging plan:

- keep warnings for parse failures that matter
- use `DEBUG` sparingly for nontrivial parsing/normalization decisions
- do not log routine string manipulation

#### `batwind/data/samples.py`

Role:

- sample file discovery and path helpers

Logging plan:

- `DEBUG` for successful resolution and path selection
- `WARNING` if a sample expectation is violated
- no `INFO` for ordinary path lookup success

### Physics

#### `batwind/physics/alfven_radius.py`

Role:

- Alfven-radius diagnostics

Logging plan:

- `INFO` for main diagnostic start
- `DEBUG` for:
  - grid sizes
  - crossing search setup
  - branch decisions
  - result count
  - completion with elapsed time

#### `batwind/physics/curve.py`

Role:

- curve-relative physical quantities

Logging plan:

- `INFO` for meaningful curve-physics calculations
- `DEBUG` for:
  - frame choice
  - field availability choices
  - reduction sizes
  - completion

#### `batwind/physics/orbit_surface.py`

Role:

- orbit-surface pressure and torque workflows

Logging plan:

- `INFO` for each major workflow start
- `DEBUG` for:
  - chosen pressure terms
  - relative-frame vs inertial-frame path
  - source-field set
  - surface grid shape
  - skipped optional terms
  - totals and reductions
  - completion with elapsed time

#### `batwind/physics/orbits.py`

Role:

- orbit geometry helpers

Logging plan:

- mostly `DEBUG`
- log:
  - orbital parameter interpretation
  - generated orbit shapes
  - special-case branch choices

#### `batwind/physics/pressure.py`

Role:

- pressure component formulas

Logging plan:

- mostly quiet
- keep warnings for invalid/unexpected states
- add `DEBUG` only if a function has multiple meaningful formula branches

#### `batwind/physics/torque.py`

Role:

- torque density and integrated torque workflows

Logging plan:

- `INFO` for major torque workflow starts
- `DEBUG` for:
  - component inclusion
  - chosen frame
  - source-field set
  - shell/surface wrapper setup
  - output shapes
  - completion with elapsed time

#### `batwind/physics/wind_scaling.py`

Role:

- wind-scaling relations

Logging plan:

- mostly quiet
- keep warnings for invalid regimes
- add `DEBUG` if a scaling branch or normalization choice is nontrivial

### Pipelines

#### `batwind/pipelines/batwind_pipe.py`

Role:

- main pipeline runner

Logging plan:

- `INFO` for run start and per-file pipeline dispatch
- `DEBUG` for:
  - discovery summary
  - formatter selection
  - pipeline routing decision
  - processed/failed/skipped counts
  - run completion

#### `batwind/pipelines/batwind_pipe_results.py`

Role:

- CLI for pipeline state/result inspection

Logging plan:

- `INFO` for top-level query or dump actions
- `DEBUG` for:
  - loaded state size
  - selected result subset
  - completion

#### `batwind/pipelines/dummy_pipeline.py`

Role:

- simple pipeline scaffold

Logging plan:

- `INFO` for pipeline start
- `DEBUG` for derived output names and completion

#### `batwind/pipelines/equatorial_stitch.py`

Role:

- equatorial stitch workflow

Logging plan:

- `INFO` for:
  - input loading
  - stitch computation start
  - output writing
- `DEBUG` for:
  - selected files
  - inferred geometry
  - overlap decisions
  - stitch dimensions
  - output artifact details
  - completion with elapsed time

#### `batwind/pipelines/recorder.py`

Role:

- pipeline state persistence

Logging plan:

- `DEBUG` for:
  - missing state files
  - loaded counts
  - saved counts
  - serialization paths
- `INFO` only for major persistence actions if they are user-visible

#### `batwind/pipelines/shell.py`

Role:

- shell pipeline

Logging plan:

- `INFO` for major stage starts
- `DEBUG` for:
  - completed stage timing
  - field availability choices
  - output writing

#### `batwind/pipelines/slice.py`

Role:

- slice pipeline

Logging plan:

- `INFO` for major stage starts
- `DEBUG` for:
  - plotting inputs
  - field availability choices
  - completed stage timing
  - output writing

#### `batwind/pipelines/utils.py`

Role:

- pipeline helper utilities

Logging plan:

- mostly `DEBUG`
- log file/path routing decisions and helper inference
- avoid `INFO` unless a helper actually starts meaningful work

#### `batwind/pipelines/volume.py`

Role:

- 3D volume pipeline and fake LOS image workflow

Logging plan:

- `INFO` for:
  - dataset load
  - shell sampling
  - mass loss / torque / flux stages
  - fake LOS image stage
  - output writing
- `DEBUG` for:
  - elapsed time per stage
  - chosen energy-field behavior
  - octree reuse
  - LOS grid shape
  - output artifact details

### Recipes

#### `batwind/recipes/batsrus.py`

Role:

- BATSRUS graph construction

Logging plan:

- `INFO` for top-level graph build start
- `DEBUG` for:
  - incoming variable count
  - `gamma` presence
  - `body_radius_m` presence
  - graph sections merged
  - canonical fields added
  - completion

#### `batwind/recipes/spherical.py`

Role:

- spherical geometry and spherical-component graph construction

Logging plan:

- `INFO` for top-level spherical graph build
- `DEBUG` for:
  - coordinate field tuple
  - geometry fields added
  - detected vector triplets
  - spherical component names added
  - completion

#### `batwind/recipes/vectors.py`

Role:

- generic vector graph construction from `*_x/*_y/*_z` triplets

Logging plan:

- `INFO` for vector graph build start
- `DEBUG` for:
  - exact triplets detected
  - generated vector and magnitude field names
  - completion

### Visualisation

#### `batwind/visualisation/histograms.py`

Role:

- histogram plotting helpers

Logging plan:

- mostly `DEBUG`
- log:
  - field count
  - bin configuration
  - normalization mode
  - weighted/unweighted choice

#### `batwind/visualisation/profile_plots.py`

Role:

- profile plotting

Logging plan:

- mostly `DEBUG`
- log:
  - profile field names
  - series count
  - plotting mode
  - completion

#### `batwind/visualisation/slice.py`

Role:

- slice plotting helpers

Logging plan:

- mostly `DEBUG`
- log:
  - slice field
  - plot style
  - normalization mode
  - grid shape
  - completion

## Explicit Exclusions

This plan does not cover:

- package `__init__.py` files
- anything under `test/`

Those should stay quiet unless there is a very specific reason not to.

## Acceptance Standard

This plan is complete only when:

- every production `*.py` file listed above has an explicit logging stance
- every nontrivial workflow module uses `INFO` and `DEBUG` deliberately
- `DEBUG` logs explain algorithm behavior rather than merely repeating function names
- expensive stages use the `INFO ... / DEBUG ... complete in %.2f s` pattern
- quiet modules remain intentionally quiet rather than accidentally silent
