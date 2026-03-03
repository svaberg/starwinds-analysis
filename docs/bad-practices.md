# Bad Practices (Anti-Patterns) To Avoid

This is a living list of patterns we want to avoid in `starwinds-analysis`.

The goal is not "purity". The goal is:

- short readable code
- reusable general APIs
- notebooks/examples that demonstrate easy usage
- SI-first analysis (with explicit exceptions like `R` and Gauss-for-plotting)

## Core Style Rule: Prefer Short Code

Rule:

- Prefer short code when the meaning stays clear.
- Do not add ceremony (casts, wrappers, helper layers, dict bundles, re-boxing) around
  one or two real lines of computation.
- If a function is mostly boilerplate and only contains a tiny amount of real work,
  that is a code smell.

Implication:

- Write the math/operation directly when it is simple.
- Split code only for real reuse, real boundaries, or real complexity.

## Branching Rule: No Gratuitous Guards

Rule:

- Do not add guard branches, fallback branches, or conditional attachment logic unless the design explicitly requires them.
- Do not "fix" a downstream composition problem by branching earlier in the call path.
- Prefer one direct model over multiple conditional code paths.

Bad:

- attaching core graph fragments only for some files
- skipping behavior upstream because one downstream composition path is noisy
- adding random guards that hide the real source of a problem

Preferred pattern:

- attach the intended fragments consistently
- fix the actual composition point where the incorrect behavior appears
- keep branching only where it is part of the real API or domain model

## Data Geometry Rule: Treat Input Data Honestly

Rule:

- In general, data read by this project are corner-point / nodal data.
- Do not pretend imported coordinates are cell centres just to make plotting or integration easier.
- If cell-centred values or cell areas/volumes are needed, compute them explicitly from the nodal geometry and nodal values.

Bad:

- treating node coordinates as if they were cell centres
- inferring a fake centred mesh from nodal coordinates and then plotting/integrating as if that were the native data model
- hiding node-to-cell conversion inside plotting convenience code

Preferred pattern:

- keep imported coordinates as corner nodes
- keep imported fields as nodal values
- when a cell-based operation is needed, make the node-to-cell conversion explicit in code

Terminology rule:

- Unless code is directly reading file-provided latitude/longitude fields or labeling a plot, prefer `polar` and `azimuth` over `latitudinal`.
- Use `lat` / `lon` when the file fields are explicitly latitude/longitude.

## Logging (Good and Bad)

Logging is required for workflow visibility, especially in pipeline and analysis entry points.

Good:

- Add `log.info(...)` at workflow start/end with key context (`n_radii`, `method`, sampling mode, output path).
- Add `log.debug(...)` for intermediate diagnostics that help debugging but are too verbose for normal runs.
- Use module loggers (`logging.getLogger(__name__)`) and logger hierarchy (for example child loggers in pipelines).
- Keep logging focused on boundaries and major steps, not every line.
- When implementing a TODO in workflow code, add appropriate logging as part of the implementation.

Bad:

- No logging in long-running workflows (hard to debug, no progress visibility).
- Logging inside hot inner loops or per-point/per-cell operations (log spam and slowdowns).
- `print(...)` instead of logger usage in library code.
- Logging large raw arrays directly.
- Swallowing exceptions without logging context.
- parenthesized multi-import blocks (`from x import (...)`)

Rule:

- Use flat imports only, one import per line.

## Pipeline Rule: Keep Pipelines Clean And Simple

Rule:

- Pipeline modules are thin orchestration layers, not implementation layers.
- Pipelines should mostly call reusable functions from `physics/` and `visualisation/`.
- Pipeline `process_plt_file(...)` should stay short and user-serviceable:
  - load input
  - call reusable compute/plot functions
  - save outputs
  - log/record key results
- In pipelines, do work in sequence:
  - compute one quantity
  - plot that quantity
  - record that quantity
  - then move to the next quantity
- Do not group many quantities first and then unpack them later in the pipeline.
- Do not hold results for a final end-of-pipeline summary record; record each result when it is produced.
- Pipelines should normally need only a single smoke test.
- Do not keep production code alive just to satisfy pipeline-specific tests.
- If logic needs detailed tests, that logic belongs in lower reusable layers (`physics/`, `analysis/`, `visualisation/`), not inside `pipelines/`.
- If a test mainly locks in the current API shape, output schema, or implementation details rather than core functionality, mark it with `@pytest.mark.design_lockin`.
- Do not build wrappers, shims, or compatibility layers just to satisfy a `design_lockin` test; those tests are warnings, not architecture requirements.

Bad:

- putting physics formulas directly in `pipelines/`
- putting plotting implementation details directly in `pipelines/`
- large nested control flow and dict plumbing inside pipeline entry points
- grouping results into temporary bundles and unpacking them later inside the pipeline
- recording a synthetic final pipeline summary instead of recording results at the point they are computed
- keeping dead wrappers or compatibility layers only because a pipeline test expects them
- writing detailed behavior tests for pipeline-only glue instead of testing the reusable functions it calls

Preferred pattern:

- move reusable computations to `physics/`
- move reusable plotting primitives to `visualisation/`
- keep pipeline code as composition/orchestration only
- keep one smoke test for the pipeline entrypoint and test the real logic elsewhere

## 0. Library Purity (Hard Boundary)

Rule:

- The library should be **pure**:
  - general
  - reusable
  - composable
- The library should **not** contain functions that are only used once (especially notebook-only helpers/wrappers).

What goes where:

- `starwinds_analysis/...`:
  - reusable analysis logic
  - reusable data/sampling/topology logic
  - general plotting primitives only when genuinely reused
- `examples/*.ipynb` and `examples/*.py`:
  - one-off workflow code
  - direct plotting calls
  - non-reused glue

Litmus test:

- If a function exists only to make one notebook cell shorter, it probably does **not** belong in the library.
- If code is specific to one example narrative and not reused elsewhere, keep it in the notebook/script.

## 1. Hard-Coded Quantity-Specific Plot Functions (Primary Anti-Pattern)

Bad:

- `plot_slice_velocity(...)`
- `plot_slice_magfield(...)`
- `plot_slice_pressure(...)`

Why this is bad:

- explodes API surface area
- duplicates plotting logic
- encourages inconsistent behavior across quantities
- makes new quantities harder to support

Preferred pattern:

- general plotting functions with parameters
- examples pass the field/quantity to plot

Examples of good direction:

- a generic native 2D slice plot helper + field name parameter
- a generic shell scalar lon/lat plot helper + scalar array + label

Rule:

- If the only difference is the field name/label/unit/cmap/scale, do not create a new function.
- Plotting functions (functions that take or create `fig`/`ax`) should not live in `analysis/`.
- `analysis/` is for data/sampling/reduction/orchestration; plotting belongs in notebooks/examples or a clearly separate visualisation layer when truly reusable.
- Physical-quantity names in `analysis` function names are a smell (for example `*_energy_*`, `*_mass_*`, `*_torque_*`) unless the function is truly algorithmically distinct.
- In `analysis`, prefer names that describe the operation/topology/reduction, not the specific physical quantity.

## 2. Notebook Business Logic ("Slop")

Bad:

- notebook cells containing masking/finite checks/colorbar plumbing/field transforms
- notebook-side triangulation and resampling boilerplate
- long defensive logic that makes examples look hard to use

Preferred pattern:

- notebook = load data + call library helper + plot
- move repeated or fiddly logic into library helpers
- keep notebook code short and readable

Important nuance:

- Do **not** move code into helpers unless needed.
- One or two clear lines that teach usage can stay in a notebook.

## Notebook Code (Special Section)

This project treats notebooks as a real part of the workflow, not as disposable output.

Notebook code should optimize for:

- showing how to use the library
- directness
- readability in one screen
- preserving user intent/comments/TODOs

### Notebook Rules (Do)

- Use direct Matplotlib primitives:
  - `ax.pcolormesh(...)`
  - `ax.contour(...)`
  - `ax.tripcolor(...)`
  - `ax.tricontour(...)`
- Reuse the library for what the library is for:
  - data access (`SmartDs`)
  - derived quantities / physics
  - geometry/topology helpers (for example `auto_coords(...)`, `triangles(...)`)
  - sampling/integration primitives
- Prefer one clear plotting call over local mini-configuration patterns.
- Prefer normal local variable names in notebooks (`x_field`, `color_field`, `n_polar`) unless there is a clear reason to mirror Matplotlib examples/standards.
- Keep TODO comments unless they are actually implemented.
- When a TODO is implemented, prefer changing `TODO` -> `DONE` rather than silently deleting it.
- Commit notebooks in stripped ("naked") form: no output blobs, no execution-count churn.
- `nbstripout` is required for this project and should be installed/enabled for local git usage.

### Notebook Rules (Do Not)

- Do not build a mini plotting framework in a notebook (switches, config systems, wrappers).
- Do not add defensive-programming clutter in notebooks (`np.isfinite` checks, masking pipelines, fallback branches) unless the notebook is specifically demonstrating that behavior.
- Do not sprinkle `matplotlib.rcParams[...]` tweaks around notebook cells.
- Do not use `np.asarray(...)` in notebooks. Use the values directly, or `np.array(...)` only when conversion is actually needed.
- Do not replace whole cells and drop user comments/TODOs.
- Do not move simple plotting code into helper files just to "clean up" the notebook.
- Do not use notebook-only orchestration wrappers when direct library calls are clearer.
  Example bad pattern in a notebook:
  - calling a convenience wrapper that computes and plots everything (`quicklook_*_figure(...)`)
    when the notebook is meant to show the explicit workflow
  Preferred:
  - call the analysis functions directly in the notebook
  - then plot with direct Matplotlib primitives

Preferred `rcParams` usage in notebooks:

- avoid it entirely unless it is genuinely useful for readability/reproducibility
- if needed, set it once near the top of the notebook (one small setup cell), not scattered across plotting cells

### Practical Split (Library vs Notebook)

Library:

- compute quantities
- sample/interpolate data
- provide reusable geometry/topology

Notebook:

- choose fields
- call plotting primitives
- arrange figures and labels
- explain what the plot means

### Canonical Example Style

Good notebook style looks like:

- get coordinates/topology from library helpers
- pass arrays directly into Matplotlib
- avoid extra wrappers unless there is repeated complexity

This is preferred over "generalized" notebook code with mode switches and helper indirection.

## 3. String-Hardcoded Units Scattered Through Analysis/Plot Code

Bad:

- unit assumptions spread across many functions/notebooks
- labels like `"Pressure [dyne/cm^2]"` hard-coded in multiple places
- derived analysis depending on non-SI field names directly

Preferred pattern:

- analysis computes in SI (except `R` coordinate convenience)
- raw/non-SI fields are allowed for plotting/inspection
- unit handling and field resolution should be centralized (eventually in `SmartDs`)

Related rule (local variables):

- Do not add unit suffixes to local variable names when the values are already in the
  default/base SI unit for that context.
- Example bad notebook variable names: `b_r_T`, `u_m_s`, `rho_si`
- Prefer plain quantity names (`b_r`, `u`, `rho`) and state units in:
  - field names (`"B_r [T]"`)
  - plot labels/titles
  - a short nearby comment when needed

## 4. Duplicated Field/Unit Resolution Logic (`resolve_*` proliferation)

Bad:

- many helper functions that all perform local field-name candidate lookup and scaling
- explicit `resolve_*` helper usage in analysis/notebook code

Why this is a smell:

- duplicates naming conventions and conversion logic
- makes behavior inconsistent across modules

Preferred direction:

- `SmartDs` should own resolution/canonicalization
- ask `SmartDs` for the quantity you want, in the units you want (SI for analysis)
- let `griblet` + `SmartDs` decide the computation/conversion path
- field resolution API should return data plus parsed unit metadata

Note:

- Existing `resolve_*` functions remain for now, but are marked with TODOs for migration.

Hard rule:

- Do not introduce new `resolve_*` methods/functions.
- Do not call `resolve_*` helpers in new analysis code when `SmartDs` can provide the quantity directly.
- `resolve_*` is the wrong layer: this is what `griblet` + `SmartDs` are for.
- In analysis code, request the physical quantity directly in SI units and let `SmartDs` provide it.

## 5. Duplicated Physical Quantity Definitions

Bad:

- defining the same physical quantity/formula in multiple modules
- one version in analysis code and another slightly different version in a notebook
- multiple names/normalizations for the same quantity without a single source of truth

Examples:

- mass flux (`rho * U_r`) reimplemented in several places
- magnetic pressure / ram pressure / Mach numbers computed ad hoc in plotting code

Why this is bad:

- easy to introduce subtle physics inconsistencies
- bug fixes do not propagate everywhere
- units and sign conventions drift between implementations

Preferred pattern:

- one authoritative implementation per physical quantity (or quantity family)
- plotting functions consume the computed quantity, they do not redefine it
- notebooks call the library and avoid re-deriving physics

Rule:

- If a quantity already exists in the library, reuse it.
- If it does not exist, add it once in the library (in the right module), then reuse it.

## 6. Silent Fallbacks That Hide Important Behavior

Bad:

- code silently switching quantity/unit/source without making it obvious

Preferred pattern:

- explicit raw-display vs SI-diagnostic presets
- obvious parameter names (`scale='log'|'linear'`, `sampling='grid'|'fibonacci'`)
- return metadata/summary where helpful

## 6b. NaN-Tolerance As A Default Coding Style

Bad:

- writing code that is "NaN-tolerant" by default (`nanmean`, `nanmax`, blanket finite masks, silent NaN filtering)
- treating unexpected NaNs as normal and just plotting/integrating around them

Why this is a smell:

- hides bugs in field resolution, resampling, or physics definitions
- makes bad data look like a normal edge case
- spreads defensive clutter through notebooks and library code

Preferred pattern:

- assume valid data unless there is a specific, known reason NaNs can occur
- let failures be visible during development
- handle NaNs only where the behavior is intentional and documented (for example: known undefined coordinates at poles/origin)

Rule:

- NaN-tolerance is a bad smell.
- If NaN handling is needed, make the reason explicit in code/comments at that exact location.

## 6c. Sprinkled Numerical Constants (Especially Physical Constants)

Bad:

- repeating raw numerical constants in multiple files/functions
- inline physical constants (especially `mu_0` / `MU0`) typed directly in formulas

Why this is a smell:

- easy to introduce inconsistent values
- obscures meaning of formulas
- makes refactors/reviews harder because the same constant is hidden in many places

Preferred pattern:

- define constants once in the appropriate deep layer/module
- import and reuse the named constant everywhere else
- prefer descriptive names over anonymous numbers

Rule:

- Do not sprinkle numerical constants around the codebase.
- Physical constants (especially magnetic permeability / `MU0`) must come from a single shared definition.

## 6d. Circular Imports / Reversed Inclusion Paths

Bad:

- circular imports between layers (`analysis <-> physics`, etc.)
- importing "up" or sideways across the intended layer boundary
- fixing cycles with lazy imports instead of fixing the dependency direction

Why this is a smell:

- usually means the layer split is wrong or responsibilities are mixed
- causes fragile import-time behavior and hidden coupling
- encourages "just move an import inside a function" patches instead of architectural fixes

Preferred pattern:

- define a clear one-way dependency direction between layers
- keep lower/deeper layers independent of higher/shallow layers
- move shared primitives to the correct layer instead of cross-importing both ways

Rule:

- Avoid circular imports entirely.
- Reversed inclusion paths are a bad smell and should be fixed at the layer boundary.
- In this project, `analysis` should not import from `physics` (if that import is needed, the split is likely wrong).

## 7. Over-Fragmentation Into Tiny Helpers

Bad:

- creating a helper for every small line of code
- hiding simple plotting calls behind unnecessary wrappers

Preferred pattern:

- make helpers only when they reduce duplication, complexity, or bug risk
- examples should remain readable and instructional

## 7a. Fat `__init__.py` Facade Modules (Too Early)

Bad:

- large `__init__.py` files that re-export many symbols from submodules
- treating package `__init__.py` as a convenience facade before the architecture is stable
- using re-exports to preserve old import paths during refactors

Why this is bad:

- hides ownership of code (where a symbol really lives)
- creates reversed layer dependencies and circular-import pressure
- makes refactors harder because imports appear to work from everywhere
- grows API surface too early, before boundaries are settled

Preferred pattern:

- keep `__init__.py` files empty or nearly empty by default
- import from the owning module directly
- only expose a small curated public surface later, when the design is stable

Rule:

- Early in this project, `__init__.py` should be empty/minimal.
- Do not build convenience facades in `__init__.py`.
- If a re-export is added, it must be justified as a deliberate stable API decision (not refactor glue).

## 7b. `np.asarray(...)` Casting Everywhere

Bad:

- sprinkling `np.asarray(...)` across library, tests, and notebooks as a reflex
- field access patterns like `np.asarray(sds("..."))` when direct arrays already work

Why this is bad:

- adds noise without clarifying intent
- hides whether conversion is actually needed
- makes code look defensive/mechanical

Preferred pattern:

- use values directly when they already behave like arrays
- use `np.array(...)` only when you explicitly need a NumPy array / dtype conversion / copy semantics

Rule:

- Do not use `np.asarray(...)` in this repo.
- If a conversion is required, make that choice explicit with `np.array(...)` (or a more specific conversion).

## 8. Wrapper Bloat / Low-Signal Functions

Bad:

- functions with 2-5 real lines of math wrapped in many lines of:
  - redundant local recasting (`x = np.array(x, dtype=float)`)
  - renaming every argument to a new local with the same meaning
  - re-boxing outputs (`np.array(...)`) when the expression already returns arrays
  - oversized docstrings/comments explaining obvious code

Why this is bad:

- hides the actual formula/operation
- makes the code look defensive and mechanical
- increases maintenance surface without adding behavior
- encourages one-function-per-quantity file sprawl

Preferred pattern:

- write the core expression directly
- keep only the checks/conversions that are actually required
- let wrong inputs fail naturally unless there is a clear user-facing reason to validate
- if the function is only a thin alias around one expression, consider deleting it

Rule:

- Do not pad simple functions with casting/renaming/boxing boilerplate.
- The signal (actual computation) should dominate the function body.

## 8b. Low-Signal `Used by` Docstring Entries

Bad:

- `Used by: <same file>` entries.
- `Used by:` entries that list only tests.
- In `Used by:` docstrings, do not list:
  - tests
  - the same file the function is defined in
- long `Used by:` lists dominated by local/internal call paths.

Why this is bad:

- hides real integration call sites
- gives false confidence about runtime usage
- adds noisy docstring churn with little design value

Preferred pattern:

- `Used by:` should prioritize real external/runtime consumers:
  - other modules
  - pipelines
  - notebooks/scripts that matter for usage
- avoid listing same-file usage and avoid test-only usage in `Used by:`.
- if no external/runtime consumer exists, say so explicitly (for example:
  `Used by: no external runtime call sites found`).

## 9. Per-Function Custom Data Containers

Bad:

- every analysis function invents its own return container/class
- specialized containers tied to one workflow instead of shared data abstractions

Why this is bad:

- creates parallel mini data models
- makes APIs inconsistent and harder to compose
- encourages analysis code to be built around one-off return types

Preferred pattern:

- reuse existing shared data abstractions (`SmartDs`, arrays + explicit metadata, shared structs)
- add small shared metadata objects only when genuinely needed (and reused)
- do not invent a new container just because one function needs a convenient shape

## 10. Bypassing `SmartDs` For Resampling Without A Specific Reason

Bad:

- ad hoc resampling/interpolation code inside analysis functions
- custom sampling paths that do not use `SmartDs.resample(...)`
- multiple resampling implementations with slightly different behavior

Why this is bad:

- duplicates interpolation behavior and edge handling
- makes structured-vs-flat output behavior inconsistent
- increases the chance of silent differences between workflows

Preferred pattern:

- use `SmartDs.resample(...)` as the default resampling path
- preserve output shape (flat point list or structured grid) through `SmartDs`
- only bypass `SmartDs` resampling for a very specific reason, and document that reason in code

## 11. Internal Shims / Compatibility Re-Export Modules

Bad:

- creating shim modules inside this repo just to preserve old import paths during refactors
- files whose only purpose is `from ... import ...` re-exports for internal callers
- "compatibility" modules for code we control in the same library

Why this is bad:

- adds indirection and confusion about ownership
- hides real dependencies and layer direction
- preserves bad structure instead of fixing imports at the call sites
- creates dead modules that linger after refactors

Preferred pattern:

- update internal imports to the owning module directly
- delete moved modules instead of leaving shim files behind
- only add compatibility shims for external/public API migrations (and document them clearly)

Rule:

- Do not add compatibility shim modules for internal library code.
- When refactoring internal module structure, update imports and delete the old module.

## 12. Dictionary Return Bundles (Usually A Smell)

Bad:

- functions returning large ad hoc dictionaries of mixed values
- string-key API bundles used instead of clear return values / shared structures
- returning dicts just to avoid defining a real abstraction or to bundle one workflow step

Why this is bad:

- weakens the API contract (keys drift, typos become runtime bugs)
- encourages wrapper/orchestration style code and string-key plumbing
- makes refactors noisy because call sites depend on many magic keys

Preferred pattern:

- return explicit values / tuples for small local formulas
- return shared abstractions (`SmartDs`, arrays + explicit metadata) for reusable workflows
- keep dicts for truly map-like data (metadata, summaries, serialization/export payloads)

Exception (current project-specific):

- computed quantities that are intentionally stored in BATSRUS-style `aux` metadata may
  reasonably be represented as dictionaries/maps (though this is still an open design choice).

Rule:

- Default to not returning dictionaries from library functions.
- If returning a dict, document why a map-shaped result is the right abstraction.

## Review Checklist (Use Before Adding New Code)

- Is this function general/parameterized, or is it hard-coded to one quantity?
- Is notebook code doing library work?
- Is unit logic centralized or scattered?
- Am I duplicating field-resolution logic?
- Am I redefining a physical quantity/formula that already exists somewhere else?
- Would a user understand this example as "easy"?
- Am I creating a helper because it is needed, or just to move code around?
- Am I sprinkling `rcParams` changes or using unnecessary ALL_CAPS names in a notebook?
- Am I inventing a one-off data container instead of reusing a shared abstraction?
- Am I bypassing `SmartDs.resample(...)` without a specific documented reason?
- Is this pipeline file doing implementation work that belongs in `physics/` or `visualisation/`?
- Is this function mostly boilerplate around 1-3 lines of actual computation?
- Am I adding a shim/re-export module instead of fixing internal imports?
- Am I returning a dict where a clearer return shape should exist?

## Current Priority Enforcement

Highest priority to avoid:

1. Hard-coded quantity-specific plotting functions
2. Notebook business logic/slop
3. Duplicated physical quantity definitions
4. New duplicated `resolve_*` helpers
5. Per-function custom data containers
6. Wrapper bloat / low-signal functions
7. Bypassing `SmartDs` for resampling without a specific reason
8. Internal compatibility shims / re-export modules
9. Ad hoc dictionary return bundles
