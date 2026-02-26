# Bad Practices (Anti-Patterns) To Avoid

This is a living list of patterns we want to avoid in `starwinds-analysis`.

The goal is not "purity". The goal is:

- short readable code
- reusable general APIs
- notebooks/examples that demonstrate easy usage
- SI-first analysis (with explicit exceptions like `R` and Gauss-for-plotting)

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

## 7. Over-Fragmentation Into Tiny Helpers

Bad:

- creating a helper for every small line of code
- hiding simple plotting calls behind unnecessary wrappers

Preferred pattern:

- make helpers only when they reduce duplication, complexity, or bug risk
- examples should remain readable and instructional

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

## 8. Per-Function Custom Data Containers

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

## 9. Bypassing `SmartDs` For Resampling Without A Specific Reason

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

## Current Priority Enforcement

Highest priority to avoid:

1. Hard-coded quantity-specific plotting functions
2. Notebook business logic/slop
3. Duplicated physical quantity definitions
4. New duplicated `resolve_*` helpers
5. Per-function custom data containers
6. Bypassing `SmartDs` for resampling without a specific reason
