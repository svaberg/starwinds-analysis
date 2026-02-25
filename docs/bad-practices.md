# Bad Practices (Anti-Patterns) To Avoid

This is a living list of patterns we want to avoid in `starwinds-analysis`.

The goal is not "purity". The goal is:

- short readable code
- reusable general APIs
- notebooks/examples that demonstrate easy usage
- SI-first analysis (with explicit exceptions like `R` and Gauss-for-plotting)

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
- Keep TODO comments unless they are actually implemented.
- When a TODO is implemented, prefer changing `TODO` -> `DONE` rather than silently deleting it.

### Notebook Rules (Do Not)

- Do not build a mini plotting framework in a notebook (switches, config systems, wrappers).
- Do not add defensive-programming clutter in notebooks (`np.isfinite` checks, masking pipelines, fallback branches) unless the notebook is specifically demonstrating that behavior.
- Do not replace whole cells and drop user comments/TODOs.
- Do not move simple plotting code into helper files just to "clean up" the notebook.

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

## 4. Duplicated Field/Unit Resolution Logic (`resolve_*` proliferation)

Bad:

- many helper functions that all perform local field-name candidate lookup and scaling

Why this is a smell:

- duplicates naming conventions and conversion logic
- makes behavior inconsistent across modules

Preferred direction:

- `SmartDs` should own resolution/canonicalization
- field resolution API should return data plus parsed unit metadata

Note:

- Existing `resolve_*` functions remain for now, but are marked with TODOs for migration.

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

## 7. Over-Fragmentation Into Tiny Helpers

Bad:

- creating a helper for every small line of code
- hiding simple plotting calls behind unnecessary wrappers

Preferred pattern:

- make helpers only when they reduce duplication, complexity, or bug risk
- examples should remain readable and instructional

## Review Checklist (Use Before Adding New Code)

- Is this function general/parameterized, or is it hard-coded to one quantity?
- Is notebook code doing library work?
- Is unit logic centralized or scattered?
- Am I duplicating field-resolution logic?
- Am I redefining a physical quantity/formula that already exists somewhere else?
- Would a user understand this example as "easy"?
- Am I creating a helper because it is needed, or just to move code around?

## Current Priority Enforcement

Highest priority to avoid:

1. Hard-coded quantity-specific plotting functions
2. Notebook business logic/slop
3. Duplicated physical quantity definitions
4. New duplicated `resolve_*` helpers
