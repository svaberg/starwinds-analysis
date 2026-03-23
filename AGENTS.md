# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Project character
This is a numerics-oriented scientific codebase.

Priorities, in order:

1. correctness
2. clarity
3. directness
4. performance
5. minimal surface area

This is not a framework, not a plugin system, and not a compatibility layer.

## Project direction
Keep the core centered on:

- `batread.Dataset` for raw BATSRUS/SWMF data access
- `numpy` / `scipy` for analysis, transforms, and resampling
- `pyvista` / `vtk` mainly for plotting and geometry operations

Prefer to keep numerical and analysis logic independent of PyVista/VTK unless the
task is explicitly about visualization or mesh operations.

## General coding bias
Prefer the simplest implementation that satisfies the current requirements.

Do not add:
- shims
- adapters
- wrappers for their own sake
- compatibility layers
- deprecation paths
- fallback implementations
- defensive handling for hypothetical future cases
- defensive numpy casting
- abstraction for unneeded extensibility
- version-dependent branching unless explicitly required
- optional parameters added only for future flexibility
- base classes, registries, factories, or plugin mechanisms unless explicitly requested

Do not "future-proof" the code unless the task explicitly requires it.

Do not preserve old APIs, old argument names, old data formats, or old behaviors
unless the task explicitly requires backward compatibility.

## Rename and Git discipline
- Treat renames as version-control operations, not filesystem churn.
- When renaming tracked files or folders, complete the tracked rename before updating references.
- Verify that Git records renames cleanly instead of delete/add noise.
- Do not make a dirty commit and then a cleanup commit when one clean commit was possible.
- Check staged contents before every commit.
- Do not assume untracked local files belong in the repo. This workspace may contain untracked examples, docs, and sample data.

## Commit discipline
- Before starting a new task, check whether there are tracked changes from the previous task.
- If there are, do not begin the new task until they are either:
  - committed, or
  - explicitly discarded by the user, or
  - explicitly carried forward with the user's approval.
- Do not let tracked changes from one task silently roll into the next task.
- Before every commit, check staged contents with `git diff --cached --stat` and `git diff --cached`.
- Keep commits narrow and topic-specific.
- Do not include untracked files unless the user explicitly asked for them.

## Executing tests, code and notebooks
- If you see an `environment.yml` file, assume that the environment exists and use it when running tests, code, and notebooks.
- In this repo, prefer `conda run -n batwind ...` for repeatable verification.
- After updating notebooks leave them in a run state.

## CI and release discipline
- Before saying a branch is ready, run the same checks that CI runs, not a looser subset.
- In this repo that means at least:
  - `conda run -n batwind flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - `conda run -n batwind flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`
  - `conda run -n batwind pytest --cov=batwind`
  - `conda run -n batwind pytest --cov=batwind --cov-report=xml:pytest-cobertura.xml`
- During important work, especially before pushing or releasing, check that GitHub Actions is green on the exact pushed commit.
- Remove generated test artifacts before finishing work.

## Public-facing text discipline
- Do not publish release notes, package descriptions, Zenodo text, README summaries, or other public-facing copy without treating them as their own writing task.
- Show the exact proposed text before publishing when the text is user-facing or public-facing.
- Say explicitly when text is a rough draft.
- Keep maintainer checklist items out of public release summaries.
- Do not present items like license, tests passing, wheel builds, or metadata cleanup as release highlights.
- Public-facing text should explain:
  - what the package does
  - what changed
  - what users can now do
- Before finalizing public-facing text, do a render/readability pass instead of stopping at the first technically correct draft.

## Notebook hygiene
- Notebooks should demonstrate basic use cases and showcase the library code.
- Notebooks should be committed in unexecuted form; use standard tools and git plugins to manage this.
- When you run notebooks as part of development and testing be sure to leave them in a run state.
- When notebooks generate plots you should check whether they look right.

## Style
Prefer:
- plain functions over class hierarchies
- explicit data flow over implicit state
- direct NumPy-style array operations
- small, composable helpers only when they reduce real duplication
- concrete names tied to the mathematics or algorithm
- straightforward control flow

Avoid:
- one-use indirection
- pass-through helper layers
- thin wrappers around existing functions
- "helper" functions that merely rename an operation
- object-oriented structure where stateless functions are sufficient
- configuration plumbing unless it is required by the task

## Numerics-specific guidance
Assume the code is written for known scientific/numerical use cases, not hostile or arbitrary inputs, unless stated otherwise.

Do not add guards for:
- impossible states that cannot occur under current invariants
- unsupported dtypes not mentioned in the task
- unsupported shapes not mentioned in the task
- hypothetical malformed data unless malformed input handling is part of the task
- hypothetical cross-version behavior differences unless explicitly required

Validate only what is necessary for correctness of the present task.

When checks are needed, prefer:
- clear preconditions
- explicit shape/unit/domain checks
- failures that happen early and noisily

Do not add silent correction, silent fallback, or silent coercion.

## Error handling
Do not use broad exception handling.

Avoid:
- `except Exception`
- retry logic
- fallback branches
- "best effort" behavior
- warning-based recovery for core numerical logic

Prefer:
- specific exceptions
- explicit precondition checks
- immediate failure on unsupported cases

Errors should expose unsupported usage, not hide it.

## Compatibility
Assume the current supported environment only.

Do not add:
- legacy paths
- compatibility aliases
- conditional imports for old environments
- old/new API bridging
- polyfills
- dual implementations for historical reasons
- function-local or deep imports unless they are genuinely required to avoid a real dependency cycle or startup-time problem

If compatibility is genuinely required, it must be explicitly requested by the task.

## API discipline
Keep the public API small.

Do not:
- add new public functions unless they are needed
- add convenience overloads
- add optional flags for speculative use cases
- expose internal plumbing

Prefer changing internals over expanding the public surface.

## Refactoring bias
When editing code, prefer reducing indirection.

Good refactors:
- remove dead branches
- remove unused helpers
- inline one-use wrappers
- collapse unnecessary abstraction
- simplify argument passing
- make data flow more explicit
- keep algorithmic structure close to the mathematics

Bad refactors:
- introducing a layer to "keep options open"
- adding abstractions before a second real use case exists
- extracting helpers that make the code harder to read
- replacing direct numerical code with framework-like structure

## Shortcut discipline
Do not take a weak organizing shortcut silently.

If you are about to add:
- a generic bucket folder
- a vague helper module
- a grab-bag utility
- a convenience abstraction with unclear ownership
- any other "park it here for now" structure

then either:
- ask first, or
- if you already did it, say explicitly afterward:
  - that it was a lazy shortcut
  - what shortcut you took
  - why you took it
  - why it was a bad design move

Do not describe the action only. Explain the design failure plainly.

## Performance
Do not pessimize numerical code for the sake of abstraction.

Prefer:
- contiguous array-oriented operations where appropriate
- avoiding unnecessary allocations
- avoiding unnecessary conversions
- keeping hot paths obvious

Do not add extra layers in hot paths unless they provide a clear benefit.

## Documentation and comments
Comments should explain:
- the mathematical idea
- the algorithmic choice
- units, conventions, indexing, or invariants

Comments should not restate obvious code.

Do not add verbose defensive commentary about hypothetical future adaptations.

## Tests
Tests should target:
- current required behavior
- numerical correctness
- important invariants
- edge cases that are actually relevant to the supported domain

Do not add tests for speculative compatibility behavior that the project does not claim to support.

Tests should use pytest. In general modules should be tested in test modules with the same name.

## When uncertain
Use this decision rule:

If a layer, option, guard, wrapper, compatibility path, or abstraction is not required by the current task, do not add it.

Before finishing, check for:
- unnecessary wrappers
- compatibility code that was not requested
- speculative defensive logic
- optional parameters that are not needed
- abstraction without a present use case
- broad exception handling
- silent fallback behavior
