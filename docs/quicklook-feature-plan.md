# Quicklook Feature Plan (2D, Shells, Analytics Only)

## Scope

This document plans the migration of the most-used non-3D parts of the old
`starwinds_batplotlib.quicklook.py` workflow into this repo.

In scope:

- 2D slice quicklook plots
- shell/spherical analytics (radial profiles and integrals)
- analytical/local estimates

Out of scope for this plan:

- all 3D plotting and 3D geometry workflows (isosurfaces, streamlines, marching cubes, volumetric rendering)
- any hard dependency on `vtk` / `pyvista`

## Design Constraints (Current Direction)

- `SmartDs` is the access layer for BATSRUS data (`starwinds_readplt.Dataset`).
- Derived quantities should be computed on demand.
- `griblet` should provide recipe/path resolution for derived fields.
- Base SI should be the internal target as early as practical.
- Core quicklook/analysis features should use `numpy` / `scipy` / `matplotlib`, not VTK.

## Progress (Current)

Implemented from this plan so far (non-3D, NumPy/SciPy only):

- spherical shell sampling + area-weighted integration primitives
- Fibonacci-sphere shell quadrature now used as the default first-pass sampler for mass-loss/torque/open-flux/energy shell integrals (grid sampler retained for axisymmetric flux)
- weighted statistics helpers
- wind mass-loss profile on spherical shells (`dotM(r)`)
- wind torque profile on spherical shells (magnetic + dynamic + total)
- open magnetic flux and axisymmetric open-flux fraction profiles
- energy flux profile on spherical shells
- a first `quicklook2d` wrapper (slice presets/overlays + shell summary figure)
- pure local analytical estimate helpers (mass loss / torque formulas + summaries)
- circular-orbit sampling wrappers + local-vs-shell comparison helpers
- pure-NumPy Kepler/elliptic orbit sampling wrappers + local-vs-shell comparison helpers (time-weighted summaries)
- orbit pressure-component analytics on sampled paths (thermal/magnetic/ram + stand-off proxy) with quicklook plotting wrappers
- orbit-surface (surface-of-revolution) pressure/standoff analytics on 3D files without VTK/PyVista
- orbit-surface (surface-of-revolution) torque diagnostics (`T1..T4` + total) on 3D files without VTK/PyVista
- `run_quicklook2d` support for orbit-surface pressure/torque figures and bundle export (JSON/NPZ summaries + PNGs)
- named-planet orbit presets (Mercury/Venus/Earth/Mars) and direct quicklook runner support for orbit/orbit-surface diagnostics
- generic explicit-surface torque integrator core (points/normals/areas; non-VTK), validated against spherical-shell torque
- modernized radial "monster" histogram quicklook mode (`hist2d` radius-vs-value maps)
- shell summary export helpers (`JSON` + `NPZ`) and radius/scatter quicklook wrappers
- one-shot `quicklook2d` runner for batch figure generation and bundle export
- structured XZ slice resampling (3D -> 2D quad grid) to support 2D slice quicklooks on 3D BATSRUS outputs
- weighted shell-band summary helpers (mean/std/quantiles over selected radius ranges)
- open-wind magnetisation (`Upsilon_open`) scaling helper and quicklook plotting support (old quicklook-style analytical diagnostic)
- local-vs-shell orbit comparison plots (mass loss and torque) in `quicklook2d`
- orbit local-vs-shell comparison summary export (`JSON` + `NPZ`) in quicklook bundles

These are first-pass shell implementations intended to be short/readable and easy to extend.

## What Is Already Usable in the New Codebase

The following pieces already exist and can be used as building blocks for a new quicklook module.

### Data / Derived Fields

- `SmartDs` wrapper with raw passthrough and on-demand fields
- optional `griblet`-based field resolution (`SmartDs.explain(...)` available)
- spherical coordinate fields on demand (`R`, `theta`, `phi`)
- spherical vector components on demand (`B_r`, `U_r`, etc.)
- BATSRUS-oriented SI normalization and some derived fields (`B`, `U`, `c_s`, `c_A`, `M_A`)

### 2D Plotting Utilities (Matplotlib)

- XZ slice tripcolor plots with marginals
- XZ slice tripcolor plots with cross-cut quantiles
- XZ slice plots with marginal scatter points
- XZ slice plots with marginal quantiles by unique coordinates
- cumulative histograms
- scatter vs radius
- binned plots vs radius

### Spherical Sampling Primitives

- `fibonacci_sphere(...)`
- `PolarAzimuthalGrid` with spherical surface cell areas
- point resampling via `SmartDs.resample(...)` (returns a new wrapped dataset)

## Old `quicklook.py` Feature Inventory (Non-3D Only) and Migration Status

The table below maps high-use non-3D features from the old script to the new repo status.

| Old quicklook feature | Status in new repo | Notes |
| --- | --- | --- |
| 2D scalar slice plots (`Rho`, `B_r`, `U_r`, `ti`, `te`, `MA`, etc.) | Implemented (v1 wrapper) | `quicklook2d` preset wrapper + one-shot runner + structured XZ resampling from 3D datasets |
| 2D contour overlays (`B_r=0`, `Ma=1`, `MA=1`, `beta=1`) | Implemented (v1 wrapper) | Optional Matplotlib `tricontour` overlays with preset defaults |
| Scatter plots vs height/radius | Implemented (v1 wrapper) | `quicklook2d` radius quicklook wrapper added (scatter/binned/CDF modes) |
| Radial “monster” histogram plots | Implemented (modernized v1) | `hist2d` radius-vs-value histogram quicklook mode added; not a byte-for-byte port of the old volume-weighted style |
| Shell mass-loss integral (`add_integral_mass`) | Implemented (v1, Fibonacci default) | Spherical-shell profile API added (NumPy/SciPy, no Tecplot/VTK) |
| Shell torque integral (`add_integral_momentum`) | Implemented (v1, Fibonacci default) | Spherical-shell magnetic/dynamic/total profile API added |
| Shell open magnetic flux (`add_integral_open_flux`) | Implemented (v1, Fibonacci default) | Signed and unsigned/open flux shell profiles added |
| Shell energy flux (`add_integral_energy`) | Implemented (v1, Fibonacci default) | Uses `E * U_r` shell integration |
| Axisymmetric open flux + fraction | Implemented (v1, grid sampler) | Azimuthal-mean `B_r` shell diagnostic added; kept on grid sampler because it requires explicit azimuthal structure |
| Generic surface torque (`surface_torque` / `integrate_surface_torque`) | Implemented (explicit-surface v1) | Non-VTK explicit-surface torque term/integral engine added for sampled points+normals+areas, with spherical-shell and orbit-surface workflows; automatic surface extraction remains deferred |
| Weighted summary stats / quantiles | Implemented (v1) | Includes shell-band weighted summaries and quantiles in quicklook JSON export |
| Local analytical mass-loss estimate (`local_massloss_estimate`) | Implemented (v2) | Core formula + circular and Kepler/elliptic orbit sampling wrappers + shell comparison helper |
| Local analytical torque estimate (`local_torque_estimate`) | Implemented (v2) | Core formula + circular and Kepler/elliptic orbit sampling wrappers + shell comparison helper |
| Orbit pressure-component diagnostics (`orbit_surface_ram_pressure`-adjacent workflow) | Implemented (v3) | Local-path and orbit-surface (surface-of-revolution) thermal/magnetic/ram (+relative ram) and stand-off proxy, plus named-planet preset support, no Tecplot/VTK |
| Orbit-surface torque diagnostics (new non-VTK workflow) | Implemented (v1) | Surface-of-revolution torque terms (`T1..T4` + total) via explicit surface points/normals/areas, plus quicklook plotting wrappers |
| Shell summary persistence (`.p` pickle aux outputs) | Implemented (redesigned v1) | `quicklook2d` bundle export writes shell summaries to JSON/NPZ |

## High-Priority Feature Plan

Wind mass loss and wind torque are the first-class targets.

### Priority 1: Wind Mass Loss on Spherical Shells

Target capability:

- Compute `dotM(r)` over a list of shell radii from a `SmartDs`
- Plot `dotM` vs height/radius
- Return numeric arrays for scripting/tests

Initial formulation (spherical shells):

- `dotM(r) = ∮ rho * U_r * dA`

Implementation approach (no VTK):

- build shell sampling helper using `PolarAzimuthalGrid` (preferred) and/or `fibonacci_sphere`
- resample source fields to shell points using `SmartDs.resample(...)`
- compute quadrature weights from shell cell areas
- integrate `rho * U_r` with SI fields from `SmartDs` + `griblet`

Required fields (SI target):

- `Rho [kg/m^3]`
- `U_r [m/s]` (or `U_xyz` + spherical conversion fallback)

Validation:

- compare shell profile behavior against old `batplotlib` quicklook outputs for known examples
- use old `batplotlib` integral tests as formula references (especially mass-flux tests)
- test radius-independence in quasi-steady regions (within tolerance)

### Priority 2: Wind Torque on Spherical Shells

Target capability:

- Compute wind torque vs shell radius
- Return magnetic, dynamic, and total torque components
- Produce the old-style torque profile plot (2D Matplotlib)

Start with spherical-shell formulation (no generic arbitrary surface yet):

- magnetic torque term (Vidotto-style shell form)
- dynamic torque term
- total torque = magnetic + dynamic

Why start here:

- It delivers the main scientific quicklook output without requiring 3D surface extraction.
- It avoids Tecplot-specific normal-vector machinery.

Required fields (SI target):

- `Rho [kg/m^3]`
- `B_r [T]`, `B_phi [T]` (or equivalent azimuthal component convention)
- `U_r [m/s]`, `U_phi [m/s]`
- `X [m]`, `Y [m]`, `Z [m]` (or `R [m]`, `theta`, `phi`) for cylindrical radius `varpi`

Notes:

- The old quicklook also computed a four-component generic surface torque (`surface_torque` / `integrate_surface_torque`).
- For this plan, implement spherical-shell torque first.
- Generic-surface torque surface-extraction workflows can be a later extension after the shell path is stable.
  - The non-VTK explicit-surface torque integrator is implemented; remaining work is automatic surface extraction (a VTK/Tecplot-adjacent concern).

Validation:

- compare against old `batplotlib` torque routines on shared example datasets
- use old `batplotlib/tests/test_torque.py` as a reference for formula decomposition and expected behavior
- test consistency across a radius range beyond the acceleration region (case-dependent tolerance)

## 2D Slice Quicklook Plan (No 3D Overlays)

The new quicklook entry point should be a thin orchestration layer built on existing plotting helpers.

### Features to Include Early

- XZ slice scalar map for requested field
- optional marginal summaries (mean / quantiles / scatter)
- preset plot bundles matching common old quicklook use:
  - density (`Rho`)
  - radial magnetic field (`B_r`)
  - radial velocity (`U_r`)
  - electron/ion temperature (`te`, `ti`)
  - Alfvén Mach (`M_A`)
- optional contour overlays:
  - `B_r = 0`
  - `M_A = 1`
  - `Ma = 1` (after sonic Mach recipe exists)
  - `beta = 1` (after plasma-beta recipe exists)

### Features to Defer

- Tecplot-style streamtraces in 2D quicklook
- any 3D slice widgets or VTK-backed slicing

## Shell/Radial Analytics Plan (Beyond Mass Loss and Torque)

These are valuable and should follow soon after the mass-loss/torque core is stable.

### Near-Term (after mass loss + torque)

- open magnetic flux profile vs radius
- axisymmetric open flux and axisymmetric fraction
- energy flux profile vs radius
- weighted summary stats and quantiles over shell radius sets

### Next Wave

- radial histogram “monster” plots (volume-weighted)
- modernized shell summary output (JSON/NPZ + plots)

## Analytical / Local Estimate Plan

These were useful in old quicklook and can be ported cleanly as pure analysis utilities.

### Planned Features

- local mass-loss estimate from sampled local `rho`, `u`, and orbital radius:
  - `4 * pi * r^2 * rho * u`
- local torque estimate from local samples (including comparison to shell torque)
- quantile summaries for local estimates

### Dependencies

- orbital/path sampling helper (separate from 3D visualization)
  - Implemented (v2): circular + pure-NumPy Kepler/elliptic orbit samplers
- shell mass-loss and shell torque implementations (for comparison diagnostics)
- weighted quantile/statistics utilities

## Proposed New Module Layout (Non-3D Quicklook)

This keeps physics, sampling, and plotting separate.

- `starwinds_analysis/analysis/shells.py`
  - shell grid construction
  - shell sampling from `SmartDs`
  - area weights / quadrature
  - generic shell integral helper
- `starwinds_analysis/analysis/mass_loss.py`
  - `mass_loss_vs_radius(...)`
  - local mass-loss estimates
- `starwinds_analysis/analysis/torque.py`
  - spherical-shell torque components and totals
  - torque-vs-radius helpers
- `starwinds_analysis/analysis/stats.py`
  - weighted mean/std
  - weighted quantiles
- `starwinds_analysis/quicklook2d.py`
  - orchestration only
  - preset plot bundles
  - saves figures / tabular summaries

## Delivery Phases

### Phase 1 (Immediate): Shell Core + Mass Loss

- add weighted stats helpers
- add shell sampler + area-weighted integration
- implement `mass_loss_vs_radius(...)`
- implement shell mass-loss plot helper
- tests on example `.plt` data in `/sample_data`

### Phase 2 (Immediate Next): Shell Torque

- implement spherical-shell magnetic/dynamic/total torque profile
- implement torque profile plot helper
- compare against old torque formulas/results on reference cases

### Phase 3: Shell Diagnostics Package

- open flux
- axisymmetric flux + fraction
- energy flux
- shell summary outputs (JSON/NPZ)

### Phase 4: 2D Quicklook Presets

- preset slice variables
- contour overlays
- scatter/radial summaries
- one command/function for batch quicklook figure generation (2D only)

Status:
- Implemented (v1): wrappers for slice presets/overlays, radius summaries, shell summary figure, and a one-shot `quicklook2d` runner with optional bundle export.

### Phase 5: Analytical Local Estimates

- local mass-loss estimate
- local torque estimate
- shell-vs-local comparison plots/summary stats

Status:
- Implemented (v2): pure local formula helpers, circular + Kepler/elliptic orbit sampling wrappers, shell comparison summaries, local-vs-shell comparison plots, and orbit comparison summary export in `quicklook2d`.

## Definition of Success for the New "Quicklook" (Non-3D)

The migration is successful when the new quicklook can, without Tecplot and without VTK/PyVista:

- generate standard 2D XZ slice figures for core wind variables
- compute and plot wind mass loss vs radius
- compute and plot wind torque vs radius (magnetic + dynamic + total)
- compute open flux and key shell diagnostics
- produce reproducible numerical outputs in SI units for scripting and tests
