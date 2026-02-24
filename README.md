# 3D analysis and visualisation of BATSRUS output
Free and open source analysis with the `pyvista` package!

## Direction

This repo is moving toward a lightweight analysis core built around:

- `starwinds_readplt.Dataset` (raw BATSRUS/SWMF data access)
- `numpy` / `scipy` (analysis, transforms, resampling)
- optional `pyvista` / `vtk` only for 3D plotting / geometry operations

The goal is to avoid a large "precomputed quantity" workflow and instead compute
derived fields on demand.

## `SmartDs` (experimental wrapper)

`starwinds_analysis.smart_ds.SmartDs` wraps a `starwinds_readplt.Dataset` and adds:

- raw field passthrough (`.variable(name)`, `sds(name)`)
- on-demand computed fields via registered functions
- optional `griblet` computation-graph resolution (when `griblet` is installed)
- point resampling that returns a **new wrapped dataset**

### Example

```python
from starwinds_analysis.smart_ds import SmartDs

sds = SmartDs.from_file("examples/3d__var_1_n00000000.plt")

# Register on-demand spherical geometry + vector components from Cartesian fields
sds.add_spherical_fields(vectors=("B", "U"))

r = sds.variable("R [R]")
br = sds.variable("B_r [Gauss]")
uphi = sds.variable("U_phi [km/s]")
```

If `griblet` is installed, you can also attach spherical *recipes* (dependency-path
resolution) instead of local field functions:

```python
sds.add_spherical_graph(vectors=("B", "U"))
print(sds.explain("B_r [Gauss]"))
```

Spherical conventions used by the helper recipes:

- `theta [rad]`: colatitude in `[0, pi]`
- `phi [rad]`: azimuth from `atan2(y, x)` in `[-pi, pi]`
- singular points (e.g. `r=0`, polar axis for some components) produce `NaN`
