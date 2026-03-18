# batwind

3D analysis and visualisation of BATSRUS output.

## Direction

This repo is moving toward a lightweight analysis core built around:

- `batread.Dataset` (raw BATSRUS/SWMF data access)
- `numpy` / `scipy` (analysis, transforms, resampling)

The goal is to avoid a large "precomputed quantity" workflow and instead compute
derived fields on demand.

## `SmartDs` (experimental wrapper)

`batwind.smart_ds.SmartDs` wraps a `batread.Dataset` and adds:

- raw field passthrough (`.variable(name)`, `sds(name)`)
- graph-based derived fields via `griblet`
- point resampling that returns a **new wrapped dataset**

### Example

```python
from batwind.smart_ds import SmartDs

sds = SmartDs.from_file("examples/3d__var_1_n00000000.plt")

# Attach spherical geometry + vector-component recipes
sds.add_spherical_graph(vectors=("B", "U"))

r = sds.variable("R [R]")
br = sds.variable("B_r [Gauss]")
uphi = sds.variable("U_phi [km/s]")
```

You can inspect the recipe path for any derived field:

```python
print(sds.explain("B_r [Gauss]"))
```

For BATSRUS-style inputs, `SmartDs` can also attach a graph for common unit
normalization (to SI where possible) and a few derived quantities:

```python
sds.add_batsrus_graph()

rho = sds.variable("Rho [kg/m^3]")
bx = sds.variable("B_x [T]")
c_s = sds.variable("c_s [m/s]")
m_a = sds.variable("M_A [none]")
print(sds.explain("M_A [none]"))
```

Spherical conventions used by the helper recipes:

- `theta [rad]`: colatitude in `[0, pi]`
- `phi [rad]`: azimuth from `atan2(y, x)` in `[-pi, pi]`
- singular points (e.g. `r=0`, polar axis for some components) produce `NaN`
