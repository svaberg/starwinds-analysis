# starwinds-analysis

Analysis and diagnostics tooling for BATSRUS/SWMF outputs.

## Core stack

- `starwinds_readplt.Dataset` for raw file access
- `SmartDs` for field access, recipe-graph computation, and resampling
- `numpy` / `scipy` / `matplotlib` for analysis and plotting
- optional `pyvista` for 3D workflows only

## Installation

Option 1 (conda environment):

```bash
conda env create -f environment.yml
conda activate starwinds-analysis
```

Option 2 (pip only):

```bash
pip install .
```

Extras:

```bash
pip install ".[tests]"
pip install ".[viz3d]"
```

## SmartDs quickstart

```python
from starwinds_analysis.smart_ds import SmartDs

sds = SmartDs.from_file("sample_data/3d__var_1_n00060000.plt")
sds.prepare()

print(sds)
rho = sds("Rho [kg/m^3]")
br = sds("B_r [T]")
print(sds.explain("B_r [T]"))
```

`SmartDs.prepare()` attaches BATSRUS + spherical recipe fragments.
Default spherical naming is:

- `R [R]`, `polar [rad]`, `azimuth [rad]`
- vector components: `_r`, `_p`, `_a`

## Resampling

```python
import numpy as np

# points shape: (n, 3) in the same coordinate system as coord_fields
points = np.stack([
    np.linspace(1.0, 10.0, 100),
    np.zeros(100),
    np.zeros(100),
], axis=-1)

curve = sds.resample(
    points,
    fields=("Rho [kg/m^3]", "B_r [T]", "R [R]"),
    method="nearest",
)
```

`resample(...)` returns a new wrapped `SmartDs`.

## Pipelines (`sw-pipe`)

`sw-pipe` discovers `.plt` and `.dat` files and routes by filename prefix:

- `3d* -> volume`
- `shl* -> shell`
- `x=0*`, `y=0*`, `z=0* -> slice`

Examples:

```bash
sw-pipe sample_data
sw-pipe sample_data --pipeline slice
sw-pipe sample_data --pipeline shell
sw-pipe sample_data --pipeline volume
```

Inspect recorded results:

```bash
sw-pipe-results --state sample_data/sw-pipe.slice.processed.json --list-fields
sw-pipe-results --state sample_data/sw-pipe.slice.processed.json --field slice_rho_png
```
