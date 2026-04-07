import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from batread import Dataset
from batwind.visualisation.histograms import plot_binned_vs_radius
from batwind.visualisation.histograms import plot_cumulative_hists
from batwind.visualisation.histograms import plot_vs_radius
from batwind.visualisation.slice import auto_coords
from batwind.visualisation.slice import plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords
from batwind.visualisation.slice import plot_xz_slice_with_marginal_points
from batwind.visualisation.slice import triangles
from scipy.interpolate import LinearNDInterpolator  # This is much slower but should be used for generating the final figures.

from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
import re
k_B = 1.380649e-23      # J/K
amu = 1.66053906660e-27  # kg


def temperature_K(ds):
    # Pressure: nPa → Pa
    P = ds("P [nPa]") * 1e-9

    # Mass density: amu/cm^3 → kg/m^3
    rho = ds("Rho [amu/cm^3]") * amu * 1e6

    # Number density: n = rho / m_p   (assume hydrogen plasma)
    n = rho / amu

    # Temperature
    T = P / (n * k_B)

    return T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2D quicklook plots for Earth XUV neutrals BATSRUS outputs."
    )
    parser.add_argument(
        "run_root",
        type=Path,
        help="BATSRUS run directory containing GM/IO2 output files.",
    )
    parser.add_argument(
        "--pattern",
        default="GM/IO2/y=0_var_1_n*.dat",
        help="Glob pattern relative to run_root for slice files.",
    )
    return parser.parse_args()


def output_label_from_io2_path(path):
    parts = path.parts
    for i in range(len(parts) - 1):
        if i > 0 and parts[i] == "GM" and parts[i + 1] == "IO2":
            return parts[i - 1]
    return path.name or "run"


args = parse_args()
run_root = args.run_root.expanduser().resolve()
pattern = run_root / args.pattern
output_dir = Path(__file__).resolve().parent / output_label_from_io2_path(pattern.parent)
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)


def extract_index(p):
    m = re.search(r"_n(\d+)(?:\D|$)", p.name)
    return int(m.group(1)) if m else -1


def sort_key(p):
    m = re.search(r"_n(\d+)(?:\D|$)", p.name)
    if not m:
        return (0, -1)
    num_str = m.group(1)
    num = extract_index(p)
    trailing_zeros = len(num_str) - len(num_str.rstrip("0"))
    return (-trailing_zeros, num)


files = list(pattern.parent.glob(pattern.name))

if not files:
    raise SystemExit(f"No files matched pattern: {pattern}")


files_sorted = sorted(files, key=extract_index)


files_sorted = sorted(files, key=sort_key)


def good_files(files):
    for file in files:
        try:
            ds = Dataset.from_file(str(file))
            yield file
        except ValueError as e:
            print(f"Error reading file {file}: {e}")
            continue


files_sorted = list(good_files(files_sorted))

fig, axs = plt.subplots(2, 2, figsize=(10, 8))


timesteps = [extract_index(file) for file in files_sorted]
cmap = LinearSegmentedColormap.from_list("two_color", ["tab:blue", "tab:red"])
norm = Normalize(vmin=min(timesteps), vmax=max(timesteps))
for file, ts in zip(files_sorted, timesteps):
    ds = Dataset.from_file(str(file))
    color = cmap(norm(ts))
    plot_cumulative_hists(ds, axs, color=color)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, location="right")
cbar.set_label("Timestep")
plt.savefig("planet_cumulative_hists.png")
plt.close(fig)


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
timesteps = [extract_index(file) for file in files_sorted]
cmap = LinearSegmentedColormap.from_list("two_color", ["tab:blue", "tab:red"])
norm = Normalize(vmin=min(timesteps), vmax=max(timesteps))
for file, ts in zip(files_sorted, timesteps):
    ds = Dataset.from_file(str(file))
    color = cmap(norm(ts))
    plot_vs_radius(ds, axs, color=color)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, location="right")
cbar.set_label("Timestep")
plt.savefig("planet_vs_radius.png")
plt.close(fig)


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
timesteps = [extract_index(file) for file in files_sorted]
cmap = LinearSegmentedColormap.from_list("two_color", ["tab:blue", "tab:red"])
norm = Normalize(vmin=min(timesteps), vmax=max(timesteps))
for file, ts in zip(files_sorted, timesteps):
    ds = Dataset.from_file(str(file))
    color = cmap(norm(ts))
    plot_binned_vs_radius(ds, axs, color=color)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, location="right")
for ax in axs.ravel():
    ax.set_yscale("log")
cbar.set_label("Timestep")
plt.savefig("planet_vs_radius_binned.png")
plt.close(fig)


for file in files_sorted:

    # Look for the noclobber file. if it exist we contine.
    noclobber_file = output_dir / f"planet_{extract_index(file):08d}_processed.txt"
    if noclobber_file.exists():
        print(f"Skipping already processed file {file}")
        continue

    try:
        ds = Dataset.from_file(str(file))
        print(ds)
    except ValueError as e:
        print(f"Error reading file {file}: {e}")
        continue

    i = extract_index(file)

    fig, (ax_main, ax_left, ax_bottom), cbar = plot_xz_slice_with_marginal_points(ds)

    plt.savefig(f"planet_{i:08d}_slice.png")
    plt.close()

    fig, (ax_main, ax_left, ax_bottom), cbar = plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords(ds)
    plt.savefig(f"planet_{i:08d}_slice_quantiles.png")
    plt.close()

    xuvflux = float(ds.aux.get('XuvFluxSi', 0))
    useHeatingSource = float(ds.aux.get('UseHeatingSource', 0)) > 0

    s = f"flux {xuvflux:.2e} useHeatingSource {useHeatingSource}"

    tris = triangles(ds)

    _, ax = plt.subplots()
    w_name = "XUVTAU [none]"
    w_var = ds.variable(w_name)
    if np.max(w_var) <= 0:
        norm = None
    else:
        norm = LogNorm()
    # import pdb; pdb.set_trace()
    img = ax.tripcolor(tris, w_var, shading="flat", norm=norm, cmap="grey")
    cax = plt.colorbar(img)

    # ax.set_xlim(-6, 6)
    # ax.set_ylim(-6, 6)
    plt.title(f"XUVTAU at time step {i} {s}")
    plt.savefig(f"planet_{i:08d}_tau.png")
    plt.close()

    fig, ax = plt.subplots()
    w_name = "XUVHEAT [W/m^3]"
    w_var = ds.variable(w_name)
    if np.max(w_var) <= 0:
        norm = None
    else:
        norm = LogNorm()
    img = ax.tripcolor(tris, w_var, shading="flat", norm=norm, cmap="inferno")
    cax = plt.colorbar(img)
    plt.title(f"xuvheat at time step {i} {s}")
    plt.savefig(f"planet_{i:08d}_xuvheat.png")
    plt.close()

    _, ax = plt.subplots()
    w_name = "Rho [amu/cm^3]"
    w_var = ds.variable(w_name)
    img = ax.tripcolor(tris, w_var, shading="flat", norm="log")
    cax = plt.colorbar(img)
    plt.title(f"Rho at time step {i} {s}")
    plt.savefig(f"planet_{i:08d}_rho.png")
    plt.close()

    _, ax = plt.subplots()
    T = temperature_K(ds)
    img = ax.tripcolor(tris, T, shading="flat", norm="log")
    cax = plt.colorbar(img)
    plt.title(f"Temperature at time step {i} {s}")
    plt.savefig(f"planet_{i:08d}_temperature.png")
    plt.close()

    _, ax = plt.subplots()
    w_name = "NH [1/m^3]"
    w_var = ds.variable(w_name)
    img = ax.tripcolor(tris, w_var, shading="flat", norm="log")
    cax = plt.colorbar(img)
    plt.title(f"NH at time step {i} {s}")
    plt.savefig(f"planet_{i:08d}_nh.png")
    plt.close()

    _, ax = plt.subplots()
    w_name = "NE [1/m^3]"
    w_var = ds.variable(w_name)
    img = ax.tripcolor(tris, w_var, shading="flat", norm="log")
    cax = plt.colorbar(img)
    plt.title(f"NE at time step {i} {s}")
    plt.savefig(f"planet_{i:08d}_ne.png")
    plt.close()

    _ax = plt.subplots()
    w_name = "NHP [1/m^3]"
    w_var = ds.variable(w_name)
    img = ax.tripcolor(tris, w_var, shading="flat", norm="log")
    cax = plt.colorbar(img)
    plt.title(f"NHP at time step {i} {s}")
    plt.savefig(f"planet_{i:08d}_nhp.png")
    plt.close()

    # Write a noclobber file to indicate that this time step has been processed
    noclobber_file = output_dir / f"planet_{i:08d}_processed.txt"
    noclobber_file.touch(exist_ok=False)


# Vectors
magfield = np.stack([ds.variable("B_%s [nT]" % i) for i in "xyz"], axis=-1)
magfield_mag = np.linalg.norm(magfield, axis=-1)
_, ax = plt.subplots()
img = ax.tripcolor(tris, magfield_mag, shading="gouraud", norm="log")
cax = plt.colorbar(img)

# Add quiver plot of the magnetic field vectors use the triplot coordinates for the quiver plot
ax.quiver(tris.x, tris.y,
          magfield[..., 0]/magfield_mag, magfield[..., 2]/magfield_mag, color='white', scale=50, width=0.002)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)


plt.savefig("planet_magfield.png")
plt.close()


# Vectors
velfield = np.stack([ds.variable("U_%s [km/s]" % i) for i in "xyz"], axis=-1)
velfield_mag = np.linalg.norm(velfield, axis=-1)
_, ax = plt.subplots()
img = ax.tripcolor(tris, velfield_mag, shading="gouraud", cmap="cividis")
cax = plt.colorbar(img)


# Add quiver plot of the magnetic field vectors use the triplot coordinates for the quiver plot
ax.quiver(tris.x, tris.y,
          velfield[..., 0]/velfield_mag, velfield[..., 2]/velfield_mag, color='white', scale=50, width=0.002)

lim = 30
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

plt.savefig("planet_velocities.png")
plt.close()


print("Making sc interpolator...")
_points = np.stack([ds(name) for name in auto_coords(ds)], axis=-1)
_data = np.stack([ds(name) for name in ds.variables], axis=-1)
sc = LinearNDInterpolator(_points, _data)

X, Y = np.meshgrid(np.linspace(-40, 20, 500), np.linspace(-30, 30, 500))
# X, Y = np.meshgrid(np.linspace(np.min(ds.variable("X [R]")), np.max(ds.variable("X [R]")), 100),
#                    np.linspace(np.min(ds.variable("Z [R]")), np.max(ds.variable("Z [R]")), 100))
scd = sc(X, Y)
variable_name = 'Rho [amu/cm^3]'
plt.pcolormesh(X, Y, scd[..., ds.variables.index(variable_name)], shading='gouraud', norm=LogNorm())
plt.savefig("planet_rho_interpolated.png")
plt.close()
print("Making sc interpolator complete.")


fig, ax = plt.subplots()
_bfield = scd[..., [ds.variables.index("B_%s [nT]" % i) for i in "xyz"]]


def vector_field_cartesian_to_polar(x, y, bx, by):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    br = (x*bx + y*by) / r
    btheta = (x*by - y*bx) / r
    return r, theta, br, btheta


R, _, _bfield_r, _ = vector_field_cartesian_to_polar(X, Y, _bfield[..., 0], _bfield[..., 2])

_bfield_mag = np.linalg.norm(_bfield, axis=-1)


norm = LogNorm(vmin := np.min(_bfield_mag[R > 1]), vmax=1e2 * vmin)
print(str(norm))

img = ax.pcolormesh(X, Y, _bfield_mag, shading='gouraud', norm=norm, cmap="viridis")
cax = plt.colorbar(img)
# ax.quiver(X, Y,
#           _bfield[...,0]/np.linalg.norm(_bfield, axis=-1), _bfield[...,2]/np.linalg.norm(_bfield, axis=-1), color='white', scale=50, width=0.002)

ax.streamplot(X, Y, _bfield[..., 0], _bfield[..., 2], color='white', density=1.5, linewidth=0.5)
plt.savefig("planet_magfield_interpolated.png", dpi=1200)
plt.close()


fig, ax = plt.subplots()
_vfield = scd[..., [ds.variables.index("U_%s [km/s]" % i) for i in "xyz"]]


def vector_field_cartesian_to_polar(x, y, bx, by):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    br = (x*bx + y*by) / r
    btheta = (x*by - y*bx) / r
    return r, theta, br, btheta


R, _, _vfield_r, _ = vector_field_cartesian_to_polar(X, Y, _vfield[..., 0], _vfield[..., 2])

_vfield_mag = np.linalg.norm(_vfield, axis=-1)


norm = LogNorm(vmin := np.min(_vfield_mag[R > 1]), vmax=1e2 * vmin)
print(str(norm))

img = ax.pcolormesh(X, Y, _vfield_mag, shading='gouraud', norm="linear", cmap="viridis")
cax = plt.colorbar(img)
# ax.quiver(X, Y,
#           _vfield[...,0]/np.linalg.norm(_vfield, axis=-1), _vfield[...,2]/np.linalg.norm(_vfield, axis=-1), color='white', scale=50, width=0.002)

ax.streamplot(X, Y, _vfield[..., 0], _vfield[..., 2], color='black', density=1.5, linewidth=0.5)
plt.savefig("planet_velocities_interpolated.png", dpi=1200)
plt.close()
