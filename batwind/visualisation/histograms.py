import logging

import numpy as np
from matplotlib.colors import LogNorm

log = logging.getLogger(__name__)


def plot_cumulative_hists(
    ds,
    axes,
    fields=(
        "Rho [amu/cm^3]",
        "Hp [amu/cm^3]",
        "H [amu/cm^3]",
        "FNEUTRAL [none]",
    ),
    bins=200,
    range=None,
    color=None,
    ylabel="Cumulative fraction",
):
    """
    Cumulative histogram (CDF line) for each field on the provided axes.
    """
    axes = np.asarray(axes).ravel()
    if axes.size < len(fields):
        raise ValueError("Not enough axes for number of fields")
    log.debug("plot_cumulative_hists fields=%d bins=%s", len(fields), bins)

    for i, (ax, field) in enumerate(zip(axes, fields)):
        x = np.asarray(ds[field]).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue

        if range is None:
            lo, hi = float(np.min(x)), float(np.max(x))
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            r = (lo, hi)
        else:
            r = range

        counts, edges = np.histogram(x, bins=bins, range=r)
        cdf = np.cumsum(counts).astype(float)
        if cdf[-1] > 0:
            cdf /= cdf[-1]

        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, cdf, color=color)
        ax.set_ylim(0.0, 1.0)

        ax.set_xlabel(field)
        if i == 0:
            ax.set_ylabel(ylabel)


def plot_vs_radius(
    ds,
    axes,
    fields=(
        "Rho [amu/cm^3]",
        "Hp [amu/cm^3]",
        "H [amu/cm^3]",
        "FNEUTRAL [none]",
    ),
    color=None,
    s=1.0,
    alpha=0.3,
):
    log.debug("plot_vs_radius fields=%d", len(fields))
    axes = np.asarray(axes).ravel()
    if axes.size < len(fields):
        raise ValueError("Not enough axes for number of fields")

    X = np.asarray(ds["X [R]"]).ravel()
    Y = np.asarray(ds["Y [R]"]).ravel()
    Z = np.asarray(ds["Z [R]"]).ravel()
    r = np.sqrt(X**2 + Y**2 + Z**2)

    rmask = np.isfinite(r)
    r = r[rmask]

    for ax, field in zip(axes, fields):
        f = np.asarray(ds[field]).ravel()[rmask]
        mask = np.isfinite(f)

        ax.scatter(r[mask], f[mask], s=s, color=color, alpha=alpha)
        ax.set_title(field)
        ax.set_xlabel("r [R]")


def plot_binned_vs_radius(
    ds,
    axes,
    fields=(
        "Rho [amu/cm^3]",
        "Hp [amu/cm^3]",
        "H [amu/cm^3]",
        "FNEUTRAL [none]",
    ),
    bins=200,
    range=None,
    color=None,
    statistic="mean",   # "mean", "median", or "sum"
):
    log.debug("plot_binned_vs_radius fields=%d bins=%s statistic=%s", len(fields), bins, statistic)
    axes = np.asarray(axes).ravel()
    if axes.size < len(fields):
        raise ValueError("Not enough axes for number of fields")

    X = np.asarray(ds["X [R]"]).ravel()
    Y = np.asarray(ds["Y [R]"]).ravel()
    Z = np.asarray(ds["Z [R]"]).ravel()

    r = np.sqrt(X**2 + Y**2 + Z**2)
    mask = np.isfinite(r)
    r = r[mask]

    if r.size == 0:
        return

    if range is None:
        lo, hi = float(r.min()), float(r.max())
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        r_range = (lo, hi)
    else:
        r_range = range

    edges = np.linspace(r_range[0], r_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.digitize(r, edges) - 1
    valid = (bin_index >= 0) & (bin_index < bins)

    for ax, field in zip(axes, fields):
        f = np.asarray(ds[field]).ravel()[mask]
        fmask = np.isfinite(f) & valid

        y = np.full(bins, np.nan)

        if statistic in ("mean", "sum"):
            sums = np.bincount(bin_index[fmask], weights=f[fmask], minlength=bins)
            counts = np.bincount(bin_index[fmask], minlength=bins)

            if statistic == "sum":
                y = sums
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    y[counts > 0] = sums[counts > 0] / counts[counts > 0]

        elif statistic == "median":
            for b in range(bins):
                vals = f[(bin_index == b) & fmask]
                if vals.size > 0:
                    y[b] = np.median(vals)
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'sum'")

        ax.plot(centers, y, color=color)
        ax.set_title(field)
        ax.set_xlabel("r [R]")


def plot_radial_hist2d(
    ds,
    axes,
    fields=(
        "Rho [amu/cm^3]",
        "Hp [amu/cm^3]",
        "H [amu/cm^3]",
        "FNEUTRAL [none]",
    ),
    bins=(128, 128),
    radius_range=None,
    value_range=None,
    weights=None,
    normalize: str | None = None,
    cmap="magma",
    log_color: bool = True,
):
    """
    2D histogram (radius vs field value) as a compact replacement for old "monster" plots.

    `normalize="per_radius"` scales each radial bin to unit sum, which makes the
    distribution shape visible even when density changes strongly with radius.
    """
    axes = np.asarray(axes).ravel()
    if axes.size < len(fields):
        raise ValueError("Not enough axes for number of fields")
    log.debug("plot_radial_hist2d fields=%d bins=%s normalize=%s", len(fields), bins, normalize)

    X = np.asarray(ds["X [R]"]).ravel()
    Y = np.asarray(ds["Y [R]"]).ravel()
    Z = np.asarray(ds["Z [R]"]).ravel()
    r = np.sqrt(X**2 + Y**2 + Z**2)
    rmask = np.isfinite(r)
    r = r[rmask]
    if r.size == 0:
        return

    if radius_range is None:
        rlo, rhi = float(np.nanmin(r)), float(np.nanmax(r))
        if rlo == rhi:
            rlo -= 0.5
            rhi += 0.5
        r_range = (rlo, rhi)
    else:
        r_range = radius_range

    if weights is None:
        w_all = None
    elif isinstance(weights, str):
        w_all = np.asarray(ds[weights]).ravel()[rmask]
    else:
        w_all = np.asarray(weights, dtype=float).ravel()[rmask]

    if isinstance(bins, int):
        bins = (int(bins), int(bins))
    r_bins, v_bins = int(bins[0]), int(bins[1])

    for ax, field in zip(axes, fields):
        f = np.asarray(ds[field]).ravel()[rmask]
        mask = np.isfinite(r) & np.isfinite(f)
        if w_all is not None:
            mask &= np.isfinite(w_all)
            w = w_all[mask]
        else:
            w = None
        rr = r[mask]
        ff = f[mask]
        if rr.size == 0:
            continue

        if value_range is None:
            vlo, vhi = float(np.nanmin(ff)), float(np.nanmax(ff))
            if vlo == vhi:
                vlo -= 0.5
                vhi += 0.5
            f_range = (vlo, vhi)
        else:
            f_range = value_range

        H, r_edges, f_edges = np.histogram2d(
            rr,
            ff,
            bins=(r_bins, v_bins),
            range=(r_range, f_range),
            weights=w,
        )
        H = np.asarray(H, dtype=float)

        if normalize == "per_radius":
            colsum = H.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                H = np.divide(H, colsum, out=np.zeros_like(H), where=colsum > 0)
        elif normalize not in (None, "count"):
            raise ValueError("normalize must be None, 'count', or 'per_radius'")

        plot_H = H.T
        positive = plot_H > 0
        norm = None
        if log_color and np.any(positive):
            norm = LogNorm(vmin=float(np.nanmin(plot_H[positive])), vmax=float(np.nanmax(plot_H[positive])))

        mesh = ax.pcolormesh(r_edges, f_edges, plot_H, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(field)
        ax.set_xlabel("r [R]")
        ax.set_ylabel(field)
        ax.figure.colorbar(mesh, ax=ax, pad=0.01)
