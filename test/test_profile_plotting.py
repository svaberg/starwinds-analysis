import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.visualisation.profile_plots import plot_shell_height_series
from starwinds_analysis.visualisation.profile_plots import shell_profile_height


def test_shell_profile_height_uses_height_or_radius():
    p_height = {"height [R]": [0.0, 1.0, 2.0]}
    p_radius = {"radius [R]": [1.0, 2.0, 3.0]}
    np.testing.assert_allclose(shell_profile_height(p_height), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(shell_profile_height(p_radius), [0.0, 1.0, 2.0])


def test_plot_shell_height_series_negative_mirror_and_labels():
    profile = {
        "height [R]": np.array([0.0, 1.0, 2.0]),
        "demo [u]": np.array([2.0, -3.0, 4.0]),
    }

    fig, ax = plt.subplots()
    try:
        plot_shell_height_series(
            ax,
            profile,
            "demo [u]",
            label="demo",
            ylabel="Demo [u]",
            show_negative=True,
        )
        assert len(ax.lines) == 2
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [2.0, -3.0, 4.0])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [-2.0, 3.0, -4.0])
        assert ax.get_xlabel() == "Height over surface [R]"
        assert ax.get_ylabel() == "Demo [u]"
    finally:
        plt.close(fig)
