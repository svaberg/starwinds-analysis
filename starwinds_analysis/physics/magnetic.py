"""THIS FILE contains magnetic display-unit helpers.

It should stay small and only hold generic magnetic-unit presentation helpers.
Pointwise magnetic spherical components should come from SmartDs/griblet recipes.
"""

from __future__ import annotations

def magnetic_field_unit_scale(unit: str) -> tuple[float, str]:
    key = str(unit).strip()
    table = {
        "T": (1.0, "T"),
        "Tesla": (1.0, "T"),
        "G": (1e4, "G"),
        "Gauss": (1e4, "G"),
        "nT": (1e9, "nT"),
    }
    if key not in table:
        raise ValueError(f"Unsupported magnetic display unit '{unit}'")
    return table[key]
