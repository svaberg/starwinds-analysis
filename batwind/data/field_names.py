"""Field-name parsing helpers shared across analysis modules."""

import logging

log = logging.getLogger(__name__)


DEFAULT_XYZ_NAMES = ("X [R]", "Y [R]", "Z [R]")


def unit_from_brackets(name: str) -> str | None:
    """
    Extract the unit token from a bracketed field name like `X [R]`.
    Used by: `batwind/analysis/shells.py`
    """
    text = str(name)
    i = text.rfind("[")
    j = text.rfind("]")
    if i == -1 or j == -1 or j <= i:
        log.warning("unit_from_brackets: no bracketed unit token in '%s'", text)
        return None
    out = text[i + 1 : j].strip() or None
    if out is None:
        log.warning("unit_from_brackets: empty bracketed unit token in '%s'", text)
    return out
