"""Field-name parsing helpers shared across analysis modules."""


def unit_from_brackets(name: str) -> str | None:
    """
    Extract the unit token from a bracketed field name like `X [R]`.
    Used by: `starwinds_analysis/analysis/shells.py`
    """
    text = str(name)
    i = text.rfind("[")
    j = text.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    return text[i + 1 : j].strip() or None
