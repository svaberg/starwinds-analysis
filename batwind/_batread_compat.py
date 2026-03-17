from __future__ import annotations

from batread.dataset import Dataset


def dataset_variable(dataset, index_or_name):
    """Return a raw field from either old or new reader APIs."""
    variable = getattr(dataset, "variable", None)
    if variable is not None:
        return variable(index_or_name)
    return dataset[index_or_name]


def ensure_dataset_compat():
    """Add a public .variable() alias when batread only exposes __getitem__."""
    if hasattr(Dataset, "variable"):
        return

    def variable(self, index_or_name):
        return self[index_or_name]

    Dataset.variable = variable


ensure_dataset_compat()


__all__ = ["Dataset", "dataset_variable", "ensure_dataset_compat"]
