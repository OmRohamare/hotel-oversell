"""Load the client's scoring contract."""

from importlib import import_module


def load_contract():
    """Return the original client contract module."""
    return import_module("contract")
