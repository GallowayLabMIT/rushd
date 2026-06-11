"""``rushd``: data management for humans.

Collection of helper modules for maintaining
robust, reproducible data management.
"""

from . import flow, check, io, plot, qpcr, ddpcr  # noqa
from .io import infile, outfile  # noqa
from .check import sanity_check  # noqa

submodules = ["io", "check", "flow", "plot", "qpcr", "ddpcr"]

re_exports = [
    "infile",
    "outfile",
    "sanity_check",
]
# Re-exports of common functions loaded from submodules
__all__ = submodules + re_exports


# Re-export datadir and rootdir
def __getattr__(name: str):
    """Set up the module attribute exports."""
    if name == "datadir":
        return io.datadir
    if name == "rootdir":
        return io.rootdir
    raise AttributeError(f"No attribute {name} in rushd")
