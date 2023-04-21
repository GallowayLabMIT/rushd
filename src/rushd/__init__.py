"""## rushd: data management for humans.

Collection of helper modules for maintaining
robust, reproducible data management.
"""
from . import flow, io, plot  # noqa
from .io import infile, outfile  # noqa

submodules = ["io", "flow", "plot"]

re_exports = [
    "infile",
    "outfile",
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
