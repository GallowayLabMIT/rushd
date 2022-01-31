"""
## rushd
Collection of helper modules for maintaining
robust, reproducible data management.
"""
from . import io
from .io import infile, outfile

# Re-exports of common functions loaded from submodules
__all__ = ['infile', 'outfile']

# Re-export datadir and rootdir
def __getattr__(name: str):
    """
    Sets up the module attribute exports.
    """
    if name == 'datadir':
        return io.datadir
    if name == 'rootdir':
        return io.rootdir
    raise AttributeError(f"No attribute {name} in rushd")
