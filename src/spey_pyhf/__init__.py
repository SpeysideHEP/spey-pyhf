from .manager import PyhfManager
from ._version import __version__

__all__ = ["manager", "__version__"]


def __dir__():
    return ["__version__"]


manager = PyhfManager()
