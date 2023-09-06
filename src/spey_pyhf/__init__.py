from .manager import PyhfManager
from .helper_functions import WorkspaceInterpreter
from ._version import __version__

__all__ = ["manager", "__version__", "WorkspaceInterpreter"]


def __dir__():
    return ["__version__", "WorkspaceInterpreter"]


manager = PyhfManager()
