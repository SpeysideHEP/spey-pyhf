from .manager import PyhfManager

__all__ = ["manager"]


def __dir__():
    return __all__


manager = PyhfManager()
