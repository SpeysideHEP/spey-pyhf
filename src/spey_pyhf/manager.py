"""Manager for pyhf integration"""
from typing import Text, List
import logging, importlib

from spey.system.exceptions import MethodNotAvailable


class PyhfManager:
    """Manager to unify the usage of pyhf through out the package"""

    pyhf = importlib.import_module("pyhf")

    def __init__(self):
        PyhfManager.pyhf.set_backend("numpy", precision="64b")

        PyhfManager.pyhf.pdf.log.setLevel(logging.CRITICAL)
        PyhfManager.pyhf.workspace.log.setLevel(logging.CRITICAL)

        self.shim = importlib.import_module("pyhf.optimize.common").shim
        self.backend_accessor = importlib.import_module("numpy")

    def __repr__(self) -> Text:
        return (
            f"pyhfManager(pyhf_backend='{self.backend}',"
            + f" available_backends={self.available_backends})"
        )

    @property
    def grad_available(self) -> bool:
        """
        Returns true or false depending on the availablility
        of the gradient within the current backend.
        """
        return self.backend != "numpy"

    @property
    def available_backends(self) -> List[Text]:
        """
        Retreive the names of available backends

        :return `List[Text]`: list of available backends
        """
        return [bd for bd in ["numpy", "tensorflow", "jax"] if importlib.util.find_spec(bd)]

    @property
    def backend(self) -> Text:
        """Retreive current backend name"""
        return PyhfManager.pyhf.tensorlib.name

    @backend.setter
    def backend(self, backend: Text) -> None:
        """
        Modify pyhf backend.

        :param backend (`Text`): backend type. see pyhf for details
        """
        if backend not in self.available_backends:
            raise MethodNotAvailable(f"{backend} backend currently not available.")
        PyhfManager.pyhf.set_backend(backend)
        self.backend_accessor = importlib.import_module(backend)