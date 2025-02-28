"""Manager for pyhf integration"""
import importlib
import logging
from collections.abc import Callable

from spey.system.exceptions import MethodNotAvailable

log = logging.getLogger("Spey")


class PyhfManager:
    """Manager to unify the usage of pyhf through out the package"""

    pyhf = importlib.import_module("pyhf")

    def __init__(self):
        if importlib.util.find_spec("jax") is not None:
            PyhfManager.pyhf.set_backend("jax", precision="64b")
            log.debug("pyhf backend set to Jax")
        else:
            PyhfManager.pyhf.set_backend("numpy", precision="64b")
            log.debug("pyhf backend set to NumPy")

        PyhfManager.pyhf.pdf.log.setLevel(logging.CRITICAL)
        PyhfManager.pyhf.workspace.log.setLevel(logging.CRITICAL)

        self.shim = importlib.import_module("pyhf.optimize.common").shim

    def __repr__(self) -> str:
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
    def available_backends(self) -> list[str]:
        """
        Retreive the names of available backends

        Returns:
            ``List[Text]``:
            list of available backends
        """
        return [
            bd for bd in ["numpy", "tensorflow", "jax"] if importlib.util.find_spec(bd)
        ]

    @property
    def backend(self) -> str:
        """Retreive current backend name"""
        return PyhfManager.pyhf.tensorlib.name

    @backend.setter
    def backend(self, backend: str) -> None:
        """
        Modify pyhf backend.

        Args:
            backend (`Text`): backend type. see pyhf for details
        """
        if backend not in self.available_backends:
            raise MethodNotAvailable(f"{backend} backend currently not available.")
        log.debug(f"setting pyhf backend to {backend}")
        PyhfManager.pyhf.set_backend(backend)

    @property
    def backend_accessor(self):
        """access to current pyhf backend"""
        return importlib.import_module(self.backend)

    def jit(self, function: Callable) -> Callable:
        """Jit the given function if available"""
        if self.backend == "jax":
            return self.backend_accessor.jit(function)

        return function
