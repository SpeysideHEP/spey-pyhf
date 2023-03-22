"""pyhf plugin for spey interface"""

from typing import Optional, List, Text, Union, Callable
import logging, pyhf
import numpy as np

from pyhf.infer.calculators import generate_asimov_data

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase
from .utils import twice_nll_func
from .pyhfdata import PyhfData, PyhfDataWrapper
from ._version import __version__

__all__ = ["PyhfInterface"]

pyhf.pdf.log.setLevel(logging.CRITICAL)
pyhf.workspace.log.setLevel(logging.CRITICAL)
pyhf.set_backend("numpy", precision="64b")


class PyhfInterface(BackendBase):
    """
    pyhf Interface.

    :param model (`PyhfData`): contains all the information regarding the regions, yields
    :raises AssertionError: if the input type is wrong.

    .. code-block:: python3

        >>> from spey.backends.pyhf_backend.data import SLData
        >>> from spey.backends.pyhf_backend.interface import PyhfInterface
        >>> from spey import ExpectationType
        >>> background = {
        >>>   "channels": [
        >>>     { "name": "singlechannel",
        >>>       "samples": [
        >>>         { "name": "background",
        >>>           "data": [50.0, 52.0],
        >>>           "modifiers": [{ "name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0]}]
        >>>         }
        >>>       ]
        >>>     }
        >>>   ],
        >>>   "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
        >>>   "measurements": [{"name": "Measurement", "config": { "poi": "mu", "parameters": []} }],
        >>>   "version": "1.0.0"
        >>> }
        >>> signal = [{"op": "add",
        >>>     "path": "/channels/0/samples/1",
        >>>     "value": {"name": "signal", "data": [12.0, 11.0],
        >>>       "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]}}]
        >>> model = SLData(signal=signal, background=background)
        >>> statistical_model = PyhfInterface(model=model, xsection=1.0, analysis="my_analysis")
        >>> print(statistical_model)
        >>> # StatisticalModel(analysis='my_analysis', xsection=1.000e+00 [pb], backend=pyhf)
        >>> statistical_model.exclusion_confidence_level()
        >>> # [0.9474850257628679] # 1-CLs
        >>> statistical_model.s95exp
        >>> # 1.0685773410460155 # prefit excluded cross section in pb
        >>> statistical_model.maximize_likelihood()
        >>> # (-0.0669277855002002, 12.483595567080783) # muhat and maximum negative log-likelihood
        >>> statistical_model.likelihood(poi_test=1.5)
        >>> # 16.59756909879556
        >>> statistical_model.exclusion_confidence_level(expected=ExpectationType.aposteriori)
        >>> # [0.9973937390501324, 0.9861799464393675, 0.9355467946443513, 0.7647435613928496, 0.4269637940897122]
    """

    name = "pyhf"
    version = __version__
    author = "SpeysideHEP"
    spey_requires = "0.0.1"
    datastructure = PyhfDataWrapper

    __slots__ = ["_model"]

    def __init__(self, model: PyhfData):
        assert isinstance(model, PyhfData), "Invalid statistical model."
        self._model = model

    @property
    def model(self) -> PyhfData:
        """Retrieve statistical model"""
        return self._model

    def generate_asimov_data(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> np.ndarray:
        """
        Method to generate Asimov data for given statistical model

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param test_statistics (`Text`, default `"qtilde"`): definition of test statistics. `q`, `qtilde` or `q0`
        :raises `NotImplementedError`: if the method has not been implemented
        :return ` Union[List[float], np.ndarray]`: Asimov data
        """
        _, model, data = self.model(expected=expected)

        asimov_data = generate_asimov_data(
            1.0 if test_statistics == "q0" else 0.0,
            data,
            model,
            model.config.suggested_init(),
            model.config.suggested_bounds(),
            model.config.suggested_fixed(),
            return_fitted_pars=False,
        )

        return asimov_data

    def get_twice_nll_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Generate function to compute twice negative log-likelihood for the statistical model

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
        :raises `NotImplementedError`: If the method is not implemented
        :return `Callable[[np.ndarray], float]`: function to compute twice negative log-likelihood for given nuisance parameters.
        """
        # CHECK THE MODEL BOUNDS!!
        # POI Test needs to be adjusted according to the boundaries for sake of convergence
        # see issue https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579235311
        # comment https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579299831
        # NOTE During tests we observed that shifting poi with respect to bounds is not needed.
        _, model, data_org = self.model(expected=expected)

        return twice_nll_func(model, data if data is not None else data_org)

    def get_gradient_twice_nll_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        # Currently not implemented
        return None
