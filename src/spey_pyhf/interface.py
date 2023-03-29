"""pyhf plugin for spey interface"""

from typing import Optional, List, Text, Union, Callable, Tuple
import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase
from .pyhfdata import PyhfData, PyhfDataWrapper
from .utils import objective_wrapper
from ._version import __version__
from . import manager

__all__ = ["PyhfInterface"]


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

    name: Text = "pyhf"
    version: Text = __version__
    author: Text = "SpeysideHEP"
    spey_requires: Text = "0.0.1"
    doi: List[Text] = ["10.5281/zenodo.1169739", "10.21105/joss.02823"]
    datastructure = PyhfDataWrapper

    __slots__ = ["_model"]

    def __init__(self, model: PyhfData):
        assert isinstance(model, PyhfData), "Invalid statistical model."
        self._model = model
        self.manager = manager

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

        asimov_data = self.manager.pyhf.infer.calculators.generate_asimov_data(
            1.0 if test_statistics == "q0" else 0.0,
            data,
            model,
            model.config.suggested_init(),
            model.config.suggested_bounds(),
            model.config.suggested_fixed(),
            return_fitted_pars=False,
        )

        return asimov_data

    def get_logpdf_func(
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

        return lambda pars: model.logpdf(
            pars, self.manager.pyhf.tensorlib.astensor(data_org if data is None else data)
        )[0]

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:

        logpdf = self.get_logpdf_func(expected, data)

        if self.manager.backend == "jax":
            hess = self.manager.backend_accessor.hessian(logpdf)

            def func(pars: np.ndarray) -> np.ndarray:
                """Compute hessian of logpdf"""
                pars = self.manager.backend_accessor.numpy.array(pars)
                return np.array(hess(pars))

            return func

        raise NotImplementedError(f"Hessian is not available in {self.manager.backend} backend")

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]:

        if do_grad and not self.manager.grad_available:
            raise NotImplementedError(
                f"Gradient is not available for {self.manager.backend} backend."
            )

        _, model, data_org = self.model(expected=expected)

        return objective_wrapper(
            data=data_org if data is None else data, pdf=model, do_grad=do_grad
        )

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        """
        Initialize a sampling function with the statistical model

        :param pars (`np.ndarray`): nuisance parameters
        :return `Callable[[int], np.ndarray]`: returns function to sample from
            a preconfigured statistical model
        """
        _, model, _ = self.model()
        pdf = model.make_pdf(self.manager.pyhf.tensorlib.astensor(pars))

        def sampler(number_of_samples: int) -> np.ndarray:
            """
            Sample generator for the statistical model

            :param number_of_samples (`int`): number of samples to be drawn from the model
            :return `np.ndarray`: Sampled observations
            """
            return np.array(pdf.sample((number_of_samples,)))

        return sampler
