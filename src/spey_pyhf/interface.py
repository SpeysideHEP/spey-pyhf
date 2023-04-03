"""pyhf plugin for spey interface"""

from typing import Optional, List, Text, Union, Callable, Tuple, Dict
import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase
from spey.base import ModelConfig
from .pyhfdata import PyhfDataWrapper, PyhfData
from .utils import objective_wrapper
from ._version import __version__
from . import manager

__all__ = ["PyhfInterface"]


class PyhfInterface(BackendBase):
    """
    pyhf Interface. For details on input structure please see
    `this link <https://pyhf.readthedocs.io/en/v0.7.0/likelihood.html>`_

    Args:
        signal_yields (``Union[List[Dict], float, List[float]]``): Signal yields can be given in
          three different structure:

          * ``List[Dict]``: This is a ``JSONPATCH`` type of input.
          * ``float``: single float type of input will be used to create a single bin
            statistical model. This input expects both ``data`` and ``background_yields``
            inputs are ``float`` as well.
          * ``List[float]]``: This input type will be used to create an uncorrelated multi-bin
            statistical model. This input expects both ``data`` and ``background_yields`` inputs
            are ``List[float]]`` as well.

        data (``Union[Dict, float, List[float]]]``): Data input can be given in three different forms:

          * ``Dict``: This input is expected to be background only ``JSON`` based statistical
            model input. Please see the details from the link above.
          * ``float``: single float type of input will be used to create a single bin
            statistical model. This input expects both ``signal_yields`` and ``background_yields``
            inputs are ``float`` as well.
          * ``List[float]]``: This input type will be used to create an uncorrelated multi-bin
            statistical model. This input expects both ``signal_yields`` and ``background_yields``
            inputs are ``List[float]]`` as well.

        background_yields (``Union[float, List[float]]``, default ``None``): If ``data`` and
          ``signal_yields`` inputs are ``float`` or ``List[float]`` type, this input will be used
          to set the SM background yields in the statistical model. Not used when ``signal_yields``
          and ``data`` are in ``JSON`` format.
        absolute_background_unc (``Union[float, List[float]]``, default ``None``): If ``data`` and
          ``signal_yields`` inputs are ``float`` or ``List[float]`` type, this input will be used
          to set the absolute uncertainties in the SM background. Not used when ``signal_yields``
          and ``data`` are in ``JSON`` format.

    Example:

    .. code_block:: python3
        :linenos:

        >>> import spey

        >>> background_only = {
        ...     "channels": [
        ...         {
        ...             "name": "singlechannel",
        ...             "samples": [
        ...                 {
        ...                     "name": "background",
        ...                     "data": [50.0, 52.0],
        ...                     "modifiers": [
        ...                         {
        ...                             "name": "uncorr_bkguncrt",
        ...                             "type": "shapesys",
        ...                             "data": [3.0, 7.0],
        ...                         }
        ...                     ],
        ...                 }
        ...             ],
        ...         }
        ...     ],
        ...     "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
        ...     "measurements": [{"name": "Measurement", "config": {"poi": "mu", "parameters": []}}],
        ...     "version": "1.0.0",
        ... }
        >>> signal = [
        ...     {
        ...         "op": "add",
        ...         "path": "/channels/0/samples/1",
        ...         "value": {
        ...             "name": "signal",
        ...             "data": [12.0, 11.0],
        ...             "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
        ...         },
        ...     }
        ... ]
        >>> statistical_model = spey.get_correlated_nbin_statistical_model(
        ...     analysis="simple_pyhf",
        ...     data=background_only,
        ...     signal_yields=signal,
        ... )
        >>> statistical_model.exclusion_confidence_level() # [0.9474850259721279]
    """

    name: Text = "pyhf"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = "0.0.1"
    """Spey version required for the backend"""
    doi: List[Text] = ["10.5281/zenodo.1169739", "10.21105/joss.02823"]
    """Citable DOI for the backend"""

    __slots__ = ["_model", "manager"]

    def __init__(
        self,
        signal_yields: Union[List[Dict], float, List[float]],
        data: Union[Dict, float, List[float]],
        background_yields: Optional[Union[float, List[float]]] = None,
        absolute_background_unc: Optional[Union[float, List[float]]] = None,
    ):
        self._model = PyhfDataWrapper(
            signal=signal_yields,
            background=data,
            nb=background_yields,
            delta_nb=absolute_background_unc,
            default_expectation=ExpectationType.observed,
            name="pyhf_model",
        )
        self.manager = manager
        """pyhf Manager to handle the interface with pyhf"""

    @property
    def model(self) -> PyhfData:
        """Retreive statistical model container"""
        return self._model

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``40.0``): upper bound for parameter
              of interest, :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """
        return self.model.config(
            allow_negative_signal=allow_negative_signal, poi_upper_bound=poi_upper_bound
        )

    def generate_asimov_data(
        self,
        poi_asimov: float = 0.0,
        expected: ExpectationType = ExpectationType.observed,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Backend specific method to generate Asimov data.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            kwargs: keyword arguments for the optimiser.

        Returns:
            ``List[float]``:
            Asimov data.
        """
        _, model, data = self.model(expected=expected)

        par_bounds = [
            *(kwargs.get("par_bounds", None) or model.config.suggested_bounds())
        ]
        init_pars = [*(kwargs.get("init_pars", None) or model.config.suggested_init())]

        asimov_data = self.manager.pyhf.infer.calculators.generate_asimov_data(
            poi_asimov,
            data,
            model,
            init_pars,
            par_bounds,
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

        raise NotImplementedError(
            f"Hessian is not available in {self.manager.backend} backend"
        )

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
