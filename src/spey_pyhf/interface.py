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


def __dir__():
    return __all__


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

    .. code-block:: python3
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

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return self.model.isAlive

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

    def expected_data(self, pars: List[float]) -> List[float]:
        r"""
        Compute the expected value of the statistical model

        Args:
            pars (``List[float]``): nuisance parameters, :math:`\theta` and
              parameter of interest, :math:`\mu`.

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        return self.model._model.expected_data(pars)

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Generate function to compute :math:`\log\mathcal{L}(\mu, \theta)` where :math:`\mu` is the
        parameter of interest and :math:`\theta` are nuisance parameters.

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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and computes
            :math:`\log\mathcal{L}(\mu, \theta)`.
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
        r"""
        Currently Hessian of :math:`\log\mathcal{L}(\mu, \theta)` is only used to compute
        variance on :math:`\mu`. This method returns a callable function which takes fit
        parameters (:math:`\mu` and :math:`\theta`) and returns Hessian.

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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and
            returns Hessian of :math:`\log\mathcal{L}(\mu, \theta)`.
        """

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
        r"""
        Objective function is the function to perform the optimisation on. This function is
        expected to be twice negative log-likelihood, :math:`-2\log\mathcal{L}(\mu, \theta)`.
        Additionally, if available it canbe bundled with the gradient of twice negative log-likelihood.

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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit
            do_grad (``bool``, default ``True``): If ``True`` return objective and its gradient
              as ``tuple`` (subject to availablility) if ``False`` only returns objective function.

        Returns:
            ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
            Function which takes fit parameters (:math:`\mu` and :math:`\theta`) and returns either
            objective or objective and its gradient.
        """

        if do_grad and not self.manager.grad_available:
            raise NotImplementedError(
                f"Gradient is not available for {self.manager.backend} backend."
            )

        _, model, data_org = self.model(expected=expected)

        return objective_wrapper(
            data=data_org if data is None else data, pdf=model, do_grad=do_grad
        )

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Retreives the function to sample from.

        Args:
            pars (:obj:`np.ndarray`): fit parameters (:math:`\mu` and :math:`\theta`)

        Returns:
            ``Callable[[int], np.ndarray]``:
            Function that takes ``number_of_samples`` as input and draws as many samples
            from the statistical model.
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
