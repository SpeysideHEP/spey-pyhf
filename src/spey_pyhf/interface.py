"""pyhf plugin for spey interface"""

import copy
import warnings
from typing import Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np
from spey.base import ModelConfig
from spey.base.backend_base import BackendBase
from spey.utils import ExpectationType

from . import manager
from ._version import __version__
from .data import Base, FullStatisticalModelData, SimpleModelData
from .utils import objective_wrapper

__all__ = ["UncorrelatedBackground", "FullStatisticalModel"]


def __dir__():
    return __all__


class ModelNotDefined(Exception):
    """Undefined model exception"""


class CombinationError(Exception):
    """Combination of the models are not possible"""


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
    """

    name: Text = "pyhf.base"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = ">=0.1.9,<0.2.0"
    """Spey version required for the backend"""
    doi: List[Text] = ["10.5281/zenodo.1169739", "10.21105/joss.02823"]
    """Citable DOI for the backend"""

    __slots__ = ["_model", "manager"]

    def __init__(self):
        self.manager = manager
        """pyhf Manager to handle the interface with pyhf"""

    @property
    def model(self) -> Base:
        """Retreive statistical model container"""
        if hasattr(self, "_model"):
            return self._model
        raise ModelNotDefined("Statistical model is not defined.")

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
        return self.model()[1].expected_data(pars)

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
                return np.array(hess(self.manager.pyhf.tensorlib.astensor(pars)))

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
        params = self.manager.pyhf.tensorlib.astensor(pars)

        def sampler(number_of_samples: int, include_auxiliary: bool = True) -> np.ndarray:
            """
            Sample generator for the statistical model

            Args:
                number_of_samples (``int``): number of samples to be drawn from the model
                include_auxiliary (``bool``, default ``True``): wether or not to include
                    auxiliary data coming from the constraint model.

            Returns:
                ``np.ndarray``:
                generated samples
            """
            if include_auxiliary:
                pdf = model.make_pdf(params)
            else:
                pdf = model.main_model.make_pdf(params)

            return np.array(pdf.sample((number_of_samples,)))

        return sampler


class UncorrelatedBackground(PyhfInterface):
    """
    This backend initiates ``pyhf.simplemodels.uncorrelated_background``, forming an uncorrelated
    histogram structure with given inputs.

    Args:
        signal_yields (``List[float]``): signal yields
        background_yields (``List[float]``): background yields
        data (``List[float]``): observations
        absolute_uncertainties (``List[float]``): absolute uncertainties on the background
    """

    name: Text = "pyhf.uncorrelated_background"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = PyhfInterface.spey_requires
    """Spey version required for the backend"""
    doi: List[Text] = PyhfInterface.doi
    """Citable DOI for the backend"""

    def __init__(
        self,
        signal_yields: List[float],
        background_yields: List[float],
        data: List[int],
        absolute_uncertainties: List[float],
    ):
        super().__init__()
        self._model = SimpleModelData(
            signal_yields, background_yields, data, absolute_uncertainties
        )


class FullStatisticalModel(PyhfInterface):
    """
    pyhf Interface. For details on input structure please see
    `this link <https://pyhf.readthedocs.io/en/v0.7.0/likelihood.html>`_

    Args:
        signal_patch (``List[Dict]``): Patch data for signal model. please see
            `this link <https://pyhf.readthedocs.io/en/v0.7.0/likelihood.html>`_ for details on
            the structure of the input.
        background_only_model (``Dict`` or ``Text``): This input expects background only data
            that describes the full statistical model for the background. It also accepts ``str``
            input which indicates the full path to the background only ``JSON`` file.

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
        >>> stat_wrapper = spey.get_backend("pyhf")
        >>> statistical_model = stat_wrapper(
        ...     analysis="simple_pyhf",
        ...     background_only_model=background_only,
        ...     signal_patch=signal,
        ... )
        >>> statistical_model.exclusion_confidence_level() # [0.9474850259721279]
    """

    name: Text = "pyhf"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = PyhfInterface.spey_requires
    """Spey version required for the backend"""
    doi: List[Text] = PyhfInterface.doi
    """Citable DOI for the backend"""

    def __init__(
        self,
        signal_patch: Dict,
        background_only_model: Union[Text, Dict],
    ):
        super().__init__()
        self._model = FullStatisticalModelData(signal_patch, background_only_model)

    def combine(self, other, **kwargs):
        """
        Combine full statistical models generated by pyhf interface

        Args:
            other (``FullStatisticalModel``): other statistical model to be combined with this model
            kwargs:

              pyhf specific inputs:

              * join (``str``, default ``None``): How to join the two workspaces.
                Pick from ``"none"``, ``"outer"``, ``"left outer"`` or "right outer".
              * merge_channels (``bool``): Whether or not to merge channels when performing the combine.
                This is only done with ``"outer"``, ``"left outer"``, and ``"right outer"`` options.

              non-pyhf specific inputs:

              * update_measurements (``bool``, default ``True``): In case the measurement name of two
                statistical models are the same, other statistical model's measurement name will be
                updated. If set to ``False`` measurements will remain as is.

              .. note::

                This model is ``"left"`` and other model is considered to be ``"right"``.

        Raises:
            ``CombinationError``: Raised if its not possible to combine statistical models.

        Returns:
            :obj:`~spey-pyhf.interface.FullStatisticalModel`:
            Combined statistical model.
        """
        assert isinstance(
            other, FullStatisticalModel
        ), f"Combination between {self.name} and {other.name} backends is not available."

        update_measurements = kwargs.pop("update_measurements", True)

        this_workspace = copy.deepcopy(self.model.workspace)
        other_workspace = copy.deepcopy(other.model.workspace)

        if update_measurements:
            warnings.warn(
                "Measurement names are identical which may create problems during combination."
                "The measurement name of the other statistical model will be updated. "
                "However, if this is not the desired action please set ``update_measurements`` to ``False``."
            )
            for this_measurement in this_workspace["measurements"]:
                for idy, other_measurement in enumerate(other_workspace["measurements"]):
                    if this_measurement["name"] == other_measurement["name"]:
                        other_workspace["measurements"][idy]["name"] += "_updated"

        try:
            combined_workspace = self.manager.pyhf.Workspace.combine(
                this_workspace, other_workspace, **kwargs
            )
        except Exception as err:
            raise CombinationError(
                "Unable to combine given background only models, they might not be compatible."
                "Please try with different `join` option : 'none', 'outer', 'left outer', or 'right outer'"
                " and `merge_channels` option."
            ) from err

        # Reorganise signal patch

        # Collect channel names for combined model
        combined_channel_names = [ch["name"] for ch in combined_workspace["channels"]]
        this_channel_names = [ch["name"] for ch in this_workspace["channels"]]
        other_channel_names = [ch["name"] for ch in other_workspace["channels"]]

        # genereta channel map for this signal patch to find the
        # location of the channel in the combined workspace
        this_channel_map = {
            this_channel_names[int(patch["path"].split("/")[2])]: patch
            for patch in copy.deepcopy(self.model.signal_patch)
        }
        # genereta channel map for other signal patch to find the
        # location of the channel in the combined workspace
        other_channel_map = {
            other_channel_names[int(patch["path"].split("/")[2])]: patch
            for patch in copy.deepcopy(other.model.signal_patch)
        }

        new_signal_patch = []
        to_remove = []
        for chid, chname in enumerate(combined_channel_names):
            current_map = None
            if chname in this_channel_map.keys():
                current_map = copy.deepcopy(this_channel_map)
            elif chname in other_channel_map.keys():
                current_map = copy.deepcopy(other_channel_map)
            else:
                raise CombinationError(
                    f"Signal patch is not weldefined. Can not find channel {chname} in either signal patches."
                )

            current_signal = current_map[chname]
            # update channel path
            current_path = current_signal["path"].split("/")[1:]
            current_path[1] = str(chid)
            current_signal["path"] = "/" + "/".join(current_path)

            if current_signal["op"] != "remove":
                # update signal name
                current_signal["value"][
                    "name"
                ] = f"{current_signal['value']['name']}_CID_{chid}"
                new_signal_patch.append(current_signal)
            else:
                to_remove.append(current_signal)

        # reorder the patches that will be removed i.e. lowest channel id should be at the end.
        to_remove.sort(key=lambda x: int(x["path"].split("/")[1:][1]), reverse=True)
        new_signal_patch += to_remove

        # Generate new statistical model
        return FullStatisticalModel(
            signal_patch=new_signal_patch, background_only_model=dict(combined_workspace)
        )
