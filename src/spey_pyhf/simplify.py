"""Interface to convert pyhf likelihoods to simplified likelihood framework"""
import copy
import warnings
from typing import Callable, List, Optional, Text, Union, Literal

import numpy as np
import spey
import tqdm
from scipy.stats import moment, multivariate_normal
from spey.backends.default_pdf import CorrelatedBackground, ThirdMomentExpansion
from spey.optimizer.core import fit

from . import WorkspaceInterpreter
from ._version import __version__


def __dir__():
    return []


class ConversionError(Exception):
    """Conversion error class"""


def make_constraint(index: int, value: float) -> Callable[[np.ndarray], float]:
    """
    Construct constraint

    Args:
        index (``int``): index of the parameter within the parameter list
        value (``float``): fixed value

    Returns:
        ``Callable[[np.ndarray], float]``:
        constraint function
    """

    def func(vector: np.ndarray) -> float:
        return vector[index] - value

    return func


class Simplify(spey.ConverterBase):
    r"""
    An interface to convert pyhf full statistical model prescription into simplified likelihood
    framework as either correlated background or third moment expansion model. For details on simplified
    likelihood framework please see
    `default plug-ins page <https://spey.readthedocs.io/en/main/plugins.html#default-plug-ins>`_.

    Details on methodology can be found `in the online documentation <https://spey-pyhf.readthedocs.io/en/main/simplify.html>`_.

    Args:
        statistical_model (:obj:`~spey.StatisticalModel`): constructed full statistical model
        fittype (``Text``, default ``"postfit"``): what type of fitting should be performed ``"postfit"``
            or ``"prefit"``.
        convert_to (``Text``, default ``"default_pdf.correlated_background"``): conversion type. Should
            be either ``"default_pdf.correlated_background"`` or ``"default_pdf.third_moment_expansion"``.
        number_of_samples (``int``, default ``1000``): number of samples to be generated in order to estimate
            contract the uncertainties into a single value.
        control_region_indices (``List[int]`` or ``List[Text]``, default ``None``): indices or names of the control and
            validation regions inside the background only dictionary. For most of the cases interface will be able
            to guess the names of these regions but in case the region names are not obvious, reconstruction may
            fail thus these indices will indicate the location of the VRs and CRs within the channel list.
        include_modifiers_in_control_model (``bool``, default ``False``): This flag enables the extraction of the signal
            modifiers to be used in the control model. The control model yields will still be zero and :math:`\mu=0`
            but the contribution of the signal modifiers to the nuisance covariance matrix will be taken into account.
            By default, modifiers are excluded from the control model.
        save_model (``Text``, default ``None``): Full path to save the model details. Model will be saved as
            compressed NumPy file (``.npz``), file name should be given as ``/PATH/TO/DIR/MODELNAME.npz``.

            **Reading the saved model:**

            One can read the saved model using NumPy's :func:`load` function

            .. code:: python3

                >>> import numpy as np
                >>> saved_model = np.load("/PATH/TO/DIR/MODELNAME.npz")
                >>> data = saved_model["data"]

            This model has several containers which includes the following keywords:

            * ``"covariance_matrix"``: includes covariance matrix per bin
            * ``"background_yields"``: includes background yields per bin
            * ``"third_moments"``: (if ``convert_to="default_pdf.third_moment_expansion"``) includes third moments
              per bin
            * ``"data"``: includes observed values per bin
            * ``"channel_order"``: includes information regarding the channel order to convert a signal patch to be used
              in the simplified framework.

    Raises:
        ``ConversionError``: If the requirements are not satisfied.
        :obj:`AssertionError`: If input statistical model does not have ``pyhf`` backend or ``pyhf``
            manager does not use ``jax`` backend.

    **Example:**

    As an example, lets use the JSON files provided for ATLAS-SUSY-2019-08 analysis which can be found in
    `HEPData <https://www.hepdata.net/record/resource/1934827?landing_page=true>`_. Once these are downloaded
    one can read them as and construct a model as follows;

    .. code:: python3

        >>> import json, spey
        >>> with open("1Lbb-likelihoods-hepdata/BkgOnly.json", "r") as f:
        >>>	    background_only = json.load(f)
        >>> with open("1Lbb-likelihoods-hepdata/patchset.json", "r") as f:
        >>>     signal = json.load(f)["patches"][0]["patch"]

        >>> pdf_wrapper = spey.get_backend("pyhf")
        >>> full_statistical_model = pdf_wrapper(
        ...     background_only_model=background_only, signal_patch=signal
        ... )
        >>> full_statistical_model.backend.manager.backend = "jax"

    Note that ``patchset.json`` includes more than one patch set, thats why we used only one of them.
    The last line enables the usage of ``jax`` backend in ``pyhf`` interface which in turn enables one
    to compute Hessian of the statistical model which is needed for simplification procedure.

    Now we ca call ``"pyhf.simplify"`` model to map our full likelihood to simplified likelihood framework

    .. code:: python3

        >>> converter = spey.get_backend("pyhf.simplify")
        >>> simplified_model = converter(
        ...     statistical_model=full_statistical_model,
        ...     convert_to="default_pdf.correlated_background",
        ...     control_region_indices=[
        ...	        'WREM_cuts', 'STCREM_cuts', 'TRHMEM_cuts', 'TRMMEM_cuts', 'TRLMEM_cuts'
        ...	    ]
        ... )
        >>> print(simplified_model.backend_type)
        >>> # "default_pdf.correlated_background"
    """

    name: Text = "pyhf.simplify"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = ">=0.1.5,<0.2.0"
    """Spey version required for the backend"""

    def __call__(
        self,
        statistical_model: spey.StatisticalModel,
        fittype: Literal["postfit", "prefit"] = "postfit",
        convert_to: Literal[
            "default_pdf.correlated_background", "default_pdf.third_moment_expansion"
        ] = "default_pdf.correlated_background",
        number_of_samples: int = 1000,
        control_region_indices: Optional[Union[List[int], List[Text]]] = None,
        include_modifiers_in_control_model: bool = False,
        save_model: Optional[Text] = None,
    ) -> Union[CorrelatedBackground, ThirdMomentExpansion]:

        assert statistical_model.backend_type == "pyhf", (
            "This method is currently only available for `pyhf` full statistical models."
            + "For details please see spey-pyhf package."
        )
        assert statistical_model.backend.manager.backend == "jax", (
            "Please enable jax implementation for phyf interface. "
            + "If jax is installed this can be done by "
            + "``statistical_model.backend.manager.backend = 'jax'`` command."
        )

        bkgonly_model = statistical_model.backend.model.background_only_model
        signal_patch = statistical_model.backend.model.signal_patch

        expected = {
            "postfit": spey.ExpectationType.observed,
            "prefit": spey.ExpectationType.apriori,
        }[fittype]

        interpreter = WorkspaceInterpreter(bkgonly_model)

        # configure signal patch map with respect to channel names
        signal_patch_map = interpreter.patch_to_map(signal_patch)

        # Prepare a JSON patch to separate control and validation regions
        # These regions are generally marked as CR and VR
        if control_region_indices is None:
            control_region_indices = interpreter.guess_CRVR()

        if len(control_region_indices) == 0:
            raise ConversionError(
                "Can not construct control model. Please provide ``control_region_indices``."
            )

        for channel in interpreter.get_channels(control_region_indices):
            interpreter.inject_signal(
                channel,
                [0.0] * len(signal_patch_map[channel]["data"]),
                signal_patch_map[channel]["modifiers"]
                if include_modifiers_in_control_model
                else None,
            )

        pdf_wrapper = spey.get_backend("pyhf")
        control_model = pdf_wrapper(
            background_only_model=bkgonly_model, signal_patch=interpreter.make_patch()
        )

        # Extract the nuisance parameters that maximises the likelihood at mu=0
        fit_opts = control_model.prepare_for_fit(expected=expected)
        _, fit_param = fit(
            **fit_opts,
            initial_parameters=None,
            bounds=None,
            fixed_poi_value=0.0,
        )

        # compute the hessian matrix without poi
        control_poi_index = fit_opts["model_configuration"].poi_index
        hessian_neg_logprob = np.delete(
            np.delete(
                -control_model.backend.get_hessian_logpdf_func(expected=expected)(
                    fit_param
                ),
                control_poi_index,
                axis=0,
            ),
            control_poi_index,
            axis=1,
        )

        # Construct multivariate normal distribution to capture correlations
        # between nuisance parameters
        nuisance_cov_matrix = np.linalg.inv(hessian_neg_logprob)
        nuisance_distribution = multivariate_normal(
            mean=np.delete(fit_param, control_poi_index), cov=nuisance_cov_matrix
        )

        # Retreive pyhf models and compare parameter maps
        stat_model_pyhf = statistical_model.backend.model()[1]
        control_model_pyhf = control_model.backend.model()[1]
        is_nuisance_map_different = (
            stat_model_pyhf.config.par_map != control_model_pyhf.config.par_map
        )
        fit_opts = statistical_model.prepare_for_fit(expected=expected)
        suggested_fixed = fit_opts["model_configuration"].suggested_fixed

        samples = []
        warnings_list = []
        with tqdm.tqdm(
            total=number_of_samples,
            unit="sample",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        ) as pbar:
            while len(samples) < number_of_samples:
                nui_smp = nuisance_distribution.rvs(size=(1,))
                current_nui_params = nui_smp.tolist()
                current_nui_params.insert(control_poi_index, 0.0)
                new_params = None

                if is_nuisance_map_different:
                    constraints = []
                    new_params = np.array([np.nan] * stat_model_pyhf.config.npars)
                    for key, item in stat_model_pyhf.config.par_map.items():
                        if key in control_model_pyhf.config.par_map.keys():
                            current_params = current_nui_params[
                                control_model_pyhf.config.par_map[key]["slice"]
                            ]
                            current_range = range(
                                item["slice"].start,
                                item["slice"].stop,
                                1 if item["slice"].step is None else item["slice"].step,
                            )
                            for ival, val in zip(current_range, current_params):
                                new_params[ival] = val
                                if not suggested_fixed[ival]:
                                    constraints.append(
                                        {"type": "eq", "fun": make_constraint(ival, val)}
                                    )

                    if np.any(np.isnan(new_params)):
                        current_fit_opts = copy.deepcopy(fit_opts)
                        init_params = np.where(
                            np.isnan(new_params),
                            current_fit_opts["model_configuration"].suggested_init,
                            new_params,
                        )
                        if np.isnan(current_fit_opts["logpdf"](np.array(init_params))):
                            # if the initial value is NaN continue search
                            continue
                        current_fit_opts["constraints"] += constraints
                        with warnings.catch_warnings(record=True) as w:
                            _, new_params = fit(
                                **current_fit_opts,
                                initial_parameters=init_params.tolist(),
                                bounds=None,
                            )
                            warnings_list += w

                try:
                    current_sample = statistical_model.backend.get_sampler(
                        np.array(current_nui_params if new_params is None else new_params)
                    )(1, include_auxiliary=False)
                    samples.append(current_sample)
                    pbar.update()
                except ValueError:
                    # Some of the samples can lead to problems while sampling from a poisson distribution.
                    # e.g. poisson requires positive lambda values to sample from. If sample leads to a negative
                    # lambda value continue sampling to avoid that point.
                    continue

        if len(warnings_list) > 0:
            warnings.warn(
                message=f"{len(warnings_list)} warning(s) generated during sampling."
                " This might be due to edge cases in nuisance parameter sampling.",
                category=RuntimeWarning,
            )

        samples = np.vstack(samples)

        covariance_matrix = np.cov(samples, rowvar=0)

        data = statistical_model.backend.model.workspace.data(
            stat_model_pyhf, include_auxdata=False
        )

        # NOTE: model spec might be modified within the pyhf workspace, thus
        # yields needs to be reordered properly before constructing the simplified likelihood
        signal_yields = []
        for channel_name in stat_model_pyhf.config.channels:
            signal_yields += signal_patch_map[channel_name]["data"]
        # NOTE background yields are first moments in simplified framework not the yield values
        # in the full statistical model!
        background_yields = np.mean(samples, axis=0)

        third_moments = []
        if convert_to == "default_pdf.correlated_background":
            backend = CorrelatedBackground(
                signal_yields=signal_yields,
                background_yields=background_yields,
                data=data,
                covariance_matrix=covariance_matrix,
            )
        elif convert_to == "default_pdf.third_moment_expansion":
            third_moments = moment(samples, moment=3, axis=0)

            backend = ThirdMomentExpansion(
                signal_yields=signal_yields,
                background_yields=background_yields,
                data=data,
                covariance_matrix=covariance_matrix,
                third_moment=third_moments,
            )
        else:
            raise ConversionError(
                "Currently available conversion methods are "
                + "'default_pdf.correlated_background', 'default_pdf.third_moment_expansion'"
            )

        if save_model is not None:
            if save_model.endswith(".npz"):
                np.savez_compressed(
                    save_model,
                    covariance_matrix=covariance_matrix,
                    background_yields=background_yields,
                    third_moments=third_moments,
                    data=data,
                    channel_order=stat_model_pyhf.config.channels,
                )

        return backend
