"""Interface to convert pyhf likelihoods to simplified likelihood framework"""
from typing import Text, List, Optional, Dict, Union, Callable

import spey, tqdm, copy
from spey.optimizer.core import fit
from spey.backends.default_pdf import CorrelatedBackground, ThirdMomentExpansion

import numpy as np
from scipy.stats import multivariate_normal, moment

from ._version import __version__


def __dir__():
    return []


class ConversionError(Exception):
    """Conversion error class"""


def remove_from_json(idx: int) -> Dict:
    """
    Remove channel from the json file

    Args:
        idx (``int``): index of the channel

    Returns:
        ``Dict``:
        JSON patch
    """
    return {"op": "remove", "path": f"/channels/{idx}"}


def add_to_json(idx: int, nbins: int, poi_name: Text) -> Dict:
    """
    Keep channel in the json file

    Args:
        idx (``int``): index of the channel
        nbins (``int``): number of bins
        poi_name (``Text``): name of POI

    Returns:
        ``Dict``:
        json patch
    """
    return {
        "op": "add",
        "path": f"/channels/{idx}/samples/0",
        "value": {
            "name": "smp",
            "data": [0.0] * nbins,
            "modifiers": [
                {"data": None, "name": "lumi", "type": "lumi"},
                {"data": None, "name": poi_name, "type": "normfactor"},
            ],
        },
    }


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
    """
    An interface to convert pyhf full statistical model prescription into simplified likelihood
    framework as either correlated background or third moment expansion model. For details on simplified
    likelihood framework please see
    `default plug-ins page <https://speysidehep.github.io/spey/plugins.html#default-plug-ins>`_.

    Details on methodology can be found `in the online documentation <https://speysidehep.github.io/spey-pyhf/simplify.html>`_.

    Args:
        statistical_model (:obj:`~spey.StatisticalModel`): constructed full statistical model
        expected (:obj:`~spey.ExpectationType`): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
        convert_to (``Text``, default ``"default_pdf.correlated_background"``): conversion type. Should
            be either ``"default_pdf.correlated_background"`` or ``"default_pdf.third_moment_expansion"``.
        number_of_samples (``int``, default ``1000``): number of samples to be generated in order to estimate
            contract the uncertainties into a single value.
        control_region_indices (``List[int]``, default ``None``): indices of the control and validation
            regions inside the background only dictionary. For most of the cases interface will be able to pick
            up the names of these regions but in case the region names are not obvious, reconstruction may fail
            thus these indices will indicate the location of the VRs and CRs within the channel list.
        save_model (``Text``, default ``None``): Full path to save the model details. Model will be saved as
            compressed NumPy file (``.npz``), file name should be given as ``/PATH/TO/DIR/MODELNAME.npz``.

    Raises:
        ``ConversionError``: If the requirements are not satisfied.
    """

    name: Text = "pyhf.simplify"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = ">=0.1.1"
    """Spey version required for the backend"""

    def __call__(
        self,
        statistical_model: spey.StatisticalModel,
        expected: spey.ExpectationType = spey.ExpectationType.observed,
        convert_to: Text = "default_pdf.correlated_background",
        number_of_samples: int = 1000,
        control_region_indices: Optional[List[int]] = None,
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

        # configure signal patch map with respect to channel names
        signal_patch_map = {}
        for channel in signal_patch:
            if channel["op"] == "add":
                path = int(channel["path"].split("/")[2])
                channel_name = bkgonly_model["channels"][path]["name"]
                signal_patch_map[channel_name] = channel["value"]["data"]

        # Retreive POI name
        POI_name = bkgonly_model["measurements"][0]["config"]["poi"]

        # Prepare a JSON patch to separate control and validation regions
        # These regions are generally marked as CR and VR
        control_regions = []
        tmp_remove = []
        for (
            ich,
            channel_name,
            nbins,
        ) in statistical_model.backend.model.channel_properties:
            if control_region_indices is not None:
                if ich in control_region_indices:
                    control_regions.append(add_to_json(ich, nbins, POI_name))
                else:
                    tmp_remove.append(remove_from_json(ich))
            else:
                if statistical_model.backend.model.metadata[channel_name] in ["CR", "VR"]:
                    control_regions.append(add_to_json(ich, nbins, POI_name))
                else:
                    tmp_remove.append(remove_from_json(ich))

        if len(control_regions) == 0:
            raise ConversionError(
                "Can not construct control model. Please provide ``control_region_indices``."
            )

        # Need to sort correctly the paths to the channels to be removed
        tmp_remove.sort(key=lambda p: p["path"].split("/")[-1], reverse=True)
        control_regions += tmp_remove

        pdf_wrapper = spey.get_backend("pyhf")
        control_model = pdf_wrapper(
            background_only_model=bkgonly_model, signal_patch=control_regions
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
                        logpdf = current_fit_opts["logpdf"](np.array(init_params))
                        if np.isnan(logpdf):
                            # if the initial value is NaN continue search
                            continue
                        current_fit_opts["constraints"] += constraints
                        _, new_params = fit(
                            **current_fit_opts,
                            initial_parameters=init_params.tolist(),
                            bounds=None,
                        )

                current_sample = statistical_model.backend.get_sampler(
                    np.array(current_nui_params if new_params is None else new_params)
                )(1, include_auxiliary=False)
                samples.append(current_sample)
                pbar.update()
        samples = np.vstack(samples)

        covariance_matrix = np.cov(samples, rowvar=0)

        data = statistical_model.backend.model.workspace.data(
            stat_model_pyhf, include_auxdata=False
        )

        # NOTE: model spec might be modified within the pyhf workspace, thus
        # yields needs to be reordered properly before constructing the simplified likelihood
        signal_yields = []
        for channel_name in stat_model_pyhf.config.channels:
            signal_yields += signal_patch_map[channel_name]
        background_yields = np.sum(
            np.squeeze(stat_model_pyhf.nominal_rates), axis=0
        ).tolist()

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
            if not save_model.endswith(".npz"):
                raise ValueError(
                    f"Model file extension has to be ``.npz``. '{save_model}' is given."
                )
            np.savez_compressed(
                save_model,
                covariance_matrix=covariance_matrix,
                background_yields=background_yields,
                third_moments=third_moments,
                data=data,
            )

        return backend
