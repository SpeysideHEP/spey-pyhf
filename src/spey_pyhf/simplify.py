"""Interface to convert pyhf likelihoods to simplified likelihood framework"""

from typing import Text, List, Optional, Dict, Union

import spey, tqdm
from spey.optimizer.core import fit
from spey.backends.default_pdf import CorrelatedBackground, ThirdMomentExpansion

import numpy as np
from scipy.stats import multivariate_normal, moment

from ._version import __version__
from .interface import PyhfInterface


class ConversionError(Exception):
    """Conversion error class"""


class Simplify(PyhfInterface):
    """
    An interface to convert pyhf full statistical model prescription into simplified likelihood
    framework as either correlated background or third moment expansion model. For details on simplified
    likelihood framework please see
    `default plug-ins page <https://speysidehep.github.io/spey/plugins.html#default-plug-ins>`_.

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
        control_region_indices (``Optional[List[int]]``, default ``None``): indices of the control and validation
            regions inside the background only dictionary. For most of the cases interface will be able to pick
            up the names of these regions but in case the region names are not obvious, reconstruction may fail
            thus these indices will indicate the location of the VRs and CRs within the channel list.

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
    ) -> Union[CorrelatedBackground, ThirdMomentExpansion]:
        assert statistical_model.backend_type == "pyhf", (
            "This method is currently only available for `pyhf` full statistical models."
            + "For details please see spey-pyhf package."
        )
        assert statistical_model.backend.manager.backend == "jax", (
            "Please enable jax implementation for phyf interface. "
            + "If jax is installed this can be done by ``statistical_model.backend.manager.backend = 'jax'`` command."
        )

        bkgonly_model = statistical_model.backend.model.background_only_model
        original_signal_patch = statistical_model.backend.model.signal_patch
        POI_name = bkgonly_model["measurements"][0]["config"]["poi"]

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

        def add_to_json(idx: int, nbins: int) -> Dict:
            """
            Keep channel in the json file

            Args:
                idx (``int``): index of the channel
                nbins (``int``): number of bins

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
                        {"data": None, "name": POI_name, "type": "normfactor"},
                    ],
                },
            }

        # Prepare a JSON patch to separate control and validation regions
        # These regions are generally marked as CR and VR
        control_regions = []
        tmp_remove = []
        for ich, obs in enumerate(bkgonly_model["observations"]):
            if control_region_indices is not None:
                if ich in control_region_indices:
                    control_regions.append(add_to_json(ich, len(obs["data"])))
                else:
                    tmp_remove.append(remove_from_json(ich))
            else:
                if statistical_model.backend.model.metadata[ich]["type"] in ["CR", "VR"]:
                    control_regions.append(add_to_json(ich, len(obs["data"])))
                else:
                    tmp_remove.append(remove_from_json(ich))

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
            fixed_poi_value=0,
        )

        # compute the hessian matrix without poi
        poi_index = fit_opts["model_configuration"].poi_index
        hessian_neg_logprob = np.delete(
            np.delete(
                -control_model.backend.get_hessian_logpdf_func(expected=expected)(
                    fit_param
                ),
                poi_index,
                axis=0,
            ),
            poi_index,
            axis=1,
        )

        # Construct multivariate normal distribution to capture correlations
        # between nuisance parameters
        nuisance_cov_matrix = np.linalg.inv(hessian_neg_logprob)
        nuisance_distribution = multivariate_normal(
            mean=np.delete(fit_param, poi_index), cov=nuisance_cov_matrix
        )

        samples = []
        with tqdm.tqdm(
            total=number_of_samples,
            unit="sample",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        ) as pbar:
            for nui_smp in nuisance_distribution.rvs(size=(number_of_samples,)):
                current_nui_params = nui_smp.tolist()
                current_nui_params.insert(poi_index, 0.0)
                current_sample = statistical_model.backend.get_sampler(
                    np.array(current_nui_params)
                )(1, include_auxiliary=False)
                samples.append(current_sample)
                pbar.update()
        samples = np.vstack(samples)

        covariance_matrix = np.cov(samples, rowvar=0)

        model = statistical_model.backend.model()[1]
        data = statistical_model.backend.model.workspace.data(
            model, include_auxdata=False
        )
        channels = model.config.channels

        signal_yields = []
        for channel_name in channels:
            found = False
            for ch in model.spec["channels"]:
                if ch["name"] != channel_name:
                    continue
                for smp in ch["samples"]:
                    for mod in smp["modifiers"]:
                        if mod["name"] == POI_name:
                            found = True
                            signal_yields += smp["data"]
                            break
                    if found:
                        break
                if found:
                    break

        if convert_to == "default_pdf.correlated_background":
            backend = CorrelatedBackground(
                signal_yields=signal_yields,
                background_yields=np.mean(samples, axis=0),
                data=data,
                covariance_matrix=covariance_matrix,
            )
        elif convert_to == "default_pdf.third_moment_expansion":
            third_moments = moment(samples, moment=3, axis=0)

            backend = ThirdMomentExpansion(
                signal_yields=signal_yields,
                background_yields=np.mean(samples, axis=0),
                data=data,
                covariance_matrix=covariance_matrix,
                third_moment=third_moments,
            )
        else:
            raise ConversionError(
                "Currently available conversion methods are "
                + "'default_pdf.correlated_background', 'default_pdf.third_moment_expansion'"
            )

        return backend
