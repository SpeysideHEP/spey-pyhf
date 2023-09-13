import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from spey.system.exceptions import InvalidInput
from spey.utils import ExpectationType

from . import manager

__all__ = ["initialise_workspace"]


def initialise_workspace(
    signal: Union[List[float], List[Dict]],
    background: Union[Dict, List[float]],
    nb: Optional[List[float]] = None,
    delta_nb: Optional[List[float]] = None,
    expected: ExpectationType = ExpectationType.observed,
    return_full_data: bool = False,
) -> Union[
    tuple[
        Union[list, Any],
        Union[Optional[dict], Any],
        Optional[Any],
        Optional[Any],
        Optional[manager.pyhf.Workspace],
        Any,
        Any,
        Union[Union[int, float, complex], Any],
    ],
    tuple[Optional[manager.pyhf.Workspace], Any, Any],
]:
    """
    Construct the statistical model with respect to the given inputs.

    Args:
        signal (``Union[List[float], List[Dict]]``): signal yields
        background (``Union[Dict, List[float]]``): background yields or background only json file
        nb (``Optional[List[float]]``, default ``None``): number of background yields. This input
          is only used if ``signal`` and ``background`` input are ``List[float]``.
        delta_nb (``Optional[List[float]]``, default ``None``): uncertainty on background.This input
          is only used if ``signal`` and ``background`` input are ``List[float]``.
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
        return_full_data (``bool``, default ``False``): _description_

    Raises:
        ``InvalidInput``: _description_

    Returns:
        ``Union[ tuple[ Union[list, Any], Union[Optional[dict], Any], Optional[Any], Optional[Any], Optional[manager.pyhf.Workspace], Any, Any, Union[Union[int, float, complex], Any], ], tuple[Optional[manager.pyhf.Workspace], Any, Any], ]``:
        _description_
    """
    # Check the origin of signal
    signal_from_patch = False
    if isinstance(signal, (float, np.ndarray)):
        signal = np.array(signal, dtype=np.float32).reshape(-1)
    elif isinstance(signal, list):
        if isinstance(signal[0], dict):
            signal_from_patch = True
        else:
            signal = np.array(signal, dtype=np.float32).reshape(-1)
    else:
        raise InvalidInput(f"An unexpected type of signal has been presented: {signal}")

    # check the origin of background
    bkg_from_json = False
    if isinstance(background, dict):
        bkg_from_json = True
    elif isinstance(background, (float, list, np.ndarray)):
        background = np.array(background, dtype=np.float32).reshape(-1)
    else:
        raise InvalidInput(
            f"An unexpected type of background has been presented: {background}"
        )

    if (bkg_from_json and not signal_from_patch) or (
        signal_from_patch and not bkg_from_json
    ):
        raise InvalidInput("Signal and background types does not match.")

    if not bkg_from_json:
        # check if bkg uncertainties are valid
        if isinstance(delta_nb, (float, list, np.ndarray)):
            delta_nb = np.array(delta_nb, dtype=np.float32).reshape(-1)
        else:
            raise InvalidInput(
                f"An unexpected type of background uncertainty has been presented: {delta_nb}"
            )
        # check if MC bkg data is valid
        if isinstance(nb, (float, list, np.ndarray)):
            nb = np.array(nb, dtype=np.float32).reshape(-1)
        else:
            raise InvalidInput(
                f"An unexpected type of background has been presented: {nb}"
            )
        assert (
            len(signal) == len(background) == len(nb) == len(delta_nb)
        ), "Dimensionality of the data does not match."
    else:
        delta_nb, nb = None, None

    workspace, model, data, minimum_poi = None, None, None, -np.inf

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if not bkg_from_json:
            if expected == ExpectationType.apriori:
                # set data as expected background events
                background = nb
            # Create model from uncorrelated region
            model = manager.pyhf.simplemodels.uncorrelated_background(
                signal.tolist(), nb.tolist(), delta_nb.tolist()
            )
            data = background.tolist() + model.config.auxdata

            if return_full_data:
                minimum_poi = (
                    -np.min(
                        np.true_divide(nb[signal != 0.0], signal[signal != 0.0])
                    ).astype(np.float32)
                    if len(signal[signal != 0.0]) > 0
                    else -np.inf
                )

        else:
            if expected == ExpectationType.apriori:
                # set data as expected background events
                obs = []
                for channel in background.get("channels", []):
                    current = []
                    for ch in channel["samples"]:
                        if len(current) == 0:
                            current = [0.0] * len(ch["data"])
                        current = [cur + dt for cur, dt in zip(current, ch["data"])]
                    obs.append({"name": channel["name"], "data": current})
                background["observations"] = obs

            workspace = manager.pyhf.Workspace(background)
            model = workspace.model(
                patches=[signal],
                modifier_settings={
                    "normsys": {"interpcode": "code4"},
                    "histosys": {"interpcode": "code4p"},
                },
            )

            data = workspace.data(model)

            if return_full_data and None not in [model, workspace, data]:
                min_ratio = []
                for idc, channel in enumerate(background.get("channels", [])):
                    current_signal = []
                    for sigch in signal:
                        if idc == int(sigch["path"].split("/")[2]):
                            current_signal = np.array(
                                sigch.get("value", {}).get("data", []), dtype=np.float32
                            )
                            break
                    if len(current_signal) == 0:
                        continue
                    current_bkg = []
                    for ch in channel["samples"]:
                        if len(current_bkg) == 0:
                            current_bkg = np.zeros(
                                shape=(len(ch["data"]),), dtype=np.float32
                            )
                        current_bkg += np.array(ch["data"], dtype=np.float32)
                    min_ratio.append(
                        np.min(
                            np.true_divide(
                                current_bkg[current_signal != 0.0],
                                current_signal[current_signal != 0.0],
                            )
                        )
                        if np.any(current_signal != 0.0)
                        else np.inf
                    )
                minimum_poi = (
                    -np.min(min_ratio).astype(np.float32)
                    if len(min_ratio) > 0
                    else -np.inf
                )

    if return_full_data:
        return signal, background, nb, delta_nb, workspace, model, data, minimum_poi

    return workspace, model, data


def objective_wrapper(
    data: np.ndarray, pdf: manager.pyhf.pdf, do_grad: bool
) -> Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]:
    """
    Prepare objective function and its gradient

    Args:
        data (``np.ndarray``): Input data
        pdf (``pyhf.pdf``): statistical model
        do_grad (``bool``): If ``True``, returns objective and gradient
          in a tuple.

    Returns:
        ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
        Function to compute either only the objective function, ``do_grad=False``, or
        alongside with its gradient, ``do_grad=True``.

        .. warning::

            Gradient is only available for certain pyhf backends.
    """
    minimizer_kwargs, _ = manager.shim(
        manager.pyhf.infer.mle.twice_nll,
        manager.pyhf.tensorlib.astensor(data),
        pdf,
        pdf.config.suggested_init(),
        pdf.config.suggested_bounds(),
        None,
        do_grad,
        False,
    )

    def func(pars: np.ndarray) -> Union[np.array, Tuple[np.array, np.array]]:
        pars = manager.pyhf.tensorlib.astensor(pars)
        if do_grad:
            obj, grad = minimizer_kwargs["func"](pars)
            return np.array(obj), np.array(grad)

        return np.array(minimizer_kwargs["func"](pars))

    return func
