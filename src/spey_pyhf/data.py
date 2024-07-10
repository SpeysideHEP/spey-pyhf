import copy
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

import numpy as np
from spey import ExpectationType
from spey.base import ModelConfig
from spey.system.exceptions import InvalidInput

from . import WorkspaceInterpreter, manager


class Base(ABC):
    """Base class for pyhf data input"""

    def __call__(self, expected: ExpectationType = ExpectationType.observed) -> Tuple:
        """
        Retreive pyhf workspace

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

        Returns:
            ``Tuple``:
            workspace, model and data
        """
        return None, None, None

    @abstractmethod
    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: Optional[float] = None
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

    @property
    @abstractmethod
    def isAlive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""


@dataclass
class SimpleModelData(Base):
    """
    Dataclass for simple statistical model

    Args:
        signal_yields (``List[float]``): signal yields
        background_yields (``List[float]``): background yields
        data (``List[float]``): observations
        absolute_uncertainties (``List[float]``): absolute uncertainties on the background
    """

    signal_yields: List[float]
    background_yields: List[float]
    data: List[float]
    absolute_uncertainties: List[float]

    def __post_init__(self):
        assert (
            len(self.signal_yields)
            == len(self.background_yields)
            == len(self.data)
            == len(self.absolute_uncertainties)
        ), "Lenght of all input must be the same."

    def __call__(self, expected: ExpectationType = ExpectationType.observed) -> Tuple:
        """
        Retreive pyhf workspace

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

        Returns:
            ``Tuple``:
            workspace, model and data
        """
        model = manager.pyhf.simplemodels.uncorrelated_background(
            self.signal_yields, self.background_yields, self.absolute_uncertainties
        )

        data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        ) + model.config.auxdata

        return None, model, data

    @property
    def isAlive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return any(x > 0 for x in self.signal_yields)

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: Optional[float] = None
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
        signal = np.array(self.signal_yields)
        nb = np.array(self.background_yields)
        minimum_poi = (
            -np.min(np.true_divide(nb[signal != 0.0], signal[signal != 0.0])).astype(
                np.float32
            )
            if len(signal[signal != 0.0]) > 0
            else -np.inf
        )

        _, model, _ = self()

        bounds = model.config.suggested_bounds()
        bounds[model.config.poi_index] = (
            max(minimum_poi, -10.0) if allow_negative_signal else 0.0,
            bounds[model.config.poi_index][1] if not poi_upper_bound else poi_upper_bound,
        )

        return ModelConfig(
            poi_index=model.config.poi_index,
            minimum_poi=minimum_poi,
            suggested_init=model.config.suggested_init(),
            suggested_bounds=bounds,
            parameter_names=model.config.par_names,
        )


@dataclass
class FullStatisticalModelData(Base):
    """
    Data container for the full statistical model.

    Args:
        signal_patch (``List[Dict]``): Patch data for signal model. please see
            `this link <https://pyhf.readthedocs.io/en/v0.7.0/likelihood.html>`_ for details on
            the structure of the input.
        background_only_model (``Dict`` or ``Text``): This input expects background only data
            that describes the full statistical model for the background. It also accepts ``str``
            input which indicates the full path to the background only ``JSON`` file.
    """

    signal_patch: List[Dict]
    background_only_model: Union[Dict, Text]

    def __post_init__(self):
        if isinstance(self.background_only_model, str):
            if not os.path.isfile(self.background_only_model):
                raise FileNotFoundError(f"Can not find {self.background_only_model}")
            with open(self.background_only_model, "r", encoding="uft-8") as f:
                self.background_only_model = json.load(f)

        interpreter = WorkspaceInterpreter(self.background_only_model)
        interpreter.add_patch(self.signal_patch)

        # set data as expected background events
        self.expected_background_yields = interpreter.expected_background_yields

        self.workspace = manager.pyhf.Workspace(self.background_only_model)

        self._model = None
        # Initialise config
        model = self()[1]

        min_ratio = []
        for idc, channel in enumerate(self.background_only_model.get("channels", [])):
            current_signal = []
            for sigch in self.signal_patch:
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
                    current_bkg = np.zeros(shape=(len(ch["data"]),), dtype=np.float32)
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
        self._minimum_poi = (
            -np.min(min_ratio).astype(np.float32) if len(min_ratio) > 0 else -np.inf
        )

        self._config = {
            "poi_index": model.config.poi_index,
            "minimum_poi": self._minimum_poi,
            "suggested_init": model.config.suggested_init(),
            "suggested_bounds": model.config.suggested_bounds(),
            "parameter_names": model.config.par_names,
            "suggested_fixed": model.config.suggested_fixed(),
        }

    @property
    def channels(self) -> Iterator[Text]:
        """Return channel names"""
        return (ch["name"] for ch in self.background_only_model["channels"])

    @property
    def channel_properties(self) -> Iterator[Tuple[int, Text, int]]:
        """Returns an iterator for channel index, name and number of bins"""
        for idx, channel in enumerate(self.channels):
            yield idx, channel, self.workspace.channel_nbins[channel]

    def __call__(
        self,
        expected: ExpectationType = ExpectationType.observed,
        include_aux: bool = True,
    ) -> Tuple:
        """
        Retreive pyhf workspace

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
            include_aux (`bool`, default `True`): Include auxiliary data.

        Returns:
            ``Tuple``:
            workspace, model and data
        """
        if self._model is None:
            self._model = self.workspace.model(
                patches=[self.signal_patch],
                modifier_settings={
                    "normsys": {"interpcode": "code4"},
                    "histosys": {"interpcode": "code4p"},
                },
            )

        if expected == ExpectationType.apriori:
            try:
                data = sum(
                    (
                        self.expected_background_yields[ch]
                        for ch in self._model.config.channels
                    ),
                    [],
                )
            except KeyError as err:
                # provide a useful error message to guide the user to the solution
                missing_channels = [
                    ch
                    for ch in self._model.config.channels
                    if ch not in self.expected_background_yields
                ]
                raise InvalidInput(
                    "Unable to construct expected data. "
                    + (len(missing_channels) > 0)
                    * (
                        "\nThis is likely due to missing channels in the signal patch. "
                        + "The missing channels are: "
                        + ", ".join(missing_channels)
                        + "\nPlease provide appropriate action for the missing channels to continue."
                    )
                ) from err
            if include_aux:
                data += self._model.config.auxdata
        else:
            data = self.workspace.data(self._model, include_auxdata=include_aux)

        return self.workspace, self._model, data

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: Optional[float] = None
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
        bounds = copy.deepcopy(self._config["suggested_bounds"])
        bounds[self._config["poi_index"]] = (
            max(self._minimum_poi, -10.0) if allow_negative_signal else 0.0,
            bounds[self._config["poi_index"]][1]
            if not poi_upper_bound
            else poi_upper_bound,
        )

        return ModelConfig(
            poi_index=self._config["poi_index"],
            minimum_poi=self._minimum_poi,
            suggested_init=self._config["suggested_init"],
            suggested_bounds=bounds,
            parameter_names=self._config["parameter_names"],
            suggested_fixed=self._config["suggested_fixed"],
        )

    @property
    def isAlive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        for channel in (
            ch for ch in self.signal_patch if ch.get("value", None) is not None
        ):
            if any(nsig > 0.0 for nsig in channel["value"].get("data", [])):
                return True
        return False
