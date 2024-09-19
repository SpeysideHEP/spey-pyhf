"""Helper function for creating and interpreting pyhf inputs"""
import logging
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

__all__ = ["WorkspaceInterpreter"]


def __dir__():
    return __all__


# pylint: disable=W1203, W1201, C0103

log = logging.getLogger("Spey")


def remove_from_json(idx: int) -> Dict[Text, Text]:
    """
    Remove channel from the json file

    Args:
        idx (``int``): index of the channel

    Returns:
        ``Dict[Text, Text]``:
        JSON patch
    """
    return {"op": "remove", "path": f"/channels/{idx}"}


def add_to_json(idx: int, yields: List[float], modifiers: List[Dict]) -> Dict:
    """
    Keep channel in the json file

    Args:
        idx (``int``): index of the channel
        yields (``List[float]``): data
        modifiers (``List[Dict]``): signal modifiers

    Returns:
        ``Dict``:
        json patch
    """
    return {
        "op": "add",
        "path": f"/channels/{idx}/samples/0",
        "value": {"name": "Signal", "data": yields, "modifiers": modifiers},
    }


def _default_modifiers(poi_name: Text) -> List[Dict]:
    """Retreive default modifiers"""
    return [
        {"data": None, "name": "lumi", "type": "lumi"},
        {"data": None, "name": poi_name, "type": "normfactor"},
    ]


class WorkspaceInterpreter:
    """
    A pyhf workspace interpreter to handle book keeping for the background only models
    and convert signal yields into JSONPatch compatible for pyhf.

    Args:
        background_only_model (``Dict``): descrioption for the background only statistical model
    """

    __slots__ = [
        "background_only_model",
        "_signal_dict",
        "_signal_modifiers",
        "_to_remove",
    ]

    def __init__(self, background_only_model: Dict):
        self.background_only_model = background_only_model
        """Background only statistical model description"""
        self._signal_dict = {}
        self._signal_modifiers = {}
        self._to_remove = []

    def __getitem__(self, item):
        return self.background_only_model[item]

    @property
    def channels(self) -> Iterator[List[Text]]:
        """Retreive channel names as iterator"""
        return (ch["name"] for ch in self["channels"])

    @property
    def poi_name(self) -> Dict[Text, Text]:
        """Retreive poi name per measurement"""
        return [(mes["name"], mes["config"]["poi"]) for mes in self["measurements"]]

    @property
    def bin_map(self) -> Dict[Text, int]:
        """Get number of bins per channel"""
        return {ch["name"]: len(ch["samples"][0]["data"]) for ch in self["channels"]}

    @property
    def expected_background_yields(self) -> Dict[Text, List[float]]:
        """Retreive expected background yields with respect to signal injection"""
        yields = {}
        undefined_channels = []
        for channel in self["channels"]:
            if channel["name"] not in self.remove_list:
                yields[channel["name"]] = []
                for smp in channel["samples"]:
                    if len(yields[channel["name"]]) == 0:
                        yields[channel["name"]] = [0.0] * len(smp["data"])
                    yields[channel["name"]] = [
                        ch + dt for ch, dt in zip(yields[channel["name"]], smp["data"])
                    ]
                if channel["name"] not in self._signal_dict:
                    undefined_channels.append(channel["name"])
        if len(undefined_channels) > 0:
            log.warning(
                "Some of the channels are not defined in the patch set, "
                "these channels will be kept in the statistical model. "
            )
            log.warning(
                "If these channels are meant to be removed, please indicate them in the patch set."
            )
            log.warning(
                "Please check the following channel(s): " + ", ".join(undefined_channels)
            )
        return yields

    def guess_channel_type(self, channel_name: Text) -> Text:
        """Guess the type of the channel as CR VR or SR"""
        if channel_name not in self.channels:
            raise ValueError(f"Unknown channel: {channel_name}")
        for tp in ["CR", "VR", "SR"]:
            if tp in channel_name.upper():
                return tp

        return "__unknown__"

    def guess_CRVR(self) -> List[Text]:
        """Retreive control and validation channel names by guess"""
        return [
            name
            for name in self.channels
            if self.guess_channel_type(name) in ["CR", "VR"]
        ]

    def get_channels(self, channel_index: Union[List[int], List[Text]]) -> List[Text]:
        """
        Retreive channel names with respect to their index

        Args:
            channel_index (``List[int]``): Indices of the channels

        Returns:
            ``List[Text]``:
            Names of the channels corresponding to the given indices
        """
        return [
            name
            for idx, name in enumerate(self.channels)
            if idx in channel_index or name in channel_index
        ]

    def inject_signal(
        self, channel: Text, data: List[float], modifiers: Optional[List[Dict]] = None
    ) -> None:
        """
        Inject signal to the model

        Args:
            channel (``Text``): channel name
            data (``List[float]``): signal yields
            modifiers (``List[Dict]``): uncertainties. If None, default modifiers will be added.

        Raises:
            ``ValueError``: If channel does not exist or number of yields does not match
                with the bin size of the channel
        """
        if channel not in self.channels:
            raise ValueError(
                f"{channel} does not exist. Available channels are "
                + ", ".join(self.channels)
            )
        if len(data) != self.bin_map[channel]:
            raise ValueError(
                f"Number of bins in injection does not match to the channel. "
                f"{self.bin_map[channel]} expected, {len(data)} received."
            )

        default_modifiers = _default_modifiers(self.poi_name[0][1])
        if modifiers is not None:
            for mod in default_modifiers:
                if mod not in modifiers:
                    log.warning(
                        f"Modifier `{mod['name']}` with type `{mod['type']}` is missing"
                        f" from the input. Adding `{mod['name']}`"
                    )
                    log.debug(f"Adding modifier: {mod}")
                    modifiers.append(mod)

        self._signal_dict[channel] = data
        self._signal_modifiers[channel] = (
            default_modifiers if modifiers is None else modifiers
        )

    @property
    def signal_per_channel(self) -> Dict[Text, List[float]]:
        """Return signal yields in each channel"""
        return self._signal_dict

    def make_patch(self) -> List[Dict]:
        """
        Make a JSONPatch for the background only model

        Args:
            measurement_index (``int``, default ``0``): in case of multiple measurements
                which one to be used. Detauls is always the first measurement

        Raises:
            ``ValueError``: if there is no signal.

        Returns:
            ``List[Dict]``:
            JSONPatch file for the background only model.
        """
        if not self._signal_dict:
            raise ValueError("Please add signal yields.")

        patch = []
        to_remove = []
        for ich, channel in enumerate(self.channels):
            if channel in self._signal_dict:
                patch.append(
                    add_to_json(
                        ich, self._signal_dict[channel], self._signal_modifiers[channel]
                    )
                )
            elif channel in self._to_remove:
                to_remove.append(remove_from_json(ich))
            else:
                log.warning(f"Undefined channel in the patch set: {channel}")

        to_remove.sort(key=lambda p: p["path"].split("/")[-1], reverse=True)

        return patch + to_remove

    def reset_signal(self) -> None:
        """Clear the signal map"""
        self._signal_dict = {}
        self._to_remove = []

    def add_patch(self, signal_patch: List[Dict]) -> None:
        """Inject signal patch"""
        self._signal_dict, self._signal_modifiers, self._to_remove = self.patch_to_map(
            signal_patch=signal_patch, return_remove_list=True
        )

    def remove_channel(self, channel_name: Text) -> None:
        """
        Remove channel from the likelihood

        .. versionadded:: 0.1.5

        Args:
            channel_name (``Text``): name of the channel to be removed
        """
        if channel_name in self.channels:
            if channel_name not in self._to_remove:
                self._to_remove.append(channel_name)
        else:
            log.error(
                f"Channel {channel_name} does not exist in the background only model. "
                + "The available channels are "
                + ", ".join(list(self.channels))
            )

    @property
    def remove_list(self) -> List[Text]:
        """
        Channels to be removed from the model

        .. versionadded:: 0.1.5

        """
        return self._to_remove

    def patch_to_map(
        self, signal_patch: List[Dict], return_remove_list: bool = False
    ) -> Union[
        Tuple[Dict[Text, Dict], Dict[Text, Dict], List[Text]],
        Tuple[Dict[Text, Dict], Dict[Text, Dict]],
    ]:
        """
        Convert JSONPatch into signal map

        .. code:: python3

            >>> signal_map = {channel_name: {"data" : signal_yields, "modifiers": signal_modifiers}}


        Args:
            signal_patch (``List[Dict]``): JSONPatch for the signal
            return_remove_list (``bool``, default ``False``): Inclure channels to be removed in the output

                .. versionadded:: 0.1.5

        Returns:
            ``Tuple[Dict[Text, Dict], Dict[Text, Dict], List[Text]]`` or ``Tuple[Dict[Text, Dict], Dict[Text, Dict]]``:
            signal map including the data and modifiers and the list of channels to be removed.
        """
        signal_map, modifier_map, to_remove = {}, {}, []
        for item in signal_patch:
            path = int(item["path"].split("/")[2])
            channel_name = self["channels"][path]["name"]
            if item["op"] == "add":
                signal_map[channel_name] = item["value"]["data"]
                modifier_map[channel_name] = item["value"].get(
                    "modifiers", _default_modifiers(poi_name=self.poi_name[0][1])
                )
            elif item["op"] == "remove":
                to_remove.append(channel_name)
        if return_remove_list:
            return signal_map, modifier_map, to_remove
        return signal_map, modifier_map
