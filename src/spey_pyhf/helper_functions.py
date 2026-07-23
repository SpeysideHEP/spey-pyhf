"""
Helper utilities for creating and interpreting :mod:`pyhf` workspace inputs.

This module provides :class:`WorkspaceInterpreter`, a thin layer around a
``pyhf`` background-only workspace that bookkeeps signal injection, channel
removal and conversion between signal maps and ``JSONPatch`` documents.

It also exposes a small set of pure helper functions that build the patch
operation dictionaries consumed by ``pyhf`` and convenience transformations
of the workspace such as luminosity extrapolation and systematic-uncertainty
rescaling.
"""

import copy
import logging
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple, Union

from spey import log_once

__all__ = ["WorkspaceInterpreter"]


def __dir__():
    return __all__


# pylint: disable=W1203, W1201, C0103, W0212

log = logging.getLogger("Spey")


SYSTEMATIC_MODIFIER_TYPES: FrozenSet[str] = frozenset({"normsys", "histosys", "lumi"})
# Modifier ``type`` values that represent systematic (non-statistical) uncertainties.

STATISTICAL_MODIFIER_TYPES: FrozenSet[str] = frozenset({"staterror", "shapesys"})
# Modifier ``type`` values that represent statistical (Poisson/MC) uncertainties.


def remove_from_json(idx: int) -> Dict[str, str]:
    """
    Build a ``remove`` ``JSONPatch`` operation for a channel.

    Args:
        idx (``int``): index of the channel in ``workspace["channels"]``.

    Returns:
        ``Dict[str, str]``:
        single ``JSONPatch`` operation that removes the channel at ``idx``.
    """
    return {"op": "remove", "path": f"/channels/{idx}"}


def add_to_json(idx: int, yields: List[float], modifiers: List[Dict]) -> Dict:
    """
    Build an ``add`` ``JSONPatch`` operation that injects a signal sample.

    The sample is inserted as ``samples[0]`` of the channel at ``idx`` with the
    name ``"Signal"``.

    Args:
        idx (``int``): index of the channel in ``workspace["channels"]``.
        yields (``List[float]``): signal yields, one entry per bin of the channel.
        modifiers (``List[Dict]``): modifier dictionaries attached to the signal sample.

    Returns:
        ``Dict``:
        single ``JSONPatch`` operation that adds the signal sample to the channel.
    """
    return {
        "op": "add",
        "path": f"/channels/{idx}/samples/0",
        "value": {"name": "Signal", "data": yields, "modifiers": modifiers},
    }


def _default_modifiers(poi_name: str, include_lumi: bool = True) -> List[Dict]:
    """
    Build the default modifier list attached to an injected signal sample.

    The default consists of a luminosity modifier and an unconstrained
    normalisation factor playing the role of the parameter of interest. The
    ``lumi`` modifier is only meaningful if the workspace's measurement
    config already declares a ``lumi`` parameter (with ``auxdata`` and
    ``sigmas``); otherwise ``pyhf`` fails to build the constraint term for it.

    Args:
        poi_name (``str``): name of the parameter of interest declared in the
            measurement configuration.
        include_lumi (``bool``, default ``True``): whether to include the
            ``lumi`` modifier. Should be ``False`` when the background-only
            workspace does not declare a ``lumi`` parameter in its
            measurement config.

    Returns:
        ``List[Dict]``:
        list of modifier dictionaries: optionally a ``lumi`` modifier and a
        ``normfactor`` modifier whose name is ``poi_name``.
    """
    modifiers = [{"data": None, "name": poi_name, "type": "normfactor"}]
    if include_lumi:
        modifiers.insert(0, {"data": None, "name": "lumi", "type": "lumi"})
    return modifiers


def _scale_modifier_for_lumi(modifier: Dict, factor: float) -> None:
    """
    Scale the absolute parts of a modifier in place by a luminosity factor.

    Modifiers whose ``data`` field carries absolute event-count information
    (``histosys`` alternative templates, ``shapesys`` per-bin absolute
    uncertainties, ``staterror`` per-bin absolute uncertainties) are scaled by
    ``factor`` so that, when combined with sample yields scaled by the same
    factor, the *relative* uncertainty per bin is preserved.

    Modifiers whose ``data`` field carries dimensionless or null information
    (``normsys``, ``normfactor``, ``lumi``, ``shapefactor``) are left untouched.

    Args:
        modifier (``Dict``): modifier dictionary as stored in a ``pyhf`` workspace;
            mutated in place.
        factor (``float``): luminosity scale factor (``new_lumi / old_lumi``).
    """
    mod_type = modifier.get("type")
    data = modifier.get("data")
    if data is None:
        return
    if mod_type == "histosys":
        modifier["data"] = {
            "hi_data": [d * factor for d in data["hi_data"]],
            "lo_data": [d * factor for d in data["lo_data"]],
        }
    elif mod_type in ("shapesys", "staterror"):
        modifier["data"] = [d * factor for d in data]


def _rescale_systematic_modifier(
    modifier: Dict, fraction: float, nominal: List[float]
) -> None:
    """
    Rescale the deviation carried by a systematic modifier in place.

    For ``normsys`` the up/down scale factors are rescaled around 1::

        new_hi = 1 + (hi - 1) * fraction
        new_lo = 1 + (lo - 1) * fraction

    For ``histosys`` the alternative templates are rescaled around the
    nominal yields::

        new_hi_data[i] = nominal[i] + (hi_data[i] - nominal[i]) * fraction
        new_lo_data[i] = nominal[i] + (lo_data[i] - nominal[i]) * fraction

    For ``lumi`` the ``data`` field is null in the modifier itself; the
    actual uncertainty lives in ``measurements[*]["config"]["parameters"]``
    and is therefore not rescaled at the modifier level.

    Args:
        modifier (``Dict``): modifier dictionary as stored in a ``pyhf`` workspace;
            mutated in place.
        fraction (``float``): multiplicative factor applied to the deviation
            from the nominal value (``0`` removes the systematic, ``1`` is a
            no-op, ``0.5`` halves the deviation, ...).
        nominal (``List[float]``): nominal sample yields for the bins to which
            the modifier is attached. Only consulted for ``histosys``.
    """
    mod_type = modifier.get("type")
    data = modifier.get("data")
    if mod_type == "normsys" and data is not None:
        modifier["data"] = {
            "hi": 1.0 + (data["hi"] - 1.0) * fraction,
            "lo": 1.0 + (data["lo"] - 1.0) * fraction,
        }
    elif mod_type == "histosys" and data is not None:
        modifier["data"] = {
            "hi_data": [n + (h - n) * fraction for n, h in zip(nominal, data["hi_data"])],
            "lo_data": [n + (l - n) * fraction for n, l in zip(nominal, data["lo_data"])],
        }


class WorkspaceInterpreter:
    """
    Bookkeeping wrapper around a ``pyhf`` background-only workspace.

    The interpreter holds the original background-only ``pyhf`` workspace
    dictionary together with a parallel description of any signal injection,
    control-region masking and modifier configuration provided by the user.
    Once populated it can produce the ``JSONPatch`` document that ``pyhf``
    consumes to build the signal-plus-background statistical model, and it
    can produce derived workspaces with rescaled luminosity or rescaled
    systematic uncertainties.

    Args:
        background_only_model (``Dict``): a valid ``pyhf`` workspace
            description for the background-only fit, containing at least the
            keys ``channels``, ``observations`` and ``measurements``.
    """

    __slots__ = [
        "background_only_model",
        "_signal_dict",
        "_signal_modifiers",
        "_to_remove",
    ]

    def __init__(self, background_only_model: Dict):
        self.background_only_model = background_only_model
        """``pyhf`` workspace description for the background-only fit."""
        self._signal_dict: Dict[str, List[float]] = {}
        self._signal_modifiers: Dict[str, List[Dict]] = {}
        self._to_remove: List[str] = []

    def __getitem__(self, item):
        return self.background_only_model[item]

    @property
    def channels(self) -> Iterator[str]:
        """
        Iterate over the channel names declared in the workspace.

        Returns:
            ``Iterator[str]``:
            generator yielding the channel names in the order they appear in
            ``workspace["channels"]``.
        """
        return (ch["name"] for ch in self["channels"])

    @property
    def has_lumi_parameter(self) -> bool:
        """
        Whether the workspace's measurement config declares a ``lumi`` parameter.

        A ``lumi`` modifier can only be built into a valid ``pyhf`` model if
        the corresponding measurement declares a ``lumi`` entry (with
        ``auxdata`` and ``sigmas``) in ``config["parameters"]``. Without it,
        ``pyhf`` cannot construct the constraint term for the modifier.

        Returns:
            ``bool``:
            ``True`` if any measurement declares a ``lumi`` parameter.
        """
        return any(
            param.get("name") == "lumi"
            for mes in self["measurements"]
            for param in mes.get("config", {}).get("parameters", [])
        )

    @property
    def poi_name(self) -> List[Tuple[str, str]]:
        """
        Parameter-of-interest name for each measurement.

        Returns:
            ``List[Tuple[str, str]]``:
            list of ``(measurement_name, poi_name)`` tuples, one per entry of
            ``workspace["measurements"]``.
        """
        return [(mes["name"], mes["config"]["poi"]) for mes in self["measurements"]]

    @property
    def bin_map(self) -> Dict[str, int]:
        """
        Number of bins for every channel declared in the workspace.

        Returns:
            ``Dict[str, int]``:
            mapping from channel name to the number of bins of its first sample.
        """
        return {ch["name"]: len(ch["samples"][0]["data"]) for ch in self["channels"]}

    @property
    def expected_background_yields(self) -> Dict[str, List[float]]:
        """
        Total expected background yields per channel, given the current configuration.

        Channels listed in :attr:`remove_list` are skipped. A warning is emitted
        once for any channel that is *kept* but has not been configured with a
        signal injection.

        Returns:
            ``Dict[str, List[float]]``:
            mapping from channel name to the bin-wise sum of all sample yields
            contributing to that channel.
        """
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
            log_once(
                "Some of the channels are not defined in the patch set, "
                "these channels will be kept in the statistical model. "
                "If these channels are meant to be removed, please indicate them in the patch set. "
                "Please check the following channel(s): " + ", ".join(undefined_channels),
                log_type="warning",
            )
        return yields

    def guess_channel_type(self, channel_name: str) -> str:
        """
        Heuristically classify a channel as control, validation or signal region.

        The classification is purely string-based: the uppercased channel name
        is searched for the substrings ``"CR"``, ``"VR"`` or ``"SR"`` in that
        order and the first match wins. Any other channel name returns
        ``"__unknown__"``. Because the check is a substring match, channel
        names that happen to contain these letters for unrelated reasons may
        be misclassified.

        Args:
            channel_name (``str``): name of the channel to classify.

        Raises:
            ``ValueError``: if ``channel_name`` is not a channel of this workspace.

        Returns:
            ``str``:
            one of ``"CR"``, ``"VR"``, ``"SR"`` or ``"__unknown__"``.
        """
        if channel_name not in self.channels:
            raise ValueError(f"Unknown channel: {channel_name}")
        for tp in ["CR", "VR", "SR"]:
            if tp in channel_name.upper():
                return tp

        return "__unknown__"

    def guess_CRVR(self) -> List[str]:
        """
        Return all channel names that look like control or validation regions.

        Classification follows :meth:`guess_channel_type`.

        Returns:
            ``List[str]``:
            channel names classified as ``"CR"`` or ``"VR"``.
        """
        return [
            name
            for name in self.channels
            if self.guess_channel_type(name) in ["CR", "VR"]
        ]

    def get_channels(self, channel_index: Union[List[int], List[str]]) -> List[str]:
        """
        Resolve a mix of channel indices and channel names to channel names.

        Args:
            channel_index (``Union[List[int], List[str]]``): indices and/or names
                of the channels to look up.

        Returns:
            ``List[str]``:
            channel names whose index or name appears in ``channel_index``.
        """
        return [
            name
            for idx, name in enumerate(self.channels)
            if idx in channel_index or name in channel_index
        ]

    def inject_signal(
        self, channel: str, data: List[float], modifiers: Optional[List[Dict]] = None
    ) -> None:
        """
        Register a signal injection in one channel of the workspace.

        If ``modifiers`` is provided but does not contain the default ``lumi``
        and ``normfactor`` modifiers (with ``poi_name`` taken from the first
        measurement), they are appended automatically.

        Args:
            channel (``str``): name of the target channel; must already exist
                in the background-only workspace.
            data (``List[float]``): signal yields, one entry per bin of the channel.
            modifiers (``Optional[List[Dict]]``, default ``None``): modifier
                dictionaries to attach to the signal sample. When ``None``,
                :func:`_default_modifiers` is used.

        Raises:
            ``ValueError``: if ``channel`` does not exist in the workspace, or
                if the length of ``data`` does not match the number of bins of
                ``channel``.
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

        default_modifiers = _default_modifiers(
            self.poi_name[0][1], include_lumi=self.has_lumi_parameter
        )
        if modifiers is not None:
            for mod in default_modifiers:
                if mod not in modifiers:
                    log.debug(
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
    def signal_per_channel(self) -> Dict[str, List[float]]:
        """
        Currently registered signal yields, keyed by channel name.

        Returns:
            ``Dict[str, List[float]]``:
            mapping from channel name to the signal yields registered via
            :meth:`inject_signal` or :meth:`add_patch`.
        """
        return self._signal_dict

    def make_patch(self) -> List[Dict]:
        """
        Convert the registered signal injections and removals into a ``JSONPatch``.

        The returned patch list contains, in order, one ``add`` operation per
        channel registered via :meth:`inject_signal`, followed by the
        ``remove`` operations for channels registered via :meth:`remove_channel`,
        sorted in *descending* index order so that earlier indices remain
        valid as ``pyhf`` applies the patch.

        Raises:
            ``ValueError``: if no signal has been registered yet.

        Returns:
            ``List[Dict]``:
            ``JSONPatch`` document for the signal-plus-background workspace.
        """
        if not self._signal_dict:
            raise ValueError("Please add signal yields.")

        patch = []
        to_remove = []
        for ich, channel in enumerate(self.channels):
            if channel in self._to_remove:
                to_remove.append(remove_from_json(ich))
            elif channel in self._signal_dict:
                patch.append(
                    add_to_json(
                        ich, self._signal_dict[channel], self._signal_modifiers[channel]
                    )
                )
            else:
                log.warning(f"Undefined channel in the patch set: {channel}")

        to_remove.sort(key=lambda p: int(p["path"].split("/")[-1]), reverse=True)

        return patch + to_remove

    def reset_signal(self) -> None:
        """Drop all registered signal injections and channel removals."""
        self._signal_dict = {}
        self._to_remove = []

    def add_patch(self, signal_patch: List[Dict]) -> None:
        """
        Replace the current signal configuration with one read from a ``JSONPatch``.

        Args:
            signal_patch (``List[Dict]``): ``JSONPatch`` document, typically
                produced by :meth:`make_patch`, describing signal sample
                additions and channel removals.
        """
        self._signal_dict, self._signal_modifiers, self._to_remove = self.patch_to_map(
            signal_patch=signal_patch, return_remove_list=True
        )

    def remove_channel(self, channel_name: str) -> None:
        """
        Mark a channel to be removed from the likelihood.

        .. versionadded:: 0.1.5

        Args:
            channel_name (``str``): name of the channel to be removed. Channels
                unknown to the workspace produce an error log and no modification.
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
    def remove_list(self) -> List[str]:
        """
        Names of channels marked for removal from the model.

        .. versionadded:: 0.1.5

        Returns:
            ``List[str]``:
            channel names registered via :meth:`remove_channel`.
        """
        return self._to_remove

    def patch_to_map(
        self, signal_patch: List[Dict], return_remove_list: bool = False
    ) -> Union[
        Tuple[Dict[str, List[float]], Dict[str, List[Dict]], List[str]],
        Tuple[Dict[str, List[float]], Dict[str, List[Dict]]],
    ]:
        """
        Convert a ``JSONPatch`` document into the internal signal map.

        .. code:: python3

            >>> signal_map = {channel_name: signal_yields}
            >>> modifier_map = {channel_name: signal_modifiers}

        Args:
            signal_patch (``List[Dict]``): ``JSONPatch`` document for the signal.
            return_remove_list (``bool``, default ``False``): if ``True``, also
                return the list of channel names marked for removal.

                .. versionadded:: 0.1.5

        Returns:
            ``Tuple[Dict[str, List[float]], Dict[str, List[Dict]], List[str]]`` or ``Tuple[Dict[str, List[float]], Dict[str, List[Dict]]]``:
            mapping from channel name to signal yields, mapping from channel
            name to signal modifiers, and (optionally) the list of channel
            names marked for removal.
        """
        signal_map, modifier_map, to_remove = {}, {}, []
        for item in signal_patch:
            path = int(item["path"].split("/")[2])
            channel_name = self["channels"][path]["name"]
            if item["op"] == "add":
                signal_map[channel_name] = item["value"]["data"]
                modifier_map[channel_name] = item["value"].get(
                    "modifiers",
                    _default_modifiers(
                        poi_name=self.poi_name[0][1], include_lumi=self.has_lumi_parameter
                    ),
                )
            elif item["op"] == "remove":
                to_remove.append(channel_name)
        if return_remove_list:
            return signal_map, modifier_map, to_remove
        return signal_map, modifier_map

    def extrapolate_luminosity(self, factor: float) -> "WorkspaceInterpreter":
        """
        Return a luminosity-extrapolated copy of this interpreter.

        Every sample yield, observation count and luminosity-sensitive modifier
        ``data`` field is multiplied by ``factor``. The transformation assumes
        that *relative* uncertainties remain constant, so absolute per-bin
        uncertainties (carried by ``shapesys``, ``staterror`` and ``histosys``
        alternative templates) scale linearly with the yields. Dimensionless
        modifier data (``normsys``, ``normfactor``, ``lumi``, ``shapefactor``)
        is left unchanged.

        Both the background-only workspace and any registered signal injection
        are scaled. The original interpreter is not modified.

        .. versionadded:: 0.2.1

        Args:
            factor (``float``): luminosity scale factor, typically
                ``new_lumi / old_lumi``. Must be strictly positive.

        Raises:
            ``ValueError``: if ``factor`` is not strictly positive.

        Returns:
            ``WorkspaceInterpreter``:
            a new interpreter wrapping a deep copy of the workspace with all
            yields and absolute uncertainties scaled by ``factor``, preserving
            the existing signal injections and channel-removal list.
        """
        if factor <= 0:
            raise ValueError(
                f"Luminosity factor must be strictly positive, got {factor}."
            )

        new_model = copy.deepcopy(self.background_only_model)

        for obs in new_model.get("observations", []):
            obs["data"] = [d * factor for d in obs["data"]]

        for channel in new_model.get("channels", []):
            for sample in channel.get("samples", []):
                sample["data"] = [d * factor for d in sample["data"]]
                for mod in sample.get("modifiers", []):
                    _scale_modifier_for_lumi(mod, factor)

        new_interp = WorkspaceInterpreter(new_model)
        new_interp._to_remove = list(self._to_remove)
        for ch_name, yields in self._signal_dict.items():
            new_interp._signal_dict[ch_name] = [y * factor for y in yields]
            new_modifiers = copy.deepcopy(self._signal_modifiers[ch_name])
            for mod in new_modifiers:
                _scale_modifier_for_lumi(mod, factor)
            new_interp._signal_modifiers[ch_name] = new_modifiers

        return new_interp

    def scale_systematics(
        self,
        fraction: float,
        modifier_types: Optional[List[str]] = None,
    ) -> "WorkspaceInterpreter":
        """
        Return a copy in which systematic-uncertainty deviations are rescaled.

        For each modifier whose ``type`` is in ``modifier_types`` the deviation
        from the nominal value is multiplied by ``fraction``:

        * ``normsys`` up/down scale factors are rescaled around 1, so that a
          fraction of ``0`` makes the systematic vanish (``hi = lo = 1``) and a
          fraction of ``1`` is a no-op;
        * ``histosys`` alternative templates are rescaled around the nominal
          sample yields with the same convention.

        Statistical modifiers (``shapesys``, ``staterror``) are *never* modified
        by this method, regardless of ``modifier_types``: passing one of them
        raises a ``ValueError``. Sample yields and observations are unchanged.

        The original interpreter is not modified.

        .. versionadded:: 0.2.1

        Args:
            fraction (``float``): multiplicative factor applied to each
                systematic deviation. ``1`` is a no-op, ``0`` removes the
                systematic, intermediate values shrink it. Must be non-negative.
            modifier_types (``Optional[List[str]]``, default ``None``):
                modifier ``type`` values to rescale. When ``None``, defaults to
                ``["normsys", "histosys"]``. Statistical modifier types
                (``shapesys``, ``staterror``) are not allowed.

        Raises:
            ``ValueError``: if ``fraction`` is negative, or if ``modifier_types``
                contains a statistical modifier type.

        Returns:
            ``WorkspaceInterpreter``:
            a new interpreter wrapping a deep copy of the workspace with the
            requested systematic deviations rescaled, preserving the existing
            signal injections and channel-removal list.
        """
        if fraction < 0:
            raise ValueError(f"Fraction must be non-negative, got {fraction}.")

        types_to_scale = (
            ["normsys", "histosys"] if modifier_types is None else list(modifier_types)
        )

        invalid = sorted(set(types_to_scale) & STATISTICAL_MODIFIER_TYPES)
        if invalid:
            raise ValueError(
                f"Cannot rescale statistical modifier types: {invalid}. "
                "scale_systematics() only operates on systematic uncertainties."
            )

        new_model = copy.deepcopy(self.background_only_model)

        for channel in new_model.get("channels", []):
            for sample in channel.get("samples", []):
                nominal = list(sample["data"])
                for mod in sample.get("modifiers", []):
                    if mod["type"] in types_to_scale:
                        _rescale_systematic_modifier(mod, fraction, nominal)

        new_interp = WorkspaceInterpreter(new_model)
        new_interp._to_remove = list(self._to_remove)
        for ch_name, yields in self._signal_dict.items():
            new_interp._signal_dict[ch_name] = list(yields)
            new_modifiers = copy.deepcopy(self._signal_modifiers[ch_name])
            for mod in new_modifiers:
                if mod["type"] in types_to_scale:
                    _rescale_systematic_modifier(mod, fraction, yields)
            new_interp._signal_modifiers[ch_name] = new_modifiers

        return new_interp

    def summary(
        self,
        measurement_name: Optional[str] = None,
        show_samples: bool = False,
        show_modifiers: bool = False,
        max_channels: int = 50,
    ) -> None:
        """
        Print a human-readable summary of the workspace and the signal injection state.

        The header reports workspace-level statistics (version, number of
        channels, measurements and observations). Each measurement is listed
        with its parameter of interest and parameter count. For every channel
        the summary shows its guessed region type (``CR`` / ``VR`` / ``SR``),
        bin count, observation total, expected-background total, sample count
        and an aggregated count of modifier types attached to its samples.
        Injected signals and channels marked for removal are listed at the
        bottom.

        .. versionadded:: 0.2.1

        Args:
            measurement_name (``Optional[str]``, default ``None``): if given,
                restrict the per-measurement section to the named measurement.
            show_samples (``bool``, default ``False``): if ``True``, list
                every sample name and its yield total beneath each channel.
            show_modifiers (``bool``, default ``False``): if ``True``, list
                every modifier name and type per sample. Implies
                ``show_samples``.
            max_channels (``int``, default ``50``): maximum number of channels
                to print per measurement.
        """
        ws = self.background_only_model
        sep = "=" * 60

        print(sep)
        print("pyhf Workspace Summary")
        print(f"  version       : {ws.get('version', '?')}")
        print(f"  channels      : {len(ws.get('channels', []))}")
        print(f"  measurements  : {len(ws.get('measurements', []))}")
        print(f"  observations  : {len(ws.get('observations', []))}")
        print()

        bm = self.bin_map
        obs_map = {o["name"]: o.get("data", []) for o in ws.get("observations", [])}
        channels = ws.get("channels", [])
        measurements = ws.get("measurements", [])

        if not measurements:
            print("(no measurements declared)")
            print(sep)
            return

        for mes in measurements:
            if measurement_name is not None and mes["name"] != measurement_name:
                continue
            cfg = mes.get("config", {})
            params = cfg.get("parameters", [])
            print(f"Measurement : {mes['name']}")
            print(f"  POI         : {cfg.get('poi', '?')}")
            if params:
                shown = ", ".join(p.get("name", "?") for p in params[:6])
                extra = f" ... (+{len(params) - 6} more)" if len(params) > 6 else ""
                print(f"  Parameters  : {len(params)} ({shown}{extra})")
            else:
                print("  Parameters  : 0")
            print(f"  Channels ({len(channels)}):")

            for i, ch in enumerate(channels[:max_channels]):
                cname = ch["name"]
                region = self.guess_channel_type(cname)
                tag = f"[{region}]" if region != "__unknown__" else "[   ]"
                n_bins = bm.get(cname, 0)
                samples = ch.get("samples", [])
                n_samples = len(samples)

                mod_counts: Dict[str, int] = {}
                for s in samples:
                    for m in s.get("modifiers", []):
                        mod_counts[m["type"]] = mod_counts.get(m["type"], 0) + 1
                mod_str = (
                    ", ".join(f"{k}:{v}" for k, v in sorted(mod_counts.items()))
                    if mod_counts
                    else "-"
                )

                obs_total = sum(obs_map.get(cname, []))
                bg_total = sum(sum(s.get("data", []) or []) for s in samples)

                inj_str = ""
                if cname in self._signal_dict:
                    sig_total = sum(self._signal_dict[cname])
                    inj_str = f"  <- signal: total={sig_total:g}"
                removed_str = "  [REMOVED]" if cname in self._to_remove else ""

                bin_word = "bin" if n_bins == 1 else "bins"
                smp_word = "sample" if n_samples == 1 else "samples"
                print(
                    f"    {i+1:4d}. {tag} {cname}"
                    f"  ({n_bins} {bin_word}, {n_samples} {smp_word},"
                    f" obs={obs_total:g}, bkg={bg_total:g}, mod={{{mod_str}}})"
                    f"{inj_str}{removed_str}"
                )

                if show_samples or show_modifiers:
                    for s in samples:
                        sname = s.get("name", "?")
                        yield_total = sum(s.get("data", []) or [])
                        modifiers = s.get("modifiers", [])
                        mod_word = "modifier" if len(modifiers) == 1 else "modifiers"
                        print(
                            f"              sample: {sname}"
                            f"  (total={yield_total:g}, {len(modifiers)} {mod_word})"
                        )
                        if show_modifiers:
                            for m in modifiers:
                                print(
                                    f"                  modifier: {m.get('name', '?')}"
                                    f" ({m.get('type', '?')})"
                                )

            if len(channels) > max_channels:
                print(
                    f"    ... ({len(channels) - max_channels} more not shown; "
                    f"increase max_channels to see all)"
                )
            print()

        if self._signal_dict:
            print(f"Injected signal: {len(self._signal_dict)} channel(s)")
            for cname, yields in self._signal_dict.items():
                n_bins = len(yields)
                total = sum(yields)
                mod_types = sorted(
                    {m.get("type", "?") for m in self._signal_modifiers.get(cname, [])}
                )
                mod_str = ", ".join(mod_types) if mod_types else "-"
                bin_word = "bin" if n_bins == 1 else "bins"
                print(
                    f"  {cname}: {n_bins} {bin_word}, total={total:g}, "
                    f"modifiers={{{mod_str}}}"
                )
            print()

        if self._to_remove:
            print(f"Channels to remove ({len(self._to_remove)}):")
            for cname in self._to_remove:
                print(f"  {cname}")
            print()

        print(sep)
