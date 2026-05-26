"""Unit tests for :mod:`spey_pyhf.helper_functions`."""
import copy

import pytest

from spey_pyhf.helper_functions import (
    STATISTICAL_MODIFIER_TYPES,
    SYSTEMATIC_MODIFIER_TYPES,
    WorkspaceInterpreter,
    _default_modifiers,
    _rescale_systematic_modifier,
    _scale_modifier_for_lumi,
    add_to_json,
    remove_from_json,
)


def make_workspace():
    """Build a small but feature-complete pyhf workspace dictionary for tests.

    The workspace covers a 2-bin signal region with all relevant modifier
    types, a 1-bin control region, a 1-bin validation region and a "Misc"
    channel that does not fall in any CR/VR/SR pattern.
    """
    return {
        "channels": [
            {
                "name": "SR1",
                "samples": [
                    {
                        "name": "background",
                        "data": [10.0, 20.0],
                        "modifiers": [
                            {"name": "lumi", "type": "lumi", "data": None},
                            {
                                "name": "norm_b",
                                "type": "normsys",
                                "data": {"hi": 1.1, "lo": 0.9},
                            },
                            {
                                "name": "shape_b",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [11.0, 22.0],
                                    "lo_data": [9.0, 18.0],
                                },
                            },
                            {
                                "name": "stat_b",
                                "type": "staterror",
                                "data": [1.0, 2.0],
                            },
                            {
                                "name": "shapesys_b",
                                "type": "shapesys",
                                "data": [0.5, 1.0],
                            },
                        ],
                    }
                ],
            },
            {
                "name": "CR1",
                "samples": [
                    {
                        "name": "bg",
                        "data": [50.0],
                        "modifiers": [
                            {"name": "lumi", "type": "lumi", "data": None},
                        ],
                    }
                ],
            },
            {
                "name": "VR1",
                "samples": [
                    {
                        "name": "bg",
                        "data": [30.0],
                        "modifiers": [
                            {"name": "lumi", "type": "lumi", "data": None},
                        ],
                    }
                ],
            },
            {
                "name": "Misc",
                "samples": [{"name": "bg", "data": [5.0], "modifiers": []}],
            },
        ],
        "observations": [
            {"name": "SR1", "data": [10, 20]},
            {"name": "CR1", "data": [50]},
            {"name": "VR1", "data": [30]},
            {"name": "Misc", "data": [5]},
        ],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {
                            "name": "lumi",
                            "auxdata": [1.0],
                            "bounds": [[0.9, 1.1]],
                            "inits": [1.0],
                            "sigmas": [0.025],
                        }
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


# ---------- Module-level helpers ----------


def test_remove_from_json():
    assert remove_from_json(3) == {"op": "remove", "path": "/channels/3"}


def test_add_to_json():
    mods = [{"name": "x", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}}]
    out = add_to_json(0, [1.0, 2.0], mods)
    assert out["op"] == "add"
    assert out["path"] == "/channels/0/samples/0"
    assert out["value"]["name"] == "Signal"
    assert out["value"]["data"] == [1.0, 2.0]
    assert out["value"]["modifiers"] is mods


def test_default_modifiers():
    mods = _default_modifiers("mu")
    assert {"name": "lumi", "type": "lumi", "data": None} in mods
    assert {"name": "mu", "type": "normfactor", "data": None} in mods
    assert len(mods) == 2


def test_classification_constants_are_disjoint():
    assert "normsys" in SYSTEMATIC_MODIFIER_TYPES
    assert "histosys" in SYSTEMATIC_MODIFIER_TYPES
    assert "lumi" in SYSTEMATIC_MODIFIER_TYPES
    assert "staterror" in STATISTICAL_MODIFIER_TYPES
    assert "shapesys" in STATISTICAL_MODIFIER_TYPES
    assert SYSTEMATIC_MODIFIER_TYPES.isdisjoint(STATISTICAL_MODIFIER_TYPES)


def test_scale_modifier_for_lumi_handles_each_type():
    histosys = {
        "name": "h",
        "type": "histosys",
        "data": {"hi_data": [2.0, 4.0], "lo_data": [1.0, 2.0]},
    }
    _scale_modifier_for_lumi(histosys, 3.0)
    assert histosys["data"] == {"hi_data": [6.0, 12.0], "lo_data": [3.0, 6.0]}

    shapesys = {"name": "s", "type": "shapesys", "data": [0.1, 0.2]}
    _scale_modifier_for_lumi(shapesys, 2.0)
    assert shapesys["data"] == [0.2, 0.4]

    staterror = {"name": "st", "type": "staterror", "data": [1.0, 2.0]}
    _scale_modifier_for_lumi(staterror, 4.0)
    assert staterror["data"] == [4.0, 8.0]

    normsys = {"name": "n", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}}
    _scale_modifier_for_lumi(normsys, 5.0)
    assert normsys["data"] == {"hi": 1.1, "lo": 0.9}

    lumi_mod = {"name": "lumi", "type": "lumi", "data": None}
    _scale_modifier_for_lumi(lumi_mod, 5.0)
    assert lumi_mod["data"] is None


def test_rescale_systematic_modifier():
    normsys = {"name": "n", "type": "normsys", "data": {"hi": 1.2, "lo": 0.8}}
    _rescale_systematic_modifier(normsys, 0.5, [10.0])
    assert normsys["data"] == {"hi": 1.1, "lo": 0.9}

    histosys = {
        "name": "h",
        "type": "histosys",
        "data": {"hi_data": [12.0, 24.0], "lo_data": [8.0, 16.0]},
    }
    _rescale_systematic_modifier(histosys, 0.5, [10.0, 20.0])
    assert histosys["data"]["hi_data"] == [11.0, 22.0]
    assert histosys["data"]["lo_data"] == [9.0, 18.0]

    # Statistical / null-data types are no-ops.
    staterror = {"name": "s", "type": "staterror", "data": [1.0]}
    _rescale_systematic_modifier(staterror, 0.0, [10.0])
    assert staterror["data"] == [1.0]


# ---------- Existing WorkspaceInterpreter API ----------


def test_channels_iteration_yields_all_names():
    wi = WorkspaceInterpreter(make_workspace())
    assert list(wi.channels) == ["SR1", "CR1", "VR1", "Misc"]


def test_channels_property_is_fresh_iterator_each_call():
    wi = WorkspaceInterpreter(make_workspace())
    first = list(wi.channels)
    second = list(wi.channels)
    assert first == second == ["SR1", "CR1", "VR1", "Misc"]


def test_bin_map():
    wi = WorkspaceInterpreter(make_workspace())
    assert wi.bin_map == {"SR1": 2, "CR1": 1, "VR1": 1, "Misc": 1}


def test_poi_name():
    wi = WorkspaceInterpreter(make_workspace())
    assert wi.poi_name == [("Measurement", "mu")]


def test_getitem_proxies_to_underlying_dict():
    ws = make_workspace()
    wi = WorkspaceInterpreter(ws)
    assert wi["channels"] is ws["channels"]
    assert wi["measurements"] is ws["measurements"]


def test_guess_channel_type_classifies_correctly():
    wi = WorkspaceInterpreter(make_workspace())
    assert wi.guess_channel_type("SR1") == "SR"
    assert wi.guess_channel_type("CR1") == "CR"
    assert wi.guess_channel_type("VR1") == "VR"
    assert wi.guess_channel_type("Misc") == "__unknown__"


def test_guess_channel_type_unknown_channel_raises():
    wi = WorkspaceInterpreter(make_workspace())
    with pytest.raises(ValueError, match="Unknown channel"):
        wi.guess_channel_type("DoesNotExist")


def test_guess_CRVR_returns_only_cr_and_vr():
    wi = WorkspaceInterpreter(make_workspace())
    assert sorted(wi.guess_CRVR()) == ["CR1", "VR1"]


def test_get_channels_by_index_and_name():
    wi = WorkspaceInterpreter(make_workspace())
    assert wi.get_channels([0, "VR1"]) == ["SR1", "VR1"]
    assert wi.get_channels([1, 3]) == ["CR1", "Misc"]
    assert wi.get_channels([]) == []


def test_inject_signal_default_modifiers():
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [1.0, 2.0])
    assert wi.signal_per_channel == {"SR1": [1.0, 2.0]}
    mods = wi._signal_modifiers["SR1"]
    assert {"data": None, "name": "lumi", "type": "lumi"} in mods
    assert {"data": None, "name": "mu", "type": "normfactor"} in mods


def test_inject_signal_appends_missing_default_modifiers():
    wi = WorkspaceInterpreter(make_workspace())
    custom = [{"name": "extra", "type": "normsys", "data": {"hi": 1.05, "lo": 0.95}}]
    wi.inject_signal("SR1", [1.0, 2.0], modifiers=custom)
    types = [m["type"] for m in wi._signal_modifiers["SR1"]]
    assert types.count("normsys") == 1
    assert "lumi" in types
    assert "normfactor" in types


def test_inject_signal_keeps_user_modifiers_when_complete():
    wi = WorkspaceInterpreter(make_workspace())
    full = [
        {"data": None, "name": "lumi", "type": "lumi"},
        {"data": None, "name": "mu", "type": "normfactor"},
    ]
    wi.inject_signal("SR1", [1.0, 2.0], modifiers=full)
    assert len(wi._signal_modifiers["SR1"]) == 2


def test_inject_signal_invalid_channel():
    wi = WorkspaceInterpreter(make_workspace())
    with pytest.raises(ValueError, match="does not exist"):
        wi.inject_signal("DoesNotExist", [1.0])


def test_inject_signal_wrong_size():
    wi = WorkspaceInterpreter(make_workspace())
    with pytest.raises(ValueError, match="Number of bins"):
        wi.inject_signal("SR1", [1.0])


def test_make_patch_no_signal_raises():
    wi = WorkspaceInterpreter(make_workspace())
    with pytest.raises(ValueError, match="add signal"):
        wi.make_patch()


def test_make_patch_with_signal_and_removal():
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [1.0, 2.0])
    wi.remove_channel("CR1")
    wi.remove_channel("VR1")
    patch = wi.make_patch()

    add_ops = [p for p in patch if p["op"] == "add"]
    rm_ops = [p for p in patch if p["op"] == "remove"]
    assert len(add_ops) == 1
    assert add_ops[0]["path"] == "/channels/0/samples/0"
    assert add_ops[0]["value"]["data"] == [1.0, 2.0]
    # Removes must be sorted descending by index so earlier indices stay valid.
    assert [int(p["path"].split("/")[-1]) for p in rm_ops] == [2, 1]


def test_remove_channel_unknown_does_not_modify_remove_list():
    wi = WorkspaceInterpreter(make_workspace())
    wi.remove_channel("NotAChannel")
    assert wi.remove_list == []


def test_remove_channel_idempotent():
    wi = WorkspaceInterpreter(make_workspace())
    wi.remove_channel("CR1")
    wi.remove_channel("CR1")
    assert wi.remove_list == ["CR1"]


def test_reset_signal_clears_all_state():
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [1.0, 2.0])
    wi.remove_channel("CR1")
    wi.reset_signal()
    assert wi.signal_per_channel == {}
    assert wi.remove_list == []


def test_patch_to_map_and_add_patch_roundtrip():
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [1.0, 2.0])
    wi.remove_channel("CR1")
    patch = wi.make_patch()

    wi2 = WorkspaceInterpreter(make_workspace())
    wi2.add_patch(patch)
    assert wi2.signal_per_channel == {"SR1": [1.0, 2.0]}
    assert wi2.remove_list == ["CR1"]


def test_patch_to_map_returns_two_or_three_items():
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [1.0, 2.0])
    wi.remove_channel("CR1")
    patch = wi.make_patch()

    two = wi.patch_to_map(patch)
    assert len(two) == 2

    three = wi.patch_to_map(patch, return_remove_list=True)
    assert len(three) == 3
    assert three[2] == ["CR1"]


def test_expected_background_yields_skips_removed_channels():
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [1.0, 2.0])
    wi.inject_signal("CR1", [0.5])
    wi.inject_signal("VR1", [0.3])
    wi.inject_signal("Misc", [0.1])
    wi.remove_channel("Misc")
    yields = wi.expected_background_yields
    assert "Misc" not in yields
    assert yields["SR1"] == [10.0, 20.0]
    assert yields["CR1"] == [50.0]
    assert yields["VR1"] == [30.0]


# ---------- extrapolate_luminosity ----------


def test_extrapolate_luminosity_invalid_factor():
    wi = WorkspaceInterpreter(make_workspace())
    with pytest.raises(ValueError):
        wi.extrapolate_luminosity(0)
    with pytest.raises(ValueError):
        wi.extrapolate_luminosity(-1.0)


def test_extrapolate_luminosity_returns_new_instance():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.extrapolate_luminosity(2.0)
    assert isinstance(new_wi, WorkspaceInterpreter)
    assert new_wi is not wi
    assert new_wi.background_only_model is not wi.background_only_model


def test_extrapolate_luminosity_scales_yields_and_observations():
    ws = make_workspace()
    wi = WorkspaceInterpreter(ws)
    new_wi = wi.extrapolate_luminosity(2.0)

    # Original is untouched.
    assert wi["channels"][0]["samples"][0]["data"] == [10.0, 20.0]
    assert wi["observations"][0]["data"] == [10, 20]

    # New is doubled.
    assert new_wi["channels"][0]["samples"][0]["data"] == [20.0, 40.0]
    assert new_wi["observations"][0]["data"] == [20, 40]
    assert new_wi["channels"][1]["samples"][0]["data"] == [100.0]
    assert new_wi["observations"][1]["data"] == [100]


def test_extrapolate_luminosity_scales_absolute_modifiers():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.extrapolate_luminosity(3.0)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}

    # histosys: alternative templates scale with luminosity
    assert mods["shape_b"]["data"]["hi_data"] == [33.0, 66.0]
    assert mods["shape_b"]["data"]["lo_data"] == [27.0, 54.0]

    # staterror & shapesys: absolute per-bin uncertainties scale linearly
    assert mods["stat_b"]["data"] == [3.0, 6.0]
    assert mods["shapesys_b"]["data"] == [1.5, 3.0]


def test_extrapolate_luminosity_leaves_relative_modifiers_alone():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.extrapolate_luminosity(3.0)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}

    assert mods["norm_b"]["data"] == {"hi": 1.1, "lo": 0.9}
    assert mods["lumi"]["data"] is None


def test_extrapolate_luminosity_preserves_signal_and_remove_list():
    wi = WorkspaceInterpreter(make_workspace())
    sig_mods = [
        {
            "name": "sig_shape",
            "type": "histosys",
            "data": {"hi_data": [2.2, 4.4], "lo_data": [1.8, 3.6]},
        }
    ]
    wi.inject_signal("SR1", [2.0, 4.0], modifiers=copy.deepcopy(sig_mods))
    wi.remove_channel("CR1")

    new_wi = wi.extrapolate_luminosity(2.5)

    # Signal yields scaled, original unchanged
    assert wi.signal_per_channel["SR1"] == [2.0, 4.0]
    assert new_wi.signal_per_channel["SR1"] == [5.0, 10.0]

    new_signal_mods = {m["name"]: m for m in new_wi._signal_modifiers["SR1"]}
    assert new_signal_mods["sig_shape"]["data"]["hi_data"] == [5.5, 11.0]
    assert new_signal_mods["sig_shape"]["data"]["lo_data"] == [4.5, 9.0]

    # remove_list is preserved but copied.
    assert new_wi.remove_list == ["CR1"]
    assert new_wi.remove_list is not wi.remove_list


def test_extrapolate_luminosity_does_not_alias_original_workspace():
    ws = make_workspace()
    wi = WorkspaceInterpreter(ws)
    new_wi = wi.extrapolate_luminosity(2.0)
    # Mutate the new workspace; verify the old one is untouched.
    new_wi["channels"][0]["samples"][0]["data"][0] = 999.0
    new_wi["channels"][0]["samples"][0]["modifiers"][2]["data"]["hi_data"][0] = 999.0
    assert ws["channels"][0]["samples"][0]["data"][0] == 10.0
    assert ws["channels"][0]["samples"][0]["modifiers"][2]["data"]["hi_data"][0] == 11.0


def test_extrapolate_luminosity_make_patch_uses_scaled_signal():
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [2.0, 4.0])
    new_wi = wi.extrapolate_luminosity(3.0)
    add_ops = [p for p in new_wi.make_patch() if p["op"] == "add"]
    assert add_ops[0]["value"]["data"] == [6.0, 12.0]


def test_extrapolate_luminosity_preserves_relative_uncertainty():
    """Relative uncertainty per bin should be invariant under luminosity scaling."""
    wi = WorkspaceInterpreter(make_workspace())
    factor = 4.0
    new_wi = wi.extrapolate_luminosity(factor)

    old_yield = wi["channels"][0]["samples"][0]["data"]
    old_stat = wi["channels"][0]["samples"][0]["modifiers"][3]["data"]
    new_yield = new_wi["channels"][0]["samples"][0]["data"]
    new_stat = new_wi["channels"][0]["samples"][0]["modifiers"][3]["data"]

    for i in range(len(old_yield)):
        assert old_stat[i] / old_yield[i] == pytest.approx(new_stat[i] / new_yield[i])


# ---------- scale_systematics ----------


def test_scale_systematics_invalid_fraction():
    wi = WorkspaceInterpreter(make_workspace())
    with pytest.raises(ValueError, match="non-negative"):
        wi.scale_systematics(-0.5)


def test_scale_systematics_rejects_statistical_types():
    wi = WorkspaceInterpreter(make_workspace())
    with pytest.raises(ValueError, match="statistical"):
        wi.scale_systematics(0.5, modifier_types=["staterror"])
    with pytest.raises(ValueError, match="statistical"):
        wi.scale_systematics(0.5, modifier_types=["shapesys"])
    with pytest.raises(ValueError, match="statistical"):
        wi.scale_systematics(0.5, modifier_types=["normsys", "shapesys"])


def test_scale_systematics_returns_new_instance():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.5)
    assert isinstance(new_wi, WorkspaceInterpreter)
    assert new_wi is not wi
    assert new_wi.background_only_model is not wi.background_only_model


def test_scale_systematics_rescales_normsys():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.5)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}
    # hi: 1 + (1.1 - 1) * 0.5 = 1.05
    # lo: 1 + (0.9 - 1) * 0.5 = 0.95
    assert mods["norm_b"]["data"] == {"hi": 1.05, "lo": 0.95}


def test_scale_systematics_rescales_histosys():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.5)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}
    # nominal = [10, 20]
    # hi: [10 + (11-10)*0.5, 20 + (22-20)*0.5] = [10.5, 21.0]
    # lo: [10 + (9-10)*0.5,  20 + (18-20)*0.5] = [9.5, 19.0]
    assert mods["shape_b"]["data"]["hi_data"] == [10.5, 21.0]
    assert mods["shape_b"]["data"]["lo_data"] == [9.5, 19.0]


def test_scale_systematics_zero_removes_systematic_effect():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.0)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}
    assert mods["norm_b"]["data"] == {"hi": 1.0, "lo": 1.0}
    assert mods["shape_b"]["data"]["hi_data"] == [10.0, 20.0]
    assert mods["shape_b"]["data"]["lo_data"] == [10.0, 20.0]


def test_scale_systematics_one_is_identity():
    ws = make_workspace()
    wi = WorkspaceInterpreter(ws)
    new_wi = wi.scale_systematics(1.0)
    mods_old = {m["name"]: m for m in ws["channels"][0]["samples"][0]["modifiers"]}
    mods_new = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}
    assert mods_new["norm_b"]["data"] == mods_old["norm_b"]["data"]
    assert mods_new["shape_b"]["data"] == mods_old["shape_b"]["data"]


def test_scale_systematics_preserves_statistical_modifiers():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.0)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}
    assert mods["stat_b"]["data"] == [1.0, 2.0]
    assert mods["shapesys_b"]["data"] == [0.5, 1.0]


def test_scale_systematics_preserves_yields_and_observations():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.0)
    assert new_wi["channels"][0]["samples"][0]["data"] == [10.0, 20.0]
    assert new_wi["observations"][0]["data"] == [10, 20]


def test_scale_systematics_does_not_alias_original_workspace():
    ws = make_workspace()
    wi = WorkspaceInterpreter(ws)
    new_wi = wi.scale_systematics(0.5)
    new_wi["channels"][0]["samples"][0]["modifiers"][1]["data"]["hi"] = 999
    assert ws["channels"][0]["samples"][0]["modifiers"][1]["data"]["hi"] == 1.1


def test_scale_systematics_preserves_signal_and_remove_list():
    wi = WorkspaceInterpreter(make_workspace())
    sig_mods = [
        {"name": "sig_norm", "type": "normsys", "data": {"hi": 1.2, "lo": 0.8}},
        {"name": "sig_stat", "type": "staterror", "data": [0.5, 0.5]},
    ]
    wi.inject_signal("SR1", [2.0, 4.0], modifiers=copy.deepcopy(sig_mods))
    wi.remove_channel("CR1")

    new_wi = wi.scale_systematics(0.5)
    new_signal_mods = {m["name"]: m for m in new_wi._signal_modifiers["SR1"]}

    # Systematic on the signal is rescaled.
    assert new_signal_mods["sig_norm"]["data"] == {"hi": 1.1, "lo": 0.9}
    # Statistical on the signal is left alone.
    assert new_signal_mods["sig_stat"]["data"] == [0.5, 0.5]
    # Yields are unchanged.
    assert new_wi.signal_per_channel["SR1"] == [2.0, 4.0]
    # remove_list is preserved.
    assert new_wi.remove_list == ["CR1"]
    assert new_wi.remove_list is not wi.remove_list


def test_scale_systematics_custom_modifier_types_subset():
    """Restricting to ``normsys`` only must leave ``histosys`` untouched."""
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.0, modifier_types=["normsys"])
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}
    assert mods["norm_b"]["data"] == {"hi": 1.0, "lo": 1.0}
    assert mods["shape_b"]["data"] == {
        "hi_data": [11.0, 22.0],
        "lo_data": [9.0, 18.0],
    }


# ---------- Composability ----------


def test_extrapolate_then_scale_systematics():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.extrapolate_luminosity(2.0).scale_systematics(0.5)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}

    # After lumi=2: yields=[20,40], histosys hi_data=[22,44]
    # After systematic=0.5: [20+(22-20)*0.5, 40+(44-40)*0.5] = [21.0, 42.0]
    assert new_wi["channels"][0]["samples"][0]["data"] == [20.0, 40.0]
    assert mods["shape_b"]["data"]["hi_data"] == [21.0, 42.0]
    assert mods["shape_b"]["data"]["lo_data"] == [19.0, 38.0]
    # Statistical modifiers were scaled by lumi but not by systematic-scaling.
    assert mods["stat_b"]["data"] == [2.0, 4.0]


def test_scale_systematics_then_extrapolate():
    wi = WorkspaceInterpreter(make_workspace())
    new_wi = wi.scale_systematics(0.5).extrapolate_luminosity(2.0)
    mods = {m["name"]: m for m in new_wi["channels"][0]["samples"][0]["modifiers"]}
    # After systematic=0.5: histosys hi_data=[10.5, 21.0]
    # After lumi=2: hi_data=[21.0, 42.0]
    assert mods["shape_b"]["data"]["hi_data"] == [21.0, 42.0]
    assert mods["norm_b"]["data"] == {"hi": 1.05, "lo": 0.95}


# ---------- summary ----------


def test_summary_basic_output(capsys):
    wi = WorkspaceInterpreter(make_workspace())
    wi.summary()
    out = capsys.readouterr().out

    assert "pyhf Workspace Summary" in out
    assert "version       : 1.0.0" in out
    assert "channels      : 4" in out
    assert "measurements  : 1" in out
    assert "Measurement : Measurement" in out
    assert "POI         : mu" in out
    # All channels are listed with their region tag
    assert "[SR] SR1" in out
    assert "[CR] CR1" in out
    assert "[VR] VR1" in out
    assert "Misc" in out
    # No injection / removal sections when nothing is configured
    assert "Injected signal" not in out
    assert "Channels to remove" not in out


def test_summary_reports_modifier_counts():
    """SR1 has all five modifier types attached to its background sample."""
    import io
    import contextlib

    wi = WorkspaceInterpreter(make_workspace())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wi.summary()
    out = buf.getvalue()
    # SR1 row reports each modifier type with its count
    sr_line = next(line for line in out.splitlines() if "SR1" in line)
    for mod_type in ("lumi", "normsys", "histosys", "staterror", "shapesys"):
        assert f"{mod_type}:1" in sr_line


def test_summary_with_signal_and_removal(capsys):
    wi = WorkspaceInterpreter(make_workspace())
    wi.inject_signal("SR1", [2.0, 4.0])
    wi.remove_channel("Misc")
    wi.summary()
    out = capsys.readouterr().out

    # Channel row shows signal total and removed flag
    sr_line = next(line for line in out.splitlines() if "SR1" in line)
    assert "signal: total=6" in sr_line
    misc_line = next(line for line in out.splitlines() if " Misc" in line)
    assert "[REMOVED]" in misc_line

    # Dedicated injection and removal sections
    assert "Injected signal: 1 channel(s)" in out
    assert "SR1: 2 bins, total=6" in out
    assert "Channels to remove (1):" in out


def test_summary_show_samples_and_modifiers(capsys):
    wi = WorkspaceInterpreter(make_workspace())
    wi.summary(show_modifiers=True)
    out = capsys.readouterr().out
    assert "sample: background" in out
    assert "modifier: norm_b (normsys)" in out
    assert "modifier: shape_b (histosys)" in out
    assert "modifier: stat_b (staterror)" in out


def test_summary_filters_by_measurement_name(capsys):
    """A non-matching measurement name must produce a header but no per-measurement detail."""
    wi = WorkspaceInterpreter(make_workspace())
    wi.summary(measurement_name="NotARealMeasurement")
    out = capsys.readouterr().out
    assert "pyhf Workspace Summary" in out
    assert "Measurement : Measurement" not in out
    assert "Channels (" not in out


def test_summary_max_channels_truncates(capsys):
    wi = WorkspaceInterpreter(make_workspace())
    wi.summary(max_channels=2)
    out = capsys.readouterr().out
    assert "SR1" in out
    assert "CR1" in out
    assert "VR1" not in out
    assert "2 more not shown" in out


def test_summary_handles_empty_measurements(capsys):
    ws = make_workspace()
    ws["measurements"] = []
    wi = WorkspaceInterpreter(ws)
    wi.summary()
    out = capsys.readouterr().out
    assert "(no measurements declared)" in out
