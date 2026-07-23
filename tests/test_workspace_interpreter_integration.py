"""Integration tests for :class:`WorkspaceInterpreter` combined with ``spey``.

These mirror the two-workflow comparison in ``notebooks/test.ipynb``: building
a signal-plus-background ``pyhf`` model by hand versus building the same
model through :class:`~spey_pyhf.helper_functions.WorkspaceInterpreter`, for a
background-only workspace whose measurement config declares no ``lumi``
parameter.
"""

import pytest

import spey
from spey_pyhf import WorkspaceInterpreter


@pytest.fixture
def background_only():
    return {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "background",
                        "data": [50.0, 52.0],
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": [3.0, 7.0],
                            }
                        ],
                    }
                ],
            }
        ],
        "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
        "measurements": [
            {"name": "Measurement", "config": {"poi": "mu", "parameters": []}}
        ],
        "version": "1.0.0",
    }


def test_manual_signal_patch_exclusion_cl(background_only):
    """Hand-built signal patch (normfactor-only) matches the reference value."""
    signal_patch = [
        {
            "op": "add",
            "path": "/channels/0/samples/1",
            "value": {
                "name": "signal",
                "data": [12.0, 11.0],
                "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
            },
        }
    ]

    model = spey.get_backend("pyhf")(
        analysis="simple_pyhf",
        background_only_model=background_only,
        signal_patch=signal_patch,
    )

    cls = model.exclusion_confidence_level()
    assert cls == pytest.approx([0.9474850258730709], abs=1e-6)


def test_workspace_interpreter_reproduces_manual_patch(background_only):
    """``WorkspaceInterpreter`` with no custom modifiers matches the manual patch.

    Regression test: since ``background_only``'s measurement config declares
    no ``lumi`` parameter, the interpreter must not auto-attach a ``lumi``
    modifier (it previously did unconditionally, which made ``pyhf`` build an
    invalid model and raise ``TypeError: 'NoneType' object is not iterable``).
    """
    interp = WorkspaceInterpreter(background_only)
    interp.inject_signal("singlechannel", [12.0, 11.0])

    assert interp.has_lumi_parameter is False

    patch = interp.make_patch()
    assert patch == [
        {
            "op": "add",
            "path": "/channels/0/samples/0",
            "value": {
                "name": "Signal",
                "data": [12.0, 11.0],
                "modifiers": [{"data": None, "name": "mu", "type": "normfactor"}],
            },
        }
    ]

    model = spey.get_backend("pyhf")(
        analysis="simple_pyhf",
        background_only_model=interp.background_only_model,
        signal_patch=patch,
    )
    cls = model.exclusion_confidence_level()
    assert cls == pytest.approx([0.9474850258730709], abs=1e-6)


def test_workspace_interpreter_with_custom_histosys_modifier(background_only):
    """A custom ``histosys`` modifier is kept and ``mu`` is still auto-appended."""
    interp = WorkspaceInterpreter(background_only)
    interp.inject_signal(
        "singlechannel",
        [12.0, 11.0],
        modifiers=[
            {
                "name": "Wolfgang",
                "type": "histosys",
                "data": {"hi_data": [13, 12], "lo_data": [11, 10]},
            }
        ],
    )

    mod_types = {mod["type"] for mod in interp._signal_modifiers["singlechannel"]}
    assert mod_types == {"histosys", "normfactor"}

    model = spey.get_backend("pyhf")(
        analysis="simple_pyhf",
        background_only_model=interp.background_only_model,
        signal_patch=interp.make_patch(),
    )
    cls = model.exclusion_confidence_level()
    assert cls == pytest.approx([0.9445701166818461], abs=1e-6)
