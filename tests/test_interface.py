"""Regression tests for the ``pyhf`` backends of ``spey``.

The reference values were computed with pyhf v0.7.6 and agree with
pyhf v0.8.0, so these tests guard the numerical results against changes
in pyhf.
"""

import copy
import importlib.util

import numpy as np
import pytest

import spey
from spey_pyhf import manager

jax = pytest.param(
    "jax",
    marks=pytest.mark.skipif(
        importlib.util.find_spec("jax") is None, reason="jax is not installed"
    ),
)
# Tests that need automatic differentiation, which only the jax backend provides
jax_only = pytest.mark.parametrize("backend", [jax], indirect=True)


@pytest.fixture(params=["numpy", jax])
def backend(request, monkeypatch):
    """Set the pyhf backend, restoring the original one afterwards as it is global."""
    monkeypatch.setattr(manager, "backend", request.param)
    return request.param


@pytest.fixture
def background_only():
    """Background-only workspace with a two-bin signal region and a one-bin control region."""
    return {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "background",
                        "data": [50.0, 52.0],
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": [3.0, 7.0],
                            },
                            {
                                "name": "bkgnorm",
                                "type": "normsys",
                                "data": {"hi": 1.1, "lo": 0.9},
                            },
                        ],
                    }
                ],
            },
            {
                "name": "CR",
                "samples": [
                    {
                        "name": "background",
                        "data": [100.0],
                        "modifiers": [
                            {
                                "name": "bkgnorm",
                                "type": "normsys",
                                "data": {"hi": 1.1, "lo": 0.9},
                            },
                            {"name": "cr_stat", "type": "staterror", "data": [5.0]},
                        ],
                    }
                ],
            },
        ],
        "observations": [
            {"name": "SR", "data": [51.0, 48.0]},
            {"name": "CR", "data": [103.0]},
        ],
        "measurements": [
            {"name": "Measurement", "config": {"poi": "mu", "parameters": []}}
        ],
        "version": "1.0.0",
    }


def _signal(channel_index, data):
    # Written out rather than built with WorkspaceInterpreter so that these tests
    # stay independent of the code under test in test_helper_functions.py
    return {
        "op": "add",
        "path": f"/channels/{channel_index}/samples/1",
        "value": {
            "name": "signal",
            "data": data,
            "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
        },
    }


@pytest.fixture
def signal_patch():
    return [_signal(0, [12.0, 11.0])]


@pytest.fixture
def full_model(background_only, signal_patch):
    return spey.get_backend("pyhf")(
        background_only_model=background_only, signal_patch=signal_patch
    )


@pytest.mark.parametrize(
    ("expected", "reference"),
    [
        (spey.ExpectationType.observed, [0.918726]),
        (
            spey.ExpectationType.aposteriori,
            [0.993439, 0.972009, 0.894028, 0.680558, 0.335583],
        ),
        (
            spey.ExpectationType.apriori,
            [0.993182, 0.971181, 0.891859, 0.676641, 0.331812],
        ),
    ],
)
def test_full_model_exclusion_confidence_level(backend, full_model, expected, reference):
    np.testing.assert_allclose(
        full_model.exclusion_confidence_level(expected=expected), reference, rtol=1e-4
    )


def test_full_model_fit_and_upper_limit(backend, full_model):
    np.testing.assert_allclose(
        full_model.maximize_likelihood(), [-0.119778, 14.584527], rtol=1e-4
    )
    assert full_model.poi_upper_limit() == pytest.approx(1.130877, rel=1e-4)


@jax_only
def test_full_model_hessian(backend, full_model):
    assert full_model.sigma_mu_from_hessian(poi_test=1.0) == pytest.approx(
        0.628702, rel=1e-4
    )


def test_uncorrelated_background(backend):
    model = spey.get_backend("pyhf.uncorrelated_background")(
        signal_yields=[12.0, 11.0],
        background_yields=[50.0, 52.0],
        data=[51, 48],
        absolute_uncertainties=[3.0, 7.0],
    )
    # cf. the README example, which uses the same model through a full workspace
    np.testing.assert_allclose(model.exclusion_confidence_level(), [0.947485], rtol=1e-4)
    assert model.poi_upper_limit() == pytest.approx(1.011569, rel=1e-4)


def test_combine(backend, background_only):
    # combine requires a signal patch for every channel, so only use the signal region
    first = background_only
    first["channels"] = first["channels"][:1]
    first["observations"] = first["observations"][:1]

    second = copy.deepcopy(first)
    second["channels"][0]["name"] = second["observations"][0]["name"] = "SR_2"
    # shapesys modifier names must be unique across the combined workspace
    second["channels"][0]["samples"][0]["modifiers"][0]["name"] += "_2"

    pyhf_backend = spey.get_backend("pyhf")
    combined = pyhf_backend(
        background_only_model=first, signal_patch=[_signal(0, [12.0, 11.0])]
    ).combine(
        pyhf_backend(
            background_only_model=second, signal_patch=[_signal(0, [12.0, 11.0])]
        )
    )
    np.testing.assert_allclose(
        combined.exclusion_confidence_level(), [0.938468], rtol=1e-4
    )


@jax_only
def test_simplify(backend, background_only, signal_patch):
    full_model = spey.get_backend("pyhf")(
        background_only_model=background_only,
        signal_patch=signal_patch + [_signal(1, [1.0])],
    )
    # Simplify samples nuisance parameters from NumPy's global random state
    np.random.seed(42)
    simplified = spey.get_backend("pyhf.simplify")(
        statistical_model=full_model,
        control_region_indices=["CR"],
        number_of_samples=500,
    )

    assert simplified.backend_type == "default.correlated_background"
    np.testing.assert_allclose(
        simplified.exclusion_confidence_level(), [0.864845], rtol=1e-4
    )
