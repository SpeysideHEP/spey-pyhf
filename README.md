# spey-pyhf

[![arxiv](https://img.shields.io/static/v1?style=plastic&label=arXiv&message=2307.06996&color=brightgreen)](https://arxiv.org/abs/2307.06996)
[![DOI](https://img.shields.io/static/v1?style=plastic&label=DOI&message=10.21468/SciPostPhys.16.1.032&color=blue)](https://scipost.org/10.21468/SciPostPhys.16.1.032)
[![DOI](https://zenodo.org/badge/617041391.svg)](https://zenodo.org/badge/latestdoi/617041391)
[![doc](https://img.shields.io/static/v1?style=plastic&label&message=Documentation&logo=gitbook&logoColor=white&color=gray)](http://spey-pyhf.readthedocs.io/)

<img src="https://raw.githubusercontent.com/SpeysideHEP/spey/main/docs/img/spey-plug-in.png" alt="Spey logo" style="float: right; margin-right: 20px" align="right" width=250px/>

[![github](https://img.shields.io/static/v1?style=plastic&label&message=GitHub&logo=github&logoColor=black&color=white)](https://github.com/SpeysideHEP/spey-pyhf)
[![PyPI - Version](https://img.shields.io/pypi/v/spey-pyhf?style=plastic)](https://pypi.org/project/spey-pyhf/)
[![Documentation Status](https://readthedocs.org/projects/spey-pyhf/badge/?version=main&style=plastic)](https://spey-pyhf.readthedocs.io)
[![GitHub License](https://img.shields.io/github/license/SpeysideHEP/spey-pyhf?style=plastic)](https://github.com/SpeysideHEP/spey-pyhf/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/SpeysideHEP/spey-pyhf/graph/badge.svg?token=OMIVV8Q0OA)](https://codecov.io/gh/SpeysideHEP/spey-pyhf)

## About

`spey-pyhf` is a [`pyhf`](https://pyhf.readthedocs.io/) plug-in for the [`spey`](https://speysidehep.github.io/spey/) statistical inference interface. It exposes `pyhf`'s full likelihood-building machinery — including HistFactory-style JSON workspaces and signal patches — through Spey's backend-agnostic API, so any analysis that ships a `pyhf` likelihood can be consumed by Spey alongside other backends. Once installed, Spey automatically detects the plug-in and registers three backends: `pyhf` (full statistical model from a background-only workspace plus a signal patch), `pyhf.uncorrelated_background` (a lightweight uncorrelated-background helper) and `pyhf.simplify` (workspace simplification).

## Installation

The plug-in is published on [PyPI](https://pypi.org/project/spey-pyhf/) and can be installed with:

```bash
python -m pip install spey-pyhf
```

To install directly from the `main` branch:

```bash
python -m pip install --upgrade "git+https://github.com/SpeysideHEP/spey-pyhf"
```

**Note that the `main` branch may not be the stable version.**

## Quick example

Given a `pyhf` background-only workspace and a signal patch, you can build a Spey statistical model and compute the exclusion confidence level in a few lines:

```python
import spey

background_only = {
    "channels": [
        {
            "name": "singlechannel",
            "samples": [
                {
                    "name": "background",
                    "data": [50.0, 52.0],
                    "modifiers": [
                        {"name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0]}
                    ],
                }
            ],
        }
    ],
    "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
    "measurements": [{"name": "Measurement", "config": {"poi": "mu", "parameters": []}}],
    "version": "1.0.0",
}

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

wrapper = spey.get_backend("pyhf")
model = wrapper(
    analysis="simple_pyhf",
    background_only_model=background_only,
    signal_patch=signal_patch,
)

print(model.exclusion_confidence_level())  # [0.9474850259721279]
```

Because Spey is fully backend-agnostic, every method of `spey.StatisticalModel` (likelihood scans, upper limits, combinations, etc.) works the same way on top of `pyhf`. For the full API and tutorials, see the [online documentation](http://spey-pyhf.readthedocs.io/).
