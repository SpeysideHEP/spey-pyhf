# Release notes v0.2

## Important notes

* `spey-pyhf` now requires `pyhf` `v0.8.x` and Python 3.10 or later. Please read the [pyhf `v0.8.0` release notes](https://github.com/scikit-hep/pyhf/releases/tag/v0.8.0) for the changes it brings.

* `pyhf` `v0.8.0` raises `pyhf.exceptions.InvalidModel` for model specifications with duplicate channel, sample, or modifier names. Workspaces with duplicate names that loaded with earlier versions will now raise an error.

## Improvements

* The `jax` extra now takes its requirements from the `pyhf[jax]` extra, which requires `jax>=0.4.1` and `jaxlib>=0.4.1`.

* `tqdm` is declared as a direct dependency, as `pyhf` `v0.8.0` no longer requires it.

* Numerical regression tests have been added for the `pyhf` and `pyhf.uncorrelated_background` backends, model combination, and the `pyhf.simplify` converter.

## Bug fixes

* Setting the `pyhf` backend now defers to `pyhf` to decide which backends are supported, and raises `MethodNotAvailable` for an unsupported or unimportable backend. TensorFlow is no longer listed as an available backend, as `pyhf` `v0.8.0` removed it.

## Contributors

This release contains contributions from (in alphabetical order):

* [Matthew Feickert](https://github.com/matthewfeickert)
