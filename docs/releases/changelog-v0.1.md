# Release notes v0.1

## New features since the last release

* A mapping functionality has been developed between ``pyhf``'s full
  statistical models and simplified likelihoods framework. Methodology
  and usage can be found in the online documentation.
  ([#2](https://github.com/SpeysideHEP/spey-pyhf/pull/2))

* Full statistical model mapping to effective sigma model have been implemented.
  ([#14](https://github.com/SpeysideHEP/spey-pyhf/pull/14))

## Improvements

* Sampler functionality has been extended to isolate auxiliary data.
  ([#3](https://github.com/SpeysideHEP/spey-pyhf/pull/3))

* Fixed parameters have been added to the constraints list.
  ([#2](https://github.com/SpeysideHEP/spey-pyhf/pull/2))

* Update `pyhf` version.
  ([#8](https://github.com/SpeysideHEP/spey-pyhf/pull/8))

* Model loading has been improved for prefit and postfit scenarios.
  ([#10](https://github.com/SpeysideHEP/spey-pyhf/pull/10))

* Improve undefined channel handling in the patchset
  ([#12](https://github.com/SpeysideHEP/spey-pyhf/pull/12))

* Improve undefined channel handling in the patchset for full likelihood simplification.
  ([#13](https://github.com/SpeysideHEP/spey-pyhf/pull/13))

* Add modifier check to signal injection.
  ([#13](https://github.com/SpeysideHEP/spey-pyhf/pull/13))

## Bug fixes

* Bugfix in `simplify` module, where signal injector was not initiated properly.
 ([#9](https://github.com/SpeysideHEP/spey-pyhf/pull/9))

* Bugfix in apriori likelihood computation for full likelihoods mentioned in
  [#5](https://github.com/SpeysideHEP/spey-pyhf/issues/5).
  ([#2](https://github.com/SpeysideHEP/spey-pyhf/pull/2))

* Bugfix in uncertainty quantification for full statistical model mapping on effective sigma
  ([#15](https://github.com/SpeysideHEP/spey-pyhf/pull/15))

* Issue with implementing removed channels to the patch
  ([#16](https://github.com/SpeysideHEP/spey-pyhf/pull/16))

* Issue with the sorting of the removed channels.
  ([#17](https://github.com/SpeysideHEP/spey-pyhf/pull/17))

## Contributors

This release contains contributions from (in alphabetical order):

* [Jack Araz](https://github.com/jackaraz)
