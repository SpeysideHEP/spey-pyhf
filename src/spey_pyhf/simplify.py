r"""
Convert :mod:`pyhf` full statistical models into the simplified likelihood framework.

This module implements :class:`Simplify`, a :class:`spey.ConverterBase`
plug-in that approximates a :mod:`pyhf` ``HistFactory`` likelihood by one
of the three simplified-likelihood backends shipped with :mod:`spey`:

* `"default.correlated_background"` -- multi-bin
  Poisson likelihood with a multivariate-Gaussian constraint on the
  combined background nuisance parameters;
* `"default.third_moment_expansion"` -- extension
  that captures the leading skewness of the per-bin background
  distribution through a quadratic deformation of the expected counts;
* `"default.effective_sigma"` -- variable-Gaussian
  (effective-:math:`\sigma`) treatment of asymmetric per-bin background
  uncertainties.

Mathematical setting
--------------------

Following Buckley, Citron, Fichet, Kraml, Waltenberger and Wardle
(JHEP 04 (2019) 064,
`arXiv:1809.05548 <https://arxiv.org/abs/1809.05548>`_),
an experimental likelihood with
:math:`N` independent elementary nuisance parameters
:math:`\boldsymbol{\delta}` over :math:`P` observed counts
:math:`\{n_I^{\mathrm{obs}}\}` (here :math:`P` is the number of analysis
bins) is approximated, in the regime :math:`N \geq P`, by

.. math::

    \mathcal{L}_S(\boldsymbol{\alpha}, \boldsymbol{\theta})
    = \prod_{I=1}^{P}
      \mathrm{Pois}\!\left(n_I^{\mathrm{obs}} \,\big|\,
        n_{s,I}(\boldsymbol{\alpha}) + a_I + b_I\,\theta_I + c_I\,\theta_I^2
      \right)
      \cdot
      \frac{\exp\!\left(-\tfrac{1}{2}\boldsymbol{\theta}^{\mathrm{T}}
                          \boldsymbol{\rho}^{-1}\boldsymbol{\theta}\right)}
           {\sqrt{(2\pi)^P}} ,

where :math:`\boldsymbol{\alpha}` parametrise the new-physics signal,
:math:`n_{s,I}(\boldsymbol{\alpha})` are the per-bin signal yields and
:math:`\boldsymbol{\theta} = (\theta_1, \dots, \theta_P)` are
*combined* nuisance parameters, one per bin, that summarise the action
of the elementary :math:`\boldsymbol{\delta}`. The latter are
unit-variance, centred Gaussians whose joint correlations are encoded in
the :math:`P \times P` matrix :math:`\boldsymbol{\rho}`. The coefficients
:math:`(a_I, b_I, c_I, \rho_{IJ})` are obtained by matching the first
three central moments of the background expectation at :math:`\mu = 0`
to those of the full likelihood (Buckley *et al.* eqs. 2.6--2.8):

.. math::

    m_{1,I}   &= a_I + c_I , \\
    m_{2,IJ}  &= b_I\,b_J\,\rho_{IJ} + 2\,c_I\,c_J\,\rho_{IJ}^{\,2} , \\
    m_{3,I}   &= 6\,b_I^{\,2}\,c_I + 8\,c_I^{\,3} .

Inverting these relations (eqs. 2.9--2.12) yields

.. math::

    c_I    &= -\mathrm{sign}(m_{3,I})\,\sqrt{2\,m_{2,II}}\,
              \cos\!\left[
                \frac{4\pi}{3} + \frac{1}{3}\,
                \arctan\!\sqrt{8\,\frac{m_{2,II}^{\,3}}{m_{3,I}^{\,2}} - 1}
              \right] , \\
    b_I    &= \sqrt{m_{2,II} - 2\,c_I^{\,2}} , \\
    a_I    &= m_{1,I} - c_I , \\
    \rho_{IJ} &= \frac{1}{4\,c_I\,c_J}\,
                 \left(\sqrt{(b_I\,b_J)^{2} + 8\,c_I\,c_J\,m_{2,IJ}}
                       - b_I\,b_J\right) ,

valid in the regime :math:`8\,m_{2,II}^{\,3} \geq m_{3,I}^{\,2}`. When
:math:`m_{3,I} \rightarrow 0` the quadratic correction vanishes
(:math:`c_I \rightarrow 0`) and the expansion collapses to the standard
simplified likelihood with :math:`a_I = m_{1,I}`,
:math:`b_I = \sqrt{m_{2,II}}` and a multivariate-Gaussian constraint on
:math:`\boldsymbol{\theta}` whose correlation matrix is
:math:`\rho_{IJ} = m_{2,IJ}/(b_I\,b_J)`. The latter case is implemented
by `"default.correlated_background"`, while the
full quadratic form is implemented by
`"default.third_moment_expansion"`.

For asymmetric uncertainties the same combined-parameter strategy is
used, but the per-bin expectation :math:`n^{b}_I + \theta_I\,\sigma_I`
of the standard form is replaced by the variable-Gaussian prescription
of Barlow
(`arXiv:physics/0406120 <https://arxiv.org/abs/physics/0406120>`_,
Sec. 3.6):

.. math::

    \sigma^{\mathrm{eff}}_I(\theta_I)
    = \sqrt{\sigma^{+}_I\,\sigma^{-}_I
            + (\sigma^{+}_I - \sigma^{-}_I)(\theta_I - n^{b}_I)} ,

so that the conditional standard deviation interpolates smoothly
between the upper (:math:`\sigma^{+}`) and lower (:math:`\sigma^{-}`)
absolute uncertainties of the bin. This is the form used by
`"default.effective_sigma"`.

Conversion algorithm
--------------------

The converter implements the Monte-Carlo moment extraction advocated in
Buckley *et al.* Sec. 4. For a user-supplied full statistical model
:math:`\mathcal{L}^{\mathrm{SR}}` (which need not be limited to signal
regions, but is referred to as such here for brevity), the algorithm
proceeds as follows.

1. **Control likelihood.** A control likelihood
   :math:`\mathcal{L}^{c}` is built from the background-only
   :mod:`pyhf` workspace by attaching a zero-yield signal sample to
   every channel listed in ``control_region_indices`` (defaulting to a
   substring-based guess of ``CR``/``VR`` channels). When
   ``include_modifiers_in_control_model`` is true the signal modifiers
   --- and therefore their associated nuisance parameters --- are
   propagated to the control sample so that signal-induced systematics
   contribute to the constraint covariance. Channels outside
   ``control_region_indices`` keep their background-only structure. The
   parameter of interest :math:`\mu` is retained so that the workspace
   has a valid signal+background topology but is fixed to zero in the
   following step and therefore has no effect on the per-bin yields.

2. **Conditional MLE.** :math:`\mathcal{L}^{c}` is profiled at
   :math:`\mu = 0` to obtain the conditional best-fit nuisance vector
   :math:`\hat{\boldsymbol{\theta}}_0^{c}`. Because every signal yield in
   :math:`\mathcal{L}^{c}` is zero at :math:`\mu = 0`, this profile
   coincides with the maximum-likelihood estimate of the
   background-only fit.

3. **Nuisance covariance.** The observed Fisher information

   .. math::

       V^{-1}_{ij}
       = -\,\frac{\partial^{2} \log\mathcal{L}^{c}}{\partial\theta_i\,\partial\theta_j}
         \bigg|_{(\mu,\,\boldsymbol{\theta})
                  = (0,\,\hat{\boldsymbol{\theta}}_0^{c})}

   is evaluated from the Hessian provided by ``pyhf``'s ``jax`` backend
   after deleting the row and column associated with :math:`\mu`. Its
   inverse :math:`\mathbf{V}` is the asymptotic covariance of the
   nuisance parameters at the conditional MLE.

4. **Sampling.** Nuisance draws
   :math:`\tilde{\boldsymbol{\theta}} \sim
   \mathcal{N}(\hat{\boldsymbol{\theta}}_0^{c}, \mathbf{V})` are taken.
   If :math:`\mathcal{L}^{\mathrm{SR}}` has nuisance parameters that
   are *not* present in :math:`\mathcal{L}^{c}` (for instance because
   some signal-only modifiers were excluded from the control model),
   the missing entries are profiled by maximising
   :math:`\mathcal{L}^{\mathrm{SR}}` at :math:`\mu = 0` with the entries
   shared with :math:`\mathcal{L}^{c}` held at
   :math:`\tilde{\boldsymbol{\theta}}` through equality constraints.
   Each accepted parameter vector is forwarded to the :mod:`pyhf`
   sampler of :math:`\mathcal{L}^{\mathrm{SR}}` to draw one Poisson
   realisation per bin (``include_auxiliary=False``). Draws that would
   require sampling from a Poisson with a non-positive rate are
   silently rejected and the loop continues until ``number_of_samples``
   accepted samples are collected.

5. **Moments.** With :math:`\tilde{n}_b` denoting the matrix of per-bin
   samples, the simplified-likelihood inputs are

   .. math::

       m_1 &= \mathbb{E}[\tilde{n}_b] , \\
       \Sigma &= \mathrm{cov}(\tilde{n}_b) , \\
       m_3 &= \mathbb{E}\!\left[(\tilde{n}_b - m_1)^{\,3}\right] ,

   estimated from the sample mean, sample covariance and sample third
   moment, respectively. For
   `"default.effective_sigma"` the symmetric
   :math:`(m_1, \Sigma)` summary is supplemented by the 68% sample
   quantiles that define the per-bin asymmetric envelope

   .. math::

       \sigma^{+}_I &= |\,Q_{0.8413}(\tilde{n}_{b,I}) - m_{1,I}\,| , \\
       \sigma^{-}_I &= |\,m_{1,I} - Q_{0.1587}(\tilde{n}_{b,I})\,| ,

   where :math:`Q_p` denotes the empirical :math:`p`-quantile, and
   :math:`\Sigma` is reduced to the correlation matrix
   :math:`\rho_{IJ} = \Sigma_{IJ}/\sqrt{\Sigma_{II}\Sigma_{JJ}}`.

The resulting summary statistics are passed to the chosen
simplified-likelihood backend, which evaluates the
:math:`(a, b, c, \boldsymbol{\rho})` parameters internally according
to the formulae above and returns the simplified statistical model.
When ``save_model`` is provided, the moments (and the quantile envelopes
for `"default.effective_sigma"`) are persisted to
a compressed ``.npz`` file together with the channel order inferred from
the underlying :mod:`pyhf` configuration, so that the model can be
rebuilt without re-running the sampler.

.. autoclass:: spey_pyhf.simplify.Simplify
    :members:
    :undoc-members:

References
----------

* A. Buckley, M. Citron, S. Fichet, S. Kraml, W. Waltenberger and
  N. Wardle, *The Simplified Likelihood Framework*, JHEP 04 (2019) 064,
  `arXiv:1809.05548 <https://arxiv.org/abs/1809.05548>`_. Defines the
  simplified likelihood, the moment-matching parameters and the
  Monte-Carlo extraction procedure used here.
* R. Barlow, *Asymmetric Errors*,
  `arXiv:physics/0406120 <https://arxiv.org/abs/physics/0406120>`_,
  Sec. 3.6. Source of the variable-Gaussian effective-:math:`\sigma`
  form consumed by `"default.effective_sigma"`.
* E. Schanet, ``simplify`` package
  (`<https://github.com/eschanet/simplify>`_). A complementary approach
  that emits a :mod:`pyhf` patch by collapsing the post-fit background
  into a single sample; its output can be used directly with
  :mod:`spey-pyhf` without going through this converter.
"""

import copy
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import spey
import tqdm
from scipy.stats import moment, multivariate_normal, norm
from spey.backends.default_pdf import (
    CorrelatedBackground,
    EffectiveSigma,
    ThirdMomentExpansion,
)
from spey.helper_functions import covariance_to_correlation
from spey.optimizer.core import fit
from spey.system.exceptions import warning_tracker

from . import WorkspaceInterpreter
from ._version import __version__


def __dir__():
    return []


# pylint: disable=W1203, R0903

log = logging.getLogger("Spey")


class ConversionError(Exception):
    """Conversion error class"""


def make_constraint(index: int, value: float) -> Callable[[np.ndarray], float]:
    """
    Construct constraint

    Args:
        index (``int``): index of the parameter within the parameter list
        value (``float``): fixed value

    Returns:
        ``Callable[[np.ndarray], float]``:
        constraint function
    """

    def func(vector: np.ndarray) -> float:
        return vector[index] - value

    return func


@contextmanager
def _disable_logging(highest_level: int = logging.CRITICAL):
    """
    Temporary disable logging implementation, this should move into Spey

    Args:
        highest_level (``int``, default ``logging.CRITICAL``): highest level to be set in logging
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


class Simplify(spey.ConverterBase):
    r"""
    Convert a :mod:`pyhf` full statistical model into the simplified likelihood framework.

    The converter approximates the input full statistical model by one of
    three :mod:`spey` simplified-likelihood backends:
    `"default.correlated_background"`,
    `"default.third_moment_expansion"` or
    `"default.effective_sigma"`. The methodology
    --- the construction of a control likelihood
    :math:`\mathcal{L}^{c}`, the multivariate-Gaussian sampling of its
    nuisance parameters and the Monte-Carlo extraction of the first few
    central moments of the per-bin background distribution --- is
    described in detail at the
    :doc:`module level <spey_pyhf.simplify>` and follows
    Buckley *et al.*, JHEP 04 (2019) 064
    (`arXiv:1809.05548 <https://arxiv.org/abs/1809.05548>`_). The
    asymmetric variant follows Barlow,
    `arXiv:physics/0406120 <https://arxiv.org/abs/physics/0406120>`_.

    For details on the target simplified-likelihood backends, see
    the `spey default plug-ins page
    <https://spey.readthedocs.io/en/main/plugins.html#default-plug-ins>`_;
    a user-level walk-through is also provided in the
    `spey-pyhf online documentation
    <https://spey-pyhf.readthedocs.io/en/main/simplify.html>`_.

    Args:
        statistical_model (:obj:`~spey.StatisticalModel`): constructed
            full statistical model backed by :mod:`pyhf` with the
            ``jax`` backend enabled. The ``jax`` backend is required
            because the algorithm queries the Hessian of the
            log-likelihood through automatic differentiation.
        fittype (``Text``, default ``"postfit"``): expectation type used
            when constructing and profiling the control model.
            ``"postfit"`` maps to :attr:`spey.ExpectationType.observed`
            (uses the observed auxiliary data) and ``"prefit"`` to
            :attr:`spey.ExpectationType.apriori` (uses the pre-fit
            auxiliary data).
        convert_to (``Text``, default ``"default.correlated_background"``):
            target simplified-likelihood backend. Must be one of
            ``"default.correlated_background"``,
            ``"default.third_moment_expansion"`` or
            ``"default.effective_sigma"``.
        number_of_samples (``int``, default ``1000``): number of accepted
            Monte-Carlo samples used to estimate the background moments
            and, where relevant, the asymmetric quantile envelopes.
            Samples that would require evaluating a Poisson at a
            non-positive rate are rejected and do not count toward this
            total.
        control_region_indices (``List[int]`` or ``List[Text]``, default ``None``):
            indices or names of the control and validation regions in
            the background-only workspace. These are the channels into
            which a zero-yield signal sample is injected when
            constructing :math:`\mathcal{L}^{c}`. If ``None``, the
            interface guesses CR/VR channels via the substring
            heuristic of
            :meth:`~spey_pyhf.helper_functions.WorkspaceInterpreter.guess_CRVR`.
            A :class:`ConversionError` is raised if no CR/VR channel
            can be identified.
        include_modifiers_in_control_model (``bool``, default ``False``):
            if ``True``, the signal modifiers (and therefore their
            nuisance parameters) are attached to the zero-yield signal
            sample injected into the control regions, so that
            signal-induced systematics contribute to the nuisance
            covariance :math:`\mathbf{V}` of :math:`\mathcal{L}^{c}`.
            By default modifiers are excluded.
        save_model (``Text``, default ``None``): full path to which the
            extracted summary statistics are persisted. The data is
            stored as a compressed NumPy archive (``.npz``); the suffix
            is added automatically when missing.

            **Reading the saved model:**

            One can read the saved model using NumPy's :func:`load`
            function

            .. code:: python3

                >>> import numpy as np
                >>> saved_model = np.load("/PATH/TO/DIR/MODELNAME.npz")
                >>> data = saved_model["data"]

            The archive always contains:

            * ``"covariance_matrix"``: :math:`P \times P` covariance
              matrix :math:`\Sigma` between bins, estimated from the
              accepted samples.
            * ``"background_yields"``: per-bin sample mean
              :math:`m_{1,I}`. This is the simplified-likelihood
              background expectation, **not** the original background
              yield of the full :mod:`pyhf` model.
            * ``"data"``: per-bin observed counts
              :math:`n_I^{\mathrm{obs}}` read from the underlying
              :mod:`pyhf` workspace.
            * ``"channel_order"``: channel-name list inferred from the
              :mod:`pyhf` configuration. The simplified backend assumes
              this ordering when interpreting a signal patch.

            Additional keys are written depending on ``convert_to``:

            * ``"third_moments"``: per-bin diagonal third central
              moment :math:`m_{3,I}`, written when ``convert_to ==
              "default.third_moment_expansion"``.
            * ``"absolute_uncertainty_envelops"``: per-bin pairs
              :math:`(\sigma^{+}_I, \sigma^{-}_I)` extracted from the
              68% sample quantiles, written when ``convert_to ==
              "default.effective_sigma"``.

    Raises:
        ``ConversionError``: when ``convert_to`` is not one of the
            supported targets, or when no control region can be
            identified.
        :obj:`AssertionError`: if ``statistical_model`` is not a
            :mod:`pyhf` model, or if its underlying :mod:`pyhf` manager
            does not have the ``jax`` backend enabled.

    **Example:**

    As an example, let us use the JSON files provided for the
    ATLAS-SUSY-2019-08 analysis, which can be found in
    `HEPData <https://www.hepdata.net/record/resource/1934827?landing_page=true>`_.
    Once these are downloaded one can read them and construct a full
    statistical model as follows.

    .. code:: python3

        >>> import json, spey
        >>> with open("1Lbb-likelihoods-hepdata/BkgOnly.json", "r") as f:
        >>>	    background_only = json.load(f)
        >>> with open("1Lbb-likelihoods-hepdata/patchset.json", "r") as f:
        >>>     signal = json.load(f)["patches"][0]["patch"]

        >>> pdf_wrapper = spey.get_backend("pyhf")
        >>> full_statistical_model = pdf_wrapper(
        ...     background_only_model=background_only, signal_patch=signal
        ... )
        >>> full_statistical_model.backend.manager.backend = "jax"

    Note that ``patchset.json`` includes more than one patch set, which
    is why we used only one of them. The last line enables the ``jax``
    backend in the ``pyhf`` interface, which is required to compute the
    Hessian of the statistical model used by the simplification
    procedure.

    The ``"pyhf.simplify"`` converter can then be invoked to map this
    full likelihood onto a simplified-likelihood model.

    .. code:: python3

        >>> converter = spey.get_backend("pyhf.simplify")
        >>> simplified_model = converter(
        ...     statistical_model=full_statistical_model,
        ...     convert_to="default.correlated_background",
        ...     control_region_indices=[
        ...	        'WREM_cuts', 'STCREM_cuts', 'TRHMEM_cuts', 'TRMMEM_cuts', 'TRLMEM_cuts'
        ...	    ]
        ... )
        >>> print(simplified_model.backend_type)
        >>> # "default.correlated_background"
    """

    name: str = "pyhf.simplify"
    """Name of the backend"""
    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = ">=0.2.0,<0.3.0"
    """Spey version required for the backend"""

    @warning_tracker
    def __call__(
        self,
        statistical_model: spey.StatisticalModel,
        fittype: Literal["postfit", "prefit"] = "postfit",
        convert_to: Literal[
            "default.correlated_background",
            "default.third_moment_expansion",
            "default.effective_sigma",
        ] = "default.correlated_background",
        number_of_samples: int = 1000,
        control_region_indices: Optional[Union[List[int], List[str]]] = None,
        include_modifiers_in_control_model: bool = False,
        save_model: Optional[str] = None,
    ) -> Union[CorrelatedBackground, ThirdMomentExpansion, EffectiveSigma]:

        assert statistical_model.backend_type == "pyhf", (
            "This method is currently only available for `pyhf` full statistical models."
            + "For details please see spey-pyhf package."
        )
        assert statistical_model.backend.manager.backend == "jax", (
            "Please enable jax implementation for phyf interface. "
            + "If jax is installed this can be done by "
            + "``statistical_model.backend.manager.backend = 'jax'`` command."
        )

        bkgonly_model = statistical_model.backend.model.background_only_model
        signal_patch = statistical_model.backend.model.signal_patch

        expected = {
            "postfit": spey.ExpectationType.observed,
            "prefit": spey.ExpectationType.apriori,
        }[fittype]

        interpreter = WorkspaceInterpreter(bkgonly_model)
        bin_map = interpreter.bin_map

        # configure signal patch map with respect to channel names
        signal_patch_map, signal_modifiers_map = interpreter.patch_to_map(signal_patch)

        # Prepare a JSON patch to separate control and validation regions
        # These regions are generally marked as CR and VR
        if control_region_indices is None:
            control_region_indices = interpreter.guess_CRVR()

        if len(control_region_indices) == 0:
            raise ConversionError(
                "Can not construct control model. Please provide ``control_region_indices``."
            )

        for channel in interpreter.get_channels(control_region_indices):
            if channel in signal_patch_map and channel in signal_modifiers_map:
                interpreter.inject_signal(
                    channel,
                    [0.0] * bin_map[channel],
                    signal_modifiers_map[channel]
                    if include_modifiers_in_control_model
                    else None,
                )

        pdf_wrapper = spey.get_backend("pyhf")
        with _disable_logging():
            control_model = pdf_wrapper(
                background_only_model=bkgonly_model, signal_patch=interpreter.make_patch()
            )

        # Extract the nuisance parameters that maximises the likelihood at mu=0
        fit_opts = control_model.prepare_for_fit(expected=expected)
        _, fit_param = fit(
            **fit_opts,
            initial_parameters=None,
            fixed_poi_value=0.0,
        )

        # compute the hessian matrix without poi
        control_poi_index = fit_opts["model_configuration"].poi_index
        hessian_neg_logprob = np.delete(
            np.delete(
                -control_model.backend.get_hessian_logpdf_func(expected=expected)(
                    fit_param
                ),
                control_poi_index,
                axis=0,
            ),
            control_poi_index,
            axis=1,
        )

        # Construct multivariate normal distribution to capture correlations
        # between nuisance parameters
        nuisance_cov_matrix = np.linalg.inv(hessian_neg_logprob)
        nuisance_distribution = multivariate_normal(
            mean=np.delete(fit_param, control_poi_index), cov=nuisance_cov_matrix
        )

        # Retreive pyhf models and compare parameter maps
        if include_modifiers_in_control_model:
            stat_model_pyhf = statistical_model.backend.model()[1]
        else:
            # Remove the nuisance parameters from the signal patch
            # Note that even if the signal yields are zero, nuisance parameters
            # do contribute to the statistical model and some models may be highly
            # sensitive to the shape and size of the nuisance parameters.
            with _disable_logging():
                tmp_interpreter = copy.deepcopy(interpreter)
                for channel, data in signal_patch_map.items():
                    tmp_interpreter.inject_signal(channel=channel, data=data)
                tmp_model = spey.get_backend("pyhf")(
                    background_only_model=bkgonly_model,
                    signal_patch=tmp_interpreter.make_patch(),
                )
                stat_model_pyhf = tmp_model.backend.model()[1]
                del tmp_model, tmp_interpreter
        control_model_pyhf = control_model.backend.model()[1]
        is_nuisance_map_different = (
            stat_model_pyhf.config.par_map != control_model_pyhf.config.par_map
        )
        fit_opts = statistical_model.prepare_for_fit(expected=expected)
        suggested_fixed = fit_opts["model_configuration"].suggested_fixed
        log.debug(
            "Number of parameters to be fitted during the scan: "
            f"{fit_opts['model_configuration'].npar - len(fit_param)}"
        )

        samples = []
        with tqdm.tqdm(
            total=number_of_samples,
            unit="sample",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        ) as pbar:
            while len(samples) < number_of_samples:
                nui_smp = nuisance_distribution.rvs(size=(1,))
                current_nui_params = nui_smp.tolist()
                current_nui_params.insert(control_poi_index, 0.0)
                new_params = None

                if is_nuisance_map_different:
                    constraints = []
                    new_params = np.array([np.nan] * stat_model_pyhf.config.npars)
                    for key, item in stat_model_pyhf.config.par_map.items():
                        if key in control_model_pyhf.config.par_map.keys():
                            current_params = current_nui_params[
                                control_model_pyhf.config.par_map[key]["slice"]
                            ]
                            current_range = range(
                                item["slice"].start,
                                item["slice"].stop,
                                1 if item["slice"].step is None else item["slice"].step,
                            )
                            for ival, val in zip(current_range, current_params):
                                new_params[ival] = val
                                if not suggested_fixed[ival]:
                                    constraints.append(
                                        {"type": "eq", "fun": make_constraint(ival, val)}
                                    )

                    if np.any(np.isnan(new_params)):
                        current_fit_opts = copy.deepcopy(fit_opts)
                        init_params = np.where(
                            np.isnan(new_params),
                            current_fit_opts["model_configuration"].suggested_init,
                            new_params,
                        )
                        if np.isnan(current_fit_opts["logpdf"](np.array(init_params))):
                            # if the initial value is NaN continue search
                            continue
                        current_fit_opts["constraints"] += constraints
                        _, new_params = fit(
                            **current_fit_opts,
                            initial_parameters=init_params.tolist(),
                            bounds=current_fit_opts[
                                "model_configuration"
                            ].suggested_bounds,
                        )

                try:
                    current_sample = statistical_model.backend.get_sampler(
                        np.array(current_nui_params if new_params is None else new_params)
                    )(1, include_auxiliary=False)
                    samples.append(current_sample)
                    pbar.update()
                except ValueError:
                    # Some of the samples can lead to problems while sampling from a poisson distribution.
                    # e.g. poisson requires positive lambda values to sample from. If sample leads to a negative
                    # lambda value continue sampling to avoid that point.
                    log.debug("Problem with the sample generation")
                    log.debug(
                        f"Nuisance parameters: {current_nui_params if new_params is None else new_params}"
                    )
                    continue

        samples = np.vstack(samples)

        covariance_matrix = np.cov(samples, rowvar=0)

        data = statistical_model.backend.model.workspace.data(
            stat_model_pyhf, include_auxdata=False
        )

        # NOTE: model spec might be modified within the pyhf workspace, thus
        # yields needs to be reordered properly before constructing the simplified likelihood
        signal_yields, missing_channels = [], []
        for channel_name in stat_model_pyhf.config.channels:
            try:
                signal_yields += signal_patch_map[channel_name]
            except KeyError:
                missing_channels.append(channel_name)
                signal_yields += [0.0] * bin_map[channel_name]
        if len(missing_channels) > 0:
            log.warning(
                "Following channels are not in the signal patch,"
                f" will be set to zero: {', '.join(missing_channels)}"
            )

        # NOTE background yields are first moments in simplified framework not the yield values
        # in the full statistical model!
        background_yields = np.mean(samples, axis=0)

        save_kwargs = {
            "covariance_matrix": covariance_matrix,
            "background_yields": background_yields,
            "data": data,
            "channel_order": stat_model_pyhf.config.channels,
        }

        third_moments = []
        if convert_to == "default.correlated_background":
            backend = CorrelatedBackground(
                signal_yields=signal_yields,
                background_yields=background_yields,
                data=data,
                covariance_matrix=covariance_matrix,
            )
        elif convert_to == "default.third_moment_expansion":
            third_moments = moment(samples, moment=3, axis=0)
            save_kwargs.update({"third_moments": third_moments})

            backend = ThirdMomentExpansion(
                signal_yields=signal_yields,
                background_yields=background_yields,
                data=data,
                covariance_matrix=covariance_matrix,
                third_moment=third_moments,
            )
        elif convert_to == "default.effective_sigma":
            # Get 68% quantiles
            q = (1.0 - (norm.cdf(1.0) - norm.cdf(-1.0))) / 2.0
            # upper and lower uncertainties
            absolute_uncertainty_envelops = np.stack(
                [
                    np.abs(np.quantile(samples, 1 - q, axis=0) - background_yields),
                    np.abs(background_yields - np.quantile(samples, q, axis=0)),
                ],
                axis=1,
            )
            save_kwargs.update(
                {"absolute_uncertainty_envelops": absolute_uncertainty_envelops}
            )

            backend = EffectiveSigma(
                signal_yields=signal_yields,
                background_yields=background_yields,
                data=data,
                correlation_matrix=covariance_to_correlation(
                    covariance_matrix=covariance_matrix
                ),
                absolute_uncertainty_envelops=absolute_uncertainty_envelops,
            )
        else:
            raise ConversionError(
                "Currently available conversion methods are "
                + "'default.correlated_background', 'default.third_moment_expansion',"
                + " 'default.effective_sigma'"
            )

        if save_model is not None:
            save_path = Path(save_model)
            if save_path.suffix != ".npz":
                save_path = save_path.with_suffix(".npz")
            np.savez_compressed(str(save_path), **save_kwargs)

        return backend
