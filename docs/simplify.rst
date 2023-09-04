Converting full statistical models to simplified likelihood framework
=====================================================================

Although full statistical models contain all the necessary information
to reconstruct the original analysis, it might be computationally costly. 
Thus, we implement methodologies to convert full likelihoods into simplified 
likelihood frameworks using ``"default_pdf.correlated_background"`` or 
``"default_pdf.third_moment_expansion"`` models. Details on the 
`simplified models can be found in this link <https://speysidehep.github.io/spey/plugins.html#default-plug-ins>`_.

Methodology
-----------

The Simplified likelihood framework contracts all the nuisance parameters 
into a single bin and represents the background uncertainty as a single source. 
To capture the correlations between nuisance parameters, one needs to construct 
a statistical model only from control and validation regions, which is ideally 
purely dominated by the background, henceforth called the control model. Once 
nuisance parameters are fitted for the control model without the signal, one can 
compute the covariance matrix between the nuisance parameters using the Hessian of 
the negative log probability distribution,

.. math::

    \mathbf{V}^{-1}_{ij} = - \frac{\partial^2 \log\mathcal{L}^{\rm control}(0,\theta_0^{\rm control})}
    {\partial\theta_i\partial\theta_j}

.. .. note::

..     Here we disregard possible signal leakage into the control or validation regions 
..     by setting all signal yields to zero. This is also ensured due to :math:`\mu=0`.


Covariance matrix :math:`\mathbf{V}_{ij}` allows construction of a multivariate
Gaussian distribution :math:`\mathcal{N}(\theta_0^{\rm control}, \mathbf{V}_{ij})` 
where one can sample nuisance parameters, 
:math:`\tilde{\theta}_0^{\rm control}\sim\mathcal{N}(\theta_0^{\rm control}, \mathbf{V}_{ij})`, 
without loosing the correlation between them.

To construct the covariance matrix between background yields within the signal 
regions, we will use :math:`\tilde{\theta}_0^{\rm control}` along with 
:math:`\mu=0` to create a sampler where each :math:`\tilde{n}_b` will be sampled 
from an independent :math:`\tilde{\theta}_0^{\rm control}`;

.. math::

    \tilde{n}_b \sim \mathcal{L}^{\rm original}(0, \tilde{\theta}_0^{\rm control})

The covariance matrix, :math:`\Sigma`, is constructed from the collection of 
:math:`\tilde{n}_b`. To extend the methodology to ``"default_pdf.third_moment_expansion"`` 
one can also compute the third moments directly from the collection of :math:`\tilde{n}_b`.

.. .. note::

..     Auxiliary data is not included in :math:`\tilde{n}_b`.

.. note::

    There has been other techniques employed to simplify the full statistical models
    one can find `such a method in this GitHub repository <https://github.com/eschanet/simplify>`_. 
    Since this approach provides a ``pyhf`` compatible dictionary as an output, it 
    can be directly used with ``spey-pyhf`` plug-in without any additional modifications.
    The method presented here is different from their approach.

Usage
-----

A full statistical model can be constructed using a background only JSON serialised file 
(usually found in HEPData entry for a given analysis). Details on how to construct a full
likelihood through ``spey-pyhf`` interface can be found in 
:ref:`this section <sec_quick_start>`.

Following the details in previous sections, a statistical model using ``pyhf`` interface
can be constructed as

.. code:: python3

    >>> pdf_wrapper = spey.get_backend("pyhf")
    >>> full_statistical_model = pdf_wrapper(
    ...     background_only_model=background_only, signal_patch=signal
    ... )

where ``background_only`` refers to background only JSON file retreived from HEPData and 
``signal`` refers to a signal patch constructed by the user. ``full_statistical_model``
can be converted into simplified likelihood by using ``pyhf.simplify`` backend.

.. code:: python3

    >>> converter = spey.get_backend("pyhf.simplify")
    >>> simplified_model = converter(
    ...     statistical_model=full_statistical_model, 
    ...     convert_to="default_pdf.correlated_background",
    ... )

**Arguments:**

    * ``statistical_model``: Statistical model constructed using ``pyhf`` backend.
    * ``expected``: Flag to choose if the fit to be realised with respect to the data or 
      background yields, default ``spey.ExpectationType.observed``.
    * ``convert_to``: Which simplified framework to be used as a baseline for the conversion,
      default ``"default_pdf.correlated_background"``.
    * ``number_of_samples``: Sets the number of samples to be generated to construct covariance
      matrix, :math:`\Sigma`, for the background bins, default ``1000``.
    * ``control_region_indices``: Usually algorithm can pick up the differences between signal, 
      control and validation regions, however there is no fixed convention in naming which lead to 
      choosing wrong channels for the construction of the :math:`\mathcal{L}^{\rm control}`. One can
      overwrite the system selection by providing the indices of the control and validation regions
      within the channel list from the background only statistical model dictionary. This can be 
      found by iterating over ``background_only["channels"]``.

Acknowledgements
----------------

This functionality has been discussed and requested during 
`8th (Re)interpretation Forum <https://conference.ippp.dur.ac.uk/event/1178/>`_.
Thanks to Nicholas Wardle and Wolfgang Waltenberger for the lively discussion.