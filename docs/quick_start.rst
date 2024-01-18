Quick Start
===========

Installation
------------

``spey-pyhf`` plug-in is available at `pypi <https://pypi.org>`_ , so it can be installed by running:

.. code-block:: bash

    >>> pip install spey-pyhf


Python >=3.8 is required. This will also automatically install ``pyhf`` since it is a requirement.
For Gradient and Hessian implementations in likelihood optimisation, we recommend also installing ``Jax``.

Once this package is installed, ``spey`` can automatically detect it, which can be tested using 
:func:`~spey.AvailableBackends` function;

.. code-block:: python3

    >>> import spey
    >>> print(spey.AvailableBackends())
    >>> # ['default_pdf.correlated_background',
    >>> #  'default_pdf.effective_sigma',
    >>> #  'default_pdf.third_moment_expansion',
    >>> #  'default_pdf.uncorrelated_background',
    >>> #  'pyhf',
    >>> #  'pyhf.uncorrelated_background']

.. _sec_quick_start:

First Steps
-----------

``"pyhf"`` accessor enables user to access all ``pyhf``'s likelihood building capabilities.
The function of this accessor accepts a background-only statistical model dictionary as described
in `pyhf's online documentation <https://pyhf.readthedocs.io/en/v0.7.2/likelihood.html>`_. Additionally,
a signal patch sample needs to be provided. An example of these two can be found below.

.. code-block:: python3

    >>> background_only = {
    ...     "channels": [
    ...         {
    ...             "name": "singlechannel",
    ...             "samples": [
    ...                 {
    ...                     "name": "background",
    ...                     "data": [50.0, 52.0],
    ...                     "modifiers": [
    ...                         {
    ...                             "name": "uncorr_bkguncrt",
    ...                             "type": "shapesys",
    ...                             "data": [3.0, 7.0],
    ...                         }
    ...                     ],
    ...                 }
    ...             ],
    ...         }
    ...     ],
    ...     "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
    ...     "measurements": [{"name": "Measurement", "config": {"poi": "mu", "parameters": []}}],
    ...     "version": "1.0.0",
    ... }

    >>> signal = [
    ...     {
    ...         "op": "add",
    ...         "path": "/channels/0/samples/1",
    ...         "value": {
    ...             "name": "signal",
    ...             "data": [12.0, 11.0],
    ...             "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    ...         },
    ...     }
    ... ]

The ``background_only`` dictionary refers to the statistical model that encapsulates only the background
, and the ``signal`` includes the signal-only patch set. In the following, we demonstrate the usage of these
descriptions within ``spey``;

.. code-block:: python3

    >>> import spey

    >>> stat_wrapper = spey.get_backend("pyhf")
    >>> statistical_model = stat_wrapper(
    ...     analysis="simple_pyhf",
    ...     background_only_model=background_only,
    ...     signal_patch=signal,
    ... )

    >>> statistical_model.exclusion_confidence_level() # [0.9474850259721279]

For the rest of the functionalities, please refer to the ``spey`` documentation, which can be found 
`in this link <https://speysidehep.github.io/spey/>`_. Due to Spey's fully
backend agnostic structure, all the functionalities of the :class:`~spey.StatisticalModel` class also
applies to ``pyhf`` plug-in.

**Arguments:**

 * ``background_only_model``: This background-only model dictionary includes information about
   background yields, uncertainties and observations. Details on constructing these dictionaries can be
   found in `pyhf's online documentation <https://pyhf.readthedocs.io/en/v0.7.2/likelihood.html>`_.
 * ``signal_patch``: This signal patch includes dictionaries describing which regions will be added or 
   removed from the statistical model.
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.


Additionally, this plug-in is shipped with simple uncorrelated background-attachment which accesses 
``pyhf``'s ``uncorrelated_backgound`` function can be accessed through spey with the following function

.. code-block:: python3

    >>> import spey

    >>> pdf_wrapper = spey.get_backend('pyhf.uncorrelated_background')

    >>> data = [1]
    >>> signal_yields = [0.5]
    >>> background_yields = [2.0]
    >>> background_unc = [1.1]

    >>> stat_model = pdf_wrapper(
    ...     signal_yields=signal_yields,
    ...     background_yields=background_yields,
    ...     data=data,
    ...     absolute_uncertainties=background_unc,
    ...     analysis="single_bin",
    ...     xsection=0.123,
    ... )

    >>> statistical_model.exclusion_confidence_level() # [0.32907621368190676]

**Arguments:**

 * ``signal_yields``: signal yields as a list.
 * ``background_yields``: background yields as a list.
 * ``data``: observations as a list.
 * ``absolute_uncertainties``: uncertainties on the background as a list.
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.


.. note::

    ``pyhf`` offers an interface to combine the likelihoods that are described as JSON serialised 
    files. This has been exploited in ``spey`` interface via :func:`combine` `function <https://speysidehep.github.io/spey/api.html#spey.StatisticalModel.combine>`_.
    This function combines ``pyhf`` workspaces and adjusts the signal structure accordingly. For more information
    about how ``pyhf`` handles the workspace combination `see the dedicated tutorial here <https://pyhf.github.io/pyhf-tutorial/Combinations.html>`_.