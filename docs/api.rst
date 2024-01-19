Description of all functions and classes
========================================

.. meta::
    :property=og:title: API
    :property=og:description: Description of all functions and classes
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey-pyhf.readthedocs.io/en/main/api.html

Managers
--------

.. currentmodule:: spey_pyhf

.. autosummary:: 
    :toctree: _generated/

    manager.PyhfManager
    interface.ModelNotDefined
    interface.CombinationError
    simplify.ConversionError

Interface
---------

.. autoclass:: spey_pyhf.interface.UncorrelatedBackground
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: spey_pyhf.interface.FullStatisticalModel
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: spey_pyhf.simplify.Simplify
    :members:
    :undoc-members:

Data classes
------------

.. autoclass:: spey_pyhf.data.Base
    :members:
    :undoc-members:

.. autoclass:: spey_pyhf.data.FullStatisticalModelData
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: spey_pyhf.data.SimpleModelData
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Helper functions
----------------

.. autoclass:: spey_pyhf.WorkspaceInterpreter
    :members:
    :undoc-members: