---
myst:
  html_meta:
    "property=og:title": "Utility belt for spey-pyhf plug-in"
    "property=og:description": "Various helper functions to be used with spey-pyhf plug-in"
    "property=og:image": "https://spey.readthedocs.io/en/main/_static/spey-logo.png"
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Utility belt for spey-pyhf plug-in

## Working with full likelihoods from HEPData

The `spey-pyhf` plug-in includes a set of helper functions to manipulate the input data in a spey-ready fashion.

```{code-cell} ipython3
:tags: [hide-cell]
:caption: Lets import necessary libraries first
import spey
from spey_pyhf.helper_functions import WorkspaceInterpreter
import json
```

We will use an example background-only JSON file which is downloaded from HEPData, but you can use any background-only JSON file.

```{code-block} python
:caption: Load the background file. You can use any background only JSON file for this example. Simply adjust the rest of the code accordingly.
with open("/PATH/TO/BkgOnly.json", "r") as f:
    bkg_only = json.load(f)
```

{py:obj}`spey_pyhf.helper_functions.WorkspaceInterpreter` scans the background-only dictionary and extracts relevant information such as channel names:

```{code-cell} ipython3
interpreter = WorkspaceInterpreter(bkg_only)
print(list(interpreter.channels))
```

```python
['WREM_cuts', 'STCREM_cuts', 'TRHMEM_cuts', 'TRMMEM_cuts', 'TRLMEM_cuts', 'SRHMEM_mct2', 'SRMMEM_mct2', 'SRLMEM_mct2']
```

and information about bin sizes in each channel

```{code-cell} ipython3
print(interpreter.bin_map['SRHMEM_mct2'])
```

```python
3
```

we can inject signal to any channel we like

````{margin}
```{admonition} Attention!
:class: attention
 Notice that the rest of the channels will be added without any signal yields. If some of these channels need to be removed from the patch set, they can be added to the remove list via the ``remove_channel()`` function. **Note:** This behaviour has been updated in ``v0.1.5``. In the older versions, the channels that were not declared were removed.
```
````

```{code-cell} ipython3
interpreter.inject_signal('SRHMEM_mct2', [5.0, 12.0, 4.0])
```

Notice that we only added 3 inputs since the `"SRHMEM_mct2"` region has only 3 bins. One can inject signals to as many channels as one wants, but for simplicity, we will use only one channel. Now we are ready to export this signal patch and compute the exclusion limit

```{code-cell} ipython3
pdf_wrapper = spey.get_backend("pyhf")
statistical_model = pdf_wrapper(
    analysis="simple_pyhf",
    background_only_model=interpreter.background_only_model,
    signal_patch=interpreter.make_patch(),
)
print(statistical_model.exclusion_confidence_level())
```

```python
[0.9856303068018685]
```
