---
myst:
  html_meta:
    "property=og:title": "Utility belt for spey-pyhf plug-in"
    "property=og:description": "Various helper functions to be used with spey-pyhf plug-in"
    "property=og:image": "https://spey.readthedocs.io/en/main/_static/spey-logo.png"
---

# Utility belt for spey-pyhf plug-in

`spey-pyhf` plug-in includes a set of helper functions to manipulate the input data in a spey-ready fashion.

```{code-block} python
:caption: Lets import necessary libraries first
import spey
from spey_pyhf.helper_functions import WorkspaceInterpreter
import json
```

We will use an example background-only JSON file which is downloaded from HEPData, but you can use any background-only JSON file.

```{code-block} python
:caption: Load the background file
with open("/PATH/TO/BkgOnly.json", "r") as f:
    bkg_only = json.load(f)
```

{py:obj}`spey_pyhf.helper_functions.WorkspaceInterpreter` scans the background-only dictionary and extracts relevant information such as channel names:

```{code-block} python
print(list(interpreter.channels))
```

```python
['WREM_cuts', 'STCREM_cuts', 'TRHMEM_cuts', 'TRMMEM_cuts', 'TRLMEM_cuts', 'SRHMEM_mct2', 'SRMMEM_mct2', 'SRLMEM_mct2']
```

and information about bin sizes in each channel

```python
print(interpreter.bin_map['SRHMEM_mct2'])
```

```python
3
```

we can inject signal to any channel we like

```{code-block} python
interpreter.inject_signal('SRHMEM_mct2', [5.0, 12.0, 4.0])
```

Notice that I only added 3 inputs since the `"SRHMEM_mct2"` region has only 3 bins. One can inject signals to as many channels as one wants, but for simplicity, we will use only one channel. Now we are ready to export this signal patch and compute the exclusion limit

```{code-block} python
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
