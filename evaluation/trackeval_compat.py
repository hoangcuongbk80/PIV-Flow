"""Load official TrackEval with a scoped legacy NumPy-name compatibility view."""
import sys
import numpy as np


class _NumpyView:
    def __getattr__(self, name):
        aliases = {"int": int, "float": float, "bool": bool, "complex": complex}
        return aliases[name] if name in aliases else getattr(np, name)


def load_trackeval():
    import trackeval
    # Some official releases still use np.int/np.float removed by NumPy 1.24.
    # Replace only their module bindings, never the global NumPy module.
    view = _NumpyView()
    for name, module in list(sys.modules.items()):
        if name.startswith("trackeval.") and getattr(module, "np", None) is np:
            module.np = view
    return trackeval
