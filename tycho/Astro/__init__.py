import inspect

import numpy as np

import _tycho as _tycho
from _tycho.Astro import *


def _vec6_wrap(fn):
    """Wrap a C++ function whose first arg is Vector6<double> to also accept lists/tuples."""

    def wrapper(arr_or_func, *args):
        if hasattr(arr_or_func, "eval"):  # VectorFunction overload — pass through
            return fn(arr_or_func, *args)
        return fn(np.asarray(arr_or_func, dtype=np.float64), *args)

    wrapper.__name__ = fn.__name__
    return wrapper


cartesian_to_classic = _vec6_wrap(_tycho.Astro.cartesian_to_classic)
cartesian_to_classic_true = _vec6_wrap(_tycho.Astro.cartesian_to_classic_true)
cartesian_to_modified = _vec6_wrap(_tycho.Astro.cartesian_to_modified)
classic_to_cartesian = _vec6_wrap(_tycho.Astro.classic_to_cartesian)
classic_to_modified = _vec6_wrap(_tycho.Astro.classic_to_modified)
modified_to_cartesian = _vec6_wrap(_tycho.Astro.modified_to_cartesian)
modified_to_classic = _vec6_wrap(_tycho.Astro.modified_to_classic)
propagate_cartesian = _vec6_wrap(_tycho.Astro.propagate_cartesian)
propagate_classic = _vec6_wrap(_tycho.Astro.propagate_classic)
propagate_modified = _vec6_wrap(_tycho.Astro.propagate_modified)

lambert_izzo = _tycho.Astro.lambert_izzo
Kepler = _tycho.Astro.Kepler
ModifiedDynamics = _tycho.Astro.ModifiedDynamics


from .AstroFrames import CR3BPFrame

if __name__ == "__main__":
    mlist = inspect.getmembers(_tycho.Astro)
    for m in mlist:
        print(m[0], "= _tycho.Astro." + str(m[0]))
