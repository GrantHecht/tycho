# =============================================================================
# Originally from ASSET (AlabamaASRL/asset_asrl)
# Copyright 2020-present The University of Alabama-Astrodynamics and Space
#   Research Lab. Licensed under the Apache License, Version 2.0
# License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# Original Developer: James B. Pezent
#
# Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
#   Apache 2.0 — see LICENSE.txt):
#   - Package renamed: asset_asrl -> tycho
#   - Module renamed: asset_asrl (pybind11) -> _tychopy (nanobind)
#   - Imports updated accordingly
# =============================================================================

import inspect

import _tychopy as _tychopy
import numpy as np
from _tychopy.astro import *


def _vec6_wrap(fn):
    """Wrap a C++ function whose first arg is Vector6<double> to also accept lists/tuples."""

    def wrapper(arr_or_func, *args):
        # `input_rows` is present on VectorFunction *and* arg-like VF types
        # (Segment / Element / Arguments) which deliberately omit `eval` /
        # `__call__` in the bindings. nanobind handles implicit conversion to
        # the VectorFunction overload at the C++ boundary.
        if hasattr(arr_or_func, "input_rows"):
            return fn(arr_or_func, *args)
        return fn(np.asarray(arr_or_func, dtype=np.float64), *args)

    wrapper.__name__ = fn.__name__
    return wrapper


cartesian_to_classic = _vec6_wrap(_tychopy.astro.cartesian_to_classic)
cartesian_to_classic_true = _vec6_wrap(_tychopy.astro.cartesian_to_classic_true)
cartesian_to_modified = _vec6_wrap(_tychopy.astro.cartesian_to_modified)
classic_to_cartesian = _vec6_wrap(_tychopy.astro.classic_to_cartesian)
classic_to_modified = _vec6_wrap(_tychopy.astro.classic_to_modified)
modified_to_cartesian = _vec6_wrap(_tychopy.astro.modified_to_cartesian)
modified_to_classic = _vec6_wrap(_tychopy.astro.modified_to_classic)
propagate_cartesian = _vec6_wrap(_tychopy.astro.propagate_cartesian)
propagate_classic = _vec6_wrap(_tychopy.astro.propagate_classic)
propagate_modified = _vec6_wrap(_tychopy.astro.propagate_modified)

lambert_izzo = _tychopy.astro.lambert_izzo
Kepler = _tychopy.astro.kepler


from .astro_frames import CR3BPFrame

if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.astro)
    for m in mlist:
        print(m[0], "= _tychopy.astro." + str(m[0]))
