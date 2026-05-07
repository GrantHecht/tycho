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
from _tychopy.vector_functions import *

Arguments = _tychopy.vector_functions.Arguments
ColMatrix = _tychopy.vector_functions.ColMatrix
Comparative = _tychopy.vector_functions.Comparative
Conditional = _tychopy.vector_functions.Conditional
ConstantScalar = _tychopy.vector_functions.ConstantScalar
ConstantVector = _tychopy.vector_functions.ConstantVector
Element = _tychopy.vector_functions.Element
PyScalarFunction = _tychopy.vector_functions.PyScalarFunction
PyVectorFunction = _tychopy.vector_functions.PyVectorFunction
RowMatrix = _tychopy.vector_functions.RowMatrix
ScalarFunction = _tychopy.vector_functions.ScalarFunction
Segment = _tychopy.vector_functions.Segment
Segment2 = _tychopy.vector_functions.Segment2
Segment3 = _tychopy.vector_functions.Segment3
VectorFunction = _tychopy.vector_functions.VectorFunction

arccos = _tychopy.vector_functions.arccos
arcsin = _tychopy.vector_functions.arcsin
arctan = _tychopy.vector_functions.arctan
arctan2 = _tychopy.vector_functions.arctan2
cos = _tychopy.vector_functions.cos
cosh = _tychopy.vector_functions.cosh
_cross_cpp = _tychopy.vector_functions.cross


def cross(a, b):
    if isinstance(a, (list, tuple)):
        a = np.asarray(a, dtype=np.float64)
    return _cross_cpp(a, b)


cwise_product = _tychopy.vector_functions.cwise_product
cwise_quotient = _tychopy.vector_functions.cwise_quotient
dot = _tychopy.vector_functions.dot
doublecross = _tychopy.vector_functions.doublecross
exp = _tychopy.vector_functions.exp
ifelse = _tychopy.vector_functions.ifelse
inverse_norm = _tychopy.vector_functions.inverse_norm
log = _tychopy.vector_functions.log
matmul = _tychopy.vector_functions.matmul
norm = _tychopy.vector_functions.norm
normalize = _tychopy.vector_functions.normalize
pow = _tychopy.vector_functions.pow
quat_product = _tychopy.vector_functions.quat_product
sin = _tychopy.vector_functions.sin
sinh = _tychopy.vector_functions.sinh
sqrt = _tychopy.vector_functions.sqrt
squared = _tychopy.vector_functions.squared
stack = _tychopy.vector_functions.stack
stack_scalar = _tychopy.vector_functions.stack_scalar
sum = _tychopy.vector_functions.sum
sum_elems = _tychopy.vector_functions.sum_elems
sum_scalar = _tychopy.vector_functions.sum_scalar
tan = _tychopy.vector_functions.tan
tanh = _tychopy.vector_functions.tanh


from .Extensions.DerivChecker import FDDerivChecker

if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.vector_functions)
    for m in mlist:
        print(m[0], "= _tychopy.vector_functions." + str(m[0]))
