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
from _tychopy.VectorFunctions import *

Arguments = _tychopy.VectorFunctions.Arguments
ColMatrix = _tychopy.VectorFunctions.ColMatrix
Comparative = _tychopy.VectorFunctions.Comparative
Conditional = _tychopy.VectorFunctions.Conditional
ConstantScalar = _tychopy.VectorFunctions.ConstantScalar
ConstantVector = _tychopy.VectorFunctions.ConstantVector
Element = _tychopy.VectorFunctions.Element
PyScalarFunction = _tychopy.VectorFunctions.PyScalarFunction
PyVectorFunction = _tychopy.VectorFunctions.PyVectorFunction
RowMatrix = _tychopy.VectorFunctions.RowMatrix
ScalarFunction = _tychopy.VectorFunctions.ScalarFunction
Segment = _tychopy.VectorFunctions.Segment
Segment2 = _tychopy.VectorFunctions.Segment2
Segment3 = _tychopy.VectorFunctions.Segment3
Stack = _tychopy.VectorFunctions.stack
StackScalar = _tychopy.VectorFunctions.stack_scalar
Sum = _tychopy.VectorFunctions.sum
SumElems = _tychopy.VectorFunctions.sum_elems
SumScalar = _tychopy.VectorFunctions.sum_scalar
VectorFunction = _tychopy.VectorFunctions.VectorFunction

arccos = _tychopy.VectorFunctions.arccos
arcsin = _tychopy.VectorFunctions.arcsin
arctan = _tychopy.VectorFunctions.arctan
arctan2 = _tychopy.VectorFunctions.arctan2
cos = _tychopy.VectorFunctions.cos
cosh = _tychopy.VectorFunctions.cosh
_cross_cpp = _tychopy.VectorFunctions.cross


def cross(a, b):
    if isinstance(a, (list, tuple)):
        a = np.asarray(a, dtype=np.float64)
    return _cross_cpp(a, b)


cwise_product = _tychopy.VectorFunctions.cwise_product
cwise_quotient = _tychopy.VectorFunctions.cwise_quotient
dot = _tychopy.VectorFunctions.dot
doublecross = _tychopy.VectorFunctions.doublecross
exp = _tychopy.VectorFunctions.exp
ifelse = _tychopy.VectorFunctions.ifelse
inverse_norm = _tychopy.VectorFunctions.inverse_norm
log = _tychopy.VectorFunctions.log
matmul = _tychopy.VectorFunctions.matmul
norm = _tychopy.VectorFunctions.norm
normalize = _tychopy.VectorFunctions.normalize
pow = _tychopy.VectorFunctions.pow
quat_product = _tychopy.VectorFunctions.quat_product
sin = _tychopy.VectorFunctions.sin
sinh = _tychopy.VectorFunctions.sinh
sqrt = _tychopy.VectorFunctions.sqrt
squared = _tychopy.VectorFunctions.squared
stack = _tychopy.VectorFunctions.stack
stack_scalar = _tychopy.VectorFunctions.stack_scalar
sum = _tychopy.VectorFunctions.sum
tan = _tychopy.VectorFunctions.tan
tanh = _tychopy.VectorFunctions.tanh


from .Extensions.DerivChecker import FDDerivChecker

if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.VectorFunctions)
    for m in mlist:
        print(m[0], "= _tychopy.VectorFunctions." + str(m[0]))
