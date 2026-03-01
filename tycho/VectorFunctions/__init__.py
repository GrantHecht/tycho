import inspect

import numpy as np

import _tycho as _tycho
from _tycho.VectorFunctions import *

Arguments = _tycho.VectorFunctions.Arguments
ColMatrix = _tycho.VectorFunctions.ColMatrix
Comparative = _tycho.VectorFunctions.Comparative
Conditional = _tycho.VectorFunctions.Conditional
ConstantScalar = _tycho.VectorFunctions.ConstantScalar
ConstantVector = _tycho.VectorFunctions.ConstantVector
Element = _tycho.VectorFunctions.Element
PyScalarFunction = _tycho.VectorFunctions.PyScalarFunction
PyVectorFunction = _tycho.VectorFunctions.PyVectorFunction
RowMatrix = _tycho.VectorFunctions.RowMatrix
ScalarFunction = _tycho.VectorFunctions.ScalarFunction
Segment = _tycho.VectorFunctions.Segment
Segment2 = _tycho.VectorFunctions.Segment2
Segment3 = _tycho.VectorFunctions.Segment3
Stack = _tycho.VectorFunctions.Stack
StackScalar = _tycho.VectorFunctions.StackScalar
Sum = _tycho.VectorFunctions.Sum
SumElems = _tycho.VectorFunctions.SumElems
SumScalar = _tycho.VectorFunctions.SumScalar
VectorFunction = _tycho.VectorFunctions.VectorFunction

arccos = _tycho.VectorFunctions.arccos
arcsin = _tycho.VectorFunctions.arcsin
arctan = _tycho.VectorFunctions.arctan
arctan2 = _tycho.VectorFunctions.arctan2
cos = _tycho.VectorFunctions.cos
cosh = _tycho.VectorFunctions.cosh
_cross_cpp = _tycho.VectorFunctions.cross


def cross(a, b):
    if isinstance(a, (list, tuple)):
        a = np.asarray(a, dtype=np.float64)
    return _cross_cpp(a, b)
cwiseProduct = _tycho.VectorFunctions.cwiseProduct
cwiseQuotient = _tycho.VectorFunctions.cwiseQuotient
dot = _tycho.VectorFunctions.dot
doublecross = _tycho.VectorFunctions.doublecross
exp = _tycho.VectorFunctions.exp
ifelse = _tycho.VectorFunctions.ifelse
inverse_norm = _tycho.VectorFunctions.inverse_norm
log = _tycho.VectorFunctions.log
matmul = _tycho.VectorFunctions.matmul
norm = _tycho.VectorFunctions.norm
normalize = _tycho.VectorFunctions.normalize
pow = _tycho.VectorFunctions.pow
quatProduct = _tycho.VectorFunctions.quatProduct
sin = _tycho.VectorFunctions.sin
sinh = _tycho.VectorFunctions.sinh
sqrt = _tycho.VectorFunctions.sqrt
squared = _tycho.VectorFunctions.squared
stack = _tycho.VectorFunctions.stack
stack_scalar = _tycho.VectorFunctions.stack_scalar
sum = _tycho.VectorFunctions.sum
tan = _tycho.VectorFunctions.tan
tanh = _tycho.VectorFunctions.tanh


from .Extensions.DerivChecker import FDDerivChecker

if __name__ == "__main__":
    mlist = inspect.getmembers(_tycho.VectorFunctions)
    for m in mlist:
        print(m[0], "= _tycho.VectorFunctions." + str(m[0]))
