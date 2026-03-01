import inspect

import _tycho as _tycho
from _tycho.Solvers import *

AlgorithmModes = _tycho.Solvers.AlgorithmModes
BarrierModes = _tycho.Solvers.BarrierModes
ConvergenceFlags = _tycho.Solvers.ConvergenceFlags
Jet = _tycho.Solvers.Jet
JetJobModes = _tycho.Solvers.JetJobModes
LineSearchModes = _tycho.Solvers.LineSearchModes
OptimizationProblem = _tycho.Solvers.OptimizationProblem
OptimizationProblemBase = _tycho.Solvers.OptimizationProblemBase
PDStepStrategies = _tycho.Solvers.PDStepStrategies
PSIOPT = _tycho.Solvers.PSIOPT
QPOrderingModes = _tycho.Solvers.QPOrderingModes
QPPivotModes = _tycho.Solvers.QPPivotModes

if __name__ == "__main__":
    mlist = inspect.getmembers(_tycho.Solvers)
    for m in mlist:
        print(m[0], "= _tycho.Solvers." + str(m[0]))
