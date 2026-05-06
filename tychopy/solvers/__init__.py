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
from _tychopy.solvers import *

AlgorithmModes = _tychopy.solvers.AlgorithmModes
BarrierModes = _tychopy.solvers.BarrierModes
ConvergenceFlags = _tychopy.solvers.ConvergenceFlags
Jet = _tychopy.solvers.Jet
JetJobModes = _tychopy.solvers.JetJobModes
LineSearchModes = _tychopy.solvers.LineSearchModes
OptimizationProblem = _tychopy.solvers.OptimizationProblem
OptimizationProblemBase = _tychopy.solvers.OptimizationProblemBase
PDStepStrategies = _tychopy.solvers.PDStepStrategies
PSIOPT = _tychopy.solvers.PSIOPT
QPOrderingModes = _tychopy.solvers.QPOrderingModes
QPPivotModes = _tychopy.solvers.QPPivotModes

if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.solvers)
    for m in mlist:
        print(m[0], "= _tychopy.solvers." + str(m[0]))
