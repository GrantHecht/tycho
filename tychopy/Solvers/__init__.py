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
from _tychopy.Solvers import *

AlgorithmModes = _tychopy.Solvers.AlgorithmModes
BarrierModes = _tychopy.Solvers.BarrierModes
ConvergenceFlags = _tychopy.Solvers.ConvergenceFlags
Jet = _tychopy.Solvers.Jet
JetJobModes = _tychopy.Solvers.JetJobModes
LineSearchModes = _tychopy.Solvers.LineSearchModes
OptimizationProblem = _tychopy.Solvers.OptimizationProblem
OptimizationProblemBase = _tychopy.Solvers.OptimizationProblemBase
PDStepStrategies = _tychopy.Solvers.PDStepStrategies
PSIOPT = _tychopy.Solvers.PSIOPT
QPOrderingModes = _tychopy.Solvers.QPOrderingModes
QPPivotModes = _tychopy.Solvers.QPPivotModes

if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.Solvers)
    for m in mlist:
        print(m[0], "= _tychopy.Solvers." + str(m[0]))
