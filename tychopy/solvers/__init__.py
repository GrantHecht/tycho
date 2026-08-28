# =============================================================================
# Originally from ASSET (AlabamaASRL/asset_asrl)
# Copyright 2020-present The University of Alabama-Astrodynamics and Space
#   Research Lab. Licensed under the Apache License, Version 2.0
# License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# Original Developer: James B. Pezent
#
# Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
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
InteriorPointSolver = _tychopy.solvers.InteriorPointSolver
Jet = _tychopy.solvers.Jet
LineSearchModes = _tychopy.solvers.LineSearchModes
OptimizationProblem = _tychopy.solvers.OptimizationProblem
OptimizationProblemBase = _tychopy.solvers.OptimizationProblemBase
PDStepStrategies = _tychopy.solvers.PDStepStrategies
QPOrderingModes = _tychopy.solvers.QPOrderingModes
QPPivotModes = _tychopy.solvers.QPPivotModes

# The solve() surface: Mode, the warm-start currency and its result types,
# and the engine handle classes.
Mode = _tychopy.solvers.Mode
DeclarationKey = _tychopy.solvers.DeclarationKey
WarmExtension = _tychopy.solvers.WarmExtension
WarmStartData = _tychopy.solvers.WarmStartData
StageResult = _tychopy.solvers.StageResult
PhaseResult = _tychopy.solvers.PhaseResult
SolveResult = _tychopy.solvers.SolveResult
SqpSolver = _tychopy.solvers.SqpSolver
IpoptSolver = _tychopy.solvers.IpoptSolver
StartLevel = _tychopy.solvers.StartLevel
QpMode = _tychopy.solvers.QpMode
SsnSigmaRule = _tychopy.solvers.SsnSigmaRule
SsnHintRule = _tychopy.solvers.SsnHintRule
SsnInfeasibilityRule = _tychopy.solvers.SsnInfeasibilityRule

# Short aliases.
IPM = InteriorPointSolver
SQP = SqpSolver

if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.solvers)
    for m in mlist:
        print(m[0], "= _tychopy.solvers." + str(m[0]))
