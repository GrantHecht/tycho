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
#   - Module renamed: asset_asrl (pybind11) -> _tycho (nanobind)
#   - Imports updated accordingly
# =============================================================================

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
