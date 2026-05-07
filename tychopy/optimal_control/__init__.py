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
from _tychopy.optimal_control import *

###############################################################################
## Exposing Compiled Module Elements to improve autocomplete

ControlModes = _tychopy.optimal_control.ControlModes
FiniteDiffTable = _tychopy.optimal_control.FiniteDiffTable
IntegralModes = _tychopy.optimal_control.IntegralModes
InterpFunction = _tychopy.optimal_control.InterpFunction
TranscriptionModes = _tychopy.optimal_control.TranscriptionModes

InterpFunction_1 = _tychopy.optimal_control.InterpFunction_1
InterpFunction_3 = _tychopy.optimal_control.InterpFunction_3
InterpFunction_6 = _tychopy.optimal_control.InterpFunction_6
LGLInterpTable = _tychopy.optimal_control.LGLInterpTable

ODEArguments = _tychopy.optimal_control.ODEArguments
OptimalControlProblem = _tychopy.optimal_control.OptimalControlProblem
PhaseInterface = _tychopy.optimal_control.PhaseInterface
PhaseRegionFlags = _tychopy.optimal_control.PhaseRegionFlags
IVPAlg = _tychopy.optimal_control.IVPAlg
StateConstraint = _tychopy.optimal_control.StateConstraint
StateObjective = _tychopy.optimal_control.StateObjective

ode_6 = _tychopy.optimal_control.ode_6
ode_6_3 = _tychopy.optimal_control.ode_6_3
ode_7_3 = _tychopy.optimal_control.ode_7_3
ode_x = _tychopy.optimal_control.ode_x
ode_x_u = _tychopy.optimal_control.ode_x_u
ode_x_u_p = _tychopy.optimal_control.ode_x_u_p

###############################################################################
## Expose Pure Python Extensions to OptimalControl

from .ode_base_class import ODEBase

##############################################################################
if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.optimal_control)
    for m in mlist:
        print(m[0], "= _tychopy.optimal_control." + str(m[0]))
