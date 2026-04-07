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
from _tychopy.OptimalControl import *

###############################################################################
## Exposing Compiled Module Elements to improve autocomplete

ControlModes = _tychopy.OptimalControl.ControlModes
FiniteDiffTable = _tychopy.OptimalControl.FiniteDiffTable
IntegralModes = _tychopy.OptimalControl.IntegralModes
InterpFunction = _tychopy.OptimalControl.InterpFunction
TranscriptionModes = _tychopy.OptimalControl.TranscriptionModes

InterpFunction_1 = _tychopy.OptimalControl.InterpFunction_1
InterpFunction_3 = _tychopy.OptimalControl.InterpFunction_3
InterpFunction_6 = _tychopy.OptimalControl.InterpFunction_6
LGLInterpTable = _tychopy.OptimalControl.LGLInterpTable

LinkConstraint = _tychopy.OptimalControl.LinkConstraint
LinkFlags = _tychopy.OptimalControl.LinkFlags
LinkObjective = _tychopy.OptimalControl.LinkObjective

ODEArguments = _tychopy.OptimalControl.ODEArguments
OptimalControlProblem = _tychopy.OptimalControl.OptimalControlProblem
PhaseInterface = _tychopy.OptimalControl.PhaseInterface
PhaseRegionFlags = _tychopy.OptimalControl.PhaseRegionFlags
RKOptions = _tychopy.OptimalControl.RKOptions
StateConstraint = _tychopy.OptimalControl.StateConstraint
StateObjective = _tychopy.OptimalControl.StateObjective

ode_6 = _tychopy.OptimalControl.ode_6
ode_6_3 = _tychopy.OptimalControl.ode_6_3
ode_7_3 = _tychopy.OptimalControl.ode_7_3
ode_x = _tychopy.OptimalControl.ode_x
ode_x_u = _tychopy.OptimalControl.ode_x_u
ode_x_u_p = _tychopy.OptimalControl.ode_x_u_p

###############################################################################
## Expose Pure Python Extensions to OptimalControl

from .ODEBaseClass import ODEBase

##############################################################################
if __name__ == "__main__":
    mlist = inspect.getmembers(_tychopy.OptimalControl)
    for m in mlist:
        print(m[0], "= _tychopy.OptimalControl." + str(m[0]))
