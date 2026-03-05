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
from _tycho.OptimalControl import *

###############################################################################
## Exposing Compiled Module Elements to improve autocomplete

ControlModes = _tycho.OptimalControl.ControlModes
FiniteDiffTable = _tycho.OptimalControl.FiniteDiffTable
IntegralModes = _tycho.OptimalControl.IntegralModes
InterpFunction = _tycho.OptimalControl.InterpFunction
TranscriptionModes = _tycho.OptimalControl.TranscriptionModes

InterpFunction_1 = _tycho.OptimalControl.InterpFunction_1
InterpFunction_3 = _tycho.OptimalControl.InterpFunction_3
InterpFunction_6 = _tycho.OptimalControl.InterpFunction_6
LGLInterpTable = _tycho.OptimalControl.LGLInterpTable

LinkConstraint = _tycho.OptimalControl.LinkConstraint
LinkFlags = _tycho.OptimalControl.LinkFlags
LinkObjective = _tycho.OptimalControl.LinkObjective

ODEArguments = _tycho.OptimalControl.ODEArguments
OptimalControlProblem = _tycho.OptimalControl.OptimalControlProblem
PhaseInterface = _tycho.OptimalControl.PhaseInterface
PhaseRegionFlags = _tycho.OptimalControl.PhaseRegionFlags
RKOptions = _tycho.OptimalControl.RKOptions
StateConstraint = _tycho.OptimalControl.StateConstraint
StateObjective = _tycho.OptimalControl.StateObjective

ode_2_1 = _tycho.OptimalControl.ode_2_1
ode_6 = _tycho.OptimalControl.ode_6
ode_6_3 = _tycho.OptimalControl.ode_6_3
ode_7_3 = _tycho.OptimalControl.ode_7_3
ode_x = _tycho.OptimalControl.ode_x
ode_x_u = _tycho.OptimalControl.ode_x_u
ode_x_u_p = _tycho.OptimalControl.ode_x_u_p

###############################################################################
## Expose Pure Python Extensions to OptimalControl

from .ODEBaseClass import ODEBase

##############################################################################
if __name__ == "__main__":
    mlist = inspect.getmembers(_tycho.OptimalControl)
    for m in mlist:
        print(m[0], "= _tycho.OptimalControl." + str(m[0]))
