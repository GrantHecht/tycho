#pragma once

// Tycho — Phase/ODE transcription, collocation, mesh refinement

#include "tycho/vector_functions.h"

#include "tycho/detail/OptimalControlFlags.h"
#include "tycho/detail/ODESizes.h"
#include "tycho/detail/ODEArguments.h"
#include "tycho/detail/InterfaceTypes.h"
#include "tycho/detail/StateFunction.h"
#include "tycho/detail/LinkFunction.h"
#include "tycho/detail/ODE.h"
#include "tycho/detail/ODEPhaseBase.h"
#include "tycho/detail/ODEPhase.h"
#include "tycho/detail/OptimalControlProblem.h"
#include "tycho/detail/LGLInterpTable.h"
#include "tycho/detail/LGLInterpFunctions.h"
#include "tycho/detail/MeshIterateInfo.h"
#include "tycho/detail/var_registry.h"
#include "tycho/detail/runtime_ode.h"
#include "tycho/detail/ode_builder.h"
#include "tycho/detail/phase_wrapper.h"
