// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Namespace: Tycho
// =============================================================================

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Binding-layer aggregate for VectorFunctions.
// Included in the pch_bindings PCH after tycho_vector_functions.h so that
// GenericComparative, GenericConditional, and GenericFunction are fully
// defined before these headers instantiate nb::class_<T>.

#include "VectorFunctions/CommonFunctionsBind.h"
#include "VectorFunctions/DenseFunctionBaseBind.h"
#include "VectorFunctions/GenericFunctionBind.h"
#include "VectorFunctions/PythonFunctions.h"
#include "VectorFunctions/python_arg_parsing.h"

#endif // TYCHO_PYTHON_BINDINGS
