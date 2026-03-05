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

#include "CommonFunctions/ExpressionFwdDeclarations.h"
#include "pch.h"

#ifdef TYCHO_PYTHON_BINDINGS

namespace Tycho {

/*
 * Converts list of python objects into a vector of dynamically sized GenericFunctions.
 * Can accept any of the fundamental types exposed to python as well as Python and Numpy
 * vectors and scalars.
 */
std::vector<GenericFunction<-1, -1>> ParsePythonArgs(nb::args x, int irows = 0);

/*
 * Converts list of python objects into a vector of dynamically sized scalar GenericFunctions.
 * Can accept any of the fundamental scalar types exposed to python as well as Python and Numpy
 * scalars.
 */
std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(nb::args x, int irows = 0);

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
