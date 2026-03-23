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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#ifndef PCH_H
#define PCH_H

#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include "TypeDefs/EigenTypes.h"
#include "Utils/EigenSTL.h"
#include "Utils/GetCoreCount.h"
#include "Utils/LambdaJumpTable.h"
#include "Utils/MathFunctions.h"
#include "Utils/STDExtensions.h"
#include "Utils/ThreadPool.h"
#include "Utils/Timer.h"
#include "Utils/TupleIterator.h"
#include "Utils/TypeName.h"
#include "Utils/TypeStorage.h"
#include "Utils/fmtlib.h"

// Python binding headers live in src/Bindings/pch_nb.h and are baked into the
// pch_bindings precompiled header used exclusively by _tychopy and tycho_extensions.

#endif // PCH_H
