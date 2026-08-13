// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
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

#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/interior/typedefs/eigen_types.h>

// Real private headers (no detail/ counterpart)
#include "utils/eigen_stl.h"
#include "utils/fmtlib.h"

// Utils
#include <hven/detail/interior/utils/get_core_count.h>
#include "tycho/detail/utils/lambda_jump_table.h"
#include <hven/detail/interior/utils/math_functions.h>
#include <hven/detail/interior/utils/std_extensions.h>
#include <hven/detail/interior/utils/thread_pool.h>
#include <hven/detail/interior/utils/timer.h>
#include <hven/detail/interior/utils/tuple_iterator.h>
#include <hven/detail/interior/utils/type_name.h>
#include <hven/detail/interior/utils/type_storage.h>

// Python binding headers live in src/Bindings/pch_nb.h and are baked into the
// pch_bindings precompiled header used exclusively by _tychopy and tycho_extensions.

#endif // PCH_H
