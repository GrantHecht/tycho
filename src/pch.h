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
#include "Utils/TypeErasure.h"
#include "Utils/TypeName.h"
#include "Utils/fmtlib.h"

////////////////////////////////////
#ifdef TYCHO_PYTHON_BINDINGS
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace nb = nanobind;

// TypeCasters.h must be included BEFORE the STL casters (stl/vector.h,
// stl/variant.h, …) because it adds explicit type_caster<> specializations
// for Eigen::VectorXd, Eigen::VectorXi, and variants that contain them.
// Explicit specialisations must be visible before the STL casters implicitly
// instantiate the corresponding partial Eigen specialisations; otherwise the
// program is ill-formed (C++14 [temp.expl.spec]/6).
#include "Bindings/TypeCasters.h"

#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#endif // TYCHO_PYTHON_BINDINGS

/////////////////////////////////

#endif // PCH_H
