#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace nb = nanobind;

// TypeCasters.h must be included BEFORE the STL casters (stl/vector.h,
// stl/variant.h, …) because it adds explicit type_caster<> specializations
// for Eigen::VectorXd, Eigen::VectorXi, and variants that contain them.
// Explicit specialisations must be visible before the STL casters implicitly
// instantiate the corresponding partial Eigen specialisations; otherwise the
// program is ill-formed (C++14 [temp.expl.spec]/6).
#include "TypeCasters.h"

#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

// FunctionRegistry.h: TychoBind<T> primary template + FunctionRegistry struct.
// Must come before Tycho_VectorFunctions.h in PCH so all binding TUs see the
// primary template before any specializations are instantiated.
#include "FunctionRegistry.h"

// JetBind.h: JetInvoker partial spec for nb::args must be visible in every
// binding TU that could instantiate Jet::map<T, Args1, nb::args>.
#include "JetBind.h"
