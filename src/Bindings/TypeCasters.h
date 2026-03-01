#pragma once

#include "InterfaceTypes.h"
#include "pch.h"

// ============================================================================
// pybind11 type casters for Tycho interface types.
//
// Include this header in any binding .cpp that registers functions whose
// signatures contain these types, so that pybind11 can automatically convert
// Python objects to/from the corresponding C++ variants.
// ============================================================================

namespace pybind11 {
namespace detail {

// ---------------------------------------------------------------------------
// ScaleType  =  std::variant<double, Eigen::VectorXd, std::string>
//
// Python  None   → std::string("none")   (disables auto-scaling)
// Python  str    → std::string
// Python  float  → double
// Python  array  → Eigen::VectorXd
// ---------------------------------------------------------------------------
template <> struct type_caster<Tycho::ScaleType> {
    PYBIND11_TYPE_CASTER(Tycho::ScaleType, _("Union[float, numpy.ndarray, str, None]"));

    bool load(handle src, bool convert) {
        if (src.is_none()) {
            value = std::string("none");
            return true;
        }
        if (PyUnicode_Check(src.ptr())) {
            value = src.cast<std::string>();
            return true;
        }
        if (PyFloat_Check(src.ptr()) || PyLong_Check(src.ptr())) {
            value = src.cast<double>();
            return true;
        }
        try {
            value = src.cast<Eigen::VectorXd>();
            return true;
        } catch (...) {
        }
        return false;
    }

    static handle cast(Tycho::ScaleType src, return_value_policy policy, handle parent) {
        return std::visit(
            [&](auto &&v) -> handle {
                using T = std::decay_t<decltype(v)>;
                return make_caster<T>::cast(v, policy, parent);
            },
            src);
    }
};

} // namespace detail
} // namespace pybind11
