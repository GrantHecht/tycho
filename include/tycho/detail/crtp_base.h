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
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include <functional>
#include <string>
#include <type_traits>

#include "tycho/detail/type_name.h"

namespace Tycho {

/*!
 * @brief Curiosly Recurring Template Pattern base class.
 *
 * @tparam Derived
 */
template <class Derived> struct CRTPBase {
    /*!
     * @brief Returns a reference to the instantiation of the derived class
     *
     * @return Derived&
     */
    Derived &derived() { return static_cast<Derived &>(*this); }

    /*!
     * @brief Returns a const reference to the instantiation of the derived class
     *
     * @return const Derived&
     */
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

    /*!
     * @brief Returns the demangled type name of the Derived class
     *
     * @return std::string
     */
    std::string name() const { return type_name<Derived>(); }

    template <class T> T cast() const { return T(this->derived()); }

    template <class T> T copy() const { return T(this->derived()); }

    template <class T> void deep_copy_into(T &obj) const { obj = T(this->derived()); }

    template <class T> T &find() { return this->derived(); }

    template <class T> void copy_if_same(const T &obj) {
        if constexpr (std::is_same<T, Derived>::value) {
            this->derived() = obj;
        }
    }
    template <class T> void forward_if_same(std::function<void(T &)> fun) {
        if constexpr (std::is_same<T, Derived>::value) {
            fun(this->derived());
        }
    }
};

template <class Derived, template <typename> class Mixin> struct CRTPMixin {
    Derived &derived() { return static_cast<Derived &>(*this); }
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

  private:
    CRTPMixin() {}
    friend Mixin<Derived>;
};

} // namespace Tycho
