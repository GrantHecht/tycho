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

#include "VectorFunctionTypeErasure/GenericFunction.h"
#include "pch.h"

namespace Tycho {

// Primary template — undefined; specializations in *Bind.h files
template <class T> struct TychoBind;

template <class T>
concept HasTychoBind = requires {
    typename TychoBind<T>::BuildTag;
};

#ifdef TYCHO_PYTHON_BINDINGS

template <class T> struct FuncPack {
    using type = T;
    std::string name;
    FuncPack(const T &t, std::string nm) : name(nm) {}
};

struct FunctionRegistry {
    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    nb::module_ &mod;
    nb::module_ vfmod;
    nb::module_ ocmod;
    nb::module_ solmod;
    nb::module_ extmod;

    nb::class_<VectorFunctionalX> vfuncx;
    nb::class_<ScalarFunctionalX> sfuncx;

    FunctionRegistry(nb::module_ &m)
        : mod(m), vfmod(m.def_submodule(
                      "VectorFunctions",
                      "SubModule Containing Vector and Scalar Function Types and Functions")),
          ocmod(
              m.def_submodule("OptimalControl",
                              "SubModule Containing Optimal Control ODEs, Phases, and Utilities")),
          solmod(m.def_submodule("Solvers", "SubModule Containing PSIOPT,NLP, and Solver Flags")),
          extmod(m.def_submodule("Extensions", "User Defined Extensions")),
          vfuncx(nb::class_<VectorFunctionalX>(this->vfmod, "VectorFunction")),
          sfuncx(nb::class_<ScalarFunctionalX>(this->vfmod, "ScalarFunction")) {}

    nb::module_ &getVectorFunctionsModule() { return this->vfmod; }
    nb::module_ &getOptimalControlModule() { return this->ocmod; }
    nb::module_ &getSolversModule() { return this->solmod; }

    template <int IR, int OR> struct RegSelector {
        template <class Derived> static void Register(FunctionRegistry *reg) {
            nb::implicitly_convertible<Derived, VectorFunctionalX>();
            reg->vfuncx.def(nb::init<Derived>());
        }
    };
    template <int IR> struct RegSelector<IR, 1> {
        template <class Derived> static void Register(FunctionRegistry *reg) {
            nb::implicitly_convertible<Derived, VectorFunctionalX>();
            reg->vfuncx.def(nb::init<Derived>());
            nb::implicitly_convertible<Derived, ScalarFunctionalX>();
            reg->sfuncx.def(nb::init<Derived>());
        }
    };

    template <class Derived> void Register() {
        RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template <class Derived>
        requires HasTychoBind<Derived>
    void Build_Register(nb::module_ &m) {
        TychoBind<Derived>::Build(m);
        RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template <class Derived>
        requires HasTychoBind<Derived>
    void Build_Register(const char *name) {
        TychoBind<Derived>::Build(this->mod, name);
        RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template <class Derived>
        requires HasTychoBind<Derived>
    void Build_Register(nb::module_ &m, const char *name) {
        TychoBind<Derived>::Build(m, name);
        RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
};

#endif // TYCHO_PYTHON_BINDINGS

} // namespace Tycho
