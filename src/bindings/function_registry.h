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
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#pragma once

#include "pch.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"

// Forward-declare sub-namespaces so that the using-directives below compile
// even before the module aggregate headers are included (e.g. in the bindings PCH).
namespace tycho::vf {}
namespace tycho::oc {}
namespace tycho::solvers {}
namespace tycho::astro {}
namespace tycho::integrators {}
namespace tycho::utils {}

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::integrators;
using namespace tycho::utils;

// Primary template — undefined; specializations in *_bind.h files
template <class T> struct TychoBind;

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
                      "vector_functions",
                      "SubModule Containing Vector and Scalar Function Types and Functions")),
          ocmod(
              m.def_submodule("optimal_control",
                              "SubModule Containing Optimal Control ODEs, Phases, and Utilities")),
          solmod(m.def_submodule("solvers", "SubModule Containing PSIOPT,NLP, and Solver Flags")),
          extmod(m.def_submodule("extensions", "User Defined Extensions")),
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
        requires requires(nb::module_ &m) { TychoBind<Derived>::build(m); }
    void build_register(nb::module_ &m) {
        TychoBind<Derived>::build(m);
        RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template <class Derived>
        requires requires(nb::module_ &m, const char *name) { TychoBind<Derived>::build(m, name); }
    void build_register(const char *name) {
        TychoBind<Derived>::build(this->mod, name);
        RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template <class Derived>
        requires requires(nb::module_ &m, const char *name) { TychoBind<Derived>::build(m, name); }
    void build_register(nb::module_ &m, const char *name) {
        TychoBind<Derived>::build(m, name);
        RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
};

#endif // TYCHO_PYTHON_BINDINGS

} // namespace tycho
