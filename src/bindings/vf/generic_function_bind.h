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

// Free function templates in namespace tycho::bind for GenericFunction,
// GenericComparative, and GenericConditional binding methods.
// Included from tycho_vector_functions.h under TYCHO_PYTHON_BINDINGS,
// after all three type definitions are complete.

#include "dense_function_base_bind.h"

namespace tycho {

using namespace tycho::vf;

namespace bind {

// ── GenericBuild
// ──────────────────────────────────────────────────────────────────────────────────
template <class Derived, class PYClass> void GenericBuild(PYClass &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<Derived::ORC == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    obj.def(nb::init<const Derived &>());
    if constexpr (Derived::ORC == -1 && Derived::IRC == -1) {
        obj.def("__init__", [](Derived *self, const GenericFunction<Derived::IRC, 1> &src) {
            new (self) Derived(Derived::template PyCopy<GenericFunction<Derived::IRC, 1>>(src));
        });
    }

    bind::DenseBaseBuild<Derived>(obj);
}

// ── MinMaxBuild
// ───────────────────────────────────────────────────────────────────────────────────
template <class PYCLASS> void MinMaxBuild(PYCLASS &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using GenComp = GenericComparative<-1>;

    // =========================================================================
    // ONE ARG BINDINGS

    // =========================================================================
    // TWO ARG BINDINGS
    // 1D Output Maximization Bindings
    obj.def("max", [](const GenS &f1, const GenS &f2) {
        return GenS(ComparativeFunction<GenS, GenS>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](double v1, const GenS &f2) {
        Vector1<double> v;
        v[0] = v1;
        Constant<-1, 1> f1(f2.IRows(), v);
        return GenS(ComparativeFunction<Constant<-1, 1>, GenS>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](const GenS &f1, double v2) {
        Vector1<double> v;
        v[0] = v2;
        Constant<-1, 1> f2(f1.IRows(), v);
        return GenS(ComparativeFunction<GenS, Constant<-1, 1>>(ComparativeFlags::MaxFlag, f1, f2));
    });

    // ND Output Maximization Bindings
    obj.def("max", [](const Gen &f1, const Gen &f2) {
        return Gen(ComparativeFunction<Gen, Gen>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](Eigen::VectorXd v1, const Gen &f2) {
        Constant<-1, -1> f1(f2.IRows(), v1);
        return Gen(ComparativeFunction<Constant<-1, -1>, Gen>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](const Gen &f1, Eigen::VectorXd v2) {
        Constant<-1, -1> f2(f1.IRows(), v2);
        return Gen(ComparativeFunction<Gen, Constant<-1, -1>>(ComparativeFlags::MaxFlag, f1, f2));
    });

    // 1D Output Minimization Bindings
    obj.def("min", [](const GenS &f1, const GenS &f2) {
        return GenS(ComparativeFunction<GenS, GenS>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](double v1, const GenS &f2) {
        Vector1<double> v;
        v[0] = v1;
        Constant<-1, 1> f1(f2.IRows(), v);
        return GenS(ComparativeFunction<Constant<-1, 1>, GenS>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](const GenS &f1, double v2) {
        Vector1<double> v;
        v[0] = v2;
        Constant<-1, 1> f2(f1.IRows(), v);
        return GenS(ComparativeFunction<GenS, Constant<-1, 1>>(ComparativeFlags::MinFlag, f1, f2));
    });

    // ND Output Maximization Bindings
    obj.def("min", [](const Gen &f1, const Gen &f2) {
        return Gen(ComparativeFunction<Gen, Gen>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](Eigen::VectorXd v1, const Gen &f2) {
        Constant<-1, -1> f1(f2.IRows(), v1);
        return Gen(ComparativeFunction<Constant<-1, -1>, Gen>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](const Gen &f1, Eigen::VectorXd v2) {
        Constant<-1, -1> f2(f1.IRows(), v2);
        return Gen(ComparativeFunction<Gen, Constant<-1, -1>>(ComparativeFlags::MinFlag, f1, f2));
    });

    // =========================================================================
    // THREE ARG BINDINGS
    // TODO: 3-argument min-max bindings
}

// ── ComparativeBuild
// ──────────────────────────────────────────────────────────────────────────────
inline void ComparativeBuild(nb::module_ &m) {
    using GenComp = GenericComparative<-1>;

    auto obj = nb::class_<GenComp>(m, "Comparative");

    obj.def("compute",
            [](const GenComp &a, ConstEigenRef<Eigen::VectorXd> x) { return a.compute(x); });

    bind::MinMaxBuild(obj);
}

// ── IfElseBuild
// ───────────────────────────────────────────────────────────────────────────────────
template <class PYCLASS> void IfElseBuild(PYCLASS &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using ELEM = Segment<-1, 1, -1>;
    using GenCon = GenericConditional<-1>;

    obj.def("ifelse", [](const GenCon &test, const GenS &tf, const GenS &ff) {
        return GenS(IfElseFunction{test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon &test, double tfv, const GenS &ff) {
        Vector1<double> v;
        v[0] = tfv;
        Constant<-1, 1> tf(test.IRows(), v);
        return GenS(IfElseFunction{test, tf, ff});
    });
    obj.def("ifelse", [](const GenCon &test, const GenS &tf, double ffv) {
        Vector1<double> v;
        v[0] = ffv;
        Constant<-1, 1> ff(test.IRows(), v);
        return GenS(IfElseFunction{test, tf, ff});
    });
    obj.def("ifelse", [](const GenCon &test, double tfv, double ffv) {
        Vector1<double> v1;
        v1[0] = tfv;
        Constant<-1, 1> tf(test.IRows(), v1);
        Vector1<double> v2;
        v2[0] = ffv;
        Constant<-1, 1> ff(test.IRows(), v2);
        return GenS(IfElseFunction{test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon &test, const Gen &tf, const Gen &ff) {
        return Gen(IfElseFunction{test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon &test, Eigen::VectorXd v, const Gen &ff) {
        Constant<-1, -1> tf(test.IRows(), v);
        return Gen(IfElseFunction{test, tf, ff});
    });
    obj.def("ifelse", [](const GenCon &test, const Gen &tf, Eigen::VectorXd v) {
        Constant<-1, -1> ff(test.IRows(), v);
        return Gen(IfElseFunction{test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon &test, Eigen::VectorXd v1, Eigen::VectorXd v2) {
        Constant<-1, -1> tf(test.IRows(), v1);
        Constant<-1, -1> ff(test.IRows(), v2);
        return Gen(IfElseFunction{test, tf, ff});
    });
}

// ── ConditionalBuild
// ──────────────────────────────────────────────────────────────────────────────
inline void ConditionalBuild(nb::module_ &m) {
    using GenCon = GenericConditional<-1>;

    auto obj = nb::class_<GenCon>(m, "Conditional");

    obj.def("compute",
            [](const GenCon &a, ConstEigenRef<Eigen::VectorXd> x) { return a.compute(x); });

    obj.def(
        "__and__",
        [](const GenCon &a, const GenCon &b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ANDFlag, b));
        },
        nb::is_operator());

    obj.def(
        "__or__",
        [](const GenCon &a, const GenCon &b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ORFlag, b));
        },
        nb::is_operator());

    bind::IfElseBuild(obj);
}

} // namespace bind
} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
