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

// TychoBind<T> specializations for CommonFunctions types.
// Also provides bind::SegBuild<Derived, PyClass> free function.
// Included from CommonFunctions/CommonFunctions.h under TYCHO_PYTHON_BINDINGS.

#include "dense_function_base_bind.h"

namespace tycho {

using namespace tycho::vf;

// ── bind::SegBuild
// ────────────────────────────────────────────────────────────────────────────────
namespace bind {

template <class Derived, class PyClass> void SegBuild(PyClass &obj) {
    bind::DoubleMathBuild<Derived>(obj);
    bind::UnaryMathBuild<Derived>(obj);
    bind::BinaryMathBuild<Derived>(obj);
    bind::BinaryOperatorsBuild<Derived>(obj);
    bind::FunctionIndexingBuild<Derived>(obj);
    bind::ConditionalOperatorsBuild<Derived>(obj);

    obj.def("tolist", [](const Derived &func) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (int i = 0; i < func.ORows(); i++) {
            elems.push_back(func.coeff(i));
        }
        return elems;
    });

    obj.def("tolist", [](const Derived &func, std::vector<int> coeffs) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (const auto &coeff : coeffs) {
            elems.push_back(func.coeff(coeff));
        }
        return elems;
    });

    obj.def("tolist", [](const Derived &func, std::vector<std::tuple<int, int>> seglist) {
        using ELEM = Segment<-1, 1, -1>;
        using SEG2 = Segment<-1, 2, -1>;
        using SEG3 = Segment<-1, 3, -1>;
        using SEG = Segment<-1, -1, -1>;

        std::vector<nb::object> segs;
        for (const auto &seg : seglist) {

            int start = std::get<0>(seg);
            int size = std::get<1>(seg);
            nb::object pyfun;
            if (size == 1) {
                auto f = func.coeff(start);
                pyfun = nb::cast(f);
            } else if (size == 2) {
                auto f = func.template segment<2>(start);
                pyfun = nb::cast(f);
            } else if (size == 3) {
                auto f = func.template segment<3>(start);
                pyfun = nb::cast(f);
            } else {
                auto f = func.segment(start, size);
                pyfun = nb::cast(f);
            }

            segs.push_back(pyfun);
        }
        return segs;
    });
}

} // namespace bind

// ── Constant
// ──────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR> struct TychoBind<Constant<IR, OR>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Constant<IR, OR>>(m, name);
        obj.def(nb::init<int, Eigen::VectorXd>());
        bind::DenseBaseBuild<Constant<IR, OR>>(obj);
    }
};

// ── FunctionDotProduct_Impl
// ───────────────────────────────────────────────────────────────────────
template <class Derived, class Func1, class Func2>
struct TychoBind<FunctionDotProduct_Impl<Derived, Func1, Func2>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<Func1, Func2>());
        bind::DenseBaseBuild<Derived>(obj);
    }
};

// ── NestedFunction
// ────────────────────────────────────────────────────────────────────────────────
template <class OuterFunc, class InnerFunc> struct TychoBind<NestedFunction<OuterFunc, InnerFunc>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<NestedFunction<OuterFunc, InnerFunc>>(m, name);
        obj.def(nb::init<>());
        obj.def(nb::init<OuterFunc, InnerFunc>());
        bind::DenseBaseBuild<NestedFunction<OuterFunc, InnerFunc>>(obj);
    }
};

// ── NormalizedPower_Impl
// ──────────────────────────────────────────────────────────────────────────
template <class Derived, int IR, int PW> struct TychoBind<NormalizedPower_Impl<Derived, IR, PW>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<int>());
        if constexpr (IR > 0) {
            obj.def(nb::init<>());
        }
        bind::DenseBaseBuild<Derived>(obj);
    }
};

// ── IntegralNorm_Impl
// ─────────────────────────────────────────────────────────────────────────────
template <class Derived, int USZ, int Power>
struct TychoBind<IntegralNorm_Impl<Derived, USZ, Power>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<int>());
        if constexpr (USZ > 0) {
            obj.def(nb::init<>());
        }
        bind::DenseBaseBuild<Derived>(obj);
    }
};

// ── ParsedInput
// ───────────────────────────────────────────────────────────────────────────────────
template <class Func, int IRC, int ORC> struct TychoBind<ParsedInput<Func, IRC, ORC>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<ParsedInput<Func, IRC, ORC>>(m, name);
        obj.def(nb::init<Func, const Eigen::VectorXi &, int>());
        bind::DenseBaseBuild<ParsedInput<Func, IRC, ORC>>(obj);
    }
};

// ── Arguments
// ─────────────────────────────────────────────────────────────────────────────────────
template <int IR_OR> struct TychoBind<Arguments<IR_OR>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Arguments<IR_OR>>(m, name);
        obj.def(nb::init<int>());
        obj.def("constant", [](const Arguments<IR_OR> &a, Eigen::VectorXd v) {
            return GenericFunction<-1, -1>(Constant<-1, -1>(a.IRows(), v));
        });
        obj.def("constant", [](const Arguments<IR_OR> &a, double v) {
            Eigen::Matrix<double, 1, 1> vv;
            vv[0] = v;
            return GenericFunction<-1, 1>(Constant<-1, 1>(a.IRows(), vv));
        });
        bind::DenseBaseBuild<Arguments<IR_OR>>(obj);
        bind::SegBuild<Arguments<IR_OR>>(obj);
    }
};

// ── Segment
// ───────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR, int ST> struct TychoBind<Segment<IR, OR, ST>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Segment<IR, OR, ST>>(m, name);
        obj.def(nb::init<int, int, int>());
        bind::DenseBaseBuild<Segment<IR, OR, ST>>(obj);
        bind::SegBuild<Segment<IR, OR, ST>>(obj);
    }
};

// ── TwoFunctionSum_Impl
// ───────────────────────────────────────────────────────────────────────────
template <class Derived, class Func1, class Func2, bool DoDifference>
struct TychoBind<TwoFunctionSum_Impl<Derived, Func1, Func2, DoDifference>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<Func1, Func2>());
        bind::DenseBaseBuild<Derived>(obj);
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
