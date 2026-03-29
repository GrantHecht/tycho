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

#ifdef TYCHO_PYTHON_BINDINGS

#include "dense_function_base_bind.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho {

using namespace tycho::vf;

template <int IRR, int ORR>
struct PyVectorFunction
    : VectorFunction<PyVectorFunction<IRR, ORR>, IRR, ORR, DenseDerivativeMode::FDiffFwd,
                     DenseDerivativeMode::FDiffFwd> {
    using Base = VectorFunction<PyVectorFunction<IRR, ORR>, IRR, ORR, DenseDerivativeMode::FDiffFwd,
                                DenseDerivativeMode::FDiffFwd>;

    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    template <class Scalar> using Input = typename Base::template Input<Scalar>;

    bool threadSafe = false;
    nb::object pyfun;
    nb::tuple pyargs;

    PyVectorFunction(int irr, int orr, nb::object f, Input<double> js, Input<double> hs,
                     nb::tuple p) {
        this->set_io_rows(irr, orr);
        this->pyfun = f;
        this->pyargs = p;

        if (irr != js.size() || irr != hs.size()) {
            throw std::invalid_argument("Incorrectly sized FD Step Sizes");
        }

        this->set_jac_fd_steps(js);
        this->set_hess_fd_steps(hs);
    }

    PyVectorFunction(int irr, int orr, nb::object f, double js, double hs, nb::tuple p) {
        this->set_io_rows(irr, orr);
        this->pyfun = f;
        this->pyargs = p;
        this->set_jac_fd_steps(js);
        this->set_hess_fd_steps(hs);
    }

    PyVectorFunction(int irr, nb::object f, Input<double> js, Input<double> hs, nb::tuple p)
        : PyVectorFunction(irr, ORR, f, js, hs, p) {}
    PyVectorFunction(int irr, nb::object f, double js, double hs, nb::tuple p)
        : PyVectorFunction(irr, ORR, f, js, hs, p) {}

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        nb::gil_scoped_acquire acquire; // safe no-op if GIL already held
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        auto result = pyfun(Input<double>(x), *pyargs);
        fx = nb::cast<Output<double>>(result);
    }

    bool thread_safe() const { return threadSafe; }
};

template <int IRR, int ORR>
struct NumbaVectorFunction
    : VectorFunction<NumbaVectorFunction<IRR, ORR>, IRR, ORR, DenseDerivativeMode::FDiffFwd,
                     DenseDerivativeMode::FDiffFwd> {
    using Base = VectorFunction<NumbaVectorFunction<IRR, ORR>, IRR, ORR,
                                DenseDerivativeMode::FDiffFwd, DenseDerivativeMode::FDiffFwd>;

    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    template <class Scalar> using Input = typename Base::template Input<Scalar>;

    using FType = long long unsigned int;
    typedef void (*FPtr)(double *, double *, int, int);

    bool threadSafe = false;
    bool dojac = false;
    FPtr fun;

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        Input<double> xt = x;
        Output<double> fxt;
        fxt.resize(this->output_rows()); // allocate before passing pointer to Numba

        this->fun(xt.data(), fxt.data(), this->input_rows(), this->output_rows());
        fx = fxt;
    }
    NumbaVectorFunction(int irr, int orr, const FType &f, double js, double hs) {
        this->set_io_rows(irr, orr);
        this->fun = (FPtr)f;
        this->set_jac_fd_steps(js);
        this->set_hess_fd_steps(hs);
    }
    NumbaVectorFunction(int irr, int orr, const FType &f)
        : NumbaVectorFunction(irr, orr, f, 1.0e-6, 1.0e-6) {}
    NumbaVectorFunction(const FType &f) : NumbaVectorFunction(IRR, ORR, f) {}
    NumbaVectorFunction(const FType &f, double js, double hs)
        : NumbaVectorFunction(IRR, ORR, f, js, hs) {}
    NumbaVectorFunction(int irr, const FType &f, double js, double hs)
        : NumbaVectorFunction(irr, ORR, f, js, hs) {}
    NumbaVectorFunction(int irr, const FType &f) : NumbaVectorFunction(irr, ORR, f) {}

    bool thread_safe() const { return threadSafe; }
};

// ── TychoBind<PyVectorFunction>
// ───────────────────────────────────────────────────────────────────
template <int IRR, int ORR> struct TychoBind<PyVectorFunction<IRR, ORR>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<PyVectorFunction<IRR, ORR>>(m, name);

        if constexpr (ORR != 1) {
            obj.def(nb::init<int, int, nb::object, double, double, nb::tuple>(), nb::arg("IRows"),
                    nb::arg("ORows"), nb::arg("Func"), nb::arg("Jstepsize") = 1.0e-6,
                    nb::arg("Hstepsize") = 1.0e-4, nb::arg("args") = nb::tuple());
        } else {
            obj.def(nb::init<int, nb::object, double, double, nb::tuple>(), nb::arg("IRows"),
                    nb::arg("Func"), nb::arg("Jstepsize") = 1.0e-6, nb::arg("Hstepsize") = 1.0e-4,
                    nb::arg("args") = nb::tuple());
        }

        bind::DenseBaseBuild<PyVectorFunction<IRR, ORR>>(obj);
        obj.def_prop_rw(
            "thread_safe", [](const PyVectorFunction<IRR, ORR> &self) { return self.threadSafe; },
            [](PyVectorFunction<IRR, ORR> &, bool val) {
                if (val)
                    throw std::invalid_argument(
                        "PyVectorFunction always acquires the GIL and cannot execute in "
                        "parallel. Setting thread_safe=True provides no benefit. Use "
                        "NumbaVectorFunction for parallel-safe functions.");
            });
    }
};

// ── TychoBind<NumbaVectorFunction> ───────────────────────────────────────────────────────────────
template <int IRR, int ORR> struct TychoBind<NumbaVectorFunction<IRR, ORR>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<NumbaVectorFunction<IRR, ORR>>(m, name);
        obj.def(nb::init<int, int, const typename NumbaVectorFunction<IRR, ORR>::FType &, double,
                         double>());
        obj.def(nb::init<int, int, const typename NumbaVectorFunction<IRR, ORR>::FType &>());

        if constexpr (ORR == 1) {
            obj.def(nb::init<int, const typename NumbaVectorFunction<IRR, ORR>::FType &, double,
                             double>());
            obj.def(nb::init<int, const typename NumbaVectorFunction<IRR, ORR>::FType &>());
        }

        if constexpr (IRR > 0 && ORR > 0) {
            obj.def(
                nb::init<const typename NumbaVectorFunction<IRR, ORR>::FType &, double, double>());
            obj.def(nb::init<const typename NumbaVectorFunction<IRR, ORR>::FType &>());
        }
        obj.def_rw("thread_safe", &NumbaVectorFunction<IRR, ORR>::threadSafe);
        bind::DenseBaseBuild<NumbaVectorFunction<IRR, ORR>>(obj);
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
