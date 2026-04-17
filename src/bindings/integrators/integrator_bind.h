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

// TychoBind<Integrator<DODE>> specialization and IntegratorBuildConstructors free function.
// Replaces the out-of-class Integrator<DODE>::Build() and BuildConstructors() definitions
// that were previously included from Integrator.h.

#include "dense_function_base_bind.h"

namespace tycho {

using namespace tycho::vf;
using namespace tycho::integrators;

namespace bind {

inline IVPAlg parse_ivp_alg(const std::string &str) {
    if (str == "DOPRI54" || str == "DP54")
        return IVPAlg::DOPRI54;
    if (str == "DOPRI87" || str == "DP87")
        return IVPAlg::DOPRI87;
    if (str == "Tsit5")
        return IVPAlg::Tsit5;
    throw std::invalid_argument(fmt::format(
        "Unknown IVP algorithm: '{}'; accepted values: DOPRI54, DP54, DOPRI87, DP87, Tsit5", str));
}

template <class DODE, class PyDODE> void IntegratorBuildConstructors(PyDODE &obj) {
    obj.def("integrator", [](const DODE &od, double ds) { return Integrator<DODE>(od, ds); });

    // IVPAlg enum overload
    obj.def("integrator",
            [](const DODE &od, IVPAlg meth, double ds) { return Integrator<DODE>(od, meth, ds); });
    // String overload — converts to IVPAlg at the binding layer
    obj.def("integrator", [](const DODE &od, std::string meth, double ds) {
        return Integrator<DODE>(od, parse_ivp_alg(meth), ds);
    });
    if constexpr (DODE::UV != 0) {
        obj.def("integrator", [](const DODE &od, IVPAlg meth, double ds,
                                 const typename Integrator<DODE>::ControllerType &u,
                                 const typename Integrator<DODE>::ControlIndexType &v) {
            return Integrator<DODE>(od, meth, ds, u, v);
        });
        obj.def("integrator", [](const DODE &od, std::string meth, double ds,
                                 const typename Integrator<DODE>::ControllerType &u,
                                 const typename Integrator<DODE>::ControlIndexType &v) {
            return Integrator<DODE>(od, parse_ivp_alg(meth), ds, u, v);
        });
        obj.def("integrator",
                [](const DODE &od, IVPAlg meth, double ds, std::shared_ptr<LGLInterpTable> u,
                   const Eigen::VectorXi &v) { return Integrator<DODE>(od, meth, ds, u, v); });
        obj.def("integrator", [](const DODE &od, std::string meth, double ds,
                                 std::shared_ptr<LGLInterpTable> u, const Eigen::VectorXi &v) {
            return Integrator<DODE>(od, parse_ivp_alg(meth), ds, u, v);
        });
        obj.def("integrator",
                [](const DODE &od, IVPAlg meth, double ds, std::shared_ptr<LGLInterpTable> u) {
                    return Integrator<DODE>(od, meth, ds, u);
                });
        obj.def("integrator",
                [](const DODE &od, std::string meth, double ds, std::shared_ptr<LGLInterpTable> u) {
                    return Integrator<DODE>(od, parse_ivp_alg(meth), ds, u);
                });
        obj.def("integrator",
                [](const DODE &od, double ds, const typename Integrator<DODE>::ControllerType &u,
                   const typename Integrator<DODE>::ControlIndexType &v) {
                    return Integrator<DODE>(od, ds, u, v);
                });
        obj.def("integrator", [](const DODE &od, double ds, const Eigen::VectorXd &v) {
            return Integrator<DODE>(od, ds, v);
        });
        obj.def("integrator",
                [](const DODE &od, double ds, std::shared_ptr<LGLInterpTable> u,
                   const Eigen::VectorXi &v) { return Integrator<DODE>(od, ds, u, v); });
        obj.def("integrator", [](const DODE &od, double ds, std::shared_ptr<LGLInterpTable> u) {
            return Integrator<DODE>(od, ds, u);
        });
    }
}

} // namespace bind

template <class DODE> struct TychoBind<Integrator<DODE>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Integrator<DODE>>(m, name);

        using ODEStateD = typename DODE::template Input<double>;
        using IntegRet = typename Integrator<DODE>::IntegRet;
        using IntegEventRet = typename Integrator<DODE>::IntegEventRet;
        using DenseRet = typename Integrator<DODE>::DenseRet;
        using DenseEventRet = typename Integrator<DODE>::DenseEventRet;
        using STMRet = typename Integrator<DODE>::STMRet;
        using STMEventRet = typename Integrator<DODE>::STMEventRet;
        using EventPack = typename Integrator<DODE>::EventPack;

        obj.def(nb::init<const DODE &, IVPAlg, double>());
        obj.def(nb::init<const DODE &, double>());

        if constexpr (DODE::UV != 0) {
            obj.def(nb::init<const DODE &, IVPAlg, double,
                             const typename Integrator<DODE>::ControllerType &,
                             const typename Integrator<DODE>::ControlIndexType &>());
            obj.def(
                nb::init<const DODE &, double, const typename Integrator<DODE>::ControllerType &,
                         const typename Integrator<DODE>::ControlIndexType &>());
            obj.def(nb::init<const DODE &, double, const Eigen::VectorXd &>());

            obj.def(nb::init<const DODE &, IVPAlg, double, std::shared_ptr<LGLInterpTable>,
                             const Eigen::VectorXi &>());
            obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>,
                             const Eigen::VectorXi &>());

            obj.def(nb::init<const DODE &, IVPAlg, double, std::shared_ptr<LGLInterpTable>>());
            obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>>());
        }

        obj.def("integrate",
                (IntegRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate,
                nb::arg("xt0_up"), nb::arg("tf"));

        obj.def("integrate",
                (IntegEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                     const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"));

        obj.def("integrate",
                (std::vector<IntegRet> (Integrator<DODE>::*)(const std::vector<ODEStateD> &,
                                                             const Eigen::VectorXd &) const) &
                    Integrator<DODE>::integrate,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_stm",
                (std::vector<STMRet> (Integrator<DODE>::*)(const std::vector<ODEStateD> &,
                                                           const Eigen::VectorXd &) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_stm2",
                (std::vector<std::tuple<ODEStateD, Eigen::MatrixXd, Eigen::MatrixXd>> (
                    Integrator<DODE>::*)(const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                                         const std::vector<ODEStateD> &) const) &
                    Integrator<DODE>::integrate_stm2,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("lfs"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_parallel",
                (std::vector<IntegRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_parallel",
                (std::vector<IntegEventRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("events"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        ////////////////////////////////////////////////////////////////////////////

        obj.def("integrate_dense",
                (DenseRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"));

        obj.def("integrate_dense",
                (DenseRet (Integrator<DODE>::*)(const ODEStateD &, double, int) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("n"));

        obj.def(
            "integrate_dense",
            [](const Integrator<DODE> &integ, const ODEStateD &x0, double tf, int n,
               nb::object pyfunc) {
                return integ.integrate_dense(x0, tf, n,
                                             [pyfunc](ConstEigenRef<Eigen::VectorXd> x) -> bool {
                                                 return PyObject_IsTrue(pyfunc(x).ptr()) != 0;
                                             });
            },
            nb::arg("xt0_up"), nb::arg("tf"), nb::arg("n"), nb::arg("stop_func"));

        obj.def("integrate_dense",
                (DenseEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                     const std::vector<EventPack> &, bool) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"), nb::arg("alloutput") = false);

        obj.def("integrate_dense",
                (DenseEventRet (Integrator<DODE>::*)(const ODEStateD &, double, int,
                                                     const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("nstates"), nb::arg("events"));

        ////////////////////////////////////////////////////////////////////////////

        obj.def("integrate_dense_parallel",
                (std::vector<DenseRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_dense_parallel,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def(
            "integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator<DODE>::*)(
                const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_dense_parallel,
            nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_dense_parallel",
                (std::vector<DenseRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<int> &, int))&Integrator<DODE>::integrate_dense_parallel,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("ns"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def(
            "integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator<DODE>::*)(
                const std::vector<ODEStateD> &, const Eigen::VectorXd &, const std::vector<int> &,
                const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_dense_parallel,
            nb::arg("xt0_up"), nb::arg("tf"), nb::arg("ns"), nb::arg("events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

        /////////////////////////////////////////////////////

        obj.def("integrate_stm",
                (STMRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("xt0_up"), nb::arg("tf"));
        obj.def("integrate_stm",
                (STMEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                   const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"));
        obj.def("integrate_stm_parallel",
                (STMRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                              int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());
        obj.def("integrate_stm_parallel",
                (std::vector<STMRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());
        obj.def("integrate_stm_parallel",
                (std::vector<STMEventRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        /////////////////////////////////////////////////////

        bind::DenseBaseBuild<Integrator<DODE>>(obj);

        obj.def_rw("enable_vectorization", &Integrator<DODE>::enable_vectorization_);

        obj.def_rw("def_step_size", &Integrator<DODE>::def_step_size_);
        obj.def_rw("max_step_size", &Integrator<DODE>::max_step_size_);
        obj.def_rw("min_step_size", &Integrator<DODE>::min_step_size_);
        obj.def_rw("max_step_change", &Integrator<DODE>::max_step_change_);
        obj.def_rw("fast_adaptive_stm", &Integrator<DODE>::fast_adaptive_stm_);

        obj.def_rw("step_frac", &Integrator<DODE>::step_frac_);
        obj.def_rw("err_pow_fac", &Integrator<DODE>::err_pow_fac_);

        obj.def_rw("adaptive", &Integrator<DODE>::adaptive_);
        obj.def_rw("abs_tols", &Integrator<DODE>::abs_tols_);

        obj.def("set_abs_tol", &Integrator<DODE>::set_abs_tol);
        obj.def("set_abs_tols", &Integrator<DODE>::set_abs_tols);
        obj.def("get_abs_tols", &Integrator<DODE>::get_abs_tols);

        obj.def("set_rel_tol", &Integrator<DODE>::set_rel_tol);
        obj.def("set_rel_tols", &Integrator<DODE>::set_rel_tols);
        obj.def("get_rel_tols", &Integrator<DODE>::get_rel_tols);

        obj.def("set_step_sizes", &Integrator<DODE>::set_step_sizes, nb::arg("def_step_size"),
                nb::arg("min_step_size"), nb::arg("max_step_size"));

        obj.def_rw("event_tol", &Integrator<DODE>::event_tol_);
        obj.def_rw("max_event_iters", &Integrator<DODE>::max_event_iters_);
        obj.def_rw("vectorize_batch_calls", &Integrator<DODE>::vectorize_batch_calls_);
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
