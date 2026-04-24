// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt).
// =============================================================================

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// TychoBind<Integrator<DODE>> specialization + IntegratorBuildConstructors
// free function (nanobind binding for the Integrator template).

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
    if (str == "BS3")
        return IVPAlg::BS3;
    if (str == "BS5")
        return IVPAlg::BS5;
    if (str == "Vern7")
        return IVPAlg::Vern7;
    if (str == "Vern8")
        return IVPAlg::Vern8;
    if (str == "Vern9")
        return IVPAlg::Vern9;
    throw std::invalid_argument(fmt::format(
        "Unknown IVP algorithm: '{}'; accepted values: DOPRI54, DP54, DOPRI87, DP87, Tsit5, BS3, "
        "BS5, Vern7, Vern8, Vern9",
        str));
}

inline IVPController parse_ivp_controller(const std::string &str) {
    if (str == "I")
        return IVPController::I;
    if (str == "PI")
        return IVPController::PI;
    if (str == "PID")
        return IVPController::PID;
    throw std::invalid_argument(
        fmt::format("Unknown IVPController: '{}'; accepted values: I, PI, PID", str));
}

inline ErrorNormType parse_error_norm(const std::string &str) {
    if (str == "RMS")
        return ErrorNormType::RMS;
    if (str == "MAX")
        return ErrorNormType::MAX;
    throw std::invalid_argument(
        fmt::format("Unknown ErrorNormType: '{}'; accepted values: RMS, MAX", str));
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

        obj.def("integrate_stm2", &Integrator<DODE>::integrate_stm2, nb::arg("xt0_ups"),
                nb::arg("tfs"), nb::arg("lfs"), nb::call_guard<nb::gil_scoped_release>());

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
                                                 // pyfunc(x) raises nb::python_error if the Python
                                                 // callable itself raises (nanobind converts the
                                                 // Python exception). PyObject_IsTrue accepts any
                                                 // truthy object — including numpy scalar bools
                                                 // that users commonly return (e.g. `x[1] < 0`). An
                                                 // error return (-1) from PyObject_IsTrue leaves
                                                 // the Python error indicator set; propagate as
                                                 // nb::python_error so the caller sees a clean
                                                 // exception instead of a silent truncation (the
                                                 // coerce -1→true trap).
                                                 nb::object result = pyfunc(x);
                                                 int truth = PyObject_IsTrue(result.ptr());
                                                 if (truth == -1)
                                                     throw nb::python_error();
                                                 return truth != 0;
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

        // def_step_size routes through set_initial_step_size so Python writes
        // respect the documented contract (flips HW auto-initdt off, rejects
        // non-positive values).
        obj.def_prop_rw(
            "def_step_size", [](const Integrator<DODE> &self) { return self.def_step_size_; },
            [](Integrator<DODE> &self, double h) { self.set_initial_step_size(h); });
        obj.def_prop_rw(
            "max_step_change",
            [](const Integrator<DODE> &self) { return self.get_max_step_change(); },
            [](Integrator<DODE> &self, double v) { self.set_max_step_change(v); });

        obj.def_rw("adaptive", &Integrator<DODE>::adaptive_);
        // abs_tols routes through set_abs_tols so Python writes validate size
        // and non-negativity.
        obj.def_prop_rw(
            "abs_tols", [](const Integrator<DODE> &self) { return self.get_abs_tols(); },
            [](Integrator<DODE> &self, typename Integrator<DODE>::template ODEDeriv<double> tol) {
                self.set_abs_tols(tol);
            });

        obj.def("set_abs_tol", &Integrator<DODE>::set_abs_tol);
        obj.def("set_abs_tols", &Integrator<DODE>::set_abs_tols);
        obj.def("get_abs_tols", &Integrator<DODE>::get_abs_tols);

        obj.def("set_rel_tol", &Integrator<DODE>::set_rel_tol);
        obj.def("set_rel_tols", &Integrator<DODE>::set_rel_tols);
        obj.def("get_rel_tols", &Integrator<DODE>::get_rel_tols);

        obj.def("set_initial_step_size", &Integrator<DODE>::set_initial_step_size, nb::arg("h"));
        obj.def("set_max_steps", &Integrator<DODE>::set_max_steps, nb::arg("n"));
        obj.def("get_max_steps", &Integrator<DODE>::get_max_steps);

        obj.def_prop_rw(
            "event_tol", [](const Integrator<DODE> &self) { return self.get_event_tol(); },
            [](Integrator<DODE> &self, double v) { self.set_event_tol(v); });
        obj.def_prop_rw(
            "max_event_iters",
            [](const Integrator<DODE> &self) { return self.get_max_event_iters(); },
            [](Integrator<DODE> &self, int n) { self.set_max_event_iters(n); });
        obj.def_rw("vectorize_batch_calls", &Integrator<DODE>::vectorize_batch_calls_);

        // Controller + stats + HW-initdt API
        obj.def("set_controller", &Integrator<DODE>::set_controller);
        obj.def("set_controller", [](Integrator<DODE> &self, const std::string &s) {
            self.set_controller(bind::parse_ivp_controller(s));
        });
        obj.def("get_controller", &Integrator<DODE>::get_controller);

        obj.def("set_error_norm", &Integrator<DODE>::set_error_norm);
        obj.def("set_error_norm", [](Integrator<DODE> &self, const std::string &s) {
            self.set_error_norm(bind::parse_error_norm(s));
        });
        obj.def("get_error_norm", &Integrator<DODE>::get_error_norm);

        obj.def("get_naccept", &Integrator<DODE>::get_naccept);
        obj.def("get_nreject", &Integrator<DODE>::get_nreject);
        obj.def("get_failed_event_count", &Integrator<DODE>::get_failed_event_count);

        obj.def("set_auto_initial_dt", &Integrator<DODE>::set_auto_initial_dt);
        obj.def("get_auto_initial_dt", &Integrator<DODE>::get_auto_initial_dt);
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
