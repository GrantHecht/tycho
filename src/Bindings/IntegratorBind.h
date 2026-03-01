#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class definitions of Integrator binding methods.
// Included from Integrator.h under TYCHO_PYTHON_BINDINGS.

namespace Tycho {

template <class DODE>
template <class PyDODE>
void Integrator<DODE>::BuildConstructors(PyDODE &obj) {

    obj.def("integrator", [](const DODE &od, double ds) { return Integrator<DODE>(od, ds); });

    obj.def("integrator", [](const DODE &od, std::string meth, double ds) {
        return Integrator<DODE>(od, meth, ds);
    });
    if constexpr (DODE::UV != 0) {
        obj.def("integrator",
                [](const DODE &od, std::string meth, double ds, const ControllerType &u,
                   const ControlIndexType &v) { return Integrator<DODE>(od, meth, ds, u, v); });
        obj.def("integrator", [](const DODE &od, std::string meth, double ds,
                                 std::shared_ptr<LGLInterpTable> u, const Eigen::VectorXi &v) {
            return Integrator<DODE>(od, meth, ds, u, v);
        });
        obj.def("integrator", [](const DODE &od, std::string meth, double ds,
                                 std::shared_ptr<LGLInterpTable> u) {
            return Integrator<DODE>(od, meth, ds, u);
        });
        obj.def("integrator",
                [](const DODE &od, double ds, const ControllerType &u,
                   const ControlIndexType &v) { return Integrator<DODE>(od, ds, u, v); });
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

template <class DODE>
void Integrator<DODE>::Build(nb::module_ &m, const char *name) {
    using Base = typename Integrator<DODE>::Base;
    auto obj = nb::class_<Integrator>(m, name);

    obj.def(nb::init<const DODE &, std::string, double>());
    obj.def(nb::init<const DODE &, double>());

    if constexpr (DODE::UV != 0) {
        obj.def(nb::init<const DODE &, std::string, double, const ControllerType &,
                         const ControlIndexType &>());
        obj.def(
            nb::init<const DODE &, double, const ControllerType &, const ControlIndexType &>());
        obj.def(nb::init<const DODE &, double, const Eigen::VectorXd &>());

        obj.def(nb::init<const DODE &, std::string, double, std::shared_ptr<LGLInterpTable>,
                         const Eigen::VectorXi &>());
        obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>,
                         const Eigen::VectorXi &>());

        obj.def(nb::init<const DODE &, std::string, double, std::shared_ptr<LGLInterpTable>>());
        obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>>());
    }

    obj.def("integrate",
            (IntegRet (Integrator::*)(const ODEState<double> &, double) const) &
                Integrator::integrate,
            nb::arg("Xt0UP"), nb::arg("tf"));

    obj.def("integrate",
            (IntegEventRet (Integrator::*)(const ODEState<double> &, double,
                                           const std::vector<EventPack> &) const) &
                Integrator::integrate,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"));

    obj.def("integrate",
            (std::vector<IntegRet> (Integrator::*)(const std::vector<ODEState<double>> &,
                                                   const Eigen::VectorXd &) const) &
                Integrator::integrate,
            nb::arg("Xt0UPs"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>());

    obj.def("integrate_stm",
            (std::vector<STMRet> (Integrator::*)(const std::vector<ODEState<double>> &,
                                                 const Eigen::VectorXd &) const) &
                Integrator::integrate_stm,
            nb::arg("Xt0UPs"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>());

    obj.def("integrate_stm2",
            (std::vector<std::tuple<ODEState<double>, Jacobian<double>, Hessian<double>>> (
                Integrator::*)(const std::vector<ODEState<double>> &, const Eigen::VectorXd &,
                               const std::vector<ODEState<double>> &) const) &
                Integrator::integrate_stm2,
            nb::arg("Xt0UPs"), nb::arg("tfs"), nb::arg("Lfs"),
            nb::call_guard<nb::gil_scoped_release>());

    obj.def("integrate_parallel",
            (std::vector<IntegRet> (Integrator::*)(const std::vector<ODEState<double>> &,
                                                   const Eigen::VectorXd &,
                                                   int))&Integrator::integrate_parallel,
            nb::arg("Xt0UPs"), nb::arg("tfs"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

    obj.def("integrate_parallel",
            (std::vector<IntegEventRet> (Integrator::*)(
                const std::vector<ODEState<double>> &, const Eigen::VectorXd &,
                const std::vector<EventPack> &, int))&Integrator::integrate_parallel,
            nb::arg("Xt0UPs"), nb::arg("tfs"), nb::arg("Events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

    ////////////////////////////////////////////////////////////////////////////

    obj.def("integrate_dense",
            (DenseRet (Integrator::*)(const ODEState<double> &, double) const) &
                Integrator::integrate_dense,
            nb::arg("Xt0UP"), nb::arg("tf"));

    obj.def("integrate_dense",
            (DenseRet (Integrator::*)(const ODEState<double> &, double, int) const) &
                Integrator::integrate_dense,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("n"));

    obj.def(
        "integrate_dense",
        [](const Integrator &integ, const ODEState<double> &x0, double tf, int n,
           nb::object pyfunc) {
            return integ.integrate_dense(
                x0, tf, n, [pyfunc](ConstEigenRef<Eigen::VectorXd> x) -> bool {
                    return PyObject_IsTrue(pyfunc(x).ptr()) != 0;
                });
        },
        nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("n"), nb::arg("StopFunc"));

    obj.def("integrate_dense",
            (DenseEventRet (Integrator::*)(const ODEState<double> &, double,
                                           const std::vector<EventPack> &, bool) const) &
                Integrator::integrate_dense,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"), nb::arg("alloutput") = false);

    obj.def("integrate_dense",
            (DenseEventRet (Integrator::*)(const ODEState<double> &, double, int,
                                           const std::vector<EventPack> &) const) &
                Integrator::integrate_dense,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("nstates"), nb::arg("Events"));

    ////////////////////////////////////////////////////////////////////////////

    obj.def("integrate_dense_parallel",
            (std::vector<DenseRet> (Integrator::*)(const std::vector<ODEState<double>> &,
                                                   const Eigen::VectorXd &,
                                                   int))&Integrator::integrate_dense_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

    obj.def("integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator::*)(
                const std::vector<ODEState<double>> &, const Eigen::VectorXd &,
                const std::vector<EventPack> &, int))&Integrator::integrate_dense_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

    obj.def("integrate_dense_parallel",
            (std::vector<DenseRet> (Integrator::*)(
                const std::vector<ODEState<double>> &, const Eigen::VectorXd &,
                const std::vector<int> &, int))&Integrator::integrate_dense_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("ns"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

    obj.def("integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator::*)(
                const std::vector<ODEState<double>> &, const Eigen::VectorXd &,
                const std::vector<int> &, const std::vector<EventPack> &,
                int))&Integrator::integrate_dense_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("ns"), nb::arg("Events"),
            nb::arg("threads"), nb::call_guard<nb::gil_scoped_release>());

    /////////////////////////////////////////////////////

    obj.def("integrate_stm",
            (STMRet (Integrator::*)(const ODEState<double> &, double) const) &
                Integrator::integrate_stm,
            nb::arg("Xt0UP"), nb::arg("tf"));
    obj.def("integrate_stm",
            (STMEventRet (Integrator::*)(const ODEState<double> &, double,
                                         const std::vector<EventPack> &) const) &
                Integrator::integrate_stm,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"));
    obj.def("integrate_stm_parallel",
            (STMRet (Integrator::*)(const ODEState<double> &, double,
                                    int))&Integrator::integrate_stm_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("integrate_stm_parallel",
            (std::vector<STMRet> (Integrator::*)(const std::vector<ODEState<double>> &,
                                                 const Eigen::VectorXd &,
                                                 int))&Integrator::integrate_stm_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("integrate_stm_parallel",
            (std::vector<STMEventRet> (Integrator::*)(
                const std::vector<ODEState<double>> &, const Eigen::VectorXd &,
                const std::vector<EventPack> &, int))&Integrator::integrate_stm_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

    /////////////////////////////////////////////////////

    Base::DenseBaseBuild(obj);

    obj.def_rw("EnableVectorization", &Integrator::EnableVectorization);

    obj.def_rw("DefStepSize", &Integrator::DefStepSize);
    obj.def_rw("MaxStepSize", &Integrator::MaxStepSize);
    obj.def_rw("MinStepSize", &Integrator::MinStepSize);
    obj.def_rw("MaxStepChange", &Integrator::MaxStepChange);
    obj.def_rw("FastAdaptiveSTM", &Integrator::FastAdaptiveSTM);

    obj.def_rw("StepFrac", &Integrator::StepFrac);
    obj.def_rw("ErrPowFac", &Integrator::ErrPowFac);

    obj.def_rw("Adaptive", &Integrator::Adaptive);
    obj.def_rw("AbsTols", &Integrator::AbsTols);

    obj.def("setAbsTol", &Integrator::setAbsTol);
    obj.def("setAbsTols", &Integrator::setAbsTols);
    obj.def("getAbsTols", &Integrator::getAbsTols);

    obj.def("setRelTol", &Integrator::setRelTol);
    obj.def("setRelTols", &Integrator::setRelTols);
    obj.def("getRelTols", &Integrator::getRelTols);

    obj.def("setStepSizes", &Integrator::setStepSizes, nb::arg("DefStepSize"),
            nb::arg("MinStepSize"), nb::arg("MaxStepSize"));

    obj.def_rw("EventTol", &Integrator::EventTol);
    obj.def_rw("MaxEventIters", &Integrator::MaxEventIters);
    obj.def_rw("VectorizeBatchCalls", &Integrator::VectorizeBatchCalls);
}

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
