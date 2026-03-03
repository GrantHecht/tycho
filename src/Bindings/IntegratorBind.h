#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// TychoBind<Integrator<DODE>> specialization and IntegratorBuildConstructors free function.
// Replaces the out-of-class Integrator<DODE>::Build() and BuildConstructors() definitions
// that were previously included from Integrator.h.

#include "DenseFunctionBaseBind.h"

namespace Tycho {

namespace Bind {

template <class DODE, class PyDODE> void IntegratorBuildConstructors(PyDODE &obj) {
    obj.def("integrator", [](const DODE &od, double ds) { return Integrator<DODE>(od, ds); });

    obj.def("integrator", [](const DODE &od, std::string meth, double ds) {
        return Integrator<DODE>(od, meth, ds);
    });
    if constexpr (DODE::UV != 0) {
        obj.def("integrator", [](const DODE &od, std::string meth, double ds,
                                 const typename Integrator<DODE>::ControllerType &u,
                                 const typename Integrator<DODE>::ControlIndexType &v) {
            return Integrator<DODE>(od, meth, ds, u, v);
        });
        obj.def("integrator",
                [](const DODE &od, std::string meth, double ds, std::shared_ptr<LGLInterpTable> u,
                   const Eigen::VectorXi &v) { return Integrator<DODE>(od, meth, ds, u, v); });
        obj.def("integrator",
                [](const DODE &od, std::string meth, double ds, std::shared_ptr<LGLInterpTable> u) {
                    return Integrator<DODE>(od, meth, ds, u);
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

} // namespace Bind

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

        obj.def(nb::init<const DODE &, std::string, double>());
        obj.def(nb::init<const DODE &, double>());

        if constexpr (DODE::UV != 0) {
            obj.def(nb::init<const DODE &, std::string, double,
                             const typename Integrator<DODE>::ControllerType &,
                             const typename Integrator<DODE>::ControlIndexType &>());
            obj.def(
                nb::init<const DODE &, double, const typename Integrator<DODE>::ControllerType &,
                         const typename Integrator<DODE>::ControlIndexType &>());
            obj.def(nb::init<const DODE &, double, const Eigen::VectorXd &>());

            obj.def(nb::init<const DODE &, std::string, double, std::shared_ptr<LGLInterpTable>,
                             const Eigen::VectorXi &>());
            obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>,
                             const Eigen::VectorXi &>());

            obj.def(nb::init<const DODE &, std::string, double, std::shared_ptr<LGLInterpTable>>());
            obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>>());
        }

        obj.def("integrate",
                (IntegRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate,
                nb::arg("Xt0UP"), nb::arg("tf"));

        obj.def("integrate",
                (IntegEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                     const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"));

        obj.def("integrate",
                (std::vector<IntegRet> (Integrator<DODE>::*)(const std::vector<ODEStateD> &,
                                                             const Eigen::VectorXd &) const) &
                    Integrator<DODE>::integrate,
                nb::arg("Xt0UPs"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_stm",
                (std::vector<STMRet> (Integrator<DODE>::*)(const std::vector<ODEStateD> &,
                                                           const Eigen::VectorXd &) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("Xt0UPs"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_stm2",
                (std::vector<std::tuple<ODEStateD, Eigen::MatrixXd, Eigen::MatrixXd>> (
                    Integrator<DODE>::*)(const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                                         const std::vector<ODEStateD> &) const) &
                    Integrator<DODE>::integrate_stm2,
                nb::arg("Xt0UPs"), nb::arg("tfs"), nb::arg("Lfs"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_parallel",
                (std::vector<IntegRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_parallel,
                nb::arg("Xt0UPs"), nb::arg("tfs"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_parallel",
                (std::vector<IntegEventRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_parallel,
                nb::arg("Xt0UPs"), nb::arg("tfs"), nb::arg("Events"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        ////////////////////////////////////////////////////////////////////////////

        obj.def("integrate_dense",
                (DenseRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("Xt0UP"), nb::arg("tf"));

        obj.def("integrate_dense",
                (DenseRet (Integrator<DODE>::*)(const ODEStateD &, double, int) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("n"));

        obj.def(
            "integrate_dense",
            [](const Integrator<DODE> &integ, const ODEStateD &x0, double tf, int n,
               nb::object pyfunc) {
                return integ.integrate_dense(x0, tf, n,
                                             [pyfunc](ConstEigenRef<Eigen::VectorXd> x) -> bool {
                                                 return PyObject_IsTrue(pyfunc(x).ptr()) != 0;
                                             });
            },
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("n"), nb::arg("StopFunc"));

        obj.def("integrate_dense",
                (DenseEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                     const std::vector<EventPack> &, bool) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"), nb::arg("alloutput") = false);

        obj.def("integrate_dense",
                (DenseEventRet (Integrator<DODE>::*)(const ODEStateD &, double, int,
                                                     const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("nstates"), nb::arg("Events"));

        ////////////////////////////////////////////////////////////////////////////

        obj.def("integrate_dense_parallel",
                (std::vector<DenseRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_dense_parallel,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def(
            "integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator<DODE>::*)(
                const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_dense_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

        obj.def("integrate_dense_parallel",
                (std::vector<DenseRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<int> &, int))&Integrator<DODE>::integrate_dense_parallel,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("ns"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        obj.def(
            "integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator<DODE>::*)(
                const std::vector<ODEStateD> &, const Eigen::VectorXd &, const std::vector<int> &,
                const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_dense_parallel,
            nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("ns"), nb::arg("Events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>());

        /////////////////////////////////////////////////////

        obj.def("integrate_stm",
                (STMRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("Xt0UP"), nb::arg("tf"));
        obj.def("integrate_stm",
                (STMEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                   const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"));
        obj.def("integrate_stm_parallel",
                (STMRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                              int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());
        obj.def("integrate_stm_parallel",
                (std::vector<STMRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());
        obj.def("integrate_stm_parallel",
                (std::vector<STMEventRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("Xt0UP"), nb::arg("tf"), nb::arg("Events"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>());

        /////////////////////////////////////////////////////

        Bind::DenseBaseBuild<Integrator<DODE>>(obj);

        obj.def_rw("EnableVectorization", &Integrator<DODE>::EnableVectorization);

        obj.def_rw("DefStepSize", &Integrator<DODE>::DefStepSize);
        obj.def_rw("MaxStepSize", &Integrator<DODE>::MaxStepSize);
        obj.def_rw("MinStepSize", &Integrator<DODE>::MinStepSize);
        obj.def_rw("MaxStepChange", &Integrator<DODE>::MaxStepChange);
        obj.def_rw("FastAdaptiveSTM", &Integrator<DODE>::FastAdaptiveSTM);

        obj.def_rw("StepFrac", &Integrator<DODE>::StepFrac);
        obj.def_rw("ErrPowFac", &Integrator<DODE>::ErrPowFac);

        obj.def_rw("Adaptive", &Integrator<DODE>::Adaptive);
        obj.def_rw("AbsTols", &Integrator<DODE>::AbsTols);

        obj.def("setAbsTol", &Integrator<DODE>::setAbsTol);
        obj.def("setAbsTols", &Integrator<DODE>::setAbsTols);
        obj.def("getAbsTols", &Integrator<DODE>::getAbsTols);

        obj.def("setRelTol", &Integrator<DODE>::setRelTol);
        obj.def("setRelTols", &Integrator<DODE>::setRelTols);
        obj.def("getRelTols", &Integrator<DODE>::getRelTols);

        obj.def("setStepSizes", &Integrator<DODE>::setStepSizes, nb::arg("DefStepSize"),
                nb::arg("MinStepSize"), nb::arg("MaxStepSize"));

        obj.def_rw("EventTol", &Integrator<DODE>::EventTol);
        obj.def_rw("MaxEventIters", &Integrator<DODE>::MaxEventIters);
        obj.def_rw("VectorizeBatchCalls", &Integrator<DODE>::VectorizeBatchCalls);
    }
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
