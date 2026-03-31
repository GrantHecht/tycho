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

// Out-of-class definitions of InterpTable binding functions.
// Included from src/Bindings/Tycho_VectorFunctions.cpp after all InterpTable
// headers have been included so all types are complete.

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

// ── InterpTable1D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable1DBuild(nb::module_ &m) {

    using MatType = InterpTable1D::MatType;
    auto obj = nb::class_<InterpTable1D>(m, "InterpTable1D");

    obj.def(nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &, int, std::string>(),
            nb::arg("ts"), nb::arg("Vs"), nb::arg("axis") = 0,
            nb::arg("kind") = std::string("cubic"));

    obj.def(nb::init<const Eigen::VectorXd &, const MatType &, int, std::string>(), nb::arg("ts"),
            nb::arg("Vs"), nb::arg("axis") = 0, nb::arg("kind") = std::string("cubic"));

    obj.def(nb::init<const std::vector<Eigen::VectorXd> &, int, std::string>(), nb::arg("Vts"),
            nb::arg("tvar") = -1, nb::arg("kind") = std::string("cubic"));

    obj.def("interp", nb::overload_cast<double>(&InterpTable1D::interp, nb::const_));
    obj.def("interp",
            nb::overload_cast<const Eigen::VectorXd &>(&InterpTable1D::interp, nb::const_));

    obj.def("__call__", nb::overload_cast<double>(&InterpTable1D::interp, nb::const_),
            nb::is_operator());
    obj.def("__call__",
            nb::overload_cast<const Eigen::VectorXd &>(&InterpTable1D::interp, nb::const_),
            nb::is_operator());

    obj.def("__call__", [](std::shared_ptr<InterpTable1D> &self, const GenericFunction<-1, 1> &t) {
        nb::object pyfun;
        if (self->vlen_ == 1) {
            auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
            pyfun = nb::cast(f);
        } else {
            auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
            pyfun = nb::cast(f);
        }
        return pyfun;
    });

    obj.def("__call__", [](std::shared_ptr<InterpTable1D> &self, const Segment<-1, 1, -1> &t) {
        nb::object pyfun;

        if (self->vlen_ == 1) {
            auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
            pyfun = nb::cast(f);
        } else {
            auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
            pyfun = nb::cast(f);
        }
        return pyfun;
    });

    obj.def("interp_deriv1", &InterpTable1D::interp_deriv1);
    obj.def("interp_deriv2", &InterpTable1D::interp_deriv2);

    obj.def_rw("warn_out_of_bounds", &InterpTable1D::warn_out_of_bounds_);
    obj.def_rw("throw_out_of_bounds", &InterpTable1D::throw_out_of_bounds_);

    obj.def("sf", [](std::shared_ptr<InterpTable1D> &self) {
        if (self->vlen_ != 1) {
            throw std::invalid_argument(
                "InterpTable1D storing Vector data cannot be converted to Scalar Function.");
        }
        return GenericFunction<-1, 1>(InterpFunction1D<1>(self));
    });
    obj.def("vf", [](std::shared_ptr<InterpTable1D> &self) {
        return GenericFunction<-1, -1>(InterpFunction1D<-1>(self));
    });
}

// ── InterpTable2D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable2DBuild(nb::module_ &m) {
    using MatType = InterpTable2D::MatType;
    auto obj = nb::class_<InterpTable2D>(m, "InterpTable2D");

    obj.def(nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &,
                     const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &, std::string>(),
            nb::arg("xs"), nb::arg("ys"), nb::arg("Z"), nb::arg("kind") = std::string("cubic"));

    obj.def("interp", nb::overload_cast<double, double>(&InterpTable2D::interp, nb::const_));
    obj.def("interp", nb::overload_cast<const MatType &, const MatType &>(&InterpTable2D::interp,
                                                                          nb::const_));

    obj.def_rw("warn_out_of_bounds", &InterpTable2D::warn_out_of_bounds_);
    obj.def_rw("throw_out_of_bounds", &InterpTable2D::throw_out_of_bounds_);

    obj.def("interp_deriv1", &InterpTable2D::interp_deriv1);
    obj.def("interp_deriv2", &InterpTable2D::interp_deriv2);

    obj.def("find_elem", &InterpTable2D::find_elem);

    obj.def("__call__", nb::overload_cast<double, double>(&InterpTable2D::interp, nb::const_),
            nb::is_operator());
    obj.def("__call__",
            nb::overload_cast<const MatType &, const MatType &>(&InterpTable2D::interp, nb::const_),
            nb::is_operator());

    obj.def("__call__", [](std::shared_ptr<InterpTable2D> &self, const GenericFunction<-1, 1> &x,
                           const GenericFunction<-1, 1> &y) {
        return GenericFunction<-1, 1>(InterpFunction2D(self).eval(stack(x, y)));
    });

    obj.def("__call__", [](std::shared_ptr<InterpTable2D> &self, const Segment<-1, 1, -1> &x,
                           const Segment<-1, 1, -1> &y) {
        return GenericFunction<-1, 1>(InterpFunction2D(self).eval(stack(x, y)));
    });

    obj.def("__call__", [](std::shared_ptr<InterpTable2D> &self, const Segment<-1, 2, -1> &xy) {
        return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
    });

    obj.def("__call__",
            [](std::shared_ptr<InterpTable2D> &self, const GenericFunction<-1, -1> &xy) {
                return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
            });

    obj.def("sf", [](std::shared_ptr<InterpTable2D> &self) {
        return GenericFunction<-1, 1>(InterpFunction2D(self));
    });
    obj.def("vf", [](std::shared_ptr<InterpTable2D> &self) {
        return GenericFunction<-1, -1>(InterpFunction2D(self));
    });
}

// ── InterpTable3D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable3DBuild(nb::module_ &m) {

    auto obj = nb::class_<InterpTable3D>(m, "InterpTable3D");

    obj.def(nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &,
                     const Eigen::Tensor<double, 3> &, std::string, bool>(),
            nb::arg("xs"), nb::arg("ys"), nb::arg("zs"), nb::arg("fs"),
            nb::arg("kind") = std::string("cubic"), nb::arg("cache") = false);

    obj.def("interp",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp, nb::const_));
    obj.def("interp_deriv1",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp_deriv1, nb::const_));
    obj.def("interp_deriv2",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp_deriv2, nb::const_));

    obj.def_rw("warn_out_of_bounds", &InterpTable3D::warn_out_of_bounds_);
    obj.def_rw("throw_out_of_bounds", &InterpTable3D::throw_out_of_bounds_);

    obj.def("__call__",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp, nb::const_),
            nb::is_operator());

    obj.def("__call__", [](std::shared_ptr<InterpTable3D> &self, const GenericFunction<-1, 1> &x,
                           const GenericFunction<-1, 1> &y, const GenericFunction<-1, 1> &z) {
        return GenericFunction<-1, 1>(InterpFunction3D(self).eval(stack(x, y, z)));
    });

    obj.def("__call__", [](std::shared_ptr<InterpTable3D> &self, const Segment<-1, 1, -1> &x,
                           const Segment<-1, 1, -1> &y, const Segment<-1, 1, -1> &z) {
        return GenericFunction<-1, 1>(InterpFunction3D(self).eval(stack(x, y, z)));
    });

    obj.def("__call__", [](std::shared_ptr<InterpTable3D> &self, const Segment<-1, 3, -1> &xyz) {
        return GenericFunction<-1, 1>(InterpFunction3D(self).eval(xyz));
    });

    obj.def("__call__",
            [](std::shared_ptr<InterpTable3D> &self, const GenericFunction<-1, -1> &xyz) {
                return GenericFunction<-1, 1>(InterpFunction3D(self).eval(xyz));
            });

    obj.def("sf", [](std::shared_ptr<InterpTable3D> &self) {
        return GenericFunction<-1, 1>(InterpFunction3D(self));
    });
    obj.def("vf", [](std::shared_ptr<InterpTable3D> &self) {
        return GenericFunction<-1, -1>(InterpFunction3D(self));
    });
}

// ── InterpTable4D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable4DBuild(nb::module_ &m) {

    auto obj = nb::class_<InterpTable4D>(m, "InterpTable4D");

    obj.def(
        nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &,
                 const Eigen::VectorXd &, const Eigen::Tensor<double, 4> &, std::string, bool>(),
        nb::arg("xs"), nb::arg("ys"), nb::arg("zs"), nb::arg("ws"), nb::arg("fs"),
        nb::arg("kind") = std::string("cubic"), nb::arg("cache") = false);

    obj.def("interp",
            nb::overload_cast<double, double, double, double>(&InterpTable4D::interp, nb::const_));
    obj.def("interp_deriv1", nb::overload_cast<double, double, double, double>(
                                 &InterpTable4D::interp_deriv1, nb::const_));
    obj.def("interp_deriv2", nb::overload_cast<double, double, double, double>(
                                 &InterpTable4D::interp_deriv2, nb::const_));

    obj.def_rw("warn_out_of_bounds", &InterpTable4D::warn_out_of_bounds_);
    obj.def_rw("throw_out_of_bounds", &InterpTable4D::throw_out_of_bounds_);

    obj.def("__call__",
            nb::overload_cast<double, double, double, double>(&InterpTable4D::interp, nb::const_),
            nb::is_operator());

    obj.def("__call__", [](std::shared_ptr<InterpTable4D> &self, const GenericFunction<-1, 1> &x,
                           const GenericFunction<-1, 1> &y, const GenericFunction<-1, 1> &z,
                           const GenericFunction<-1, 1> &w) {
        return GenericFunction<-1, 1>(InterpFunction4D(self).eval(stack(x, y, z, w)));
    });

    obj.def("__call__",
            [](std::shared_ptr<InterpTable4D> &self, const Segment<-1, 1, -1> &x,
               const Segment<-1, 1, -1> &y, const Segment<-1, 1, -1> &z, const Segment<-1, 1, -1> &w

            ) { return GenericFunction<-1, 1>(InterpFunction4D(self).eval(stack(x, y, z, w))); });

    obj.def("__call__", [](std::shared_ptr<InterpTable4D> &self, const Segment<-1, -1, -1> &xyzw) {
        return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
    });

    obj.def("__call__",
            [](std::shared_ptr<InterpTable4D> &self, const GenericFunction<-1, -1> &xyzw) {
                return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
            });

    obj.def("sf", [](std::shared_ptr<InterpTable4D> &self) {
        return GenericFunction<-1, 1>(InterpFunction4D(self));
    });
    obj.def("vf", [](std::shared_ptr<InterpTable4D> &self) {
        return GenericFunction<-1, -1>(InterpFunction4D(self));
    });
}

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
