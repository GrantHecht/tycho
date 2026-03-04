#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class definitions of InterpTable binding functions.
// Included from src/Bindings/Tycho_VectorFunctions.cpp after all InterpTable
// headers have been included so all types are complete.

namespace Tycho {

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
        if (self->vlen == 1) {
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

        if (self->vlen == 1) {
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

    obj.def_rw("WarnOutOfBounds", &InterpTable1D::WarnOutOfBounds);
    obj.def_rw("ThrowOutOfBounds", &InterpTable1D::ThrowOutOfBounds);

    obj.def("sf", [](std::shared_ptr<InterpTable1D> &self) {
        if (self->vlen != 1) {
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

    obj.def_rw("WarnOutOfBounds", &InterpTable2D::WarnOutOfBounds);
    obj.def_rw("ThrowOutOfBounds", &InterpTable2D::ThrowOutOfBounds);

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

    m.def("InterpTable2DSpeedTest", [](const GenericFunction<-1, 1> &tabf, double xl, double xu,
                                       double yl, double yu, int nsamps, bool lin) {
        Eigen::ArrayXd xsamps;
        xsamps.setRandom(nsamps);
        xsamps += 1;
        xsamps /= 2;
        xsamps *= (xu - xl);
        xsamps += xl;

        Eigen::ArrayXd ysamps;
        ysamps.setRandom(nsamps);
        ysamps += 1;
        ysamps /= 2;
        ysamps *= (yu - yl);
        ysamps += yl;

        if (lin) {
            xsamps.setLinSpaced(xl, xu);
            ysamps.setLinSpaced(yl, yu);
        }

        Eigen::VectorXd xy(2);
        Vector1<double> f;
        f.setZero();

        Utils::Timer Runtimer;
        Runtimer.start();

        double tmp = 0;
        for (int i = 0; i < nsamps; i++) {

            xy[0] = xsamps[i];
            xy[1] = ysamps[i];

            tabf.compute(xy, f);
            tmp += f[0] / double(i + 3);

            // fmt::print("{0:} \n",f[0]);

            f.setZero();
        }
        Runtimer.stop();
        double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
        fmt::print("Total Time: {0:} ms \n", tseconds * 1000);

        return tmp;
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

    obj.def_rw("WarnOutOfBounds", &InterpTable3D::WarnOutOfBounds);
    obj.def_rw("ThrowOutOfBounds", &InterpTable3D::ThrowOutOfBounds);

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

    m.def("InterpTable3DSpeedTest",
          [](const GenericFunction<-1, 1> &tabf, double xl, double xu, double yl, double yu,
             double zl, double zu, int nsamps, bool lin) {
              Eigen::ArrayXd xsamps;
              xsamps.setRandom(nsamps);
              xsamps += 1;
              xsamps /= 2;
              xsamps *= (xu - xl);
              xsamps += xl;

              Eigen::ArrayXd ysamps;
              ysamps.setRandom(nsamps);
              ysamps += 1;
              ysamps /= 2;
              ysamps *= (yu - yl);
              ysamps += yl;

              Eigen::ArrayXd zsamps;
              zsamps.setRandom(nsamps);
              zsamps += 1;
              zsamps /= 2;
              zsamps *= (zu - zl);
              zsamps += zl;

              if (lin) {
                  xsamps.setLinSpaced(xl, xu);
                  ysamps.setLinSpaced(yl, yu);
                  zsamps.setLinSpaced(zl, zu);
              }

              Eigen::VectorXd xyz(3);
              Vector1<double> f;
              f.setZero();

              Utils::Timer Runtimer;
              Runtimer.start();

              double tmp = 0;
              for (int i = 0; i < nsamps; i++) {

                  xyz[0] = xsamps[i];
                  xyz[1] = ysamps[i];
                  xyz[2] = zsamps[i];

                  tabf.compute(xyz, f);
                  tmp += f[0] / double(i + 3);

                  f.setZero();
              }
              Runtimer.stop();
              double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
              fmt::print("Total Time: {0:} ms \n", tseconds * 1000);

              return tmp;
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

    obj.def_rw("WarnOutOfBounds", &InterpTable4D::WarnOutOfBounds);
    obj.def_rw("ThrowOutOfBounds", &InterpTable4D::ThrowOutOfBounds);

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

    m.def("InterpTable4DSpeedTest",
          [](const GenericFunction<-1, 1> &tabf, double xl, double xu, double yl, double yu,
             double zl, double zu, double wl, double wu,

             int nsamps, bool lin) {
              Eigen::ArrayXd xsamps;
              xsamps.setRandom(nsamps);
              xsamps += 1;
              xsamps /= 2;
              xsamps *= (xu - xl);
              xsamps += xl;

              Eigen::ArrayXd ysamps;
              ysamps.setRandom(nsamps);
              ysamps += 1;
              ysamps /= 2;
              ysamps *= (yu - yl);
              ysamps += yl;

              Eigen::ArrayXd zsamps;
              zsamps.setRandom(nsamps);
              zsamps += 1;
              zsamps /= 2;
              zsamps *= (zu - zl);
              zsamps += zl;

              Eigen::ArrayXd wsamps;
              wsamps.setRandom(nsamps);
              wsamps += 1;
              wsamps /= 2;
              wsamps *= (wu - wl);
              wsamps += wl;

              if (lin) {
                  xsamps.setLinSpaced(xl, xu);
                  ysamps.setLinSpaced(yl, yu);
                  zsamps.setLinSpaced(zl, zu);
                  wsamps.setLinSpaced(wl, wu);
              }

              Eigen::VectorXd xyzw(4);
              Vector1<double> f;
              f.setZero();

              Utils::Timer Runtimer;
              Runtimer.start();

              double tmp = 0;
              for (int i = 0; i < nsamps; i++) {

                  xyzw[0] = xsamps[i];
                  xyzw[1] = ysamps[i];
                  xyzw[2] = zsamps[i];
                  xyzw[3] = wsamps[i];

                  tabf.compute(xyzw, f);
                  tmp += f[0] / double(i + 3);

                  f.setZero();
              }
              Runtimer.stop();
              double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
              fmt::print("Total Time: {0:} ms \n", tseconds * 1000);

              return tmp;
          });
}

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
