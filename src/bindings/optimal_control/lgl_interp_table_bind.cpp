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

#include "ode_phase_bind.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;
using VectorFunctionalX = GenericFunction<-1, -1>;

void TychoBind<LGLInterpTable>::build(nb::module_ &m) {
    auto obj = nb::class_<LGLInterpTable>(
        m, "LGLInterpTable",
        R"doc(Piecewise-polynomial interpolation table over a discretized trajectory.

Builds a Legendre-Gauss-Lobatto (LGL) interpolant of a trajectory so the
state (and optionally control) can be evaluated at arbitrary times between
the stored nodes. A table can be constructed from an ODE (which supplies node
derivatives) plus trajectory data, or directly from trajectory data with the
derivatives recovered by finite difference.

A trajectory is a list of packed node vectors, each ordered ``[x, t, u, p]``
(the same layout used by phases). The ``time axis`` defaults to the time
component of that layout.

Notes
-----
Calling the table like a function -- ``table(t)`` -- is shorthand for
:meth:`interpolate` at time ``t``.
)doc");
    obj.def(nb::init<VectorFunctionalX, int, int, TranscriptionModes,
                     const std::vector<Eigen::VectorXd> &, int>());

    obj.def(nb::init<VectorFunctionalX, int, int, std::string, const std::vector<Eigen::VectorXd> &,
                     int>());

    obj.def(nb::init<VectorFunctionalX, int, int, int, std::string,
                     const std::vector<Eigen::VectorXd> &, int>());

    obj.def(nb::init<VectorFunctionalX, int, int, const std::vector<Eigen::VectorXd> &>());
    obj.def(nb::init<VectorFunctionalX, int, int, int, const std::vector<Eigen::VectorXd> &>());

    obj.def(nb::init<int, const std::vector<Eigen::VectorXd> &, int>());
    obj.def(nb::init<const std::vector<Eigen::VectorXd> &>());

    obj.def(nb::init<VectorFunctionalX, int, int, TranscriptionModes>());

    obj.def(nb::init<int, int, TranscriptionModes>());
    obj.def("load_even_data", &LGLInterpTable::load_even_data,
            R"doc(Load evenly-time-spaced node data, deriving node derivatives.

Parameters
----------
xtudat : list of numpy.ndarray
    Trajectory nodes, each packed as ``[x, t, u, p]``, evenly spaced in time.
    Derivatives are computed from the bound ODE (or by finite difference if
    the table was constructed without one).
)doc");
    obj.def("get_table_ptr", &LGLInterpTable::get_table_ptr);
    obj.def("load_uneven_data", &LGLInterpTable::load_uneven_data,
            R"doc(Load arbitrarily-spaced node data and resample onto an even mesh.

Parameters
----------
dnum : int
    Number of evenly-spaced segments to resample the trajectory onto.
xtudat : list of numpy.ndarray
    Trajectory nodes, each packed as ``[x, t, u, p]``, in strictly
    increasing time order.
)doc");
    obj.def("interpolate", &LGLInterpTable::interpolate<double>,
            R"doc(Interpolate the trajectory at a single time.

Parameters
----------
t : float
    Time at which to evaluate, within ``[t0, tf]``.

Returns
-------
numpy.ndarray
    The interpolated packed node vector ``[x, t, u, p]`` at time ``t``.
)doc");

    obj.def("new_error_integral", &LGLInterpTable::new_error_integral);

    obj.def(
        "__call__", nb::overload_cast<double>(&LGLInterpTable::interpolate<double>, nb::const_),
        nb::is_operator(),
        R"doc(Interpolate the trajectory at a single time (calling shorthand for :meth:`interpolate`).

Parameters
----------
t : float
    Time at which to evaluate, within ``[t0, tf]``.

Returns
-------
numpy.ndarray
    The interpolated packed node vector ``[x, t, u, p]`` at time ``t``.
)doc");

    obj.def("interpolate_deriv", &LGLInterpTable::interpolate_deriv<double>,
            R"doc(Interpolate the trajectory and its time derivative at a single time.

Parameters
----------
t : float
    Time at which to evaluate, within ``[t0, tf]``.

Returns
-------
tuple of numpy.ndarray
    The interpolated value and its time derivative at ``t``.
)doc");
    obj.def("make_periodic", &LGLInterpTable::make_periodic,
            R"doc(Mark the trajectory periodic, snapping the last node onto the first state.)doc");

    obj.def("interp_range", &LGLInterpTable::interp_range,
            R"doc(Interpolate the trajectory at evenly-spaced samples over a time range.

Parameters
----------
num : int
    Number of output samples.
tl : float
    Lower time bound.
th : float
    Upper time bound.

Returns
-------
list of numpy.ndarray
    ``num`` interpolated node vectors spanning ``[tl, th]``.
)doc");
    obj.def("interp_whole_range", &LGLInterpTable::interp_whole_range,
            R"doc(Interpolate the trajectory at evenly-spaced samples over ``[t0, tf]``.

Parameters
----------
num : int
    Number of output samples.

Returns
-------
list of numpy.ndarray
    ``num`` interpolated node vectors spanning the full table range.
)doc");
    obj.def("error_integral", &LGLInterpTable::error_integral);

    obj.def_ro("t0", &LGLInterpTable::t0_, "float: initial time of the table.");
    obj.def_ro("tf", &LGLInterpTable::tf_, "float: final time of the table.");

    obj.def("interp_non_dim", nb::overload_cast<int, double, double>(&LGLInterpTable::nd_equidist));
}
