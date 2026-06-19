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

// Free-function binding helper for ODESize member definitions.
// Used by ODEBind.h and any other code that needs to expose ODE size members.

namespace tycho::bind {

using namespace tycho::utils;

template <int XV, int UV, int PV, class Derived, class Obj> void ODESizeBuild(Obj &obj) {
    obj.def("x_vars", &Derived::x_vars,
            R"doc(Return the number of state variables.

Returns
-------
int
    State-vector dimension ``XV``.
)doc");
    obj.def("u_vars", &Derived::u_vars,
            R"doc(Return the number of control variables.

Returns
-------
int
    Control-vector dimension ``UV``.
)doc");
    obj.def("p_vars", &Derived::p_vars,
            R"doc(Return the number of parameter variables.

Returns
-------
int
    Parameter-vector dimension ``PV``.
)doc");
    obj.def("t_var", &Derived::t_var,
            R"doc(Return the index of the time variable within the packed node vector.

The packed node layout is ``[x_0, ..., x_{XV-1}, t, u_0, ..., p_{PV-1}]``,
so the time index equals ``x_vars()``.

Returns
-------
int
    Index of the time variable.
)doc");

    obj.def("xt_vars", &Derived::xt_vars,
            R"doc(Return the combined state + time variable count.

Returns
-------
int
    ``x_vars() + 1``.
)doc");
    obj.def("xtu_vars", &Derived::xtu_vars,
            R"doc(Return the combined state + time + control variable count.

Returns
-------
int
    ``x_vars() + 1 + u_vars()``.
)doc");
    obj.def("xtup_vars", &Derived::xtu_p_vars,
            R"doc(Return the total variable count (state + time + control + parameters).

Returns
-------
int
    ``x_vars() + 1 + u_vars() + p_vars()``.
)doc");

    obj.def("x_idxs", nb::overload_cast<>(&Derived::x_idxs, nb::const_),
            R"doc(Return the absolute indices of all state variables.

Returns
-------
numpy.ndarray of int
    Index vector ``[0, 1, ..., x_vars()-1]``.
)doc");
    obj.def("x_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::x_idxs, nb::const_), "");

    obj.def("xt_idxs", nb::overload_cast<>(&Derived::xt_idxs, nb::const_),
            R"doc(Return the absolute indices of the state and time variables.

Returns
-------
numpy.ndarray of int
    Index vector ``[0, 1, ..., xt_vars()-1]``.
)doc");
    obj.def("xt_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::xt_idxs, nb::const_),
            "");

    obj.def("xtu_idxs", nb::overload_cast<>(&Derived::xtu_idxs, nb::const_),
            R"doc(Return the absolute indices of the state, time, and control variables.

Returns
-------
numpy.ndarray of int
    Index vector ``[0, 1, ..., xtu_vars()-1]``.
)doc");
    obj.def("xtu_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::xtu_idxs, nb::const_),
            "");

    obj.def("u_idxs", nb::overload_cast<>(&Derived::u_idxs, nb::const_),
            R"doc(Return the absolute indices of the control variables.

Returns
-------
numpy.ndarray of int
    Index vector ``[xt_vars(), ..., xtu_vars()-1]``.
)doc");
    obj.def("u_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::u_idxs, nb::const_), "");

    obj.def("p_idxs", nb::overload_cast<>(&Derived::p_idxs, nb::const_),
            R"doc(Return the absolute indices of the parameter variables.

Returns
-------
numpy.ndarray of int
    Index vector ``[xtu_vars(), ..., xtup_vars()-1]``.
)doc");
    obj.def("p_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::p_idxs, nb::const_), "");

    obj.def("add_idx",
            nb::overload_cast<const std::string &, const Eigen::VectorXi &>(&Derived::add_idx),
            R"doc(Register a named group of variable indices.

Parameters
----------
name : str
    Unique name for the group.
idx : numpy.ndarray of int
    Indices (into the full packed ``[x, t, u, p]`` node vector) belonging
    to the group.

Raises
------
ValueError
    If ``idx`` is empty or a group named ``name`` already exists.
)doc");
    obj.def(
        "get_idxs",
        [](const Derived &self) {
            nb::dict result;
            for (const auto &[k, v] : self.get_idxs()) {
                result[nb::cast(k)] = nb::cast(v);
            }
            return result;
        },
        R"doc(Return all named variable index groups as a dictionary.

Returns
-------
dict[str, numpy.ndarray of int]
    Mapping from group name to index vector, as registered via
    :meth:`add_idx` or :meth:`set_idxs`.
)doc");
    obj.def(
        "set_idxs",
        [](Derived &self, nb::dict d) {
            FlatMap<std::string, Eigen::VectorXi> fm;
            for (auto [k, v] : d) {
                fm[nb::cast<std::string>(k)] = nb::cast<Eigen::VectorXi>(v);
            }
            self.set_idxs(fm);
        },
        R"doc(Replace the entire named variable index group registry.

Parameters
----------
idxs : dict[str, numpy.ndarray of int]
    New registry mapping group name to index vector.  Every group must be
    non-empty; the previous registry is discarded.

Raises
------
ValueError
    If any group in ``idxs`` is empty.
)doc");
    obj.def("idx", &Derived::idx,
            R"doc(Look up a named variable index group by name.

Parameters
----------
name : str
    Name of a group previously registered with :meth:`add_idx`.

Returns
-------
numpy.ndarray of int
    The index vector registered under ``name``.

Raises
------
ValueError
    If no group named ``name`` exists.
)doc");
}

} // namespace tycho::bind

#endif // TYCHO_PYTHON_BINDINGS
