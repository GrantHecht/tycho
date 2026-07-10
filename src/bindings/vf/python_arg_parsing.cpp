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

#include "python_arg_parsing.h"

#include <fmt/format.h>

#include "function_registry.h"
#include "tycho/detail/vf/common/common_functions.h"
#include "tycho/detail/vf/core/vector_function.h"
#include "tycho/detail/vf/operators/math_overloads.h"
#include "tycho/detail/vf/operators/operator_overloads.h"
#include "tycho/detail/vf/type_erasure/generic_comparative.h"
#include "tycho/detail/vf/type_erasure/generic_conditional.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"

namespace tycho {

std::vector<GenericFunction<-1, -1>> ParsePythonArgs(nb::args x, int irows) {

    using std::cin;
    using std::cout;
    using std::endl;

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    using Rtype = Gen;

    // Static PyObject* (not nb::object) so destructors never run after interpreter shutdown.
    // The lambda trick forces assignment through nb::object to trigger accessor evaluation.
    static PyObject *vftype = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("VectorFunction");
        return o.release().ptr();
    }();
    static PyObject *sftype = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("ScalarFunction");
        return o.release().ptr();
    }();
    static PyObject *elemtype = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("Element");
        return o.release().ptr();
    }();
    static PyObject *segtype = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("Segment");
        return o.release().ptr();
    }();
    static PyObject *seg2type = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("Segment2");
        return o.release().ptr();
    }();
    static PyObject *seg3type = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("Segment3");
        return o.release().ptr();
    }();
    static PyObject *argtype = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("Arguments");
        return o.release().ptr();
    }();
    static PyObject *py_int = [] {
        nb::object o = nb::module_::import_("builtins").attr("int");
        return o.release().ptr();
    }();
    static PyObject *py_float = [] {
        nb::object o = nb::module_::import_("builtins").attr("float");
        return o.release().ptr();
    }();
    static PyObject *py_list = [] {
        nb::object o = nb::module_::import_("builtins").attr("list");
        return o.release().ptr();
    }();
    static PyObject *np_array = [] {
        nb::object o = nb::module_::import_("numpy").attr("ndarray");
        return o.release().ptr();
    }();
    static PyObject *np_float = [] {
        nb::object o = nb::module_::import_("numpy").attr("float64");
        return o.release().ptr();
    }();
    static PyObject *np_int = [] {
        nb::object o = nb::module_::import_("numpy").attr("int32");
        return o.release().ptr();
    }();
    // numpy.number covers every signed/unsigned int and float width (including 0-d array
    // scalars boxed as np.float64/np.int64), unlike the np_int/np_float exact-type checks
    // above, which only match numpy's int32/float64 aliases.
    static PyObject *np_number = [] {
        nb::object o = nb::module_::import_("numpy").attr("number");
        return o.release().ptr();
    }();

    auto is_numeric_scalar = [&](nb::handle h) {
        // Exact-type fast paths first, then the general numpy.number ABC check for every
        // other signed/unsigned int and float width (int8/16/64, uint*, float16/32, ...).
        if (h.type().is(py_float) || h.type().is(py_int) || h.type().is(np_int) ||
            h.type().is(np_float) || PyBool_Check(h.ptr())) {
            return true;
        }
        int r = PyObject_IsInstance(h.ptr(), np_number);
        if (r < 0) {
            PyErr_Clear();
        }
        return r == 1;
    };

    int i = 0;
    for (nb::handle xi : x) {
        if (xi.type().is(vftype) || xi.type().is(sftype) || xi.type().is(elemtype) ||
            xi.type().is(segtype) || xi.type().is(seg2type) || xi.type().is(seg3type) ||
            xi.type().is(argtype)) {
            int irowstmp = nb::cast<int>(xi.attr("input_rows")());
            if (irows == 0) {
                irows = irowstmp;
            } else if (irowstmp != irows) {
                throw std::invalid_argument("VectorFunctions in list must have same input size");
            }

        } else if (is_numeric_scalar(xi)) {

            // Good to go
        } else if (xi.type().is(py_list) || xi.type().is(np_array)) {
            // Loop over and check that these are arrays of doubles or ints
            int lenvec = nb::cast<int>(xi.attr("__len__")());
            for (int j = 0; j < lenvec; j++) {
                nb::object elemj = xi.attr("__getitem__")(nb::int_(j));
                if (!is_numeric_scalar(elemj)) {
                    throw std::invalid_argument(fmt::format(
                        "Vectors and lists must only contain ints or floats; got element of "
                        "type {}",
                        nb::cast<std::string>(nb::str(elemj.type()))));
                }
            }
        }

        else {
            throw std::invalid_argument(
                fmt::format("Argument cannot be converted to VectorFunction; got type {}",
                            nb::cast<std::string>(nb::str(xi.type()))));
        }

        i++;
    }

    if (irows == 0) {
        throw std::invalid_argument("Argument list must contain at least one VectorFunction.");
    }

    std::vector<Rtype> funs;
    int Elem = 0;
    for (nb::handle xi : x) {
        if (xi.type().is(vftype)) {
            funs.emplace_back(Rtype(nb::cast<Gen>(xi)));
        } else if (xi.type().is(sftype)) {
            funs.emplace_back(Rtype(nb::cast<GenS>(xi)));
        } else if (xi.type().is(elemtype)) {
            funs.emplace_back(Rtype(nb::cast<ELEM>(xi)));
        } else if (xi.type().is(segtype)) {
            funs.emplace_back(Rtype(nb::cast<SEG>(xi)));
        } else if (xi.type().is(seg2type)) {
            funs.emplace_back(Rtype(nb::cast<SEG2>(xi)));
        } else if (xi.type().is(seg3type)) {
            funs.emplace_back(Rtype(nb::cast<SEG3>(xi)));
        } else if (xi.type().is(argtype)) {
            funs.emplace_back(Rtype(nb::cast<Arguments<-1>>(xi)));
        } else if (is_numeric_scalar(xi)) {
            Vector1<double> val;
            val[0] = nb::cast<double>(xi);
            funs.emplace_back(Constant<-1, 1>(irows, val));
        } else if (xi.type().is(py_list) || xi.type().is(np_array)) {
            int lenvec = nb::cast<int>(xi.attr("__len__")());
            VectorX<double> val(lenvec);
            for (int j = 0; j < lenvec; j++) {
                val[j] = nb::cast<double>(xi.attr("__getitem__")(nb::int_(j)));
            }
            funs.emplace_back(Constant<-1, -1>(irows, val));
        } else {

            throw std::invalid_argument("Unrecognized Argument.");
        }
        Elem++;
    }

    return funs;
}

std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(nb::args x, int irows) {

    using std::cin;
    using std::cout;
    using std::endl;

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    using Rtype = GenS;

    // Static PyObject* (not nb::object) so destructors never run after interpreter shutdown.
    static PyObject *sftype = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("ScalarFunction");
        return o.release().ptr();
    }();
    static PyObject *elemtype = [] {
        nb::object o = nb::module_::import_("_tychopy.vector_functions").attr("Element");
        return o.release().ptr();
    }();
    static PyObject *py_int = [] {
        nb::object o = nb::module_::import_("builtins").attr("int");
        return o.release().ptr();
    }();
    static PyObject *py_float = [] {
        nb::object o = nb::module_::import_("builtins").attr("float");
        return o.release().ptr();
    }();
    static PyObject *np_float = [] {
        nb::object o = nb::module_::import_("numpy").attr("float64");
        return o.release().ptr();
    }();
    static PyObject *np_int = [] {
        nb::object o = nb::module_::import_("numpy").attr("int32");
        return o.release().ptr();
    }();
    // numpy.number covers every signed/unsigned int and float width (including 0-d array
    // scalars boxed as np.float64/np.int64), unlike the np_int/np_float exact-type checks
    // above, which only match numpy's int32/float64 aliases.
    static PyObject *np_number = [] {
        nb::object o = nb::module_::import_("numpy").attr("number");
        return o.release().ptr();
    }();

    auto is_numeric_scalar = [&](nb::handle h) {
        // Exact-type fast paths first, then the general numpy.number ABC check for every
        // other signed/unsigned int and float width (int8/16/64, uint*, float16/32, ...).
        if (h.type().is(py_float) || h.type().is(py_int) || h.type().is(np_int) ||
            h.type().is(np_float) || PyBool_Check(h.ptr())) {
            return true;
        }
        int r = PyObject_IsInstance(h.ptr(), np_number);
        if (r < 0) {
            PyErr_Clear();
        }
        return r == 1;
    };

    int i = 0;
    for (nb::handle xi : x) {
        if (xi.type().is(sftype) || xi.type().is(elemtype)) {
            int irowstmp = nb::cast<int>(xi.attr("input_rows")());
            if (irows == 0) {
                irows = irowstmp;
            } else if (irowstmp != irows) {
                throw std::invalid_argument("VectorFunctions in list must have same input size");
            }

        } else if (is_numeric_scalar(xi)) {
            // Good to go
        } else {
            throw std::invalid_argument(
                fmt::format("Argument cannot be converted to VectorFunction; got type {}",
                            nb::cast<std::string>(nb::str(xi.type()))));
        }

        i++;
    }
    if (irows == 0) {
        throw std::invalid_argument("Argument list must contain at least one VectorFunction.");
    }
    std::vector<Rtype> funs;
    int Elem = 0;
    for (nb::handle xi : x) {
        if (xi.type().is(sftype)) {
            funs.emplace_back(Rtype(nb::cast<GenS>(xi)));
        } else if (xi.type().is(elemtype)) {
            funs.emplace_back(Rtype(nb::cast<ELEM>(xi)));
        } else if (is_numeric_scalar(xi)) {
            Vector1<double> val;
            val[0] = nb::cast<double>(xi);
            funs.emplace_back(Constant<-1, 1>(irows, val));
        } else {
            throw std::invalid_argument("Unrecognized Argument.");
        }
        Elem++;
    }

    return funs;
}
} // namespace tycho
