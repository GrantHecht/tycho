#include "PythonArgParsing.h"

#include "CommonFunctions/CommonFunctions.h"
#include "FunctionRegistry.h"
#include "MathOverloads.h"
#include "OperatorOverloads.h"
#include "VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericComparative.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"

namespace Tycho {

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

    static nb::object vftype =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("VectorFunction");
    static nb::object sftype =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("ScalarFunction");
    static nb::object elemtype =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("Element");
    static nb::object segtype =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("Segment");
    static nb::object seg2type =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("Segment2");
    static nb::object seg3type =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("Segment3");
    static nb::object argtype =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("Arguments");

    static nb::object py_int =
        (nb::object)nb::module_::import_("builtins").attr("int");
    static nb::object py_float =
        (nb::object)nb::module_::import_("builtins").attr("float");
    static nb::object py_list =
        (nb::object)nb::module_::import_("builtins").attr("list");
    static nb::object np_array = (nb::object)nb::module_::import_("numpy").attr("ndarray");
    static nb::object np_float = (nb::object)nb::module_::import_("numpy").attr("float64");
    static nb::object np_int = (nb::object)nb::module_::import_("numpy").attr("int32");

    int i = 0;
    for (nb::handle xi : x) {
        if (xi.type().is(vftype) || xi.type().is(sftype) || xi.type().is(elemtype) ||
            xi.type().is(segtype) || xi.type().is(seg2type) ||
            xi.type().is(seg3type) || xi.type().is(argtype)) {
            int irowstmp = nb::cast<int>(xi.attr("IRows")());
            if (irows == 0) {
                irows = irowstmp;
            } else if (irowstmp != irows) {
                throw std::invalid_argument("Asset functions in list must have same input size");
            }

        } else if (xi.type().is(py_float) || xi.type().is(py_int) ||
                   xi.type().is(np_int) || xi.type().is(np_float)) {

            // Good to go
        } else if (xi.type().is(py_list) || xi.type().is(np_array)) {
            // Loop over and check that these are arrays of doubles or ints
            int lenvec = nb::cast<int>(xi.attr("__len__")());
            for (int j = 0; j < lenvec; j++) {
                auto elemj = xi.attr("__getitem__")(nb::int_(j)).type();
                if (!(elemj.is(py_float) || elemj.is(py_int) || elemj.is(np_int) ||
                      elemj.is(np_float))) {
                    nb::print(nb::str(elemj));
                    throw std::invalid_argument(
                        "Vectors and lists must only contain doubles or floats");
                }
            }
        }

        else {
            nb::print(nb::str(xi.type()));
            throw std::invalid_argument("Argument cannot be converted to Asset function");
        }

        i++;
    }

    if (irows == 0) {
        throw std::invalid_argument("Argument list must contain at least one asset function.");
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
        } else if (xi.type().is(py_float) || xi.type().is(py_int) ||
                   xi.type().is(np_int) || xi.type().is(np_float)) {
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

    static nb::object sftype =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("ScalarFunction");
    static nb::object elemtype =
        (nb::object)nb::module_::import_("_tycho.VectorFunctions").attr("Element");

    static nb::object py_int =
        (nb::object)nb::module_::import_("builtins").attr("int");
    static nb::object py_float =
        (nb::object)nb::module_::import_("builtins").attr("float");
    static nb::object np_float = (nb::object)nb::module_::import_("numpy").attr("float64");
    static nb::object np_int = (nb::object)nb::module_::import_("numpy").attr("int32");

    int i = 0;
    for (nb::handle xi : x) {
        if (xi.type().is(sftype) || xi.type().is(elemtype)) {
            int irowstmp = nb::cast<int>(xi.attr("IRows")());
            if (irows == 0) {
                irows = irowstmp;
            } else if (irowstmp != irows) {
                throw std::invalid_argument("Asset functions in list must have same input size");
            }

        } else if (xi.type().is(py_float) || xi.type().is(py_int) ||
                   xi.type().is(np_int) || xi.type().is(np_float)) {
            // Good to go
        } else {
            nb::print(nb::str(xi.type()));
            throw std::invalid_argument("Argument cannot be converted to Asset function");
        }

        i++;
    }
    if (irows == 0) {
        throw std::invalid_argument("Argument list must contain at least one asset function.");
    }
    std::vector<Rtype> funs;
    int Elem = 0;
    for (nb::handle xi : x) {
        if (xi.type().is(sftype)) {
            funs.emplace_back(Rtype(nb::cast<GenS>(xi)));
        } else if (xi.type().is(elemtype)) {
            funs.emplace_back(Rtype(nb::cast<ELEM>(xi)));
        } else if (xi.type().is(py_float) || xi.type().is(py_int) ||
                   xi.type().is(np_float) || xi.type().is(np_int)) {
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
} // namespace Tycho
