#include "Tycho_VectorFunctions.h"

namespace Tycho {

template <class FType> void DefineListEval(nb::class_<FType> &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    obj.def("__call__", [](const FType &fun, const std::vector<GenS> &funcs) {
        return FType(fun.eval(DynamicStack(funcs)));
    });
    obj.def("__call__", [](const FType &fun, const std::vector<Gen> &funcs) {
        return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType &fun, const Gen &first, nb::args x) {
        auto funcs = std::vector{first};
        auto funcsrest = ParsePythonArgs(x, first.IRows());
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType &fun, double first, nb::args x) {
        auto funcsrest = ParsePythonArgs(x);
        Vector1<double> val;
        val[0] = first;
        auto funcs = std::vector{Gen(Constant<-1, 1>(funcsrest[0].IRows(), val))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType &fun, Eigen::VectorXd first, nb::args x) {
        auto funcsrest = ParsePythonArgs(x);
        auto funcs = std::vector{Gen(Constant<-1, -1>(funcsrest[0].IRows(), first))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return FType(fun.eval(DynamicStack(funcs)));
    });
}

void VectorFunctionBuildPart2(FunctionRegistry &reg, nb::module_ &m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    Bind::DoubleMathBuild<GenS>(reg.sfuncx);
    Bind::UnaryMathBuild<GenS>(reg.sfuncx);
    Bind::BinaryMathBuild<GenS>(reg.sfuncx);
    Bind::BinaryOperatorsBuild<GenS>(reg.sfuncx);
    Bind::FunctionIndexingBuild<GenS>(reg.sfuncx);
    Bind::ConditionalOperatorsBuild<GenS>(reg.sfuncx);

    ///////////////////////////////////////
    Bind::DoubleMathBuild<Gen>(reg.vfuncx);
    Bind::FunctionIndexingBuild<Gen>(reg.vfuncx);
    Bind::BinaryOperatorsBuild<Gen>(reg.vfuncx);
    ///////////////////////////////////////

    DefineListEval(reg.vfuncx);
    DefineListEval(reg.sfuncx);
}

} // namespace Tycho