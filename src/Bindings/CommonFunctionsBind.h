#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class definitions of CommonFunctions binding Build() / SegBuild() methods.
// Included from CommonFunctions/CommonFunctions.h under TYCHO_PYTHON_BINDINGS.

namespace Tycho {

// ── Constant ──────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR>
void Constant<IR, OR>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Constant<IR, OR>>(m, name);
    obj.def(nb::init<int, Output<double>>());
    Base::DenseBaseBuild(obj);
}

// ── FunctionDotProduct_Impl ───────────────────────────────────────────────────────────────────────
template <class Derived, class Func1, class Func2>
void FunctionDotProduct_Impl<Derived, Func1, Func2>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Derived>(m, name);
    obj.def(nb::init<Func1, Func2>());
    Base::DenseBaseBuild(obj);
}

// ── NestedFunction ────────────────────────────────────────────────────────────────────────────────
template <class OuterFunc, class InnerFunc>
void NestedFunction<OuterFunc, InnerFunc>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<NestedFunction<OuterFunc, InnerFunc>>(m, name);
    obj.def(nb::init<>());
    obj.def(nb::init<OuterFunc, InnerFunc>());
    Base::DenseBaseBuild(obj);
}

// ── NormalizedPower_Impl ──────────────────────────────────────────────────────────────────────────
template <class Derived, int IR, int PW>
void NormalizedPower_Impl<Derived, IR, PW>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Derived>(m, name);
    obj.def(nb::init<int>());
    if constexpr (IR > 0) {
        obj.def(nb::init<>());
    }
    Base::DenseBaseBuild(obj);
}

// ── IntegralNorm_Impl ─────────────────────────────────────────────────────────────────────────────
template <class Derived, int USZ, int Power>
void IntegralNorm_Impl<Derived, USZ, Power>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Derived>(m, name);
    obj.def(nb::init<int>());
    if constexpr (USZ > 0) {
        obj.def(nb::init<>());
    }
    Base::DenseBaseBuild(obj);
}

// ── ParsedInput ───────────────────────────────────────────────────────────────────────────────────
template <class Func, int IRC, int ORC>
void ParsedInput<Func, IRC, ORC>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<ParsedInput<Func, IRC, ORC>>(m, name);
    obj.def(nb::init<Func, const Func_Input<int> &, int>());
    Base::DenseBaseBuild(obj);
}

// ── Arguments ─────────────────────────────────────────────────────────────────────────────────────
template <int IR_OR>
void Arguments<IR_OR>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Arguments<IR_OR>>(m, name);
    obj.def(nb::init<int>());
    obj.def("Constant", [](const Arguments<IR_OR> &a, Eigen::VectorXd v) {
        return GenericFunction<-1, -1>(Constant<-1, -1>(a.IRows(), v));
    });
    obj.def("Constant", [](const Arguments<IR_OR> &a, double v) {
        Eigen::Matrix<double, 1, 1> vv;
        vv[0] = v;
        return GenericFunction<-1, 1>(Constant<-1, 1>(a.IRows(), vv));
    });
    Base::DenseBaseBuild(obj);
    Base::SegBuild(obj);
}

// ── Segment ───────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR, int ST>
void Segment<IR, OR, ST>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Segment<IR, OR, ST>>(m, name);
    obj.def(nb::init<int, int, int>());
    Base::DenseBaseBuild(obj);
    Base::SegBuild(obj);
}

// ── Segment_Impl::SegBuild ────────────────────────────────────────────────────────────────────────
template <class Derived, int IR, int OR, int ST>
template <class PyClass>
void Segment_Impl<Derived, IR, OR, ST>::SegBuild(PyClass &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    Base::DoubleMathBuild(obj);
    Base::UnaryMathBuild(obj);
    Base::BinaryMathBuild(obj);
    Base::BinaryOperatorsBuild(obj);
    Base::FunctionIndexingBuild(obj);
    Base::ConditionalOperatorsBuild(obj);

    obj.def("tolist", [](const Derived &func) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (int i = 0; i < func.ORows(); i++) {
            elems.push_back(func.coeff(i));
        }
        return elems;
    });

    obj.def("tolist", [](const Derived &func, std::vector<int> coeffs) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (const auto &coeff : coeffs) {
            elems.push_back(func.coeff(coeff));
        }
        return elems;
    });

    obj.def("tolist", [](const Derived &func, std::vector<std::tuple<int, int>> seglist) {
        using ELEM = Segment<-1, 1, -1>;
        using SEG2 = Segment<-1, 2, -1>;
        using SEG3 = Segment<-1, 3, -1>;
        using SEG = Segment<-1, -1, -1>;

        std::vector<nb::object> segs;
        for (const auto &seg : seglist) {

            int start = std::get<0>(seg);
            int size = std::get<1>(seg);
            nb::object pyfun;
            if (size == 1) {
                auto f = func.coeff(start);
                pyfun = nb::cast(f);
            } else if (size == 2) {
                auto f = func.template segment<2>(start);
                pyfun = nb::cast(f);
            } else if (size == 3) {
                auto f = func.template segment<3>(start);
                pyfun = nb::cast(f);
            } else {
                auto f = func.segment(start, size);
                pyfun = nb::cast(f);
            }

            segs.push_back(pyfun);
        }
        return segs;
    });
}

// ── TwoFunctionSum_Impl ───────────────────────────────────────────────────────────────────────────
template <class Derived, class Func1, class Func2, bool DoDifference>
void TwoFunctionSum_Impl<Derived, Func1, Func2, DoDifference>::Build(nb::module_ &m,
                                                                      const char *name) {
    auto obj = nb::class_<Derived>(m, name);
    obj.def(nb::init<Func1, Func2>());
    Base::DenseBaseBuild(obj);
}

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
