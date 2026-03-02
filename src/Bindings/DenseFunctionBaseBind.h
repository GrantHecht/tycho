#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Free function templates implementing DenseFunctionBase binding helpers.
// These replace the out-of-class member function definitions that previously
// lived in DenseFunctionBase.h under TYCHO_PYTHON_BINDINGS.

namespace Tycho::Bind {

template <class Derived, class PYClass>
void DenseBaseBuild(PYClass &obj) {

    using Gen = GenericFunction<-1, -1>;

    obj.def("IRows", &Derived::IRows);
    obj.def("ORows", &Derived::ORows);
    obj.def("name", &Derived::name);

    obj.def("input_domain", &Derived::input_domain);
    obj.def("is_linear", &Derived::is_linear);

    // Generic lambda bodies — defined once, called from both the zero-copy
    // (ConstEigenRef/numpy) and the sequence-fallback (VectorXd) overloads.
    auto compute_body = [](const Derived &func, const auto &x) -> Eigen::VectorXd {
        if ((int)x.size() != func.IRows())
            throw std::invalid_argument("Incorrectly sized input to function");
        Eigen::VectorXd fx(func.ORows());
        fx.setZero();
        func.derived().compute(x, fx);
        return fx;
    };
    auto jacobian_body = [](const Derived &func, const auto &x) -> Eigen::MatrixXd {
        if ((int)x.size() != func.IRows())
            throw std::invalid_argument("Incorrectly sized input to function");
        Eigen::MatrixXd jx(func.ORows(), func.IRows());
        jx.setZero();
        func.derived().jacobian(x, jx);
        return jx;
    };
    auto compute_jacobian_body =
        [](const Derived &func,
           const auto &x) -> std::tuple<Eigen::VectorXd, Eigen::MatrixXd> {
        if ((int)x.size() != func.IRows())
            throw std::invalid_argument("Incorrectly sized input to function");
        Eigen::VectorXd fx(func.ORows());
        fx.setZero();
        Eigen::MatrixXd jx(func.ORows(), func.IRows());
        jx.setZero();
        func.derived().compute_jacobian(x, fx, jx);
        return std::tuple{fx, jx};
    };
    auto adjointgradient_body =
        [](const Derived &func, const auto &x, const auto &lm) -> Eigen::VectorXd {
        if ((int)x.size() != func.IRows())
            throw std::invalid_argument("Incorrectly sized input to function");
        if ((int)lm.size() != func.ORows())
            throw std::invalid_argument("Incorrectly sized multiplier input to function");
        Eigen::VectorXd ax(func.IRows());
        ax.setZero();
        func.derived().adjointgradient(x, ax, lm);
        return ax;
    };
    auto adjointhessian_body =
        [](const Derived &func, const auto &x, const auto &lm) -> Eigen::MatrixXd {
        if ((int)x.size() != func.IRows())
            throw std::invalid_argument("Incorrectly sized input to function");
        if ((int)lm.size() != func.ORows())
            throw std::invalid_argument("Incorrectly sized multiplier input to function");
        Eigen::MatrixXd hx(func.IRows(), func.IRows());
        hx.setZero();
        func.derived().adjointhessian(x, hx, lm);
        return hx;
    };
    auto computeall_body =
        [](const Derived &func,
           const auto &x,
           const auto &lm) -> std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd,
                                          Eigen::MatrixXd> {
        if ((int)x.size() != func.IRows())
            throw std::invalid_argument("Incorrectly sized input to function");
        if ((int)lm.size() != func.ORows())
            throw std::invalid_argument("Incorrectly sized multiplier input to function");
        Eigen::VectorXd fx(func.ORows());
        Eigen::MatrixXd jx(func.ORows(), func.IRows());
        Eigen::VectorXd gx(func.IRows());
        Eigen::MatrixXd hx(func.IRows(), func.IRows());
        fx.setZero();
        jx.setZero();
        gx.setZero();
        hx.setZero();
        func.derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);
        return std::tuple{fx, jx, gx, hx};
    };

    // Two overloads per method are required:
    //   1. ConstEigenRef<VectorXd> — zero-copy reference into a numpy array buffer.
    //      Uses nanobind's built-in Eigen caster; does NOT invoke our VectorXd type_caster.
    //   2. VectorXd (by value)   — invokes our custom type_caster which accepts Python
    //      lists, tuples, and other sequences via its from_python fallback path.
    // Without overload 1, every numpy call copies. Without overload 2, lists/tuples fail.
    obj.def("compute",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) { return compute_body(f, x); });
    obj.def("compute",
            [=](const Derived &f, Eigen::VectorXd x) { return compute_body(f, x); });

    obj.def("__call__",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) { return compute_body(f, x); });
    obj.def("__call__",
            [=](const Derived &f, Eigen::VectorXd x) { return compute_body(f, x); });

    obj.def("jacobian",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) { return jacobian_body(f, x); });
    obj.def("jacobian",
            [=](const Derived &f, Eigen::VectorXd x) { return jacobian_body(f, x); });

    obj.def("compute_jacobian",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) { return compute_jacobian_body(f, x); });
    obj.def("compute_jacobian",
            [=](const Derived &f, Eigen::VectorXd x) { return compute_jacobian_body(f, x); });

    obj.def("adjointgradient",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x, ConstEigenRef<Eigen::VectorXd> lm) { return adjointgradient_body(f, x, lm); });
    obj.def("adjointgradient",
            [=](const Derived &f, Eigen::VectorXd x, Eigen::VectorXd lm) { return adjointgradient_body(f, x, lm); });

    obj.def("adjointhessian",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x, ConstEigenRef<Eigen::VectorXd> lm) { return adjointhessian_body(f, x, lm); });
    obj.def("adjointhessian",
            [=](const Derived &f, Eigen::VectorXd x, Eigen::VectorXd lm) { return adjointhessian_body(f, x, lm); });

    obj.def("computeall",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x, ConstEigenRef<Eigen::VectorXd> lm) { return computeall_body(f, x, lm); });
    obj.def("computeall",
            [=](const Derived &f, Eigen::VectorXd x, Eigen::VectorXd lm) { return computeall_body(f, x, lm); });

    obj.def("rpt", &Derived::rpt);
    obj.def("vf", &Derived::template MakeGeneric<GenericFunction<-1, -1>>);

    constexpr int OR = Derived::ORC;
    if constexpr (OR == 1) {
        obj.def("sf", &Derived::template MakeGeneric<GenericFunction<-1, 1>>);
    }
}

template <class Derived, class PYClass>
void DoubleMathBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    // Prevents numpy from overriding __radd__ and __rmul__
    obj.attr("__array_ufunc__") = nb::none();

    obj.def(
        "__add__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(a + b); },
        nb::is_operator());
    obj.def(
        "__radd__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(a + b); },
        nb::is_operator());
    obj.def(
        "__sub__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(a - b); },
        nb::is_operator());
    obj.def(
        "__rsub__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(b - a); },
        nb::is_operator());

    obj.def(
        "__mul__", [](const Derived &a, double b) { return BinGen(a * b); }, nb::is_operator());
    obj.def(
        "__rmul__", [](const Derived &a, double b) { return BinGen(a * b); },
        nb::is_operator());

    obj.def("__neg__", [](const Derived &a) { return BinGen(a * (-1.0)); }, nb::is_operator());

    obj.def(
        "__truediv__", [](const Derived &a, double b) { return BinGen(a * (1.0 / b)); },
        nb::is_operator());
    obj.def(
        "__truediv__",
        [](const Derived &a, const Segment<-1, 1, -1> &b) { return BinGen(a / b); },
        nb::is_operator());

    if constexpr (OR == 1) { // Scalars
        obj.def(
            "__add__", [](const Derived &a, double b) { return BinGen(a + b); },
            nb::is_operator());
        obj.def(
            "__radd__", [](const Derived &a, double b) { return BinGen(a + b); },
            nb::is_operator());
        obj.def(
            "__sub__", [](const Derived &a, double b) { return BinGen(a - b); },
            nb::is_operator());
        obj.def(
            "__rsub__", [](const Derived &a, double b) { return BinGen(b - a); },
            nb::is_operator());
        obj.def(
            "__rtruediv__", [](const Derived &a, double b) { return BinGen(b / a); },
            nb::is_operator());

        obj.def(
            "__rmul__",
            [](const Derived &a, const Eigen::VectorXd &b) {
                return Gen(MatrixScaled<Derived, -1>(a, b));
            },
            nb::is_operator());

        obj.def(
            "__mul__",
            [](const Derived &a, const Eigen::VectorXd &b) {
                return Gen(MatrixScaled<Derived, -1>(a, b));
            },
            nb::is_operator());
    }
}

template <class Derived, class PYClass>
void UnaryMathBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    if constexpr (OR != 1) {

        if constexpr (!std::is_same<PYClass, nb::module_>::value) {
            obj.def("sum", [](const Derived &a) { return GenS(a.sum()); });
        }

        obj.def("normalized_power3", [](const Derived &a, Eigen::VectorXd b) {
            return Gen((a + b).template normalized_power<3>());
        });
        obj.def("normalized_power3", [](const Derived &a, Eigen::VectorXd b, double s) {
            return Gen(((a + b).template normalized_power<3>()) * s);
        });
        ////////////////////////////////////////////////////////////////////
        if constexpr (OR > 0) { // already constant size

            obj.def("norm", [](const Derived &a) { return GenS(a.norm()); });
            obj.def("squared_norm", [](const Derived &a) { return GenS(a.squared_norm()); });
            obj.def("cubed_norm",
                    [](const Derived &a) { return GenS(a.template norm_power<3>()); });

            obj.def("inverse_norm", [](const Derived &a) { return GenS(a.inverse_norm()); });
            obj.def("inverse_squared_norm",
                    [](const Derived &a) { return GenS(a.inverse_squared_norm()); });
            obj.def("inverse_cubed_norm",
                    [](const Derived &a) { return GenS(a.template inverse_norm_power<3>()); });
            obj.def("inverse_four_norm",
                    [](const Derived &a) { return GenS(a.template inverse_norm_power<4>()); });

            obj.def("normalized", [](const Derived &a) { return Gen(a.normalized()); });

            obj.def("normalized_power2",
                    [](const Derived &a) { return Gen(a.template normalized_power<2>()); });

            obj.def("normalized_power4",
                    [](const Derived &a) { return Gen(a.template normalized_power<4>()); });
            obj.def("normalized_power5",
                    [](const Derived &a) { return Gen(a.template normalized_power<5>()); });

            obj.def("normalized_power3",
                    [](const Derived &a) { return Gen(a.template normalized_power<3>()); });

        } else {
            /// Try to fit these to constant size 2,3 if possible

            auto SizeSwitch = [](const auto &fun, auto Lam) {
                int orr = fun.ORows();
                // if (orr == 2)     return Lam(fun, std::integral_constant<int, 2>());
                // else
                if (orr == 3)
                    return Lam(fun, std::integral_constant<int, 3>());
                return Lam(fun, std::integral_constant<int, -1>());
            };

            obj.def("norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(Norm<size.value>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("squared_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(SquaredNorm<size.value>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("cubed_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(NormPower<size.value, 3>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseNorm<size.value>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_squared_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseSquaredNorm<size.value>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_cubed_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseNormPower<size.value, 3>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_four_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseNormPower<size.value, 4>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(Normalized<size.value>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power2", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 2>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power3", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 3>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power4", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 4>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power5", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 5>(funt.ORows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
        }
    }

    if constexpr (OR == 1) {

        obj.def("sin", [](const Derived &a) { return GenS(a.Sin()); });
        obj.def("cos", [](const Derived &a) { return GenS(a.Cos()); });
        obj.def("tan", [](const Derived &a) { return GenS(a.Tan()); });
        obj.def("sqrt", [](const Derived &a) { return GenS(a.Sqrt()); });
        obj.def("exp", [](const Derived &a) { return GenS(a.Exp()); });
        obj.def("log", [](const Derived &e) { return GenS(CwiseLog<Derived>(e)); });
        obj.def("squared", [](const Derived &a) { return GenS(a.Square()); });
        obj.def("arcsin", [](const Derived &e) { return GenS(CwiseArcSin<Derived>(e)); });
        obj.def("arccos", [](const Derived &e) { return GenS(CwiseArcCos<Derived>(e)); });
        obj.def("arctan", [](const Derived &e) { return GenS(CwiseArcTan<Derived>(e)); });

        obj.def("sinh", [](const Derived &e) { return GenS(CwiseSinH<Derived>(e)); });
        obj.def("cosh", [](const Derived &e) { return GenS(CwiseCosH<Derived>(e)); });
        obj.def("tanh", [](const Derived &e) { return GenS(CwiseTanH<Derived>(e)); });

        obj.def("arcsinh", [](const Derived &e) { return GenS(CwiseArcSinH<Derived>(e)); });
        obj.def("arccosh", [](const Derived &e) { return GenS(CwiseArcCosH<Derived>(e)); });
        obj.def("arctanh", [](const Derived &e) { return GenS(CwiseArcTanH<Derived>(e)); });

        obj.def("__abs__", [](const Derived &e) { return GenS(CwiseAbs<Derived>(e)); });
        obj.def("sign", [](const Derived &e) { return GenS(SignFunction<Derived>(e)); });

        obj.def("pow", [](const Derived &e, double power) {
            return GenS(CwisePow<Derived>(e, power));
        });

        if constexpr (!std::is_same<PYClass, nb::module_>::value) {
            obj.def(
                "__pow__",
                [](const Derived &a, double b) { return GenS(CwisePow<Derived>(a, b)); },
                nb::is_operator());
            obj.def(
                "__pow__",
                [](const Derived &a, int b) {
                    if (b == 1)
                        return GenS(a);
                    else if (b == 2)
                        return GenS(a.Square());
                    return GenS(CwisePow<Derived>(a, b));
                },
                nb::is_operator());
        }
    }
}

template <class Derived, class PYClass>
void FunctionIndexingBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    obj.def("padded_lower",
            [](const Derived &a, int lpad) { return Gen(a.padded_lower(lpad)); });
    obj.def("padded_upper",
            [](const Derived &a, int upad) { return Gen(a.padded_upper(upad)); });
    obj.def("padded",
            [](const Derived &a, int upad, int lpad) { return Gen(a.padded(upad, lpad)); });

    constexpr bool is_seg = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;

    using SegRetType = typename std::conditional<is_seg, SEG, GenericFunction<-1, -1>>::type;
    using ElemRetType = typename std::conditional<is_seg, ELEM, GenericFunction<-1, 1>>::type;

    obj.def("segment", [](const Derived &a, int start, int size) {
        return SegRetType(a.segment(start, size));
    });
    obj.def("head", [](const Derived &a, int size) { return SegRetType(a.head(size)); });
    obj.def("tail", [](const Derived &a, int size) { return SegRetType(a.tail(size)); });

    obj.def("coeff", [](const Derived &a, int elem) { return ElemRetType(a.coeff(elem)); });
    obj.def(
        "__getitem__", [](const Derived &a, int elem) { return ElemRetType(a.coeff(elem)); },
        nb::is_operator());
    obj.def(
        "__getitem__",
        [](const Derived &a, const nb::slice &slice) {
            auto [start, stop, step, slicelength] = slice.compute(a.ORows());

            if (step != 1) {
                throw std::invalid_argument("Non continous slices not supported");
            }
            if (start >= a.ORows()) {
                throw std::invalid_argument("Segment index out of bounds.");
            }
            if (start > stop) {
                throw std::invalid_argument("Backward indexing not supported.");
            }
            if (slicelength == 0) {
                throw std::invalid_argument("Slice length must be greater than 0.");
            }

            int start_ = start;
            int size_ = stop - start;

            return SegRetType(a.segment(start_, size_));
        },
        nb::is_operator());

    // if constexpr (OR != 1) {
    //     obj.def("colmatrix", [](const Derived& a, int rows, int cols) {
    //         Gen agen(a);
    //         return agen.colmatrix(rows, cols);
    //         });
    //     obj.def("rowmatrix", [](const Derived& a, int rows, int cols) {
    //         Gen agen(a);
    //         return agen.rowmatrix(rows, cols);
    //         });
    // }

    if constexpr (OR < 0 || OR > 2) {
        using Seg2RetType =
            typename std::conditional<is_seg, SEG2, GenericFunction<-1, -1>>::type;

        auto seg2 = [](const Derived &a, int start) {
            return Seg2RetType(a.template segment<2>(start));
        };
        auto head2 = [](const Derived &a) { return Seg2RetType(a.template segment<2>(0)); };
        auto tail2 = [](const Derived &a) {
            return Seg2RetType(a.template segment<2>(a.ORows() - 2));
        };

        obj.def("segment_2", seg2);
        obj.def("head_2", head2);
        obj.def("tail_2", tail2);
        obj.def("segment2", seg2);
        obj.def("head2", head2);
        obj.def("tail2", tail2);
    }
    if constexpr (OR < 0 || OR > 3) {
        using Seg3RetType =
            typename std::conditional<is_seg, SEG3, GenericFunction<-1, -1>>::type;
        auto seg3 = [](const Derived &a, int start) {
            return Seg3RetType(a.template segment<3>(start));
        };
        auto head3 = [](const Derived &a) { return Seg3RetType(a.template segment<3>(0)); };
        auto tail3 = [](const Derived &a) {
            return Seg3RetType(a.template segment<3>(a.ORows() - 3));
        };
        obj.def("segment_3", seg3);
        obj.def("head_3", head3);
        obj.def("tail_3", tail3);
        obj.def("segment3", seg3);
        obj.def("head3", head3);
        obj.def("tail3", tail3);
    }
}

template <class Derived, class PYClass>
void BinaryMathBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    constexpr bool is_seg = std::is_same<Derived, SEG>::value;
    constexpr bool is_seg2 = std::is_same<Derived, SEG2>::value;
    constexpr bool is_seg3 = std::is_same<Derived, SEG3>::value;

    constexpr bool is_elem = std::is_same<Derived, ELEM>::value;
    constexpr bool is_gen = std::is_same<Derived, Gen>::value;
    constexpr bool is_gens = std::is_same<Derived, GenS>::value;

    if constexpr (OR == 3 || OR == -1) {

        if constexpr (!is_seg)
            obj.def("cross", [](const Derived &seg1, const SEG &seg2) {
                return Gen(crossProduct(seg1, seg2));
            });

        if constexpr (!is_seg3)
            obj.def("cross", [](const Derived &seg1, const SEG3 &seg2) {
                return Gen(crossProduct(seg1, seg2));
            });
        if constexpr (!is_gen)
            obj.def("cross", [](const Derived &seg1, const Gen &seg2) {
                return Gen(crossProduct(seg1, seg2));
            });

        obj.def("cross", [](const Derived &seg1, const Vector3<double> &seg2) {
            return Gen(crossProduct(seg1, Constant<-1, 3>(seg1.IRows(), seg2)));
        });

        obj.def("cross", [](const Derived &seg1, const Derived &seg2) {
            return Gen(crossProduct(seg1, seg2));
        });
    }

    if constexpr (OR != 1) {
        if constexpr (!is_seg)
            obj.def("dot", [](const Derived &seg1, const SEG &seg2) {
                return GenS(dotProduct(seg1, seg2));
            });

        if constexpr (!is_gen)
            obj.def("dot", [](const Derived &seg1, const Gen &seg2) {
                return GenS(dotProduct(seg1, seg2));
            });

        ///////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        if constexpr (!is_seg)
            obj.def("cwiseProduct", [](const Derived &seg1, const SEG &seg2) {
                return Gen(CwiseFunctionProduct<Derived, SEG>(seg1, seg2));
            });
        if constexpr (!is_gen)
            obj.def("cwiseProduct", [](const Derived &seg1, const Gen &seg2) {
                return Gen(CwiseFunctionProduct<Derived, Gen>(seg1, seg2));
            });

        ///////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////

        if constexpr (!is_seg)
            obj.def("cwiseQuotient", [](const Derived &seg1, const SEG &seg2) {
                return Gen(cwiseQuotient(seg1, seg2));
            });
        if constexpr (!is_gen)
            obj.def("cwiseQuotient", [](const Derived &seg1, const Gen &seg2) {
                return Gen(cwiseQuotient(seg1, seg2));
            });
    }

    obj.def("dot", [](const Derived &seg1, const Derived &seg2) {
        return GenS(dotProduct(seg1, seg2));
    });
    obj.def("cwiseProduct", [](const Derived &seg1, const Derived &seg2) {
        return BinGen(CwiseFunctionProduct<Derived, Derived>(seg1, seg2));
    });

    obj.def("cwiseQuotient", [](const Derived &seg1, const Derived &seg2) {
        return BinGen(cwiseQuotient(seg1, seg2));
    });

    //////////////////////////////////////
    obj.def("dot", [](const Derived &seg1, const Eigen::VectorXd &seg2) {
        return GenS(dotProduct(seg1, Constant<-1, Derived::ORC>(seg1.IRows(), seg2)));
    });

    obj.def("cwiseProduct", [](const Derived &seg1, const Eigen::VectorXd &seg2) {
        return BinGen(RowScaled<Derived>(seg1, seg2));
    });

    obj.def("cwiseQuotient", [](const Derived &seg1, const Eigen::VectorXd &seg2) {
        Eigen::VectorXd seg2i = seg2.cwiseInverse();
        return BinGen(RowScaled<Derived>(seg1, seg2i));
    });
}

template <class Derived, class PYClass>
void BinaryOperatorsBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using ARGS = Arguments<-1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    constexpr bool is_seg = std::is_same<Derived, SEG>::value;
    constexpr bool is_seg2 = std::is_same<Derived, SEG2>::value;
    constexpr bool is_seg3 = std::is_same<Derived, SEG3>::value;

    constexpr bool is_elem = std::is_same<Derived, ELEM>::value;
    constexpr bool is_gen = std::is_same<Derived, Gen>::value;
    constexpr bool is_gens = std::is_same<Derived, GenS>::value;

    constexpr bool is_arglike = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;

    if constexpr (!is_arglike) {

        obj.def("eval", [](const Derived &a, const ELEM &b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived &a, const SEG &b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived &a, const SEG2 &b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived &a, const SEG3 &b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived &a, const Gen &b) { return BinGen(a.eval(b)); });

        /////////////////////////////////////////////////////
        obj.def(
            "__call__", [](const Derived &a, const Gen &b) { return BinGen(a.eval(b)); },
            nb::is_operator());
        obj.def(
            "__call__", [](const Derived &a, const ELEM &b) { return BinGen(a.eval(b)); },
            nb::is_operator());
        obj.def("__call__", [](const Derived &a, const SEG &b) { return BinGen(a.eval(b)); });
        obj.def(
            "__call__", [](const Derived &a, const SEG2 &b) { return BinGen(a.eval(b)); },
            nb::is_operator());
        obj.def(
            "__call__", [](const Derived &a, const SEG3 &b) { return BinGen(a.eval(b)); },
            nb::is_operator());
        ///////////////////////////////////////////////////////////////
        obj.def("eval", [](const Derived &a, int ir, Eigen::VectorXi v) {
            return BinGen(ParsedInput<Derived, -1, Derived::ORC>(a, v, ir));
        });
    }

    obj.def(
        "apply",
        [](const Derived &a, const Gen &b) {
            // return Gen(b.eval(a));
            return Gen(NestedFunction<Derived, Gen>(a, b));
        },
        nb::is_operator());
    obj.def(
        "apply",
        [](const Derived &a, const GenS &b) {
            // return GenS(b.eval(a));
            return Gen(NestedFunction<Derived, GenS>(a, b));
        },
        nb::is_operator());

    obj.def(
        "__add__", [](const Derived &a, const Derived &b) { return BinGen(a + b); },
        nb::is_operator());
    obj.def(
        "__sub__", [](const Derived &a, const Derived &b) { return BinGen(a - b); },
        nb::is_operator());

    obj.def(
        "__mul__", [](const Derived &a, const ELEM &b) { return BinGen(a * b); },
        nb::is_operator());
    obj.def(
        "__mul__", [](const Derived &a, const GenS &b) { return BinGen(a * b); },
        nb::is_operator());

    obj.def(
        "__truediv__", [](const Derived &a, const ELEM &b) { return BinGen(a / b); },
        nb::is_operator());

    obj.def(
        "__truediv__", [](const Derived &a, const GenS &b) { return BinGen(a / b); },
        nb::is_operator());

    if constexpr (is_seg || is_elem || is_seg2 || is_seg3) { //
        obj.def(
            "__add__",
            [](const Derived &a, const BinGen &b) {
                return BinGen(TwoFunctionSum<Derived, BinGen>(a, b));
            },
            nb::is_operator());
        obj.def(
            "__sub__",
            [](const Derived &a, const BinGen &b) {
                return BinGen(FunctionDifference<Derived, BinGen>(a, b));
            },
            nb::is_operator());
    }

    if constexpr (OR != 1) {

        obj.def(
            "__rmul__", [](const Derived &a, const ELEM &b) { return BinGen(a * b); },
            nb::is_operator());

        obj.def(
            "__rmul__", [](const Derived &a, const GenS &b) { return BinGen(a * b); },
            nb::is_operator());
    }

    ////////////////////////////////////////////////////////////////
}

template <class Derived, class PYClass>
void ConditionalOperatorsBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    constexpr int IR = Derived::IRC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using ELEM = Segment<-1, 1, -1>;
    using GenCon = GenericConditional<IR>;

    if constexpr (OR == 1) {
        obj.def(
            "__lt__", [](const Derived &a, double b) { return GenCon(a < b); },
            nb::is_operator());
        obj.def(
            "__gt__", [](const Derived &a, double b) { return GenCon(a > b); },
            nb::is_operator());
        obj.def(
            "__rlt__", [](const Derived &a, double b) { return GenCon(a > b); },
            nb::is_operator());
        obj.def(
            "__rgt__", [](const Derived &a, double b) { return GenCon(a < b); },
            nb::is_operator());
        obj.def(
            "__le__", [](const Derived &a, double b) { return GenCon(a <= b); },
            nb::is_operator());
        obj.def(
            "__ge__", [](const Derived &a, double b) { return GenCon(a >= b); },
            nb::is_operator());

        obj.def(
            "__lt__", [](const Derived &a, const Derived &b) { return GenCon(a < b); },
            nb::is_operator());
        obj.def(
            "__gt__", [](const Derived &a, const Derived &b) { return GenCon(a > b); },
            nb::is_operator());
        obj.def(
            "__le__", [](const Derived &a, const Derived &b) { return GenCon(a <= b); },
            nb::is_operator());
        obj.def(
            "__ge__", [](const Derived &a, const Derived &b) { return GenCon(a >= b); },
            nb::is_operator());

        if constexpr (std::is_same<Derived, ELEM>::value) {
            obj.def(
                "__lt__", [](const Derived &a, const GenS &b) { return GenCon(a < b); },
                nb::is_operator());
            obj.def(
                "__gt__", [](const Derived &a, const GenS &b) { return GenCon(a > b); },
                nb::is_operator());
            obj.def(
                "__le__", [](const Derived &a, const GenS &b) { return GenCon(a <= b); },
                nb::is_operator());
            obj.def(
                "__ge__", [](const Derived &a, const GenS &b) { return GenCon(a >= b); },
                nb::is_operator());
        }
    }
}

} // namespace Tycho::Bind

#endif // TYCHO_PYTHON_BINDINGS
