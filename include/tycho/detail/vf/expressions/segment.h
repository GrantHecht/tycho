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
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/vf/common/constant.h"
#include "tycho/detail/vf/core/vector_function.h"
#include "tycho/detail/vf/derivatives/detect_diagonal.h"
#include "tycho/detail/vf/type_erasure/conditional.h"

namespace tycho::vf {

template <class Derived, int IR, int OR, int ST> struct Segment_Impl;

template <int IR_OR> struct Arguments : Segment_Impl<Arguments<IR_OR>, IR_OR, IR_OR, 0> {
    using Base = Segment_Impl<Arguments<IR_OR>, IR_OR, IR_OR, 0>;
    using Base::Base;
    Arguments(int iror) : Base(iror, iror, 0) {}
};
template <int IR, int OR, int ST> struct Segment : Segment_Impl<Segment<IR, OR, ST>, IR, OR, ST> {
    using Base = Segment_Impl<Segment<IR, OR, ST>, IR, OR, ST>;
    using Base::Base;
};

template <class T> struct Is_Segment : std::false_type {};
template <int IR, int OR, int ST> struct Is_Segment<Segment<IR, OR, ST>> : std::true_type {};

template <class T> struct Is_Arguments : std::false_type {};
template <int IR> struct Is_Arguments<Arguments<IR>> : std::true_type {};

template <class T> struct Is_ScaledSegment : std::false_type {};
template <int IR, int OR, int ST>
struct Is_ScaledSegment<Scaled<Segment<IR, OR, ST>>> : std::true_type {};
template <int IR, int OR, int ST, class VALUE>
struct Is_ScaledSegment<StaticScaled<Segment<IR, OR, ST>, VALUE>> : std::true_type {};

template <int ST> struct SegStartHolder {
    static constexpr int seg_start_ = ST;
    void set_seg_start(int st) {};
};
template <> struct SegStartHolder<-1> {
    int seg_start_ = -1;
    void set_seg_start(int st) { this->seg_start_ = st; };
};

template <class Derived, int IR, int OR, int ST>
struct Segment_Impl : VectorFunction<Derived, IR, OR>, SegStartHolder<ST> {
    using INPUT_DOMAIN = SingleDomain<IR, ST, OR>;
    using Base = VectorFunction<Derived, IR, OR>;
    // using SegStartHolder<ST>::seg_start_;
    VF_TYPE_ALIASES(Base);

    Segment_Impl() {}
    Segment_Impl(int irows, int orows, int start) {
        this->set_io_rows(irows, orows);
        this->set_seg_start(start);
        DomainMatrix dmn(2, 1);
        dmn(0, 0) = start;
        dmn(1, 0) = orows;

        this->set_input_domain(irows, {dmn});

        if (start + orows > this->input_rows() || start < 0) {
            throw std::invalid_argument("Segment/Element Index Out of Bounds");
        }
    }

    static constexpr bool is_linear_function = true;
    static constexpr bool is_vectorizable = true;

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.template segment<OR>(this->seg_start_, this->output_rows());
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        const int OROWS = this->output_rows();
        Scalar ONE = Scalar(1.0);
        fx = x.template segment<OR>(this->seg_start_, OROWS);
        jx.template middleCols<OR>(this->seg_start_, OROWS).diagonal().setConstant(ONE);
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        //  MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        const int OROWS = this->output_rows();
        Scalar ONE = Scalar(1.0);

        fx = x.template segment<OR>(this->seg_start_, OROWS);
        jx.template middleCols<OR>(this->seg_start_, OROWS).diagonal().setConstant(ONE);
        adjgrad.template segment<OR>(this->seg_start_, OROWS) = adjvars;
    }

    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_,
                                       CEigRef<Left> left, CEigRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        MatRef<Target> target = target_.const_cast_derived();
        typedef typename Target::Scalar Scalar;

        auto Impl = [&](auto &diag) {
            diag = right.derived()
                       .template middleCols<OR>(this->seg_start_, this->output_rows())
                       .diagonal();

            if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
                if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                                  decltype(left.derived())>::type>::value) {
                    target.template middleCols<OR>(this->seg_start_, this->output_rows())
                        .diagonal() = left.derived().diagonal().cwiseProduct(diag);
                } else {
                    if constexpr (Aliased)
                        target.template middleCols<OR>(this->seg_start_, this->output_rows()) =
                            left.derived() * diag.asDiagonal();
                    else
                        target.template middleCols<OR>(this->seg_start_, this->output_rows())
                            .noalias() = left.derived() * diag.asDiagonal();
                }

            } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
                if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                                  decltype(left.derived())>::type>::value) {
                    target.template middleCols<OR>(this->seg_start_, this->output_rows())
                        .diagonal() += left.derived().diagonal().cwiseProduct(diag);
                } else {
                    if constexpr (Aliased)
                        target.template middleCols<OR>(this->seg_start_, this->output_rows()) +=
                            left.derived() * diag.asDiagonal();
                    else
                        target.template middleCols<OR>(this->seg_start_, this->output_rows())
                            .noalias() += left.derived() * diag.asDiagonal();
                }

            } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {

                if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                                  decltype(left.derived())>::type>::value) {
                    target.template middleCols<OR>(this->seg_start_, this->output_rows())
                        .diagonal() -= left.derived().diagonal().cwiseProduct(diag);
                } else {
                    if constexpr (Aliased)
                        target.template middleCols<OR>(this->seg_start_, this->output_rows()) -=
                            left.derived() * diag.asDiagonal();
                    else
                        target.template middleCols<OR>(this->seg_start_, this->output_rows())
                            .noalias() -= left.derived() * diag.asDiagonal();
                }

            } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
                target.template middleCols<OR>(this->seg_start_, this->output_rows()).noalias() =
                    assign.value * left.derived() * diag.asDiagonal();
            } else if constexpr (std::is_same<Assignment,
                                              ScaledPlusEqualsAssignment<Scalar>>::value) {
                target.template middleCols<OR>(this->seg_start_, this->output_rows()).noalias() +=
                    assign.value * left.derived() * diag.asDiagonal();
            } else {
                std::cout << "right_jacobian_product has not been implemented for: " << this->name
                          << std::endl
                          << std::endl;
            }
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }

    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_,
                                          CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        MatRef<Target> target = target_.const_cast_derived();
        typedef typename Target::Scalar Scalar;

        Eigen::DiagonalMatrix<Scalar, OR> diag;
        diag.diagonal() = right.derived()
                              .template middleCols<OR>(this->seg_start_, this->output_rows())
                              .diagonal();
        diag.diagonal() = diag.diagonal().cwiseProduct(diag.diagonal());

        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() = left.derived() * diag;
        } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
            if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                              decltype(left.derived())>::type>::value) {
                target
                    .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                            this->output_rows())
                    .diagonal() += left.derived().diagonal().cwiseProduct(diag.diagonal());
            } else {
                target
                    .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                            this->output_rows())
                    .noalias() += left.derived() * diag;
            }

        } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() -= left.derived() * diag;
        } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() = assign.value * left.derived() * diag;
        } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() += assign.value * left.derived() * diag;
        } else {
            std::cout << "symetric_jacobian_product has not been implemented for: " << this->name
                      << std::endl
                      << std::endl;
        }
    }

    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_,
                                    CEigRef<JacType> right, Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
            target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() =
                right.derived()
                    .template middleCols<OR>(this->seg_start_, this->output_rows())
                    .diagonal();
        } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
            target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() +=
                right.derived()
                    .template middleCols<OR>(this->seg_start_, this->output_rows())
                    .diagonal();
        } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
            target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() -=
                right.derived()
                    .template middleCols<OR>(this->seg_start_, this->output_rows())
                    .diagonal();
        } else {
        }
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_,
                                    CEigRef<JacType> right, Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
            target.template segment<OR>(this->seg_start_, this->output_rows()) =
                right.derived().template segment<OR>(this->seg_start_, this->output_rows());
        } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
            target.template segment<OR>(this->seg_start_, this->output_rows()) +=
                right.derived().template segment<OR>(this->seg_start_, this->output_rows());
        } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
            target.template segment<OR>(this->seg_start_, this->output_rows()) -=
                right.derived().template segment<OR>(this->seg_start_, this->output_rows());
        } else {
        }
    }
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() *= s;
    }
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        target.template segment<OR>(this->seg_start_, this->output_rows()) *= s;
    }
    template <class Func, int FuncIRC>
    decltype(auto) rearged(const DenseFunctionBase<Func, FuncIRC, IR> &f) const {
        return Base::template EVALOP<Func>::make_nested(this->derived(), f.derived());
    }
};

} // namespace tycho::vf
