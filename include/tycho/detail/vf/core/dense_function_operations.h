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

#include "tycho/detail/vf/core/assignment_types.h"
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::vf {

/// @internal
/// @brief Computes the matrix product `left * right` into `target` per @p Assignment.
/// @tparam Target  Eigen destination expression type.
/// @tparam Left  Eigen left-operand expression type.
/// @tparam Right  Eigen right-operand expression type.
/// @tparam Assignment  Assignment-tag type selecting `=`, `+=`, `-=`, or scaled.
/// @tparam Aliased  Whether the operands may alias `target`.
/// @param target_  Destination matrix.
/// @param left  Left operand.
/// @param right  Right operand.
/// @param assign  Assignment-tag instance (carries scale for scaled tags).
/// @param aliased  Compile-time aliasing flag.
/// @endinternal
template <class Target, class Left, class Right, class Assignment, bool Aliased>
void right_jacobian_product_impl(const Eigen::MatrixBase<Target> &target_,
                                 const Eigen::EigenBase<Left> &left,
                                 const Eigen::EigenBase<Right> &right, Assignment assign,
                                 std::bool_constant<Aliased> aliased) {
    Eigen::MatrixBase<Target> &target = target_.const_cast_derived();
    typedef typename Target::Scalar Scalar;
    if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
        if constexpr (Aliased) {
            target = left.derived() * right.derived();
        } else {
            target.noalias() = left.derived() * right.derived();
        }
    } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
        if constexpr (Aliased) {
            target += left.derived() * right.derived();
        } else {
            target.noalias() += left.derived() * right.derived();
        }
    } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
        if constexpr (Aliased) {
            target -= left.derived() * right.derived();
        } else {
            target.noalias() -= left.derived() * right.derived();
        }
    } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
        if constexpr (Aliased) {
            target = assign.value * left.derived() * right.derived();
        } else {
            target.noalias() = assign.value * left.derived() * right.derived();
        }
    } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
        if constexpr (Aliased) {
            target += assign.value * left.derived() * right.derived();
        } else {
            target.noalias() += assign.value * left.derived() * right.derived();
        }
    } else {
        std::cout << "right_jacobian_product has not been implemented for: "

                  << std::endl;
    }
}

/// @internal
/// @brief Right Jacobian product restricted to a compile-time sub-domain layout.
/// @tparam INPUT_DOMAIN  Compile-time domain type providing `sub_domains`.
/// @tparam Target  Eigen destination expression type.
/// @tparam Left  Eigen left-operand expression type.
/// @tparam Right  Eigen right-operand expression type.
/// @tparam Assignment  Assignment-tag type.
/// @tparam Aliased  Whether the operands may alias `target`.
/// @param sub_domains  Compile-time sub-domain layout instance.
/// @param target_  Destination matrix.
/// @param left  Left operand.
/// @param right  Right operand.
/// @param assign  Assignment-tag instance.
/// @param aliased  Compile-time aliasing flag.
/// @endinternal
template <class INPUT_DOMAIN, class Target, class Left, class Right, class Assignment, bool Aliased>
void right_jacobian_product_constant_impl(const INPUT_DOMAIN &sub_domains,
                                          const Eigen::MatrixBase<Target> &target_,
                                          const Eigen::EigenBase<Left> &left,
                                          const Eigen::EigenBase<Right> &right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) {
    constexpr int sds = INPUT_DOMAIN::sub_domains.size();

    const Eigen::MatrixBase<Right> &right_ref(right.derived());
    Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

    tycho::utils::constexpr_for_loop(
        std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
            constexpr int Start1 = INPUT_DOMAIN::sub_domains[i.value][0];
            constexpr int Size1 = INPUT_DOMAIN::sub_domains[i.value][1];
            right_jacobian_product_impl(target_ref.template middleCols<Size1>(Start1, Size1), left,
                                        right_ref.template middleCols<Size1>(Start1, Size1), assign,
                                        aliased);
        });
}

/// @internal
/// @brief Right Jacobian product restricted to a runtime sub-domain layout.
/// @tparam Target  Eigen destination expression type.
/// @tparam Left  Eigen left-operand expression type.
/// @tparam Right  Eigen right-operand expression type.
/// @tparam Assignment  Assignment-tag type.
/// @tparam Aliased  Whether the operands may alias `target`.
/// @param sub_domains  Runtime sub-domain layout (`{start, size}` columns).
/// @param target_  Destination matrix.
/// @param left  Left operand.
/// @param right  Right operand.
/// @param assign  Assignment-tag instance.
/// @param aliased  Compile-time aliasing flag.
/// @endinternal
template <class Target, class Left, class Right, class Assignment, bool Aliased>
void right_jacobian_product_dynamic_impl(const DomainMatrix &sub_domains,
                                         const Eigen::MatrixBase<Target> &target_,
                                         const Eigen::EigenBase<Left> &left,
                                         const Eigen::EigenBase<Right> &right, Assignment assign,
                                         std::bool_constant<Aliased> aliased) {
    const int sds = sub_domains.cols();

    if (sds == 0) {
        // right_jacobian_product_impl(target_, left, right, assign, aliased);
    } else {
        const Eigen::MatrixBase<Right> &right_ref(right.derived());
        Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

        for (int i = 0; i < sds; i++) {
            int start = sub_domains(0, i);
            int size = sub_domains(1, i);

            right_jacobian_product_impl(target_ref.middleCols(start, size), left,
                                        right_ref.middleCols(start, size), assign, aliased);
        }
    }
}

///////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Computes the symmetric product `rightᵀ * left * right` into `target`.
/// @tparam Target  Eigen destination expression type.
/// @tparam Left  Eigen middle-operand expression type.
/// @tparam Right  Eigen outer-operand expression type.
/// @tparam Assignment  Assignment-tag type.
/// @tparam Aliased  Whether the operands may alias `target`.
/// @param target_  Destination matrix.
/// @param left  Middle (symmetric) operand.
/// @param right  Outer operand applied on both sides.
/// @param assign  Assignment-tag instance.
/// @param aliased  Compile-time aliasing flag.
/// @endinternal
template <class Target, class Left, class Right, class Assignment, bool Aliased>
void symetric_jacobian_product_impl(const Eigen::MatrixBase<Target> &target_,
                                    const Eigen::EigenBase<Left> &left,
                                    const Eigen::EigenBase<Right> &right, Assignment assign,
                                    std::bool_constant<Aliased> aliased) {
    Eigen::MatrixBase<Target> &target = target_.const_cast_derived();
    typedef typename Target::Scalar Scalar;
    if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
        if constexpr (Aliased) {
            target = right.derived().transpose() * left.derived() * right.derived();
        } else {
            target.noalias() = right.derived().transpose() * left.derived() * right.derived();
        }
    } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
        if constexpr (Aliased) {
            target += right.derived().transpose() * left.derived() * right.derived();
        } else {
            if constexpr (Left::MaxRowsAtCompileTime == 1 && Left::MaxColsAtCompileTime == 1) {
                target.noalias() +=
                    right.derived().transpose() * (Scalar(left.derived()[0]) * right.derived());

            } else {
                target.noalias() += right.derived().transpose() * left.derived() * right.derived();
            }
        }
    } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
        if constexpr (Aliased) {
            target -= right.derived().transpose() * left.derived() * right.derived();
        } else {
            target.noalias() -= right.derived().transpose() * left.derived() * right.derived();
        }
    } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
        if constexpr (Aliased) {
            target = assign.value * right.derived().transpose() * left.derived() * right.derived();
        } else {
            target.noalias() =
                assign.value * right.derived().transpose() * left.derived() * right.derived();
        }
    } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
        if constexpr (Aliased) {
            target += assign.value * right.derived().transpose() * left.derived() * right.derived();
        } else {
            target.noalias() +=
                assign.value * right.derived().transpose() * left.derived() * right.derived();
        }
    } else {
        std::cout << "symetric_jacobian_product has not been implemented for: "

                  << std::endl;
    }
}

/// @internal
/// @brief Symmetric Jacobian product over a compile-time sub-domain layout.
/// @tparam INPUT_DOMAIN  Compile-time domain type providing `sub_domains`.
/// @tparam Target  Eigen destination expression type.
/// @tparam Left  Eigen middle-operand expression type.
/// @tparam Right  Eigen outer-operand expression type.
/// @tparam Assignment  Assignment-tag type.
/// @tparam Aliased  Whether the operands may alias `target`.
/// @param sub_domains  Compile-time sub-domain layout instance.
/// @param target_  Destination matrix.
/// @param left  Middle (symmetric) operand.
/// @param right  Outer operand.
/// @param assign  Assignment-tag instance.
/// @param aliased  Compile-time aliasing flag.
/// @endinternal
template <class INPUT_DOMAIN, class Target, class Left, class Right, class Assignment, bool Aliased>
void symetric_jacobian_product_constant_impl(const INPUT_DOMAIN &sub_domains,
                                             const Eigen::MatrixBase<Target> &target_,
                                             const Eigen::EigenBase<Left> &left,
                                             const Eigen::EigenBase<Right> &right,
                                             Assignment assign,
                                             std::bool_constant<Aliased> aliased) {
    constexpr int sds = INPUT_DOMAIN::sub_domains.size();

    const Eigen::MatrixBase<Right> &right_ref(right.derived());
    Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

    if constexpr (sds != 1) {
        symetric_jacobian_product_impl(target_, left, right, assign, aliased);

    } else {
        tycho::utils::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                constexpr int Start1 = INPUT_DOMAIN::sub_domains[i.value][0];
                constexpr int Size1 = INPUT_DOMAIN::sub_domains[i.value][1];
                symetric_jacobian_product_impl(
                    target_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1), left,
                    right_ref.template middleCols<Size1>(Start1, Size1), assign, aliased);
            });
    }
}

/// @internal
/// @brief Symmetric Jacobian product over a runtime sub-domain layout.
/// @tparam Target  Eigen destination expression type.
/// @tparam Left  Eigen middle-operand expression type.
/// @tparam Right  Eigen outer-operand expression type.
/// @tparam Assignment  Assignment-tag type.
/// @tparam Aliased  Whether the operands may alias `target`.
/// @param sub_domains  Runtime sub-domain layout (`{start, size}` columns).
/// @param target_  Destination matrix.
/// @param left  Middle (symmetric) operand.
/// @param right  Outer operand.
/// @param assign  Assignment-tag instance.
/// @param aliased  Compile-time aliasing flag.
/// @endinternal
template <class Target, class Left, class Right, class Assignment, bool Aliased>
void symetric_jacobian_product_dynamic_impl(const DomainMatrix &sub_domains,
                                            const Eigen::MatrixBase<Target> &target_,
                                            const Eigen::EigenBase<Left> &left,
                                            const Eigen::EigenBase<Right> &right, Assignment assign,
                                            std::bool_constant<Aliased> aliased) {
    const int sds = sub_domains.cols();

    if (sds == 0 || sds > 1) {
        symetric_jacobian_product_impl(target_, left, right, assign, aliased);
    } else {
        const Eigen::MatrixBase<Right> &right_ref(right.derived());
        Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

        for (int i = 0; i < sds; i++) {
            int start = sub_domains(0, i);
            int size = sub_domains(1, i);

            symetric_jacobian_product_impl(target_ref.block(start, start, size, size), left,
                                           right_ref.middleCols(start, size), assign, aliased);

            /*for (int j = i+1; j < sds; j++) {
                int start2 = sub_domains(0, j);
                int size2 = sub_domains(1, j);

                target_ref.block(start, start, size, size) +=
                    right_ref.middleCols(start2, size2).transpose() * left.derived() *
            right_ref.middleCols(start, size);

            }*/
        }
    }
}

////////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Accumulates `right` into `target` per @p Assignment.
/// @tparam Target  Eigen destination expression type.
/// @tparam JacType  Eigen source expression type.
/// @tparam Assignment  Assignment-tag type.
/// @param target_  Destination matrix.
/// @param right  Source matrix.
/// @param assign  Assignment-tag instance.
/// @endinternal
template <class Target, class JacType, class Assignment>
void accumulate_impl(const Eigen::MatrixBase<Target> &target_,
                     const Eigen::MatrixBase<JacType> &right, Assignment assign) {
    Eigen::MatrixBase<Target> &target = target_.const_cast_derived();
    typedef typename Target::Scalar Scalar;

    if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
        target = right;
    } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
        target += right;
    } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
        target -= right;
    } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
        target = assign.value * right.derived();
    } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
        target += assign.value * right.derived();
    }
}

/// @internal
/// @brief Accumulates a matrix column-block-wise over a runtime sub-domain layout.
/// @tparam Target  Eigen destination expression type.
/// @tparam JacType  Eigen source expression type.
/// @tparam Assignment  Assignment-tag type.
/// @param sub_domains  Runtime sub-domain layout (`{start, size}` columns).
/// @param target_  Destination matrix.
/// @param right  Source matrix.
/// @param assign  Assignment-tag instance.
/// @endinternal
template <class Target, class JacType, class Assignment>
void accumulate_matrix_dynamic_domain_impl(const DomainMatrix &sub_domains,
                                           const Eigen::MatrixBase<Target> &target_,
                                           const Eigen::MatrixBase<JacType> &right,
                                           Assignment assign) {
    int sds = sub_domains.cols();
    if (sds == 0) {
        // accumulate_impl(target_, right, assign);
    } else {
        const Eigen::MatrixBase<JacType> &right_ref(right.derived());
        Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

        for (int i = 0; i < sds; i++) {
            int start = sub_domains(0, i);
            int size = sub_domains(1, i);

            accumulate_impl(target_ref.middleCols(start, size), right_ref.middleCols(start, size),
                            assign);
        }
    }
}

/// @internal
/// @brief Accumulates a symmetric matrix block-wise over a runtime sub-domain layout.
/// @tparam Target  Eigen destination expression type.
/// @tparam JacType  Eigen source expression type.
/// @tparam Assignment  Assignment-tag type.
/// @param sub_domains  Runtime sub-domain layout (`{start, size}` columns).
/// @param target_  Destination matrix.
/// @param right  Source matrix.
/// @param assign  Assignment-tag instance.
/// @endinternal
template <class Target, class JacType, class Assignment>
void accumulate_symetric_matrix_dynamic_domain_impl(const DomainMatrix &sub_domains,
                                                    const Eigen::MatrixBase<Target> &target_,
                                                    const Eigen::MatrixBase<JacType> &right,
                                                    Assignment assign) {
    int sds = sub_domains.cols();
    const Eigen::MatrixBase<JacType> &right_ref(right.derived());
    Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

    if (sds == 0) {
        // accumulate_impl(target_, right, assign);
    } else if (sds == 1) {

        int Start1 = sub_domains(0, 0);
        int Size1 = sub_domains(1, 0);
        accumulate_impl(target_ref.block(Start1, Start1, Size1, Size1),
                        right_ref.block(Start1, Start1, Size1, Size1), assign);
    } else if (sds == 2) {

        int Start1 = sub_domains(0, 0);
        int Size1 = sub_domains(1, 0);
        int Start2 = sub_domains(0, 1);
        int Size2 = sub_domains(1, 1);

        accumulate_impl(target_ref.block(Start1, Start1, Size1, Size1),
                        right_ref.block(Start1, Start1, Size1, Size1), assign);

        accumulate_impl(target_ref.block(Start2, Start1, Size2, Size1),
                        right_ref.block(Start2, Start1, Size2, Size1), assign);

        accumulate_impl(target_ref.block(Start1, Start2, Size1, Size2),
                        right_ref.block(Start1, Start2, Size1, Size2), assign);

        accumulate_impl(target_ref.block(Start2, Start2, Size2, Size2),
                        right_ref.block(Start2, Start2, Size2, Size2), assign);

    } else {

        for (int i = 0; i < sds; i++) {
            int start = sub_domains(0, i);
            int size = sub_domains(1, i);

            accumulate_impl(target_ref.middleCols(start, size), right_ref.middleCols(start, size),
                            assign);
        }
    }
}

/// @internal
/// @brief Accumulates a vector segment-wise over a runtime sub-domain layout.
/// @tparam Target  Eigen destination expression type.
/// @tparam JacType  Eigen source expression type.
/// @tparam Assignment  Assignment-tag type.
/// @param sub_domains  Runtime sub-domain layout (`{start, size}` columns).
/// @param target_  Destination vector.
/// @param right  Source vector.
/// @param assign  Assignment-tag instance.
/// @endinternal
template <class Target, class JacType, class Assignment>
void accumulate_vector_dynamic_domain_impl(const DomainMatrix &sub_domains,
                                           const Eigen::MatrixBase<Target> &target_,
                                           const Eigen::MatrixBase<JacType> &right,
                                           Assignment assign) {
    int sds = sub_domains.cols();
    if (sds == 0) {
        // accumulate_impl(target_, right, assign);
    } else {
        const Eigen::MatrixBase<JacType> &right_ref(right.derived());
        Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

        for (int i = 0; i < sds; i++) {
            int start = sub_domains(0, i);
            int size = sub_domains(1, i);
            accumulate_impl(target_ref.segment(start, size), right_ref.segment(start, size),
                            assign);
        }
    }
}

/// @internal
/// @brief Scales a vector in place segment-wise over a runtime sub-domain layout.
/// @tparam Target  Eigen destination expression type.
/// @tparam Scalar  Scale-factor type.
/// @param sub_domains  Runtime sub-domain layout (`{start, size}` columns).
/// @param target_  Vector to scale.
/// @param s  Scale factor.
/// @endinternal
template <class Target, class Scalar>
void scale_vector_dynamic_domain_impl(const DomainMatrix &sub_domains,
                                      const Eigen::MatrixBase<Target> &target_, Scalar s) {
    int sds = sub_domains.cols();
    Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

    if (sds == 0) {
        target_ref *= s;
    } else {
        for (int i = 0; i < sds; i++) {
            int start = sub_domains(0, i);
            int size = sub_domains(1, i);
            target_ref.segment(start, size) *= s;
        }
    }
}

/// @internal
/// @brief Scales a matrix in place column-block-wise over a runtime sub-domain layout.
/// @tparam Target  Eigen destination expression type.
/// @tparam Scalar  Scale-factor type.
/// @param sub_domains  Runtime sub-domain layout (`{start, size}` columns).
/// @param target_  Matrix to scale.
/// @param s  Scale factor.
/// @endinternal
template <class Target, class Scalar>
void scale_matrix_dynamic_domain_impl(const DomainMatrix &sub_domains,
                                      const Eigen::MatrixBase<Target> &target_, Scalar s) {
    int sds = sub_domains.cols();
    Eigen::MatrixBase<Target> &target_ref(target_.const_cast_derived());

    if (sds == 0) {
        target_ref *= s;
    } else {
        for (int i = 0; i < sds; i++) {
            int start = sub_domains(0, i);
            int size = sub_domains(1, i);
            target_ref.middleCols(start, size) *= s;
        }
    }
}

} // namespace tycho::vf
