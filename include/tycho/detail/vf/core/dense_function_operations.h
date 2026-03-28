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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
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
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/crtp_base.h"

namespace tycho::vf {

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
            right_jacobian_product_impl(
                target_ref.template middleCols<Size1>(Start1, Size1), left,
                right_ref.template middleCols<Size1>(Start1, Size1), assign, aliased);
        });
}

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
                                                  right_ref.middleCols(start, size), assign,
                                                  aliased);

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

            accumulate_impl(target_ref.middleCols(start, size),
                                   right_ref.middleCols(start, size), assign);
        }
    }
}

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

            accumulate_impl(target_ref.middleCols(start, size),
                                   right_ref.middleCols(start, size), assign);
        }
    }
}

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
