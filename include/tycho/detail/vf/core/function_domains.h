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

using utils::const_tuple_for_each;
using utils::make_array;

/// @brief Compile-time input domain of a single contiguous range.
/// @ingroup vf
///
/// Describes a VectorFunction that reads exactly one contiguous block of its
/// input vector: `Size` elements starting at offset `Start`.
/// @tparam IR  Total input row count.
/// @tparam Start  Offset of the active range.
/// @tparam Size  Length of the active range.
template <int IR, int Start, int Size> struct SingleDomain {
    static constexpr int DomainSize = IR; ///< @brief Total input row count.
    /// @brief The single contiguous sub-domain `{Start, Size}`.
    static constexpr std::array<std::array<int, 2>, 1> sub_domains = {
        std::array<int, 2>{Start, Size}};
    static constexpr int start = Start; ///< @brief Offset of the active range.
    static constexpr int size = Size;   ///< @brief Length of the active range.
};

/// @brief Compile-time input domain formed by merging several sub-domains.
/// @ingroup vf
///
/// Combines the sub-domains of `T, Ts...` into a minimal set of contiguous
/// ranges over `[0, IR)`, computed entirely at compile time.
/// @tparam IR  Total input row count.
/// @tparam T  First contributing domain type.
/// @tparam Ts  Remaining contributing domain types.
template <int IR, class T, class... Ts> struct CompositeDomain {
    static constexpr int DomainSize = IR; ///< @brief Total input row count.

    /// @brief Tests whether input index @p n is covered by any sub-domain.
    /// @param n  Input index to test.
    /// @return `true` if @p n lies within a contributing sub-domain.
    static constexpr bool contains_elem(int n) {
        std::tuple<T, Ts...> ts;
        bool t = false;
        const_tuple_for_each(ts, [&](const auto &func_i) {
            if (!t) {
                for (int i = 0; i < func_i.sub_domains.size(); i++) {
                    int start = func_i.sub_domains[i][0];
                    int size = func_i.sub_domains[i][1];
                    if (n >= start && n < (start + size))
                        t = true;
                }
            }
        });
        return t;
    }
    /// @brief Counts consecutive covered indices starting at @p start.
    /// @param start  Index at which to begin counting.
    /// @return Length of the contiguous covered run beginning at @p start.
    static constexpr int max_range(int start) {
        int maxx = 0;
        for (int i = start; i < IR; i++) {
            if (contains_elem(i))
                maxx++;
            else
                break;
        }
        return maxx;
    }
    /// @brief Per-index run lengths of contiguous coverage.
    static constexpr std::array<int, IR> dmn = {make_array<IR>(max_range)};
    /// @brief Counts the number of distinct contiguous sub-domains.
    /// @param v  Per-index run-length array (see @ref dmn).
    /// @return Number of merged contiguous ranges.
    static constexpr int count_sub_domains(std::array<int, IR> v) {
        int sr = 0;
        int i = 0;
        while (i < IR) {
            if (v[i] > 0) {
                sr += 1;
                i += v[i];
            } else {
                i += 1;
            }
        }
        return sr;
    }
    static constexpr int NumSubDomains =
        count_sub_domains(dmn); ///< @brief Number of merged sub-domains.
    /// @brief Computes the `{start, size}` of the @p sd-th merged sub-domain.
    /// @param sd  Zero-based sub-domain index.
    /// @return The `{start, size}` pair, or `{0, 0}` if out of range.
    static constexpr std::array<int, 2> calc_sub_domains(int sd) {
        std::array<int, 2> v = {0, 0};
        int sr = 0;
        int i = 0;
        while (i < IR) {
            if (dmn[i] > 0) {
                if (sr == sd) {
                    v[0] = i;
                    v[1] = dmn[i];
                    return v;
                }
                sr += 1;
                i += dmn[i];
            } else {
                i += 1;
            }
        }
        return v;
    }
    /// @brief The merged contiguous sub-domains as `{start, size}` pairs.
    static constexpr std::array<std::array<int, 2>, NumSubDomains> sub_domains = {
        make_array<NumSubDomains>(calc_sub_domains)};

    // CompositeDomain(int ir, T b, Ts... a) {}
    // CompositeDomain() = default;
};

/// @brief Dynamic-size specialization of @ref CompositeDomain.
/// @ingroup vf
///
/// Used when the total input size is not known at compile time; the
/// sub-domain layout is resolved at runtime instead.
/// @tparam T  First contributing domain type.
/// @tparam Ts  Remaining contributing domain types.
template <class T, class... Ts> struct CompositeDomain<-1, T, Ts...> {

    static constexpr int DomainSize = -1; ///< @brief Dynamic-size marker.
    /// @brief Placeholder sub-domain `{-1, -1}` for the dynamic case.
    static constexpr std::array<std::array<int, 2>, 1> sub_domains = {std::array<int, 2>{-1, -1}};
    static constexpr int start = -1; ///< @brief Dynamic-size start marker.
    static constexpr int size = -1;  ///< @brief Dynamic-size length marker.

    // CompositeDomain(int ir, T b, Ts... a) {}
    // CompositeDomain() = default;
};

/// @brief Holds and reports a VectorFunction's input domain (static size).
/// @ingroup vf
///
/// For a statically-sized function the domain is the single contiguous range
/// `[0, IR)`, so `set_input_domain` is a no-op.
/// @tparam IR  Compile-time input row count.
template <int IR> struct DomainHolder {
    /// @brief Returns the input domain as a `[0, IR)` domain matrix.
    /// @return The 2x1 domain matrix `{0, IR}`.
    [[nodiscard]] DomainMatrix input_domain() const {
        DomainMatrix dmn(2, 1);
        dmn(0, 0) = 0;
        dmn(1, 0) = IR;
        return dmn;
    }
    /// @brief No-op for statically-sized domains.
    /// @param irr  Total input row count (ignored).
    /// @param sub_domains  Candidate sub-domains (ignored).
    void set_input_domain(int irr, const std::vector<DomainMatrix> &sub_domains) {};
};

/// @brief Dynamic-size specialization of @ref DomainHolder.
/// @ingroup vf
///
/// Stores the resolved sub-domain layout at runtime and merges overlapping or
/// adjacent candidate ranges in `set_input_domain`.
template <> struct DomainHolder<-1> {
    DomainMatrix sub_domains; ///< @brief Runtime sub-domain layout (`{start, size}` columns).

    /// @brief Returns the stored runtime input domain.
    /// @return The stored sub-domain matrix.
    [[nodiscard]] DomainMatrix input_domain() const { return sub_domains; }
    /// @brief Resolves and stores the merged input sub-domain layout.
    /// @param irr  Total input row count.
    /// @param sub_domains  Candidate sub-domains to merge.
    void set_input_domain(int irr, const std::vector<DomainMatrix> &sub_domains) {
        if (sub_domains.size() == 1) {
            this->sub_domains = sub_domains[0];
            return;
        }
        Eigen::VectorXi full(irr);
        full.setZero();

        for (auto &dmn : sub_domains) {
            for (int i = 0; i < dmn.cols(); i++) {
                full.segment(dmn(0, i), dmn(1, i)).setOnes();
            }
            if (full.sum() == irr) {
                this->sub_domains.resize(2, 1);
                this->sub_domains(0, 0) = 0;
                this->sub_domains(1, 0) = irr;

                return;
            }
        }
        std::vector<std::array<int, 2>> sds;

        bool find = true;
        for (int i = 0; i < irr; i++) {
            if (full[i] == 1) {
                if (find) {
                    sds.emplace_back(std::array<int, 2>{i, 1});
                    find = false;
                } else
                    sds.back()[1]++;
            } else
                find = true;
        }

        this->sub_domains.resize(2, sds.size());
        for (int i = 0; i < sds.size(); i++) {
            this->sub_domains(0, i) = sds[i][0];
            this->sub_domains(1, i) = sds[i][1];
        }
    }
};

} // namespace tycho::vf
