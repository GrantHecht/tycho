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
#include "tycho/detail/utils/crtp_base.h"
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

template <int IR, int Start, int Size> struct SingleDomain {
    static const int DomainSize = IR;
    static constexpr std::array<std::array<int, 2>, 1> sub_domains = {
        std::array<int, 2>{Start, Size}};
    static const int start = Start;
    static const int size = Size;
};

template <int IR, class T, class... Ts> struct CompositeDomain {
    static const int DomainSize = IR;

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
    static constexpr std::array<int, IR> dmn = {make_array<IR>(max_range)};
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
    static const int NumSubDomains = count_sub_domains(dmn);
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
    static constexpr std::array<std::array<int, 2>, NumSubDomains> sub_domains = {
        make_array<NumSubDomains>(calc_sub_domains)};

    // CompositeDomain(int ir, T b, Ts... a) {}
    // CompositeDomain() = default;
};

template <class T, class... Ts> struct CompositeDomain<-1, T, Ts...> {

    static const int DomainSize = -1;
    static constexpr std::array<std::array<int, 2>, 1> sub_domains = {std::array<int, 2>{-1, -1}};
    static const int start = -1;
    static const int size = -1;

    // CompositeDomain(int ir, T b, Ts... a) {}
    // CompositeDomain() = default;
};

template <int IR> struct DomainHolder {
    [[nodiscard]] DomainMatrix input_domain() const {
        DomainMatrix dmn(2, 1);
        dmn(0, 0) = 0;
        dmn(1, 0) = IR;
        return dmn;
    }
    void set_input_domain(int irr, const std::vector<DomainMatrix> &sub_domains) {};
};

template <> struct DomainHolder<-1> {
    DomainMatrix sub_domains;

    [[nodiscard]] DomainMatrix input_domain() const { return sub_domains; }
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
