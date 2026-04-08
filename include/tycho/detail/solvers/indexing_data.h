// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// This file implements the struct SolverIndexingData which holds all meta data
// necessary for an asset vector function to be used as a constraint or objective inside of psiopt.
// It is coupled with a function by the interface classes ConstraintFunction and ObjectiveFunction.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/functional_flags.h"
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

namespace tycho::solvers {

struct SolverIndexingData {
    using MatrixXi = Eigen::MatrixXi;
    using VectorXi = Eigen::VectorXi;

    int input_size_ = 0;
    int output_size_ = 0;
    int num_funcappl_ = 0;

    bool vindex_init_ = false;
    bool cindex_init_ = false;
    bool unique_constraints_ = true;

    /// <summary>
    /// Matrix whose columns contains the ordered indices of the variables
    /// that will be forwarded to a constraint or objective function
    /// </summary>
    MatrixXi v_index_;

    /// <summary>
    /// Matrix whose columns constraint the ordered constraint output indices for the corresponding
    /// column in v_index_. This matrix is empy for objective functions.
    /// </summary>
    MatrixXi c_index_;

    /// <summary>
    /// Each element indicates whether the corresponding indices in v_index_ are sorted
    /// and contigous (ie: 10,11,12...)
    /// </summary>
    std::vector<vf::ParsedIOFlags> v_index_continuity_;

    /// <summary>
    /// Each element indicates whether the corresponding indices in c_index_ are sorted
    /// and contigous (ie: 10,11,12...)
    /// </summary>
    std::vector<vf::ParsedIOFlags> c_index_continuity_;

    /// <summary>
    /// Holds the index of the start of the region of memory allocated by Psiopt to sum the
    /// constraint output of the ith call of a constraint function.
    /// </summary>
    VectorXi inner_constraint_starts_;

    /// <summary>
    /// Holds the index of the start of the region of memory allocated by Psiopt to sum the gradient
    /// output of the ith call of a constraint or objective function.
    /// </summary>
    VectorXi inner_gradient_starts_;

    /// <summary>
    /// Holds the index of the start of the region of memory allocated by Psiopt to store the
    /// locations where the derivatives of the ith call to a function should be summed into the
    /// global KKT matrix.
    /// </summary>
    VectorXi inner_kkt_starts_;

    SolverIndexingData() {}
    SolverIndexingData(int irr, int orr, const MatrixXi &vindex, const MatrixXi &cindex)
        : input_size_(irr), output_size_(orr) {
        this->set_v_index_c_index(vindex, cindex);
    }

    SolverIndexingData(int irr, const MatrixXi &vindex) : input_size_(irr), output_size_(1) {
        this->set_v_index(vindex);
    }

    void get_gradient_space(EigenRef<VectorXi> GXrows, int &freeloc) {
        this->inner_gradient_starts_.resize(this->num_appl());
        for (int V = 0; V < this->num_appl(); V++) {
            this->inner_gradient_starts_[V] = freeloc;
            for (int i = 0; i < this->input_size_; i++) {
                GXrows[freeloc] = this->v_loc(i, V);
                freeloc++;
            }
        }
    }
    void get_constraint_space(EigenRef<VectorXi> FXrows, int &freeloc) {
        this->inner_constraint_starts_.resize(this->num_appl());
        for (int V = 0; V < this->num_appl(); V++) {
            this->inner_constraint_starts_[V] = freeloc;
            for (int j = 0; j < this->output_size_; j++) {
                FXrows[freeloc] = this->c_loc(j, V);
                freeloc++;
            }
        }
    }

    std::vector<SolverIndexingData> thread_split(int Threads) const {
        std::vector<SolverIndexingData> split;
        split.reserve(Threads);

        int cols = this->num_funcappl_;
        int colpThr = cols / Threads;
        int rempThr = cols % Threads;

        VectorXi perThr = VectorXi::Constant(Threads, colpThr);
        perThr.head(rempThr) += VectorXi::Constant(rempThr, 1);
        int start = 0;
        int range;
        if (colpThr > 0)
            range = Threads;
        else
            range = rempThr;
        for (int i = 0; i < range; i++) {
            if (this->cindex_init_) {
                split.emplace_back(SolverIndexingData(this->input_size_, this->output_size_,
                                                      this->v_index_.middleCols(start, perThr[i]),
                                                      this->c_index_.middleCols(start, perThr[i])));
            } else {
                split.emplace_back(SolverIndexingData(this->input_size_,
                                                      this->v_index_.middleCols(start, perThr[i])));
            }
            split.back().unique_constraints_ = this->unique_constraints_;
            start += perThr[i];
        }
        return split;
    }

    void set_v_index(const MatrixXi &vt) {
        this->v_index_ = vt;
        this->vindex_init_ = true;
        this->num_funcappl_ = this->v_index_.cols();
        this->v_index_continuity_.resize(this->v_index_.cols());
        for (int i = 0; i < this->v_index_.cols(); i++) {
            this->v_index_continuity_[i] = this->check_continuity(this->v_index_.col(i));
        }
    }
    void set_c_index(const MatrixXi &ct) {
        this->c_index_ = ct;
        this->c_index_continuity_.resize(this->c_index_.cols());
        this->cindex_init_ = true;

        for (int i = 0; i < this->c_index_.cols(); i++) {
            this->c_index_continuity_[i] = this->check_continuity(this->c_index_.col(i));
        }
    }
    const MatrixXi &get_v_index() const { return this->v_index_; }
    const MatrixXi &get_c_index() const { return this->c_index_; }
    void set_v_index_c_index(const MatrixXi &vt, const MatrixXi &ct) {
        this->set_v_index(vt);
        this->set_c_index(ct);
    }
    void push_v_index(int push) {
        for (int i = 0; i < this->v_index_.size(); i++) {
            this->v_index_(i) += push;
        }
    }
    void push_c_index(int push) {
        for (int i = 0; i < c_index_.size(); i++) {
            this->c_index_(i) += push;
        }
    }
    void push_v_index_c_index(int vpush, int cpush) {
        this->push_v_index(vpush);
        this->push_c_index(cpush);
    }

    inline int num_appl() const { return this->num_funcappl_; }
    inline int c_loc(int loc, int col) const { return this->c_index_(loc, col); }
    inline int v_loc(int loc, int col) const { return this->v_index_(loc, col); }

    static vf::ParsedIOFlags check_continuity(const Eigen::VectorXi &ix) {
        int s = 0;
        for (int i = 0; i < (ix.size() - 1); i++) {
            s = ix[i + 1] - ix[i] - 1;
            if (s != 0)
                return vf::ParsedIOFlags::NotContiguous;
        }
        return vf::ParsedIOFlags::Contiguous;
    }
};

} // namespace tycho::solvers
