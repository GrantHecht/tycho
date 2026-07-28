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
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
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

#include <fmt/format.h>

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

/// @brief Canonical KKT lock column for the physical slot coupling global indices
/// @p a and @p b: the smaller of the two.
///
/// The KKT sparsity routine (NonLinearProgram::analyze_sparsity) canonicalizes every
/// element to the lower triangle (col <= row) and stores its value offset under the
/// SMALLER endpoint, so both mirror orderings of a symmetric Hessian pair collapse to
/// one physical double filed under min(a, b). Any two writers of that double are only
/// serialized if they take the same mutex, so every KKT scatter site
/// (DenseFunctionBase::kkt_fill_all / kkt_fill_hess) keys its per-element lock on this
/// function, and NonLinearProgram::get_mat_space marks contested columns (and sizes
/// kkt_locks_) with this same function. Because all claimants of a slot derive their
/// lock column from this single shared keying, cross-partition agreement is structural
/// -- there is no per-site convention that can drift. Do NOT introduce a second keying
/// convention at any of those sites.
inline constexpr int kkt_canonical_lock_col(int a, int b) { return (a < b) ? a : b; }

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
    /// Where this function's outputs go in the SOLVER's variable space, when
    /// that differs from the problem's own variable space.
    ///
    /// v_index_ above is the function's INPUT map: which entries of the primal
    /// vector its arguments are gathered from. It always addresses the full
    /// problem space and is never rewritten -- a function always reads the same
    /// variables it was declared over.
    ///
    /// This is the OUTPUT map: which KKT column, and which gradient row, each of
    /// those arguments corresponds to once the solver has eliminated the
    /// variables whose bounds fix them. Entry (i, V) is the solver-space index
    /// of the same variable v_index_(i, V) names, or -1 when that variable has
    /// been eliminated and the corresponding outputs have nowhere to go.
    ///
    /// Empty on the identity path: with nothing eliminated, the solver's space
    /// IS the problem's space and v_index_ serves as both maps, so no second
    /// table is built and no copy is made. Derived state -- always regenerated
    /// from v_index_, never edited in place, so repeated configuration cannot
    /// compound.
    /// </summary>
    MatrixXi v_out_index_;

    /// True while v_out_index_ is live, i.e. the output map differs from the
    /// input map. THE flag the KKT scatters hoist their branch on: false selects
    /// the untouched original loops, and it is false for every function on every
    /// problem with no bound-fixed variables. Setup-time emitters read
    /// v_scatter_loc(), which dispatches on the same flag.
    bool v_out_reduced_ = false;

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

    /// <summary>
    /// Installs the solver-space output map. @p vout must have the same shape as
    /// v_index_; -1 entries mark eliminated variables. Regenerated wholesale by
    /// the caller from v_index_ on every configuration, so this never has to
    /// undo a previous mapping.
    /// </summary>
    void set_output_v_index(const MatrixXi &vout) {
        if (vout.rows() != this->v_index_.rows() || vout.cols() != this->v_index_.cols()) {
            throw std::invalid_argument(fmt::format(
                "SolverIndexingData::set_output_v_index: expected a {}x{} map to "
                "match the input map, got {}x{}",
                this->v_index_.rows(), this->v_index_.cols(), vout.rows(), vout.cols()));
        }
        this->v_out_index_ = vout;
        this->v_out_reduced_ = true;
    }

    /// Drops the output map, returning this function to the identity path where
    /// v_index_ is both maps.
    void clear_output_v_index() {
        this->v_out_index_.resize(0, 0);
        this->v_out_reduced_ = false;
    }

    /// Solver-space output index, valid only while v_out_reduced_ is true --
    /// which is exactly when the KKT scatters take their reduced loop.
    inline int v_out_loc(int loc, int col) const { return this->v_out_index_(loc, col); }

    /// Output index for setup-time emitters (KKT/gradient location tables),
    /// which run once per configuration and can afford the dispatch: the
    /// solver-space index when an output map is live, the problem-space index
    /// otherwise. Returns -1 for an eliminated variable.
    inline int v_scatter_loc(int loc, int col) const {
        return this->v_out_reduced_ ? this->v_out_index_(loc, col) : this->v_index_(loc, col);
    }

    void get_gradient_space(EigenRef<VectorXi> GXrows, int &freeloc) {
        this->inner_gradient_starts_.resize(this->num_appl());
        for (int V = 0; V < this->num_appl(); V++) {
            this->inner_gradient_starts_[V] = freeloc;
            for (int i = 0; i < this->input_size_; i++) {
                // -1 for an eliminated variable: the claim stays the same size
                // (the function still writes input_size_ gradient values into
                // it) but that value has no row to be summed into, and the RHS
                // fill skips it.
                GXrows[freeloc] = this->v_scatter_loc(i, V);
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
        if (Threads <= 0)
            throw std::invalid_argument(
                fmt::format("thread_split: Threads must be positive, got {}", Threads));

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
        if (vt.rows() != this->input_size_)
            throw std::invalid_argument(
                fmt::format("SolverIndexingData::set_v_index: expected {} rows (input_size_), "
                            "got {}",
                            this->input_size_, vt.rows()));
        this->v_index_ = vt;
        this->vindex_init_ = true;
        this->num_funcappl_ = this->v_index_.cols();
        this->v_index_continuity_.resize(this->v_index_.cols());
        for (int i = 0; i < this->v_index_.cols(); i++) {
            this->v_index_continuity_[i] = this->check_continuity(this->v_index_.col(i));
        }
    }
    void set_c_index(const MatrixXi &ct) {
        if (ct.rows() != this->output_size_)
            throw std::invalid_argument(
                fmt::format("SolverIndexingData::set_c_index: expected {} rows (output_size_), "
                            "got {}",
                            this->output_size_, ct.rows()));
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
