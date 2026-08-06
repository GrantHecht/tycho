// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// The internal equality row the MakeConstraint fixed-variable treatment installs:
// x[index] - value = 0, one per variable whose declared bounds are equal.
//
// The row used to be built as a VectorFunction expression, which made the only
// caller -- NonLinearProgram, the heart of the solver -- depend on the whole
// expression machinery for one line of arithmetic. It is written here as a
// plain solver-side constraint instead: one input, one output, a residual that
// subtracts a constant, and a Jacobian that is the constant 1. Everything the
// solver asks a constraint for is spelled out below, and each method mirrors
// what the general VectorFunction path did for a function of this shape, so the
// row occupies exactly the same KKT slots and writes exactly the same values it
// did before.
// =============================================================================

#pragma once

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <fmt/format.h>

#include "tycho/detail/solvers/constraint_function.h"
#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/solvers/threading_flags.h"
#include "tycho/detail/typedefs/eigen_types.h"

namespace tycho::solvers {

/// <summary>
/// The residual x - value as a solver constraint: one variable in, one
/// constraint row out, per application.
///
/// The variable it reads and the row it writes are not stored here -- they live
/// in the SolverIndexingData the solver hands to every method, exactly as they
/// do for a user's own constraint, which is what lets one instance serve every
/// application in a partition and lets thread_split slice it like any other.
///
/// Structurally this is a linear function: its Hessian is identically zero, so
/// it claims no Hessian slot in the KKT matrix and its Jacobian is the single
/// element 1.0 in the column of the variable it pins. The general
/// VectorFunction path reached the same two conclusions from
/// `is_linear_function` and a dense 1x1 Jacobian; here they are simply the
/// shape of the code.
/// </summary>
struct FixedVariableRow {
    /// The value the variable is pinned to.
    double value_ = 0.0;

    FixedVariableRow() = default;
    explicit FixedVariableRow(double value) : value_(value) {}

    // ---- Sizing / identity ----
    std::string name() const { return "FixedVariableRow"; }
    int input_rows() const { return 1; }
    int output_rows() const { return 1; }
    bool thread_safe() const { return true; }

    // ---- Constraint evaluation ----
    void constraints(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {
        for (int V = 0; V < data.num_appl(); V++) {
            FX[data.inner_constraint_starts_[V]] = this->residual(X, data, V);
        }
    }

    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {
        for (int V = 0; V < data.num_appl(); V++) {
            FX[data.inner_constraint_starts_[V]] = this->residual(X, data, V);
            // The adjoint gradient is J^T * l, and J is 1.
            AGX[data.inner_gradient_starts_[V]] = L[data.c_loc(0, V)];
        }
    }

    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              EigenRef<Eigen::VectorXi> KKTLocations,
                              EigenRef<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const {
        for (int V = 0; V < data.num_appl(); V++) {
            FX[data.inner_constraint_starts_[V]] = this->residual(X, data, V);
            this->scatter_jacobian(V, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
    }

    void constraints_jacobian_adjointgradient(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        for (int V = 0; V < data.num_appl(); V++) {
            FX[data.inner_constraint_starts_[V]] = this->residual(X, data, V);
            AGX[data.inner_gradient_starts_[V]] = L[data.c_loc(0, V)];
            this->scatter_jacobian(V, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
    }

    /// The Hessian of this row is zero, so the second-derivative pass differs
    /// from the first only in also writing the adjoint gradient: there is no
    /// Hessian slot to claim and therefore none to fill.
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        this->constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes,
                                                   KKTLocks, data);
    }

    // ---- KKT structure ----

    /// One Jacobian element per application when the caller wants Jacobian
    /// space, and never a Hessian element.
    int num_kkt_elements(bool dojac, bool dohess) const { return dojac ? 1 : 0; }

    /// Claims this row's KKT slots, in the same order and under the same rules
    /// the general path claims them: one slot per application, recorded at the
    /// constraint row (offset by @p conoffset) and the variable's column. An
    /// application whose variable the solver has eliminated still claims its
    /// slot -- so the claims stay contiguous and the scatter's cursor walks them
    /// in lockstep -- but names no matrix entry, which is what (-1, -1) means to
    /// NonLinearProgram::analyze_sparsity.
    void get_kkt_space(EigenRef<Eigen::VectorXi> KKTrows, EigenRef<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       SolverIndexingData &data) {
        data.inner_kkt_starts_.resize(data.num_appl());
        for (int V = 0; V < data.num_appl(); V++) {
            data.inner_kkt_starts_[V] = freeloc;
            if (!dojac) {
                continue;
            }
            const int col = data.v_scatter_loc(0, V);
            KKTrows[freeloc] = (col < 0) ? -1 : data.c_loc(0, V) + conoffset;
            KKTcols[freeloc] = col;
            freeloc++;
        }
    }

  private:
    /// x - value for application @p V.
    double residual(ConstEigenRef<Eigen::VectorXd> X, const SolverIndexingData &data, int V) const {
        return X[data.v_loc(0, V)] - this->value_;
    }

    /// Sums this row's single Jacobian element, the constant 1, into the KKT
    /// value array.
    ///
    /// The cursor is the one get_kkt_space claimed: the slot for application
    /// @p Apl is where that application's claims start, because this row claims
    /// exactly one slot per application. An eliminated variable's slot is
    /// stepped over rather than written -- there is no matrix entry behind it --
    /// which the cursor bookkeeping here is trivial about only because the row
    /// has a single element.
    ///
    /// The lock protocol is the shared one (see kkt_canonical_lock_col in
    /// indexing_data.h): a Jacobian element's canonical lock column is the
    /// variable's own column, since constraint rows sort above every variable
    /// column, and the column is left unlocked when this application's
    /// constraint rows are unique to it, so no other partition can be writing
    /// the same slot.
    void scatter_jacobian(int Apl, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                          EigenRef<Eigen::VectorXi> KKTLocs, EigenRef<Eigen::VectorXi> VarClashes,
                          std::vector<std::mutex> &ClashLocks,
                          const SolverIndexingData &data) const {
        const int active = data.v_scatter_loc(0, Apl);
        if (active < 0) {
            // Eliminated variable: the claim exists, the matrix entry does not.
            return;
        }
        const int freeloc = data.inner_kkt_starts_[Apl];
        const bool lock_column = !data.unique_constraints_ && (VarClashes[active] != -1);
        if (lock_column) {
            ClashLocks[VarClashes[active]].lock();
        }
        KKTmat.valuePtr()[KKTLocs.data()[freeloc]] += 1.0;
        if (lock_column) {
            ClashLocks[VarClashes[active]].unlock();
        }
    }
};

/// <summary>
/// Builds one internal equality row pinning primal variable @p index to
/// @p value and writing constraint row @p row, addressed over the problem's own
/// variable space.
///
/// The MakeConstraint fixed-variable treatment's whole payload.
/// </summary>
inline ConstraintFunction make_fixed_variable_row(int index, double value, int row) {
    if (index < 0) {
        throw std::invalid_argument(fmt::format(
            "make_fixed_variable_row: variable index must be non-negative (got {0})", index));
    }
    if (row < 0) {
        throw std::invalid_argument(fmt::format(
            "make_fixed_variable_row: constraint row must be non-negative (got {0})", row));
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            fmt::format("make_fixed_variable_row: the value variable {0} is fixed at must be "
                        "finite (got {1})",
                        index, value));
    }

    // One application, one input, one output: the row reads the variable it pins
    // and writes the constraint row it was given.
    Eigen::MatrixXi v_index(1, 1);
    v_index(0, 0) = index;
    Eigen::MatrixXi c_index(1, 1);
    c_index(0, 0) = row;

    ConstraintFunction fix_row(FixedVariableRow(value), v_index, c_index);
    // The policy the transcription gives a thread-safe single-application
    // function, so these rows spread over the work partitions the same way the
    // user's own single-application constraints do.
    fix_row.thread_mode_ = ThreadingFlags::RoundRobin;
    return fix_row;
}

} // namespace tycho::solvers
