// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// The internal equality row the MakeConstraint fixed-variable treatment installs:
// x[index] - value = 0, one per variable whose declared bounds are equal.
//
// Its own translation unit on purpose. The row is a VectorFunction expression, so
// building it needs the VectorFunction headers; keeping it here leaves the
// NonLinearProgram's own translation unit -- which is compiled far more often and
// has nothing else to do with the expression machinery -- free of them.
// =============================================================================

#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/vector_functions.h"

tycho::solvers::ConstraintFunction tycho::solvers::make_fixed_variable_row(int index, double value,
                                                                          int row) {
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
    auto args = tycho::vf::Arguments<1>();
    auto residual = args.coeff<0>() - value;

    Eigen::MatrixXi v_index(1, 1);
    v_index(0, 0) = index;
    Eigen::MatrixXi c_index(1, 1);
    c_index(0, 0) = row;

    ConstraintFunction fix_row(tycho::vf::GenericFunction<-1, -1>(residual), v_index, c_index);
    // The policy the transcription gives a thread-safe single-application
    // function, so these rows spread over the work partitions the same way the
    // user's own single-application constraints do.
    fix_row.thread_mode_ = ThreadingFlags::RoundRobin;
    return fix_row;
}
