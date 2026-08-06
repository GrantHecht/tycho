// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <string>

#include <Eigen/Core>

#include "tycho/detail/typedefs/eigen_types.h"

namespace tycho::solvers {

/// Solver-neutral sparse NLP in the Ipopt TNLP shape:
///
///   min  f(x)   s.t.  g_lower <= g(x) <= g_upper,  x_lower <= x <= x_upper
///
/// Subclass this (in C++ or Python) and hand it to NLPSolver. Conventions are
/// Ipopt's, verbatim: the Lagrangian is L = obj_factor*f + lambda^T g; eval_hess
/// fills the LOWER TRIANGLE of grad^2 L (row >= col); lambda is in THIS
/// problem's own row space in every signature here. Rows with
/// g_lower == g_upper are equalities; +/-infinity means unbounded on that side;
/// rows with two finite, unequal bounds are handled (internally split);
/// rows unbounded on both sides are dropped.
///
/// Structures are queried once at setup and must not change afterwards.
/// Evaluation callbacks must be pure (same x -> same values): results are
/// cached per iterate. Duplicate (row, col) entries in a structure are legal;
/// their values are summed.
class NLPProblem {
  public:
    virtual ~NLPProblem() = default;

    // --- Sizing (required) ---
    virtual int num_vars() const = 0;
    virtual int num_cons() const = 0;          // 0 = unconstrained is legal
    virtual int num_jac_nonzeros() const = 0;  // must be 0 when num_cons() == 0
    virtual int num_hess_nonzeros() const = 0; // lower triangle of the Lagrangian Hessian

    // --- Bounds (required; +/-infinity = unbounded) ---
    virtual void bounds(Eigen::Ref<Eigen::VectorXd> x_lower, Eigen::Ref<Eigen::VectorXd> x_upper,
                        Eigen::Ref<Eigen::VectorXd> g_lower,
                        Eigen::Ref<Eigen::VectorXd> g_upper) const = 0;

    // --- Evaluation (required; pure) ---
    virtual void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const = 0;
    virtual void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                             Eigen::Ref<Eigen::VectorXd> grad) const = 0;
    virtual void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const = 0;

    // --- Sparsity structures (required; queried once; 0-based) ---
    virtual void jac_structure(Eigen::Ref<Eigen::VectorXi> rows,
                               Eigen::Ref<Eigen::VectorXi> cols) const = 0;
    virtual void hess_structure(Eigen::Ref<Eigen::VectorXi> rows,
                                Eigen::Ref<Eigen::VectorXi> cols) const = 0;

    // --- Derivative values (required; same slot ordering as the structures) ---
    virtual void eval_jac(ConstEigenRef<Eigen::VectorXd> x,
                          Eigen::Ref<Eigen::VectorXd> vals) const = 0;
    virtual void eval_hess(ConstEigenRef<Eigen::VectorXd> x, double obj_factor,
                           ConstEigenRef<Eigen::VectorXd> lambda,
                           Eigen::Ref<Eigen::VectorXd> vals) const = 0;

    // --- Warm start (the primal guess is a solve argument; multipliers are
    //     optional). Return true and fill lambda to seed the solver's
    //     constraint multipliers; the default leaves the solver's own
    //     initialization untouched. ---
    virtual bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const {
        (void)lambda;
        return false;
    }

    virtual std::string name() const { return "NLPProblem"; }
};

} // namespace tycho::solvers
