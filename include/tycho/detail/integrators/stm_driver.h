// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace tycho::integrators {

using vf::GenericFunction;

namespace detail {

/// STM-specific finite-check. Placed outside the hot inner loops (called
/// once per STM output, not per-step), so the overhead is a single
/// Eigen::allFinite sweep per trajectory — negligible against Jacobian
/// multiplication cost.
template <class Derived>
inline void check_stm_finite_or_throw(const Eigen::MatrixBase<Derived> &m, const char *site,
                                      int trajectory_idx = -1) {
    if (m.allFinite())
        return;
    std::string msg = "Non-finite STM output from ";
    msg += site;
    if (trajectory_idx >= 0)
        msg += " (trajectory=" + std::to_string(trajectory_idx) + ")";
    msg += ". The single-step Jacobian (or adjoint Hessian) diverged at some "
           "point in the trajectory — this usually indicates the ODE dynamics "
           "passed through a singularity (division by near-zero, sqrt of "
           "negative, pole in gravity model, etc.). Check the transcription "
           "dense-output states for the offending step.";
    throw std::runtime_error(msg);
}

} // namespace detail

/// STM (State Transition Matrix) computation driver.
///
/// Provides Jacobian and Hessian chain computation through RK step sequences,
/// extracted from Integrator's calculate_jacobian/hessian methods.
///
/// Takes the transcription stepper VF as a parameter rather than owning it.
struct STMDriver {

    /// Compute the full Jacobian (STM) from a sequence of dense states.
    ///
    /// Chains single-step Jacobians from the transcription stepper through
    /// the entire trajectory, optionally using SuperScalar batching.
    template <class DODE>
    static Eigen::MatrixXd
    calculate_jacobian(const GenericFunction<-1, -1> &stepper, const DODE &ode,
                       const std::vector<typename DODE::template Input<double>> &xs, int input_rows,
                       int output_rows, bool enable_vectorization) {

        using Scalar = double;
        using ODEState = typename DODE::template Input<Scalar>;

        Eigen::MatrixXd jx(output_rows, input_rows);
        jx.setZero();
        Eigen::MatrixXd jxall(input_rows, input_rows);
        jxall.setIdentity();

        Eigen::VectorXd stepper_input(input_rows);
        Eigen::VectorXd stepper_output(ode.input_rows());
        Eigen::MatrixXd stepper_jacobian(output_rows, input_rows);
        Eigen::MatrixXd jactmp(output_rows, input_rows);

        int numsteps = xs.size() - 1;

        constexpr int vsize = tycho::DefaultSuperScalar::SizeAtCompileTime;

        Eigen::Matrix<tycho::DefaultSuperScalar, -1, 1> stepper_input_ss(input_rows);
        Eigen::Matrix<tycho::DefaultSuperScalar, -1, 1> stepper_output_ss(ode.input_rows());
        Eigen::Matrix<tycho::DefaultSuperScalar, -1, -1> stepper_jacobian_ss(output_rows,
                                                                             input_rows);

        auto scalar_impl = [&](int i) {
            stepper_input.head(ode.input_rows()) = xs[i];
            stepper_input[ode.input_rows()] = xs[i + 1][ode.t_var()];

            stepper_output.setZero();
            stepper_jacobian.setZero();

            stepper.compute_jacobian(stepper_input, stepper_output, stepper_jacobian);
            jactmp.noalias() = stepper_jacobian * jxall;
            jxall.topRows(output_rows) = jactmp;
        };

        auto vector_impl = [&](int i) {
            stepper_output_ss.setZero();
            stepper_jacobian_ss.setZero();

            for (int j = 0; j < vsize; j++) {
                for (int k = 0; k < ode.input_rows(); k++) {
                    stepper_input_ss.head(ode.input_rows())[k][j] = xs[i + j][k];
                }
                stepper_input_ss[ode.input_rows()][j] = xs[i + j + 1][ode.t_var()];
            }
            stepper.compute_jacobian(stepper_input_ss, stepper_output_ss, stepper_jacobian_ss);

            for (int j = 0; j < vsize; j++) {
                for (int k = 0; k < input_rows; k++) {
                    for (int l = 0; l < output_rows; l++) {
                        stepper_jacobian(l, k) = stepper_jacobian_ss(l, k)[j];
                    }
                }
                jactmp.noalias() = stepper_jacobian * jxall;
                jxall.topRows(output_rows) = jactmp;
            }
        };

        int packs = enable_vectorization ? numsteps / vsize : 0;

        for (int i = 0; i < packs; i++) {
            vector_impl(i * vsize);
        }
        for (int i = packs * vsize; i < numsteps; i++) {
            scalar_impl(i);
        }

        jx = jxall.topRows(output_rows);
        detail::check_stm_finite_or_throw(jx, "STMDriver::calculate_jacobian");
        return jx;
    }

    /// Compute Jacobian + Hessian from a sequence of dense states + adjoint.
    template <class DODE>
    static std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
    calculate_jacobian_hessian(const GenericFunction<-1, -1> &stepper, const DODE &ode,
                               const std::vector<typename DODE::template Input<double>> &xs,
                               const typename DODE::template Input<double> &lf, int input_rows,
                               int output_rows) {

        using ODEState = typename DODE::template Input<double>;

        Eigen::MatrixXd jx(output_rows, input_rows);
        jx.setZero();

        Eigen::MatrixXd jxall(output_rows, input_rows);
        jxall.setZero();
        jxall.leftCols(output_rows).setIdentity();

        Eigen::MatrixXd hxall(input_rows, input_rows);
        hxall.setZero();

        Eigen::VectorXd stepper_input(input_rows);
        Eigen::VectorXd stepper_grad(input_rows);
        Eigen::VectorXd stepper_output(ode.input_rows());
        Eigen::MatrixXd stepper_jacobian(output_rows, input_rows);
        Eigen::MatrixXd stepper_hessian(input_rows, input_rows);

        ODEState stepper_adjvars = lf;

        Eigen::MatrixXd jtwist(input_rows, input_rows);
        jtwist.setZero();
        jtwist(input_rows - 1, input_rows - 1) = 1.0;

        int numsteps = xs.size() - 1;

        for (int i = 0; i < numsteps; i++) {
            stepper_input.head(ode.input_rows()) = xs[numsteps - i - 1];
            stepper_input[ode.input_rows()] = xs[numsteps - i][ode.t_var()];

            stepper_output.setZero();
            stepper_jacobian.setZero();
            stepper_grad.setZero();
            stepper_hessian.setZero();

            stepper.compute_jacobian_adjointgradient_adjointhessian(
                stepper_input, stepper_output, stepper_jacobian, stepper_grad, stepper_hessian,
                stepper_adjvars);

            jtwist.topRows(output_rows) = stepper_jacobian;
            jxall = jxall * jtwist;
            if (i == 0) {
                jxall.rightCols(1) = stepper_jacobian.rightCols(1);
            }

            hxall = jtwist.transpose() * hxall * jtwist;
            hxall += stepper_hessian;
            stepper_adjvars = stepper_grad.head(output_rows);
        }

        jx = jxall.topRows(output_rows);
        detail::check_stm_finite_or_throw(jx, "STMDriver::calculate_jacobian_hessian (jacobian)");
        detail::check_stm_finite_or_throw(hxall, "STMDriver::calculate_jacobian_hessian (hessian)");
        return {jx, hxall};
    }

    /// Batch Jacobians via SuperScalar packing.
    template <class DODE>
    static std::vector<Eigen::MatrixXd>
    calculate_jacobians(const GenericFunction<-1, -1> &stepper, const DODE &ode,
                        const std::vector<std::vector<typename DODE::template Input<double>>> &xs_s,
                        int input_rows, int output_rows) {

        constexpr int vsize = tycho::DefaultSuperScalar::SizeAtCompileTime;

        Eigen::Matrix<tycho::DefaultSuperScalar, -1, 1> stepper_input_ss(input_rows);
        Eigen::Matrix<tycho::DefaultSuperScalar, -1, 1> stepper_output_ss(ode.input_rows());
        Eigen::Matrix<tycho::DefaultSuperScalar, -1, -1> stepper_jacobian_ss(output_rows,
                                                                             input_rows);
        Eigen::Matrix<tycho::DefaultSuperScalar, -1, -1> jxall_ss(input_rows, input_rows);
        Eigen::Matrix<tycho::DefaultSuperScalar, -1, -1> jactmp_ss(output_rows, input_rows);

        int ntrajs = xs_s.size();

        std::vector<int> idxs(ntrajs);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(),
                  [&](auto a, auto b) { return xs_s[a].size() < xs_s[b].size(); });

        int npacks = ntrajs / vsize;
        int nrem = ntrajs % vsize;
        if (nrem > 0)
            npacks++;

        std::vector<Eigen::MatrixXd> jxs(ntrajs);

        if (ntrajs < vsize) {
            for (int v = 0; v < vsize; v++) {
                for (int j = 0; j < ode.input_rows(); j++) {
                    stepper_input_ss[j][v] = xs_s[0][0][j];
                }
            }
        }

        for (int n = 0; n < npacks; n++) {
            int vmax = std::min(vsize, ntrajs - n * vsize);

            for (int v = 0; v < vmax; v++) {
                int idx = idxs[n * vsize + v];
                for (int j = 0; j < ode.input_rows(); j++) {
                    stepper_input_ss[j][v] = xs_s[idx][0][j];
                }
            }

            int ncalls = xs_s[idxs[n * vsize]].size() - 1;
            jxall_ss.setIdentity();
            jactmp_ss.setZero();

            for (int i = 0; i < ncalls; i++) {
                for (int v = 0; v < vmax; v++) {
                    int idx = idxs[n * vsize + v];
                    for (int j = 0; j < ode.input_rows(); j++) {
                        stepper_input_ss[j][v] = xs_s[idx][i][j];
                    }
                    stepper_input_ss[ode.input_rows()][v] = xs_s[idx][i + 1][ode.t_var()];
                }

                stepper_output_ss.setZero();
                stepper_jacobian_ss.setZero();
                stepper.compute_jacobian(stepper_input_ss, stepper_output_ss, stepper_jacobian_ss);
                jactmp_ss.noalias() = stepper_jacobian_ss * jxall_ss;
                jxall_ss.topRows(output_rows) = jactmp_ss;
                stepper_input_ss.head(ode.input_rows()) = stepper_output_ss;
            }

            for (int v = 0; v < vmax; v++) {
                int idx = idxs[n * vsize + v];
                Eigen::MatrixXd jxall_scalar(input_rows, input_rows);

                for (int k = 0; k < input_rows; k++) {
                    for (int l = 0; l < input_rows; l++) {
                        jxall_scalar(l, k) = jxall_ss(l, k)[v];
                    }
                }

                // Tail-handling for heterogeneous trajectory lengths: when this
                // lane has more steps than the pack's shortest, finish the
                // remaining steps scalar-style and multiply into the packed
                // Jacobian. Matches the production path previously in
                // Integrator::calculate_jacobians.
                if (ncalls == static_cast<int>(xs_s[idx].size()) - 1) {
                    jxs[idx] = jxall_scalar.topRows(output_rows);
                } else {
                    std::vector<typename DODE::template Input<double>> xs_tail(
                        xs_s[idx].begin() + ncalls, xs_s[idx].end());
                    Eigen::MatrixXd jtail = STMDriver::calculate_jacobian(
                        stepper, ode, xs_tail, input_rows, output_rows,
                        /*enable_vectorization=*/false);
                    jxs[idx] = jtail * jxall_scalar;
                }
                detail::check_stm_finite_or_throw(jxs[idx], "STMDriver::calculate_jacobians", idx);
            }
        }

        return jxs;
    }
};

} // namespace tycho::integrators
