// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/vf/type_erasure/gf_type_erasure.h"
#include "tycho/vector_functions.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <hven/solver_interface_adapter.h>
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

#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/interior/typedefs/eigen_types.h>
#include <hven/detail/interior/utils/flat_map.h>
#include <hven/detail/interior/utils/function_return_type.h>
#include <hven/detail/interior/utils/get_core_count.h>
#include <hven/detail/interior/utils/math_functions.h>
#include <hven/detail/interior/utils/sizing_helpers.h>
#include <hven/detail/interior/utils/std_extensions.h>
#include <hven/detail/interior/utils/thread_pool.h>
#include <hven/detail/interior/utils/type_name.h>
#include <hven/detail/interior/utils/type_storage.h>

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::Arguments;
using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::Is_SuperScalar;
using vf::MatRef;
using vf::StackedOutputs;
using vf::ThreadingFlags;
using vf::VecRef;
using vf::VectorExpression;
using vf::VectorFunction;

// Solvers types
using tycho::solvers::SolverIndexingData;

/// @internal
/// @brief Expression builder for a central-shooting defect via a VF integrator expression.
///
/// Builds the defect that integrates both arcs to the interval midpoint and
/// returns their state difference, using the VectorFunction integrator
/// expression (the expression-tree formulation of @ref ShootingDefect).
/// @tparam DODE        The ODE type whose dynamics are shot.
/// @tparam Integrator  The integrator-expression type.
/// @endinternal
template <class DODE, class Integrator> struct ShootingDefect_Impl {
    /// @internal
    /// @brief Build the shooting-defect expression.
    /// @param ode    The ODE to shoot.
    /// @param integ  The integrator expression.
    /// @return The VectorFunction expression for the shooting defect.
    /// @endinternal
    static auto Definition(const DODE &ode, const Integrator &integ) {
        constexpr int IRC = SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value;
        int input_rows = ode.xtu_vars() * 2 + ode.p_vars();

        auto args = Arguments<IRC>(input_rows);
        // Input[x1,t1,u1,x2,t2,u2,pv]

        auto x1 = args.template head<DODE::XtUV>(ode.xtu_vars());
        auto t1 = x1.template coeff<DODE::XV>(ode.x_vars());
        auto x2 = args.template segment<DODE::XtUV, DODE::XtUV>(ode.xtu_vars(), ode.xtu_vars());
        auto t2 = x2.template coeff<DODE::XV>(ode.x_vars());

        auto tm = 0.5 * (t1 + t2);

        auto pvars = args.template tail<DODE::PV>(ode.p_vars());

        auto make_state = [&](auto xx) {
            if constexpr (DODE::PV == 0) {
                return StackedOutputs{xx, tm};
            } else {
                return StackedOutputs{xx, pvars, tm};
            }
        };

        auto Arc1Input = make_state(x1);
        auto Arc2Input = make_state(x2);

        auto defect = integ.eval(Arc1Input).template head<DODE::XV>(ode.x_vars()) -
                      integ.eval(Arc2Input).template head<DODE::XV>(ode.x_vars());

        return defect;
    }
};

/// @ingroup optimal_control
/// @brief Expression-tree central-shooting defect constraint.
///
/// VectorFunction-expression formulation of the central-shooting defect built
/// by @c ShootingDefect_Impl; the legacy shooting-defect path.
/// @tparam DODE        The ODE type whose dynamics are shot.
/// @tparam Integrator  The integrator-expression type.
template <class DODE, class Integrator>
struct ShootingDefect
    : VectorExpression<ShootingDefect<DODE, Integrator>, ShootingDefect_Impl<DODE, Integrator>,
                       const DODE &, const Integrator &> {
    /// @brief Convenience alias for the VectorExpression base class.
    using Base =
        VectorExpression<ShootingDefect<DODE, Integrator>, ShootingDefect_Impl<DODE, Integrator>,
                         const DODE &, const Integrator &>;
    // using Base::Base;
    /// @brief Default constructor; leaves the ODE and integrator unset.
    ShootingDefect() {}
    /// @brief Construct from an ODE and an integrator expression.
    /// @param ode    The ODE to shoot.
    /// @param integ  The integrator expression.
    ShootingDefect(const DODE &ode, const Integrator &integ) : Base(ode, integ) {}
    /// @brief Whether to exploit Hessian sparsity.
    ///
    /// @note Not honored on this legacy expression-tree path; the KKT fill runs
    ///   through the generic VectorExpression machinery, which has no static
    ///   sparsity mask. Hessian-sparsity exploitation is implemented only on the
    ///   default @ref CentralShootingDefect path.
    bool enable_hessian_sparsity_ = false;
};

/// @ingroup optimal_control
/// @brief Central-shooting defect constraint with hand-written STM derivatives.
///
/// The equality constraint VectorFunction enforcing that two arcs integrated to
/// the interval midpoint agree. Derivatives are assembled from the integrator's
/// state-transition matrices (and second-order STMs for the Hessian); supports
/// SuperScalar batched evaluation.
/// @tparam DODE        The ODE type whose dynamics are shot.
/// @tparam Integrator  The integrator type providing STM integration.
template <class DODE, class Integrator>
struct CentralShootingDefect
    : VectorFunction<CentralShootingDefect<DODE, Integrator>,
                     SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value, DODE::XV> {

    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<CentralShootingDefect<DODE, Integrator>,
                                SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value, DODE::XV>;

    VF_TYPE_ALIASES(Base);

    /// @brief ODE input-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    /// @brief ODE output-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;
    /// @brief Integrator Jacobian (STM) type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using IntegJac = typename Integrator::template Jacobian<Scalar>;

    static constexpr bool is_vectorizable = true; ///< Supports SuperScalar batched evaluation.
    bool enable_hessian_sparsity_ = false;        ///< Whether to exploit Hessian sparsity.
    Eigen::MatrixXi nz_locs_;                     ///< Nonzero-pattern mask of the adjoint Hessian.

    DODE ode_;         ///< The ODE whose dynamics are shot.
    Integrator integ_; ///< The integrator providing STM integration.

    /// @internal
    /// @brief Per-instance scratch buffers reused across evaluations (OC §2.1 fix).
    ///
    /// `compute_impl`/`compute_jacobian_impl`/
    /// `compute_jacobian_adjointgradient_adjointhessian_impl` and the batched
    /// `compute_impl_v`/`compute_jacobian_impl_v`/`compute_all_impl_v` used to
    /// declare fresh `std::vector`s on every call -- the O(segments x InteriorPointSolver
    /// iterations) hot path in the transcription layer. These are now
    /// per-instance `mutable` scratch buffers that get `.resize()`d (not
    /// freshly declared) each call: `std::vector::resize()` to an unchanged
    /// size never reallocates, and `input_rows()`/`output_rows()`/
    /// `ode_.input_rows()` are fixed once the defect is constructed, so after
    /// the first (warm-up) call these buffers cost zero heap traffic.
    ///
    /// Thread-safety: `mutable` scratch on a `const`-qualified evaluation path
    /// is only safe if no two threads ever call these methods on the *same*
    /// instance concurrently. That holds here:
    /// `ConstraintFunction::thread_split()` (constraint_function.h) partitions
    /// the shooting constraints once, before the iterative solve, via
    /// `TypeStorage::clone_into` (type_storage.h) -- a deep copy that gives
    /// every partition (and hence every worker thread) its *own*
    /// `CentralShootingDefect` instance, including independent copies of these
    /// scratch buffers. Within a partition, `NonLinearProgram`'s KKT-eval loop
    /// (non_linear_program.cpp) dispatches each partition's work via
    /// `tycho::utils::parallel_sequence`, which enqueues one task per
    /// partition and *joins all of them* before returning -- so the same
    /// partition's defect instance is never invoked by two threads at once,
    /// and there is a proper happens-before edge between one KKT-eval call and
    /// the next. So this is genuinely per-instance, effectively
    /// single-threaded state, not shared mutable state.
    ///
    /// `xs_scratch_`/`tfs_scratch_`/`lfs_scratch_` specifically must remain
    /// plain `std::vector<ODEState<double>>`/`Eigen::VectorXd` (not
    /// arena-backed `BumpAllocator` temporaries, unlike e.g.
    /// `TrapezoidalDefects`) because they are passed by const-ref directly
    /// into `Integrator::integrate`/`integrate_stm`/`integrate_stm2`, whose
    /// public signature (owned by the integrators layer, out of scope for
    /// this change) requires exactly that container type -- an
    /// arena-allocated `Eigen::Map` temporary cannot bind there without a
    /// copy that would defeat the purpose.
    /// @endinternal
    /// @internal Per-lane packed inputs. Shared by compute/jacobian/JGH entry
    /// points — safe while each top-level call fully consumes it before
    /// returning; do NOT delegate between those hooks while it is live.
    mutable std::vector<Input<double>> x1x2s_scratch_;
    mutable std::vector<Output<double>> l_scratch_;    ///< @internal Per-lane adjoint multipliers.
    mutable std::vector<ODEState<double>>
        xs_scratch_;                      ///< @internal Two-per-lane ODE states (integrator input).
    mutable Eigen::VectorXd tfs_scratch_; ///< @internal Two-per-lane integration targets.
    mutable std::vector<ODEState<double>>
        lfs_scratch_; ///< @internal Two-per-lane ODE-sized multiplier seeds.
    mutable std::vector<Output<double>> fxs_scratch_;   ///< @internal Per-lane defect residuals.
    mutable std::vector<Jacobian<double>> jxs_scratch_; ///< @internal Per-lane Jacobians.
    mutable std::vector<Hessian<double>> hxs_scratch_;  ///< @internal Per-lane adjoint Hessians.

    /// @brief Construct from an ODE and integrator and size the defect.
    /// @param ode    The ODE to shoot.
    /// @param integ  The integrator providing STM integration.
    CentralShootingDefect(const DODE &ode, const Integrator &integ) : ode_(ode), integ_(integ) {
        this->set_io_rows(2 * this->ode_.xtu_vars() + this->ode_.p_vars(), this->ode_.x_vars());

        // Build the static adjoint-Hessian sparsity mask. The shooting-defect
        // input layout is [x1(xtu), x2(xtu), p] with 2 cardinal endpoints, the
        // same structure the trapezoidal defect uses (trapezoidal_defects.h). The
        // adjoint Hessian couples: each endpoint's xtu block with itself, the p
        // block with itself, each endpoint's xtu block with p, and (via the shared
        // midpoint time tm = (t1 + t2) / 2) the two endpoint time rows/columns.
        // The only structural zero is the endpoint-to-endpoint cross block: the
        // two arcs' second-order STMs never couple non-time states of one endpoint
        // to non-time states of the other (the integrator Hessian is block
        // diagonal per arc).
        nz_locs_.resize(this->input_rows(), this->input_rows());
        nz_locs_.setZero();

        int xtu = this->ode_.xtu_vars();
        nz_locs_.topLeftCorner(xtu, xtu).setOnes();
        nz_locs_.block(xtu, xtu, xtu, xtu).setOnes();
        nz_locs_.bottomRightCorner(this->ode_.p_vars(), this->ode_.p_vars()).setOnes();

        int j = 0;
        int cardinals = 2;
        nz_locs_
            .block(j * this->ode_.xtu_vars(), cardinals * this->ode_.xtu_vars(),
                   this->ode_.xtu_vars(), this->ode_.p_vars())
            .setOnes();

        nz_locs_
            .block(cardinals * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(),
                   this->ode_.p_vars(), this->ode_.xtu_vars())
            .setOnes();
        j = 1;
        nz_locs_
            .block(j * this->ode_.xtu_vars(), cardinals * this->ode_.xtu_vars(),
                   this->ode_.xtu_vars(), this->ode_.p_vars())
            .setOnes();

        nz_locs_
            .block(cardinals * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(),
                   this->ode_.p_vars(), this->ode_.xtu_vars())
            .setOnes();

        nz_locs_.col(this->ode_.t_var()).setOnes();
        nz_locs_.col(this->ode_.t_var() + this->ode_.xtu_vars() * (cardinals - 1)).setOnes();
        nz_locs_.row(this->ode_.t_var()).setOnes();
        nz_locs_.row(this->ode_.t_var() + this->ode_.xtu_vars() * (cardinals - 1)).setOnes();
    }

    /// @internal
    /// @brief Whether the given adjoint-Hessian element is (possibly) nonzero.
    /// @param row  Hessian row index.
    /// @param col  Hessian column index.
    /// @return True if the element may be nonzero (always true unless sparsity is enabled).
    /// @endinternal
    inline bool hessian_elem_is_nonzero(int row, int col) const {
        if (this->enable_hessian_sparsity_) {
            return bool(this->nz_locs_(row, col));
        } else {
            return true;
        }
    }

    /// @internal
    /// @brief Accumulate one adjoint-Hessian element into the sparse storage.
    /// @param v        The value to add.
    /// @param row      Hessian row index.
    /// @param col      Hessian column index.
    /// @param mpt      Pointer to the dense value storage.
    /// @param lpt      Pointer to the element-location index table.
    /// @param freeloc  In/out next free slot in @p lpt; advanced if the element is stored.
    /// @endinternal
    inline void add_hessian_elem(double v, int row, int col, double *mpt, const int *lpt,
                                 int &freeloc) const {
        if (this->enable_hessian_sparsity_) {
            if (bool(this->nz_locs_(row, col))) {
                // Slot validity, on the path where the cursor contract is a RUNTIME
                // predicate rather than a compile-time one and can therefore drift:
                // an eliminated element's location stays -1, so a cursor out of step
                // with get_kkt_space's claims would write through valuePtr()[-1].
                // Debug-only; compiled out under NDEBUG.
                assert(lpt[freeloc] >= 0 &&
                       "KKT Hessian fill cursor landed on an eliminated element's slot");
                mpt[lpt[freeloc]] += v;
                freeloc++;
            }
        } else {
            assert(lpt[freeloc] >= 0 &&
                   "KKT Hessian fill cursor landed on an eliminated element's slot");
            mpt[lpt[freeloc]] += v;
            freeloc++;
        }
    }

    /// @brief Default constructor; leaves the ODE and integrator unset.
    CentralShootingDefect() {}

    /// @internal
    /// @brief Unpack a SuperScalar input pack into per-lane double input vectors.
    /// @tparam InType  Eigen input-vector (SuperScalar) type.
    /// @param X1X2   The packed SuperScalar input.
    /// @param X1X2s  Output per-lane double input vectors.
    /// @endinternal
    template <class InType>
    void extract_scalar_inputs(CVecRef<InType> X1X2, std::vector<Input<double>> &X1X2s) const {

        typedef typename InType::Scalar Scalar;

        X1X2s.resize(Scalar::SizeAtCompileTime);
        for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
            X1X2s[v].resize(this->input_rows());
            for (int i = 0; i < this->input_rows(); i++) {
                X1X2s[v][i] = X1X2[i][v];
            }
        }
    }

    /// @internal
    /// @brief Unpack a SuperScalar multiplier pack into per-lane double vectors.
    /// @tparam InType  Eigen vector (SuperScalar) type.
    /// @param Lf   The packed SuperScalar multipliers.
    /// @param Lfs  Output per-lane double multiplier vectors.
    /// @endinternal
    template <class InType>
    void extract_scalar_lmults(CVecRef<InType> Lf, std::vector<Output<double>> &Lfs) const {

        typedef typename InType::Scalar Scalar;

        Lfs.resize(Scalar::SizeAtCompileTime);
        for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
            Lfs[v].resize(this->output_rows());
            for (int i = 0; i < this->output_rows(); i++) {
                Lfs[v][i] = Lf[i][v];
            }
        }
    }

    /// @internal
    /// @brief Split each two-endpoint input into two ODE states with a shared midpoint time.
    /// @param X1X2s  Per-application packed two-endpoint inputs.
    /// @param Xs     Output ODE input states (two per application).
    /// @param tfs    Output integration target times (midpoint, two per application).
    /// @endinternal
    void get_input_states_tfs(const std::vector<Input<double>> &X1X2s,
                              std::vector<ODEState<double>> &Xs, Eigen::VectorXd &tfs) const {

        Xs.resize(2 * X1X2s.size());
        tfs.resize(2 * X1X2s.size());

        for (int i = 0; i < X1X2s.size(); i++) {

            Xs[2 * i].resize(this->ode_.input_rows());
            Xs[2 * i + 1].resize(this->ode_.input_rows());

            Xs[2 * i].head(this->ode_.xtu_vars()) = X1X2s[i].head(this->ode_.xtu_vars());
            Xs[2 * i + 1].head(this->ode_.xtu_vars()) =
                X1X2s[i].segment(this->ode_.xtu_vars(), this->ode_.xtu_vars());

            double tm = (Xs[2 * i][this->ode_.t_var()] + Xs[2 * i + 1][this->ode_.t_var()]) / 2.0;

            tfs[2 * i] = tm;
            tfs[2 * i + 1] = tm;

            if constexpr (DODE::PV != 0) {

                Xs[2 * i].tail(this->ode_.p_vars()) = X1X2s[i].tail(this->ode_.p_vars());
                Xs[2 * i + 1].tail(this->ode_.p_vars()) = X1X2s[i].tail(this->ode_.p_vars());
            }
        }
    }
    /// @internal
    /// @brief Lift output multipliers into ODE-sized seed vectors for both arcs.
    /// @param Ls   Per-application output multipliers.
    /// @param Lfs  Output ODE-sized seed vectors (two per application).
    /// @endinternal
    void get_lmults(const std::vector<Output<double>> &Ls,
                    std::vector<ODEState<double>> &Lfs) const {

        Lfs.resize(2 * Ls.size());

        for (int i = 0; i < Ls.size(); i++) {

            Lfs[2 * i].resize(this->ode_.input_rows());
            Lfs[2 * i + 1].resize(this->ode_.input_rows());
            Lfs[2 * i].setZero();
            Lfs[2 * i + 1].setZero();

            Lfs[2 * i].head(this->ode_.x_vars()) = Ls[i];
            Lfs[2 * i + 1].head(this->ode_.x_vars()) = Ls[i];
        }
    }

    /// @internal
    /// @brief Batched primal evaluation: integrate both arcs and difference them.
    /// @param X1X2s  Per-application packed two-endpoint inputs.
    /// @return Per-application defect-residual vectors (reference to reused scratch storage;
    ///   valid until the next call on this instance).
    /// @endinternal
    std::vector<Output<double>> &compute_impl_v(const std::vector<Input<double>> &X1X2s) const {

        this->get_input_states_tfs(X1X2s, this->xs_scratch_, this->tfs_scratch_);

        auto Xfs = this->integ_.integrate(this->xs_scratch_, this->tfs_scratch_);

        this->fxs_scratch_.resize(X1X2s.size());

        for (int i = 0; i < X1X2s.size(); i++) {
            this->fxs_scratch_[i] =
                Xfs[2 * i].head(this->ode_.x_vars()) - Xfs[2 * i + 1].head(this->ode_.x_vars());
        }
        return this->fxs_scratch_;
    }

    /// @internal
    /// @brief Batched residual + Jacobian via the integrator state-transition matrices.
    /// @param X1X2s  Per-application packed two-endpoint inputs.
    /// @return Tuple of per-application {defect residuals, Jacobians} (references to reused
    ///   scratch storage; valid until the next call on this instance).
    /// @endinternal
    std::tuple<std::vector<Output<double>> &, std::vector<Jacobian<double>> &>
    compute_jacobian_impl_v(const std::vector<Input<double>> &X1X2s) const {

        this->get_input_states_tfs(X1X2s, this->xs_scratch_, this->tfs_scratch_);
        auto Xfs_Jfs = this->integ_.integrate_stm(this->xs_scratch_, this->tfs_scratch_);

        this->fxs_scratch_.resize(X1X2s.size());
        this->jxs_scratch_.resize(X1X2s.size());
        auto &fxs = this->fxs_scratch_;
        auto &jxs = this->jxs_scratch_;

        Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(
            ode_.output_rows(), integ_.input_rows() * 2);
        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(
            integ_.input_rows() * 2, this->input_rows());

        XJac.setZero();

        XJac.topLeftCorner(ode_.xtu_vars(), ode_.xtu_vars()).setIdentity();
        XJac.block(ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(), ode_.p_vars())
            .setIdentity();
        XJac(ode_.input_rows(), ode_.t_var()) = .5;
        XJac(ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        XJac.block(integ_.input_rows(), ode_.xtu_vars(), ode_.xtu_vars(), ode_.xtu_vars())
            .setIdentity();
        XJac.block(integ_.input_rows() + ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(),
                   ode_.p_vars())
            .setIdentity();

        XJac(integ_.input_rows() + ode_.input_rows(), ode_.t_var()) = .5;
        XJac(integ_.input_rows() + ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        for (int i = 0; i < X1X2s.size(); i++) {

            auto &[Xf1, Jf1] = Xfs_Jfs[2 * i];
            auto &[Xf2, Jf2] = Xfs_Jfs[2 * i + 1];

            Jf2 *= -1.0;

            fxs[i] = Xf1.head(ode_.x_vars()) - Xf2.head(ode_.x_vars());

            IJac.leftCols(integ_.input_rows()) = Jf1.topRows(ode_.x_vars());
            IJac.rightCols(integ_.input_rows()) = Jf2.topRows(ode_.x_vars());

            jxs[i].noalias() = IJac * XJac;
        }
        return std::tuple<std::vector<Output<double>> &, std::vector<Jacobian<double>> &>{fxs, jxs};
    }

    /// @internal
    /// @brief Batched residual, Jacobian, and adjoint Hessian via first/second-order STMs.
    /// @param X1X2s  Per-application packed two-endpoint inputs.
    /// @param Ls     Per-application output multipliers.
    /// @return Tuple of per-application {defect residuals, Jacobians, adjoint Hessians}
    ///   (references to reused scratch storage; valid until the next call on this instance).
    /// @endinternal
    std::tuple<std::vector<Output<double>> &, std::vector<Jacobian<double>> &,
               std::vector<Hessian<double>> &>
    compute_all_impl_v(const std::vector<Input<double>> &X1X2s,
                       const std::vector<Output<double>> &Ls) const {

        this->get_input_states_tfs(X1X2s, this->xs_scratch_, this->tfs_scratch_);
        this->get_lmults(Ls, this->lfs_scratch_);

        auto Xfs_Jfs_Hfs =
            this->integ_.integrate_stm2(this->xs_scratch_, this->tfs_scratch_, this->lfs_scratch_);

        this->fxs_scratch_.resize(X1X2s.size());
        this->jxs_scratch_.resize(X1X2s.size());
        this->hxs_scratch_.resize(X1X2s.size());
        auto &fxs = this->fxs_scratch_;
        auto &jxs = this->jxs_scratch_;
        auto &hxs = this->hxs_scratch_;

        Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(
            ode_.output_rows(), integ_.input_rows() * 2);
        IJac.setZero();

        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value,
                      SZ_PROD<Integrator::IRC, 2>::value>
            IHess(integ_.input_rows() * 2, integ_.input_rows() * 2);

        IHess.setZero();

        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(
            integ_.input_rows() * 2, this->input_rows());
        XJac.setZero();

        XJac.topLeftCorner(ode_.xtu_vars(), ode_.xtu_vars()).setIdentity();
        XJac.block(ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(), ode_.p_vars())
            .setIdentity();
        XJac(ode_.input_rows(), ode_.t_var()) = .5;
        XJac(ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        XJac.block(integ_.input_rows(), ode_.xtu_vars(), ode_.xtu_vars(), ode_.xtu_vars())
            .setIdentity();
        XJac.block(integ_.input_rows() + ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(),
                   ode_.p_vars())
            .setIdentity();

        XJac(integ_.input_rows() + ode_.input_rows(), ode_.t_var()) = .5;
        XJac(integ_.input_rows() + ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        for (int i = 0; i < X1X2s.size(); i++) {

            auto &[Xf1, Jf1, Hf1] = Xfs_Jfs_Hfs[2 * i];
            auto &[Xf2, Jf2, Hf2] = Xfs_Jfs_Hfs[2 * i + 1];

            Jf2 *= -1.0;
            Hf2 *= -1.0;

            fxs[i] = Xf1.head(this->ode_.x_vars()) - Xf2.head(this->ode_.x_vars());

            IJac.leftCols(integ_.input_rows()) = Jf1.topRows(ode_.x_vars());
            IJac.rightCols(integ_.input_rows()) = Jf2.topRows(ode_.x_vars());

            IHess.topLeftCorner(integ_.input_rows(), integ_.input_rows()) = Hf1;
            IHess.bottomRightCorner(integ_.input_rows(), integ_.input_rows()) = Hf2;

            jxs[i].noalias() = IJac * XJac;
            hxs[i].noalias() = XJac.transpose() * IHess * XJac;
        }
        return std::tuple<std::vector<Output<double>> &, std::vector<Jacobian<double>> &,
                          std::vector<Hessian<double>> &>{fxs, jxs, hxs};
    }

    /// @internal
    /// @brief Evaluate the shooting-defect residual (scalar or SuperScalar dispatch).
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Packed two-endpoint input.
    /// @param fx_  Output defect-residual vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto &X1X2s = this->x1x2s_scratch_;

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            X1X2s.resize(1);
            X1X2s[0] = x;
        } else {
            this->extract_scalar_inputs(x, X1X2s);
        }

        auto &fxs = this->compute_impl_v(X1X2s);

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            fx = fxs.front();
        } else {
            for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i][v] = fxs[v][i];
                }
            }
        }
    }
    /// @internal
    /// @brief Evaluate the shooting-defect residual and Jacobian (scalar or SuperScalar).
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Packed two-endpoint input.
    /// @param fx_  Output defect-residual vector to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto &X1X2s = this->x1x2s_scratch_;

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            X1X2s.resize(1);
            X1X2s[0] = x;
        } else {
            this->extract_scalar_inputs(x, X1X2s);
        }

        auto [fxs, jxs] = this->compute_jacobian_impl_v(X1X2s);

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            fx = fxs.front();
            jx = jxs.front();

        } else {
            for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i][v] = fxs[v][i];
                }

                for (int j = 0; j < this->input_rows(); j++) {
                    for (int i = 0; i < this->output_rows(); i++) {
                        jx(i, j)[v] = jxs[v](i, j);
                    }
                }
            }
        }
    }
    /// @internal
    /// @brief Evaluate the shooting-defect residual, Jacobian, adjoint gradient, and Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Packed two-endpoint input.
    /// @param fx_      Output defect-residual vector to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjhess_ Output adjoint Hessian to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        auto &X1X2s = this->x1x2s_scratch_;
        auto &Lfs = this->l_scratch_;

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            X1X2s.resize(1);
            X1X2s[0] = x;
            Lfs.resize(1);
            Lfs[0] = adjvars;
        } else {
            this->extract_scalar_inputs(x, X1X2s);
            this->extract_scalar_lmults(adjvars, Lfs);
        }

        auto [fxs, jxs, hxs] = this->compute_all_impl_v(X1X2s, Lfs);

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            fx = fxs.front();
            jx = jxs.front();
            adjhess = hxs.front();
        } else {
            for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {

                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i][v] = fxs[v][i];
                }

                for (int j = 0; j < this->input_rows(); j++) {
                    for (int i = 0; i < this->output_rows(); i++) {
                        jx(i, j)[v] = jxs[v](i, j);
                    }
                }

                for (int j = 0; j < this->input_rows(); j++) {
                    for (int i = 0; i < this->input_rows(); i++) {
                        adjhess(i, j)[v] = hxs[v](i, j);
                    }
                }
            }
        }

        adjgrad = jx.transpose() * adjvars;
    }

    /// @internal
    /// @brief Evaluate constraints/gradients and fill the KKT matrix for all applications.
    /// @param X             Full decision-variable vector.
    /// @param L             Full multiplier vector.
    /// @param FX            Output constraint-residual vector.
    /// @param AGX           Output adjoint-gradient vector.
    /// @param KKTmat        The KKT matrix to fill.
    /// @param KKTLocations  Per-element KKT storage locations.
    /// @param KKTClashes    Per-element KKT clash counts.
    /// @param KKTLocks      Per-element KKT write locks.
    /// @param data          Solver indexing data for this function.
    /// @endinternal
    void constraints_jacobian_adjointgradient_adjointhessian_test(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {

        Input<double> x(this->input_rows());
        Output<double> l(this->output_rows());

        Eigen::Map<Output<double>> fx(NULL, this->output_rows());
        Eigen::Map<Input<double>> agx(NULL, this->input_rows());

        std::vector<Input<double>> X1X2s;
        std::vector<Output<double>> Lfs;

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            this->gather_mult(L, l, V, data);

            X1X2s.push_back(x);
            Lfs.push_back(l);
        }

        auto [fxs, jxs, hxs] = this->compute_all_impl_v(X1X2s, Lfs);

        for (int V = 0; V < data.num_appl(); V++) {

            new (&fx) Eigen::Map<Output<double>>(FX.data() + data.inner_constraint_starts_[V],
                                                 this->output_rows());
            new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.inner_gradient_starts_[V],
                                                 this->input_rows());

            fx = fxs[V];
            agx = jxs[V].transpose() * Lfs[V];
            this->derived().kkt_fill_all(V, jxs[V], hxs[V], KKTmat, KKTLocations, KKTClashes,
                                         KKTLocks, data);
        }
    }
};

} // namespace tycho::oc

// Both shooting defects are plain function objects, so they are stored as
// themselves when converted to a solver constraint — one erasure, one virtual
// dispatch per solver call. The registrations sit beside the definitions
// because that is the only placement no translation unit can miss; see
// hven/solver_interface_adapter.h for the contract.
namespace hven::solvers {

template <class DODE, class Integrator>
struct SolverInterfaceAdapter<tycho::oc::ShootingDefect<DODE, Integrator>>
    : ::tycho::solvers::DirectVFAdapter<tycho::oc::ShootingDefect<DODE, Integrator>> {};

template <class DODE, class Integrator>
struct SolverInterfaceAdapter<tycho::oc::CentralShootingDefect<DODE, Integrator>>
    : ::tycho::solvers::DirectVFAdapter<tycho::oc::CentralShootingDefect<DODE, Integrator>> {};

} // namespace hven::solvers
