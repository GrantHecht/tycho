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

#include "tycho/detail/integrators/integrator.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
#include "tycho/detail/optimal_control/transcription/blocked_ode_wrapper.h"
#include "tycho/detail/optimal_control/transcription/lgl_defects.h"
#include "tycho/detail/optimal_control/transcription/shooting_defects.h"
#include "tycho/detail/optimal_control/transcription/trapezoidal_defects.h"
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

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using integrators::Integrator;
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::Arguments;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::IOScaled;
using vf::RowScaled;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

/// @ingroup optimal_control
/// @brief A statically-typed optimal control phase for a concrete ODE.
///
/// The typed phase the user constructs from an ODE. It owns the ODE and a
/// scaled copy of it, a reference integrator, and (via @ref ODEPhaseBase) the
/// constraints, objectives, mesh, and solver. It supplies the transcription
/// machinery (defect functions, mesh-error estimators, re-integrators)
/// specialized to the ODE's compile-time dimensions.
/// @tparam DODE  The concrete ODE type this phase discretizes.
template <class DODE> struct ODEPhase : ODEPhaseBase {
    using VectorXi = Eigen::VectorXi; ///< @brief Convenience alias for an integer vector.
    using MatrixXi = Eigen::MatrixXi; ///< @brief Convenience alias for an integer matrix.

    using VectorXd = Eigen::VectorXd; ///< @brief Convenience alias for a double vector.
    using MatrixXd = Eigen::MatrixXd; ///< @brief Convenience alias for a double matrix.

    using VectorFunctionalX =
        GenericFunction<-1, -1>; ///< @brief Type-erased vector-valued function.
    using ScalarFunctionalX =
        GenericFunction<-1, 1>; ///< @brief Type-erased scalar-valued function.

    using StateConstraint =
        StateFunction<VectorFunctionalX>; ///< @brief A constraint bound to a region.
    using StateObjective =
        StateFunction<ScalarFunctionalX>; ///< @brief An objective bound to a region.

    /// @brief Convenience alias for an LGL defect function of the given cardinal-state count.
    /// @tparam ODET  The ODE type the defect wraps.
    /// @tparam CS    Number of cardinal states per defect interval.
    template <class ODET, int CS> using LGLType = LGLDefects<ODET, CS>;

    /// @brief The ODE's input (state/time/control/param) vector type.
    /// @tparam Scalar  The arithmetic scalar type.
    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    /// @brief The ODE's output (state-derivative) vector type.
    /// @tparam Scalar  The arithmetic scalar type.
    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;

    /// @brief Type-erased, IO-scaled copy of the ODE used under auto-scaling.
    using ScaledODE = GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>;

    DODE ode_;             ///< The concrete ODE this phase discretizes.
    ScaledODE ode_scaled_; ///< IO-scaled copy of the ODE for auto-scaling.

    Integrator<DODE> integrator_;          ///< Reference integrator for the ODE.
    bool enable_hessian_sparsity_ = false; ///< Whether to exploit Hessian sparsity in defects.
    bool old_shooting_defect_ = false;     ///< Use the legacy shooting-defect formulation.

    /// @brief Construct a phase from an ODE and a transcription mode.
    /// @param ode    The ODE to discretize.
    /// @param Tmode  The transcription mode.
    ODEPhase(const DODE &ode, TranscriptionModes Tmode)
        : ODEPhaseBase(ode.x_vars(), ode.u_vars(), ode.p_vars()) {
        this->ode_ = ode;
        this->integrator_ = Integrator<DODE>(ode, .01);
        this->set_idxs(this->ode_.get_idxs());
        this->set_transcription_mode(Tmode);
        this->set_units(Eigen::VectorXd::Ones(this->xtu_p_vars()));
    }
    /// @brief Construct a phase and set an initial trajectory on a uniform mesh.
    /// @param ode     The ODE to discretize.
    /// @param Tmode   The transcription mode.
    /// @param Traj    Initial trajectory (one packed state per node).
    /// @param numdef  Number of defect intervals.
    ODEPhase(const DODE &ode, TranscriptionModes Tmode, const std::vector<Eigen::VectorXd> &Traj,
             int numdef)
        : ODEPhase(ode, Tmode) {
        this->set_traj(Traj, numdef);
    }
    /// @brief Construct a phase and set an initial trajectory, optionally interpolating.
    /// @param ode     The ODE to discretize.
    /// @param Tmode   The transcription mode.
    /// @param Traj    Initial trajectory (one packed state per node).
    /// @param numdef  Number of defect intervals.
    /// @param LerpIG  Whether to linearly interpolate the trajectory onto the mesh.
    ODEPhase(const DODE &ode, TranscriptionModes Tmode, const std::vector<Eigen::VectorXd> &Traj,
             int numdef, bool LerpIG)
        : ODEPhase(ode, Tmode) {
        this->set_traj(Traj, numdef, LerpIG);
    }
    /// @brief Construct a phase and set an initial trajectory, inferring the mesh.
    /// @param ode    The ODE to discretize.
    /// @param Tmode  The transcription mode.
    /// @param Traj   Initial trajectory (one packed state per node).
    ODEPhase(const DODE &ode, TranscriptionModes Tmode, const std::vector<Eigen::VectorXd> &Traj)
        : ODEPhase(ode, Tmode) {
        this->set_traj(Traj);
    }

    /// @brief Construct a phase from an ODE and a transcription-mode name.
    /// @param ode    The ODE to discretize.
    /// @param Tmode  The transcription-mode name (see @c strto_TranscriptionMode).
    ODEPhase(const DODE &ode, std::string Tmode) : ODEPhase(ode, strto_TranscriptionMode(Tmode)) {}
    /// @brief Construct a phase by mode name and set a trajectory on a uniform mesh.
    /// @param ode     The ODE to discretize.
    /// @param Tmode   The transcription-mode name.
    /// @param Traj    Initial trajectory (one packed state per node).
    /// @param numdef  Number of defect intervals.
    ODEPhase(const DODE &ode, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj,
             int numdef)
        : ODEPhase(ode, strto_TranscriptionMode(Tmode), Traj, numdef) {}
    /// @brief Construct a phase by mode name and set a trajectory (mesh inferred).
    /// @param ode    The ODE to discretize.
    /// @param Tmode  The transcription-mode name.
    /// @param Traj   Initial trajectory (one packed state per node).
    ODEPhase(const DODE &ode, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj)
        : ODEPhase(ode, strto_TranscriptionMode(Tmode), Traj) {}
    /// @brief Construct a phase by mode name and set a trajectory, optionally interpolating.
    /// @param ode     The ODE to discretize.
    /// @param Tmode   The transcription-mode name.
    /// @param Traj    Initial trajectory (one packed state per node).
    /// @param numdef  Number of defect intervals.
    /// @param LerpIG  Whether to linearly interpolate the trajectory onto the mesh.
    ODEPhase(const DODE &ode, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj,
             int numdef, bool LerpIG)
        : ODEPhase(ode, strto_TranscriptionMode(Tmode), Traj, numdef, LerpIG) {}

    /// @brief Set the scaling units and rebuild the scaled ODE used for auto-scaling.
    /// @param xtup_units  Per-variable scaling units for the state/time/control/ODE-param block.
    /// @throws std::invalid_argument if @p xtup_units has the wrong size.
    void set_units(const Eigen::VectorXd &xtup_units) {

        if (xtup_units.size() != this->xtu_p_vars()) {
            throw std::invalid_argument("Incorrect size for input units vector");
        }

        this->xtup_units_ = xtup_units;
        VectorXd output_scales = this->xtup_units_.head(this->x_vars()).cwiseInverse() *
                                 this->xtup_units_[this->x_vars()];
        VectorFunctionalX odetemp;

        if constexpr (DODE::IsGenericODE) {
            odetemp = this->ode_.func_;
        } else {
            odetemp = this->ode_;
        }

        auto tmp = IOScaled<VectorFunctionalX>(odetemp, this->xtup_units_, output_scales);

        this->ode_scaled_ = ScaledODE(tmp, this->x_vars(), this->u_vars(), this->p_vars());
    }

    /// @internal
    /// @brief Build a re-integrator over the scaled ODE, wiring in the control interpolant.
    /// @return An integrator over the scaled ODE configured to match this phase.
    /// @endinternal
    virtual Integrator<ScaledODE> make_scaled_reintegrator() const {
        Integrator<ScaledODE> Integ;
        if (this->u_vars() == 0 || this->control_mode_ == ControlModes::BlockConstant) {
            Integ = Integrator<ScaledODE>{this->ode_scaled_, this->integrator_.def_step_size_};
        } else {
            Eigen::VectorXi ulocs;
            ulocs.setLinSpaced(this->u_vars(), this->t_var() + 1, this->t_var() + this->u_vars());

            Eigen::VectorXi varlocs(1);
            varlocs[0] = this->t_var();

            double tscale = this->xtup_units_[this->t_var()];
            auto tabcon = InterpFunction<-1>(std::make_shared<LGLInterpTable>(this->table_), ulocs)
                              .eval(Arguments<1>(1) * tscale);
            VectorXd Uscales =
                this->xtup_units_.segment(this->t_var() + 1, this->u_vars()).cwiseInverse();
            auto ucon = RowScaled<decltype(tabcon)>(tabcon, Uscales);
            Integ = Integrator<ScaledODE>{this->ode_scaled_, this->integrator_.def_step_size_, ucon,
                                          varlocs};
        }

        Integ.copy_settings_from(this->integrator_);
        Integ.enable_vectorization_ = this->enable_vectorization_;

        return Integ;
    }

    /// @internal
    /// @brief Build a re-integrator over the unscaled ODE, wiring in the control interpolant.
    /// @return An integrator over the ODE configured to match this phase.
    /// @endinternal
    virtual Integrator<DODE> make_reintegrator() const {
        Integrator<DODE> Integ;

        if (this->u_vars() == 0 || this->control_mode_ == ControlModes::BlockConstant) {
            Integ = Integrator<DODE>{this->ode_, this->integrator_.def_step_size_};
        } else {
            Integ = Integrator<DODE>{this->ode_, this->integrator_.def_step_size_,
                                     std::make_shared<LGLInterpTable>(this->table_)};
        }

        Integ.copy_settings_from(this->integrator_);
        Integ.enable_vectorization_ = this->enable_vectorization_;

        return Integ;
    }

    /// @brief Set the transcription scheme and rebuild the interpolation table.
    /// @param m  The transcription mode.
    /// @throws std::invalid_argument if @p m is not a valid transcription method.
    virtual void set_transcription_mode(TranscriptionModes m) {
        this->reset_transcription();

        this->transcription_mode_ = m;

        VectorFunctionalX odetemp;

        if constexpr (DODE::IsGenericODE) {
            odetemp = this->ode_.func_;
        } else {
            odetemp = this->ode_;
        }

        switch (this->transcription_mode_) {
        case TranscriptionModes::LGL7:
            this->table_ =
                LGLInterpTable(odetemp, this->x_vars(), this->u_vars() + this->p_vars(), m);
            this->order_ = 7.0;
            this->num_tran_card_states_ = 4;
            break;
        case TranscriptionModes::LGL5:
            this->table_ =
                LGLInterpTable(odetemp, this->x_vars(), this->u_vars() + this->p_vars(), m);
            this->order_ = 5.0;
            this->num_tran_card_states_ = 3;
            break;
        case TranscriptionModes::LGL3:
            this->table_ =
                LGLInterpTable(odetemp, this->x_vars(), this->u_vars() + this->p_vars(), m);

            this->order_ = 3.0;
            this->num_tran_card_states_ = 2;
            break;
        case TranscriptionModes::Trapezoidal:
            this->table_ = LGLInterpTable(odetemp, this->x_vars(), this->u_vars() + this->p_vars(),
                                          TranscriptionModes::LGL3);

            this->order_ = 2.0;
            this->num_tran_card_states_ = 2;
            break;
        case TranscriptionModes::CentralShooting:
            this->table_ = LGLInterpTable(odetemp, this->x_vars(), this->u_vars() + this->p_vars(),
                                          TranscriptionModes::LGL3);
            this->order_ = 7.0;
            this->num_tran_card_states_ = 2;
            // Default Central Shooting to BlockConstant!!!
            this->set_control_mode(ControlModes::BlockConstant);

            break;
        default: {
            throw std::invalid_argument("Invalid Transcription Method");
            break;
        }
        }
    }

    /// @internal
    /// @brief Build and register the ODE defect constraints for the active transcription.
    /// @throws std::invalid_argument if the transcription mode is invalid.
    /// @endinternal
    virtual void transcribe_dynamics() {
        VectorXi StateT(this->ode_.xtu_vars());
        for (int i = 0; i < this->ode_.xtu_vars(); i++)
            StateT[i] = i;
        VectorXi OParT(this->ode_.p_vars());
        for (int i = 0; i < this->ode_.p_vars(); i++)
            OParT[i] = i;
        VectorXi empty(0);
        empty.resize(0);

        auto lgldef = [&](auto cs, auto ode_t) {
            if (this->control_mode_ == ControlModes::BlockConstant) {

                if constexpr (DODE::UV == 0 && DODE::PV == 0) {
                    LGLType<decltype(ode_t), cs.value> lgl(ode_t);
                    lgl.enable_vectorization_ = this->enable_vectorization_;
                    this->dynamics_func_index_ =
                        this->indexer_.add_equality(lgl, PhaseRegionFlags::DefectPath, StateT,
                                                    OParT, empty, ThreadingFlags::ByApplication);
                } else {

                    using BlockedODE = Blocked_ODE_Wrapper<decltype(ode_t)>;

                    auto lgl = LGLDefects<BlockedODE, cs.value>(BlockedODE(ode_t));
                    lgl.enable_vectorization_ = this->enable_vectorization_;
                    this->dynamics_func_index_ =
                        this->indexer_.add_equality(lgl, PhaseRegionFlags::BlockDefectPath, StateT,
                                                    OParT, empty, ThreadingFlags::ByApplication);
                }
            } else {
                LGLType<decltype(ode_t), cs.value> lgl(ode_t);
                lgl.enable_vectorization_ = this->enable_vectorization_;
                this->dynamics_func_index_ =
                    this->indexer_.add_equality(lgl, PhaseRegionFlags::DefectPath, StateT, OParT,
                                                empty, ThreadingFlags::ByApplication);
            }
        };

        auto trapdef = [&](auto ode_t) {
            if (this->control_mode_ == ControlModes::BlockConstant) {

                if constexpr (DODE::UV == 0 && DODE::PV == 0) {
                    TrapezoidalDefects<decltype(ode_t)> trap(ode_t);
                    trap.enable_vectorization_ = this->enable_vectorization_;
                    this->dynamics_func_index_ =
                        this->indexer_.add_equality(trap, PhaseRegionFlags::DefectPath, StateT,
                                                    OParT, empty, ThreadingFlags::ByApplication);
                } else {
                    using BlockedODE = Blocked_ODE_Wrapper<decltype(ode_t)>;

                    auto trap = TrapezoidalDefects<BlockedODE>(BlockedODE(ode_t));
                    trap.enable_vectorization_ = this->enable_vectorization_;
                    // trap.enable_hessian_sparsity_ = this->enable_hessian_sparsity_;
                    this->dynamics_func_index_ =
                        this->indexer_.add_equality(trap, PhaseRegionFlags::BlockDefectPath, StateT,
                                                    OParT, empty, ThreadingFlags::ByApplication);
                }

            } else {
                TrapezoidalDefects<decltype(ode_t)> trap(ode_t);
                trap.enable_vectorization_ = this->enable_vectorization_;
                this->dynamics_func_index_ =
                    this->indexer_.add_equality(trap, PhaseRegionFlags::DefectPath, StateT, OParT,
                                                empty, ThreadingFlags::ByApplication);
            }
        };

        switch (this->transcription_mode_) {
        case TranscriptionModes::LGL7: {

            if (this->auto_scaling_) {
                lgldef(int_const<4>(), this->ode_scaled_);
            } else {
                lgldef(int_const<4>(), this->ode_);
            }

            break;
        }
        case TranscriptionModes::LGL5: {
            if (this->auto_scaling_) {
                lgldef(int_const<3>(), this->ode_scaled_);
            } else {
                lgldef(int_const<3>(), this->ode_);
            }
            break;
        }
        case TranscriptionModes::LGL3: {
            if (this->auto_scaling_) {
                lgldef(int_const<2>(), this->ode_scaled_);
            } else {
                lgldef(int_const<2>(), this->ode_);
            }
            break;
        }
        case TranscriptionModes::Trapezoidal: {
            if (this->auto_scaling_) {
                trapdef(this->ode_scaled_);
            } else {
                trapdef(this->ode_);
            }
            break;
        }
        case TranscriptionModes::CentralShooting: {

            auto shooter = this->make_shooter();
            this->dynamics_func_index_ =
                this->indexer_.add_equality(shooter, PhaseRegionFlags::DefectPath, StateT, OParT,
                                            empty, ThreadingFlags::ByApplication);

            break;
        }
        default:
            throw std::invalid_argument("Invalid Transcription Method");
        }
    }

    /// @internal
    /// @brief Build the central-shooting defect constraint for the active configuration.
    /// @return A type-erased constraint implementing the shooting defect.
    /// @endinternal
    virtual tycho::solvers::ConstraintInterface make_shooter() {

        if (this->auto_scaling_) {
            auto Integ = Integrator<ScaledODE>{this->ode_scaled_, this->integrator_.def_step_size_};
            Integ.copy_settings_from(this->integrator_);
            Integ.enable_vectorization_ = this->enable_vectorization_;

            if (old_shooting_defect_) {
                auto shooter = ShootingDefect{this->ode_scaled_, Integ};
                shooter.enable_hessian_sparsity_ = this->enable_hessian_sparsity_;
                return tycho::solvers::ConstraintInterface(shooter);
            } else {
                auto shooter = CentralShootingDefect{this->ode_scaled_, Integ};
                shooter.enable_hessian_sparsity_ = this->enable_hessian_sparsity_;
                shooter.enable_vectorization_ = this->enable_vectorization_;
                return tycho::solvers::ConstraintInterface(shooter);
            }
        }
        {

            auto Integ = Integrator<DODE>{this->ode_, this->integrator_.def_step_size_};
            Integ.copy_settings_from(this->integrator_);
            Integ.enable_vectorization_ = this->enable_vectorization_;

            if (old_shooting_defect_) {
                auto shooter = ShootingDefect{this->ode_, Integ};
                shooter.enable_hessian_sparsity_ = this->enable_hessian_sparsity_;
                return tycho::solvers::ConstraintInterface(shooter);
            } else {
                auto shooter = CentralShootingDefect{this->ode_, Integ};
                shooter.enable_hessian_sparsity_ = this->enable_hessian_sparsity_;
                shooter.enable_vectorization_ = this->enable_vectorization_;
                return tycho::solvers::ConstraintInterface(shooter);
            }
        }
    }

    /// @internal
    /// @brief Compute the end-to-end error by re-integrating the trajectory through the mesh.
    /// @return Per-state absolute end-to-end error.
    /// @endinternal
    virtual Eigen::VectorXd calc_global_error() const {

        auto CalcError = [&](auto &Integ, const auto &Traj) {
            ODEState<double> Xin;
            ODEState<double> Xout;

            double T0 = Traj[0][this->t_var()];
            double TF = Traj.back()[this->t_var()];

            int BlockSize = this->num_tran_card_states_;
            int num_blocks = (Traj.size() - 1) / (BlockSize - 1);

            Xin = Traj[0];

            for (int i = 0; i < num_blocks; i++) {
                int start = (BlockSize - 1) * i;
                int stop = (BlockSize - 1) * (i + 1);

                double tf = Traj[stop][this->t_var()];
                Xout = Integ.integrate(Xin, tf);
                Xin = Xout;
            }

            Eigen::VectorXd gerror = (Traj.back() - Xout).head(this->x_vars()).cwiseAbs();
            return gerror;
        };

        if (this->auto_scaling_) {
            Integrator<ScaledODE> Integ = this->make_scaled_reintegrator();
            auto ActiveTrajTmp = this->active_traj_;
            for (auto &T : ActiveTrajTmp) {
                T = T.cwiseQuotient(this->xtup_units_);
            }
            return CalcError(Integ, ActiveTrajTmp);

        } else {
            Integrator<DODE> Integ = this->make_reintegrator();
            return CalcError(Integ, this->active_traj_);
        }
    }

    /// @internal
    /// @brief Estimate per-segment mesh error/distribution via the de Boor method.
    /// @param tsnd         Output non-dimensional node times.
    /// @param mesh_errors  Output per-segment error matrix.
    /// @param mesh_dist    Output per-segment error-distribution matrix.
    /// @endinternal
    virtual void get_meshinfo_deboor(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                                     Eigen::MatrixXd &mesh_dist) const {

        double T0 = this->active_traj_[0][this->t_var()];
        double TF = this->active_traj_.back()[this->t_var()];

        int BlockSize = this->num_tran_card_states_;
        int num_blocks = (this->active_traj_.size() - 1) / (BlockSize - 1);

        mesh_errors.resize(this->x_vars(), num_blocks + 1);
        mesh_dist.resize(this->x_vars(), num_blocks + 1);
        tsnd.resize(num_blocks + 1);

        Eigen::VectorXd XerrWeights(this->num_tran_card_states_);
        Eigen::VectorXd DXerrWeights(this->num_tran_card_states_);
        std::vector<ODEDeriv<double>> Derivs(this->active_traj_.size(),
                                             ODEDeriv<double>::Zero(this->x_vars()));

        for (int i = 0; i < this->active_traj_.size(); i++) {
            this->ode_.compute(this->active_traj_[i], Derivs[i]);
        }

        ////////////////////////////////////////////////////
        auto factorial = [](int n) {
            double fact = 1;
            for (int i = 1; i <= n; i++)
                fact = fact * i;
            return fact;
        };

        double PolyFact = factorial(this->order_);
        double error_weight;

        auto FillErrorInfo = [&](auto CardStates) {
            for (int i = 0; i < CardStates.value; i++) {
                XerrWeights[i] =
                    LGLCoeffs<CardStates.value>::Cardinal_XPower_Weights[i][0] * PolyFact;
                DXerrWeights[i] =
                    LGLCoeffs<CardStates.value>::Cardinal_DXPower_Weights[i][0] * PolyFact;
            }
            error_weight = LGLCoeffs<CardStates.value>::error_weight_;
        };

        switch (this->transcription_mode_) {
        case TranscriptionModes::LGL7:
            FillErrorInfo(int_const<4>());
            break;
        case TranscriptionModes::LGL5:
            FillErrorInfo(int_const<3>());
            break;
        case TranscriptionModes::LGL3:
            FillErrorInfo(int_const<2>());
            break;
        case TranscriptionModes::CentralShooting:
            FillErrorInfo(int_const<2>());
            break;
        case TranscriptionModes::Trapezoidal:
            error_weight = 1 / 12.0;
            XerrWeights.setZero();
            DXerrWeights[0] = -1;
            DXerrWeights[1] = 1;
            break;
        default: {
            throw std::invalid_argument("Invalid Transcription Method");
            break;
        }
        }
        ////////////////////////////////////////////////////
        std::vector<ODEDeriv<double>> yvecs(num_blocks);
        Eigen::VectorXd hs(num_blocks);

        for (int i = 0; i < num_blocks; i++) {
            int start = (BlockSize - 1) * i;

            hs[i] = this->active_traj_[start + (BlockSize - 1)][this->t_var()] -
                    this->active_traj_[start][this->t_var()];

            tsnd[i] = (this->active_traj_[start][this->t_var()] - T0) / (TF - T0);

            ODEDeriv<double> yvec(this->x_vars());
            yvec.setZero();
            double powh = std::pow(hs[i], this->order_);

            ODEDeriv<double> dtemp(this->x_vars());

            if (this->u_vars() != 0 && this->control_mode_ == ControlModes::BlockConstant) {
                ODEState<double> tmp = this->active_traj_[start + BlockSize - 1];
                tmp.segment(this->xt_vars(), this->u_vars()) =
                    this->active_traj_[start].segment(this->xt_vars(), this->u_vars());

                dtemp = Derivs[start + BlockSize - 1];
                Derivs[start + BlockSize - 1].setZero();
                this->ode_.compute(tmp, Derivs[start + BlockSize - 1]);
            }

            for (int j = 0; j < BlockSize; j++) {
                yvec += this->active_traj_[start + j].head(this->x_vars()) * XerrWeights[j] / powh;
                yvec += Derivs[start + j].head(this->x_vars()) * DXerrWeights[j] * hs[i] / powh;
            }

            if (this->u_vars() != 0 && this->control_mode_ == ControlModes::BlockConstant) {
                Derivs[start + BlockSize - 1] = dtemp;
            }

            yvecs[i] = yvec;
        }

        /////////////////////////
        if (this->auto_scaling_) {
            // All errors assessed in scaled units
            // yvecs has dims of X/t^Order
            for (int i = 0; i < num_blocks; i++) {
                yvecs[i] = (yvecs[i].cwiseQuotient(this->xtup_units_.head(this->x_vars()))).eval();
                yvecs[i] *= std::pow(this->xtup_units_[this->x_vars()], this->order_);
                hs[i] /= this->xtup_units_[this->x_vars()];
            }
        }
        ////////////////////////

        tsnd[num_blocks] = 1.0;

        for (int i = 0; i < num_blocks; i++) {
            ODEDeriv<double> err_tmp(this->x_vars());

            if (i > 0 && i < (num_blocks - 1)) {

                err_tmp = ((yvecs[i] - yvecs[i - 1]) / (hs[i] + hs[i - 1])).cwiseAbs() +
                          ((yvecs[i + 1] - yvecs[i]) / (hs[i] + hs[i + 1])).cwiseAbs();

            } else if (i == 0) {
                err_tmp = (2 * (yvecs[i] - yvecs[i + 1]) / (hs[i] + hs[i + 1])).cwiseAbs();
            } else {
                err_tmp = (2 * (yvecs[i] - yvecs[i - 1]) / (hs[i] + hs[i - 1])).cwiseAbs();
            }

            mesh_dist.col(i) = err_tmp.array().pow(1 / (this->order_ + 1));
            mesh_errors.col(i) =
                err_tmp * std::pow(std::abs(hs[i]), this->order_ + 1) * error_weight;
        }

        mesh_dist.col(num_blocks) = mesh_dist.col(num_blocks - 1);
        mesh_errors.col(num_blocks) = mesh_errors.col(num_blocks - 1);
    }

    /// @internal
    /// @brief Estimate per-segment mesh error/distribution via reference integration.
    /// @param tsnd         Output non-dimensional node times.
    /// @param mesh_errors  Output per-segment error matrix.
    /// @param mesh_dist    Output per-segment error-distribution matrix.
    /// @endinternal
    virtual void get_meshinfo_integrator(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                                         Eigen::MatrixXd &mesh_dist) const {

        auto CalcError = [&](auto &Integ, const auto &Traj) {
            double T0 = Traj[0][this->t_var()];
            double TF = Traj.back()[this->t_var()];

            int BlockSize = this->num_tran_card_states_;
            int num_blocks = (Traj.size() - 1) / (BlockSize - 1);

            mesh_errors.resize(this->x_vars(), num_blocks + 1);
            mesh_dist.resize(this->x_vars(), num_blocks + 1);
            tsnd.resize(num_blocks + 1);

            Eigen::MatrixXd tmp_mat(this->x_vars(), Traj.size() - 1);

            std::vector<ODEState<double>> Xins(Traj.size() - 1);
            Eigen::VectorXd tfs(Traj.size() - 1);

            for (int i = 0; i < Traj.size() - 1; i++) {
                Xins[i] = Traj[i];
                tfs[i] = Traj[i + 1][this->t_var()];
            }
            auto Xouts = Integ.integrate(Xins, tfs);

            for (int i = 0; i < Traj.size() - 1; i++) {
                tmp_mat.col(i) =
                    (Xouts[i].head(this->x_vars()) - Traj[i + 1].head(this->x_vars())).cwiseAbs();
            }

            double max_err = tmp_mat.maxCoeff();
            ODEDeriv<double> evec(this->x_vars());

            for (int i = 0; i < num_blocks; i++) {
                int start = (BlockSize - 1) * i;
                int stop = (BlockSize - 1) * (i + 1);

                double t0 = Traj[start][this->t_var()];
                double tf = Traj[stop][this->t_var()];

                tsnd[i] = (t0 - T0) / (TF - T0);

                evec.setZero();

                for (int j = 0; j < BlockSize - 1; j++) {
                    evec += tmp_mat.col(start + j) / (BlockSize - 1);
                }

                evec.setZero();

                for (int j = 0; j < BlockSize - 1; j++) {
                    double ti = Traj[start + j][this->t_var()];
                    double tn = Traj[start + j + 1][this->t_var()];

                    evec += tmp_mat.col(start + j) * std::abs((tn - ti) / (tf - t0));
                }

                double h = std::abs(tf - t0);
                mesh_errors.col(i) = evec;
                mesh_dist.col(i) = mesh_errors.col(i) / (std::pow(h, this->order_ + 1) * max_err);
                mesh_dist.col(i) = (mesh_dist.col(i).array().pow(1 / (this->order_ + 1))).eval();
            }

            mesh_errors.col(num_blocks) = mesh_errors.col(num_blocks - 1);
            mesh_dist.col(num_blocks) = mesh_dist.col(num_blocks - 1);
            tsnd[num_blocks] = 1.0;
        };

        if (this->auto_scaling_) {
            Integrator<ScaledODE> Integ = this->make_scaled_reintegrator();
            auto ActiveTrajTmp = this->active_traj_;
            for (auto &T : ActiveTrajTmp) {
                T = T.cwiseQuotient(this->xtup_units_);
            }
            return CalcError(Integ, ActiveTrajTmp);

        } else {
            Integrator<DODE> Integ = this->make_reintegrator();
            return CalcError(Integ, this->active_traj_);
        }
    }

    /// @brief Return the ODE defect function for the active transcription mode.
    /// @return A type-erased defect VectorFunction.
    /// @throws std::invalid_argument if the transcription mode is invalid.
    VectorFunctionalX get_defect() {
        VectorFunctionalX func;

        switch (this->transcription_mode_) {
        case TranscriptionModes::LGL7: {
            return func = LGLDefects<DODE, 4>(this->ode_);
            break;
        }
        case TranscriptionModes::LGL5: {
            return func = LGLDefects<DODE, 3>(this->ode_);
            break;
        }
        case TranscriptionModes::LGL3: {
            return func = LGLDefects<DODE, 2>(this->ode_);
            break;
        }
        case TranscriptionModes::Trapezoidal: {
            return func = LGLDefects<DODE, 2>(this->ode_);
            break;
        }
        case TranscriptionModes::CentralShooting: {
            return func = LGLDefects<DODE, 2>(this->ode_);
            break;
        }
        default:
            throw std::invalid_argument("Invalid Transcription Method");
        }

        return func;
    }
};

} // namespace tycho::oc
