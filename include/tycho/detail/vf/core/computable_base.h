// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Implements Base class of all dense and sparse expressions with a compute method.
// Template parameters are the derived class (Derived), and the compile time value of the input (IR)
// and output (OR) rows of the vectorfunction.
//
// Provides derived() cast directly (CRTP pattern).
// Inherits from InputOutputSize<IR, OR> to gain fields for holding input and output sizes if
// necessary for dynamic sized functions (IR=OR=-1).
//
// Defines the default set of constexpr boolean info about functions that are used by the expression
// system, such as is_vectorizable. Derived classes are intended to override these as necessary.
//
//
// Adds functions for getting and setting the input and output rows.
//
// Defines the .compute function in terms of derived().compute_impl which must be implemented by a
// derived class.
//
// Also defines implements the constraints function in terms of the compute function. The
// constraints functions is part of a vector functions interface to the non-linear optimizer PSIOPT.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include <concepts>

#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/vf/core/eigen_ref_aliases.h"
#include "tycho/detail/vf/core/functional_flags.h"
#include "tycho/detail/vf/core/input_output_size.h"
#include "tycho/detail/vf/derivatives/detect_super_scalar.h"
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

#include "tycho/detail/utils/memory_management.h"

namespace tycho::vf {

/// @brief CRTP base for anything exposing a `compute` function.
///
/// `ComputableBase` is the root of the VectorFunction evaluation hierarchy. It
/// supplies the CRTP `derived()` cast, holds (for dynamic-sized functions) the
/// runtime input/output sizes via @ref InputOutputSize, defines the default set
/// of compile-time `constexpr` trait flags used by the expression system (e.g.
/// @ref is_vectorizable), and implements `compute` / `adjointgradient` in terms
/// of the derived class's `compute_impl`. It also provides the `constraints`
/// interface by which a VectorFunction is invoked by the PSIOPT optimizer.
///
/// @tparam Derived  CRTP derived (user) function type.
/// @tparam IR       Input rows at compile time (`-1` for dynamic).
/// @tparam OR       Output rows at compile time (`-1` for dynamic).
/// @ingroup vf
template <class Derived, int IR, int OR> struct ComputableBase : InputOutputSize<IR, OR> {
    /// @brief Downcast to the CRTP derived type.
    /// @return Mutable reference to the derived function.
    Derived &derived() { return static_cast<Derived &>(*this); }
    /// @brief Downcast to the CRTP derived type (const overload).
    /// @return Const reference to the derived function.
    const Derived &derived() const { return static_cast<const Derived &>(*this); }
    /// @brief Demangled type name of the derived function.
    /// @return Human-readable name of @p Derived.
    std::string name() const { return type_name<Derived>(); }

    ///////////////////////TypeDefs////////////////////////////////////////////
    /// @brief Output column-vector type of length @ref ORC.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Output = Eigen::Matrix<Scalar, OR, 1>;
    /// @brief Input column-vector type of length @ref IRC.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Input = Eigen::Matrix<Scalar, IR, 1>;
    /// @brief Gradient column-vector type of length @ref IRC.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Gradient = Eigen::Matrix<Scalar, IR, 1>;

    /// @brief Input rows at compile time (`-1` if dynamic).
    static constexpr int IRC = IR;
    /// @brief Output rows at compile time (`-1` if dynamic).
    static constexpr int ORC = OR;

    static constexpr bool InputIsDynamic = (IR < 0); ///< @brief True if the input size is dynamic.
    static constexpr bool OutputIsDynamic =
        (OR < 0); ///< @brief True if the output size is dynamic.
    static constexpr bool JacobianIsDynamic =
        (IR < 0 || OR < 0); ///< @brief True if either Jacobian dimension is dynamic.
    static constexpr bool FullyDynamic =
        (IR < 0 && OR < 0); ///< @brief True if both input and output sizes are dynamic.

    static constexpr bool is_vectorizable =
        false; ///< @brief Whether SuperScalar (SIMD pack) evaluation is supported.
    static constexpr bool is_linear_function =
        false; ///< @brief Whether the function is linear (zero Hessian).
    static constexpr bool has_diagonal_jacobian =
        false; ///< @brief Whether the Jacobian is diagonal.
    static constexpr bool has_diagonal_hessian = false; ///< @brief Whether the Hessian is diagonal.
    static constexpr bool is_cwise_operator =
        false; ///< @brief Whether this is a coefficient-wise operator.
    static constexpr bool is_generic_function =
        false; ///< @brief Whether this is a type-erased generic function.
    static constexpr bool is_conditional =
        false; ///< @brief Whether evaluation is conditional/branching.

    mutable bool enable_vectorization_ =
        false; ///< @brief Runtime flag toggling vectorized solver-interface paths.

    /// @brief Enable or disable vectorized evaluation at runtime.
    /// @param b  `true` to enable SuperScalar packed evaluation.
    void enable_vectorization(bool b) const { this->enable_vectorization_ = b; }

    /// @brief Whether the derived function is linear.
    /// @return `Derived::is_linear_function`.
    [[nodiscard]] constexpr bool is_linear() const { return Derived::is_linear_function; }

    /// @brief Set the runtime input row count (only for dynamic-input functions).
    /// @param inputrows  Number of input rows.
    void set_input_rows(int inputrows) {
        if constexpr (IR < 0) {
            this->input_rows_val = inputrows;
        }
    }
    /// @brief Set the runtime output row count (only for dynamic-output functions).
    /// @param outputrows  Number of output rows.
    void set_output_rows(int outputrows) {
        if constexpr (OR < 0) {
            this->output_rows_val = outputrows;
        }
    }
    /// @brief Set both input and output row counts.
    /// @param inputrows   Number of input rows.
    /// @param outputrows  Number of output rows.
    void set_io_rows(int inputrows, int outputrows) {
        this->set_input_rows(inputrows);
        this->set_output_rows(outputrows);
    }
    /// @brief Number of input rows (compile-time value if fixed, else runtime value).
    /// @return The input row count.
    [[nodiscard]] inline int input_rows() const {
        if constexpr (IR < 0) {
            return this->input_rows_val;
        } else {
            return IR;
        }
    }
    /// @brief Number of output rows (compile-time value if fixed, else runtime value).
    /// @return The output row count.
    [[nodiscard]] inline int output_rows() const {
        if constexpr (OR < 0) {
            return this->output_rows_val;
        } else {
            return OR;
        }
    }

    /// @brief Whether this function is safe to evaluate concurrently from multiple threads.
    /// @return `true` for the base class; derived classes may override.
    [[nodiscard]] bool thread_safe() const { return true; }
    //////////////////////////////////////////////////////////////////////////////
    /// @brief Default-construct (sizes taken from compile-time template parameters).
    ComputableBase() {}
    /// @brief Construct with explicit runtime input/output sizes.
    /// @param inputrows   Number of input rows.
    /// @param outputrows  Number of output rows.
    ComputableBase(int inputrows, int outputrows) { this->set_io_rows(inputrows, outputrows); }

    /// @brief Evaluate the function: `fx = f(x)`.
    ///
    /// CRTP entry point that dispatches to `Derived::compute_impl`. When invoked
    /// with a SuperScalar (`Eigen::Array`) pack but the derived function is not
    /// vectorizable, this scalarizes the call lane-by-lane.
    ///
    /// @tparam InType   Eigen column-vector type of the input @p x.
    /// @tparam OutType  Eigen column-vector type of the output @p fx_.
    /// @param x    Input vector, shape (IR,).
    /// @param fx_  Output vector, shape (OR,); overwritten with `f(x)`.
    template <class InType, class OutType>
    inline void compute(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        if constexpr (!Vectorizable<Derived>) {
            if constexpr (IsSuperScalar<Scalar>) {
                VecRef<OutType> fx = fx_.const_cast_derived();

                typedef typename Scalar::Scalar RealScalar;

                Input<RealScalar> x_r;
                Output<RealScalar> fx_r;

                const int IRR = this->input_rows();
                const int ORR = this->output_rows();

                if constexpr (InputIsDynamic)
                    x_r.resize(IRR);
                if constexpr (OutputIsDynamic)
                    fx_r.resize(ORR);

                for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                    for (int j = 0; j < IRR; j++)
                        x_r[j] = x[j][i];
                    this->derived().compute_impl(x_r, fx_r);
                    for (int j = 0; j < ORR; j++)
                        fx[j][i] = fx_r[j];
                    fx_r.setZero();
                }

            } else {
                this->derived().compute_impl(x, fx_);
            }
        } else {
            this->derived().compute_impl(x, fx_);
        }
    }

    /// @brief Evaluate the function and return the result by value.
    ///
    /// @tparam InType  Eigen column-vector type of the input @p x.
    /// @param x  Input vector, shape (IR,).
    /// @return The output vector `f(x)`, shape (OR,).
    template <class InType>
    inline Output<typename InType::Scalar> compute(CVecRef<InType> x) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx(this->output_rows());
        fx.setZero();
        this->derived().compute(x, fx);
        return fx;
    }

    /// @brief Evaluate the function and the adjoint (vector-Jacobian) gradient.
    ///
    /// Computes `fx = f(x)` and `adjgrad = J(x)^T * adjvars`.
    ///
    /// @tparam InType       Eigen type of the input @p x.
    /// @tparam OutType      Eigen type of the output @p fx_.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient @p adjgrad_.
    /// @tparam AdjVarType   Eigen type of the adjoint (multiplier) vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param fx_      Output vector, shape (OR,); overwritten with `f(x)`.
    /// @param adjgrad_ Output adjoint gradient, shape (IR,).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    template <class InType, class OutType, class AdjGradType, class AdjVarType>
    inline void compute_adjointgradient(CVecRef<InType> x, CVecRef<OutType> fx_,
                                        CVecRef<AdjGradType> adjgrad_,
                                        CVecRef<AdjVarType> adjvars) const {
        this->derived().compute_adjointgradient(x, fx_, adjgrad_, adjvars);
    }

    /// @brief Compute the adjoint gradient (function value discarded).
    ///
    /// @tparam InType       Eigen type of the input @p x.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient @p adjgrad_.
    /// @tparam AdjVarType   Eigen type of the adjoint (multiplier) vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param adjgrad_ Output adjoint gradient, shape (IR,).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    template <class InType, class AdjGradType, class AdjVarType>
    inline void adjointgradient(CVecRef<InType> x, CVecRef<AdjGradType> adjgrad_,
                                CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx(this->output_rows());
        fx.setZero();
        this->derived().compute_adjointgradient(x, fx, adjgrad_, adjvars);
    }

    /// @brief Compute and return the adjoint gradient by value.
    ///
    /// @tparam InType      Eigen type of the input @p x.
    /// @tparam AdjVarType  Eigen type of the adjoint (multiplier) vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    /// @return The adjoint gradient `J(x)^T * adjvars`, shape (IR,).
    template <class InType, class AdjVarType>
    inline Gradient<typename InType::Scalar> adjointgradient(CVecRef<InType> x,
                                                             CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Gradient<Scalar> adjgrad(this->input_rows());
        adjgrad.setZero();
        this->derived().adjointgradient(x, adjgrad, adjvars);
        return adjgrad;
    }

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /// @brief PSIOPT interface: evaluate this function as a problem constraint.
    ///
    /// Evaluates the function once per application listed in @p data, scattering
    /// the results into the global constraint vector @p FX. @p X is the full
    /// optimization variable vector; @p data holds, for each call, the locations
    /// of the input variables within @p X and the target rows within @p FX.
    /// When the derived function is vectorizable and vectorization is enabled,
    /// applications are processed in SuperScalar packs with a scalar tail.
    ///
    /// @param X     Full optimization variable vector.
    /// @param FX    Full equality/inequality constraint vector (written into).
    /// @param data  Solver indexing metadata for each application.
    /// @note Users should not normally override this method.
    void constraints(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                     const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());

            const int IRR = this->input_rows();
            const int ORR = this->output_rows();

            // Scalar non-vectorized call to the function
            auto ScalarImpl = [&](int start, int stop) {
                for (int V = start; V < stop; V++) {
                    this->gather_input(X, x, V, data);
                    new (&fx) Eigen::Map<Output<double>>(
                        FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                    fx.setZero();
                    this->derived().compute(x, fx);
                }
            };

            /*
            Super Scalar vectorized call to the function.
            Does as many fully packed vectorized calls as possible
            then reverts to the scalar implementation.
            */

            auto VectorImpl = [&]() {
                using SuperScalar = tycho::DefaultSuperScalar;
                constexpr int vsize = SuperScalar::SizeAtCompileTime;
                int Packs = data.num_appl() / vsize;

                Input<SuperScalar> x_vect(this->input_rows());
                Output<SuperScalar> fx_vect(this->output_rows());

                for (int i = 0; i < Packs; i++) {
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        this->gather_input(X, x, V, data);
                        for (int k = 0; k < IRR; k++) {
                            x_vect[k][j] = x[k];
                        }
                    }
                    fx_vect.setZero();
                    this->derived().compute(x_vect, fx_vect);
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        new (&fx) Eigen::Map<Output<double>>(
                            FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                        for (int l = 0; l < ORR; l++) {
                            fx[l] = fx_vect[l][j];
                        }
                    }
                }
                ScalarImpl(Packs * vsize, data.num_appl());
            };

            // Only try vectorized impl if Derived allows and it is enabled
            if constexpr (Vectorizable<Derived>) {
                if (this->derived().enable_vectorization_) {
                    VectorImpl();
                } else {
                    ScalarImpl(0, data.num_appl());
                }
            } else {
                ScalarImpl(0, data.num_appl());
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1));
    }

    /// @brief PSIOPT interface: evaluate constraint values and their adjoint gradients.
    ///
    /// For each application in @p data, gathers the local input and multiplier
    /// vectors from @p X and @p L, evaluates `f(x)` and the adjoint gradient
    /// `J(x)^T * l`, and scatters them into @p FX and @p AGX respectively.
    ///
    /// @param X     Full optimization variable vector.
    /// @param L     Full Lagrange-multiplier vector.
    /// @param FX    Full constraint vector (written into).
    /// @param AGX   Full adjoint-gradient vector (written into).
    /// @param data  Solver indexing metadata for each application.
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &l) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());
            Eigen::Map<Input<double>> agx(NULL, this->input_rows());

            for (int V = 0; V < data.num_appl(); V++) {
                this->gather_input(X, x, V, data);
                this->gather_mult(L, l, V, data);
                new (&fx) Eigen::Map<Output<double>>(FX.data() + data.inner_constraint_starts_[V],
                                                     this->output_rows());
                new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.inner_gradient_starts_[V],
                                                     this->input_rows());
                fx.setZero();
                agx.setZero();
                this->derived().compute_adjointgradient(x, fx, agx, l);
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Output<double>>(this->output_rows(), 1));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////

  protected:
    /// @brief Gather the input variables for the @p V-th application into @p xt.
    /// @internal
    /// Copies the input variables for application @p V out of the global variable
    /// vector @p X. When @p data marks the variables as contiguous they are read
    /// with a single segment copy; otherwise they are gathered element-by-element.
    /// Specialized on the compile-time input size for a modest perf gain.
    ///
    /// @tparam XtType  Eigen vector type of the local input buffer @p xt.
    /// @param X     Full optimization variable vector.
    /// @param xt    Local input buffer (written into).
    /// @param V     Application index.
    /// @param data  Solver indexing metadata.
    /// @endinternal
    template <class XtType>
        requires std::derived_from<std::remove_cvref_t<XtType>,
                                   Eigen::EigenBase<std::remove_cvref_t<XtType>>>
    inline void gather_input(ConstEigenRef<Eigen::VectorXd> X, XtType &xt, int V,
                             const tycho::solvers::SolverIndexingData &data) const {
        if (data.v_index_continuity_[V] == ParsedIOFlags::Contiguous) {
            xt = X.template segment<IR>(data.v_loc(0, V), this->input_rows());
        } else {
            for (int i = 0; i < this->input_rows(); i++)
                xt(i) = X(data.v_loc(i, V));
        }
    }

    /// @brief Gather the Lagrange multipliers for the @p V-th application into @p l.
    /// @internal
    /// Copies the multipliers for application @p V out of the global multiplier
    /// vector @p L, using a single segment copy when contiguous. See @ref
    /// gather_input for the rationale.
    ///
    /// @tparam LType  Eigen vector type of the local multiplier buffer @p l.
    /// @param L     Full Lagrange-multiplier vector.
    /// @param l     Local multiplier buffer (written into).
    /// @param V     Application index.
    /// @param data  Solver indexing metadata.
    /// @endinternal
    template <class LType>
        requires std::derived_from<std::remove_cvref_t<LType>,
                                   Eigen::EigenBase<std::remove_cvref_t<LType>>>
    inline void gather_mult(ConstEigenRef<Eigen::VectorXd> L, LType &l, int V,
                            const tycho::solvers::SolverIndexingData &data) const {
        if (data.c_index_continuity_[V] == ParsedIOFlags::Contiguous) {
            l = L.template segment<OR>(data.c_loc(0, V), this->output_rows());
        } else {
            for (int i = 0; i < this->output_rows(); i++)
                l(i) = L(data.c_loc(i, V));
        }
    }
};

} // namespace tycho::vf
