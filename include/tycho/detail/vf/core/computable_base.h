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
// Inherits from CRTP to gain derived casting capabilities.
// Inherits from InputOutputSize<IR, OR> to gain fields for holding input and output sizes if necessary
// for dynamic sized functions (IR=OR=-1).
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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once
#include <concepts>

#include "tycho/detail/vf/derivatives/detect_super_scalar.h"
#include "tycho/detail/vf/core/functional_flags.h"
#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/vf/core/input_output_size.h"
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
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/crtp_base.h"

#include "tycho/detail/utils/memory_management.h"

namespace tycho::vf {

/*!
 * @brief A computable is anything with a \code compute \endcode function
 *
 * @tparam Derived CRTP Derived class
 * @tparam IR Input Rows
 * @tparam OR Output Rows
 */
template <class Derived, int IR, int OR>
struct ComputableBase : tycho::utils::CRTPBase<Derived>, InputOutputSize<IR, OR> {
    ///////////////////////TypeDefs////////////////////////////////////////////
    template <class Scalar> using Output = Eigen::Matrix<Scalar, OR, 1>;
    template <class Scalar> using Input = Eigen::Matrix<Scalar, IR, 1>;
    template <class Scalar> using Gradient = Eigen::Matrix<Scalar, IR, 1>;

    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;
    template <class Scalar> using VectorBaseRef = Eigen::MatrixBase<Scalar> &;

    /// Input Rows at Compile Time (-1 if Dynamic)
    static const int IRC = IR;
    /// Output Rows at Compile Time (-1 if Dynamic)
    static const int ORC = OR;

    static const bool InputIsDynamic = (IR < 0);
    static const bool OutputIsDynamic = (OR < 0);
    static const bool JacobianIsDynamic = (IR < 0 || OR < 0);
    static const bool FullyDynamic = (IR < 0 && OR < 0);

    static const bool is_vectorizable = false;
    static const bool is_linear_function = false;
    static const bool has_diagonal_jacobian = false;
    static const bool HasDiagonalHessian = false;
    static const bool IsCwiseOperator = false;
    static const bool is_generic_function = false;
    static const bool IsConditional = false;

    mutable bool EnableVectorization = false;

    void enable_vectorization(bool b) const { this->EnableVectorization = b; }

    [[nodiscard]] constexpr bool is_linear() const { return Derived::is_linear_function; }

    void set_input_rows(int inputrows) {
        if constexpr (IR < 0) {
            this->input_rows_val = inputrows;
        }
    }
    void set_output_rows(int outputrows) {
        if constexpr (OR < 0) {
            this->output_rows_val = outputrows;
        }
    }
    void set_io_rows(int inputrows, int outputrows) {
        this->set_input_rows(inputrows);
        this->set_output_rows(outputrows);
    }
    [[nodiscard]] inline int input_rows() const {
        if constexpr (IR < 0) {
            return this->input_rows_val;
        } else {
            return IR;
        }
    }
    [[nodiscard]] inline int output_rows() const {
        if constexpr (OR < 0) {
            return this->output_rows_val;
        } else {
            return OR;
        }
    }

    [[nodiscard]] bool thread_safe() const { return true; }
    //////////////////////////////////////////////////////////////////////////////
    ComputableBase() {}
    ComputableBase(int inputrows, int outputrows) { this->set_io_rows(inputrows, outputrows); }

    /*!
     * @brief Calls compute on the derived class.
     *
     * @tparam InType Eigen type of input vector
     * @tparam OutType Eigen type of output vector
     * @param x Input vector
     * @param fx_ Output vector
     */
    template <class InType, class OutType>
    inline void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        if constexpr (!Vectorizable<Derived>) {
            if constexpr (IsSuperScalar<Scalar>) {
                VectorBaseRef<OutType> fx = fx_.const_cast_derived();

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

    /*!
     * @brief Returns the output of compute on the derived class
     *
     * @tparam InType Eigen type of input vector
     * @param x Input vector
     * @return Output<typename InType::Scalar> Output type
     */
    template <class InType>
    inline Output<typename InType::Scalar> compute(ConstVectorBaseRef<InType> x) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx(this->output_rows());
        fx.setZero();
        this->derived().compute(x, fx);
        return fx;
    }

    template <class InType, class OutType, class AdjGradType, class AdjVarType>
    inline void compute_adjointgradient(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstVectorBaseRef<AdjGradType> adjgrad_,
                                        ConstVectorBaseRef<AdjVarType> adjvars) const {
        this->derived().compute_adjointgradient(x, fx_, adjgrad_, adjvars);
    }

    template <class InType, class AdjGradType, class AdjVarType>
    inline void adjointgradient(ConstVectorBaseRef<InType> x,
                                ConstVectorBaseRef<AdjGradType> adjgrad_,
                                ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx(this->output_rows());
        fx.setZero();
        this->derived().compute_adjointgradient(x, fx, adjgrad_, adjvars);
    }

    template <class InType, class AdjVarType>
    inline Gradient<typename InType::Scalar>
    adjointgradient(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Gradient<Scalar> adjgrad(this->input_rows());
        adjgrad.setZero();
        this->derived().adjointgradient(x, adjgrad, adjvars);
        return adjgrad;
    }

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /*
    * This function is the interface allowing the vector function to be used as a constraint inside
    psiopt.
    * Vector X is the total variables vector for the full optimization problem, and FX is the total
    equality or inequality constraints vector for the problem. Data pertaining to the location of the
    input variables in X for each distinct call as well as the target location for the output
    variables in FX is stored in the Tycho::SolverIndexingData object. In general users should not directly
    overload this function unless they have a very good reason.
    */
    void constraints(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                     const Tycho::SolverIndexingData &data) const {

        auto Impl = [&](auto &x) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());

            const int IRR = this->input_rows();
            const int ORR = this->output_rows();

            // Scalar non-vectorized call to the function
            auto ScalarImpl = [&](int start, int stop) {
                for (int V = start; V < stop; V++) {
                    this->gather_input(X, x, V, data);
                    new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V],
                                                         this->output_rows());
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
                int Packs = data.NumAppl() / vsize;

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
                            FX.data() + data.InnerConstraintStarts[V], this->output_rows());
                        for (int l = 0; l < ORR; l++) {
                            fx[l] = fx_vect[l][j];
                        }
                    }
                }
                ScalarImpl(Packs * vsize, data.NumAppl());
            };

            // Only try vectorized impl if Derived allows and it is enabled
            if constexpr (Vectorizable<Derived>) {
                if (this->derived().EnableVectorization) {
                    VectorImpl();
                } else {
                    ScalarImpl(0, data.NumAppl());
                }
            } else {
                ScalarImpl(0, data.NumAppl());
            }
        };

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1));
    }

    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const Tycho::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &l) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());
            Eigen::Map<Input<double>> agx(NULL, this->input_rows());

            for (int V = 0; V < data.NumAppl(); V++) {
                this->gather_input(X, x, V, data);
                this->gather_mult(L, l, V, data);
                new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V],
                                                     this->output_rows());
                new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.InnerGradientStarts[V],
                                                     this->input_rows());
                fx.setZero();
                agx.setZero();
                this->derived().compute_adjointgradient(x, fx, agx, l);
            }
        };

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
                                    tycho::utils::TempSpec<Output<double>>(this->output_rows(), 1));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////

  protected:
    /*
     * This helper is used to gather the required variables for the Vth function call
     * from the optimizer variables vector, X, into the local input vector of the function, xt.
     * The solver indexing data structure contains the locations of the variables for the Vth
     * function call as well as precomputed metadata indicating whether these variables are stored
     * contiguously. In the case of contiguous storage the variables are retrieved in a single call
     * This is implemented here to potentially take advantage of the fact the the input size is a
     * known compile time constant. This was observed to provide a modest but measurable perf
     * improvement.
     */
    template <class XtType>
        requires std::derived_from<std::remove_cvref_t<XtType>,
                                   Eigen::EigenBase<std::remove_cvref_t<XtType>>>
    inline void gather_input(ConstEigenRef<Eigen::VectorXd> X, XtType &xt, int V,
                            const Tycho::SolverIndexingData &data) const {
        if (data.VindexContinuity[V] == ParsedIOFlags::Contiguous) {
            xt = X.template segment<IR>(data.VLoc(0, V), this->input_rows());
        } else {
            for (int i = 0; i < this->input_rows(); i++)
                xt(i) = X(data.VLoc(i, V));
        }
    }

    /*
     * This helper is used to gather the required lagrange multipliers for the Vth function call
     * from the optimizer multipler vector, L, into the local multiplier vector of the function, lt.
     * See above for justification.
     */
    template <class LType>
        requires std::derived_from<std::remove_cvref_t<LType>,
                                   Eigen::EigenBase<std::remove_cvref_t<LType>>>
    inline void gather_mult(ConstEigenRef<Eigen::VectorXd> L, LType &l, int V,
                           const Tycho::SolverIndexingData &data) const {
        if (data.CindexContinuity[V] == ParsedIOFlags::Contiguous) {
            l = L.template segment<OR>(data.CLoc(0, V), this->output_rows());
        } else {
            for (int i = 0; i < this->output_rows(); i++)
                l(i) = L(data.CLoc(i, V));
        }
    }
};

} // namespace tycho::vf
