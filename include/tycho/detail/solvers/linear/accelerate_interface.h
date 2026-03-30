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

#ifndef EIGEN_ACCELERATESUPPORT_H
#define EIGEN_ACCELERATESUPPORT_H

#include <Accelerate/Accelerate.h>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

#include <Eigen/Sparse>
#include <cmath>
#include <numeric>

#include "tycho/detail/solvers/linear/accelerate_utils.h"

#include <limits>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

/*
The classes in this file are directly based on the AccelerateSupport module from Eigen 3.4 and
are subject to Eigen's MPL2 license, which can be found in the notices folder of the GitHub
repository. Changes include the addition of several member variables and functions to provide more
fine grained control of the Accelerate Sparse solvers, to avoid making unnecessary
copies/allocations, to add an iterative refinement implementation (to bring solution accuracy more
in line with PARDISO), as well as to more closely align with the methods of the PARDISO interface in
PardisoInterface.h (particularly for methods that Tycho employs extensively). Note that the MPL2
license is only applied to this particular file and not the rest of the project, as per the MPL2
license.

*/

namespace Eigen {

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
class AccelerateImpl;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLLT
 * \brief A direct Cholesky (LLT) factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLLT
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLLT =
    AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationCholesky, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLT
 * \brief The default Cholesky (LDLT) factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLT
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLT = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLT, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLTUnpivoted
 * \brief A direct Cholesky-like LDL^T factorization and solver based on Accelerate with only 1x1
 * pivots and no pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTUnpivoted
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTUnpivoted =
    AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTUnpivoted, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLTSBK
 * \brief A direct Cholesky (LDLT) factorization and solver based on Accelerate with Supernode
 * Bunch-Kaufman and static pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTSBK
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTSBK =
    AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTSBK, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLTTPP
 * \brief A direct Cholesky (LDLT) factorization and solver based on Accelerate with full threshold
 * partial pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTTPP
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTTPP =
    AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTTPP, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateQR
 * \brief A QR factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateQR
 */
template <typename MatrixType>
using AccelerateQR = AccelerateImpl<MatrixType, 0, SparseFactorizationQR, false>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateCholeskyAtA
 * \brief A QR factorization and solver based on Accelerate without storing Q (equivalent to A^TA =
 * R^T R)
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateCholeskyAtA
 */
template <typename MatrixType>
using AccelerateCholeskyAtA = AccelerateImpl<MatrixType, 0, SparseFactorizationCholeskyAtA, false>;

namespace internal {
template <typename T> struct AccelFactorizationDeleter {
    void operator()(T *sym) {
        if (sym) {
            SparseCleanup(*sym);
            delete sym;
            sym = nullptr;
        }
    }
};

template <typename DenseVecT, typename DenseMatT, typename SparseMatT, typename NumFactT>
struct SparseTypesTraitBase {
    typedef DenseVecT AccelDenseVector;
    typedef DenseMatT AccelDenseMatrix;
    typedef SparseMatT AccelSparseMatrix;

    typedef SparseOpaqueSymbolicFactorization SymbolicFactorization;
    typedef NumFactT NumericFactorization;

    typedef AccelFactorizationDeleter<SymbolicFactorization> SymbolicFactorizationDeleter;
    typedef AccelFactorizationDeleter<NumericFactorization> NumericFactorizationDeleter;
};

template <typename Scalar> struct SparseTypesTrait {};

template <>
struct SparseTypesTrait<double>
    : SparseTypesTraitBase<DenseVector_Double, DenseMatrix_Double, SparseMatrix_Double,
                           SparseOpaqueFactorization_Double> {};

template <>
struct SparseTypesTrait<float>
    : SparseTypesTraitBase<DenseVector_Float, DenseMatrix_Float, SparseMatrix_Float,
                           SparseOpaqueFactorization_Float> {};

// Taken from
// https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/accelerate_sparse.cc
inline void *resizeForAccelerateAlignment(const size_t required_size,
                                          std::vector<uint8_t> *mem_vec) {
    // As per the Accelerate documentation, all workspace memory passed to the
    // sparse solver functions must be 16-byte aligned.
    constexpr int kAccelerateRequiredAlignment = 16;
    // Although malloc() on macOS should always be 16-byte aligned, it is unclear
    // if this holds for new(), or on other Apple OSs (phoneOS, watchOS etc).
    // As such we assume it is not and use std::align() to create a (potentially
    // offset) 16-byte aligned sub-buffer of the specified size within workspace.
    mem_vec->resize(required_size + kAccelerateRequiredAlignment);
    size_t size_from_aligned_start = mem_vec->size();
    void *aligned_solve_workspace_start = reinterpret_cast<void *>(mem_vec->data());
    aligned_solve_workspace_start =
        std::align(kAccelerateRequiredAlignment, required_size, aligned_solve_workspace_start,
                   size_from_aligned_start);
    return aligned_solve_workspace_start;
}

} // end namespace internal

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
class AccelerateImpl
    : public SparseSolverBase<AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>> {
  protected:
    using Base = SparseSolverBase<AccelerateImpl>;
    using Base::derived;
    using Base::m_isInitialized;

  public:
    using Base::_solve_impl;

    typedef MatrixType_ MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    enum { ColsAtCompileTime = Dynamic, MaxColsAtCompileTime = Dynamic };
    enum { UpLo = UpLo_ };

    using AccelDenseVector = typename internal::SparseTypesTrait<Scalar>::AccelDenseVector;
    using AccelDenseMatrix = typename internal::SparseTypesTrait<Scalar>::AccelDenseMatrix;
    using AccelSparseMatrix = typename internal::SparseTypesTrait<Scalar>::AccelSparseMatrix;
    using SymbolicFactorization =
        typename internal::SparseTypesTrait<Scalar>::SymbolicFactorization;
    using NumericFactorization = typename internal::SparseTypesTrait<Scalar>::NumericFactorization;
    using SymbolicFactorizationDeleter =
        typename internal::SparseTypesTrait<Scalar>::SymbolicFactorizationDeleter;
    using NumericFactorizationDeleter =
        typename internal::SparseTypesTrait<Scalar>::NumericFactorizationDeleter;

    AccelerateImpl() {
        m_isInitialized = false;

        auto check_flag_set = [](int value, int flag) { return ((value & flag) == flag); };

        if (check_flag_set(UpLo_, Symmetric)) {
            m_sparseKind = SparseSymmetric;
            m_triType = (UpLo_ & Lower) ? SparseLowerTriangle : SparseUpperTriangle;
        } else if (check_flag_set(UpLo_, UnitLower)) {
            m_sparseKind = SparseUnitTriangular;
            m_triType = SparseLowerTriangle;
        } else if (check_flag_set(UpLo_, UnitUpper)) {
            m_sparseKind = SparseUnitTriangular;
            m_triType = SparseUpperTriangle;
        } else if (check_flag_set(UpLo_, StrictlyLower)) {
            m_sparseKind = SparseTriangular;
            m_triType = SparseLowerTriangle;
        } else if (check_flag_set(UpLo_, StrictlyUpper)) {
            m_sparseKind = SparseTriangular;
            m_triType = SparseUpperTriangle;
        } else if (check_flag_set(UpLo_, Lower)) {
            m_sparseKind = SparseTriangular;
            m_triType = SparseLowerTriangle;
        } else if (check_flag_set(UpLo_, Upper)) {
            m_sparseKind = SparseTriangular;
            m_triType = SparseUpperTriangle;
        } else {
            m_sparseKind = SparseOrdinary;
            m_triType = (UpLo_ & Lower) ? SparseLowerTriangle : SparseUpperTriangle;
        }

        order_ = SparseOrderMetis;
        do_iterative_refinement_ = false;
        iterative_refinement_iterations_ = 2;
    }

    explicit AccelerateImpl(const MatrixType &matrix) : AccelerateImpl() { compute(matrix); }

    ~AccelerateImpl() {}

    inline Index cols() const { return n_cols_; }
    inline Index rows() const { return n_rows_; }

    ComputationInfo info() const {
        eigen_assert(m_isInitialized && "Decomposition is not initialized.");
        return info_;
    }

    void set_matrix(const MatrixType &matrix);

    void analyze_pattern(const MatrixType &matrix);

    void factorize(const MatrixType &matrix);

    void compute(const MatrixType &matrix);

    template <typename Rhs, typename Dest>
    void _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const;

    /** Sets the ordering algorithm to use. */
    void set_order(SparseOrder_t order) { order_ = order; }

    /** Sets the number of threads for accelerate */
    inline void set_num_threads(int num_threads) { accelerate_set_num_threads(num_threads); }

    void set_iterative_refinement(bool iterativeRefinement) {
        do_iterative_refinement_ = iterativeRefinement;
    }

    void set_iterative_refinement_iterations(int iterations) {
        eigen_assert(iterations >= 0 && "Number of iterations must be non-negative.");
        iterative_refinement_iterations_ = iterations;
    }

    void set_pivot_tolerance(Scalar tol) { pivot_tolerance_ = tol; }
    void set_zero_tolerance(Scalar tol) { zero_tolerance_ = tol; }

    MatrixType &get_matrix() { return matrix_; }

    template <int U = UpLo>
    typename std::enable_if<!bool(U & Symmetric), void>::type get_matrix(const MatrixType &matrix) {
        matrix_ = matrix;
        matrix_.makeCompressed();
    }

    template <int U = UpLo>
    typename std::enable_if<bool(U &Symmetric), void>::type get_matrix(const MatrixType &matrix) {
        // Similar to Pardiso, use selfadjointView to ensure symmetry
        PermutationMatrix<Dynamic, Dynamic, StorageIndex> p_null;
        matrix_.resize(matrix.rows(), matrix.cols());

        constexpr int TriangleType = (UpLo & Lower) ? Lower : Upper;
        matrix_.template selfadjointView<TriangleType>() =
            matrix.template selfadjointView<TriangleType>().twistedBy(p_null);
        matrix_.makeCompressed();
    }

    template <int U = UpLo>
    typename std::enable_if<bool(U &Symmetric), MatrixType>::type
    get_matrix_twisted(const MatrixType &matrix) {
        eigen_assert(!permutation_.empty() &&
                     "Permutation not available. Call compute() or analyze_pattern() first.");
        eigen_assert(matrix.rows() == matrix.cols() &&
                     "Matrix must be square for twisted operation.");
        eigen_assert(static_cast<size_t>(matrix.rows()) == permutation_.size() &&
                     "Matrix size must match permutation size.");

        // Create permutation matrix from the stored permutation vector
        PermutationMatrix<Dynamic, Dynamic, StorageIndex> p_perm;
        p_perm.indices() =
            Map<const Matrix<StorageIndex, Dynamic, 1>>(permutation_.data(), permutation_.size());

        MatrixType result;
        result.resize(matrix.rows(), matrix.cols());

        constexpr int TriangleType = (U & Lower) ? Lower : Upper;
        result.template selfadjointView<TriangleType>() =
            matrix.template selfadjointView<TriangleType>().twistedBy(p_perm);

        result.makeCompressed();
        return result;
    }

    template <SparseFactorization_t S = Solver_>
    std::enable_if_t<S == SparseFactorizationLDLTTPP, Index> peigs() const {
        return static_cast<Index>(peigs_);
    }

    template <SparseFactorization_t S = Solver_>
    std::enable_if_t<S == SparseFactorizationLDLTTPP, Index> neigs() const {
        return static_cast<Index>(neigs_);
    }

    template <SparseFactorization_t S = Solver_>
    std::enable_if_t<S == SparseFactorizationLDLTTPP, Index> zeigs() const {
        return static_cast<Index>(zeigs_);
    }

    inline int ppivs() const {
        // Accelerate doesn't provide direct pivot perturbation count
        // Return 0 as a safe default for now
        return 0;
    }

    // This method initializes the internal AccelSparseMatrix. This must be called
    // after changing the sparsity pattern of matrix_ via the reference returned
    // from MatrixType& get_matrix()
    void reinitialize_internal_matrix_representation();

    // Internal factorization methods
    AccelerateImpl &analyze_pattern_internal();
    AccelerateImpl &factorize_internal();
    AccelerateImpl &refactorize_internal();
    AccelerateImpl &compute_internal();

    // Cleanup method
    void release();

    // Performance metrics
    mutable int flops_ = 0;
    mutable int mem_ = 0;

  private:
    void *getAlignedPointer(std::vector<uint8_t> &storage) const {
        constexpr int kAlign = 16;
        void *ptr = reinterpret_cast<void *>(storage.data());
        size_t space = storage.size();
        return std::align(kAlign, space - kAlign, ptr, space);
    }

    void cacheInertia() {
        if constexpr (Solver_ == SparseFactorizationLDLTTPP) {
            if (m_numericFactorization) {
                int np = 0, nz = 0, nn = 0;
                if (SparseGetInertia(*m_numericFactorization, &np, &nz, &nn) == 0) {
                    peigs_ = np;
                    neigs_ = nn;
                    zeigs_ = nz;
                } else {
                    peigs_ = 0;
                    neigs_ = 0;
                    zeigs_ = 0;
                }
            }
        }
    }

    void updatePerformanceMetrics() {
        if (m_symbolicFactorization) {
            mem_ = std::is_same<Scalar, double>::value ? m_symbolicFactorization->factorSize_Double
                                                       : m_symbolicFactorization->factorSize_Float;
        }
        flops_ = 0;
    }

    void buildAccelSparseMatrix() {
        SparseAttributes_t attributes{};
        attributes.kind = m_sparseKind;

        SparseMatrixStructure structure{};
        structure.blockSize = 1;

        if constexpr (MatrixType::IsRowMajor) { // RowMajor (CSR) format
            // For CSR, Accelerate expects CSC. We use the 'transpose' attribute
            // to tell Accelerate to interpret the CSR matrix as a transposed CSC matrix.
            const Index nRowStarts = matrix_.rows() + 1;
            m_columnStarts.resize(nRowStarts); // Reuse m_columnStarts for rowStarts
            std::copy_n(matrix_.outerIndexPtr(), nRowStarts, m_columnStarts.data());

            structure.rowCount = static_cast<int>(matrix_.cols());    // Swapped
            structure.columnCount = static_cast<int>(matrix_.rows()); // Swapped
            structure.columnStarts = m_columnStarts.data();           // These are now rowStarts
            structure.rowIndices =
                const_cast<int *>(matrix_.innerIndexPtr()); // These are now columnIndices
            attributes.transpose = true;

            // When transposing, we need to flip the triangle type for symmetric matrices
            if (m_sparseKind == SparseSymmetric) {
                attributes.triangle =
                    (m_triType == SparseLowerTriangle) ? SparseUpperTriangle : SparseLowerTriangle;
            } else {
                attributes.triangle = m_triType;
            }
        } else { // CSC format
            const Index nColumnsStarts = matrix_.cols() + 1;
            m_columnStarts.resize(nColumnsStarts);
            std::copy_n(matrix_.outerIndexPtr(), nColumnsStarts, m_columnStarts.data());

            structure.rowCount = static_cast<int>(matrix_.rows());
            structure.columnCount = static_cast<int>(matrix_.cols());
            structure.columnStarts = m_columnStarts.data();
            structure.rowIndices = const_cast<int *>(matrix_.innerIndexPtr());
            attributes.transpose = false;
            attributes.triangle = m_triType;
        }

        structure.attributes = attributes;
        accel_matrix_.structure = structure;
        accel_matrix_.data = const_cast<Scalar *>(matrix_.valuePtr());
    }

    void doAnalysis() {
        m_numericFactorization.reset(nullptr);

        // Only resize permutation if necessary to avoid unnecessary allocations
        if (permutation_.size() != static_cast<size_t>(n_rows_)) {
            permutation_.resize(n_rows_);
        }
        std::iota(permutation_.begin(), permutation_.end(), 0); // Initialize with identity

        SparseSymbolicFactorOptions fopts{};
        fopts.control = SparseDefaultControl;
        fopts.orderMethod = order_;
        fopts.order = permutation_.data(); // Provide storage for computed permutation
        fopts.ignoreRowsAndColumns = nullptr;
        fopts.malloc = malloc;
        fopts.free = free;

        fopts.reportError = [](const char *msg) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Accelerate Sparse Symbolic Factorization Error: {}\n", msg);
        };

        m_symbolicFactorization.reset(
            new SymbolicFactorization(SparseFactor(Solver_, accel_matrix_.structure, fopts)));

        SparseStatus_t status = m_symbolicFactorization->status;

        updateInfoStatus(status);

        if (status != SparseStatusOK) {
            m_symbolicFactorization.reset(nullptr);
        }
    }

    void doFactorization() {
        SparseStatus_t status = SparseStatusReleased;

        if (m_symbolicFactorization) {

            SparseNumericFactorOptions nopts{};
            nopts.control = SparseDefaultControl;
            nopts.scalingMethod = SparseScalingDefault;
            nopts.scaling = nullptr;
            // Default values set by Apple
            nopts.pivotTolerance = pivot_tolerance_;
            nopts.zeroTolerance = zero_tolerance_;

            // Get factor and workspace size
            const int factorSize = std::is_same<Scalar, double>::value
                                       ? m_symbolicFactorization->factorSize_Double
                                       : m_symbolicFactorization->factorSize_Float;
            const int workspaceSize = std::is_same<Scalar, double>::value
                                          ? m_symbolicFactorization->workspaceSize_Double
                                          : m_symbolicFactorization->workspaceSize_Float;

            m_numericFactorization.reset(new NumericFactorization(
                SparseFactor(*m_symbolicFactorization, accel_matrix_, nopts,
                             internal::resizeForAccelerateAlignment(factorSize, &factor_storage_),
                             internal::resizeForAccelerateAlignment(workspaceSize, &workspace_))));

            status = m_numericFactorization->status;

            if (status != SparseStatusOK)
                m_numericFactorization.reset(nullptr);
            else
                cacheInertia();
        }

        updateInfoStatus(status);
    }

    void doRefactorization() {
        if (!m_numericFactorization || !m_symbolicFactorization) {
            doFactorization();
            if (m_numericFactorization)
                updatePerformanceMetrics();
            return;
        }

        void *ws = getAlignedPointer(workspace_);
        SparseRefactor(accel_matrix_, m_numericFactorization.get(), ws);

        SparseStatus_t status = m_numericFactorization->status;
        updateInfoStatus(status);

        if (status == SparseStatusOK) {
            cacheInertia();
            updatePerformanceMetrics();
        }
    }

  protected:
    void updateInfoStatus(SparseStatus_t status) const {
        switch (status) {
        case SparseStatusOK:
            info_ = Success;
            break;
        case SparseFactorizationFailed:
        case SparseMatrixIsSingular:
            info_ = NumericalIssue;
            break;
        case SparseInternalError:
        case SparseParameterError:
        case SparseStatusReleased:
        default:
            info_ = InvalidInput;
            break;
        }
    }

    std::vector<long> m_columnStarts;
    mutable MatrixType matrix_;
    mutable AccelSparseMatrix accel_matrix_;
    mutable ComputationInfo info_;
    mutable std::vector<uint8_t> factor_storage_;
    mutable std::vector<uint8_t> workspace_;
    mutable std::vector<uint8_t> solve_workspace_; // Cache solve workspace
    mutable std::vector<Scalar> r_mem_;
    mutable int cached_solve_workspace_size_ = 0; // Track cached size
    Index n_rows_, n_cols_;
    std::unique_ptr<SymbolicFactorization, SymbolicFactorizationDeleter> m_symbolicFactorization;
    std::unique_ptr<NumericFactorization, NumericFactorizationDeleter> m_numericFactorization;
    SparseKind_t m_sparseKind;
    SparseTriangle_t m_triType;
    SparseOrder_t order_;
    bool do_iterative_refinement_;
    int iterative_refinement_iterations_;
    Scalar pivot_tolerance_ = Scalar(0.01);
    Scalar zero_tolerance_ = Scalar(1e-4) * std::numeric_limits<Scalar>::epsilon();
    mutable int peigs_ = 0;
    mutable int neigs_ = 0;
    mutable int zeigs_ = 0;
    mutable std::vector<StorageIndex> permutation_; // Store permutation from factorization
};

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::set_matrix(
    const MatrixType &matrix) {
    if (EnforceSquare_) {
        eigen_assert(matrix.rows() == matrix.cols());
    }

    // matrix_ = matrix;
    get_matrix(matrix);
    n_rows_ = matrix_.rows();
    n_cols_ = matrix_.cols();

    buildAccelSparseMatrix();

    m_isInitialized = false;
    m_symbolicFactorization.reset(nullptr);
    m_numericFactorization.reset(nullptr);
    cached_solve_workspace_size_ = 0; // Clear cached workspace size
    info_ = Success;
}

/** Computes the symbolic and numeric decomposition of matrix \a a */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::compute(const MatrixType &a) {
    set_matrix(a);

    doAnalysis();

    if (m_symbolicFactorization)
        doFactorization();

    m_isInitialized = true;
}

/** Performs a symbolic decomposition on the sparsity pattern of matrix \a a.
 *
 * This function is particularly useful when solving for several problems having the same structure.
 *
 * \sa factorize()
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::analyze_pattern(
    const MatrixType &a) {
    if (EnforceSquare_) {
        eigen_assert(a.rows() == a.cols());
    }
    set_matrix(a);

    doAnalysis();

    m_isInitialized = true;
}

/** Performs a numeric decomposition of matrix \a a.
 *
 * The given matrix must have the same sparsity pattern as the matrix on which the symbolic
 * decomposition has been performed.
 *
 * \sa analyze_pattern()
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::factorize(const MatrixType &a) {
    eigen_assert(m_symbolicFactorization && "You must first call analyze_pattern()");
    eigen_assert(n_rows_ == a.rows() && n_cols_ == a.cols());

    if (EnforceSquare_) {
        eigen_assert(a.rows() == a.cols());
    }

    matrix_ = a;
    buildAccelSparseMatrix();
    m_numericFactorization.reset(nullptr);

    doFactorization();
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
template <typename Rhs, typename Dest>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::_solve_impl(
    const MatrixBase<Rhs> &b, MatrixBase<Dest> &x) const {
    if (!m_numericFactorization) {
        info_ = InvalidInput;
        return;
    }

    eigen_assert(n_rows_ == b.rows());
    eigen_assert(((b.cols() == 1) || b.outerStride() == b.rows()));

    SparseStatus_t status = SparseStatusOK;

    Scalar *b_ptr = const_cast<Scalar *>(b.derived().data());
    Scalar *x_ptr = const_cast<Scalar *>(x.derived().data());

    AccelDenseMatrix xmat{};
    xmat.attributes = SparseAttributes_t();
    xmat.columnCount = static_cast<int>(x.cols());
    xmat.rowCount = static_cast<int>(x.rows());
    xmat.columnStride = xmat.rowCount;
    xmat.data = x_ptr;

    AccelDenseMatrix bmat{};
    bmat.attributes = SparseAttributes_t();
    bmat.columnCount = static_cast<int>(b.cols());
    bmat.rowCount = static_cast<int>(b.rows());
    bmat.columnStride = bmat.rowCount;
    bmat.data = b_ptr;

    const int nrhs = (bmat.attributes.transpose) ? bmat.rowCount : bmat.columnCount;
    const int workspaceSize = m_numericFactorization->solveWorkspaceRequiredStatic +
                              nrhs * m_numericFactorization->solveWorkspaceRequiredPerRHS;

    // Use cached solve workspace to avoid repeated allocations
    void *ws;
    if (workspaceSize != cached_solve_workspace_size_) {
        ws = internal::resizeForAccelerateAlignment(workspaceSize, &solve_workspace_);
        cached_solve_workspace_size_ = workspaceSize;
    } else {
        ws = getAlignedPointer(solve_workspace_);
    }
    assert(ws != nullptr && "Accelerate workspace alignment failed");

    SparseSolve(*m_numericFactorization, bmat, xmat, ws);

    updateInfoStatus(status);

    if (do_iterative_refinement_) {
        auto n = vDSP_Length(x.rows() * x.cols());
        if (r_mem_.size() < n) {
            r_mem_.resize(n);
        }
        AccelDenseMatrix ref_mat{};
        ref_mat.attributes = SparseAttributes_t();
        ref_mat.columnCount = static_cast<int>(x.cols());
        ref_mat.rowCount = static_cast<int>(x.rows());
        ref_mat.columnStride = ref_mat.rowCount;
        ref_mat.data = r_mem_.data();

        for (int i = 0; i < iterative_refinement_iterations_; ++i) {
            // Calculate residual: r = -b + A*x
            if constexpr (std::is_same_v<Scalar, double>) {
                vDSP_vnegD(bmat.data, 1, ref_mat.data, 1, n);
            } else {
                vDSP_vneg(bmat.data, 1, ref_mat.data, 1, n);
            }
            SparseMultiplyAdd(accel_matrix_, xmat, ref_mat);

            // Solve for correction: ref = A^{-1} * r
            SparseSolve(*m_numericFactorization, ref_mat, ws);

            // Update solution: x -= correction
            if constexpr (std::is_same_v<Scalar, double>) {
                vDSP_vsubD(ref_mat.data, 1, xmat.data, 1, xmat.data, 1, n);
            } else {
                vDSP_vsub(ref_mat.data, 1, xmat.data, 1, xmat.data, 1, n);
            }
        }
    }
}

/** Initializes the internal AccelSparseMatrix from the internal Eigen sparse matrix
 *
 * This method must be called after changing the sparsity pattern of the matrix_
 * member via the reference returned from MatrixType& get_matrix().
 *
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_,
                    EnforceSquare_>::reinitialize_internal_matrix_representation() {
    // Update matrix dimensions in case they changed when the sparsity pattern was modified
    n_rows_ = matrix_.rows();
    n_cols_ = matrix_.cols();

    // Build/rebuild the internal AccelSparseMatrix from the current state of matrix_
    buildAccelSparseMatrix();

    // Reset factorizations since the matrix structure has changed
    m_symbolicFactorization.reset(nullptr);
    m_numericFactorization.reset(nullptr);

    // Clear cached workspace size since matrix structure changed
    cached_solve_workspace_size_ = 0;

    // Reset initialization state - will need to recompute factorizations
    m_isInitialized = false;

    // Reset computation info
    info_ = Success;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_> &
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::analyze_pattern_internal() {
    eigen_assert(matrix_.rows() > 0 && matrix_.cols() > 0 &&
                 "Matrix must be set before calling analyze_pattern_internal()");

    m_symbolicFactorization.reset(nullptr);
    doAnalysis();

    m_isInitialized = true;
    return *this;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_> &
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::factorize_internal() {
    eigen_assert(m_symbolicFactorization && "You must first call analyze_pattern_internal()");

    m_numericFactorization.reset(nullptr);
    doFactorization();

    if (m_numericFactorization)
        updatePerformanceMetrics();

    return *this;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_> &
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::refactorize_internal() {
    doRefactorization();
    return *this;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_> &
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::compute_internal() {
    eigen_assert(matrix_.rows() > 0 && matrix_.cols() > 0 &&
                 "Matrix must be set before calling compute_internal()");

    m_symbolicFactorization.reset(nullptr);
    m_numericFactorization.reset(nullptr);

    doAnalysis();

    if (m_symbolicFactorization) {
        doFactorization();

        if (m_numericFactorization)
            updatePerformanceMetrics();
    }

    m_isInitialized = true;
    return *this;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::release() {
    // Clean up factorizations
    m_numericFactorization.reset(nullptr);
    m_symbolicFactorization.reset(nullptr);

    // Clear storage vectors
    factor_storage_.clear();
    workspace_.clear();
    solve_workspace_.clear();
    r_mem_.clear();

    // Clear matrix data
    matrix_.resize(0, 0);
    matrix_.data().squeeze();

    // Clear permutation
    permutation_.clear();

    // Reset cached sizes
    cached_solve_workspace_size_ = 0;

    // Reset performance metrics
    flops_ = 0;
    mem_ = 0;

    // Reset initialization state
    m_isInitialized = false;
    info_ = Success;
}

} // end namespace Eigen

#endif // EIGEN_ACCELERATESUPPORT_H
