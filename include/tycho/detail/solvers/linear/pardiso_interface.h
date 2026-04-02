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

#ifndef EIGEN_PARDISOSUPPORT_H
#define EIGEN_PARDISOSUPPORT_H

#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <mkl_pardiso.h>

#include <Eigen/SparseCore>

/*
The classes in this file are directly based on the PardisoSupport module from Eigen 3.4 and
are subject to Eigen's MPL2 license, which can be found in the notices folder of the GitHub
repository. Changes include the addition of several member functions to access internal PARDISO
parameters, and additional methods to perform phases of the factorization without making unnecessary
copies of the input matrix. Note that the MPL2 license is only applied to this particular file and
not the rest of the project, as per the MPL2 license.

*/

namespace Eigen {

template <typename _MatrixType> class PardisoLU;
template <typename _MatrixType, int Options = Upper> class PardisoLLT;
template <typename _MatrixType, int Options = Upper> class PardisoLDLT;

namespace internal {
template <typename IndexType> struct pardiso_run_selector {
    static IndexType run(_MKL_DSS_HANDLE_t pt, IndexType maxfct, IndexType mnum, IndexType type,
                         IndexType phase, IndexType n, void *a, IndexType *ia, IndexType *ja,
                         IndexType *perm, IndexType nrhs, IndexType *iparm, IndexType msglvl,
                         void *b, void *x) {
        IndexType error = 0;
        ::pardiso(pt, &maxfct, &mnum, &type, &phase, &n, a, ia, ja, perm, &nrhs, iparm, &msglvl, b,
                  x, &error);
        return error;
    }
};
template <> struct pardiso_run_selector<long long int> {
    typedef long long int IndexType;
    static IndexType run(_MKL_DSS_HANDLE_t pt, IndexType maxfct, IndexType mnum, IndexType type,
                         IndexType phase, IndexType n, void *a, IndexType *ia, IndexType *ja,
                         IndexType *perm, IndexType nrhs, IndexType *iparm, IndexType msglvl,
                         void *b, void *x) {
        IndexType error = 0;
        ::pardiso_64(pt, &maxfct, &mnum, &type, &phase, &n, a, ia, ja, perm, &nrhs, iparm, &msglvl,
                     b, x, &error);
        return error;
    }
};

template <class Pardiso> struct pardiso_traits;

template <typename _MatrixType> struct pardiso_traits<PardisoLU<_MatrixType>> {
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::StorageIndex StorageIndex;
};

template <typename _MatrixType, int Options>
struct pardiso_traits<PardisoLLT<_MatrixType, Options>> {
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::StorageIndex StorageIndex;
};

template <typename _MatrixType, int Options>
struct pardiso_traits<PardisoLDLT<_MatrixType, Options>> {
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::StorageIndex StorageIndex;
};

} // end namespace internal

template <class Derived> class PardisoImpl : public SparseSolverBase<Derived> {
  protected:
    typedef SparseSolverBase<Derived> Base;
    using Base::derived;
    using Base::m_isInitialized;

    typedef internal::pardiso_traits<Derived> Traits;

  public:
    using Base::_solve_impl;

    typedef typename Traits::MatrixType MatrixType;
    typedef typename Traits::Scalar Scalar;
    typedef typename Traits::RealScalar RealScalar;
    typedef typename Traits::StorageIndex StorageIndex;
    typedef SparseMatrix<Scalar, RowMajor, StorageIndex> SparseMatrixType;
    typedef Matrix<Scalar, Dynamic, 1> VectorType;
    typedef Matrix<StorageIndex, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<StorageIndex, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    typedef Array<StorageIndex, 64, 1, DontAlign> ParameterType;
    enum {
        ScalarIsComplex = NumTraits<Scalar>::IsComplex,
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
    };

    PardisoImpl() {
        eigen_assert((sizeof(StorageIndex) >= sizeof(_INTEGER_t) && sizeof(StorageIndex) <= 8) &&
                     "Non-supported index type");
        iparm_.setZero();
        msglvl_ = 0; // No output
        m_isInitialized = false;
    }

    ~PardisoImpl() { pardiso_release(); }

    inline Index cols() const { return size_; }
    inline Index rows() const { return size_; }

    /** \brief Reports whether previous computation was successful.
     *
     * \returns \c Success if computation was succesful,
     *          \c NumericalIssue if the matrix appears to be negative.
     */
    ComputationInfo info() const {
        eigen_assert(m_isInitialized && "Decomposition is not initialized.");
        return info_;
    }

    /** \warning for advanced usage only.
     * \returns a reference to the parameter array controlling PARDISO.
     * See the PARDISO manual to know how to use it. */
    ParameterType &pardiso_parameter_array() { return iparm_; }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
     *
     * This function is particularly useful when solving for several problems
     * having the same structure.
     *
     * \sa factorize()
     */
    Derived &analyze_pattern(const MatrixType &matrix);

    /** Performs a numeric decomposition of \a matrix
     *
     * The given matrix must has the same sparcity than the matrix on which the
     * symbolic decomposition has been performed.
     *
     * \sa analyze_pattern()
     */
    Derived &factorize(const MatrixType &matrix);

    Derived &compute(const MatrixType &matrix);

    Derived &factorize_internal();
    Derived &refactorize_internal() { return factorize_internal(); }
    Derived &compute_internal();
    Derived &analyze_pattern_internal();

    template <typename Rhs, typename Dest>
    void _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const;

  protected:
    void pardiso_release() {
        if (m_isInitialized) // Factorization ran at least once
        {
            internal::pardiso_run_selector<StorageIndex>::run(
                pt_, 1, 1, type_, -1, internal::convert_index<StorageIndex>(size_), 0, 0, 0,
                perm_.data(), 0, iparm_.data(), msglvl_, NULL, NULL);
            m_isInitialized = false;
        }
    }

    void pardiso_init(int type) {
        type_ = type;
        bool symmetric = std::abs(type_) < 10;
        iparm_[0] = 1;                  // No solver default
        iparm_[1] = 3;                  // use Metis for the ordering
        iparm_[2] = 0;                  // Reserved. Set to zero. (??Numbers of processors, value
                                        // of OMP_NUM_THREADS??)
        iparm_[3] = 0;                  // No iterative-direct algorithm
        iparm_[4] = 0;                  // No user fill-in reducing permutation
        iparm_[5] = 0;                  // Write solution into x, b is left unchanged
        iparm_[6] = 0;                  // Not in use
        iparm_[7] = 2;                  // Max numbers of iterative refinement steps
        iparm_[8] = 0;                  // Not in use
        iparm_[9] = 13;                 // Perturb the pivot elements with 1E-13
        iparm_[10] = symmetric ? 0 : 1; // Use nonsymmetric permutation and scaling MPS
        iparm_[11] = 0;                 // Not in use
        iparm_[12] = symmetric ? 0 : 1; // Maximum weighted matching algorithm is
                                        // switched-off (default for symmetric). Try
                                        // iparm_[12] = 1 in case of inappropriate accuracy
        iparm_[13] = 0;                 // Output: Number of perturbed pivots
        iparm_[14] = 0;                 // Not in use
        iparm_[15] = 0;                 // Not in use
        iparm_[16] = 0;                 // Not in use
        iparm_[17] = -1;                // Output: Number of nonzeros in the factor LU
        iparm_[18] = -1;                // Output: Mflops for LU factorization
        iparm_[19] = 0;                 // Output: Numbers of CG Iterations

        iparm_[20] = 0; // 1x1 pivoting
        iparm_[23] = 1; // Two level
        // iparm_[24] = 1;  // Parsolve

        iparm_[26] = 0; // No matrix checker
        iparm_[27] = (sizeof(RealScalar) == 4) ? 1 : 0;
        iparm_[34] = 1; // C indexing
        iparm_[36] = 0; // CSR
        iparm_[59] = 0; // 0 - In-Core ; 1 - Automatic switch between In-Core and
                        // Out-of-Core modes ; 2 - Out-of-Core

        for (int i = 0; i < 64; i++)
            pt_[i] = 0;
        // memset(pt_, 0, sizeof(pt_));
    }

  protected:
    // cached data to reduce reallocation, etc.

    void manage_error_code(Index error) const {
        switch (error) {
        case 0:
            info_ = Success;
            break;
        case -4:
        case -7:
            info_ = NumericalIssue;
            break;
        default:
            info_ = InvalidInput;
        }
    }

    mutable SparseMatrixType matrix_;
    mutable ComputationInfo info_;
    bool analysis_is_ok_, factorization_is_ok_;
    StorageIndex type_, msglvl_;
    mutable void *pt_[64];
    mutable ParameterType iparm_;
    mutable IntColVectorType perm_;
    Index size_;

    int neigs_; // iparm[22]
    int peigs_; // iparm[21]
    int ppiv_;  // iparm[13]
    int flops_;
    int mem_;
};

template <class Derived> Derived &PardisoImpl<Derived>::compute(const MatrixType &a) {
    size_ = a.rows();
    eigen_assert(a.rows() == a.cols());

    pardiso_release();
    perm_.setZero(size_);
    derived().get_matrix(a);

    Index error;
    error = internal::pardiso_run_selector<StorageIndex>::run(
        pt_, 1, 1, type_, 12, internal::convert_index<StorageIndex>(size_), matrix_.valuePtr(),
        matrix_.outerIndexPtr(), matrix_.innerIndexPtr(), perm_.data(), 0, iparm_.data(), msglvl_,
        NULL, NULL);
    peigs_ = iparm_[21];
    neigs_ = iparm_[22];
    ppiv_ = iparm_[13];

    flops_ = iparm_[18];
    mem_ = iparm_[17];

    manage_error_code(error);
    analysis_is_ok_ = true;
    factorization_is_ok_ = true;
    m_isInitialized = true;
    return derived();
}
template <class Derived> Derived &PardisoImpl<Derived>::compute_internal() {
    size_ = matrix_.rows();

    pardiso_release();
    perm_.setZero(size_);

    Index error;
    error = internal::pardiso_run_selector<StorageIndex>::run(
        pt_, 1, 1, type_, 12, internal::convert_index<StorageIndex>(size_), matrix_.valuePtr(),
        matrix_.outerIndexPtr(), matrix_.innerIndexPtr(), perm_.data(), 0, iparm_.data(), msglvl_,
        NULL, NULL);
    peigs_ = iparm_[21];
    neigs_ = iparm_[22];
    ppiv_ = iparm_[13];
    flops_ = iparm_[18];
    mem_ = iparm_[17];

    manage_error_code(error);
    analysis_is_ok_ = true;
    factorization_is_ok_ = true;
    m_isInitialized = true;
    return derived();
}

template <class Derived> Derived &PardisoImpl<Derived>::analyze_pattern(const MatrixType &a) {
    size_ = a.rows();
    eigen_assert(size_ == a.cols());

    pardiso_release();
    perm_.setZero(size_);
    derived().get_matrix(a);

    Index error;
    error = internal::pardiso_run_selector<StorageIndex>::run(
        pt_, 1, 1, type_, 11, internal::convert_index<StorageIndex>(size_), matrix_.valuePtr(),
        matrix_.outerIndexPtr(), matrix_.innerIndexPtr(), perm_.data(), 0, iparm_.data(), msglvl_,
        NULL, NULL);

    manage_error_code(error);
    analysis_is_ok_ = true;
    factorization_is_ok_ = false;
    m_isInitialized = true;
    return derived();
}

template <class Derived> Derived &PardisoImpl<Derived>::analyze_pattern_internal() {
    size_ = matrix_.rows();
    pardiso_release();
    // perm_.setZero(size_);
    Index error;
    error = internal::pardiso_run_selector<StorageIndex>::run(
        pt_, 1, 1, type_, 11, internal::convert_index<StorageIndex>(size_), matrix_.valuePtr(),
        matrix_.outerIndexPtr(), matrix_.innerIndexPtr(), perm_.data(), 0, iparm_.data(), msglvl_,
        NULL, NULL);
    manage_error_code(error);
    analysis_is_ok_ = true;
    factorization_is_ok_ = false;
    m_isInitialized = true;
    return derived();
}

template <class Derived> Derived &PardisoImpl<Derived>::factorize(const MatrixType &a) {
    eigen_assert(analysis_is_ok_ && "You must first call analyze_pattern()");
    eigen_assert(size_ == a.rows() && size_ == a.cols());

    derived().get_matrix(a);

    Index error;
    error = internal::pardiso_run_selector<StorageIndex>::run(
        pt_, 1, 1, type_, 22, internal::convert_index<StorageIndex>(size_), matrix_.valuePtr(),
        matrix_.outerIndexPtr(), matrix_.innerIndexPtr(), perm_.data(), 0, iparm_.data(), msglvl_,
        NULL, NULL);
    peigs_ = iparm_[21];
    neigs_ = iparm_[22];
    ppiv_ = iparm_[13];
    flops_ = iparm_[18];
    mem_ = iparm_[17];

    manage_error_code(error);
    factorization_is_ok_ = true;
    return derived();
}
template <class Derived> Derived &PardisoImpl<Derived>::factorize_internal() {
    eigen_assert(analysis_is_ok_ && "You must first call analyze_pattern()");

    Index error;
    error = internal::pardiso_run_selector<StorageIndex>::run(
        pt_, 1, 1, type_, 22, internal::convert_index<StorageIndex>(size_), matrix_.valuePtr(),
        matrix_.outerIndexPtr(), matrix_.innerIndexPtr(), perm_.data(), 0, iparm_.data(), msglvl_,
        NULL, NULL);
    peigs_ = iparm_[21];
    neigs_ = iparm_[22];
    ppiv_ = iparm_[13];
    flops_ = iparm_[18];
    mem_ = iparm_[17];

    manage_error_code(error);
    factorization_is_ok_ = true;
    return derived();
}

template <class Derived>
template <typename BDerived, typename XDerived>
void PardisoImpl<Derived>::_solve_impl(const MatrixBase<BDerived> &b,
                                       MatrixBase<XDerived> &x) const {
    if (iparm_[0] == 0) // Factorization was not computed
    {
        info_ = InvalidInput;
        return;
    }

    // Index n = matrix_.rows();
    Index nrhs = Index(b.cols());
    eigen_assert(size_ == b.rows());
    eigen_assert(((MatrixBase<BDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) &&
                 "Row-major right hand sides are not supported");
    eigen_assert(((MatrixBase<XDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) &&
                 "Row-major matrices of unknowns are not supported");
    eigen_assert(((nrhs == 1) || b.outerStride() == b.rows()));

    //  switch (transposed) {
    //    case SvNoTrans    : iparm_[11] = 0 ; break;
    //    case SvTranspose  : iparm_[11] = 2 ; break;
    //    case SvAdjoint    : iparm_[11] = 1 ; break;
    //    default:
    //      //std::cerr << "Eigen: transposition  option \"" << transposed << "\"
    //      not supported by the PARDISO backend\n"; iparm_[11] = 0;
    //  }

    Scalar *rhs_ptr = const_cast<Scalar *>(b.derived().data());
    Matrix<Scalar, Dynamic, Dynamic, ColMajor> tmp;

    // Pardiso cannot solve in-place
    if (rhs_ptr == x.derived().data()) {
        tmp = b;
        rhs_ptr = tmp.data();
    }

    Index error;
    error = internal::pardiso_run_selector<StorageIndex>::run(
        pt_, 1, 1, type_, 33, internal::convert_index<StorageIndex>(size_), matrix_.valuePtr(),
        matrix_.outerIndexPtr(), matrix_.innerIndexPtr(), perm_.data(),
        internal::convert_index<StorageIndex>(nrhs), iparm_.data(), msglvl_, rhs_ptr,
        x.derived().data());

    manage_error_code(error);
}

/** \ingroup PardisoSupport_Module
 * \class PardisoLU
 * \brief A sparse direct LU factorization and solver based on the PARDISO
 * library
 *
 * This class allows to solve for A.X = B sparse linear problems via a direct LU
 * factorization using the Intel MKL PARDISO library. The sparse matrix A must
 * be squared and invertible. The vectors or matrices X and B can be either
 * dense or sparse.
 *
 * By default, it runs in in-core mode. To enable PARDISO's out-of-core feature,
 * set: \code solver.pardiso_parameter_array()[59] = 1; \endcode
 *
 * \tparam _MatrixType the type of the sparse matrix A, it must be a
 * SparseMatrix<>
 *
 * \implsparsesolverconcept
 *
 * \sa \ref TutorialSparseSolverConcept, class SparseLU
 */
template <typename MatrixType> class PardisoLU : public PardisoImpl<PardisoLU<MatrixType>> {
  protected:
    typedef PardisoImpl<PardisoLU> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    using Base::matrix_;
    using Base::pardiso_init;
    friend class PardisoImpl<PardisoLU<MatrixType>>;

  public:
    using Base::compute;
    using Base::solve;

    PardisoLU() : Base() { pardiso_init(Base::ScalarIsComplex ? 13 : 11); }

    explicit PardisoLU(const MatrixType &matrix) : Base() {
        pardiso_init(Base::ScalarIsComplex ? 13 : 11);
        compute(matrix);
    }

  protected:
    void get_matrix(const MatrixType &matrix) {
        matrix_ = matrix;
        matrix_.makeCompressed();
    }
};

/** \ingroup PardisoSupport_Module
 * \class PardisoLLT
 * \brief A sparse direct Cholesky (LLT) factorization and solver based on the
 * PARDISO library
 *
 * This class allows to solve for A.X = B sparse linear problems via a LL^T
 * Cholesky factorization using the Intel MKL PARDISO library. The sparse matrix
 * A must be selfajoint and positive definite. The vectors or matrices X and B
 * can be either dense or sparse.
 *
 * By default, it runs in in-core mode. To enable PARDISO's out-of-core feature,
 * set: \code solver.pardiso_parameter_array()[59] = 1; \endcode
 *
 * \tparam MatrixType the type of the sparse matrix A, it must be a
 * SparseMatrix<> \tparam UpLo can be any bitwise combination of Upper, Lower.
 * The default is Upper, meaning only the upper triangular part has to be used.
 *         Upper|Lower can be used to tell both triangular parts can be used as
 * input.
 *
 * \implsparsesolverconcept
 *
 * \sa \ref TutorialSparseSolverConcept, class SimplicialLLT
 */
template <typename MatrixType, int _UpLo>
class PardisoLLT : public PardisoImpl<PardisoLLT<MatrixType, _UpLo>> {
  protected:
    typedef PardisoImpl<PardisoLLT<MatrixType, _UpLo>> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    using Base::matrix_;
    using Base::pardiso_init;
    friend class PardisoImpl<PardisoLLT<MatrixType, _UpLo>>;

  public:
    typedef typename Base::StorageIndex StorageIndex;
    enum { UpLo = _UpLo };
    using Base::compute;

    PardisoLLT() : Base() { pardiso_init(Base::ScalarIsComplex ? 4 : 2); }

    explicit PardisoLLT(const MatrixType &matrix) : Base() {
        pardiso_init(Base::ScalarIsComplex ? 4 : 2);
        compute(matrix);
    }

  protected:
    void get_matrix(const MatrixType &matrix) {
        // PARDISO supports only upper, row-major matrices
        PermutationMatrix<Dynamic, Dynamic, StorageIndex> p_null;
        matrix_.resize(matrix.rows(), matrix.cols());
        matrix_.template selfadjointView<Upper>() =
            matrix.template selfadjointView<UpLo>().twistedBy(p_null);
        matrix_.makeCompressed();
    }
};

/** \ingroup PardisoSupport_Module
 * \class PardisoLDLT
 * \brief A sparse direct Cholesky (LDLT) factorization and solver based on the
 * PARDISO library
 *
 * This class allows to solve for A.X = B sparse linear problems via a LDL^T
 * Cholesky factorization using the Intel MKL PARDISO library. The sparse matrix
 * A is assumed to be selfajoint and positive definite. For complex matrices, A
 * can also be symmetric only, see the \a Options template parameter. The
 * vectors or matrices X and B can be either dense or sparse.
 *
 * By default, it runs in in-core mode. To enable PARDISO's out-of-core feature,
 * set: \code solver.pardiso_parameter_array()[59] = 1; \endcode
 *
 * \tparam MatrixType the type of the sparse matrix A, it must be a
 * SparseMatrix<> \tparam Options can be any bitwise combination of Upper,
 * Lower, and Symmetric. The default is Upper, meaning only the upper triangular
 * part has to be used. Symmetric can be used for symmetric, non-selfadjoint
 * complex matrices, the default being to assume a selfadjoint matrix.
 *         Upper|Lower can be used to tell both triangular parts can be used as
 * input.
 *
 * \implsparsesolverconcept
 *
 * \sa \ref TutorialSparseSolverConcept, class SimplicialLDLT
 */
template <typename MatrixType, int Options>
class PardisoLDLT : public PardisoImpl<PardisoLDLT<MatrixType, Options>> {
  protected:
    typedef PardisoImpl<PardisoLDLT<MatrixType, Options>> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    // typedef typename Base::Index Index;
    using Base::pardiso_init;

    // using Base::neigs_;
    // using Base::peigs_;
    friend class PardisoImpl<PardisoLDLT<MatrixType, Options>>;

  public:
    using Base::flops_;
    using Base::iparm_;
    using Base::matrix_;
    using Base::mem_;
    using Base::msglvl_;
    using Base::neigs_;
    using Base::peigs_;
    using Base::perm_;
    using Base::ppiv_;

    typedef typename Base::StorageIndex StorageIndex;
    using Base::compute;
    enum { UpLo = Options & (Upper | Lower) };

    PardisoLDLT() : Base() {
        pardiso_init(Base::ScalarIsComplex ? (bool(Options & Symmetric) ? 6 : -4) : -2);
    }

    explicit PardisoLDLT(const MatrixType &matrix) : Base() {
        pardiso_init(Base::ScalarIsComplex ? (bool(Options & Symmetric) ? 6 : -4) : -2);
        compute(matrix);
    }

    void get_matrix(const MatrixType &matrix) {
        // PARDISO supports only upper, row-major matrices
        PermutationMatrix<Dynamic, Dynamic, StorageIndex> p_null;
        matrix_.resize(matrix.rows(), matrix.cols());
        matrix_.template selfadjointView<Upper>() =
            matrix.template selfadjointView<UpLo>().twistedBy(p_null);
        matrix_.makeCompressed();
    }
    MatrixType get_matrix_twisted(const MatrixType &matrix) {
        PermutationMatrix<Dynamic, Dynamic, StorageIndex> p_null(perm_);
        MatrixType j;
        j.resize(matrix.rows(), matrix.cols());
        j.template selfadjointView<Upper>() =
            matrix.template selfadjointView<UpLo>().twistedBy(p_null);
        j.makeCompressed();
        return j;
    }

    inline int neigs() const { return neigs_; }
    inline int peigs() const { return peigs_; }
    inline int ppivs() const { return ppiv_; }
    void release() {
        Base::pardiso_release();
        this->matrix_.resize(0, 0);
        this->matrix_.data().squeeze();
        this->perm_.resize(0);
    }
    MatrixType &get_matrix() { return matrix_; }
    void set_matrix(const MatrixType &mat) { this->matrix_ = mat; }

    int ord_ = 3;
    int pivotstrat_ = 0;
    int pivotpert_ = 13;
    int matching_ = 0;
    int scaling_ = 0;
    int iterref_ = 2; // iparm[7]
    int alg_ = 0;
    int store_perm_ = 2;
    int parsolve_ = 0;
    int threads_ = 0;
    void set_params() {
        iparm_[0] = 1;           // No solver default
        iparm_[1] = ord_;        // use Metis for the ordering
        iparm_[2] = 0;           // Reserved. Set to zero. (??Numbers of processors, value
                                 // of OMP_NUM_THREADS??)
        iparm_[3] = 0;           // No iterative-direct algorithm
        iparm_[4] = store_perm_; // No user fill-in reducing permutation
        iparm_[5] = 0;           // Write solution into x, b is left unchanged
        iparm_[6] = 0;           // Not in use
        iparm_[7] = iterref_;    // Max numbers of iterative refinement steps
        iparm_[8] = 0;           // Not in use
        iparm_[9] = pivotpert_;  // Perturb the pivot elements with 1E-13
        iparm_[10] = scaling_;   // Use nonsymmetric permutation and scaling MPS
        iparm_[11] = 0;          // Not in use
        iparm_[12] = matching_;  // Maximum weighted matching algorithm is switched-off
                                 // (default for symmetric). Try iparm_[12] = 1 in case of
                                 // inappropriate accuracy
        iparm_[13] = 0;          // Output: Number of perturbed pivots
        iparm_[14] = 0;          // Not in use
        iparm_[15] = 0;          // Not in use
        iparm_[16] = 0;          // Not in use
        iparm_[17] = -1;         // Output: Number of nonzeros in the factor LU
        iparm_[18] = -1;         // Output: Mflops for LU factorization
        iparm_[19] = 0;          // Output: Numbers of CG Iterations

        iparm_[20] = pivotstrat_; // 1x1 pivoting
        iparm_[23] = alg_;        // Two level
        iparm_[24] = parsolve_;   // Two level

        iparm_[26] = 0; // No matrix checker
        iparm_[27] = (sizeof(RealScalar) == 4) ? 1 : 0;
        iparm_[33] = threads_;
        iparm_[34] = 1; // C indexing
        iparm_[36] = 0; // CSR
        iparm_[59] = 0; // 0 - In-Core ; 1 - Automatic switch between In-Core and
                        // Out-of-Core modes ; 2 - Out-of-Core
    }
};

} // end namespace Eigen

#include <Eigen/src/Core/util/ReenableStupidWarnings.h>

#endif // EIGEN_PARDISOSUPPORT_H
