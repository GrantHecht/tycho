// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// Phase 5a: SuperScalar dispatch via scalarize-per-lane.
//
// Validates that a Vectorizable<Derived>=true VectorFunction with EnzymeAD
// produces lane-equivalent results to the scalar EnzymeAD path applied to
// each lane independently.  Phase 1 forbade this combination; Phase 5a lifts
// the restriction.
// =============================================================================

#include <gtest/gtest.h>

#include <tycho/tycho.h>

#include "tycho/detail/utils/simd_math.h"

#include "enzyme_test_dynamics.h"

namespace tycho_enzyme_test {

// Brachistochrone variant marked vectorizable. Body is identical to
// BrachODEModed except for the is_vectorizable opt-in.
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct BrachVectorizable
    : tycho::vf::VectorFunction<BrachVectorizable<Jm, Hm>, 5, 3, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<BrachVectorizable<Jm, Hm>, 5, 3, Jm, Hm>;
    VF_TYPE_ALIASES(Base)

    static constexpr bool is_vectorizable = true;

    double g_;
    BrachVectorizable(double g = 32.2) : g_{g} {}

    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        using tycho::math::cos;
        using tycho::math::sin;
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();

        const Scalar v = x[2];
        const Scalar theta = x[4];
        fx[0] = sin(theta) * v;
        fx[1] = -cos(theta) * v;
        fx[2] = Scalar(g_) * cos(theta);
    }
};

using BrachVectorizableEnzymeAD = BrachVectorizable<
    tycho::vf::DenseDerivativeMode::EnzymeAD,
    tycho::vf::DenseDerivativeMode::FDiffFwd>;

using BrachVectorizableEnzymeFull = BrachVectorizable<
    tycho::vf::DenseDerivativeMode::EnzymeAD,
    tycho::vf::DenseDerivativeMode::EnzymeAD>;

// CR3BP variant marked Vectorizable=true (sqrt-only body — Phase 6 SIMD
// Hessian eligible).
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct CR3BPVectorizable
    : tycho::vf::VectorFunction<CR3BPVectorizable<Jm, Hm>, 7, 6, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<CR3BPVectorizable<Jm, Hm>, 7, 6, Jm, Hm>;
    VF_TYPE_ALIASES(Base)

    static constexpr bool is_vectorizable = true;

    double mu_;
    CR3BPVectorizable(double mu = 0.0123) : mu_{mu} {}

    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        using std::sqrt;
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        tycho::Vector3<Scalar> X = x.template head<3>();
        tycho::Vector3<Scalar> V = x.template segment<3>(3);
        tycho::Vector3<Scalar> p1loc; p1loc[0] = Scalar(-mu_);
        tycho::Vector3<Scalar> p2loc; p2loc[0] = Scalar(1.0 - mu_);
        tycho::Vector3<Scalar> dvec = X - p1loc;
        tycho::Vector3<Scalar> rvec = X - p2loc;
        Scalar d = sqrt((dvec.array() * dvec.array()).sum());
        Scalar r = sqrt((rvec.array() * rvec.array()).sum());
        fx.template head<3>() = V;
        fx.template segment<3>(3) =
            (Scalar(-(1.0 - mu_)) / (d * d * d)) * dvec
          + (Scalar(-mu_) / (r * r * r)) * rvec;
        fx[3] += Scalar(2.0) * V[1] + X[0];
        fx[4] += Scalar(-2.0) * V[0] + X[1];
    }
};

using CR3BPVectorizableEnzymeFull = CR3BPVectorizable<
    tycho::vf::DenseDerivativeMode::EnzymeAD,
    tycho::vf::DenseDerivativeMode::EnzymeAD>;

} // namespace tycho_enzyme_test

namespace {

// -----------------------------------------------------------------------------
// Phase 5a primary check: the Vectorizable+EnzymeAD combination compiles
// (Phase 1 had a static_assert blocking this; Phase 5a's if-constexpr lifts
// it) and computes correctly per lane.
// -----------------------------------------------------------------------------
TEST(EnzymeVectorized, JacobianMatchesScalar) {
    using SS = Eigen::Array<double, 4, 1>;
    tycho_enzyme_test::BrachVectorizableEnzymeAD f(32.2);

    Eigen::Matrix<SS, 5, 1> x;
    for (int j = 0; j < 5; ++j)
        for (int lane = 0; lane < 4; ++lane)
            x(j)[lane] = 0.1 * (j + 1) * (lane + 1);

    Eigen::Matrix<SS, 3, 1> fx;
    Eigen::Matrix<SS, 3, 5> jx;
    fx.setZero();
    jx.setZero();

    f.compute_jacobian(x, fx, jx);

    // Per-lane equality vs the scalar path.
    tycho_enzyme_test::BrachEnzymeAD f_scalar(32.2);
    for (int lane = 0; lane < 4; ++lane) {
        Eigen::Matrix<double, 5, 1> x_lane;
        for (int j = 0; j < 5; ++j) x_lane[j] = x(j)[lane];

        Eigen::Matrix<double, 3, 1> fx_lane;
        Eigen::Matrix<double, 3, 5> jx_lane;
        fx_lane.setZero();
        jx_lane.setZero();
        f_scalar.compute_jacobian(x_lane, fx_lane, jx_lane);

        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(fx(j)[lane], fx_lane[j], 1e-12)
                << "lane=" << lane << " fx[" << j << "]";
        }
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 5; ++i) {
                EXPECT_NEAR(jx(j, i)[lane], jx_lane(j, i), 1e-12)
                    << "lane=" << lane << " jx(" << j << "," << i << ")";
            }
    }
}

// -----------------------------------------------------------------------------
// Vectorized full-Enzyme Hessian: <EnzymeAD, EnzymeAD> with is_vectorizable.
// Per-lane comparison against the scalar full-Enzyme path.
// -----------------------------------------------------------------------------
TEST(EnzymeVectorized, HessianMatchesScalar) {
    using SS = Eigen::Array<double, 4, 1>;
    tycho_enzyme_test::BrachVectorizableEnzymeFull f(32.2);

    Eigen::Matrix<SS, 5, 1> x;
    Eigen::Matrix<SS, 3, 1> lam;
    for (int j = 0; j < 5; ++j)
        for (int lane = 0; lane < 4; ++lane)
            x(j)[lane] = 0.1 * (j + 1) * (lane + 1);
    for (int j = 0; j < 3; ++j)
        for (int lane = 0; lane < 4; ++lane)
            lam(j)[lane] = 0.5 + 0.1 * j - 0.05 * lane;

    Eigen::Matrix<SS, 3, 1> fx;
    Eigen::Matrix<SS, 3, 5> jx;
    Eigen::Matrix<SS, 5, 1> g;
    Eigen::Matrix<SS, 5, 5> h;
    fx.setZero(); jx.setZero(); g.setZero(); h.setZero();

    f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, g, h, lam);

    tycho_enzyme_test::BrachEnzymeFull f_scalar(32.2);
    for (int lane = 0; lane < 4; ++lane) {
        Eigen::Matrix<double, 5, 1> x_lane;
        Eigen::Matrix<double, 3, 1> lam_lane;
        for (int j = 0; j < 5; ++j) x_lane[j] = x(j)[lane];
        for (int j = 0; j < 3; ++j) lam_lane[j] = lam(j)[lane];

        Eigen::Matrix<double, 3, 1> fx_lane;
        Eigen::Matrix<double, 3, 5> jx_lane;
        Eigen::Matrix<double, 5, 1> g_lane;
        Eigen::Matrix<double, 5, 5> h_lane;
        fx_lane.setZero(); jx_lane.setZero();
        g_lane.setZero();  h_lane.setZero();

        f_scalar.compute_jacobian_adjointgradient_adjointhessian(
            x_lane, fx_lane, jx_lane, g_lane, h_lane, lam_lane);

        for (int i = 0; i < 5; ++i) {
            EXPECT_NEAR(g(i)[lane], g_lane[i], 1e-12)
                << "lane=" << lane << " g[" << i << "]";
            for (int j = 0; j < 5; ++j)
                EXPECT_NEAR(h(j, i)[lane], h_lane(j, i), 1e-10)
                    << "lane=" << lane << " h(" << j << "," << i << ")";
        }
    }
}

// -----------------------------------------------------------------------------
// Phase 6 direct-SIMD vs Phase 5a scalarize-per-lane.  Calls the SIMD impl
// directly (bypassing dispatch) so the comparison holds regardless of how
// TYCHO_ENZYME_SIMD_HESSIAN routes a given input.  Guarded by the same flag:
// when the SIMD impl isn't compiled, the test is skipped.
// -----------------------------------------------------------------------------
#if defined(TYCHO_ENZYME_SIMD_HESSIAN_ENABLED) \
    && defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)

template <class VF, int IR, int OR>
void check_hessian_simd_vs_scalarized(VF& f) {
    using SS = Eigen::Array<double, 4, 1>;

    Eigen::Matrix<SS, IR, 1> x;
    Eigen::Matrix<SS, OR, 1> lam;
    for (int j = 0; j < IR; ++j)
        for (int lane = 0; lane < 4; ++lane)
            x(j)[lane] = 0.13 * (j + 1) - 0.07 * lane;
    for (int j = 0; j < OR; ++j)
        for (int lane = 0; lane < 4; ++lane)
            lam(j)[lane] = 0.4 - 0.05 * j + 0.03 * lane;

    Eigen::Matrix<SS, OR, 1> fx_simd;
    Eigen::Matrix<SS, OR, IR> jx_simd;
    Eigen::Matrix<SS, IR, 1> g_simd;
    Eigen::Matrix<SS, IR, IR> h_simd;
    fx_simd.setZero(); jx_simd.setZero();
    g_simd.setZero(); h_simd.setZero();

    f.simd_compute_jacobian_adjointgradient_adjointhessian_impl(
        x, fx_simd, jx_simd, g_simd, h_simd, lam);

    for (int lane = 0; lane < 4; ++lane) {
        Eigen::Matrix<double, IR, 1> x_lane;
        Eigen::Matrix<double, OR, 1> lam_lane;
        for (int j = 0; j < IR; ++j) x_lane[j] = x(j)[lane];
        for (int j = 0; j < OR; ++j) lam_lane[j] = lam(j)[lane];

        Eigen::Matrix<double, OR, 1> fx_lane;
        Eigen::Matrix<double, OR, IR> jx_lane;
        Eigen::Matrix<double, IR, 1> g_lane;
        Eigen::Matrix<double, IR, IR> h_lane;
        fx_lane.setZero(); jx_lane.setZero();
        g_lane.setZero();  h_lane.setZero();

        f.scalar_compute_jacobian_adjointgradient_adjointhessian_impl(
            x_lane, fx_lane, jx_lane, g_lane, h_lane, lam_lane);

        for (int j = 0; j < OR; ++j)
            EXPECT_NEAR(fx_simd(j)[lane], fx_lane[j], 1e-10)
                << "lane=" << lane << " fx[" << j << "]";
        for (int j = 0; j < OR; ++j)
            for (int i = 0; i < IR; ++i)
                EXPECT_NEAR(jx_simd(j, i)[lane], jx_lane(j, i), 1e-10)
                    << "lane=" << lane << " jx(" << j << "," << i << ")";
        for (int i = 0; i < IR; ++i) {
            EXPECT_NEAR(g_simd(i)[lane], g_lane[i], 1e-10)
                << "lane=" << lane << " g[" << i << "]";
            for (int j = 0; j < IR; ++j)
                EXPECT_NEAR(h_simd(j, i)[lane], h_lane(j, i), 1e-10)
                    << "lane=" << lane << " h(" << j << "," << i << ")";
        }
    }
}

TEST(EnzymeVectorized, HessianSIMDMatchesScalarized_Brach) {
    tycho_enzyme_test::BrachVectorizableEnzymeFull f(32.2);
    check_hessian_simd_vs_scalarized<decltype(f), 5, 3>(f);
}

TEST(EnzymeVectorized, HessianSIMDMatchesScalarized_CR3BP) {
    tycho_enzyme_test::CR3BPVectorizableEnzymeFull f(0.0123);
    check_hessian_simd_vs_scalarized<decltype(f), 7, 6>(f);
}

#endif // TYCHO_ENZYME_SIMD_HESSIAN_ENABLED && ForwardOverReverse

} // namespace
