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

// MEE variant marked Vectorizable=true.  Body mirrors MEEBench in
// bench/cpp/bench_enzyme.cpp; trig routed via tycho::math::cos/sin so
// Phase 5b primal evaluation works under EnzymeAD.  Used to validate
// Phase 7 SIMD FoF on the composite trig+sqrt+division body that the
// Phase 6 SIMD FoR path cannot handle.
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct MEEVectorizable
    : tycho::vf::VectorFunction<MEEVectorizable<Jm, Hm>, 9, 6, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<MEEVectorizable<Jm, Hm>, 9, 6, Jm, Hm>;
    VF_TYPE_ALIASES(Base)

    static constexpr bool is_vectorizable = true;

    double mu_, sqm_;
    MEEVectorizable(double mu = 1.0) : mu_{mu}, sqm_{std::sqrt(mu)} {}

    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        using tycho::math::cos;
        using tycho::math::sin;
        using std::sqrt;
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        Scalar x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4];
        Scalar x5 = x[5], x6 = x[6], x7 = x[7], x8 = x[8];
        Scalar sqx0 = sqrt(x0);
        Scalar x9 = Scalar(1.0 / sqm_);
        Scalar x10 = cos(x5);
        Scalar x11 = sin(x5);
        Scalar x12 = x1 * x10 + x11 * x2;
        Scalar x13 = x12 + 1.0;
        Scalar x14 = 1.0 / x13;
        Scalar x15 = x14 * x7;
        Scalar x16 = x10 * x4;
        Scalar x17 = x11 * x3;
        Scalar x18 = x14 * x8;
        Scalar x19 = x12 + 2.0;
        Scalar x20 = sqx0 * x9;
        Scalar x21 = x18 * (-x16 + x17);
        Scalar x22 = 0.5 * x18 * x20 * ((x3 * x3) + (x4 * x4) + 1.0);
        fx[0] = 2.0 * (x0 * sqx0) * x15 * x9;
        fx[1] = x20 * (x11 * x6 + x15 * (x1 + x10 * x19) + x18 * x2 * (x16 - x17));
        fx[2] = x20 * (x1 * x21 - x10 * x6 + x15 * (x11 * x19 + x2));
        fx[3] = x10 * x22;
        fx[4] = x11 * x22;
        fx[5] = x20 * (mu_ * x13 * x13 / (x0 * x0) + 1.0 * x21);
    }
};

using MEEVectorizableEnzymeFull = MEEVectorizable<
    tycho::vf::DenseDerivativeMode::EnzymeAD,
    tycho::vf::DenseDerivativeMode::EnzymeAD>;

// IR=8, OR=4 synthetic fixture for the Phase 7+ doubly-batched FoF helper.
// IR is divisible by BW=4 so the doubly-batched rectangle covers the full
// matrix without tail handling — keeps the proof-of-concept test minimal.
// Body: simple polynomial mixture, no transcendentals; Vectorizable=true.
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct PolyVectorizable8x4
    : tycho::vf::VectorFunction<PolyVectorizable8x4<Jm, Hm>, 8, 4, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<PolyVectorizable8x4<Jm, Hm>, 8, 4, Jm, Hm>;
    VF_TYPE_ALIASES(Base)

    static constexpr bool is_vectorizable = true;

    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        // Quadratic + cubic mixture so the Hessian has non-trivial entries
        // across both diagonal and off-diagonal positions.
        fx[0] = x[0] * x[1] + x[2] * x[3] + x[4] * x[4];
        fx[1] = x[1] * x[2] * x[5] + x[6];
        fx[2] = x[3] + x[4] * x[5] + x[6] * x[7];
        fx[3] = x[0] * x[7] + x[2] * x[6] - x[5];
    }
};

using PolyVectorizable8x4EnzymeFull = PolyVectorizable8x4<
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
// Direct-SIMD Hessian vs scalarize-per-lane reference.  The
// `simd_compute_jacobian_adjointgradient_adjointhessian_impl` method is
// defined under both Hessian strategies (Phase 6 FoR and Phase 7 FoF
// combined J+H), so this helper works for either — the test cases below
// pick the strategy-specific name.  Guarded by the SIMD flag only.
// -----------------------------------------------------------------------------
#if defined(TYCHO_ENZYME_SIMD_HESSIAN_ENABLED)

template <class VF, int IR, int OR>
void check_hessian_simd_vs_scalarized(VF& f) {
    using SS = Eigen::Array<double, 4, 1>;

    Eigen::Matrix<SS, IR, 1> x;
    Eigen::Matrix<SS, OR, 1> lam;
    // Additive seeds keep x positive across all lanes — matters for MEE
    // (sqrt(x[0])) and other VFs with restricted input domains.
    for (int j = 0; j < IR; ++j)
        for (int lane = 0; lane < 4; ++lane)
            x(j)[lane] = 0.13 * (j + 1) + 0.05 * lane;
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

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)
// Phase 6 SIMD FoR cases.  Brach + CR3BP only — MEE-class composites
// (cos/sin/sqrt/division) opt out of FoR SIMD; see Phase 6 docs.
TEST(EnzymeVectorized, HessianSIMDMatchesScalarized_Brach) {
    tycho_enzyme_test::BrachVectorizableEnzymeFull f(32.2);
    check_hessian_simd_vs_scalarized<decltype(f), 5, 3>(f);
}

TEST(EnzymeVectorized, HessianSIMDMatchesScalarized_CR3BP) {
    tycho_enzyme_test::CR3BPVectorizableEnzymeFull f(0.0123);
    check_hessian_simd_vs_scalarized<decltype(f), 7, 6>(f);
}
#endif // ForwardOverReverse

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)
// Phase 7 SIMD FoF combined-J+H cases.  All three fixtures — including
// MEE — work under FoF SIMD (no reverse-mode tape, so the IR-deduction
// limit that forced MEE out of FoR SIMD doesn't apply).
TEST(EnzymeVectorized, HessianFoFSIMDMatchesScalarized_Brach) {
    tycho_enzyme_test::BrachVectorizableEnzymeFull f(32.2);
    check_hessian_simd_vs_scalarized<decltype(f), 5, 3>(f);
}

TEST(EnzymeVectorized, HessianFoFSIMDMatchesScalarized_CR3BP) {
    tycho_enzyme_test::CR3BPVectorizableEnzymeFull f(0.0123);
    check_hessian_simd_vs_scalarized<decltype(f), 7, 6>(f);
}

TEST(EnzymeVectorized, HessianFoFSIMDMatchesScalarized_MEE) {
    tycho_enzyme_test::MEEVectorizableEnzymeFull f(1.0);
    check_hessian_simd_vs_scalarized<decltype(f), 9, 6>(f);
}

// Phase 7+ doubly-batched FoF helper — proof-of-concept on a synthetic
// IR=8 OR=4 fixture (IR divisible by BW=4 so the full matrix is covered
// by the doubly-batched rectangle, no tail handling).  Validates that
// nested __enzyme_fwddiff(enzyme_width=BW) over inner __enzyme_fwddiff
// (enzyme_width=BW) composes correctly and produces BW² Hessian elements
// per outer call.
TEST(EnzymeVectorized, HessianFoFSIMDdb_PolyMatchesScalarized) {
    using SS = Eigen::Array<double, 4, 1>;
    constexpr int IR = 8;
    constexpr int OR = 4;
    tycho_enzyme_test::PolyVectorizable8x4EnzymeFull f;

    Eigen::Matrix<SS, IR, 1> x;
    Eigen::Matrix<SS, OR, 1> lam;
    for (int j = 0; j < IR; ++j)
        for (int lane = 0; lane < 4; ++lane)
            x(j)[lane] = 0.13 * (j + 1) + 0.05 * lane;
    for (int j = 0; j < OR; ++j)
        for (int lane = 0; lane < 4; ++lane)
            lam(j)[lane] = 0.4 - 0.05 * j + 0.03 * lane;

    Eigen::Matrix<SS, OR, 1> fx_simd;
    Eigen::Matrix<SS, OR, IR> jx_simd;
    Eigen::Matrix<SS, IR, IR> h_simd;
    fx_simd.setZero(); jx_simd.setZero(); h_simd.setZero();

    f.compute_jacobian_adjoint_hessian_fof_simd_db_(
        x, fx_simd, jx_simd, h_simd, lam, IR, OR);

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
        for (int i = 0; i < IR; ++i)
            for (int j = 0; j < IR; ++j)
                EXPECT_NEAR(h_simd(j, i)[lane], h_lane(j, i), 1e-10)
                    << "lane=" << lane << " h(" << j << "," << i << ")";
    }
}
#endif // ForwardOverForward

#endif // TYCHO_ENZYME_SIMD_HESSIAN_ENABLED

} // namespace
