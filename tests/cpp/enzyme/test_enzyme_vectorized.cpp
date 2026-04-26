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
        using std::cos;
        using std::sin;
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
    tycho::vf::DenseDerivativeMode::AutodiffFwd>;

using BrachVectorizableEnzymeFull = BrachVectorizable<
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

} // namespace
