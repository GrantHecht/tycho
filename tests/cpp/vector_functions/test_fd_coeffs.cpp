// =============================================================================
// Tests for tycho::vf::FDCoeffs (VF_REVIEW 1.9): moment-condition coverage
// over every table in fd_coeffs.h, plus a regression guard for the
// backward-stencil flip() out-of-bounds read.
// =============================================================================

#include <cmath>

#include <gtest/gtest.h>

#include <tycho/detail/vf/derivatives/fd_coeffs.h>

using namespace tycho::vf;

namespace {

// A finite-difference stencil of order `order` using nodes
// first_node, first_node+1, ..., first_node+N-1 must satisfy the moment
// conditions:
//   sum_k weights[k] * node(k)^m == 0        for m != order
//   sum_k weights[k] * node(k)^m == order!   for m == order
// This is both the definition of a consistent FD stencil and a strong check
// that the table wasn't transcribed incorrectly (wrong node count, wrong
// coefficient, wrong sign, etc.).
template <class Table> void check_moments(int order, int first_node) {
    constexpr int N = Table::N;
    ASSERT_EQ(static_cast<int>(Table::weights.size()), N);
    double fact = 1.0;
    for (int m = 1; m <= order; m++) {
        fact *= m;
    }
    for (int m = 0; m <= order; m++) {
        double s = 0;
        for (int k = 0; k < N; k++) {
            s += Table::weights[k] * std::pow(double(first_node + k), m);
        }
        EXPECT_NEAR(s, m == order ? fact : 0.0, 1e-6) << "moment m=" << m << " order=" << order;
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Forward stencils, Shift = 0: nodes 0 .. N-1.
// -----------------------------------------------------------------------------
TEST(FDCoeffs, ForwardShift0Tables) {
    check_moments<FDCoeffs<1, 2, FDCoeffType::Forwards, 0>>(1, 0);
    check_moments<FDCoeffs<1, 4, FDCoeffType::Forwards, 0>>(1, 0);
    check_moments<FDCoeffs<1, 6, FDCoeffType::Forwards, 0>>(1, 0);
    check_moments<FDCoeffs<2, 2, FDCoeffType::Forwards, 0>>(2, 0);
    check_moments<FDCoeffs<2, 4, FDCoeffType::Forwards, 0>>(2, 0);
    check_moments<FDCoeffs<2, 6, FDCoeffType::Forwards, 0>>(2, 0);
    check_moments<FDCoeffs<3, 2, FDCoeffType::Forwards, 0>>(3, 0);
    check_moments<FDCoeffs<3, 4, FDCoeffType::Forwards, 0>>(3, 0);
    check_moments<FDCoeffs<3, 6, FDCoeffType::Forwards, 0>>(3, 0); // corrupt pre-fix
    check_moments<FDCoeffs<4, 2, FDCoeffType::Forwards, 0>>(4, 0);
    check_moments<FDCoeffs<4, 4, FDCoeffType::Forwards, 0>>(4, 0);
}

// -----------------------------------------------------------------------------
// Forward stencils, Shift = 1: nodes -1 .. N-2.
// -----------------------------------------------------------------------------
TEST(FDCoeffs, ForwardShift1Tables) {
    check_moments<FDCoeffs<1, 4, FDCoeffType::Forwards, 1>>(1, -1);
    check_moments<FDCoeffs<1, 6, FDCoeffType::Forwards, 1>>(1, -1);
    check_moments<FDCoeffs<2, 4, FDCoeffType::Forwards, 1>>(2, -1);
    check_moments<FDCoeffs<2, 6, FDCoeffType::Forwards, 1>>(2, -1);
    check_moments<FDCoeffs<3, 2, FDCoeffType::Forwards, 1>>(3, -1);
    check_moments<FDCoeffs<3, 4, FDCoeffType::Forwards, 1>>(3, -1);
    check_moments<FDCoeffs<3, 6, FDCoeffType::Forwards, 1>>(3, -1);
    check_moments<FDCoeffs<4, 2, FDCoeffType::Forwards, 1>>(4, -1);
    check_moments<FDCoeffs<4, 4, FDCoeffType::Forwards, 1>>(4, -1);
}

// -----------------------------------------------------------------------------
// Forward stencils, Shift = 2: nodes -2 .. N-3.
// -----------------------------------------------------------------------------
TEST(FDCoeffs, ForwardShift2Tables) {
    check_moments<FDCoeffs<1, 6, FDCoeffType::Forwards, 2>>(1, -2);
    check_moments<FDCoeffs<2, 6, FDCoeffType::Forwards, 2>>(2, -2);
    check_moments<FDCoeffs<3, 4, FDCoeffType::Forwards, 2>>(3, -2);
    check_moments<FDCoeffs<3, 6, FDCoeffType::Forwards, 2>>(3, -2);
    check_moments<FDCoeffs<4, 4, FDCoeffType::Forwards, 2>>(4, -2);
}

// -----------------------------------------------------------------------------
// Central stencils: nodes -(N-1)/2 .. (N-1)/2.
// -----------------------------------------------------------------------------
TEST(FDCoeffs, CentralTables) {
    check_moments<FDCoeffs<1, 2, FDCoeffType::Central, 0>>(1, -1);
    check_moments<FDCoeffs<1, 4, FDCoeffType::Central, 0>>(1, -2);
    check_moments<FDCoeffs<1, 6, FDCoeffType::Central, 0>>(1, -3);
    check_moments<FDCoeffs<1, 8, FDCoeffType::Central, 0>>(1, -4);
    check_moments<FDCoeffs<2, 2, FDCoeffType::Central, 0>>(2, -1);
    check_moments<FDCoeffs<2, 4, FDCoeffType::Central, 0>>(2, -2);
    check_moments<FDCoeffs<2, 6, FDCoeffType::Central, 0>>(2, -3);
    check_moments<FDCoeffs<2, 8, FDCoeffType::Central, 0>>(2, -4);
    check_moments<FDCoeffs<3, 2, FDCoeffType::Central, 0>>(3, -2);
    check_moments<FDCoeffs<3, 4, FDCoeffType::Central, 0>>(3, -3);
    check_moments<FDCoeffs<3, 6, FDCoeffType::Central, 0>>(3, -4);
    check_moments<FDCoeffs<4, 2, FDCoeffType::Central, 0>>(4, -2);
    check_moments<FDCoeffs<4, 4, FDCoeffType::Central, 0>>(4, -3);
    check_moments<FDCoeffs<4, 6, FDCoeffType::Central, 0>>(4, -4);
}

// -----------------------------------------------------------------------------
// Backward stencils: derived generically from the matching Forward table by
// FDCoeffs<Order, Accuracy, Backwards, Shift>::flip(). Pre-fix, flip() reads
// a[N - i], which is out-of-bounds at i == 0 -- a constexpr hard compile
// error the first time any Backwards table is odr-used. Nodes run
// Shift-(N-1) .. Shift.
// -----------------------------------------------------------------------------
TEST(FDCoeffs, BackwardFlipWellFormed) {
    using B1 = FDCoeffs<1, 2, FDCoeffType::Backwards, 0>;
    check_moments<B1>(1, -(B1::N - 1));

    using B2 = FDCoeffs<2, 2, FDCoeffType::Backwards, 0>;
    check_moments<B2>(2, -(B2::N - 1));

    using B3 = FDCoeffs<3, 2, FDCoeffType::Backwards, 0>;
    check_moments<B3>(3, -(B3::N - 1));

    using B4 = FDCoeffs<4, 2, FDCoeffType::Backwards, 0>;
    check_moments<B4>(4, -(B4::N - 1));

    // Non-zero shift: nodes run Shift-(N-1) .. Shift.
    using B3s1 = FDCoeffs<3, 4, FDCoeffType::Backwards, 1>;
    check_moments<B3s1>(3, 1 - (B3s1::N - 1));

    using B4s2 = FDCoeffs<4, 4, FDCoeffType::Backwards, 2>;
    check_moments<B4s2>(4, 2 - (B4s2::N - 1));
}
