// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// RK tableau A-row consistency.
//
// Butcher tableau row-sum identity:
//     c_i = Σⱼ A[i][j]
// for every non-trivial row i (the first stage has c_0 = 0 and no A row,
// so storage starts at row 1 and indexes i ∈ [0, Stages-2] over the A array).
//
// The existing compile-time checks in rk_coeffs.h cover:
//   - BmidStages schema (storage layout)
//   - Σ B == 1 (main-weight normalization)
//   - Σ Bmid == 1 (midpoint-weight normalization)
// but do NOT check the A/C row-sum identity. A single-digit typo in any
// A-row — the most common source of a subtly-broken tableau — would pass
// all existing checks and still silently break every integration of that
// method. This test closes that gap across all 8 user-selectable methods.

#include <tycho/detail/integrators/rk_coeffs.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

// Same tolerance used by the existing compile-time weight-sum checks
// (see rk_coeffs.h detail::kRKWeightTol).
constexpr double kARowTol = 1e-9;

// Sum every column of A row i (trailing zeros are harmless because A is
// stored as a fixed-width ragged array padded with zeros). C is indexed
// from 0 but corresponds to stage i+1, matching the documented A-row
// layout in rk_coeffs.h (row i holds coefficients for stage i+1 with
// c-value C[i]).
template <class RKRow> double row_sum(const RKRow &row) {
    double s = 0.0;
    for (std::size_t j = 0; j < row.size(); ++j) {
        s += row[j];
    }
    return s;
}

template <tycho::IVPAlg Alg> void check_a_row_consistency(const char *name) {
    using RK = tycho::integrators::RKCoeffs<Alg>;
    // A.size() is the number of rows in storage; each row i corresponds
    // to stage i+1, whose c-value is at C[i].
    ASSERT_EQ(RK::A.size(), RK::C.size())
        << name << ": A.size() must equal C.size() (one c-value per A-row).";
    for (std::size_t i = 0; i < RK::A.size(); ++i) {
        const double sum = row_sum(RK::A[i]);
        const double c = RK::C[i];
        EXPECT_NEAR(sum, c, kARowTol)
            << name << ": row " << i << " Σⱼ A[i][j] = " << sum
            << " but C[i] = " << c << "; |Δ| = " << std::abs(sum - c) << ".";
    }
}

} // namespace

TEST(RKTableauConsistencyTest, ARowSumsEqualC_AllUserMethods) {
    check_a_row_consistency<tycho::IVPAlg::DOPRI54>("DOPRI54");
    check_a_row_consistency<tycho::IVPAlg::DOPRI87>("DOPRI87");
    check_a_row_consistency<tycho::IVPAlg::Tsit5>("Tsit5");
    check_a_row_consistency<tycho::IVPAlg::BS3>("BS3");
    check_a_row_consistency<tycho::IVPAlg::BS5>("BS5");
    check_a_row_consistency<tycho::IVPAlg::Vern7>("Vern7");
    check_a_row_consistency<tycho::IVPAlg::Vern8>("Vern8");
    check_a_row_consistency<tycho::IVPAlg::Vern9>("Vern9");
}
