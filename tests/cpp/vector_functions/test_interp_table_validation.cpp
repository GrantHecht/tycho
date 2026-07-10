// Gridded-table validation regression tests (OC review §3.3 + §3.1 cluster).
//
// §3.3: InterpTable1D::set_data / InterpTable2D::set_data used a strict `<`
// comparison to check for an ascending grid, so two consecutive equal nodes
// (a duplicate abscissa) were silently accepted. A duplicate node produces a
// zero grid step, and later interpolation divides by that step -- a silent
// divide-by-zero. The comparison is now `<=`, rejecting duplicates outright.
//
// The same defect existed verbatim in the InterpTable3D / InterpTable4D
// constructors (copies of the same validation loop), so those are fixed and
// covered here as well.
//
// Also covers the companion fixes: ChebTable::from_values (both the 1-D and
// N-D overloads) now rejects non-finite (NaN/Inf) grid samples instead of
// letting them propagate through the DCT-I transform and poison every
// Chebyshev coefficient; ChebFunction's shared_ptr<ChebTable> constructor now
// calls require_populated() so a non-null but default-constructed (empty)
// table is rejected up front instead of producing garbage evaluations; and
// the InterpFunction1D/2D/3D/4D shared_ptr<InterpTable...> constructors now
// reject a null table instead of dereferencing it in the constructor body.

#include "test_utils.h"
#include <fmt/color.h> // Eigen::Tensor TUs need this directly (see CMakeLists note)
#include <gtest/gtest.h>
#include <tycho/tycho.h>

#include <cmath>
#include <limits>

using namespace tycho;
using namespace TychoTest;

namespace {

// Five ascending time nodes with a duplicate at index 1/2 -- the minimum
// table size (tsize_ >= 5) with exactly one non-strictly-ascending pair.
Eigen::VectorXd duplicate_ts() {
    Eigen::VectorXd ts(5);
    ts << 0.0, 1.0, 1.0, 2.0, 3.0;
    return ts;
}

Eigen::VectorXd strictly_ascending_ts() {
    Eigen::VectorXd ts(5);
    ts << 0.0, 1.0, 2.0, 3.0, 4.0;
    return ts;
}

} // namespace

class InterpTableValidationTest : public VectorFunctionFixture {};

// --- §3.3: InterpTable1D strictly-ascending grid -----------------------------

TEST_F(InterpTableValidationTest, DuplicateNodesRejected1D) {
    auto ts = duplicate_ts();
    Eigen::VectorXd vs(5);
    vs << 0.0, 1.0, 1.0, 4.0, 9.0;
    EXPECT_THROW((oc::InterpTable1D(ts, vs, InterpType::Cubic)), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, StrictlyAscendingNodesAccepted1D) {
    // Negative control: a genuinely strictly-ascending grid must still work.
    auto ts = strictly_ascending_ts();
    Eigen::VectorXd vs(5);
    vs << 0.0, 1.0, 4.0, 9.0, 16.0;
    EXPECT_NO_THROW((oc::InterpTable1D(ts, vs, InterpType::Cubic)));
}

// --- §3.3: InterpTable2D strictly-ascending grids (both axes) ---------------

TEST_F(InterpTableValidationTest, DuplicateNodesRejected2D) {
    auto xs = duplicate_ts(); // duplicate on the X axis
    auto ys = strictly_ascending_ts();
    oc::InterpTable2D::MatType zs = oc::InterpTable2D::MatType::Zero(5, 5);
    EXPECT_THROW((oc::InterpTable2D(xs, ys, zs, InterpType::Cubic)), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, DuplicateYNodesRejected2D) {
    auto xs = strictly_ascending_ts();
    auto ys = duplicate_ts(); // duplicate on the Y axis
    oc::InterpTable2D::MatType zs = oc::InterpTable2D::MatType::Zero(5, 5);
    EXPECT_THROW((oc::InterpTable2D(xs, ys, zs, InterpType::Cubic)), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, StrictlyAscendingNodesAccepted2D) {
    auto xs = strictly_ascending_ts();
    auto ys = strictly_ascending_ts();
    oc::InterpTable2D::MatType zs = oc::InterpTable2D::MatType::Zero(5, 5);
    EXPECT_NO_THROW((oc::InterpTable2D(xs, ys, zs, InterpType::Cubic)));
}

// --- §3.3: InterpTable3D / InterpTable4D strictly-ascending grids -----------

TEST_F(InterpTableValidationTest, DuplicateNodesRejected3D) {
    auto xs = strictly_ascending_ts();
    auto ys = duplicate_ts(); // duplicate on the Y axis
    auto zs = strictly_ascending_ts();
    Eigen::Tensor<double, 3> fs(5, 5, 5);
    fs.setZero();
    EXPECT_THROW((oc::InterpTable3D(xs, ys, zs, fs, InterpType::Cubic, false)),
                 std::invalid_argument);
}

TEST_F(InterpTableValidationTest, DuplicateNodesRejected4D) {
    auto xs = strictly_ascending_ts();
    auto ys = strictly_ascending_ts();
    auto zs = strictly_ascending_ts();
    auto ws = duplicate_ts(); // duplicate on the W axis
    Eigen::Tensor<double, 4> fs(5, 5, 5, 5);
    fs.setZero();
    EXPECT_THROW((oc::InterpTable4D(xs, ys, zs, ws, fs, InterpType::Cubic, false)),
                 std::invalid_argument);
}

// --- InterpFunction null-table guards (1D/2D/3D/4D wrappers) -----------------

TEST_F(InterpTableValidationTest, NullTableRejected1DFunction) {
    std::shared_ptr<oc::InterpTable1D> null_tab;
    EXPECT_THROW((oc::InterpFunction1D<-1>(null_tab)), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, NullTableRejected2DFunction) {
    std::shared_ptr<oc::InterpTable2D> null_tab;
    EXPECT_THROW((oc::InterpFunction2D(null_tab)), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, NullTableRejected3DFunction) {
    std::shared_ptr<oc::InterpTable3D> null_tab;
    EXPECT_THROW((oc::InterpFunction3D(null_tab)), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, NullTableRejected4DFunction) {
    std::shared_ptr<oc::InterpTable4D> null_tab;
    EXPECT_THROW((oc::InterpFunction4D(null_tab)), std::invalid_argument);
}

// --- ChebTable::from_values non-finite sample rejection ---------------------

TEST_F(InterpTableValidationTest, ChebFromValuesRejectsNonFinite) {
    const int order = 4;
    oc::ChebTable::MatType vals(order + 1, 1);
    vals.setOnes();
    vals(2, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(oc::ChebTable::from_values(vals, 0.0, 1.0, order), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, ChebFromValuesRejectsInfinite) {
    const int order = 4;
    oc::ChebTable::MatType vals(order + 1, 1);
    vals.setOnes();
    vals(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(oc::ChebTable::from_values(vals, 0.0, 1.0, order), std::invalid_argument);
}

TEST_F(InterpTableValidationTest, ChebFromValuesNdRejectsNonFinite) {
    std::vector<int> orders{3, 3};
    Eigen::VectorXd lb(2), ub(2);
    lb << 0.0, 0.0;
    ub << 1.0, 1.0;
    oc::ChebTable::MatType vals =
        oc::ChebTable::MatType::Ones((orders[0] + 1) * (orders[1] + 1), 1);
    vals(5, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(oc::ChebTable::from_values(vals, lb, ub, orders), std::invalid_argument);
}

// --- ChebFunction rejects a non-null but unpopulated (default-constructed) table

TEST_F(InterpTableValidationTest, ChebFunctionRejectsUnpopulatedTable) {
    auto tab = std::make_shared<oc::ChebTable>(); // default ctor -- never populated
    EXPECT_THROW((oc::ChebFunction<-1>(tab)), std::invalid_argument);
}
