// Unit tests for ChebTable's rank-generic multi-index / stride helpers.
#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

class ChebNdIndexTest : public VectorFunctionFixture {};

// row-major strides for shape {3,4,5}: {20,5,1}; tsize 60; round-trip flatten.
TEST_F(ChebNdIndexTest, StridesAndRoundTrip) {
    std::vector<int> shape{3, 4, 5};
    auto strides = oc::ChebTable::row_major_strides(shape);  // static helper
    ASSERT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[0], 20);
    EXPECT_EQ(strides[1], 5);
    EXPECT_EQ(strides[2], 1);
    long tsize = 3 * 4 * 5;
    for (long flat = 0; flat < tsize; ++flat) {
        std::vector<int> mi;
        oc::ChebTable::unflatten(flat, shape, strides, mi);
        EXPECT_EQ(oc::ChebTable::flat_index(mi, strides), flat);
    }
}

// hess_index: upper-triangle packing for dim=3 ({0,1,2}x{0,1,2}, j<=k).
// Expected packed order: (0,0)=0, (0,1)=1, (0,2)=2, (1,1)=3, (1,2)=4, (2,2)=5
TEST_F(ChebNdIndexTest, HessIndexDim3) {
    EXPECT_EQ(oc::ChebTable::hess_index(0, 0, 3), 0);
    EXPECT_EQ(oc::ChebTable::hess_index(0, 1, 3), 1);
    EXPECT_EQ(oc::ChebTable::hess_index(0, 2, 3), 2);
    EXPECT_EQ(oc::ChebTable::hess_index(1, 1, 3), 3);
    EXPECT_EQ(oc::ChebTable::hess_index(1, 2, 3), 4);
    EXPECT_EQ(oc::ChebTable::hess_index(2, 2, 3), 5);
    // Symmetry: (j,k) == (k,j)
    EXPECT_EQ(oc::ChebTable::hess_index(1, 0, 3), oc::ChebTable::hess_index(0, 1, 3));
    EXPECT_EQ(oc::ChebTable::hess_index(2, 0, 3), oc::ChebTable::hess_index(0, 2, 3));
    EXPECT_EQ(oc::ChebTable::hess_index(2, 1, 3), oc::ChebTable::hess_index(1, 2, 3));
}

// hess_index: boundary cases dim=1 and dim=2 (easiest to mis-implement).
TEST_F(ChebNdIndexTest, HessIndexSmallDims) {
    // dim=1: only (0,0) -> 0
    EXPECT_EQ(oc::ChebTable::hess_index(0, 0, 1), 0);
    // dim=2 upper-triangle packed order: (0,0)=0, (0,1)=1, (1,1)=2
    EXPECT_EQ(oc::ChebTable::hess_index(0, 0, 2), 0);
    EXPECT_EQ(oc::ChebTable::hess_index(0, 1, 2), 1);
    EXPECT_EQ(oc::ChebTable::hess_index(1, 1, 2), 2);
    // Symmetry: (1,0) == (0,1)
    EXPECT_EQ(oc::ChebTable::hess_index(1, 0, 2), oc::ChebTable::hess_index(0, 1, 2));
}

// unflatten: multi-index coords are in [0, shape[d]) and are correct.
TEST_F(ChebNdIndexTest, UnflattenCoords) {
    std::vector<int> shape{2, 3, 4};
    auto strides = oc::ChebTable::row_major_strides(shape);
    ASSERT_EQ(strides[0], 12);
    ASSERT_EQ(strides[1], 4);
    ASSERT_EQ(strides[2], 1);
    // flat=13 -> i=1,j=0,k=1  (13 = 1*12 + 0*4 + 1)
    std::vector<int> mi;
    oc::ChebTable::unflatten(13, shape, strides, mi);
    ASSERT_EQ(int(mi.size()), 3);
    EXPECT_EQ(mi[0], 1);
    EXPECT_EQ(mi[1], 0);
    EXPECT_EQ(mi[2], 1);
    EXPECT_EQ(oc::ChebTable::flat_index(mi, strides), 13);
}
