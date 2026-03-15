///////////////////////////////////////////////////////////////////////////////
// BumpAllocator unit tests
//
// Tests BumpAllocator from src/Utils/MemoryManagement.h — resize, size
// queries, allocate_run correctness, overflow, learning, and nesting.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include <gtest/gtest.h>

using namespace Tycho;

TEST(BumpAllocatorTest, ResizeAndQuery) {
    BumpAllocator::resize(128, 64);
    EXPECT_EQ(BumpAllocator::size_scalar(), 128);
    EXPECT_EQ(BumpAllocator::size_super_scalar(), 64);
}

TEST(BumpAllocatorTest, ResizeSameForBoth) {
    BumpAllocator::resize(100);
    EXPECT_EQ(BumpAllocator::size_scalar(), 100);
    EXPECT_EQ(BumpAllocator::size_super_scalar(), 100);
}

TEST(BumpAllocatorTest, MultipleResizes) {
    BumpAllocator::resize(64, 64);
    EXPECT_EQ(BumpAllocator::size_scalar(), 64);

    BumpAllocator::resize(256, 128);
    EXPECT_EQ(BumpAllocator::size_scalar(), 256);
    EXPECT_EQ(BumpAllocator::size_super_scalar(), 128);

    BumpAllocator::resize(32, 32);
    EXPECT_EQ(BumpAllocator::size_scalar(), 32);
    EXPECT_EQ(BumpAllocator::size_super_scalar(), 32);
}

TEST(BumpAllocatorTest, AllocateRunZeroInit) {
    // Temporaries from allocate_run must be zero-initialized
    BumpAllocator::resize(256);
    using VType = Eigen::VectorXd;
    BumpAllocator::allocate_run(
        [](auto &v) {
            EXPECT_EQ(v.rows(), 10);
            EXPECT_EQ(v.cols(), 1);
            for (int i = 0; i < v.size(); ++i) {
                EXPECT_EQ(v[i], 0.0) << "Non-zero at index " << i;
            }
        },
        TempSpec<VType>(10, 1));
}

TEST(BumpAllocatorTest, AllocateRunMatrixZeroInit) {
    BumpAllocator::resize(256);
    using MType = Eigen::MatrixXd;
    BumpAllocator::allocate_run(
        [](auto &m) {
            EXPECT_EQ(m.rows(), 6);
            EXPECT_EQ(m.cols(), 7);
            for (int i = 0; i < m.size(); ++i) {
                EXPECT_EQ(m.data()[i], 0.0) << "Non-zero at flat index " << i;
            }
        },
        TempSpec<MType>(6, 7));
}

TEST(BumpAllocatorTest, AllocateRunMultipleTemps) {
    BumpAllocator::resize(256);
    using VType = Eigen::VectorXd;
    using MType = Eigen::MatrixXd;
    BumpAllocator::allocate_run(
        [](auto &v, auto &m) {
            EXPECT_EQ(v.rows(), 6);
            EXPECT_EQ(m.rows(), 6);
            EXPECT_EQ(m.cols(), 7);
            // They must not overlap
            bool overlap = (v.data() >= m.data() && v.data() < m.data() + m.size()) ||
                           (m.data() >= v.data() && m.data() < v.data() + v.size());
            EXPECT_FALSE(overlap) << "Temporaries overlap in memory";
        },
        TempSpec<VType>(6, 1), TempSpec<MType>(6, 7));
}

TEST(BumpAllocatorTest, OverflowProducesCorrectTemps) {
    // Use a tiny arena to force overflow
    BumpAllocator::resize(4);
    using VType = Eigen::VectorXd;
    BumpAllocator::allocate_run(
        [](auto &v) {
            EXPECT_EQ(v.rows(), 64);
            for (int i = 0; i < v.size(); ++i) {
                EXPECT_EQ(v[i], 0.0) << "Non-zero at index " << i;
            }
        },
        TempSpec<VType>(64, 1));
    // Restore arena to reasonable size
    BumpAllocator::resize(256);
}

TEST(BumpAllocatorTest, NestedAllocations) {
    BumpAllocator::resize(512);
    using VType = Eigen::VectorXd;
    double *outer_ptr = nullptr;
    BumpAllocator::allocate_run(
        [&](auto &outer) {
            EXPECT_EQ(outer.rows(), 10);
            outer_ptr = outer.data();
            outer.setOnes();

            BumpAllocator::allocate_run(
                [&](auto &inner) {
                    EXPECT_EQ(inner.rows(), 8);
                    // Inner must not overlap outer
                    bool overlap =
                        (inner.data() >= outer_ptr && inner.data() < outer_ptr + 10) ||
                        (outer_ptr >= inner.data() && outer_ptr < inner.data() + 8);
                    EXPECT_FALSE(overlap) << "Inner and outer temporaries overlap";
                    // Inner must be zero
                    for (int i = 0; i < inner.size(); ++i) {
                        EXPECT_EQ(inner[i], 0.0);
                    }
                },
                TempSpec<VType>(8, 1));

            // After inner restore, outer must still be intact
            for (int i = 0; i < 10; ++i) {
                EXPECT_EQ(outer[i], 1.0) << "Outer corrupted at index " << i;
            }
        },
        TempSpec<VType>(10, 1));
}

TEST(BumpAllocatorTest, ArrayOfTempSpecsCorrectness) {
    BumpAllocator::resize(512);
    using VType = Eigen::VectorXd;
    BumpAllocator::allocate_run(
        [](auto &arr) {
            EXPECT_EQ(arr.size(), 4u);
            for (int i = 0; i < 4; ++i) {
                EXPECT_EQ(arr[i].rows(), 6);
                EXPECT_EQ(arr[i].cols(), 1);
                for (int j = 0; j < arr[i].size(); ++j) {
                    EXPECT_EQ(arr[i].data()[j], 0.0);
                }
            }
            // Verify arrays don't overlap
            for (int i = 0; i < 3; ++i) {
                EXPECT_NE(arr[i].data(), arr[i + 1].data());
            }
        },
        ArrayOfTempSpecs<VType, 4>(6, 1));
}

TEST(BumpAllocatorTest, ConstantSizeExactPack) {
    // Constant-size TempSpecs should use stack allocation (ExactTempPack)
    using V3 = Eigen::Vector3d;
    BumpAllocator::allocate_run(
        [](auto &v) {
            EXPECT_EQ(v.rows(), 3);
            EXPECT_EQ(v.cols(), 1);
        },
        TempSpec<V3>(3, 1));
}

TEST(BumpAllocatorTest, SaveRestoreExactState) {
    // Verify arena returns to exact prior state after allocate_run
    BumpAllocator::resize(256);
    using VType = Eigen::VectorXd;

    // Run once to get baseline
    double *first_ptr = nullptr;
    BumpAllocator::allocate_run(
        [&](auto &v) { first_ptr = v.data(); }, TempSpec<VType>(10, 1));

    // Run again — should get same pointer (arena restored)
    double *second_ptr = nullptr;
    BumpAllocator::allocate_run(
        [&](auto &v) { second_ptr = v.data(); }, TempSpec<VType>(10, 1));

    EXPECT_EQ(first_ptr, second_ptr) << "Arena did not restore to prior state";
}

TEST(BumpAllocatorTest, VFComputeAfterResize) {
    // Verify that VF compute works after resize (integration check)
    BumpAllocator::resize(64, 64);
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    n.compute(x, fx);
    EXPECT_NEAR(fx[0], 5.0, 1e-12);
}
