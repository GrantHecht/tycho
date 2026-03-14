///////////////////////////////////////////////////////////////////////////////
// MemoryManager unit tests
//
// Tests MemoryManager from src/Utils/MemoryManagement.h — resize, arena
// enable/disable, size queries. The allocate_run API is tightly coupled
// to VF expression evaluation and is exercised through the VF tests.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include <gtest/gtest.h>

using namespace Tycho;

TEST(MemoryManagerTest, ResizeAndQuery) {
    MemoryManager::resize(128, 64);
    EXPECT_EQ(MemoryManager::size_scalar(), 128);
    EXPECT_EQ(MemoryManager::size_super_scalar(), 64);
}

TEST(MemoryManagerTest, ResizeSameForBoth) {
    MemoryManager::resize(100);
    EXPECT_EQ(MemoryManager::size_scalar(), 100);
    EXPECT_EQ(MemoryManager::size_super_scalar(), 100);
}

TEST(MemoryManagerTest, EnableDisableArena) {
    MemoryManager::enable_arena_memory(32, 32);
    EXPECT_TRUE(MemoryManager::arena_memory_enabled());
    MemoryManager::disable_arena_memory();
    EXPECT_FALSE(MemoryManager::arena_memory_enabled());
}

TEST(MemoryManagerTest, EnableArenaDefaultSize) {
    MemoryManager::enable_arena_memory();
    EXPECT_TRUE(MemoryManager::arena_memory_enabled());
    MemoryManager::disable_arena_memory();
}

TEST(MemoryManagerTest, EnableArenaSingleSize) {
    MemoryManager::enable_arena_memory(48);
    EXPECT_TRUE(MemoryManager::arena_memory_enabled());
    MemoryManager::disable_arena_memory();
}

TEST(MemoryManagerTest, ResizeAfterArena) {
    MemoryManager::enable_arena_memory(32, 32);
    MemoryManager::resize(200, 150);
    EXPECT_EQ(MemoryManager::size_scalar(), 200);
    EXPECT_EQ(MemoryManager::size_super_scalar(), 150);
    MemoryManager::disable_arena_memory();
}

TEST(MemoryManagerTest, MultipleResizes) {
    MemoryManager::resize(64, 64);
    EXPECT_EQ(MemoryManager::size_scalar(), 64);

    MemoryManager::resize(256, 128);
    EXPECT_EQ(MemoryManager::size_scalar(), 256);
    EXPECT_EQ(MemoryManager::size_super_scalar(), 128);

    MemoryManager::resize(32, 32);
    EXPECT_EQ(MemoryManager::size_scalar(), 32);
    EXPECT_EQ(MemoryManager::size_super_scalar(), 32);
}

TEST(MemoryManagerTest, VFComputeAfterResize) {
    // Verify that VF compute works after resize (integration check)
    MemoryManager::resize(64, 64);
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    n.compute(x, fx);
    EXPECT_NEAR(fx[0], 5.0, 1e-12);
}
