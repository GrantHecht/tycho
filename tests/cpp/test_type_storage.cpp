///////////////////////////////////////////////////////////////////////////////
// TypeStorage SBO container tests
//
// Tests TypeStorage<C, SBO_CAP> from src/Utils/TypeStorage.h — value
// semantics, SBO inline/heap paths, copy/move correctness.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/TypeStorage.h"
#include <gtest/gtest.h>

using namespace Tycho;

// ---------------------------------------------------------------------------
// Minimal test base/model types
// ---------------------------------------------------------------------------

struct TestBase {
    virtual ~TestBase() = default;
    virtual int value() const = 0;
    virtual void clone_into(TypeStorage<TestBase> &s) const = 0;
};

struct TestModel : TestBase {
    int val_;
    explicit TestModel(int v) : val_(v) {}
    int value() const override { return val_; }
    void clone_into(TypeStorage<TestBase> &s) const override { s.emplace<TestModel>(val_); }
};

// A type deliberately larger than the default 128-byte SBO buffer
struct LargeModel : TestBase {
    std::array<char, 256> padding_{};
    int val_;
    explicit LargeModel(int v) : val_(v) {}
    int value() const override { return val_; }
    void clone_into(TypeStorage<TestBase> &s) const override { s.emplace<LargeModel>(val_); }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(TypeStorageTest, DefaultConstructEmpty) {
    TypeStorage<TestBase> s;
    EXPECT_TRUE(s.empty());
}

TEST(TypeStorageTest, EmplaceAndGet) {
    TypeStorage<TestBase> s;
    s.emplace<TestModel>(42);
    EXPECT_FALSE(s.empty());
    EXPECT_EQ(s.get().value(), 42);
}

TEST(TypeStorageTest, CopyConstruct) {
    TypeStorage<TestBase> a;
    a.emplace<TestModel>(7);

    TypeStorage<TestBase> b(a);
    EXPECT_FALSE(b.empty());
    EXPECT_EQ(b.get().value(), 7);

    // Verify independence: modifying the source via re-emplace shouldn't affect copy
    a.emplace<TestModel>(99);
    EXPECT_EQ(b.get().value(), 7);
    EXPECT_EQ(a.get().value(), 99);
}

TEST(TypeStorageTest, MoveConstruct) {
    TypeStorage<TestBase> a;
    a.emplace<TestModel>(13);

    TypeStorage<TestBase> b(std::move(a));
    EXPECT_FALSE(b.empty());
    EXPECT_EQ(b.get().value(), 13);
    EXPECT_TRUE(a.empty());
}

TEST(TypeStorageTest, CopyAssign) {
    TypeStorage<TestBase> a;
    a.emplace<TestModel>(5);

    TypeStorage<TestBase> b;
    b.emplace<TestModel>(10);

    b = a;
    EXPECT_EQ(b.get().value(), 5);

    // Verify independence
    a.emplace<TestModel>(99);
    EXPECT_EQ(b.get().value(), 5);
}

TEST(TypeStorageTest, MoveAssign) {
    TypeStorage<TestBase> a;
    a.emplace<TestModel>(20);

    TypeStorage<TestBase> b;
    b.emplace<TestModel>(30);

    b = std::move(a);
    EXPECT_EQ(b.get().value(), 20);
    EXPECT_TRUE(a.empty());
}

TEST(TypeStorageTest, SelfCopyAssign) {
    TypeStorage<TestBase> a;
    a.emplace<TestModel>(42);

    auto &ref = a;
    a = ref;
    EXPECT_FALSE(a.empty());
    EXPECT_EQ(a.get().value(), 42);
}

TEST(TypeStorageTest, SelfMoveAssign) {
    TypeStorage<TestBase> a;
    a.emplace<TestModel>(42);

    auto &ref = a;
    a = std::move(ref);
    // After self-move, the object should remain valid (self-move is guarded)
    EXPECT_FALSE(a.empty());
    EXPECT_EQ(a.get().value(), 42);
}

TEST(TypeStorageTest, HeapAllocation) {
    // LargeModel (>128 bytes) forces the heap path
    TypeStorage<TestBase> a;
    a.emplace<LargeModel>(77);
    EXPECT_FALSE(a.empty());
    EXPECT_EQ(a.get().value(), 77);

    // Copy via clone_into (heap path)
    TypeStorage<TestBase> b(a);
    EXPECT_EQ(b.get().value(), 77);

    // Move (pointer steal)
    TypeStorage<TestBase> c(std::move(a));
    EXPECT_EQ(c.get().value(), 77);
    EXPECT_TRUE(a.empty());
}
