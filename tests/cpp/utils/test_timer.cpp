///////////////////////////////////////////////////////////////////////////////
// Timer unit tests
//
// Tests Utils::Timer from src/Utils/Timer.h — construction, start/stop
// accumulation, reset, and duration type queries.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/Timer.h"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using namespace Tycho;

TEST(TimerTest, DefaultNotStarted) {
    Utils::Timer t;
    EXPECT_EQ(t.count<std::chrono::microseconds>(), 0);
}

TEST(TimerTest, ConstructWithStart) {
    Utils::Timer t(true);
    // Timer is running — count should be non-negative (may be 0 on fast machines)
    EXPECT_GE(t.count<std::chrono::microseconds>(), 0);
}

TEST(TimerTest, StartStopAccumulates) {
    Utils::Timer t;
    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    t.stop();
    auto elapsed = t.count<std::chrono::milliseconds>();
    EXPECT_GT(elapsed, 0) << "Timer should have accumulated > 0 ms";
}

TEST(TimerTest, ResetClearsAccumulated) {
    Utils::Timer t(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    t.stop();
    EXPECT_GT(t.count<std::chrono::milliseconds>(), 0);
    t.reset();
    EXPECT_EQ(t.count<std::chrono::milliseconds>(), 0);
}

TEST(TimerTest, StopWithoutStartIsNoOp) {
    Utils::Timer t;
    t.stop(); // Should not crash
    EXPECT_EQ(t.count<std::chrono::microseconds>(), 0);
}

TEST(TimerTest, CountWithDurationTypes) {
    Utils::Timer t(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    t.stop();
    // Microseconds should be larger than milliseconds value
    auto us = t.count<std::chrono::microseconds>();
    auto ms = t.count<std::chrono::milliseconds>();
    EXPECT_GT(us, 0);
    EXPECT_GE(us, ms * 1000);
}
