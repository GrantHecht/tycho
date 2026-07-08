///////////////////////////////////////////////////////////////////////////////
// LGLInterpTable robustness tests (OC review §3.2)
//
// Pre-fix, LGLInterpTable's loaders printed a diagnostic to stdout and called
// exit(1) on malformed input instead of throwing (T6 rule violation);
// load_even_data() never validated monotonic/duplicate times at all;
// num_states_-1 not evenly divisible by block_size_-1 silently truncated
// num_blocks_, dropping trailing nodes; several loaders/ctors indexed
// xtudat[0] without checking for an empty vector; error_integral() read
// ts[1] and called ode_.compute() without guarding num_samps < 2 or
// has_ode_ == false; and the periodic time-wrap used int(frac) truncation.
// last_block_accessed_ (a search-hint cache written by the const find_block())
// was a plain `mutable int`, racy under concurrent const interpolation calls.
//
// This file exercises each rejection/guard path plus copy semantics for the
// new atomic search-hint member and a concurrent-interpolation sanity check.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <atomic>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <thread>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Step 1 (brief-mandated): load_even_data() rejection paths.
///////////////////////////////////////////////////////////////////////////////

TEST(LGLTableRobustness, DimMismatchThrowsNotExits) {
    EXPECT_THROW(load_even_wrong_dim(), std::invalid_argument);
}

TEST(LGLTableRobustness, DuplicateTimesRejected) {
    EXPECT_THROW(load_even_duplicate_times(), std::invalid_argument);
}

TEST(LGLTableRobustness, MisSizedBlocksRejected) {
    EXPECT_THROW(load_even_missized_blocks(), std::invalid_argument);
}

TEST(LGLTableRobustness, EmptyInputRejected) {
    EXPECT_THROW(load_even_empty(), std::invalid_argument);
}

/// Sanity: well-formed input must NOT throw (guards above aren't over-eager).
TEST(LGLTableRobustness, ValidInputNoThrow) {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL5);
    std::vector<Eigen::VectorXd> xtudat;
    constexpr int num_nodes = 7; // num_states_-1 == 6, divisible by block_size_-1 == 2
    for (int i = 0; i < num_nodes; ++i) {
        double t = static_cast<double>(i) / (num_nodes - 1);
        Eigen::VectorXd node(2);
        node << std::sin(t), t;
        xtudat.push_back(node);
    }
    EXPECT_NO_THROW(tab.load_even_data(xtudat));
    EXPECT_EQ(tab.num_states_, num_nodes);
    EXPECT_EQ(tab.num_blocks_, 3);
}

///////////////////////////////////////////////////////////////////////////////
// Coverage for the other two exit(1) -> throw sites (load_even_data2,
// load_uneven_data) and the empty-input guards added alongside them.
///////////////////////////////////////////////////////////////////////////////

TEST(LGLTableRobustness, LoadEvenData2WrongDimThrows) {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL3);
    std::vector<Eigen::VectorXd> xtudat, xdotdat;
    for (int i = 0; i < 3; ++i) {
        Eigen::VectorXd node(1); // wrong: expects xtu_vars_ == 2
        node << double(i);
        xtudat.push_back(node);
        Eigen::VectorXd d(1);
        d << 0.0;
        xdotdat.push_back(d);
    }
    EXPECT_THROW(tab.load_even_data2(xtudat, xdotdat), std::invalid_argument);
}

TEST(LGLTableRobustness, LoadEvenData2EmptyThrows) {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL3);
    std::vector<Eigen::VectorXd> xtudat, xdotdat;
    EXPECT_THROW(tab.load_even_data2(xtudat, xdotdat), std::invalid_argument);
}

TEST(LGLTableRobustness, LoadUnevenDataWrongDimThrows) {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL5);
    std::vector<Eigen::VectorXd> xtudat;
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXd node(1); // wrong: expects xtu_vars_ == 2
        node << double(i);
        xtudat.push_back(node);
    }
    EXPECT_THROW(tab.load_uneven_data(4, xtudat), std::invalid_argument);
}

TEST(LGLTableRobustness, LoadUnevenDataEmptyThrows) {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL3);
    std::vector<Eigen::VectorXd> xtudat;
    EXPECT_THROW(tab.load_uneven_data(4, xtudat), std::invalid_argument);
}

/// The dimension-inferring constructor (LGLInterpTable(xtudat)) indexes
/// xtudat[0] directly, before any loader runs -- must guard separately.
TEST(LGLTableRobustness, InferringCtorEmptyThrows) {
    std::vector<Eigen::VectorXd> xtudat;
    EXPECT_THROW(LGLInterpTable{xtudat}, std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// error_integral() guards: has_ode_ and num_samps < 2 (reads ts[1] unguarded
// pre-fix).
///////////////////////////////////////////////////////////////////////////////

TEST(LGLTableRobustness, ErrorIntegralNoOdeThrows) {
    constexpr int num_nodes = 7;
    std::vector<Eigen::VectorXd> xtudat;
    xtudat.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        double t = static_cast<double>(i) / (num_nodes - 1);
        Eigen::VectorXd node(2);
        node << std::sin(t), t;
        xtudat.push_back(node);
    }
    // This ctor never sets has_ode_ -- derivatives come from finite differences.
    LGLInterpTable tab(/*xv=*/1, xtudat, /*dnum=*/num_nodes - 1);
    ASSERT_FALSE(tab.has_ode_);
    EXPECT_THROW(tab.error_integral(5), std::invalid_argument);
}

TEST(LGLTableRobustness, ErrorIntegralTooFewSamplesThrows) {
    auto phase = make_linear_phase();
    LGLInterpTable tab = phase->return_traj_table();
    ASSERT_TRUE(tab.has_ode_);
    EXPECT_THROW(tab.error_integral(0), std::invalid_argument);
    EXPECT_THROW(tab.error_integral(1), std::invalid_argument);
}

TEST(LGLTableRobustness, ErrorIntegralValidInputNoThrow) {
    auto phase = make_linear_phase();
    LGLInterpTable tab = phase->return_traj_table();
    EXPECT_NO_THROW(tab.error_integral(5));
}

///////////////////////////////////////////////////////////////////////////////
// num_states_-1 divisibility guard on the exact-data loaders (brought up to
// the same standard as load_even_data).
///////////////////////////////////////////////////////////////////////////////

TEST(LGLTableRobustness, LoadExactDataMisSizedBlocksRejected) {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL5);
    std::vector<Eigen::VectorXd> xtudat, xdotdat;
    constexpr int num_nodes = 4; // num_states_-1 == 3, not divisible by block_size_-1 == 2
    for (int i = 0; i < num_nodes; ++i) {
        Eigen::VectorXd node(2);
        node << std::sin(double(i)), double(i);
        xtudat.push_back(node);
        Eigen::VectorXd d(1);
        d << std::cos(double(i));
        xdotdat.push_back(d);
    }
    EXPECT_THROW(tab.load_exact_data(xtudat, xdotdat), std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// last_block_accessed_ atomic search hint: copy semantics + concurrent use.
///////////////////////////////////////////////////////////////////////////////

/// Copy ctor/assignment must compile despite the atomic member and must reset
/// the hint to 0 in the copy while preserving the table's logical data (and
/// therefore identical interpolated values).
TEST(LGLTableRobustness, CopyPreservesDataAndResetsHint) {
    auto tab = make_exact_lgl_table(TranscriptionModes::LGL3);
    // Prime the hint away from 0 via a normal const interpolation call: with 7
    // uniformly-spaced nodes (block_size_ == 2, num_blocks_ == 6) over t in
    // [0, 1], t == 0.9 falls in the last block (index 5), moving the hint off 0.
    tab->interpolate(0.9);
    EXPECT_GT(tab->last_block_accessed_.load(), 0);

    LGLInterpTable copy_ctor_tab(*tab);
    EXPECT_EQ(copy_ctor_tab.last_block_accessed_.load(), 0);
    EXPECT_EQ(copy_ctor_tab.num_states_, tab->num_states_);
    EXPECT_EQ(copy_ctor_tab.x_vars_, tab->x_vars_);

    LGLInterpTable copy_assign_tab;
    copy_assign_tab = *tab;
    EXPECT_EQ(copy_assign_tab.last_block_accessed_.load(), 0);
    EXPECT_EQ(copy_assign_tab.num_states_, tab->num_states_);

    Eigen::VectorXd orig_val = tab->interpolate(0.5);
    Eigen::VectorXd ctor_val = copy_ctor_tab.interpolate(0.5);
    Eigen::VectorXd assign_val = copy_assign_tab.interpolate(0.5);
    EXPECT_NEAR((orig_val - ctor_val).norm(), 0.0, 1e-12);
    EXPECT_NEAR((orig_val - assign_val).norm(), 0.0, 1e-12);
}

/// Concurrent const interpolate() calls (each internally calling find_block(),
/// which reads/writes last_block_accessed_) must not crash or corrupt results
/// -- the search-hint is only ever a perf hint, so any interleaving of the
/// relaxed load/store must still resolve to the correct containing block.
TEST(LGLTableRobustness, ConcurrentInterpolateNoCrash) {
    // load_exact_data() leaves even_data_ == false, so find_block() takes the
    // last_block_accessed_ search-hint path (the arithmetic even_data_ path
    // never touches the hint).
    auto tab = make_exact_lgl_table(TranscriptionModes::LGL5);
    ASSERT_FALSE(tab->even_data_);

    constexpr int kThreads = 8;
    constexpr int kQueriesPerThread = 2000;
    std::vector<std::thread> threads;
    std::atomic<bool> all_ok{true};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(1000u + static_cast<unsigned>(t));
            std::uniform_real_distribution<double> dist(tab->t0_, tab->tf_);
            for (int i = 0; i < kQueriesPerThread; ++i) {
                double tq = dist(rng);
                Eigen::VectorXd fx = tab->interpolate(tq);
                if (fx.size() != tab->xtu_vars_ || !fx.allFinite()) {
                    all_ok.store(false, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto &th : threads) {
        th.join();
    }
    EXPECT_TRUE(all_ok.load());
}

///////////////////////////////////////////////////////////////////////////////
// Periodic time-wrap sanity (int(frac) -> std::floor(frac)).
///////////////////////////////////////////////////////////////////////////////

/// A periodic, evenly-spaced table must map query times several periods in
/// the future or past back onto the same in-range value.
TEST(LGLTableRobustness, PeriodicWrapMultiPeriodSanity) {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL3);
    constexpr int num_nodes = 9;
    std::vector<Eigen::VectorXd> xtudat;
    for (int i = 0; i < num_nodes; ++i) {
        double t = static_cast<double>(i) / (num_nodes - 1);
        Eigen::VectorXd node(2);
        node << std::sin(2.0 * std::numbers::pi * t), t;
        xtudat.push_back(node);
    }
    tab.load_even_data(xtudat);
    tab.make_periodic();
    ASSERT_TRUE(tab.periodic_);
    ASSERT_TRUE(tab.even_data_);

    double period = tab.total_t_;
    double query = tab.t0_ + 0.3 * period;
    Eigen::VectorXd baseline = tab.interpolate(query);
    Eigen::VectorXd future = tab.interpolate(query + 5.0 * period);
    Eigen::VectorXd past = tab.interpolate(query - 5.0 * period);

    EXPECT_NEAR((baseline.head(1) - future.head(1)).norm(), 0.0, 1e-8);
    EXPECT_NEAR((baseline.head(1) - past.head(1)).norm(), 0.0, 1e-8);
}
