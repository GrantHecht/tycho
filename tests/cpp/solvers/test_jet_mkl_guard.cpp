///////////////////////////////////////////////////////////////////////////////
// Jet MKL thread-local pin guard tests
//
// Regression: Jet::map's per-job MKL pin must not outlive the job
// (CODEBASE_REVIEW 1.3). The thread-local setting overrides the global
// mkl_set_num_threads() PSIOPT applies; an unreset pin on the calling
// thread (serial fallback) single-threads every later Pardiso solve.
///////////////////////////////////////////////////////////////////////////////

#ifndef USE_ACCELERATE_SPARSE

#include "tycho/detail/solvers/jet.h"
#include <gtest/gtest.h>
#include <mkl.h>

TEST(JetMklGuard, RestoresThreadDefaultOnExit) {
    int before = mkl_get_max_threads();
    if (before <= 1)
        GTEST_SKIP() << "host resolves to 1 MKL thread; pin and restore are indistinguishable";
    {
        tycho::solvers::detail::MklLocalPinGuard guard;
        EXPECT_EQ(mkl_get_max_threads(), 1);
    }
    EXPECT_EQ(mkl_get_max_threads(), before);
}

TEST(JetMklGuard, RestoresPriorThreadLocalPin) {
    // The guard restores the caller's own thread-local pin, not just the
    // global default (ctor captures mkl_set_num_threads_local's return).
    mkl_set_num_threads_local(2);
    if (mkl_get_max_threads() != 2) {
        mkl_set_num_threads_local(0);
        GTEST_SKIP() << "MKL ignores thread-local settings on this host";
    }
    {
        tycho::solvers::detail::MklLocalPinGuard guard;
        EXPECT_EQ(mkl_get_max_threads(), 1);
    }
    EXPECT_EQ(mkl_get_max_threads(), 2);
    mkl_set_num_threads_local(0); // revert to global for later tests
}

#endif // USE_ACCELERATE_SPARSE
