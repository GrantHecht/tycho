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
    {
        tycho::solvers::detail::MklLocalPinGuard guard;
        EXPECT_EQ(mkl_get_max_threads(), 1);
    }
    EXPECT_EQ(mkl_get_max_threads(), before);
}

#endif // USE_ACCELERATE_SPARSE
