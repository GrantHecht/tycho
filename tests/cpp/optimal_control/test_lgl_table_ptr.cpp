///////////////////////////////////////////////////////////////////////////////
// LGLInterpTable::get_table_ptr ownership tests
//
// Regression: get_table_ptr() must be non-owning (OPTIMAL_CONTROL_REVIEW 1.1).
// Pre-fix it returned std::shared_ptr(this) with the default deleter; dropping
// the last reference ran `delete this` on storage it never owned.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST(LGLInterpTablePtr, GetTablePtrIsNonOwning) {
    oc::LGLInterpTable tab(2, 0, TranscriptionModes::LGL3);
    auto p = tab.get_table_ptr();
    EXPECT_EQ(p.get(), &tab);
    EXPECT_EQ(p.use_count(), 0);  // aliasing ctor with empty owner
    p.reset();  // pre-fix: delete &tab on a stack object -> crash/UB
    SUCCEED();
}
