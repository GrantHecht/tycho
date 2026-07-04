///////////////////////////////////////////////////////////////////////////////
// Type-erasure guard tests (VF_REVIEW 1.14)
//
// Default-constructed (empty) erased objects previously null-dereferenced on
// first use — a segfault, not a clean failure. These tests confirm the guards
// added to GFStorage::get() (gf_type_erasure.h), GenericComparative's
// storage.get() call sites (generic_comparative.h), and GenericFunction's
// erasing constructor + cachedata() runtime size check (generic_function.h)
// now fail cleanly via C++ exceptions instead of crashing.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::vf;
using namespace TychoTest;

class VFTypeErasureGuards : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// GFStorage::get() guard (gf_type_erasure.h ~:620)
//
// Pre-fix: a default-constructed GenericFunction holds a null shared_ptr in
// its GFStorage; calling compute() reaches GFStorage::get(), which did
// `return *ptr_;` unconditionally — a null dereference (segfault, not a
// catchable failure). Post-fix, get() throws std::runtime_error when empty.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFTypeErasureGuards, DefaultConstructedGenericFunctionThrowsOnUse) {
    GenericFunction<-1, -1> f; // empty — never assigned a function
    Eigen::VectorXd x(2), fx(1);
    x.setZero();
    fx.setZero();
    EXPECT_THROW(f.compute(x, fx), std::runtime_error);
}

///////////////////////////////////////////////////////////////////////////////
// Erasing-ctor size-mismatch guard (generic_function.h ~:76-81, ~:117-121)
//
// The erasing constructor's compile-time `requires` constraint only rejects a
// *statically*-known size mismatch; a dynamically-sized source (IRC == -1)
// always satisfies it, since its true size isn't known until runtime. The
// runtime check added to cachedata() is what actually catches this case: it
// compares the stored function's runtime input_rows()/output_rows() against
// the target GenericFunction<IR,OR>'s fixed IR/OR and throws
// std::invalid_argument on mismatch (previously this silently truncated —
// e.g. accepted a 7-row source into a 6-row fixed-size GenericFunction and
// left the extra row unread).
//
// Path taken by this test: `seven` is GenericFunction<-1,-1> (IRC == -1), so
// the compile-time constraint on GenericFunction<6,3>'s erasing ctor is
// satisfied (T::IRC == -1 passes trivially) and the call compiles; the
// runtime check in cachedata() is what throws.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFTypeErasureGuards, ErasingCtorRejectsRuntimeSizeMismatch) {
    auto args = Arguments<7>();
    GenericFunction<-1, -1> seven(args * 2.0); // 7-input, 7-output, fully dynamic wrapper
    EXPECT_EQ(seven.input_rows(), 7);

    EXPECT_THROW((GenericFunction<6, 3>(seven)), std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// GenericComparative storage.get() guards (generic_comparative.h ~:91, ~:99->130)
//
// A default-constructed GenericComparative holds an empty TypeStorage; both
// input_rows() and compute() previously called storage.get() unconditionally,
// which is UB on empty TypeStorage (documented as such — TypeStorage::get()
// itself is intentionally left unguarded, since it's a general utility with an
// Empty-is-UB contract). The erased-wrapper boundary (GenericComparative) now
// guards each call site with storage.empty() and throws std::runtime_error.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFTypeErasureGuards, DefaultComparativeThrowsOnUse) {
    GenericComparative<-1> c; // empty — never assigned a predicate
    Eigen::VectorXd x(2);
    x.setZero();
    EXPECT_THROW(c.compute(x), std::runtime_error);
}

TEST_F(VFTypeErasureGuards, DefaultComparativeInputRowsThrows) {
    GenericComparative<-1> c;
    EXPECT_THROW(c.input_rows(), std::runtime_error);
}

///////////////////////////////////////////////////////////////////////////////
// GenericConditional storage.get() guards (generic_conditional.h input_rows() /
// compute()). Same defect class as GenericComparative above: a default-
// constructed GenericConditional holds an empty TypeStorage. GenericConditional
// is the TestFunc behind the Python ifelse() DSL, so it is user-reachable.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFTypeErasureGuards, DefaultConditionalThrowsOnUse) {
    GenericConditional<-1> c; // empty — never assigned a predicate
    Eigen::VectorXd x(2);
    x.setZero();
    EXPECT_THROW(c.compute(x), std::runtime_error);
}

TEST_F(VFTypeErasureGuards, DefaultConditionalInputRowsThrows) {
    GenericConditional<-1> c;
    EXPECT_THROW(c.input_rows(), std::runtime_error);
}
