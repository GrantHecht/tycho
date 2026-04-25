///////////////////////////////////////////////////////////////////////////////
// exception_what unit tests — pin all three branches: null exception_ptr,
// std::exception subclass (path used by aggregated dispatch error reports),
// and the non-std catch-all fallback.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/utils/exception_what.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

TEST(ExceptionWhat, NullExceptionPtrReturnsSentinel) {
    std::exception_ptr p;
    EXPECT_FALSE(static_cast<bool>(p));
    EXPECT_EQ(tycho::utils::exception_what(p), "<no exception>");
}

TEST(ExceptionWhat, StdExceptionReturnsWhat) {
    std::exception_ptr p;
    try {
        throw std::runtime_error("kepler diverged at t=1.5");
    } catch (...) {
        p = std::current_exception();
    }
    EXPECT_EQ(tycho::utils::exception_what(p), "kepler diverged at t=1.5");
}

TEST(ExceptionWhat, StdExceptionSubclassPreservesMessage) {
    // invalid_argument — a subclass we route through the typed-rethrow ladder
    // in decorate_trajectory; same path goes through exception_what for the
    // aggregated message.
    std::exception_ptr p;
    try {
        throw std::invalid_argument("bad direction value");
    } catch (...) {
        p = std::current_exception();
    }
    EXPECT_EQ(tycho::utils::exception_what(p), "bad direction value");
}

TEST(ExceptionWhat, NonStdExceptionFallsBackToSentinel) {
    // std::make_exception_ptr accepts non-std types; the helper must catch via
    // catch(...) and emit the documented sentinel without crashing.
    std::exception_ptr p = std::make_exception_ptr(42);
    EXPECT_EQ(tycho::utils::exception_what(p), "<non-std::exception>");
}
