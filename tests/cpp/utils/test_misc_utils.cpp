///////////////////////////////////////////////////////////////////////////////
// Miscellaneous Utils unit tests
//
// Tests TypeName, GetCoreCount, STDExtensions, and FunctionReturnType
// utilities.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/FunctionReturnType.h"
#include "Utils/GetCoreCount.h"
#include "Utils/STDExtensions.h"
#include "Utils/TypeName.h"
#include <gtest/gtest.h>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// TypeNameTest
///////////////////////////////////////////////////////////////////////////////

TEST(TypeNameTest, IntName) {
    auto name = type_name<int>();
    EXPECT_NE(name.find("int"), std::string::npos);
}

TEST(TypeNameTest, DoubleName) {
    auto name = type_name<double>();
    EXPECT_NE(name.find("double"), std::string::npos);
}

TEST(TypeNameTest, ConstRef) {
    auto name = type_name<const int &>();
    // Should contain "int" and either "const" or "&"
    EXPECT_NE(name.find("int"), std::string::npos);
}

///////////////////////////////////////////////////////////////////////////////
// GetCoreCountTest
///////////////////////////////////////////////////////////////////////////////

TEST(GetCoreCountTest, AtLeastOne) { EXPECT_GE(get_core_count(), 1); }

///////////////////////////////////////////////////////////////////////////////
// STDExtensionsTest
///////////////////////////////////////////////////////////////////////////////

TEST(STDExtensionsTest, RemoveConstRef) {
    static_assert(std::is_same_v<std::remove_const_reference<const int &>::type, int>);
    SUCCEED();
}

TEST(STDExtensionsTest, PassThroughDouble) {
    static_assert(std::is_same_v<std::remove_const_reference<double>::type, double>);
    SUCCEED();
}

///////////////////////////////////////////////////////////////////////////////
// FunctionReturnTypeTest
///////////////////////////////////////////////////////////////////////////////

static double free_func(int, float) { return 0.0; }

struct TestClass {
    int member_func(double) { return 0; }
    float const_member(int) const { return 0.0f; }
};

TEST(FunctionReturnTypeTest, FreeFunction) {
    static_assert(std::is_same_v<return_type_t<decltype(&free_func)>, double>);
    SUCCEED();
}

TEST(FunctionReturnTypeTest, MemberFunction) {
    static_assert(std::is_same_v<return_type_t<decltype(&TestClass::member_func)>, int>);
    SUCCEED();
}

TEST(FunctionReturnTypeTest, ConstMemberFunction) {
    static_assert(std::is_same_v<return_type_t<decltype(&TestClass::const_member)>, float>);
    SUCCEED();
}
