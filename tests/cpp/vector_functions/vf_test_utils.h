///////////////////////////////////////////////////////////////////////////////
// Shared VectorFunction test fixtures
//
// Empty fixture classes that inherit VectorFunctionFixture and are used
// across multiple VF test files.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "test_utils.h"

namespace TychoTest {

class CommonFunctionsTest : public VectorFunctionFixture {};
class GenericFunctionTest : public VectorFunctionFixture {};
class VFCompositionTest : public VectorFunctionFixture {};

} // namespace TychoTest
