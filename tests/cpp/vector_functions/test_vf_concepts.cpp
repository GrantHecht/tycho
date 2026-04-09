// Compile-time verification that VF concepts match the trait flags they replaced.
// If a concept evaluates differently from the underlying static constexpr bool,
// these static_asserts will catch it at compile time.

#include <gtest/gtest.h>

#include "tycho/vector_functions.h"

namespace tycho::vf {

// ---------------------------------------------------------------------------
// Vectorizable — true iff T::is_vectorizable == true
// ---------------------------------------------------------------------------
static_assert(Vectorizable<Arguments<-1>>);
static_assert(Vectorizable<Segment<-1, -1, -1>>);
static_assert(Vectorizable<Constant<-1, -1>>);
static_assert(Vectorizable<GenericFunction<-1, -1>>);
static_assert(Vectorizable<GenericFunction<-1, 1>>);

// ComputableBase defaults: is_vectorizable = false
// (No leaf type currently exercises the false case because all common leaves
//  override to true. The base default is tested indirectly via GenericFunction
//  which explicitly sets is_vectorizable = true.)

// ---------------------------------------------------------------------------
// LinearVF — true iff T::is_linear_function == true
// ---------------------------------------------------------------------------
static_assert(LinearVF<Arguments<-1>>);
static_assert(LinearVF<Arguments<7>>);
static_assert(LinearVF<Segment<-1, -1, -1>>);
static_assert(LinearVF<Segment<7, 3, 0>>);
static_assert(LinearVF<Constant<-1, -1>>);
static_assert(LinearVF<Constant<-1, 1>>);

// GenericFunction wraps arbitrary functions — not linear
static_assert(!LinearVF<GenericFunction<-1, -1>>);
static_assert(!LinearVF<GenericFunction<-1, 1>>);

// ---------------------------------------------------------------------------
// IsGenericVF — true iff T::is_generic_function == true
// ---------------------------------------------------------------------------
static_assert(IsGenericVF<GenericFunction<-1, -1>>);
static_assert(IsGenericVF<GenericFunction<-1, 1>>);
static_assert(!IsGenericVF<Arguments<-1>>);
static_assert(!IsGenericVF<Segment<-1, -1, -1>>);
static_assert(!IsGenericVF<Constant<-1, -1>>);

// ---------------------------------------------------------------------------
// ConditionalVF — true iff T::is_conditional == true
// ---------------------------------------------------------------------------
// ConditionalStatement sets is_conditional = true
using TestConditional = ConditionalStatement<GenericFunction<-1, 1>, GenericFunction<-1, 1>>;
static_assert(ConditionalVF<TestConditional>);

// Regular VF types are not conditional
static_assert(!ConditionalVF<Arguments<-1>>);
static_assert(!ConditionalVF<GenericFunction<-1, -1>>);

// ---------------------------------------------------------------------------
// HasDiagonalJac / HasDiagonalHess / CwiseVF
// No type currently sets these to true — verify the defaults hold.
// ---------------------------------------------------------------------------
static_assert(!HasDiagonalJac<Arguments<-1>>);
static_assert(!HasDiagonalJac<GenericFunction<-1, -1>>);
static_assert(!HasDiagonalHess<Arguments<-1>>);
static_assert(!HasDiagonalHess<GenericFunction<-1, -1>>);
static_assert(!CwiseVF<Arguments<-1>>);
static_assert(!CwiseVF<GenericFunction<-1, -1>>);

// ---------------------------------------------------------------------------
// StaticallySized / DynamicallySized
// ---------------------------------------------------------------------------
static_assert(StaticallySized<Arguments<7>>);
static_assert(StaticallySized<Segment<7, 3, 0>>);
static_assert(!StaticallySized<Arguments<-1>>);
static_assert(!StaticallySized<GenericFunction<-1, -1>>);

static_assert(DynamicallySized<Arguments<-1>>);
static_assert(DynamicallySized<GenericFunction<-1, -1>>);
static_assert(!DynamicallySized<Arguments<7>>);
static_assert(!DynamicallySized<Segment<7, 3, 0>>);

// ---------------------------------------------------------------------------
// Composable<Inner, Outer> — Inner::ORC matches Outer::IRC (or dynamic)
// ---------------------------------------------------------------------------
// Arguments<N>: IRC=ORC=N
static_assert(Composable<Arguments<3>, Arguments<3>>);
static_assert(!Composable<Arguments<5>, Arguments<3>>);
static_assert(Composable<Arguments<-1>, Arguments<3>>);
static_assert(Composable<Arguments<3>, Arguments<-1>>);

// Segment<7,3,0>: IRC=7, ORC=3
static_assert(Composable<Segment<7, 3, 0>, Arguments<3>>);
static_assert(!Composable<Segment<7, 3, 0>, Arguments<7>>);

// ---------------------------------------------------------------------------
// Stackable<F1, F2> — IRC must match (or dynamic)
// ---------------------------------------------------------------------------
static_assert(Stackable<Arguments<7>, Arguments<7>>);
static_assert(!Stackable<Arguments<7>, Arguments<5>>);
static_assert(Stackable<Arguments<-1>, Arguments<7>>);
static_assert(Stackable<Arguments<7>, Arguments<-1>>);
static_assert(Stackable<Arguments<-1>, Arguments<-1>>);

// MutuallyStackable — all pairs must satisfy Stackable
static_assert(MutuallyStackable<Arguments<7>, Arguments<7>, Arguments<7>>);
static_assert(!MutuallyStackable<Arguments<7>, Arguments<-1>, Arguments<5>>);
static_assert(MutuallyStackable<Arguments<-1>, Arguments<-1>, Arguments<-1>>);

// ---------------------------------------------------------------------------
// DenseVectorFunction — structural concept for the full VF interface
// ---------------------------------------------------------------------------
static_assert(DenseVectorFunction<Arguments<-1>>);
static_assert(DenseVectorFunction<Arguments<7>>);
static_assert(DenseVectorFunction<Segment<-1, -1, -1>>);
static_assert(DenseVectorFunction<Segment<7, 3, 0>>);
static_assert(DenseVectorFunction<Constant<-1, -1>>);
static_assert(DenseVectorFunction<Constant<-1, 1>>);
static_assert(DenseVectorFunction<GenericFunction<-1, -1>>);
static_assert(DenseVectorFunction<GenericFunction<-1, 1>>);

// ---------------------------------------------------------------------------
// Dummy test to satisfy Google Test (all checks are compile-time)
// ---------------------------------------------------------------------------
TEST(VFConcepts, StaticAssertsCompile) { SUCCEED(); }

} // namespace tycho::vf
