#pragma once

// Extern template declarations for frequently-used VF types.
//
// These suppress implicit instantiation in consuming TUs. The matching
// explicit instantiation definitions live in
// src/vf/extern_template_instantiations.cpp (the vf_instantiations library).
//
// Only the classes with significant method bodies are listed here.
// Empty pass-through classes in the CRTP chain (VectorFunction,
// DenseDerivatives, DenseFirstDerivatives, DenseSecondDerivatives,
// DenseFunction, Computable) are omitted — they contain only type aliases
// for Analytic derivative mode and have negligible instantiation cost.

#include "tycho/detail/vf/type_erasure/generic_function.h"

namespace tycho::vf {

// ---------------------------------------------------------------------------
// GenericFunction<-1, -1>  (dynamic-size vector function)
// ---------------------------------------------------------------------------
extern template struct ComputableBase<GenericFunction<-1, -1>, -1, -1>;
extern template struct DenseFunctionBase<GenericFunction<-1, -1>, -1, -1>;
extern template struct GenericFunction<-1, -1>;

// ---------------------------------------------------------------------------
// GenericFunction<-1, 1>  (dynamic-size scalar function)
// ---------------------------------------------------------------------------
extern template struct ComputableBase<GenericFunction<-1, 1>, -1, 1>;
extern template struct DenseFunctionBase<GenericFunction<-1, 1>, -1, 1>;
extern template struct GenericFunction<-1, 1>;

// ---------------------------------------------------------------------------
// Common leaf types — CRTP chain (DenseFunctionBase + ComputableBase)
//
// These leaf types appear in many TUs as building blocks of larger
// expressions. Pre-instantiating their DenseFunctionBase methods avoids
// redundant instantiation across TUs.
// ---------------------------------------------------------------------------
extern template struct ComputableBase<Arguments<-1>, -1, -1>;
extern template struct DenseFunctionBase<Arguments<-1>, -1, -1>;

extern template struct ComputableBase<Segment<-1, -1, -1>, -1, -1>;
extern template struct DenseFunctionBase<Segment<-1, -1, -1>, -1, -1>;

extern template struct ComputableBase<Constant<-1, -1>, -1, -1>;
extern template struct DenseFunctionBase<Constant<-1, -1>, -1, -1>;

extern template struct ComputableBase<Constant<-1, 1>, -1, 1>;
extern template struct DenseFunctionBase<Constant<-1, 1>, -1, 1>;

// ---------------------------------------------------------------------------
// GFModelCommon / GFModel for common leaf types
//
// These are the most frequently duplicated type-erasure instantiations across
// binding and test TUs (30,568 duplicate weak symbols for GFModelCommon<-1,-1>
// alone). Each unique T stored in GenericFunction requires a full
// GFModelCommon + GFModel instantiation with ~25 virtual method overrides.
// ---------------------------------------------------------------------------

// Arguments<-1> — used in virtually every VF expression
extern template struct GFModelCommon<-1, -1, Arguments<-1>>;
extern template struct GFModel<-1, -1, Arguments<-1>>;

// Segment<-1, -1, -1> — the most common indexing operation
extern template struct GFModelCommon<-1, -1, Segment<-1, -1, -1>>;
extern template struct GFModel<-1, -1, Segment<-1, -1, -1>>;

// Constant<-1, -1> — constant vector function
extern template struct GFModelCommon<-1, -1, Constant<-1, -1>>;
extern template struct GFModel<-1, -1, Constant<-1, -1>>;

// Constant<-1, 1> — constant scalar function (stored in GenericFunction<-1,1>)
extern template struct GFModelCommon<-1, 1, Constant<-1, 1>>;
extern template struct GFModel<-1, 1, Constant<-1, 1>>;

} // namespace tycho::vf
