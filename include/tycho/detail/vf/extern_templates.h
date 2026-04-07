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
#include "tycho/detail/vf/core/dense_scalar_function_base.h"

namespace tycho::vf {

// ---------------------------------------------------------------------------
// GenericFunction<-1, -1>  (dynamic-size vector function)
// ---------------------------------------------------------------------------
extern template struct ComputableBase<GenericFunction<-1, -1>, -1, -1>;
extern template struct DenseFunctionBase<GenericFunction<-1, -1>, -1, -1>;
extern template struct GenericFunction<-1, -1>;

// ---------------------------------------------------------------------------
// GenericFunction<-1, 1>  (dynamic-size scalar function)
// Uses DenseScalarFunctionBase<D, IR> instead of DenseFunctionBase<D, IR, OR>
// ---------------------------------------------------------------------------
extern template struct ComputableBase<GenericFunction<-1, 1>, -1, 1>;
extern template struct DenseFunctionBase<GenericFunction<-1, 1>, -1, 1>;
extern template struct DenseScalarFunctionBase<GenericFunction<-1, 1>, -1>;
extern template struct GenericFunction<-1, 1>;

} // namespace tycho::vf
