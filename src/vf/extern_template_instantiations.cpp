// Explicit instantiation definitions for common VF types.
// Matches the extern template declarations in
// include/tycho/detail/vf/extern_templates.h.
//
// This TU compiles all the template method bodies that are suppressed in
// consuming TUs by extern template. It will be heavy (~3-8 GB RSS).

#include "tycho/tycho.h"

namespace tycho::vf {

// ---------------------------------------------------------------------------
// GenericFunction<-1, -1>  (dynamic-size vector function)
// ---------------------------------------------------------------------------
template struct ComputableBase<GenericFunction<-1, -1>, -1, -1>;
template struct DenseFunctionBase<GenericFunction<-1, -1>, -1, -1>;
template struct GenericFunction<-1, -1>;

// ---------------------------------------------------------------------------
// GenericFunction<-1, 1>  (dynamic-size scalar function)
// ---------------------------------------------------------------------------
template struct ComputableBase<GenericFunction<-1, 1>, -1, 1>;
template struct DenseFunctionBase<GenericFunction<-1, 1>, -1, 1>;
template struct GenericFunction<-1, 1>;

// ---------------------------------------------------------------------------
// Common leaf types — CRTP chain
// ---------------------------------------------------------------------------
template struct ComputableBase<Arguments<-1>, -1, -1>;
template struct DenseFunctionBase<Arguments<-1>, -1, -1>;

template struct ComputableBase<Segment<-1, -1, -1>, -1, -1>;
template struct DenseFunctionBase<Segment<-1, -1, -1>, -1, -1>;

template struct ComputableBase<Constant<-1, -1>, -1, -1>;
template struct DenseFunctionBase<Constant<-1, -1>, -1, -1>;

template struct ComputableBase<Constant<-1, 1>, -1, 1>;
template struct DenseFunctionBase<Constant<-1, 1>, -1, 1>;

// ---------------------------------------------------------------------------
// GFModelCommon / GFModel for common leaf types
// ---------------------------------------------------------------------------
template struct GFModelCommon<-1, -1, Arguments<-1>>;
template struct GFModel<-1, -1, Arguments<-1>>;

template struct GFModelCommon<-1, -1, Segment<-1, -1, -1>>;
template struct GFModel<-1, -1, Segment<-1, -1, -1>>;

template struct GFModelCommon<-1, -1, Constant<-1, -1>>;
template struct GFModel<-1, -1, Constant<-1, -1>>;

template struct GFModelCommon<-1, 1, Constant<-1, 1>>;
template struct GFModel<-1, 1, Constant<-1, 1>>;

} // namespace tycho::vf
