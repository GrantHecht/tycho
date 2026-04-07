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
template struct DenseScalarFunctionBase<GenericFunction<-1, 1>, -1>;
template struct GenericFunction<-1, 1>;

} // namespace tycho::vf
