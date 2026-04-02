#pragma once
#include "tycho/vector_functions.h"

namespace tycho::vf {

GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>> &elems);
GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>> &elems);
GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>> &elems);
GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>> &elems);

} // namespace tycho::vf
