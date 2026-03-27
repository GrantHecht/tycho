#pragma once

#include "tycho/typedefs.h"
#include "tycho/utils.h"

// Core VF types
#include "tycho/detail/vf/core/vector_function.h"
#include "tycho/detail/vf/core/dense_function_base.h"
#include "tycho/detail/vf/operators/operator_overloads.h"
#include "tycho/detail/vf/operators/math_overloads.h"

// Type erasure
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include "tycho/detail/vf/type_erasure/generic_conditional.h"
#include "tycho/detail/vf/type_erasure/generic_comparative.h"

// Common functions library
#include "tycho/detail/vf/common/common_functions.h"

// Concepts
#include "tycho/detail/vf/core/vector_function_concepts.h"
