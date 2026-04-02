// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/vf/common/constant.h"
#include "tycho/detail/vf/core/vector_function.h"
#include "tycho/detail/vf/expressions/call_and_append.h"
#include "tycho/detail/vf/operators/arc_tan2.h"
#include "tycho/detail/vf/type_erasure/autodiff_function.h"
#include "tycho/detail/vf/type_erasure/comparative.h"
#include "tycho/detail/vf/type_erasure/conditional.h"

#include "tycho/detail/vf/common/elements.h"
#include "tycho/detail/vf/common/value.h"
#include "tycho/detail/vf/expressions/function_holder.h"
#include "tycho/detail/vf/expressions/lambda_function.h"
#include "tycho/detail/vf/expressions/nested_function.h"
#include "tycho/detail/vf/expressions/parsed_input.h"
#include "tycho/detail/vf/expressions/segment.h"
#include "tycho/detail/vf/expressions/stacked.h"
#include "tycho/detail/vf/expressions/summation.h"
#include "tycho/detail/vf/operators/cwise_operators.h"
#include "tycho/detail/vf/operators/cwise_product.h"
#include "tycho/detail/vf/operators/cwise_sum.h"
#include "tycho/detail/vf/operators/dot_product.h"
#include "tycho/detail/vf/operators/function_vector_sums.h"
#include "tycho/detail/vf/operators/matrix_function.h"
#include "tycho/detail/vf/operators/matrix_inverse.h"
#include "tycho/detail/vf/operators/matrix_product.h"
#include "tycho/detail/vf/operators/normalized.h"
#include "tycho/detail/vf/operators/norms.h"
#include "tycho/detail/vf/operators/root_finder.h"
#include "tycho/detail/vf/operators/sign_function.h"
#include "tycho/detail/vf/operators/vector_products.h"
#include "tycho/detail/vf/operators/vector_scalar_function_division.h"
#include "tycho/detail/vf/operators/vector_scalar_function_product.h"
#include "tycho/detail/vf/scaling/padded.h"
#include "tycho/detail/vf/scaling/scaled.h"
