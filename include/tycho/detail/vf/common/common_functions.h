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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/VectorFunction.h"
#include "tycho/detail/ArcTan2.h"
#include "tycho/detail/AutoDiffFunction.h"
#include "tycho/detail/CallAndAppend.h"
#include "tycho/detail/Comparative.h"
#include "tycho/detail/Conditional.h"
#include "tycho/detail/Constant.h"

#include "tycho/detail/CwiseOperators.h"
#include "tycho/detail/CwiseProduct.h"
#include "tycho/detail/CwiseSum.h"
#include "tycho/detail/DotProduct.h"
#include "tycho/detail/Elements.h"
#include "tycho/detail/FunctionHolder.h"
#include "tycho/detail/FunctionVectorSums.h"
#include "tycho/detail/LambdaFunction.h"
#include "tycho/detail/MatrixFunction.h"
#include "tycho/detail/MatrixInverse.h"
#include "tycho/detail/MatrixProduct.h"
#include "tycho/detail/NestedFunction.h"
#include "tycho/detail/Normalized.h"
#include "tycho/detail/Norms.h"
#include "tycho/detail/Padded.h"
#include "tycho/detail/ParsedInput.h"
#include "tycho/detail/RootFinder.h"
#include "tycho/detail/Scaled.h"
#include "tycho/detail/Segment.h"
#include "tycho/detail/SignFunction.h"
#include "tycho/detail/Stacked.h"
#include "tycho/detail/Summation.h"
#include "tycho/detail/Value.h"
#include "tycho/detail/VectorProducts.h"
#include "tycho/detail/VectorScalarFunctionDivision.h"
#include "tycho/detail/VectorScalarFunctionProduct.h"
