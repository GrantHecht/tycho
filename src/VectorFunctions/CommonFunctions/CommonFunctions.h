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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include "../VectorFunction.h"
#include "ArcTan2.h"
#include "AutoDiffFunction.h"
#include "CallAndAppend.h"
#include "Comparative.h"
#include "Conditional.h"
#include "Constant.h"

// #include "CrossProduct.h"
#include "CwiseOperators.h"
#include "CwiseProduct.h"
#include "CwiseSum.h"
#include "DotProduct.h"
#include "Elements.h"
#include "FunctionHolder.h"
#include "FunctionVectorSums.h"
#include "LambdaFunction.h"
#include "MatrixFunction.h"
#include "MatrixInverse.h"
#include "MatrixProduct.h"
#include "NestedFunction.h"
#include "Normalized.h"
#include "Norms.h"
#include "Padded.h"
#include "ParsedInput.h"
#include "RootFinder.h"
#include "Scaled.h"
#include "Segment.h"
#include "SignFunction.h"
#include "Stacked.h"
#include "Summation.h"
#include "Value.h"
#include "VectorProducts.h"
#include "VectorScalarFunctionDivision.h"
#include "VectorScalarFunctionProduct.h"
