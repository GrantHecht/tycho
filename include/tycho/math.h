#pragma once

// Tycho — opt-in math wrappers for VFs differentiated by Enzyme.
//
// The wrappers in `tycho::math` (currently `cos`/`sin`) provide ADL-friendly
// overloads for `double` and `Eigen::Array<double, W, 1>` (W in {2, 4, 8}).
// They sidestep Eigen's vectorised packet trig, which Enzyme cannot
// differentiate, while preserving Phase 5b SIMD primal evaluation and Phase 3
// batched tangents. Use these in `compute_impl` bodies that pair a
// trig-bearing VF with `EnzymeAD`; see CLAUDE.md "Trig-bearing bodies under
// Phase 5b" for the rule of thumb.

#include "tycho/detail/utils/simd_math.h"
