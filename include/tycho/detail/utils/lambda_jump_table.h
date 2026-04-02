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

#include <type_traits>

namespace tycho::utils {

template <int J1, int J2, int J3> struct LambdaJumpTable {
    template <class Ftype> static void run(Ftype &f, int crit_size) {
        if (crit_size <= J1) {
            f(std::integral_constant<int, J1>());
        } else if (crit_size <= J2) {
            f(std::integral_constant<int, J2>());
        } else if (crit_size <= J3) {
            f(std::integral_constant<int, J3>());
        } else {
            f(std::integral_constant<int, -1>());
        }
    }
};

} // namespace tycho::utils
