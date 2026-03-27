#!/usr/bin/env python3
"""
Migrate OptimalControl headers from src/OptimalControl/ to include/tycho/detail/.

For each header (except Tycho_OptimalControl.h):
  1. Copy to include/tycho/detail/ with include path rewrites
  2. Convert original in src/ to a forwarding include

Then create include/tycho/optimal_control.h and update Tycho_OptimalControl.h.
"""

import os
import re

REPO = "/home/ghecht/Projects/tycho"
SRC_OC = os.path.join(REPO, "src", "OptimalControl")
DETAIL = os.path.join(REPO, "include", "tycho", "detail")
INCLUDE_TYCHO = os.path.join(REPO, "include", "tycho")

# All OC headers to migrate (everything except Tycho_OptimalControl.h)
OC_HEADERS = [
    "OptimalControlFlags.h",
    "ODESizes.h",
    "ODEArguments.h",
    "InterfaceTypes.h",
    "StateFunction.h",
    "LinkFunction.h",
    "ODE.h",
    "ODEPhaseBase.h",
    "ODEPhase.h",
    "OptimalControlProblem.h",
    "LGLInterpTable.h",
    "LGLInterpFunctions.h",
    "MeshIterateInfo.h",
    "OptimalControlFwdDecl.h",
    "LGLCoeffs.h",
    "LGLControlSplines.h",
    "LGLDefects.h",
    "LGLIntegrals.h",
    "TrapezoidalDefects.h",
    "TrapezoidalIntegrals.h",
    "ShootingDefects.h",
    "FDCoeffs.h",
    "FDDerivArbitrary.h",
    "FDDerivUniform.h",
    "PhaseIndexer.h",
    "TranscriptionSizing.h",
    "ValueLock.h",
    "AutoScalingUtils.h",
    "MeshSpacingConstraints.h",
    "Blocked_ODE_Wrapper.h",
]

# Set of all OC header basenames for matching
OC_HEADER_SET = set(OC_HEADERS)

# Set of all existing detail headers (for collision check)
existing_detail = set(os.listdir(DETAIL))

# Utils name mapping: src/Utils/Foo.h -> tycho/detail/bar.h
UTILS_MAP = {
    "SizingHelpers.h": "sizing_helpers.h",
    "FlatMap.h": "flat_map.h",
    "CRTPBase.h": "crtp_base.h",
    "ThreadPool.h": "thread_pool.h",
    "GetCoreCount.h": "get_core_count.h",
    "MathFunctions.h": "math_functions.h",
    "STDExtensions.h": "std_extensions.h",
    "TypeName.h": "type_name.h",
    "TypeStorage.h": "type_storage.h",
    "LambdaJumpTable.h": "lambda_jump_table.h",
    "Timer.h": "timer.h",
    "TupleIterator.h": "tuple_iterator.h",
    "EigenSTL.h": "eigen_stl.h",
    "fmtlib.h": "fmtlib.h",
}

# The pch.h replacement block: system includes + Eigen + utils
PCH_REPLACEMENT = """#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <fmt/format.h>

#include "tycho/detail/eigen_types.h"
#include "tycho/detail/std_extensions.h"
#include "tycho/detail/math_functions.h"
#include "tycho/detail/type_name.h"
"""


def rewrite_includes(content, filename):
    """Rewrite includes in an OC header for the detail/ location."""
    lines = content.split("\n")
    new_lines = []
    pch_replaced = False

    for line in lines:
        stripped = line.strip()

        # Replace #include "pch.h"
        if stripped == '#include "pch.h"':
            if not pch_replaced:
                new_lines.append(PCH_REPLACEMENT.rstrip())
                pch_replaced = True
            # Skip duplicate pch includes
            continue

        # Match #include "..." patterns
        m = re.match(r'^(\s*)#include\s+"([^"]+)"(.*)$', line)
        if m:
            indent = m.group(1)
            inc_path = m.group(2)
            trailing = m.group(3)

            new_path = map_include(inc_path, filename)
            if new_path != inc_path:
                new_lines.append(f'{indent}#include "{new_path}"{trailing}')
                continue

        new_lines.append(line)

    return "\n".join(new_lines)


def map_include(inc_path, source_filename):
    """Map an include path to its new location."""

    # Bare OC header: "Foo.h" where Foo.h is in our OC set
    basename = os.path.basename(inc_path)
    if "/" not in inc_path and basename in OC_HEADER_SET:
        return f"tycho/detail/{basename}"

    # VectorFunctions/Tycho_VectorFunctions.h -> tycho/vector_functions.h
    if inc_path == "VectorFunctions/Tycho_VectorFunctions.h":
        return "tycho/vector_functions.h"

    # VectorFunctions/VectorFunction.h -> tycho/detail/VectorFunction.h
    if inc_path.startswith("VectorFunctions/"):
        vf_name = inc_path.split("/", 1)[1]
        return f"tycho/detail/{vf_name}"

    # VectorFunctionTypeErasure/Foo.h -> tycho/detail/Foo.h
    if inc_path.startswith("VectorFunctionTypeErasure/"):
        te_name = inc_path.split("/", 1)[1]
        return f"tycho/detail/{te_name}"

    # CommonFunctions/Foo.h -> tycho/detail/Foo.h
    if inc_path.startswith("CommonFunctions/"):
        cf_name = inc_path.split("/", 1)[1]
        return f"tycho/detail/{cf_name}"

    # Utils/Foo.h -> tycho/detail/<mapped_name>
    if inc_path.startswith("Utils/"):
        util_name = inc_path.split("/", 1)[1]
        if util_name in UTILS_MAP:
            return f"tycho/detail/{UTILS_MAP[util_name]}"
        # Fallback: keep original for unmapped utils
        return inc_path

    # TypeDefs/EigenTypes.h -> tycho/detail/eigen_types.h
    if inc_path == "TypeDefs/EigenTypes.h":
        return "tycho/detail/eigen_types.h"

    # Solvers/*.h -> keep as-is (not yet migrated)
    if inc_path.startswith("Solvers/"):
        return inc_path

    # Integrators/*.h -> keep as-is (not yet migrated)
    if inc_path.startswith("Integrators/"):
        return inc_path

    # PyDocString/*.h -> keep as-is
    if inc_path.startswith("PyDocString/"):
        return inc_path

    # System includes (<...>) are not matched here
    # Anything else: keep as-is
    return inc_path


def make_forwarding_include(header_name):
    """Create forwarding include content."""
    return f'#pragma once\n#include "tycho/detail/{header_name}"\n'


def check_collisions():
    """Check for filename collisions with existing detail headers."""
    collisions = []
    for h in OC_HEADERS:
        if h in existing_detail:
            collisions.append(h)
    return collisions


def main():
    # Step 0: Check collisions
    collisions = check_collisions()
    if collisions:
        print(f"WARNING: Filename collisions detected: {collisions}")
        print("These files already exist in detail/. Aborting.")
        return

    print("No filename collisions detected.")

    # Step 0.5: Read Tycho_OptimalControl.h BEFORE modifying anything
    tycho_oc_path = os.path.join(SRC_OC, "Tycho_OptimalControl.h")
    with open(tycho_oc_path, "r") as f:
        original_tycho_oc = f.read()

    # Extract binding-related lines
    binding_lines = []
    for line in original_tycho_oc.split("\n"):
        if "Bindings/" in line and "#include" in line:
            binding_lines.append(line)

    # Step 1: Copy each OC header to detail/ with include rewrites
    for header in OC_HEADERS:
        src_path = os.path.join(SRC_OC, header)
        dst_path = os.path.join(DETAIL, header)

        if not os.path.exists(src_path):
            print(f"  SKIP (not found): {header}")
            continue

        with open(src_path, "r") as f:
            content = f.read()

        rewritten = rewrite_includes(content, header)

        with open(dst_path, "w") as f:
            f.write(rewritten)

        print(f"  Copied + rewritten: {header}")

    # Step 2: Convert originals to forwarding includes
    for header in OC_HEADERS:
        src_path = os.path.join(SRC_OC, header)
        if not os.path.exists(src_path):
            continue

        forwarding = make_forwarding_include(header)
        with open(src_path, "w") as f:
            f.write(forwarding)

        print(f"  Forwarding: src/OptimalControl/{header}")

    # Step 3: Create include/tycho/optimal_control.h
    optimal_control_h = os.path.join(INCLUDE_TYCHO, "optimal_control.h")
    oc_content = """#pragma once

#include "tycho/vector_functions.h"

#include "tycho/detail/OptimalControlFlags.h"
#include "tycho/detail/ODESizes.h"
#include "tycho/detail/ODEArguments.h"
#include "tycho/detail/InterfaceTypes.h"
#include "tycho/detail/StateFunction.h"
#include "tycho/detail/LinkFunction.h"
#include "tycho/detail/ODE.h"
#include "tycho/detail/ODEPhaseBase.h"
#include "tycho/detail/ODEPhase.h"
#include "tycho/detail/OptimalControlProblem.h"
#include "tycho/detail/LGLInterpTable.h"
#include "tycho/detail/LGLInterpFunctions.h"
#include "tycho/detail/MeshIterateInfo.h"
#include "tycho/detail/OptimalControlFwdDecl.h"
"""
    with open(optimal_control_h, "w") as f:
        f.write(oc_content)
    print(f"  Created: include/tycho/optimal_control.h")

    # Step 4: Update Tycho_OptimalControl.h
    new_tycho_oc = """// =============================================================================
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

#include "tycho/optimal_control.h"
#include "FDDerivArbitrary.h"
#include "FDDerivUniform.h"
#include "LGLInterpFunctions.h"
#include "ODE.h"
#include "ODEPhase.h"
#include "OptimalControlProblem.h"
"""

    if binding_lines:
        new_tycho_oc += "\n" + "\n".join(binding_lines) + "\n"

    new_tycho_oc += "\nnamespace Tycho {} // namespace Tycho\n"

    with open(tycho_oc_path, "w") as f:
        f.write(new_tycho_oc)
    print(f"  Updated: src/OptimalControl/Tycho_OptimalControl.h")

    print("\nMigration complete!")
    print(f"\nSummary:")
    print(f"  - {len(OC_HEADERS)} headers copied to include/tycho/detail/")
    print(f"  - {len(OC_HEADERS)} source headers converted to forwarding includes")
    print(f"  - Created include/tycho/optimal_control.h")
    print(f"  - Updated src/OptimalControl/Tycho_OptimalControl.h")
    if binding_lines:
        print(
            f"  - Preserved {len(binding_lines)} binding include(s) in Tycho_OptimalControl.h"
        )


if __name__ == "__main__":
    main()
