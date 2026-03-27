#!/usr/bin/env python3
"""
Create detail/ copies for OC headers that still need them.
Reads originals from git HEAD since source files were already converted to forwarding stubs.
"""

import os
import re
import subprocess

REPO = "/home/ghecht/Projects/tycho"
DETAIL = os.path.join(REPO, "include", "tycho", "detail")

# Headers that need detail copies created (source already has forwarding stubs)
REMAINING_HEADERS = [
    "LGLControlSplines.h",
    "LGLDefects.h",
    "LGLIntegrals.h",
    "LGLInterpTable.h",
    "LGLInterpFunctions.h",
    "TrapezoidalDefects.h",
    "TrapezoidalIntegrals.h",
    "ShootingDefects.h",
    "PhaseIndexer.h",
    "AutoScalingUtils.h",
    "MeshSpacingConstraints.h",
    "Blocked_ODE_Wrapper.h",
    "ODE.h",
    "ODEPhaseBase.h",
    "ODEPhase.h",
    "OptimalControlProblem.h",
]

# All OC header names (for include matching)
ALL_OC_HEADERS = {
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
}

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


def get_original_from_git(header):
    """Read original file content from git HEAD."""
    git_path = f"src/OptimalControl/{header}"
    result = subprocess.run(
        ["git", "show", f"HEAD:{git_path}"], cwd=REPO, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: Could not read {git_path} from git: {result.stderr}")
        return None
    return result.stdout


def map_include(inc_path):
    """Map an include path to its new detail/ location."""
    basename = os.path.basename(inc_path)

    # Bare OC header
    if "/" not in inc_path and basename in ALL_OC_HEADERS:
        return f"tycho/detail/{basename}"

    # VectorFunctions/Tycho_VectorFunctions.h
    if inc_path == "VectorFunctions/Tycho_VectorFunctions.h":
        return "tycho/vector_functions.h"

    # VectorFunctions/Foo.h
    if inc_path.startswith("VectorFunctions/"):
        return f"tycho/detail/{inc_path.split('/', 1)[1]}"

    # VectorFunctionTypeErasure/Foo.h
    if inc_path.startswith("VectorFunctionTypeErasure/"):
        return f"tycho/detail/{inc_path.split('/', 1)[1]}"

    # CommonFunctions/Foo.h
    if inc_path.startswith("CommonFunctions/"):
        return f"tycho/detail/{inc_path.split('/', 1)[1]}"

    # Utils/Foo.h
    if inc_path.startswith("Utils/"):
        util_name = inc_path.split("/", 1)[1]
        if util_name in UTILS_MAP:
            return f"tycho/detail/{UTILS_MAP[util_name]}"

    # TypeDefs/EigenTypes.h
    if inc_path == "TypeDefs/EigenTypes.h":
        return "tycho/detail/eigen_types.h"

    return inc_path


def rewrite_includes(content):
    """Rewrite includes in an OC header for the detail/ location."""
    lines = content.split("\n")
    new_lines = []
    pch_replaced = False

    for line in lines:
        stripped = line.strip()

        if stripped == '#include "pch.h"':
            if not pch_replaced:
                new_lines.append(PCH_REPLACEMENT.rstrip())
                pch_replaced = True
            continue

        m = re.match(r'^(\s*)#include\s+"([^"]+)"(.*)$', line)
        if m:
            indent, inc_path, trailing = m.group(1), m.group(2), m.group(3)
            new_path = map_include(inc_path)
            if new_path != inc_path:
                new_lines.append(f'{indent}#include "{new_path}"{trailing}')
                continue

        new_lines.append(line)

    return "\n".join(new_lines)


def main():
    for header in REMAINING_HEADERS:
        dst_path = os.path.join(DETAIL, header)

        if os.path.exists(dst_path):
            print(f"  SKIP (already exists): {header}")
            continue

        content = get_original_from_git(header)
        if content is None:
            continue

        rewritten = rewrite_includes(content)

        with open(dst_path, "w") as f:
            f.write(rewritten)

        print(f"  Created detail: {header}")

    print("\nDone creating remaining detail copies!")


if __name__ == "__main__":
    main()
