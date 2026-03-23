# =============================================================================
# Originally from ASSET (AlabamaASRL/asset_asrl)
# Copyright 2020-present The University of Alabama-Astrodynamics and Space
#   Research Lab. Licensed under the Apache License, Version 2.0
# License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# Original Developer: James B. Pezent
#
# Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
#   Apache 2.0 — see LICENSE.txt):
#   - Package renamed: asset_asrl -> tycho
#   - Module renamed: asset_asrl (pybind11) -> _tychopy (nanobind)
#   - Imports updated accordingly
# =============================================================================

import inspect

import _tychopy as _tychopy

import tychopy.Astro
import tychopy.OptimalControl
import tychopy.Solvers
import tychopy.Utils
import tychopy.VectorFunctions

SoftwareInfo = _tychopy.SoftwareInfo

if __name__ == "__main__":
    _tychopy.SoftwareInfo()
    mlist = inspect.getmembers(_asset)
    for m in mlist:
        print(m[0], "= _tychopy." + str(m[0]))
