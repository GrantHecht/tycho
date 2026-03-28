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

namespace tycho::solvers {

struct IterateInfo {

    int iter = 0;

    double Mu = 0;
    double PrimObj = 0;
    double BarrObj = 0;
    double KKTInf = 0;
    double BarrInf = 0;
    double EConInf = 0;
    double IConInf = 0;

    double PenPar1 = 0.0;
    double PenPar2 = 0.0;

    int LSiters = 0;
    double alphaP = 1.0;
    double alphaD = 1.0;
    double alphaT = 1.0;

    double Hpert = 0;
    int Hfacs = 0;

    double KKTNormErr = 0;
    double BarrNormErr = 0;
    double EConNormErr = 0;
    double IConNormErr = 0;
    double AllConNormErr = 0;

    int PPivots = 0;
    double MaxEMult = 0;
    double MaxIMult = 0;
    double MeritVal = 0.0;
};

} // namespace tycho::solvers
