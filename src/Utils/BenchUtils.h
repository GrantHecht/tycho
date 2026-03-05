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

#include "bench/BenchTimer.h"
#include "pch.h"

namespace Tycho {

template <class Functor, class... Args>
double BenchLoopFunctor(std::string name, int n, const Functor &f, Args... args) {
    Eigen::BenchTimer t;

    std::cout << "Bench Start: " << name << std::endl;
    t.start();
    for (int i = 0; i < n; i++) {
        f(n, args...);
    }
    t.stop();
    std::cout << "Bench Stop : " << t.total() * 1000.0 << " ms" << std::endl << std::endl;
    return t.total();
}

template <class Functor, class... Args>
double BenchFunctor(std::string name, const Functor &f, Args... args) {
    Eigen::BenchTimer t;
    std::cout << "Bench Start: " << name << std::endl;
    t.start();
    f(args...);
    t.stop();
    std::cout << "Bench Stop : " << t.total() * 1000.0 << " ms" << std::endl << std::endl;
    return t.total();
}

} // namespace Tycho
