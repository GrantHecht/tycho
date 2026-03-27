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
#include <Eigen/Core>
#include <vector>

namespace tycho::utils {

template <class Scalar, int Size>
std::vector<Scalar> eigenvector_to_stdvector(const Eigen::Matrix<Scalar, Size, 1> &eigvec) {
    int size = eigvec.size();
    std::vector<double> stdvec(size);
    for (int i = 0; i < size; i++) {
        stdvec[i] = eigvec[i];
    }
    return stdvec;
}

template <class Scalar>
Eigen::Matrix<Scalar, -1, 1> stdvector_to_eigenvector(const std::vector<Scalar> &stdvec) {
    int size = stdvec.size();
    Eigen::Matrix<Scalar, -1, 1> eigvec(size);
    for (int i = 0; i < size; i++) {
        eigvec[i] = stdvec[i];
    }
    return eigvec;
}

} // namespace tycho::utils