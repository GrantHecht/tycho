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

#include <algorithm>
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

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"

namespace tycho {

enum class PhaseRegionFlags {
    NotSet,
    Front,
    Back,
    FrontandBack,
    BackandFront,
    Path,
    InnerPath,
    NodalPath,
    DefectPath,
    PairWisePath,
    DefectPairWisePath,
    FrontNodalBackPath,
    Params,
    ODEParams,
    StaticParams,
    Accumulate,
    BlockDefectPath,
};

enum class LinkFlags {
    BackToFront,
    FrontToBack,
    FrontToFront,
    BackToBack,
    ParamsToParams,
    LinkParams,
    BackTwoToTwoFront,
    FrontTwoToTwoBack,
    PathToPath,
    ReadRegions,
};

enum class ControlModes {
    HighestOrderSpline,
    FirstOrderSpline,
    NoSpline,
    BlockConstant,
};

enum class TranscriptionModes {
    LGL3,
    LGL5,
    LGL7,
    Trapezoidal,
    CentralShooting,
};

enum class IntegralModes {
    BaseIntegral,
    SimpsonIntegral,
    TrapIntegral,
};

enum class ScaleModes {
    AUTO,
    CUSTOM,
    NONE,
};

enum class MeshErrorEstimators {
    DEBOOR,
    INTEGRATOR,
};

enum class MeshErrorAggregation {
    MAX,
    AVG,
    GEOMETRIC,
    ENDTOEND,
};

inline ScaleModes strto_ScaleMode(const std::string &str) {
    if (str == "auto")
        return ScaleModes::AUTO;
    else if (str == "custom")
        return ScaleModes::CUSTOM;
    else if (str == "none")
        return ScaleModes::NONE;
    else {
        throw std::invalid_argument(fmt::format("Unrecognized ScaleMode: {0}", str));
    }
}

inline MeshErrorEstimators strto_MeshErrorEstimator(const std::string &str) {
    if (str == "deboor")
        return MeshErrorEstimators::DEBOOR;
    else if (str == "integrator")
        return MeshErrorEstimators::INTEGRATOR;
    else {
        throw std::invalid_argument(fmt::format("Unrecognized MeshErrorEstimator: {0}", str));
    }
}

inline MeshErrorAggregation strto_MeshErrorAggregation(const std::string &str) {
    if (str == "max")
        return MeshErrorAggregation::MAX;
    else if (str == "avg")
        return MeshErrorAggregation::AVG;
    else if (str == "geometric")
        return MeshErrorAggregation::GEOMETRIC;
    else if (str == "endtoend")
        return MeshErrorAggregation::ENDTOEND;
    else {
        throw std::invalid_argument(fmt::format("Unrecognized MeshErrorAggregation: {0}", str));
    }
}

inline PhaseRegionFlags strto_PhaseRegionFlag(const std::string &str) {

    if (str == "Front" || str == "First")
        return PhaseRegionFlags::Front;
    else if (str == "Back" || str == "Last")
        return PhaseRegionFlags::Back;
    else if (str == "Path")
        return PhaseRegionFlags::Path;
    else if (str == "ODEParams")
        return PhaseRegionFlags::ODEParams;
    else if (str == "StaticParams")
        return PhaseRegionFlags::StaticParams;
    else if (str == "FrontandBack" || str == "FirstandLast")
        return PhaseRegionFlags::FrontandBack;
    else if (str == "BackandFront" || str == "LastandFirst")
        return PhaseRegionFlags::BackandFront;
    else if (str == "InnerPath")
        return PhaseRegionFlags::InnerPath;
    else if (str == "PairWisePath")
        return PhaseRegionFlags::PairWisePath;
    else {
        auto msg = fmt::format("Unrecognized PhaseRegionFlag: {0}\n", str);
        throw std::invalid_argument(msg);
    }
}

inline Eigen::Matrix<PhaseRegionFlags, -1, 1>
strto_PhaseRegionFlag(const std::vector<std::string> &strs) {
    Eigen::Matrix<PhaseRegionFlags, -1, 1> regvec(strs.size());

    for (int i = 0; i < strs.size(); i++) {
        regvec[i] = strto_PhaseRegionFlag(strs[i]);
    }
    return regvec;
}

inline TranscriptionModes strto_TranscriptionMode(const std::string &str) {

    if (str == "LGL3")
        return TranscriptionModes::LGL3;
    else if (str == "LGL5")
        return TranscriptionModes::LGL5;
    else if (str == "LGL7")
        return TranscriptionModes::LGL7;
    else if (str == "CentralShooting")
        return TranscriptionModes::CentralShooting;
    else if (str == "Trapezoidal")
        return TranscriptionModes::Trapezoidal;
    else {
        auto msg = fmt::format("Unrecognized TranscriptionModes: {0}\n", str);
        throw std::invalid_argument(msg);
    }
}

inline LinkFlags strto_LinkFlag(const std::string &str) {

    if (str == "BackToBack" || str == "LastToLast")
        return LinkFlags::BackToBack;
    else if (str == "BackToFront" || str == "LastToFirst")
        return LinkFlags::BackToFront;
    else if (str == "FrontToBack" || str == "FirstToLast")
        return LinkFlags::FrontToBack;
    else if (str == "FrontToFront" || str == "FirstToFirst")
        return LinkFlags::FrontToFront;
    else if (str == "LinkParams")
        return LinkFlags::LinkParams;
    else if (str == "PathToPath")
        return LinkFlags::PathToPath;
    else {
        auto msg = fmt::format("Unrecognized LinkFlag: {0}\n", str);
        throw std::invalid_argument(msg);
    }
}

inline ControlModes strto_ControlMode(const std::string &str) {

    if (str == "FirstOrderSpline")
        return ControlModes::FirstOrderSpline;
    else if (str == "BlockConstant")
        return ControlModes::BlockConstant;
    else if (str == "HighestOrderSpline")
        return ControlModes::HighestOrderSpline;
    else if (str == "NoSpline")
        return ControlModes::NoSpline;
    else {
        auto msg = fmt::format("Unrecognized ControlMode: {0}\n", str);
        throw std::invalid_argument(msg);
    }
}

} // namespace tycho
