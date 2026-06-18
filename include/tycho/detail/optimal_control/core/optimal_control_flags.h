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
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/type_name.h"

namespace tycho {

/// @ingroup optimal_control
/// @brief Region of a phase that a state function (constraint or objective) acts on.
///
/// Selects which discretized state(s) — endpoints, interior nodes, defect
/// points, or parameter blocks — are passed as arguments to a state function
/// when it is attached to a phase.
enum class PhaseRegionFlags {
    NotSet,             ///< Sentinel: region not yet assigned.
    Front,              ///< The first (initial-time) node of the phase.
    Back,               ///< The last (final-time) node of the phase.
    FrontandBack,       ///< Both endpoints, front state followed by back state.
    BackandFront,       ///< Both endpoints, back state followed by front state.
    Path,               ///< Every node of the phase (full path constraint).
    InnerPath,          ///< Every interior node, excluding the two endpoints.
    NodalPath,          ///< Every nodal (mesh) point along the path.
    DefectPath,         ///< Every defect (collocation) point along the path.
    PairWisePath,       ///< Each consecutive pair of nodes along the path.
    DefectPairWisePath, ///< Each consecutive pair of defect points along the path.
    FrontNodalBackPath, ///< Front node, all nodal points, then back node.
    Params,             ///< The combined parameter vector of the phase.
    ODEParams,          ///< The ODE (dynamics) parameter sub-vector.
    StaticParams,       ///< The static (non-ODE) parameter sub-vector.
    Accumulate,         ///< Accumulated (summed) contribution across the path.
    BlockDefectPath,    ///< Defect points grouped into transcription blocks.
};

/// @ingroup optimal_control
/// @brief Pairing of phase regions joined by a link (multi-phase) constraint.
///
/// Selects which states of two linked phases are passed to a link function in a
/// multi-phase optimal control problem.
enum class LinkFlags {
    BackToFront,       ///< Back of the first phase to front of the second.
    FrontToBack,       ///< Front of the first phase to back of the second.
    FrontToFront,      ///< Front of the first phase to front of the second.
    BackToBack,        ///< Back of the first phase to back of the second.
    ParamsToParams,    ///< Parameter vector of one phase to that of another.
    LinkParams,        ///< The shared link-parameter vector.
    BackTwoToTwoFront, ///< Last two states of one phase to first two of another.
    FrontTwoToTwoBack, ///< First two states of one phase to last two of another.
    PathToPath,        ///< Path nodes of one phase to path nodes of another.
    ReadRegions,       ///< Regions are read from an explicit specification.
};

/// @ingroup optimal_control
/// @brief Interpolation/representation scheme for the control history.
enum class ControlModes {
    HighestOrderSpline, ///< Spline matching the transcription's highest order.
    FirstOrderSpline,   ///< Piecewise-linear (first-order) control spline.
    NoSpline,           ///< Independent control values, no inter-node spline.
    BlockConstant,      ///< Control held constant over each transcription block.
};

/// @ingroup optimal_control
/// @brief Collocation/transcription scheme used to discretize a phase.
enum class TranscriptionModes {
    LGL3,            ///< 3rd-order Legendre-Gauss-Lobatto collocation.
    LGL5,            ///< 5th-order Legendre-Gauss-Lobatto collocation.
    LGL7,            ///< 7th-order Legendre-Gauss-Lobatto collocation.
    Trapezoidal,     ///< Trapezoidal (2nd-order) collocation.
    CentralShooting, ///< Central (forward/backward) shooting transcription.
};

/// @ingroup optimal_control
/// @brief Quadrature rule used to evaluate integral terms over a phase.
enum class IntegralModes {
    BaseIntegral,    ///< Default quadrature matched to the transcription order.
    SimpsonIntegral, ///< Simpson's-rule quadrature.
    TrapIntegral,    ///< Trapezoidal-rule quadrature.
};

/// @ingroup optimal_control
/// @brief Source of the variable/equation scaling applied to a phase.
enum class ScaleModes {
    AUTO,   ///< Scales are computed automatically from the problem.
    CUSTOM, ///< Scales are supplied explicitly by the user.
    NONE,   ///< No scaling is applied (unit scales).
};

/// @ingroup optimal_control
/// @brief Method used to estimate the per-interval mesh (discretization) error.
enum class MeshErrorEstimators {
    DEBOOR,     ///< de Boor collocation-residual error estimate.
    INTEGRATOR, ///< Error estimate from a reference numerical integrator.
};

/// @ingroup optimal_control
/// @brief Reduction used to aggregate per-interval mesh errors into a scalar.
enum class MeshErrorAggregation {
    MAX,       ///< Maximum error over all intervals.
    AVG,       ///< Arithmetic mean of per-interval errors.
    GEOMETRIC, ///< Geometric mean of per-interval errors.
    ENDTOEND,  ///< End-to-end (accumulated) trajectory error.
};

/// @brief Parse a ScaleModes value from its string name.
/// @param str  One of "auto", "custom", or "none".
/// @return The matching ScaleModes enumerator.
/// @throws std::invalid_argument if @p str is not a recognized scale mode.
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

/// @brief Parse a MeshErrorEstimators value from its string name.
/// @param str  One of "deboor" or "integrator".
/// @return The matching MeshErrorEstimators enumerator.
/// @throws std::invalid_argument if @p str is not a recognized estimator.
inline MeshErrorEstimators strto_MeshErrorEstimator(const std::string &str) {
    if (str == "deboor")
        return MeshErrorEstimators::DEBOOR;
    else if (str == "integrator")
        return MeshErrorEstimators::INTEGRATOR;
    else {
        throw std::invalid_argument(fmt::format("Unrecognized MeshErrorEstimator: {0}", str));
    }
}

/// @brief Parse a MeshErrorAggregation value from its string name.
/// @param str  One of "max", "avg", "geometric", or "endtoend".
/// @return The matching MeshErrorAggregation enumerator.
/// @throws std::invalid_argument if @p str is not a recognized aggregation.
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

/// @brief Parse a single PhaseRegionFlags value from its string name.
/// @param str  Region name (e.g. "Front", "Back", "Path", "ODEParams");
///             several names accept a "First"/"Last" synonym.
/// @return The matching PhaseRegionFlags enumerator.
/// @throws std::invalid_argument if @p str is not a recognized region.
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

/// @brief Parse a vector of PhaseRegionFlags from their string names.
/// @param strs  Region names, one per output entry.
/// @return Column vector of the matching PhaseRegionFlags enumerators.
/// @throws std::invalid_argument if any name is not a recognized region.
inline Eigen::Matrix<PhaseRegionFlags, -1, 1>
strto_PhaseRegionFlag(const std::vector<std::string> &strs) {
    Eigen::Matrix<PhaseRegionFlags, -1, 1> regvec(strs.size());

    for (int i = 0; i < strs.size(); i++) {
        regvec[i] = strto_PhaseRegionFlag(strs[i]);
    }
    return regvec;
}

/// @brief Parse a TranscriptionModes value from its string name.
/// @param str  One of "LGL3", "LGL5", "LGL7", "CentralShooting", or "Trapezoidal".
/// @return The matching TranscriptionModes enumerator.
/// @throws std::invalid_argument if @p str is not a recognized transcription mode.
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

/// @brief Parse a LinkFlags value from its string name.
/// @param str  Link name (e.g. "BackToBack", "BackToFront", "LinkParams",
///             "PathToPath"); the directional names accept "First"/"Last" synonyms.
/// @return The matching LinkFlags enumerator.
/// @throws std::invalid_argument if @p str is not a recognized link flag.
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

/// @brief Parse a ControlModes value from its string name.
/// @param str  One of "FirstOrderSpline", "BlockConstant", "HighestOrderSpline",
///             or "NoSpline".
/// @return The matching ControlModes enumerator.
/// @throws std::invalid_argument if @p str is not a recognized control mode.
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
