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
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Namespace: Tycho
// =============================================================================

#include <signal.h>

#include "Astro/tycho_astro.h"
#include "OptimalControl/tycho_optimal_control.h"
#include "Solvers/tycho_solvers.h"
#include "Utils/tycho_utils.h"
#include "VectorFunctions/tycho_vector_functions.h"
#include "pch.h"

namespace Tycho {
void VectorFunctionBuild(FunctionRegistry &reg, nb::module_ &m);
void SolversBuild(FunctionRegistry &reg, nb::module_ &m);
void OptimalControlBuild(FunctionRegistry &reg, nb::module_ &m);
void UtilsBuild(nb::module_ &m);
void AstroBuild(FunctionRegistry &reg, nb::module_ &m);
void ExtensionsBuild(FunctionRegistry &reg, nb::module_ &m);
} // namespace Tycho

using namespace Tycho;

////////////////////////////////////////////////////////////////////////////

void SoftwareInfo() {

    int tcount = std::thread::hardware_concurrency();
    int ccount = tycho::utils::get_core_count();
    int vsize = tycho::DefaultSuperScalar::SizeAtCompileTime;

    std::string assetversion = std::string(TYCHO_VERSIONSTRING);
    std::string osversion = std::string(TYCHO_OS) + " " + std::string(TYCHO_OSVERSION);

    std::string syscorecount = std::to_string(ccount);
    std::string systhreadcount = std::to_string(tcount);

    std::string compiler =
        std::string(TYCHO_COMPILERSTRING) + std::string(" ") + std::string(TYCHO_COMPILERVERSION);
    std::string pythonv = std::to_string(PY_MAJOR_VERSION) + "." + std::to_string(PY_MINOR_VERSION);
    std::string vecversion;
    if (vsize == 2) {
#ifdef __ARM_NEON
        vecversion = "ARM NEON - 128 bit - 2 doubles";
#else
        vecversion = "SSE - 128 bit - 2 doubles";
#endif
    } else if (vsize == 4) {
        vecversion = "AVX2 - 256 bit - 4 doubles";
    } else if (vsize == 8) {
        vecversion = "AVX512 - 512 bit - 8 doubles";
    }

    std::string TYCHO_STR = R"(                  ████████╗██╗   ██╗ ██████╗██╗  ██╗ ██████╗
                  ╚══██╔══╝╚██╗ ██╔╝██╔════╝██║  ██║██╔═══██╗
                     ██║    ╚████╔╝ ██║     ███████║██║   ██║
                     ██║     ╚██╔╝  ██║     ██╔══██║██║   ██║
                     ██║      ██║   ╚██████╗██║  ██║╚██████╔╝
                     ╚═╝      ╚═╝    ╚═════╝╚═╝  ╚═╝ ╚═════╝)";

    fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
    fmt::print(fmt::fg(fmt::color::crimson), "\n{}\n", TYCHO_STR);
    fmt::print(fmt::fg(fmt::color::crimson), "{0:^{1}}\n", "Trajectory Optimization Library", 79);
    fmt::print(fmt::fg(fmt::color::white), "{0:^{1}}\n",
               "Forked from ASSET (github.com/AlabamaASRL/asset_asrl)", 79);
    fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n\n", "", 79);
    fmt::print(fmt::fg(fmt::color::royal_blue), " Software Version     : ");
    fmt::print(fmt::fg(fmt::color::white), "{}", assetversion);
    fmt::print("\n");
    fmt::print(fmt::fg(fmt::color::royal_blue), " Python   Version     : ");
    fmt::print(fmt::fg(fmt::color::white), "{}", pythonv);
    fmt::print("\n");

    fmt::print(fmt::fg(fmt::color::royal_blue), " System Core Count    : ");
    fmt::print(fmt::fg(fmt::color::white), "{}", syscorecount);
    fmt::print("\n");
    fmt::print(fmt::fg(fmt::color::royal_blue), " System Thread Count  : ");
    fmt::print(fmt::fg(fmt::color::white), "{}", systhreadcount);
    fmt::print("\n");
    fmt::print(fmt::fg(fmt::color::royal_blue), " Vectorization Mode   : ");
    fmt::print(fmt::fg(fmt::color::white), "{}", vecversion);
    fmt::print("\n");
    fmt::print(fmt::fg(fmt::color::royal_blue), " Linear Solver        : ");
#ifndef USE_ACCELERATE_SPARSE
    fmt::print(fmt::fg(fmt::color::white), "Intel MKL Pardiso");
#else
    fmt::print(fmt::fg(fmt::color::white), "Apple Accelerate Sparse");
#endif
    fmt::print("\n");

    fmt::print(fmt::fg(fmt::color::royal_blue), " Compiled With        : ");
    fmt::print(fmt::fg(fmt::color::white), "{}", compiler);
    fmt::print("\n");
    fmt::print(fmt::fg(fmt::color::royal_blue), " Compiled On          : ");
    fmt::print(fmt::fg(fmt::color::white), "{}", osversion);
    fmt::print("\n");

    fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n\n", "", 79);
    fmt::print(fmt::fg(fmt::color::royal_blue),
               " Copyright/Licensing Notices : See package's .dist.data folder for full text\n");

    fmt::print(fmt::fg(fmt::color::white), "  ASSET    :Apache 2.0 | Copyright (c) 2020-present "
                                           "The University of Alabama-Astrodynamics "
                                           "and Space Research Lab\n");
    fmt::print(fmt::fg(fmt::color::white),
               "  nanobind :BSD | Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>\n");
    fmt::print(
        fmt::fg(fmt::color::white),
        "  Intel MKL:Intel Simplified Software License (Version October 2022) | Copyright (c) 2022 "
        "Intel Corporation \n");
    fmt::print(fmt::fg(fmt::color::white),
               "  Eigen    :MPL-2.0. | Copyright (c) Eigen Developers \n");
    fmt::print(fmt::fg(fmt::color::white),
               "  fmt      :MIT | Copyright (c) 2012 - present, Victor Zverovich \n");
}

void signal_callback(int sig) {
    fmt::print(fmt::fg(fmt::color::red),
               "Interrupt signal [{0}] received, terminating program.\n\n\n\n\n\n\n\n", sig);
    exit(sig);
}

int main() {
    using std::cin;
    using std::cout;
    using std::endl;

    tycho::utils::enable_color_console();

    signal(SIGINT, signal_callback);

    SoftwareInfo();

    return 0;
}

NB_MODULE(_tychopy, m) {

    tycho::utils::enable_color_console(); // This only does something on windows

    signal(SIGINT, signal_callback);

    m.doc() = "Tycho"; // optional module docstring
    m.def("PyMain", &main);
    m.def("SoftwareInfo", &SoftwareInfo);

    FunctionRegistry reg(m);     // Must be built first
    VectorFunctionBuild(reg, m); // Must be built second
    SolversBuild(reg, m);        // Builds Third so that PSIOPT shows up better in autocomplete
    OptimalControlBuild(reg, m);
    UtilsBuild(m);
    AstroBuild(reg, m);

    ExtensionsBuild(reg, reg.extmod);
}
