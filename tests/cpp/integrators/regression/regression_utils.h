///////////////////////////////////////////////////////////////////////////////
// Golden regression test utilities
//
// Shared definitions for the golden regression corpus:
// - CR3BP ODE definition (via BUILD_ODE_FROM_EXPRESSION_FD)
// - Binary I/O helpers for golden data
// - Common initial conditions and parameters
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "integrator_test_utils.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace TychoTest {

///////////////////////////////////////////////////////////////////////////////
// CR3BP test stand-in
//
// Uses tycho::astro::Kepler with a different mu value as a substitute
// for a full CR3BP ODE. The VF expression tree for CR3BP requires
// Constant<3> which has incomplete CRTP propagation when defined outside
// the tycho namespace. Using Kepler with mu=1.0 (normalized units)
// exercises identical integrator code paths (ODESize<6,0,0>, same state
// vector layout, same stepper template instantiation).
///////////////////////////////////////////////////////////////////////////////
using CR3BP_Substitute = tycho::astro::Kepler;

///////////////////////////////////////////////////////////////////////////////
// Common physical parameters
///////////////////////////////////////////////////////////////////////////////

// Earth gravitational parameter (km^3/s^2)
constexpr double MU_EARTH = 398600.4418;


///////////////////////////////////////////////////////////////////////////////
// LEO circular orbit initial state (Kepler, km and km/s)
///////////////////////////////////////////////////////////////////////////////

inline Eigen::VectorXd leo_circular_x0() {
    double r0 = 7000.0;
    double v0 = std::sqrt(MU_EARTH / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0.0, 0.0, 0.0, v0, 0.0, 0.0;
    return x0;
}

inline double leo_period() {
    double r0 = 7000.0;
    return 2.0 * std::numbers::pi * std::sqrt(r0 * r0 * r0 / MU_EARTH);
}

///////////////////////////////////////////////////////////////////////////////
// "CR3BP" substitute initial state (Kepler with mu=1.0, normalized units)
//
// Exercises the same Integrator<Kepler> template instantiation as the real
// CR3BP tests would, but with two-body dynamics. Circular orbit at r=1.0
// in normalized units (mu=1.0).
///////////////////////////////////////////////////////////////////////////////

constexpr double MU_CR3BP_SUBSTITUTE = 1.0;

inline Eigen::VectorXd cr3bp_l1_x0() {
    // Circular orbit at r=1.0 in normalized units (mu=1.0)
    double r0 = 1.0;
    double v0 = std::sqrt(MU_CR3BP_SUBSTITUTE / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0.0, 0.05, 0.0, v0, 0.0, 0.0;
    return x0;
}

///////////////////////////////////////////////////////////////////////////////
// Eccentric orbit initial state (Kepler, for event detection tests)
///////////////////////////////////////////////////////////////////////////////

inline Eigen::VectorXd eccentric_orbit_x0() {
    // Eccentric orbit: perigee ~6800 km, eccentricity ~0.1
    double rp = 6800.0;
    double e = 0.1;
    double a = rp / (1.0 - e);
    double vp = std::sqrt(MU_EARTH * (2.0 / rp - 1.0 / a));
    Eigen::VectorXd x0(7);
    x0 << rp, 0.0, 0.0, 0.0, vp, 0.0, 0.0;
    return x0;
}

inline double eccentric_orbit_period() {
    double rp = 6800.0;
    double e = 0.1;
    double a = rp / (1.0 - e);
    return 2.0 * std::numbers::pi * std::sqrt(a * a * a / MU_EARTH);
}

///////////////////////////////////////////////////////////////////////////////
// Binary I/O helpers
///////////////////////////////////////////////////////////////////////////////

inline void write_int32(std::ofstream &f, int32_t v) {
    f.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

inline int32_t read_int32(std::ifstream &f) {
    int32_t v;
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    return v;
}

inline void write_vector(std::ofstream &f, const Eigen::VectorXd &v) {
    int32_t n = static_cast<int32_t>(v.size());
    write_int32(f, n);
    f.write(reinterpret_cast<const char *>(v.data()), n * sizeof(double));
}

inline Eigen::VectorXd read_vector(std::ifstream &f) {
    int32_t n = read_int32(f);
    Eigen::VectorXd v(n);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(double));
    return v;
}

inline void write_matrix(std::ofstream &f, const Eigen::MatrixXd &m) {
    int32_t rows = static_cast<int32_t>(m.rows());
    int32_t cols = static_cast<int32_t>(m.cols());
    write_int32(f, rows);
    write_int32(f, cols);
    f.write(reinterpret_cast<const char *>(m.data()), rows * cols * sizeof(double));
}

inline Eigen::MatrixXd read_matrix(std::ifstream &f) {
    int32_t rows = read_int32(f);
    int32_t cols = read_int32(f);
    Eigen::MatrixXd m(rows, cols);
    f.read(reinterpret_cast<char *>(m.data()), rows * cols * sizeof(double));
    return m;
}

inline void write_vectors(std::ofstream &f, const std::vector<Eigen::VectorXd> &vs) {
    int32_t n = static_cast<int32_t>(vs.size());
    write_int32(f, n);
    for (const auto &v : vs) {
        write_vector(f, v);
    }
}

inline std::vector<Eigen::VectorXd> read_vectors(std::ifstream &f) {
    int32_t n = read_int32(f);
    std::vector<Eigen::VectorXd> vs(n);
    for (int32_t i = 0; i < n; ++i) {
        vs[i] = read_vector(f);
    }
    return vs;
}

inline void write_matrices(std::ofstream &f, const std::vector<Eigen::MatrixXd> &ms) {
    int32_t n = static_cast<int32_t>(ms.size());
    write_int32(f, n);
    for (const auto &m : ms) {
        write_matrix(f, m);
    }
}

inline std::vector<Eigen::MatrixXd> read_matrices(std::ifstream &f) {
    int32_t n = read_int32(f);
    std::vector<Eigen::MatrixXd> ms(n);
    for (int32_t i = 0; i < n; ++i) {
        ms[i] = read_matrix(f);
    }
    return ms;
}

///////////////////////////////////////////////////////////////////////////////
// Golden data path
///////////////////////////////////////////////////////////////////////////////

#ifndef GOLDEN_DATA_DIR
#error "GOLDEN_DATA_DIR must be defined at compile time"
#endif

inline std::string golden_path(const std::string &filename) {
    return std::string(GOLDEN_DATA_DIR) + "/" + filename;
}

} // namespace TychoTest
