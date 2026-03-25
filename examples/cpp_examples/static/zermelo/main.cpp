///////////////////////////////////////////////////////////////////////////////
// Zermelo Navigation — C++ example (static DSL)
//
// Minimum-time navigation through wind fields.  A boat travels from point A
// to point B at maximum speed vMax while the wind pushes it.  The control
// variable theta selects the heading.
//
// State  : [x, y]    (2D position)
// Control: [theta]   (heading angle, measured from +x)
//
// Augmented phase vector layout: [x, y, t, theta]
//                                  0  1  2     3
//
// ODE:
//   xdot = vMax * cos(theta) + wx(x,y,t)
//   ydot = vMax * sin(theta) + wy(x,y,t)
//
// Corresponds to the Python example in examples/Zermelo.py.
//
// === PAIN POINTS (for Phase 7 static DSL improvements) ===
// 1. No composable wind functions: each wind model requires its own ODE type.
//    In Python, the wind function is a callable passed to the ODE constructor.
//    In C++, you must define a separate _Impl struct + BUILD_ODE_FROM_EXPRESSION
//    per wind model, duplicating the boat dynamics in each.
// 2. The navigate() solver function must be a template parameterised on the
//    ODE type, even though the problem structure is identical for all wind
//    models.  A runtime-polymorphic ODE (GenericODE) would avoid this but adds
//    type-erasure overhead and a different construction pattern.
// 3. Manual index tracking: coeff<0>(), coeff<3>() instead of named variables.
// 4. Time-dependent models require Arguments<N> (total size) — there is no
//    ODEArguments equivalent in the static DSL that auto-calculates the layout.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// ODE definitions — one per wind model
///////////////////////////////////////////////////////////////////////////////

// No wind
struct ZermeloNoWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax) {
        auto args = Arguments<4>(); // [x, y, t, theta]
        auto theta = args.coeff<3>();
        return StackedOutputs{vMax * cos(theta), vMax * sin(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloNoWind, ZermeloNoWind_Impl, double);

// Uniform wind (constant velocity and direction)
struct ZermeloUniformWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax, double wVel, double wAng) {
        auto args = Arguments<4>();
        auto theta = args.coeff<3>();
        return StackedOutputs{
            vMax * cos(theta) + wVel * std::cos(wAng),
            vMax * sin(theta) + wVel * std::sin(wAng)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloUniformWind, ZermeloUniformWind_Impl, double, double, double);

// Constant-direction wind with position-dependent speed
struct ZermeloConstDirWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax, double wAng) {
        auto args = Arguments<4>();
        auto pos = args.head<2>();
        auto theta = args.coeff<3>();
        auto vel = cos(pos.norm());
        return StackedOutputs{
            vMax * cos(theta) + vel * std::cos(wAng),
            vMax * sin(theta) + vel * std::sin(wAng)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloConstDirWind, ZermeloConstDirWind_Impl, double, double);

// Variable-direction wind: speed and angle depend on position
struct ZermeloVarWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax) {
        auto args = Arguments<4>();
        auto pos = args.head<2>();
        auto x = args.coeff<0>();
        auto y = args.coeff<1>();
        auto theta = args.coeff<3>();
        auto vel = sin(pos.norm());
        auto ang = 2.0 * (x + y);
        return StackedOutputs{
            vMax * cos(theta) + vel * cos(ang),
            vMax * sin(theta) + vel * sin(ang)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloVarWind, ZermeloVarWind_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Solver — templated on ODE type (pain point: cannot take a runtime wind fn)
///////////////////////////////////////////////////////////////////////////////

template <typename ODE>
std::vector<Eigen::VectorXd> navigate(ODE &ode, const Eigen::VectorXd &A,
                                      const Eigen::VectorXd &B, double vMax) {
    constexpr int nSeg = 250;
    constexpr double tol = 1e-12;

    // Linear initial guess
    const double dist = (B - A).norm();
    const double t0 = dist / vMax;
    const Eigen::VectorXd d = (B - A) / dist;
    const double ang = std::atan2(d[1], d[0]);

    std::vector<Eigen::VectorXd> trajG;
    trajG.reserve(nSeg);
    for (int i = 0; i < nSeg; ++i) {
        const double s = static_cast<double>(i) / (nSeg - 1);
        Eigen::VectorXd pt(4);
        pt << A[0] + d[0] * s, A[1] + d[1] * s, t0 * s, ang;
        trajG.push_back(pt);
    }

    // Phase
    auto phase = std::make_shared<ODEPhase<ODE>>(ode, TranscriptionModes::LGL3);
    phase->setTraj(trajG, nSeg);
    phase->setNumPartitions(10);

    // Boundary conditions
    Eigen::VectorXi xy_idx(2);
    xy_idx << 0, 1;
    Eigen::VectorXi t_idx(1);
    t_idx << 2;
    Eigen::VectorXd t0_val(1);
    t0_val << 0.0;

    phase->addBoundaryValue(PhaseRegionFlags::Front, xy_idx, A, ScaleModes::AUTO);
    phase->addBoundaryValue(PhaseRegionFlags::Front, t_idx, t0_val, ScaleModes::AUTO);
    phase->addBoundaryValue(PhaseRegionFlags::Back, xy_idx, B, ScaleModes::AUTO);

    // Control bounds
    phase->addLUVarBound(PhaseRegionFlags::Path, 3, -M_PI, M_PI, 1.0);

    // Minimise travel time
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);

    // Solver settings (match Python: only tolerances)
    phase->optimizer->set_EContol(tol);
    phase->optimizer->set_KKTtol(tol);

    const auto status = phase->solve_optimize();
    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "  FAILED: navigation did not converge (status "
                  << static_cast<int>(status) << ")\n";
        return {};
    }
    return phase->returnTraj();
}

///////////////////////////////////////////////////////////////////////////////
// main — compare wind models (mirrors Python compareWind)
///////////////////////////////////////////////////////////////////////////////

int main() {
    int failures = 0;

    // Quick Jacobian sanity check
    {
        ZermeloVarWind ode_test(1.25);
        Eigen::VectorXd tp(4);
        tp << 0.2, -0.5, 0.5, 1.0;
        Eigen::VectorXd fx(2);
        ode_test.compute(tp, fx);
        Eigen::MatrixXd jac(2, 4);
        ode_test.jacobian(tp, jac);
        // Python: [[ 0.842 -0.079 0 -1.052] [0.667 1.297 0 0.675]]
        bool ok = std::abs(jac(0, 1) - (-0.0787)) < 0.01;
        std::cout << "ODE Jacobian dy column: " << jac(0, 1) << " " << jac(1, 1)
                  << (ok ? "  OK" : "  BUG") << "\n\n";
        if (!ok) ++failures;
    }

    Eigen::VectorXd A(2), B(2);
    A << 0.0, -1.0;
    B << 1.0, 1.0;

    // No wind
    std::cout << "Solving: no wind ... " << std::flush;
    ZermeloNoWind ode_nw(1.0);
    auto traj1 = navigate(ode_nw, A, B, 1.0);
    if (traj1.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf1 = traj1.back()[2];
        std::cout << std::fixed << std::setprecision(4) << "tf = " << tf1 << " s\n";
    }

    // Uniform wind
    std::cout << "Solving: uniform wind ... " << std::flush;
    constexpr double vM = 1.25;
    ZermeloUniformWind ode_uw(vM, 0.5, 135.0 * M_PI / 180.0);
    auto traj2 = navigate(ode_uw, A, B, vM);
    if (traj2.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf2 = traj2.back()[2];
        std::cout << "tf = " << tf2 << " s\n";
    }

    // Constant-direction wind
    std::cout << "Solving: constant-direction wind ... " << std::flush;
    ZermeloConstDirWind ode_cd(vM, 45.0 * M_PI / 180.0);
    auto traj3 = navigate(ode_cd, A, B, vM);
    if (traj3.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf3 = traj3.back()[2];
        std::cout << "tf = " << tf3 << " s\n";
    }

    // Variable-direction wind
    std::cout << "Solving: variable-direction wind ... " << std::flush;
    ZermeloVarWind ode_vw(vM);
    auto traj4 = navigate(ode_vw, A, B, vM);
    if (traj4.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf4 = traj4.back()[2];
        std::cout << "tf = " << tf4 << " s\n";
    }

    // Summary
    std::cout << "\nZermelo: " << (4 - failures) << "/4 wind models converged\n";
    return failures > 0 ? 1 : 0;
}
