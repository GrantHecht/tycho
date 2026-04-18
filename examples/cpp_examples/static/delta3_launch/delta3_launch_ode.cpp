///////////////////////////////////////////////////////////////////////////////
// Delta-III launch ODE — concrete VectorFunctions + erased factories
//
// RocketODE expression tree and MakeTargetOrbit terminal constraint tree live
// here. All compute_jacobian_adjointgradient_adjointhessian instantiations for
// these expressions are emitted exactly once in this TU.
///////////////////////////////////////////////////////////////////////////////

#include "delta3_launch_ode.h"

#include <cmath>

namespace tycho_examples {

namespace {

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

///////////////////////////////////////////////////////////////////////////////
// Non-dimensional constants (file-scope constants + lazily built struct)
///////////////////////////////////////////////////////////////////////////////

constexpr double Lstar = 6378145.0;
constexpr double Tstar = 961.0;
constexpr double Mstar = 301454.0;

const double Astar_v = Lstar / (Tstar * Tstar);
const double Vstar_v = Lstar / Tstar;
const double Rhostar_v = Mstar / (Lstar * Lstar * Lstar);
const double Mustar_v = (Lstar * Lstar * Lstar) / (Tstar * Tstar);
const double Fstar_v = Astar_v * Mstar;

const double mu_v = 3.986012e14 / Mustar_v;
const double Re_v = 6378145.0 / Lstar;
const double We_v = 7.29211585e-5 * Tstar;

const double RhoAir_v = 1.225 / Rhostar_v;
const double h_scale_v = 7200.0 / Lstar;
const double g0 = 9.80665;
const double g_v = g0 / Astar_v;

constexpr double CD = 0.5;
const double S_v = 4.0 * M_PI / (Lstar * Lstar);

// Engine parameters
const double TS = 628500.0 / Fstar_v;
const double T1c = 1083100.0 / Fstar_v;
const double T2c = 110094.0 / Fstar_v;

const double IS = 283.33364 / Tstar;
const double I1 = 301.68 / Tstar;
const double I2 = 467.21 / Tstar;

const double tS = 75.2 / Tstar;
const double t1 = 261.0 / Tstar;
const double t2 = 700.0 / Tstar;

// Stage masses
const double TMS = 19290.0 / Mstar;
const double TM1 = 104380.0 / Mstar;
const double TM2 = 19300.0 / Mstar;
const double TMPay = 4164.0 / Mstar;

const double PMS = 17010.0 / Mstar;
const double PM1 = 95550.0 / Mstar;
const double PM2 = 16820.0 / Mstar;

const double SMS = TMS - PMS;
const double SM1 = TM1 - PM1;

// Phase thrust and mass flow rates
const double T_phase1_v = 6.0 * TS + T1c;
const double T_phase2_v = 3.0 * TS + T1c;
const double T_phase3_v = T1c;
const double T_phase4_v = T2c;

const double mdot_phase1_v = (6.0 * TS / IS + T1c / I1) / g_v;
const double mdot_phase2_v = (3.0 * TS / IS + T1c / I1) / g_v;
const double mdot_phase3_v = T1c / (g_v * I1);
const double mdot_phase4_v = T2c / (g_v * I2);

// Phase times
const double tf_phase1_v = tS;
const double tf_phase2_v = 2.0 * tS;
const double tf_phase3_v = t1;
const double tf_phase4_v = t1 + t2;

// Phase masses
const double m0_phase1_v = 9.0 * TMS + TM1 + TM2 + TMPay;
const double mf_phase1_v = m0_phase1_v - 6.0 * PMS - (tS / t1) * PM1;

const double m0_phase2_v = mf_phase1_v - 6.0 * SMS;
const double mf_phase2_v = m0_phase2_v - 3.0 * PMS - (tS / t1) * PM1;

const double m0_phase3_v = mf_phase2_v - 3.0 * SMS;
const double mf_phase3_v = m0_phase3_v - (1.0 - 2.0 * tS / t1) * PM1;

const double m0_phase4_v = mf_phase3_v - SM1;
const double mf_phase4_v = m0_phase4_v - PM2;

///////////////////////////////////////////////////////////////////////////////
// RocketODE — parameterised by (thrust, mass_flow_rate)
///////////////////////////////////////////////////////////////////////////////

struct RocketODE_Impl : ODESize<7, 3, 0> {
    static auto Definition(double T, double mdot) {
        auto args = ODEArguments<7, 3, 0>();
        auto R = args[XSeg<0, 3>];
        auto V = args[XSeg<3, 3>];
        auto m = args[XVar<6>];
        auto u = args[UVec].normalized();

        auto h = R.norm() - Re_v;

        // Atmospheric density: rho = RhoAir * exp(-h / h_scale)
        auto exp_neg_h = exp(h * (-1.0 / h_scale_v));

        // Relative velocity: Vr = V + R × [0, 0, We]
        Eigen::Vector3d omega_val(0.0, 0.0, We_v);
        auto omega = Constant<11, 3>(11, omega_val);
        auto Vr = V + R.cross(omega);

        // Drag: D = -0.5 * CD * S * rho * ||Vr|| * Vr
        const double drag_rho = -0.5 * CD * S_v * RhoAir_v;
        auto D = (drag_rho * exp_neg_h) * (Vr * Vr.norm());

        auto Rdot = V;
        auto Vdot = (-mu_v) * R.normalized_power<3>() + (T * u + D) / m;

        // Constant mass flow rate as a VF (must match IR=11)
        Eigen::Matrix<double, 1, 1> neg_mdot_val;
        neg_mdot_val[0] = -mdot;
        auto neg_mdot_vf = Constant<11, 1>(11, neg_mdot_val);

        return StackedOutputs{Rdot, Vdot, neg_mdot_vf};
    }
};
BUILD_ODE_FROM_EXPRESSION_FD(RocketODE, RocketODE_Impl, double, double);

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Constants accessor
///////////////////////////////////////////////////////////////////////////////

const Delta3Constants &delta3_constants() {
    static const Delta3Constants c{
        Lstar,     Tstar,     Mstar,       Astar_v,
        Vstar_v,   Fstar_v,   Mustar_v,    Rhostar_v,
        mu_v,      Re_v,      We_v,

        T_phase1_v, T_phase2_v, T_phase3_v, T_phase4_v,
        mdot_phase1_v, mdot_phase2_v, mdot_phase3_v, mdot_phase4_v,
        tf_phase1_v, tf_phase2_v, tf_phase3_v, tf_phase4_v,
        m0_phase1_v, mf_phase1_v,
        m0_phase2_v, mf_phase2_v,
        m0_phase3_v, mf_phase3_v,
        m0_phase4_v, mf_phase4_v,
    };
    return c;
}

///////////////////////////////////////////////////////////////////////////////
// Factories
///////////////////////////////////////////////////////////////////////////////

tycho::vf::GenericFunction<-1, -1> make_rocket_ode(double T, double mdot) {
    return tycho::vf::GenericFunction<-1, -1>(RocketODE(T, mdot));
}

tycho::vf::GenericFunction<-1, -1>
make_target_orbit_residuals(double a_t, double e_t, double i_t, double raan_t, double argp_t) {
    using namespace tycho::vf;
    auto args = Arguments<6>();
    auto R = args.head<3>();
    auto V = args.tail<3>();

    auto r = R.norm();

    // Angular momentum: h = R × V
    auto hvec = R.cross(V);

    // Node vector: n = [0,0,1] × h
    Eigen::Vector3d z_hat(0.0, 0.0, 1.0);
    auto z_const = Constant<6, 3>(6, z_hat);
    auto nvec = z_const.cross(hvec);

    // Specific energy: eps = 0.5 * v^2 - mu/r
    auto eps = V.squared_norm() * 0.5 - mu_v / r;

    // Semi-major axis
    auto a = -0.5 * mu_v / eps;

    // Eccentricity vector: e_vec = (V × h)/mu - R_hat
    auto evec = V.cross(hvec) / mu_v - R.normalized();
    auto e = evec.norm();

    // Inclination
    auto inc = acos(hvec.normalized().coeff<2>());

    // RAAN — quadrant-corrected
    auto O_raw = acos(nvec.normalized().coeff<0>());
    auto O = IfElseFunction(nvec.coeff<1>() > 0.0, O_raw, 2.0 * M_PI - O_raw);

    // Argument of periapse — quadrant-corrected
    auto W_raw = acos(nvec.normalized().dot(evec.normalized()));
    auto W = IfElseFunction(evec.coeff<2>() > 0.0, W_raw, 2.0 * M_PI - W_raw);

    return tycho::vf::GenericFunction<-1, -1>(
        StackedOutputs{a - a_t, e - e_t, inc - i_t, O - raan_t, W - argp_t});
}

} // namespace tycho_examples
