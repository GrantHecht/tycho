///////////////////////////////////////////////////////////////////////////////
// Delta-III launch ODE — type-erased factories
//
// RocketODE is parameterised by (thrust, mass_flow_rate). Each of the 4 launch
// phases needs a different instance. Declares a factory returning a
// GenericFunction<-1,-1> so main.cpp does not see the concrete VectorFunction
// type. Also exposes the target-orbit terminal constraint as an erased
// GenericFunction<-1,-1> so its expression tree only instantiates once.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>

namespace tycho_examples {

// Physical and engine constants re-exported for main.cpp's scaling & boundary
// conditions.  Values are non-dimensionalised with the characteristic
// scales Lstar = 6 378 145 m, Tstar = 961 s, Mstar = 301 454 kg.
struct Delta3Constants {
    double Lstar, Tstar, Mstar;
    double Astar, Vstar, Fstar, Mustar, Rhostar;
    double mu, Re, We;

    double T_phase1, T_phase2, T_phase3, T_phase4;
    double mdot_phase1, mdot_phase2, mdot_phase3, mdot_phase4;
    double tf_phase1, tf_phase2, tf_phase3, tf_phase4;
    double m0_phase1, mf_phase1;
    double m0_phase2, mf_phase2;
    double m0_phase3, mf_phase3;
    double m0_phase4, mf_phase4;
};

const Delta3Constants &delta3_constants();

// Factory: rocket dynamics with thrust T and mass flow rate mdot.
// State: [Rx, Ry, Rz, Vx, Vy, Vz, m]  (7)
// Control: [ux, uy, uz]  (3)
tycho::vf::GenericFunction<-1, -1> make_rocket_ode(double T, double mdot);

// Target-orbit equality constraint, evaluated on [R(3), V(3)] (6 inputs),
// returning residuals on (a, e, i, RAAN, omega) in that order (5 outputs).
// Returns a GenericFunction<-1,-1> so its instantiation lives in this TU.
tycho::vf::GenericFunction<-1, -1>
make_target_orbit_residuals(double a_t, double e_t, double i_t, double raan_t, double argp_t);

} // namespace tycho_examples
