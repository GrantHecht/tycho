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

#include <array>

namespace tycho {

// RKCoeffs dense-output schema:
//
//   Main integration tableau:
//     Stages         — main integration f-evals per step
//     A, B, C        — standard Butcher tableau
//     Bhat           — embedded (error-estimator) weights (zeros if !HasEmbedded)
//     FSAL           — last main stage equals next step's first stage
//     HasMidpoint    — does this method support midpoint dense output?
//
//   Dense-output extras:
//     InterpStages   — # of *additional* f-evals needed for the method's
//                      interpolation polynomial, beyond the main Stages.
//                      0 means "no extras" (Tsit5, BS3, DOPRI54, DOPRI87).
//     LastStageIsFxf — does k_{Stages-1} (the last main stage) equal f(xf)?
//                      True for FSAL methods (by construction). Also true for
//                      BS5/Vern7/Vern8/Vern9 (their perform_step evaluates the
//                      last main stage at (t+h, xf) with a-row equal to B).
//                      False only for DOPRI87 (needs an explicit f(xf)).
//     BmidStages     — total k-values referenced by the midpoint sum:
//                      Stages + InterpStages + (LastStageIsFxf ? 0 : 1).
//     ExtraA         — InterpStages × (Stages + InterpStages) a-matrix; row e
//                      builds the argument to f for extra stage e:
//                        x_e = x + h · Σ_{j<Stages+e} ExtraA[e][j] · k_j
//                      (k_0..k_{Stages-1} are main; k_Stages..k_{Stages+e-1} are
//                       prior extras.)
//     ExtraC         — c-value for each extra stage (t_e = t0 + ExtraC[e]·h).
//     Bmid           — weights for the midpoint sum, stored as 2·b_i(Θ=0.5)
//                      where b_i(Θ) is Julia's interpolation polynomial.
//                      Factor of 2 cancels the /2 applied inside
//                      stepper_compute_impl. Σ Bmid[i] = 1.0 sanity-check.
//                      Layout:
//                        Bmid[0..Stages-1]                     — main stages
//                        Bmid[Stages]                          — f(xf) term (only if
//                        !LastStageIsFxf) Bmid[Stages + offset .. BmidStages-1] — extra stages
//                        (offset = LastStageIsFxf ? 0 : 1)
//
// The doxygen list below marks which enum values are user-selectable via
// the Python binding + string parser. The remaining values (RK4Classic,
// DOPRI5, *Trans tags) serve as internal template-dispatch tags — some
// adaptive steppers instantiate against the non-public tableau (e.g.
// DOPRI54's adaptive stepper uses IVPAlg::DOPRI5 coefficients), and the
// transcription paths use the *Trans variants; rk_steppers.h branches on
// them. Passing a non-user-selectable value to set_method throws.
enum class IVPAlg {
    DOPRI54, ///< Dormand-Prince 5(4) — 7 stages, adaptive
    DOPRI87, ///< Dormand-Prince 8(7) — 13 stages, adaptive (default)
    Tsit5,   ///< Tsitouras 5(4) — 7 stages, FSAL, adaptive
    BS3,     ///< Bogacki-Shampine 3(2) — 4 stages, FSAL, adaptive
    BS5,     ///< Bogacki-Shampine 5(4) — 8 stages + 3 interp extras
    Vern7,   ///< Verner 7(6) — 10 stages + 6 interp extras
    Vern8,   ///< Verner 8(7) — 13 stages + 8 interp extras
    Vern9,   ///< Verner 9(8) — 16 stages + 10 interp extras
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// set_method() throws on this value. Do not expose via bindings.
    RK4Classic,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// set_method() throws on this value. Do not expose via bindings.
    DOPRI5,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// 6-stage non-FSAL transcription companion for Tsit5. Do not expose via bindings.
    Tsit5Trans,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// 3-stage non-FSAL transcription companion for BS3. Do not expose via bindings.
    BS3Trans,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// 7-stage non-FSAL transcription companion for BS5 (drops k_8 which equals f(xf)).
    /// Do not expose via bindings.
    BS5Trans,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// 9-stage non-FSAL transcription companion for Vern7 (drops k_10 which is a
    /// helper evaluated at t+h but not equal to f(xf)). Do not expose via bindings.
    Vern7Trans,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// 12-stage non-FSAL transcription companion for Vern8 (drops k_13 which is
    /// only used for error estimation). Do not expose via bindings.
    Vern8Trans,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// 15-stage non-FSAL transcription companion for Vern9 (drops k_16 which is
    /// only used for error estimation). Do not expose via bindings.
    Vern9Trans,
};

} // namespace tycho

namespace tycho::integrators {

using tycho::IVPAlg;

// Compile-time rational helper for transcribing Butcher tableau coefficients.
//
// Mirrors Julia's `N // D` rational notation used in OrdinaryDiffEq.jl's
// generic (`T::Type`) tableau variants. For Float64 targets, `rat(N, D)` is
// bit-identical to Julia's `convert(Float64, N // D)` because both perform
// IEEE 754 round-to-nearest on a single division.
//
// Usage convention in RKCoeffs specializations:
//   - Use `rat(N, D)` for rational coefficients (matches Julia `N // D`).
//   - Use Float64 literals for irrational coefficients, copied from Julia's
//     `CompiledFloats` variant (matches Julia `convert(Float64, big"...")`).
//
// `long long` covers any denominator we encounter (up to 2^63 - 1); all
// existing Verner/BS/Tsit5 denominators fit well within this range.
constexpr double rat(long long n, long long d) {
    return static_cast<double>(n) / static_cast<double>(d);
}

template <IVPAlg opt> struct RKCoeffs {};

template <> struct RKCoeffs<IVPAlg::RK4Classic> {
    static constexpr int Stages = 4;
    static constexpr int Order = 4;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // fixed-step, no dense output

    // Dense-output schema fields — all zero because HasMidpoint=false.
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    static constexpr std::array<std::array<double, 3>, 3> A = {
        std::array<double, 3>{rat(1, 2), 0.0, 0.0}, std::array<double, 3>{0.0, rat(1, 2), 0.0},
        std::array<double, 3>{0.0, 0.0, 1.0}};

    static constexpr std::array<double, 3> C = {rat(1, 2), rat(1, 2), 1.0};
    static constexpr std::array<double, 4> B = {rat(1, 6), rat(1, 3), rat(1, 3), rat(1, 6)};
    static constexpr std::array<double, 4> Bhat = {0, 0, 0, 0};
};

template <> struct RKCoeffs<IVPAlg::DOPRI54> {
    static constexpr int Stages = 7;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 4;
    static constexpr bool FSAL = true;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // Dense-output schema fields (no extra interpolation stages — DOPRI54
    // uses its own 5th-order interpolation polynomial over existing main stages).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = FSAL;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // Butcher tableau — matches Julia OrdinaryDiffEqLowOrderRK DP5ConstantCache
    // (generic T::Type variant with rational coefficients).
    static constexpr std::array<std::array<double, 6>, 6> A = {
        std::array<double, 6>{rat(1, 5), 0, 0, 0, 0, 0},
        std::array<double, 6>{rat(3, 40), rat(9, 40), 0, 0, 0, 0},
        std::array<double, 6>{rat(44, 45), rat(-56, 15), rat(32, 9), 0, 0, 0},
        std::array<double, 6>{rat(19372, 6561), rat(-25360, 2187), rat(64448, 6561), rat(-212, 729),
                              0, 0},
        std::array<double, 6>{rat(9017, 3168), rat(-355, 33), rat(46732, 5247), rat(49, 176),
                              rat(-5103, 18656), 0},
        std::array<double, 6>{rat(35, 384), 0.0, rat(500, 1113), rat(125, 192), rat(-2187, 6784),
                              rat(11, 84)}};

    static constexpr std::array<double, 6> C = {rat(1, 5), rat(3, 10), rat(4, 5),
                                                rat(8, 9), 1.0,        1.0};
    static constexpr std::array<double, 7> B = {
        rat(35, 384), 0.0, rat(500, 1113), rat(125, 192), rat(-2187, 6784), rat(11, 84), 0.0};
    static constexpr std::array<double, 7> Bhat = {
        rat(5179, 57600), 0.0,       rat(7571, 16695), rat(393, 640), rat(-92097, 339200),
        rat(187, 2100),   rat(1, 40)};

    static constexpr std::array<double, BmidStages> Bmid = {
        0.2002686376600479, 0.0000000000000000,  0.7836643588368518, -0.0596492035318963,
        0.1178653667448159, -0.0899577761820872, 0.0478086164722679};
};

template <> struct RKCoeffs<IVPAlg::DOPRI5> {
    static constexpr int Stages = 6;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // transcription-only, no dense output

    // Dense-output schema fields — unused (HasMidpoint=false).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // 6-stage non-FSAL transcription companion for DOPRI54 — same first 6
    // stages as DOPRI54's A matrix with the FSAL-last row dropped.
    static constexpr std::array<std::array<double, 5>, 5> A = {
        std::array<double, 5>{rat(1, 5), 0, 0, 0, 0},
        std::array<double, 5>{rat(3, 40), rat(9, 40), 0, 0, 0},
        std::array<double, 5>{rat(44, 45), rat(-56, 15), rat(32, 9), 0, 0},
        std::array<double, 5>{rat(19372, 6561), rat(-25360, 2187), rat(64448, 6561), rat(-212, 729),
                              0},
        std::array<double, 5>{rat(9017, 3168), rat(-355, 33), rat(46732, 5247), rat(49, 176),
                              rat(-5103, 18656)}};

    static constexpr std::array<double, 5> C = {rat(1, 5), rat(3, 10), rat(4, 5), rat(8, 9), 1.0};
    static constexpr std::array<double, 6> B = {
        rat(35, 384), 0.0, rat(500, 1113), rat(125, 192), rat(-2187, 6784), rat(11, 84)};
    static constexpr std::array<double, 6> Bhat = {rat(5179, 57600),    0.0,
                                                   rat(7571, 16695),    rat(393, 640),
                                                   rat(-92097, 339200), rat(187, 2100)};
};

template <> struct RKCoeffs<IVPAlg::DOPRI87> {
    static constexpr int Stages = 13;
    static constexpr int Order = 8;
    static constexpr int ErrorOrder = 7;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // Dense-output schema fields — DOPRI87's last main stage is not f(xf),
    // so the midpoint branch must evaluate f(xf) as an extra k-value. No other
    // interpolation stages are needed.
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // Butcher tableau — matches Julia OrdinaryDiffEqHighOrderRK DP8ConstantCache
    // (generic T::Type variant with rational coefficients).
    static constexpr std::array<std::array<double, 12>, 12> A = {
        std::array<double, 12>{rat(1, 18), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{rat(1, 48), rat(1, 16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{rat(1, 32), 0, rat(3, 32), 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{rat(5, 16), 0, rat(-75, 64), rat(75, 64), 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{rat(3, 80), 0, 0, rat(3, 16), rat(3, 20), 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{rat(29443841, 614563906), 0, 0, rat(77736538, 692538347),
                               rat(-28693883, 1125000000), rat(23124283, 1800000000), 0, 0, 0, 0, 0,
                               0},
        std::array<double, 12>{rat(16016141, 946692911), 0, 0, rat(61564180, 158732637),
                               rat(22789713, 633445777), rat(545815736, 2771057229),
                               rat(-180193667, 1043307555), 0, 0, 0, 0, 0},
        std::array<double, 12>{rat(39632708, 573591083), 0, 0, rat(-433636366, 683701615),
                               rat(-421739975, 2616292301), rat(100302831, 723423059),
                               rat(790204164, 839813087), rat(800635310, 3783071287), 0, 0, 0, 0},
        std::array<double, 12>{rat(246121993, 1340847787), 0, 0, rat(-37695042795, 15268766246),
                               rat(-309121744, 1061227803), rat(-12992083, 490766935),
                               rat(6005943493, 2108947869), rat(393006217, 1396673457),
                               rat(123872331, 1001029789), 0, 0, 0},
        std::array<double, 12>{rat(-1028468189, 846180014), 0, 0, rat(8478235783, 508512852),
                               rat(1311729495, 1432422823), rat(-10304129995, 1701304382),
                               rat(-48777925059, 3047939560), rat(15336726248, 1032824649),
                               rat(-45442868181, 3398467696), rat(3065993473, 597172653), 0, 0},
        std::array<double, 12>{rat(185892177, 718116043), 0, 0, rat(-3185094517, 667107341),
                               rat(-477755414, 1098053517), rat(-703635378, 230739211),
                               rat(5731566787, 1027545527), rat(5232866602, 850066563),
                               rat(-4093664535, 808688257), rat(3962137247, 1805957418),
                               rat(65686358, 487910083), 0},
        std::array<double, 12>{rat(403863854, 491063109), 0, 0, rat(-5068492393, 434740067),
                               rat(-411421997, 543043805), rat(652783627, 914296604),
                               rat(11173962825, 925320556), rat(-13158990841, 6184727034),
                               rat(3936647629, 1978049680), rat(-160528059, 685178525),
                               rat(248638103, 1413531060), 0},
    };

    static constexpr std::array<double, 12> C = {rat(1, 18),   rat(1, 12),
                                                 rat(1, 8),    rat(5, 16),
                                                 rat(3, 8),    rat(59, 400),
                                                 rat(93, 200), rat(5490023248, 9719169821),
                                                 rat(13, 20),  rat(1201146811, 1299019798),
                                                 1.0,          1.0};
    static constexpr std::array<double, 13> B = {rat(14005451, 335480064),
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 rat(-59238493, 1068277825),
                                                 rat(181606767, 758867731),
                                                 rat(561292985, 797845732),
                                                 rat(-1041891430, 1371343529),
                                                 rat(760417239, 1151165299),
                                                 rat(118820643, 751138087),
                                                 rat(-528747749, 2220607170),
                                                 rat(1, 4)};
    static constexpr std::array<double, 13> Bhat = {rat(13451932, 455176623),
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    rat(-808719846, 976000145),
                                                    rat(1757004468, 5645159321),
                                                    rat(656045339, 265891186),
                                                    rat(-3867574721, 1518517206),
                                                    rat(465885868, 322736535),
                                                    rat(53011238, 667516719),
                                                    rat(2, 45),
                                                    0};

    static constexpr std::array<double, BmidStages> Bmid = {
        0.0820626072147879,  0.0000000000000000, 0.0000000000000000,  0.0000000000000000,
        0.0000000000000000,  0.1020112560398276, 0.4777861354824404,  0.6193740287992207,
        -0.4344650943510704, 0.1566681135866386, -0.0037228739431160, 0.0141456884053577,
        0.0138169474640115,  -0.0276768086980947};
};

template <> struct RKCoeffs<IVPAlg::Tsit5> {
    static constexpr int Stages = 7;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 4;
    static constexpr bool FSAL = true;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // Dense-output schema fields (Tsit5's own 4th-order interpolation
    // polynomial uses only the 7 main stages; no extras needed).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = FSAL;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 6 rows (stages 2..7). Row i holds A[i+2, 0..i+1].
    // Source: OrdinaryDiffEqTsit5 Tsit5ConstantCacheActual (CompiledFloats variant)
    static constexpr std::array<std::array<double, 6>, 6> A = {
        std::array<double, 6>{0.161, 0, 0, 0, 0, 0},
        std::array<double, 6>{-0.008480655492356989, 0.335480655492357, 0, 0, 0, 0},
        std::array<double, 6>{2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0, 0},
        std::array<double, 6>{5.325864828439257, -11.748883564062828, 7.4955393428898365,
                              -0.09249506636175525, 0, 0},
        std::array<double, 6>{5.86145544294642, -12.92096931784711, 8.159367898576159,
                              -0.071584973281401, -0.028269050394068383, 0},
        // Row 7 (FSAL) = b vector itself:
        std::array<double, 6>{0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
                              -3.290069515436081, 2.324710524099774}};

    static constexpr std::array<double, 6> C = {0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0};

    // B (5th-order weights): 7th row of A (FSAL structure) with b7 = 0.
    static constexpr std::array<double, 7> B = {
        0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081,
        2.324710524099774,   0.0};

    // Bhat = B - btilde. btilde from tsit_tableaus.jl:
    // (-0.001780011052225777, -0.0008164344596567469, 0.007880878010261995,
    //  -0.1447110071732629, 0.5823571654525552, -0.45808210592918697, 0.015151515151515152)
    static constexpr std::array<double, 7> Bhat = {0.09646076681806523 - (-0.001780011052225777),
                                                   0.01 - (-0.0008164344596567469),
                                                   0.4798896504144996 - 0.007880878010261995,
                                                   1.379008574103742 - (-0.1447110071732629),
                                                   -3.290069515436081 - 0.5823571654525552,
                                                   2.324710524099774 - (-0.45808210592918697),
                                                   0.0 - 0.015151515151515152};

    // Bmid: stored in Tycho convention as 2·b_i(Θ=0.5), where b_i is Julia's
    // Tsit5Interp polynomial. The factor of 2 cancels the 1/2 applied in
    // stepper_compute_impl's midpoint sum, yielding sol(Θ=0.5) exactly.
    // Derived via bench/julia_reference/src/compute_bmid.jl.
    // Sanity: Σ Bmid[i] = 1.0 (matches existing DOPRI54 Bmid convention).
    static constexpr std::array<double, BmidStages> Bmid = {
        2.14824704601937561e-01,  2.27124999999999966e-02, 7.91218061120906202e-01,
        -6.89504287051870612e-01, 2.63237072991632992e+00, -2.03412170858730335e+00,
        6.25000000000000000e-02};
};

template <> struct RKCoeffs<IVPAlg::BS3> {
    static constexpr int Stages = 4;
    static constexpr int Order = 3;
    static constexpr int ErrorOrder = 2;
    static constexpr bool FSAL = true;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // Dense-output schema fields (BS3 falls back to Hermite cubic in Julia;
    // no extras needed).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = FSAL;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 3 rows (stages 2..4). Row 3 (FSAL) equals b-vector.
    // Source: OrdinaryDiffEqLowOrderRK BS3ConstantCache (generic T::Type
    // variant with rational coefficients).
    static constexpr std::array<std::array<double, 3>, 3> A = {
        std::array<double, 3>{rat(1, 2), 0, 0}, std::array<double, 3>{0, rat(3, 4), 0},
        std::array<double, 3>{rat(2, 9), rat(1, 3), rat(4, 9)}};

    static constexpr std::array<double, 3> C = {rat(1, 2), rat(3, 4), 1.0};
    static constexpr std::array<double, 4> B = {rat(2, 9), rat(1, 3), rat(4, 9), 0.0};

    // Bhat = 2nd-order embedded weights from Bogacki-Shampine 1989 paper.
    // Equivalent to Julia's b + btilde where btilde = bhat - b = (5/72, -1/12, -1/9, 1/8).
    static constexpr std::array<double, 4> Bhat = {rat(7, 24), rat(1, 4), rat(1, 3), rat(1, 8)};

    // Bmid: BS3 has no dedicated interp polynomial in OrdinaryDiffEqLowOrderRK
    // (confirmed by grep for BS3 in interpolants.jl, interp_func.jl,
    // low_order_rk_addsteps.jl — zero matches). Julia falls back to Hermite
    // cubic at Θ=0.5 via the generic dense-output path.
    //
    //   Hermite cubic at Θ=0.5:
    //     y_mid = y_0 + h·Σ [B_i/2 + (1/8)·δ_{i,1} - (1/8)·δ_{i,FSAL-last}] · f_i
    //   BS3 FSAL: last stage k_4 = f(xf), B_4 = 0.
    //     b_1(0.5) = B_1/2 + 1/8 = 1/9 + 1/8 = 17/72
    //     b_2(0.5) = B_2/2       = 1/6
    //     b_3(0.5) = B_3/2       = 2/9
    //     b_4(0.5) = B_4/2 - 1/8 = -1/8
    //   Σ b_i(0.5) = 1/2 ✓. Tycho stores 2·b_i(0.5): Σ = 1.0 ✓.
    static constexpr std::array<double, BmidStages> Bmid = {rat(17, 36), rat(1, 3), rat(4, 9),
                                                            rat(-1, 4)};
};

template <> struct RKCoeffs<IVPAlg::BS3Trans> {
    static constexpr int Stages = 3;
    static constexpr int Order = 3;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // transcription-only, no dense output

    // Dense-output schema fields — unused (HasMidpoint=false).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 2 rows (stages 2..3), same as RKCoeffs<IVPAlg::BS3> rows 0..1
    static constexpr std::array<std::array<double, 2>, 2> A = {std::array<double, 2>{rat(1, 2), 0},
                                                               std::array<double, 2>{0, rat(3, 4)}};

    static constexpr std::array<double, 2> C = {rat(1, 2), rat(3, 4)};

    // B = BS3's b-vector (without the FSAL trailing zero):
    static constexpr std::array<double, 3> B = {rat(2, 9), rat(1, 3), rat(4, 9)};
    static constexpr std::array<double, 3> Bhat = {0, 0, 0}; // unused
};

// Bogacki-Shampine 5(4) — 8-stage non-FSAL method with 3-stage lazy
// interpolant. The last main stage (k_8) is evaluated at (t+h, xf) with
// a_{8,*} equal to the b-vector, so k_8 = f(xf) (LastStageIsFxf=true) even
// though b_8 = 0 (not strict-FSAL). The adaptive step uses dofsal=true to
// reuse k_8 as k_1 of the next step — mirrors Julia's integrator.fsallast.
//
// Source: OrdinaryDiffEqLowOrderRK BS5ConstantCache (generic T::Type variant
// with rational coefficients). Interpolation extras from BS5Interp, poly
// weights from BS5Interp_polyweights (CompiledFloats variant via Julia
// compute_bmid_bs5.jl).
template <> struct RKCoeffs<IVPAlg::BS5> {
    static constexpr int Stages = 8;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 4;
    static constexpr bool FSAL = false; // B_8 = 0; not strict-FSAL
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // Dense-output schema fields. BS5's perform_step evaluates k_8 at
    // (t+h, xf) so k_8 = f(xf); 3 additional stages (k_9, k_10, k_11) needed
    // for the BS5Interp polynomial's dense output.
    static constexpr int InterpStages = 3;
    static constexpr bool LastStageIsFxf = true;
    static constexpr int BmidStages = Stages + InterpStages; // 11

    // A: 7 rows (stages 2..8). Row 6 equals BS5's b-vector (with a82 = 0).
    static constexpr std::array<std::array<double, 7>, 7> A = {
        std::array<double, 7>{rat(1, 6), 0, 0, 0, 0, 0, 0},
        std::array<double, 7>{rat(2, 27), rat(4, 27), 0, 0, 0, 0, 0},
        std::array<double, 7>{rat(183, 1372), rat(-162, 343), rat(1053, 1372), 0, 0, 0, 0},
        std::array<double, 7>{rat(68, 297), rat(-4, 11), rat(42, 143), rat(1960, 3861), 0, 0, 0},
        std::array<double, 7>{rat(597, 22528), rat(81, 352), rat(63099, 585728), rat(58653, 366080),
                              rat(4617, 20480), 0, 0},
        std::array<double, 7>{rat(174197, 959244), rat(-30942, 79937), rat(8152137, 19744439),
                              rat(666106, 1039181), rat(-29421, 29068), rat(482048, 414219), 0},
        std::array<double, 7>{rat(587, 8064), 0, rat(4440339, 15491840), rat(24353, 124800),
                              rat(387, 44800), rat(2152, 5985), rat(7267, 94080)}};

    static constexpr std::array<double, 7> C = {rat(1, 6), rat(2, 9), rat(3, 7), rat(2, 3),
                                                rat(3, 4), 1.0,       1.0};

    // B: 8 elements. B_1..B_7 = row A[6]; B_8 = 0 (k_8 is computed for FSAL
    // reuse and dense output but not used in the u update).
    static constexpr std::array<double, 8> B = {rat(587, 8064),         0.0,
                                                rat(4440339, 15491840), rat(24353, 124800),
                                                rat(387, 44800),        rat(2152, 5985),
                                                rat(7267, 94080),       0.0};

    // Bhat = B - btilde. btilde values from BS5ConstantCache (utilde main
    // embedded-error weights); btilde_2 = 0 implicit.
    static constexpr std::array<double, 8> Bhat = {rat(587, 8064) - rat(-3817, 1959552),
                                                   0.0 - 0.0,
                                                   rat(4440339, 15491840) - rat(140181, 15491840),
                                                   rat(24353, 124800) - rat(-4224731, 272937600),
                                                   rat(387, 44800) - rat(8557, 403200),
                                                   rat(2152, 5985) - rat(-57928, 4363065),
                                                   rat(7267, 94080) - rat(-23930231, 4366535040),
                                                   0.0 - rat(3293, 556956)};

    // Extra interpolation stages (k_9, k_10, k_11). ExtraA rows are
    // (Stages + InterpStages)=11 wide; unused entries zero.
    // From BS5Interp (rational T::Type variant) in low_order_rk_tableaus.jl.
    static constexpr std::array<std::array<double, 11>, 3> ExtraA = {
        // k_9 uses k_1..k_8:
        std::array<double, 11>{rat(455, 6144), 0, rat(10256301, 35409920), rat(2307361, 17971200),
                               rat(-387, 102400), rat(73, 5130), rat(-7267, 215040), rat(1, 32), 0,
                               0, 0},
        // k_10 uses k_1..k_9:
        std::array<double, 11>{
            rat(-837888343715LL, 13176988637184LL), rat(30409415, 52955362),
            rat(-48321525963LL, 759168069632LL), rat(8530738453321LL, 197654829557760LL),
            rat(1361640523001LL, 1626788720640LL), rat(-13143060689LL, 38604458898LL),
            rat(18700221969LL, 379584034816LL), rat(-5831595, 847285792), rat(-5183640, 26477681),
            0, 0},
        // k_11 uses k_1..k_10:
        std::array<double, 11>{rat(98719073263LL, 1551965184000LL), rat(1307, 123552),
                               rat(4632066559387LL, 70181753241600LL),
                               rat(7828594302389LL, 382182512025600LL), rat(40763687, 11070259200),
                               rat(34872732407LL, 224610586200LL), rat(-2561897, 30105600),
                               rat(1, 10), rat(-1, 10), rat(-1403317093LL, 11371610250LL), 0}};

    static constexpr std::array<double, 3> ExtraC = {rat(1, 2), rat(5, 6), rat(1, 9)};

    // Bmid = 2·b_i(Θ=0.5) where b_i(Θ) is the BS5Interp polynomial
    // (CompiledFloats variant via bench/julia_reference/src/compute_bmid_bs5.jl).
    // Layout: [0..7] main stages k_1..k_8; [8..10] extra stages k_9..k_11.
    // Bmid[1] = 0 (BS5's b_2 contribution is zero).
    // Sanity: Σ Bmid[i] ≈ 1.0 (FP-rounded).
    static constexpr std::array<double, BmidStages> Bmid = {
        3.42702600730507845e-02, 0.00000000000000000e+00,  1.71102321961401849e-01,
        1.10413935309037481e-01, -5.24794697971777552e-03, -4.21488869254374676e-01,
        1.59313217474489510e-01, -1.93847656250000000e-01, 4.27083333333333148e-01,
        3.12151404332777549e-01, 4.06250000000000000e-01};
};

template <> struct RKCoeffs<IVPAlg::BS5Trans> {
    static constexpr int Stages = 7;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // transcription-only, no dense output

    // Dense-output schema fields — unused (HasMidpoint=false).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 6 rows (stages 2..7), same as RKCoeffs<IVPAlg::BS5> rows 0..5.
    static constexpr std::array<std::array<double, 6>, 6> A = {
        std::array<double, 6>{rat(1, 6), 0, 0, 0, 0, 0},
        std::array<double, 6>{rat(2, 27), rat(4, 27), 0, 0, 0, 0},
        std::array<double, 6>{rat(183, 1372), rat(-162, 343), rat(1053, 1372), 0, 0, 0},
        std::array<double, 6>{rat(68, 297), rat(-4, 11), rat(42, 143), rat(1960, 3861), 0, 0},
        std::array<double, 6>{rat(597, 22528), rat(81, 352), rat(63099, 585728), rat(58653, 366080),
                              rat(4617, 20480), 0},
        std::array<double, 6>{rat(174197, 959244), rat(-30942, 79937), rat(8152137, 19744439),
                              rat(666106, 1039181), rat(-29421, 29068), rat(482048, 414219)}};

    static constexpr std::array<double, 6> C = {rat(1, 6), rat(2, 9), rat(3, 7),
                                                rat(2, 3), rat(3, 4), 1.0};

    // B = BS5's a_{8,*} row (the implicit b-vector when B_8 = 0):
    static constexpr std::array<double, 7> B = {rat(587, 8064),         0.0,
                                                rat(4440339, 15491840), rat(24353, 124800),
                                                rat(387, 44800),        rat(2152, 5985),
                                                rat(7267, 94080)};
    static constexpr std::array<double, 7> Bhat = {0, 0, 0, 0, 0, 0, 0}; // unused
};

// Verner 7(6) — 10-stage non-FSAL method with 6-stage lazy interpolant.
// The last main stage (k_10) is evaluated at t+h but with a helper a-row that
// does NOT equal the b-vector, so k_10 ≠ f(xf). The midpoint branch therefore
// evaluates f(xf) explicitly as k_vals_extra[0] (LastStageIsFxf=false).
// However, Julia's Vern7 interpolation polynomial does not reference f(xf);
// its Bmid weight for the f(xf) slot is zero.
//
// Source: OrdinaryDiffEqVerner Vern7Tableau / Vern7ExtraStages /
// Vern7InterpolationCoefficients (CompiledFloats Float64 literal variant,
// because the generic T::Type variant uses BigInt rationals).
// dofsal=false (matches Julia behavior — no FSAL reuse; k_1 computed fresh).
template <> struct RKCoeffs<IVPAlg::Vern7> {
    static constexpr int Stages = 10;
    static constexpr int Order = 7;
    static constexpr int ErrorOrder = 6;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // Dense-output schema fields. The 6 extra stages (k_11..k_16) form the
    // Vern7Interp polynomial. f(xf) is evaluated separately in the midpoint
    // branch to correctly update xdot_prev (required by stepper_compute_impl).
    static constexpr int InterpStages = 6;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + InterpStages + 1; // 17

    // A: 9 rows (stages 2..10). Row 8 is the Vern7 helper g10 which uses only
    // k_1, k_3..k_7 (a10_2 = a10_8 = a10_9 = 0).
    static constexpr std::array<std::array<double, 9>, 9> A = {
        std::array<double, 9>{0.005, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 9>{-1.07679012345679, 1.185679012345679, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 9>{0.04083333333333333, 0, 0.1225, 0, 0, 0, 0, 0, 0},
        std::array<double, 9>{0.6389139236255726, 0, -2.455672638223657, 2.272258714598084, 0, 0, 0,
                              0, 0},
        std::array<double, 9>{-2.6615773750187572, 0, 10.804513886456137, -8.3539146573962,
                              0.820487594956657, 0, 0, 0, 0},
        std::array<double, 9>{6.067741434696772, 0, -24.711273635911088, 20.427517930788895,
                              -1.9061579788166472, 1.006172249242068, 0, 0, 0},
        std::array<double, 9>{12.054670076253203, 0, -49.75478495046899, 41.142888638604674,
                              -4.461760149974004, 2.042334822239175, -0.09834843665406107, 0, 0},
        std::array<double, 9>{10.138146522881808, 0, -42.6411360317175, 35.76384003992257,
                              -4.3480228403929075, 2.0098622683770357, 0.3487490460338272,
                              -0.27143900510483127, 0},
        std::array<double, 9>{-45.030072034298676, 0, 187.3272437654589, -154.02882369350186,
                              18.56465306347536, -7.141809679295079, 1.3088085781613787, 0, 0}};

    static constexpr std::array<double, 9> C = {0.005,
                                                0.10888888888888888,
                                                0.16333333333333333,
                                                0.4555,
                                                0.6095094489978381,
                                                0.884,
                                                0.925,
                                                1.0,
                                                1.0};

    // B: 10 elements. B_1 = b1; B_4..B_9 = b4..b9; B_2 = B_3 = B_10 = 0.
    static constexpr std::array<double, 10> B = {0.04715561848627222,
                                                 0,
                                                 0,
                                                 0.25750564298434153,
                                                 0.26216653977412624,
                                                 0.15216092656738558,
                                                 0.4939969170032485,
                                                 -0.29430311714032503,
                                                 0.08131747232495111,
                                                 0};

    // Bhat = B - btilde. Julia gives btilde for stages {1, 4..10}; btilde_{2,3} = 0.
    static constexpr std::array<double, 10> Bhat = {0.04715561848627222 - 0.002547011879931045,
                                                    0,
                                                    0,
                                                    0.25750564298434153 - (-0.00965839487279575),
                                                    0.26216653977412624 - 0.04206470975639691,
                                                    0.15216092656738558 - (-0.0666822437469301),
                                                    0.4939969170032485 - 0.2650097464621281,
                                                    -0.29430311714032503 - (-0.29430311714032503),
                                                    0.08131747232495111 - 0.08131747232495111,
                                                    0 - (-0.02029518466335628)};

    // Extra stage tableau — 6 rows (k_11..k_16) × (Stages + InterpStages) = 16 cols.
    // Columns 0..9 reference main k_1..k_10 (col 1, 2, 9 always 0 — Vern7's
    // extras skip k_2, k_3, k_10). Columns 10..15 reference prior extras
    // k_11..k_16.
    static constexpr std::array<std::array<double, 16>, 6> ExtraA = {
        // k_11:
        std::array<double, 16>{0.04715561848627222, 0, 0, 0.25750564298434153, 0.2621665397741262,
                               0.15216092656738558, 0.49399691700324844, -0.29430311714032503,
                               0.0813174723249511, 0, 0, 0, 0, 0, 0, 0},
        // k_12:
        std::array<double, 16>{0.0523222769159969, 0, 0, 0.22495861826705715, 0.017443709248776376,
                               -0.007669379876829393, 0.03435896044073285, -0.0410209723009395,
                               0.025651133005205617, 0, -0.0160443457, 0, 0, 0, 0, 0},
        // k_13:
        std::array<double, 16>{0.053053341257859085, 0, 0, 0.12195301011401886,
                               0.017746840737602496, -0.0005928372667681495, 0.008381833970853752,
                               -0.01293369259698612, 0.009412056815253861, 0, -0.005353253107275676,
                               -0.06666729992455811, 0, 0, 0, 0},
        // k_14:
        std::array<double, 16>{0.03887903257436304, 0, 0, -0.0024403203308301317,
                               -0.0013928917214672623, -0.00047446291558680135,
                               0.00039207932413159514, -0.00040554733285128004,
                               0.00019897093147716726, 0, -0.00010278198793179169,
                               0.03385661513870267, 0.1814893063199928, 0, 0, 0},
        // k_15:
        std::array<double, 16>{0.05723681204690013, 0, 0, 0.22265948066761182, 0.12344864200186899,
                               0.04006332526666491, -0.05269894848581452, 0.04765971214244523,
                               -0.02138895885042213, 0, 0.015193891064036402, 0.12060546716289655,
                               -0.022779423016187374, 0, 0, 0},
        // k_16:
        std::array<double, 16>{0.051372038802756814, 0, 0, 0.5414214473439406, 0.350399806692184,
                               0.14193112269692182, 0.10527377478429423, -0.031081847805874016,
                               -0.007401883149519145, 0, -0.006377932504865363,
                               -0.17325495908361865, -0.18228156777622026, 0, 0, 0}};

    static constexpr std::array<double, 6> ExtraC = {1.0, 0.29, 0.125, 0.25, 0.53, 0.79};

    // Bmid = 2·b_i(Θ=0.5) via Vern7Interp polynomial (CompiledFloats). Layout:
    // [0..9] main k_1..k_10 (indices 1, 2, 9 = 0); [10] f(xf) slot (=0; not in
    // poly); [11..16] extras k_11..k_16.
    // Derived via bench/julia_reference/src/compute_bmid_vern7.jl.
    static constexpr std::array<double, BmidStages> Bmid = {1.01568172253994393e-01,
                                                            0,
                                                            0,
                                                            2.34071209963178983e-01,
                                                            2.38307939451734452e-01,
                                                            1.38313443456900220e-01,
                                                            4.49040474379289734e-01,
                                                            -2.67519911123506482e-01,
                                                            7.39171340777436958e-02,
                                                            0,
                                                            0,
                                                            -5.63903701328063445e-02,
                                                            -4.32115132904069832e-02,
                                                            9.59844798466543692e-02,
                                                            3.22065326928104412e-01,
                                                            7.82416025520005221e-03,
                                                            -2.93970546066088634e-01};
};

template <> struct RKCoeffs<IVPAlg::Vern7Trans> {
    static constexpr int Stages = 9;
    static constexpr int Order = 7;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false;

    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 8 rows (stages 2..9), same as RKCoeffs<IVPAlg::Vern7> rows 0..7
    static constexpr std::array<std::array<double, 8>, 8> A = {
        std::array<double, 8>{0.005, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 8>{-1.07679012345679, 1.185679012345679, 0, 0, 0, 0, 0, 0},
        std::array<double, 8>{0.04083333333333333, 0, 0.1225, 0, 0, 0, 0, 0},
        std::array<double, 8>{0.6389139236255726, 0, -2.455672638223657, 2.272258714598084, 0, 0, 0,
                              0},
        std::array<double, 8>{-2.6615773750187572, 0, 10.804513886456137, -8.3539146573962,
                              0.820487594956657, 0, 0, 0},
        std::array<double, 8>{6.067741434696772, 0, -24.711273635911088, 20.427517930788895,
                              -1.9061579788166472, 1.006172249242068, 0, 0},
        std::array<double, 8>{12.054670076253203, 0, -49.75478495046899, 41.142888638604674,
                              -4.461760149974004, 2.042334822239175, -0.09834843665406107, 0},
        std::array<double, 8>{10.138146522881808, 0, -42.6411360317175, 35.76384003992257,
                              -4.3480228403929075, 2.0098622683770357, 0.3487490460338272,
                              -0.27143900510483127}};

    static constexpr std::array<double, 8> C = {
        0.005, 0.10888888888888888, 0.16333333333333333, 0.4555, 0.6095094489978381, 0.884, 0.925,
        1.0};

    // B = Vern7's b-vector truncated to 9 elements (drops trailing zero).
    static constexpr std::array<double, 9> B = {0.04715561848627222,
                                                0,
                                                0,
                                                0.25750564298434153,
                                                0.26216653977412624,
                                                0.15216092656738558,
                                                0.4939969170032485,
                                                -0.29430311714032503,
                                                0.08131747232495111};
    static constexpr std::array<double, 9> Bhat = {0, 0, 0, 0, 0, 0, 0, 0, 0}; // unused
};

// Verner 8(7) — 13-stage non-FSAL method with 8-stage lazy interpolant.
// Similar to Vern7: k_12 and k_13 are helper stages evaluated at t+h but NOT
// equal to f(xf) (their a-rows differ from the b-vector). LastStageIsFxf=false.
// dofsal=false (no FSAL reuse). Vern8Interp polynomial does not reference
// f(xf); the Bmid[Stages] slot is 0.
//
// Source: OrdinaryDiffEqVerner Vern8Tableau / Vern8ExtraStages /
// Vern8InterpolationCoefficients (CompiledFloats Float64 literal variant).
template <> struct RKCoeffs<IVPAlg::Vern8> {
    static constexpr int Stages = 13;
    static constexpr int Order = 8;
    static constexpr int ErrorOrder = 7;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    static constexpr int InterpStages = 8;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + InterpStages + 1; // 22

    // A: 12 rows (stages 2..13). Row 11 is the g13 helper that uses only
    // k_1, k_4..k_10 (k_11, k_12 excluded).
    static constexpr std::array<std::array<double, 12>, 12> A = {
        std::array<double, 12>{0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{-0.0069931640625, 0.1135556640625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{0.0399609375, 0, 0.1198828125, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{0.36139756280045754, 0, -1.3415240667004928, 1.3701265039000352, 0,
                               0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{0.049047202797202795, 0, 0, 0.23509720422144048, 0.18085559298135673,
                               0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{0.06169289044289044, 0, 0, 0.11236568314640277, -0.03885046071451367,
                               0.01979188712522046, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{-1.767630240222327, 0, 0, -62.5, -6.061889377376669,
                               5.6508231982227635, 65.62169641937624, 0, 0, 0, 0, 0},
        std::array<double, 12>{-1.1809450665549708, 0, 0, -41.50473441114321, -4.434438319103725,
                               4.260408188586133, 43.75364022446172, 0.00787142548991231, 0, 0, 0,
                               0},
        std::array<double, 12>{-1.2814059994414884, 0, 0, -45.047139960139866, -4.731362069449576,
                               4.514967016593808, 47.44909557172985, 0.01059228297111661,
                               -0.0057468422638446166, 0, 0, 0},
        std::array<double, 12>{-1.7244701342624853, 0, 0, -60.92349008483054, -5.951518376222392,
                               5.556523730698456, 63.98301198033305, 0.014642028250414961,
                               0.06460408772358203, -0.0793032316900888, 0, 0},
        std::array<double, 12>{-3.301622667747079, 0, 0, -118.01127235975251, -10.141422388456112,
                               9.139311332232058, 123.37594282840426, 4.62324437887458,
                               -3.3832777380682018, 4.527592100324618, -5.828495485811623, 0},
        std::array<double, 12>{-3.039515033766309, 0, 0, -109.26086808941763, -9.290642497400293,
                               8.43050498176491, 114.20100103783314, -0.9637271342145479,
                               -5.0348840888021895, 5.958130824002923, 0, 0}};

    static constexpr std::array<double, 12> C = {0.05,  0.1065625, 0.15984375, 0.39,
                                                 0.465, 0.155,     0.943,      0.901802041735857,
                                                 0.909, 0.94,      1.0,        1.0};

    // B: 13 elements. Only b1, b6..b12 nonzero; b2..b5 = b13 = 0.
    static constexpr std::array<double, 13> B = {0.04427989419007951,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0.3541049391724449,
                                                 0.24796921549564377,
                                                 -15.694202038838085,
                                                 25.084064965558564,
                                                 -31.738367786260277,
                                                 22.938283273988784,
                                                 -0.2361324633071542,
                                                 0};

    // Bhat = B - btilde. btilde{2..5} = 0; btilde{13} specified.
    static constexpr std::array<double, 13> Bhat = {0.04427989419007951 - (-3.272103901028138e-5),
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0.3541049391724449 - (-0.0005046250618777704),
                                                    0.24796921549564377 - 0.0001211723589784759,
                                                    -15.694202038838085 - (-20.142336771313868),
                                                    25.084064965558564 - 5.2371785994398286,
                                                    -31.738367786260277 - (-8.156744408794658),
                                                    22.938283273988784 - 22.938283273988784,
                                                    -0.2361324633071542 - (-0.2361324633071542),
                                                    0 - 0.36016794372897754};

    // ExtraA: 8 rows (k_14..k_21) × (13 + 8) = 21 cols. Cols 0..12 reference
    // main k_1..k_13 (most are zero — Julia's extras use k_1, k_6..k_12 only,
    // never k_13). Cols 13..20 reference prior extras k_14..k_21.
    static constexpr std::array<std::array<double, 21>, 8> ExtraA = {
        // k_14 (uses k_1, k_6..k_12):
        std::array<double, 21>{0.04427989419007951,
                               0,
                               0,
                               0,
                               0,
                               0.3541049391724449,
                               0.2479692154956438,
                               -15.694202038838084,
                               25.084064965558564,
                               -31.738367786260277,
                               22.938283273988784,
                               -0.2361324633071542,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_15:
        std::array<double, 21>{0.04620700646754963,
                               0,
                               0,
                               0,
                               0,
                               0.045039041608424805,
                               0.23368166977134244,
                               37.83901368421068,
                               -15.949113289454246,
                               23.028368351816102,
                               -44.85578507769412,
                               -0.06379858768647444,
                               0,
                               -0.012595035543861663,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_16:
        std::array<double, 21>{0.05037946855482041,
                               0,
                               0,
                               0,
                               0,
                               0.041098361310460796,
                               0.17180541533481958,
                               4.614105319981519,
                               -1.7916678830853965,
                               2.531658930485041,
                               -5.324977860205731,
                               -0.03065532595385635,
                               0,
                               -0.005254479979429613,
                               -0.08399194644224793,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_17:
        std::array<double, 21>{0.0408289713299708,
                               0,
                               0,
                               0,
                               0,
                               0.4244479514247632,
                               0.23260915312752345,
                               2.677982520711806,
                               0.7420826657338945,
                               0.1460377847941461,
                               -3.579344509890565,
                               0.11388443896001738,
                               0,
                               0.012677906510331901,
                               -0.07443436349946675,
                               0.047827480797578516,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_18:
        std::array<double, 21>{0.052126823936684136,
                               0,
                               0,
                               0,
                               0,
                               0.053925083967447975,
                               0.01660758097434641,
                               -4.45448575792678,
                               6.835218278632146,
                               -8.711334822181994,
                               6.491635839232917,
                               -0.07072551809844346,
                               0,
                               -0.018540314919932164,
                               0.023504021054353848,
                               0.2344795103407822,
                               -0.08241072501152899,
                               0,
                               0,
                               0,
                               0},
        // k_19:
        std::array<double, 21>{0.05020102870355714,
                               0,
                               0,
                               0,
                               0,
                               0.1552209034795498,
                               0.1264268424089235,
                               -5.149206303539847,
                               8.46834099903693,
                               -10.662130681081495,
                               7.541833224959729,
                               -0.07436968113832143,
                               0,
                               -0.020558876866183826,
                               0.07753795264710298,
                               0.10462592203525443,
                               -0.11792133064519794,
                               0,
                               0,
                               0,
                               0},
        // k_20:
        std::array<double, 21>{0.03737341446457826,
                               0,
                               0,
                               0,
                               0,
                               0.35049307053383166,
                               0.49226528193730257,
                               8.553695439359313,
                               -10.353172990305913,
                               13.83320427252915,
                               -12.280924330784618,
                               0.17191515956565098,
                               0,
                               0.036415831143144964,
                               0.02961920580288763,
                               -0.2651793938627067,
                               0.09429503961738067,
                               0,
                               0,
                               0,
                               0},
        // k_21:
        std::array<double, 21>{0.039390583455282506,
                               0,
                               0,
                               0,
                               0,
                               0.3558516141234424,
                               0.419738222595261,
                               0.8720449778071941,
                               0.8989520834876595,
                               -0.6305806161059884,
                               -1.1218872205954835,
                               0.04298219512400197,
                               0,
                               0.013325575668739157,
                               0.018762270539641482,
                               -0.18594111329221055,
                               0.17736142719246029,
                               0,
                               0,
                               0,
                               0}};

    static constexpr std::array<double, 8> ExtraC = {
        1.0, 0.3110177634953864, 0.1725, 0.7846, 0.37, 0.5, 0.7, 0.9};

    // Bmid via bench/julia_reference/src/compute_bmid_vern8.jl at Θ=0.5.
    static constexpr std::array<double, BmidStages> Bmid = {8.84199315183124002e-02,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            7.32246901575567932e-01,
                                                            5.12770847413673181e-01,
                                                            -3.24537433521794583e+01,
                                                            5.18708631765496193e+01,
                                                            -6.56311700335731985e+01,
                                                            4.74336418297316413e+01,
                                                            -4.88293851597161321e-01,
                                                            0,
                                                            0,
                                                            -5.92952958438524025e-02,
                                                            2.98097661122986857e-03,
                                                            -2.13025559023485300e-02,
                                                            -7.92148276237192306e-03,
                                                            2.32062806306903013e-02,
                                                            -2.16300646981446576e-01,
                                                            -4.56869806201808615e-01,
                                                            -3.29232918972026667e-01};
};

template <> struct RKCoeffs<IVPAlg::Vern8Trans> {
    static constexpr int Stages = 12;
    static constexpr int Order = 8;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false;

    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 11 rows (stages 2..12), same as RKCoeffs<IVPAlg::Vern8> rows 0..10
    static constexpr std::array<std::array<double, 11>, 11> A = {
        std::array<double, 11>{0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 11>{-0.0069931640625, 0.1135556640625, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 11>{0.0399609375, 0, 0.1198828125, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 11>{0.36139756280045754, 0, -1.3415240667004928, 1.3701265039000352, 0,
                               0, 0, 0, 0, 0, 0},
        std::array<double, 11>{0.049047202797202795, 0, 0, 0.23509720422144048, 0.18085559298135673,
                               0, 0, 0, 0, 0, 0},
        std::array<double, 11>{0.06169289044289044, 0, 0, 0.11236568314640277, -0.03885046071451367,
                               0.01979188712522046, 0, 0, 0, 0, 0},
        std::array<double, 11>{-1.767630240222327, 0, 0, -62.5, -6.061889377376669,
                               5.6508231982227635, 65.62169641937624, 0, 0, 0, 0},
        std::array<double, 11>{-1.1809450665549708, 0, 0, -41.50473441114321, -4.434438319103725,
                               4.260408188586133, 43.75364022446172, 0.00787142548991231, 0, 0, 0},
        std::array<double, 11>{-1.2814059994414884, 0, 0, -45.047139960139866, -4.731362069449576,
                               4.514967016593808, 47.44909557172985, 0.01059228297111661,
                               -0.0057468422638446166, 0, 0},
        std::array<double, 11>{-1.7244701342624853, 0, 0, -60.92349008483054, -5.951518376222392,
                               5.556523730698456, 63.98301198033305, 0.014642028250414961,
                               0.06460408772358203, -0.0793032316900888, 0},
        std::array<double, 11>{-3.301622667747079, 0, 0, -118.01127235975251, -10.141422388456112,
                               9.139311332232058, 123.37594282840426, 4.62324437887458,
                               -3.3832777380682018, 4.527592100324618, -5.828495485811623}};

    static constexpr std::array<double, 11> C = {0.05,  0.1065625, 0.15984375, 0.39,
                                                 0.465, 0.155,     0.943,      0.901802041735857,
                                                 0.909, 0.94,      1.0};

    static constexpr std::array<double, 12> B = {0.04427989419007951,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0.3541049391724449,
                                                 0.24796921549564377,
                                                 -15.694202038838085,
                                                 25.084064965558564,
                                                 -31.738367786260277,
                                                 22.938283273988784,
                                                 -0.2361324633071542};
    static constexpr std::array<double, 12> Bhat = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

// Verner 9(8) — 16-stage non-FSAL method with 10-stage lazy interpolant.
// Like Vern7/Vern8, k_15 and k_16 are helper stages evaluated at t+h but NOT
// equal to f(xf). LastStageIsFxf=false; dofsal=false.
//
// Julia renumbers the interpolation k-array: k[1..10] = (k_1, k_8..k_16) so
// stages k_2..k_7 are discarded from the interp. k_16 is additionally not
// referenced by the interp polynomial itself (used only in utilde error
// estimator). In Tycho's sequential layout we still compute all 16 main
// stages; the corresponding Bmid weights for k_2..k_7 and k_16 are zero.
//
// Source: OrdinaryDiffEqVerner Vern9Tableau / Vern9ExtraStages /
// Vern9InterpolationCoefficients (CompiledFloats Float64 literal variant).
template <> struct RKCoeffs<IVPAlg::Vern9> {
    static constexpr int Stages = 16;
    static constexpr int Order = 9;
    static constexpr int ErrorOrder = 8;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    static constexpr int InterpStages = 10;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + InterpStages + 1; // 27

    // A: 15 rows (stages 2..16). Row 14 (stage 16) uses k_1, k_6..k_13 only
    // (not k_14 — a_{16,14} = 0).
    static constexpr std::array<std::array<double, 15>, 15> A = {
        std::array<double, 15>{0.03462, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 15>{-0.03893354388572875, 0.13595789452450918, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0},
        std::array<double, 15>{0.03638413148954267, 0, 0.10915239446862801, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0},
        std::array<double, 15>{2.0257639143939694, 0, -7.638023836496291, 6.173259922102322, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 15>{0.05112275589406061, 0, 0, 0.17708237945550218,
                               0.0008027762409222536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 15>{0.13160063579752163, 0, 0, -0.2957276252669636, 0.08781378035642955,
                               0.6213052975225274, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 15>{0.07166666666666667, 0, 0, 0, 0, 0.33055335789153195,
                               0.2427799754418014, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 15>{0.071806640625, 0, 0, 0, 0, 0.3294380283228177, 0.1165190029271823,
                               -0.034013671875, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 15>{0.04836757646340646, 0, 0, 0, 0, 0.03928989925676164,
                               0.10547409458903446, -0.021438652846483126, -0.10412291746271944, 0,
                               0, 0, 0, 0, 0},
        std::array<double, 15>{-0.026645614872014785, 0, 0, 0, 0, 0.03333333333333333,
                               -0.1631072244872467, 0.03396081684127761, 0.1572319413814626,
                               0.21522674780318796, 0, 0, 0, 0, 0},
        std::array<double, 15>{0.03689009248708622, 0, 0, 0, 0, -0.1465181576725543,
                               0.2242577768172024, 0.02294405717066073, -0.0035850052905728597,
                               0.08669223316444385, 0.43838406519683376, 0, 0, 0, 0},
        std::array<double, 15>{-0.4866012215113341, 0, 0, 0, 0, -6.304602650282853,
                               -0.2812456182894729, -2.679019236219849, 0.5188156639241577,
                               1.3653531876033418, 5.8850910885039465, 2.8028087862720628, 0, 0, 0},
        std::array<double, 15>{0.4185367457753472, 0, 0, 0, 0, 6.724547581906459,
                               -0.42544428016461133, 3.3432791530012653, 0.6170816631175374,
                               -0.9299661239399329, -6.099948804751011, -3.002206187889399,
                               0.2553202529443446, 0, 0},
        std::array<double, 15>{-0.7793740861228848, 0, 0, 0, 0, -13.937342538107776,
                               1.2520488533793563, -14.691500408016868, -0.494705058533141,
                               2.2429749091462368, 13.367893803828643, 14.396650486650687,
                               -0.79758133317768, 0.4409353709534278, 0},
        std::array<double, 15>{2.0580513374668867, 0, 0, 0, 0, 22.357937727968032,
                               0.9094981099755646, 35.89110098240264, -3.442515027624454,
                               -4.865481358036369, -18.909803813543427, -34.26354448030452,
                               1.2647565216956427, 0, 0}};

    static constexpr std::array<double, 15> C = {0.03462,
                                                 0.09702435063878045,
                                                 0.14553652595817068,
                                                 0.561,
                                                 0.22900791159048503,
                                                 0.544992088409515,
                                                 0.645,
                                                 0.48375,
                                                 0.06757,
                                                 0.25,
                                                 0.6590650618730999,
                                                 0.8206,
                                                 0.9012,
                                                 1.0,
                                                 1.0};

    // B: 16 elements. Only b1, b8..b15 nonzero; b2..b7 = b16 = 0.
    static constexpr std::array<double, 16> B = {0.014611976858423152,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 -0.3915211862331339,
                                                 0.23109325002895065,
                                                 0.12747667699928525,
                                                 0.2246434176204158,
                                                 0.5684352689748513,
                                                 0.058258715572158275,
                                                 0.13643174034822156,
                                                 0.030570139830827976,
                                                 0};

    // Bhat = B - btilde. btilde values from Vern9Tableau: (btilde1, btilde8..btilde16).
    static constexpr std::array<double, 16> Bhat = {0.014611976858423152 - (-0.005357988290444578),
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    -0.3915211862331339 - (-2.583020491182464),
                                                    0.23109325002895065 - 0.14252253154686625,
                                                    0.12747667699928525 - 0.013420653512688676,
                                                    0.2246434176204158 - (-0.02867296291409493),
                                                    0.5684352689748513 - 2.624999655215792,
                                                    0.058258715572158275 - (-0.2825509643291537),
                                                    0.13643174034822156 - 0.13643174034822156,
                                                    0.030570139830827976 - 0.030570139830827976,
                                                    0 - (-0.04834231373823958)};

    // ExtraA: 10 rows (k_17..k_26) × (16 + 10) = 26 cols.
    // Columns 0..15 reference main k_1..k_16; Julia's extras use only k_1,
    // k_8..k_15 (cols 0, 7..14). Cols 1..6 (k_2..k_7) and 15 (k_16) always 0.
    // Columns 16..25 reference prior extras k_17..k_26.
    static constexpr std::array<std::array<double, 26>, 10> ExtraA = {
        // k_17:
        std::array<double, 26>{0.014611976858423152,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               -0.3915211862331339,
                               0.23109325002895065,
                               0.12747667699928525,
                               0.2246434176204158,
                               0.5684352689748513,
                               0.058258715572158275,
                               0.13643174034822156,
                               0.030570139830827976,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_18:
        std::array<double, 26>{0.015499736681895594,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0.3355153219059635,
                               0.20036139441918607,
                               0.12520606592835493,
                               0.22986763931842066,
                               -0.20202506534761813,
                               0.05917103230665457,
                               -0.026518347830476387,
                               -0.023840946021309713,
                               0,
                               0.027181715702085017,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_19:
        std::array<double, 26>{0.013024539431143383,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               -0.7452850902413112,
                               0.2643867896429301,
                               0.1313961758372754,
                               0.21672538151229273,
                               0.8734117564076053,
                               0.011859056439357767,
                               0.05876002941689551,
                               0.003266518630202088,
                               0,
                               -0.00895930864841793,
                               0.06941415157202692,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_20:
        std::array<double, 26>{0.013970899969259426,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               -0.46657653359576745,
                               0.24163727872162571,
                               0.12903633413456747,
                               0.22167006717351054,
                               0.6257275123364645,
                               0.04355312415679284,
                               0.10119624916672908,
                               0.01808582254679721,
                               0,
                               -0.020798755876891697,
                               -0.09022232517086219,
                               -0.12127967356222542,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_21:
        std::array<double, 26>{0.016046388883181127,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0.09517712399458336,
                               0.13591872646553177,
                               0.1237765280959854,
                               0.2335656264102966,
                               -0.09051508172625873,
                               -0.02537574270006131,
                               -0.13596316968871622,
                               -0.04679214284145113,
                               0,
                               0.05177958859391748,
                               0.09672595677476774,
                               0.14773126903407427,
                               -0.11507507129585039,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_22:
        std::array<double, 26>{0.018029186238936207,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0.06983601042028874,
                               -0.025412476607916634,
                               0.008487827035463275,
                               -0.002427525516089802,
                               -0.10478397528938199,
                               -0.014731477952480419,
                               -0.03916338390816177,
                               -0.010056573432939595,
                               0,
                               0.011025103922048344,
                               0.005092830749095398,
                               0.04759715599420645,
                               0.03386307003288383,
                               0.02764422831404798,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_23:
        std::array<double, 26>{0.01677431640522778,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0.6220437408820475,
                               -0.2060859809768842,
                               0.11563949897660589,
                               0.026641017933783588,
                               -0.937681079341877,
                               -0.13678064667021603,
                               -0.3678480995268297,
                               -0.09547871314402478,
                               0,
                               0.10134920184223697,
                               -0.08911323084568594,
                               0.46641409889747604,
                               0.450273629235458,
                               0.18385224633268188,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_24:
        std::array<double, 26>{0.010711497314914442,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               -0.07094336118221108,
                               0.10021649003400916,
                               0.13834539804680251,
                               0.17963306335781634,
                               0.09048246545576182,
                               -0.005460662294523339,
                               -0.030004579051196197,
                               -0.011451920269627991,
                               0,
                               0.010033946861093851,
                               -0.09506485282809046,
                               0.04853358804093592,
                               0.08013325919783924,
                               -0.1251643326835242,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_25:
        std::array<double, 26>{0.014101720888692213,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               -0.3713379753704491,
                               0.22312655481171803,
                               0.12870053459181202,
                               0.22246006596754947,
                               0.5382853042550702,
                               0.05417202616988763,
                               0.1256968791308744,
                               0.027844927890020542,
                               0,
                               -0.0307740924620506,
                               0.008569805293689777,
                               -0.15351746905870445,
                               -0.021799570305481963,
                               0.014471288197371868,
                               0,
                               0,
                               0,
                               0,
                               0},
        // k_26:
        std::array<double, 26>{0.014246004117356466,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               -0.3767107393295407,
                               0.22523997807304214,
                               0.128360307629253,
                               0.22302387052616926,
                               0.5463127827750747,
                               0.0552619079137578,
                               0.12856135087499826,
                               0.028572506812964065,
                               0,
                               -0.02398761886357109,
                               0.055562244589105095,
                               -0.017406756507628386,
                               -0.03815462365996979,
                               0.011118785048989178,
                               0,
                               0,
                               0,
                               0,
                               0}};

    static constexpr std::array<double, 10> ExtraC = {
        1.0, 0.7404185470631561, 0.888, 0.696, 0.487, 0.025, 0.15, 0.32, 0.78, 0.96};

    // Bmid via bench/julia_reference/src/compute_bmid_vern9.jl at Θ=0.5.
    static constexpr std::array<double, BmidStages> Bmid = {-1.91946461894023646e-02,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            -3.08098721573430101e-01,
                                                            1.81853594139268715e-01,
                                                            1.00314881020319380e-01,
                                                            1.76778044745506957e-01,
                                                            4.47317248278085167e-01,
                                                            4.58453754724861273e-02,
                                                            1.07361865108100085e-01,
                                                            2.40564785032854611e-02,
                                                            0,
                                                            0,
                                                            2.02956288866669166e-02,
                                                            6.92004602300377858e-03,
                                                            2.94339642108891741e-02,
                                                            -8.30992078831003234e-02,
                                                            -2.37540876368252940e-03,
                                                            1.33047947204947725e-01,
                                                            1.62188392637059614e-01,
                                                            2.67182162436755188e-01,
                                                            -1.61418207477961984e-01,
                                                            -1.28409436778841979e-01};
};

template <> struct RKCoeffs<IVPAlg::Vern9Trans> {
    static constexpr int Stages = 15;
    static constexpr int Order = 9;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false;

    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 14 rows (stages 2..15), same as RKCoeffs<IVPAlg::Vern9> rows 0..13
    static constexpr std::array<std::array<double, 14>, 14> A = {
        std::array<double, 14>{0.03462, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 14>{-0.03893354388572875, 0.13595789452450918, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0},
        std::array<double, 14>{0.03638413148954267, 0, 0.10915239446862801, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0},
        std::array<double, 14>{2.0257639143939694, 0, -7.638023836496291, 6.173259922102322, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 14>{0.05112275589406061, 0, 0, 0.17708237945550218,
                               0.0008027762409222536, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 14>{0.13160063579752163, 0, 0, -0.2957276252669636, 0.08781378035642955,
                               0.6213052975225274, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 14>{0.07166666666666667, 0, 0, 0, 0, 0.33055335789153195,
                               0.2427799754418014, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 14>{0.071806640625, 0, 0, 0, 0, 0.3294380283228177, 0.1165190029271823,
                               -0.034013671875, 0, 0, 0, 0, 0, 0},
        std::array<double, 14>{0.04836757646340646, 0, 0, 0, 0, 0.03928989925676164,
                               0.10547409458903446, -0.021438652846483126, -0.10412291746271944, 0,
                               0, 0, 0, 0},
        std::array<double, 14>{-0.026645614872014785, 0, 0, 0, 0, 0.03333333333333333,
                               -0.1631072244872467, 0.03396081684127761, 0.1572319413814626,
                               0.21522674780318796, 0, 0, 0, 0},
        std::array<double, 14>{0.03689009248708622, 0, 0, 0, 0, -0.1465181576725543,
                               0.2242577768172024, 0.02294405717066073, -0.0035850052905728597,
                               0.08669223316444385, 0.43838406519683376, 0, 0, 0},
        std::array<double, 14>{-0.4866012215113341, 0, 0, 0, 0, -6.304602650282853,
                               -0.2812456182894729, -2.679019236219849, 0.5188156639241577,
                               1.3653531876033418, 5.8850910885039465, 2.8028087862720628, 0, 0},
        std::array<double, 14>{0.4185367457753472, 0, 0, 0, 0, 6.724547581906459,
                               -0.42544428016461133, 3.3432791530012653, 0.6170816631175374,
                               -0.9299661239399329, -6.099948804751011, -3.002206187889399,
                               0.2553202529443446, 0},
        std::array<double, 14>{-0.7793740861228848, 0, 0, 0, 0, -13.937342538107776,
                               1.2520488533793563, -14.691500408016868, -0.494705058533141,
                               2.2429749091462368, 13.367893803828643, 14.396650486650687,
                               -0.79758133317768, 0.4409353709534278}};

    static constexpr std::array<double, 14> C = {0.03462,
                                                 0.09702435063878045,
                                                 0.14553652595817068,
                                                 0.561,
                                                 0.22900791159048503,
                                                 0.544992088409515,
                                                 0.645,
                                                 0.48375,
                                                 0.06757,
                                                 0.25,
                                                 0.6590650618730999,
                                                 0.8206,
                                                 0.9012,
                                                 1.0};

    static constexpr std::array<double, 15> B = {0.014611976858423152,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 -0.3915211862331339,
                                                 0.23109325002895065,
                                                 0.12747667699928525,
                                                 0.2246434176204158,
                                                 0.5684352689748513,
                                                 0.058258715572158275,
                                                 0.13643174034822156,
                                                 0.030570139830827976};
    static constexpr std::array<double, 15> Bhat = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

template <> struct RKCoeffs<IVPAlg::Tsit5Trans> {
    static constexpr int Stages = 6;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // transcription-only, no dense output

    // Dense-output schema fields — unused (HasMidpoint=false).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 5 rows (stages 2..6), same as RKCoeffs<IVPAlg::Tsit5> rows 0..4
    static constexpr std::array<std::array<double, 5>, 5> A = {
        std::array<double, 5>{0.161, 0, 0, 0, 0},
        std::array<double, 5>{-0.008480655492356989, 0.335480655492357, 0, 0, 0},
        std::array<double, 5>{2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0},
        std::array<double, 5>{5.325864828439257, -11.748883564062828, 7.4955393428898365,
                              -0.09249506636175525, 0},
        std::array<double, 5>{5.86145544294642, -12.92096931784711, 8.159367898576159,
                              -0.071584973281401, -0.028269050394068383}};

    static constexpr std::array<double, 5> C = {0.161, 0.327, 0.9, 0.9800255409045097, 1.0};

    // B = Tsit5's row 7 of A (the b-vector via FSAL structure):
    static constexpr std::array<double, 6> B = {0.09646076681806523, 0.01,
                                                0.4798896504144996,  1.379008574103742,
                                                -3.290069515436081,  2.324710524099774};
    static constexpr std::array<double, 6> Bhat = {0, 0, 0, 0, 0, 0}; // unused
};

// =============================================================================
// Compile-time tableau invariants (Review hardening)
//
// These checks catch authoring typos in any RKCoeffs specialization at
// compile time, without any runtime cost. Each user-selectable method
// must satisfy:
//   1. BmidStages == Stages + InterpStages + (LastStageIsFxf ? 0 : 1)
//      (the documented schema invariant — see header comment at top of
//      file)
//   2. |Σ B[i] - 1| < 1e-13 — main weight vector consistency.
//   3. |Σ Bmid[i] - 1| < 1e-13 — midpoint weights consistency
//      (only checked when HasMidpoint is true).
//
// The constexpr helpers below allow the static_asserts to read directly
// from each RKCoeffs<Alg> specialization. A failed assert names the
// offending method.
// =============================================================================

namespace detail {

constexpr double rk_constexpr_abs(double x) { return x < 0 ? -x : x; }

template <std::size_t N> constexpr double rk_array_sum(const std::array<double, N> &arr) {
    double s = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        s += arr[i];
    return s;
}

// Tolerance is loose enough to absorb the FP drift accumulated by the
// large alternating-sign Vern* dense-output coefficients (Vern8's Bmid
// sums to 1 + ~1.7e-11 due to round-trip through Julia's symbolic
// computation), but tight enough that any human-introduced typo (sign
// flip, missing term, off-by-an-order-of-magnitude) produces deviations
// >> 1e-9 and triggers the assert.
constexpr double kRKWeightTol = 1e-9;

} // namespace detail

#define TYCHO_VALIDATE_RK_TABLEAU(ALG, NAME)                                                       \
    static_assert(RKCoeffs<IVPAlg::ALG>::BmidStages ==                                             \
                      RKCoeffs<IVPAlg::ALG>::Stages + RKCoeffs<IVPAlg::ALG>::InterpStages +        \
                          (RKCoeffs<IVPAlg::ALG>::LastStageIsFxf ? 0 : 1),                         \
                  NAME ": BmidStages must equal Stages + InterpStages + "                          \
                       "(LastStageIsFxf ? 0 : 1).");                                               \
    static_assert(detail::rk_constexpr_abs(detail::rk_array_sum(RKCoeffs<IVPAlg::ALG>::B) - 1.0) < \
                      detail::kRKWeightTol,                                                        \
                  NAME ": main B weights must sum to 1.")

#define TYCHO_VALIDATE_RK_BMID(ALG, NAME)                                                          \
    static_assert(detail::rk_constexpr_abs(detail::rk_array_sum(RKCoeffs<IVPAlg::ALG>::Bmid) -     \
                                           1.0) < detail::kRKWeightTol,                            \
                  NAME ": midpoint Bmid weights must sum to 1.")

TYCHO_VALIDATE_RK_TABLEAU(DOPRI54, "DOPRI54");
TYCHO_VALIDATE_RK_BMID(DOPRI54, "DOPRI54");
TYCHO_VALIDATE_RK_TABLEAU(DOPRI87, "DOPRI87");
TYCHO_VALIDATE_RK_BMID(DOPRI87, "DOPRI87");
TYCHO_VALIDATE_RK_TABLEAU(Tsit5, "Tsit5");
TYCHO_VALIDATE_RK_BMID(Tsit5, "Tsit5");
TYCHO_VALIDATE_RK_TABLEAU(BS3, "BS3");
TYCHO_VALIDATE_RK_BMID(BS3, "BS3");
TYCHO_VALIDATE_RK_TABLEAU(BS5, "BS5");
TYCHO_VALIDATE_RK_BMID(BS5, "BS5");
TYCHO_VALIDATE_RK_TABLEAU(Vern7, "Vern7");
TYCHO_VALIDATE_RK_BMID(Vern7, "Vern7");
TYCHO_VALIDATE_RK_TABLEAU(Vern8, "Vern8");
TYCHO_VALIDATE_RK_BMID(Vern8, "Vern8");
TYCHO_VALIDATE_RK_TABLEAU(Vern9, "Vern9");
TYCHO_VALIDATE_RK_BMID(Vern9, "Vern9");

#undef TYCHO_VALIDATE_RK_TABLEAU
#undef TYCHO_VALIDATE_RK_BMID

} // namespace tycho::integrators
