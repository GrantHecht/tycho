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

// RKCoeffs dense-output schema (SP2):
//
//   Main integration tableau:
//     Stages         — main integration f-evals per step
//     A, B, C        — standard Butcher tableau
//     Bhat           — embedded (error-estimator) weights (zeros if !HasEmbedded)
//     FSAL           — last main stage equals next step's first stage
//     HasMidpoint    — does this method support midpoint dense output?
//
//   Dense-output extras (SP2):
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
//                        Bmid[Stages]                          — f(xf) term (only if !LastStageIsFxf)
//                        Bmid[Stages + offset .. BmidStages-1] — extra stages
//                        (offset = LastStageIsFxf ? 0 : 1)
//
// User-selectable IVP algorithms: DOPRI54, DOPRI87, Tsit5. Only these are
// exposed via the Python binding and the string parser. RK4Classic, DOPRI5,
// and Tsit5Trans remain in the enum because they serve as internal
// template-dispatch tags (DOPRI54's adaptive stepper is instantiated against
// IVPAlg::DOPRI5 coefficients; Tsit5's transcription path uses Tsit5Trans; see
// integrator.h set_method), and rk_steppers.h uses them in constexpr branches.
// They are not runtime-selectable — passing them to set_method throws.
enum class IVPAlg {
    DOPRI54, ///< Dormand-Prince 5(4) — 7 stages, adaptive
    DOPRI87, ///< Dormand-Prince 8(7) — 13 stages, adaptive (default)
    Tsit5,   ///< Tsitouras 5(4) — 7 stages, FSAL, adaptive (SP2)
    BS3,     ///< Bogacki-Shampine 3(2) — 4 stages, FSAL, adaptive (SP2)
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

    // SP2 dense-output schema fields — all zero because HasMidpoint=false.
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    static constexpr std::array<std::array<double, 3>, 3> A = {
        std::array<double, 3>{rat(1, 2), 0.0, 0.0},
        std::array<double, 3>{0.0, rat(1, 2), 0.0},
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

    // SP2 dense-output schema fields (no extra interpolation stages — DOPRI54
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
        std::array<double, 6>{rat(19372, 6561), rat(-25360, 2187), rat(64448, 6561),
                              rat(-212, 729), 0, 0},
        std::array<double, 6>{rat(9017, 3168), rat(-355, 33), rat(46732, 5247), rat(49, 176),
                              rat(-5103, 18656), 0},
        std::array<double, 6>{rat(35, 384), 0.0, rat(500, 1113), rat(125, 192), rat(-2187, 6784),
                              rat(11, 84)}};

    static constexpr std::array<double, 6> C = {rat(1, 5), rat(3, 10), rat(4, 5),
                                                rat(8, 9),  1.0,       1.0};
    static constexpr std::array<double, 7> B = {
        rat(35, 384), 0.0, rat(500, 1113), rat(125, 192), rat(-2187, 6784), rat(11, 84), 0.0};
    static constexpr std::array<double, 7> Bhat = {
        rat(5179, 57600),   0.0,            rat(7571, 16695), rat(393, 640),
        rat(-92097, 339200), rat(187, 2100), rat(1, 40)};

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

    // SP2 dense-output schema fields — unused (HasMidpoint=false).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // 6-stage non-FSAL transcription companion for DOPRI54 — same first 6
    // stages as DOPRI54's A matrix with the FSAL-last row dropped.
    static constexpr std::array<std::array<double, 5>, 5> A = {
        std::array<double, 5>{rat(1, 5), 0, 0, 0, 0},
        std::array<double, 5>{rat(3, 40), rat(9, 40), 0, 0, 0},
        std::array<double, 5>{rat(44, 45), rat(-56, 15), rat(32, 9), 0, 0},
        std::array<double, 5>{rat(19372, 6561), rat(-25360, 2187), rat(64448, 6561),
                              rat(-212, 729), 0},
        std::array<double, 5>{rat(9017, 3168), rat(-355, 33), rat(46732, 5247), rat(49, 176),
                              rat(-5103, 18656)}};

    static constexpr std::array<double, 5> C = {rat(1, 5), rat(3, 10), rat(4, 5), rat(8, 9), 1.0};
    static constexpr std::array<double, 6> B = {
        rat(35, 384), 0.0, rat(500, 1113), rat(125, 192), rat(-2187, 6784), rat(11, 84)};
    static constexpr std::array<double, 6> Bhat = {
        rat(5179, 57600), 0.0, rat(7571, 16695), rat(393, 640), rat(-92097, 339200), rat(187, 2100)};
};

template <> struct RKCoeffs<IVPAlg::DOPRI87> {
    static constexpr int Stages = 13;
    static constexpr int Order = 8;
    static constexpr int ErrorOrder = 7;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // SP2 dense-output schema fields — DOPRI87's last main stage is not f(xf),
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

    static constexpr std::array<double, 12> C = {rat(1, 18),      rat(1, 12),
                                                 rat(1, 8),       rat(5, 16),
                                                 rat(3, 8),       rat(59, 400),
                                                 rat(93, 200),    rat(5490023248, 9719169821),
                                                 rat(13, 20),     rat(1201146811, 1299019798),
                                                 1.0,             1.0};
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

    // SP2 dense-output schema fields (Tsit5's own 4th-order interpolation
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
        std::array<double, 6>{0.09646076681806523, 0.01, 0.4798896504144996,
                              1.379008574103742, -3.290069515436081, 2.324710524099774}};

    static constexpr std::array<double, 6> C = {0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0};

    // B (5th-order weights): 7th row of A (FSAL structure) with b7 = 0.
    static constexpr std::array<double, 7> B = {0.09646076681806523, 0.01, 0.4798896504144996,
                                                1.379008574103742, -3.290069515436081,
                                                2.324710524099774, 0.0};

    // Bhat = B - btilde. btilde from tsit_tableaus.jl:
    // (-0.001780011052225777, -0.0008164344596567469, 0.007880878010261995,
    //  -0.1447110071732629, 0.5823571654525552, -0.45808210592918697, 0.015151515151515152)
    static constexpr std::array<double, 7> Bhat = {
        0.09646076681806523 - (-0.001780011052225777),
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
        2.14824704601937561e-01,  2.27124999999999966e-02,  7.91218061120906202e-01,
        -6.89504287051870612e-01, 2.63237072991632992e+00,  -2.03412170858730335e+00,
        6.25000000000000000e-02};
};

template <> struct RKCoeffs<IVPAlg::BS3> {
    static constexpr int Stages = 4;
    static constexpr int Order = 3;
    static constexpr int ErrorOrder = 2;
    static constexpr bool FSAL = true;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    // SP2 dense-output schema fields (BS3 falls back to Hermite cubic in Julia;
    // no extras needed).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = FSAL;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 3 rows (stages 2..4). Row 3 (FSAL) equals b-vector.
    // Source: OrdinaryDiffEqLowOrderRK BS3ConstantCache (generic T::Type
    // variant with rational coefficients).
    static constexpr std::array<std::array<double, 3>, 3> A = {
        std::array<double, 3>{rat(1, 2), 0, 0},
        std::array<double, 3>{0, rat(3, 4), 0},
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
    static constexpr std::array<double, BmidStages> Bmid = {
        rat(17, 36), rat(1, 3), rat(4, 9), rat(-1, 4)};
};

template <> struct RKCoeffs<IVPAlg::BS3Trans> {
    static constexpr int Stages = 3;
    static constexpr int Order = 3;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // transcription-only, no dense output

    // SP2 dense-output schema fields — unused (HasMidpoint=false).
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = false;
    static constexpr int BmidStages = Stages + (LastStageIsFxf ? 0 : 1);

    // A: 2 rows (stages 2..3), same as RKCoeffs<IVPAlg::BS3> rows 0..1
    static constexpr std::array<std::array<double, 2>, 2> A = {
        std::array<double, 2>{rat(1, 2), 0}, std::array<double, 2>{0, rat(3, 4)}};

    static constexpr std::array<double, 2> C = {rat(1, 2), rat(3, 4)};

    // B = BS3's b-vector (without the FSAL trailing zero):
    static constexpr std::array<double, 3> B = {rat(2, 9), rat(1, 3), rat(4, 9)};
    static constexpr std::array<double, 3> Bhat = {0, 0, 0}; // unused
};

template <> struct RKCoeffs<IVPAlg::Tsit5Trans> {
    static constexpr int Stages = 6;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // transcription-only, no dense output

    // SP2 dense-output schema fields — unused (HasMidpoint=false).
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
    static constexpr std::array<double, 6> B = {0.09646076681806523, 0.01, 0.4798896504144996,
                                                1.379008574103742, -3.290069515436081,
                                                2.324710524099774};
    static constexpr std::array<double, 6> Bhat = {0, 0, 0, 0, 0, 0}; // unused
};

} // namespace tycho::integrators
