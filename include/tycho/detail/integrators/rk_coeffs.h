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

// User-selectable IVP algorithms: DOPRI54, DOPRI87. Only these are exposed via
// the Python binding and the string parser. RK4Classic and DOPRI5 remain in the
// enum because they serve as internal template-dispatch tags (DOPRI54's
// adaptive stepper is instantiated against IVPAlg::DOPRI5 coefficients; see
// integrator.h set_method), and rk_steppers.h uses them in constexpr branches.
// They are not runtime-selectable — passing them to set_method throws.
enum class IVPAlg {
    DOPRI54, ///< Dormand-Prince 5(4) — 7 stages, adaptive
    DOPRI87, ///< Dormand-Prince 8(7) — 13 stages, adaptive (default)
    Tsit5,   ///< Tsitouras 5(4) — 7 stages, FSAL, adaptive (SP2)
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// set_method() throws on this value. Do not expose via bindings.
    RK4Classic,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// set_method() throws on this value. Do not expose via bindings.
    DOPRI5,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    /// 6-stage non-FSAL transcription companion for Tsit5. Do not expose via bindings.
    Tsit5Trans,
};

} // namespace tycho

namespace tycho::integrators {

using tycho::IVPAlg;

template <IVPAlg opt> struct RKCoeffs {};

template <> struct RKCoeffs<IVPAlg::RK4Classic> {
    static constexpr int Stages = 4;
    static constexpr int Order = 4;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // fixed-step, no dense output

    static constexpr std::array<std::array<double, 3>, 3> A = {
        std::array<double, 3>{0.5, 0.0, 0.0}, std::array<double, 3>{0.0, 0.5, 0.0},
        std::array<double, 3>{0.0, 0.0, 1.0}};

    static constexpr std::array<double, 3> C = {0.5, 0.5, 1.0};
    static constexpr std::array<double, 4> B = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    static constexpr std::array<double, 4> Bhat = {0, 0, 0, 0};
};

template <> struct RKCoeffs<IVPAlg::DOPRI54> {
    static constexpr int Stages = 7;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 4;
    static constexpr bool FSAL = true;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;
    static constexpr int BmidSize = FSAL ? Stages : Stages + 1;

    static constexpr std::array<std::array<double, 6>, 6> A = {
        std::array<double, 6>{1 / 5.0, 0, 0, 0, 0, 0},
        std::array<double, 6>{3.0 / 40.0, 9 / 40.0, 0, 0, 0, 0},
        std::array<double, 6>{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0, 0, 0},
        std::array<double, 6>{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
                            0, 0},
        std::array<double, 6>{9017.0 / 3168.0, -355.0 / 33.0, 46732 / 5247.0, 49.0 / 176.0,
                            -5103.0 / 18656.0, 0},
        std::array<double, 6>{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125 / 192.0, -2187.0 / 6784.0,
                            11.0 / 84.0}};

    static constexpr std::array<double, 6> C = {1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0,
                                                8.0 / 9.0, 1.0,        1.0};
    static constexpr std::array<double, 7> B = {
        35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187 / 6784.0, 11.0 / 84.0, 0.0};
    static constexpr std::array<double, 7> Bhat = {
        5179.0 / 57600.0,  0.0,          7571.0 / 16695.0, 393 / 640.0,
        -92097 / 339200.0, 187 / 2100.0, 1.0 / 40.0};

    static constexpr std::array<double, BmidSize> Bmid = {
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

    static constexpr std::array<std::array<double, 5>, 5> A = {
        std::array<double, 5>{1 / 5.0, 0, 0, 0, 0},
        std::array<double, 5>{3.0 / 40.0, 9 / 40.0, 0, 0, 0},
        std::array<double, 5>{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0, 0},
        std::array<double, 5>{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
                            0},
        std::array<double, 5>{9017.0 / 3168.0, -355.0 / 33.0, 46732 / 5247.0, 49.0 / 176.0,
                            -5103.0 / 18656.0}};

    static constexpr std::array<double, 5> C = {1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0};
    static constexpr std::array<double, 6> B = {
        35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187 / 6784.0, 11.0 / 84.0};
    static constexpr std::array<double, 6> Bhat = {
        5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393 / 640.0, -92097 / 339200.0, 187 / 2100.0};
};

template <> struct RKCoeffs<IVPAlg::DOPRI87> {
    static constexpr int Stages = 13;
    static constexpr int Order = 8;
    static constexpr int ErrorOrder = 7;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;
    static constexpr int BmidSize = FSAL ? Stages : Stages + 1;

    static constexpr std::array<std::array<double, 12>, 12> A = {
        std::array<double, 12>{1.0 / 18.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{1.0 / 48.0, 1.0 / 16.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{1.0 / 32.0, 0, 3.0 / 32.0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{5.0 / 16.0, 0, -75.0 / 64.0, 75.0 / 64.0, 0, 0, 0, 0, 0, 0, 0, 0},

        std::array<double, 12>{3.0 / 80.0, 0, 0, 3.0 / 16.0, 3.0 / 20.0, 0, 0, 0, 0, 0, 0, 0},
        std::array<double, 12>{29443841.0 / 614563906.0, 0, 0, 77736538.0 / 692538347.0,
                             -28693883.0 / 1125000000.0, 23124283.0 / 1800000000.0, 0, 0, 0, 0, 0,
                             0},
        std::array<double, 12>{16016141.0 / 946692911.0, 0, 0, 61564180.0 / 158732637.0,
                             22789713.0 / 633445777.0, 545815736.0 / 2771057229.0,
                             -180193667.0 / 1043307555.0, 0, 0, 0, 0, 0},
        std::array<double, 12>{39632708.0 / 573591083.0, 0, 0, -433636366.0 / 683701615.0,
                             -421739975.0 / 2616292301.0, 100302831.0 / 723423059.0,
                             790204164.0 / 839813087.0, 800635310.0 / 3783071287.0, 0, 0, 0, 0},

        std::array<double, 12>{246121993.0 / 1340847787.0, 0, 0, -37695042795.0 / 15268766246.0,
                             -309121744.0 / 1061227803.0, -12992083.0 / 490766935.0,
                             6005943493.0 / 2108947869.0, 393006217.0 / 1396673457.0,
                             123872331.0 / 1001029789.0, 0, 0, 0},
        std::array<double, 12>{-1028468189.0 / 846180014.0, 0, 0, 8478235783.0 / 508512852.0,
                             1311729495.0 / 1432422823.0, -10304129995.0 / 1701304382.0,
                             -48777925059.0 / 3047939560.0, 15336726248.0 / 1032824649,
                             -45442868181.0 / 3398467696, 3065993473.0 / 597172653.0, 0, 0},
        std::array<double, 12>{185892177.0 / 718116043.0, 0, 0, -3185094517.0 / 667107341.0,
                             -477755414.0 / 1098053517.0, -703635378.0 / 230739211.0,
                             5731566787.0 / 1027545527, 5232866602.0 / 850066563.0,
                             -4093664535.0 / 808688257.0, 3962137247.0 / 1805957418,
                             65686358.0 / 487910083.0, 0},
        std::array<double, 12>{403863854.0 / 491063109.0, 0, 0, -5068492393.0 / 434740067.0,
                             -411421997.0 / 543043805.0, 652783627.0 / 914296604.0,
                             11173962825.0 / 925320556, -13158990841.0 / 6184727034.0,
                             3936647629.0 / 1978049680.0, -160528059.0 / 685178525,
                             248638103.0 / 1413531060.0, 0},

    };

    static constexpr std::array<double, 12> C = {1.0 / 18.0,   1.0 / 12.0,
                                                  1.0 / 8,      5.0 / 16.0,
                                                  3.0 / 8.0,    59.0 / 400.0,
                                                  93.0 / 200.0, 5490023248.0 / 9719169821.0,
                                                  13.0 / 20.0,  1201146811.0 / 1299019798.0,
                                                  1.0,          1.0};
    static constexpr std::array<double, 13> B = {14005451.0 / 335480064.0,
                                                  0,
                                                  0,
                                                  0,
                                                  0,
                                                  -59238493.0 / 1068277825.0,
                                                  181606767.0 / 758867731.0,
                                                  561292985.0 / 797845732.0,
                                                  -1041891430.0 / 1371343529.0,
                                                  760417239.0 / 1151165299.0,
                                                  118820643.0 / 751138087.0,
                                                  -528747749.0 / 2220607170.0,
                                                  1.0 / 4.0};
    static constexpr std::array<double, 13> Bhat = {13451932.0 / 455176623.0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     -808719846.0 / 976000145.0,
                                                     1757004468.0 / 5645159321.0,
                                                     656045339.0 / 265891186.0,
                                                     -3867574721.0 / 1518517206.0,
                                                     465885868.0 / 322736535.0,
                                                     53011238.0 / 667516719.0,
                                                     2.0 / 45.0,
                                                     0};

    static constexpr std::array<double, BmidSize> Bmid = {
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
    static constexpr int BmidSize = FSAL ? Stages : Stages + 1;

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
    static constexpr std::array<double, BmidSize> Bmid = {
        2.14824704601937561e-01,  2.27124999999999966e-02,  7.91218061120906202e-01,
        -6.89504287051870612e-01, 2.63237072991632992e+00,  -2.03412170858730335e+00,
        6.25000000000000000e-02};
};

template <> struct RKCoeffs<IVPAlg::Tsit5Trans> {
    static constexpr int Stages = 6;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 0;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    static constexpr bool HasMidpoint = false; // transcription-only, no dense output

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
