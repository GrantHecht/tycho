// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// Enzyme-mode test ODE definitions.  Mirror existing structs in
// extensions/Tycho_Extensions.h, parameterized by derivative mode so we can
// pair <EnzymeAD, AutodiffFwd> against <AutodiffFwd, AutodiffFwd> for
// lane-for-lane Jacobian comparison in Task 1.8.
//
// CR3BP and MEE bodies are lifted verbatim from
// extensions/Tycho_Extensions.h::CR3BPAD::compute_impl and
// extensions/Tycho_Extensions.h::ModifiedDynamicsAD::compute_impl
// (only the constant member access is renamed: mu -> mu_, sqm -> sqm_).
// =============================================================================
#pragma once

#include <tycho/tycho.h>

namespace tycho_enzyme_test {

using tycho::vf::CVecRef;
using tycho::vf::DenseDerivativeMode;
using tycho::vf::VecRef;
using tycho::vf::VectorFunction;
using tycho::Vector3;

// 5 inputs (x, y, v, t, theta) -> 3 outputs.
template <DenseDerivativeMode Jm, DenseDerivativeMode Hm>
struct BrachODEModed : VectorFunction<BrachODEModed<Jm, Hm>, 5, 3, Jm, Hm> {
    using Base = VectorFunction<BrachODEModed<Jm, Hm>, 5, 3, Jm, Hm>;
    VF_TYPE_ALIASES(Base)

    double g_;
    BrachODEModed(double g = 32.2) : g_{g} {}

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        using std::cos;
        using std::sin;
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        const Scalar v = x[2];
        const Scalar theta = x[4];
        fx[0] = sin(theta) * v;
        fx[1] = -cos(theta) * v;
        fx[2] = Scalar(g_) * cos(theta);
    }
};

// 7 inputs, 6 outputs.  Body lifted from
// extensions/Tycho_Extensions.h::CR3BPAD::compute_impl verbatim
// (mu -> mu_).
template <DenseDerivativeMode Jm, DenseDerivativeMode Hm>
struct CR3BPModed : VectorFunction<CR3BPModed<Jm, Hm>, 7, 6, Jm, Hm> {
    using Base = VectorFunction<CR3BPModed<Jm, Hm>, 7, 6, Jm, Hm>;
    VF_TYPE_ALIASES(Base)

    double mu_;
    CR3BPModed(double mu = 0.0123) : mu_(mu) {}

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {

        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        Vector3<Scalar> X = x.template head<3>();
        Vector3<Scalar> V = x.template segment<3>(3);

        Vector3<Scalar> p1loc;
        p1loc[0] = -mu_;

        Vector3<Scalar> p2loc;
        p2loc[0] = 1.0 - mu_;

        Vector3<Scalar> dvec = X - p1loc;
        Vector3<Scalar> rvec = X - p2loc;

        Scalar d = dvec.norm();
        Scalar r = rvec.norm();

        fx.template head<3>() = V;
        fx.template segment<3>(3) = -(1.0 - mu_) * dvec / (d * d * d) - mu_ * rvec / (r * r * r);
        fx[3] += 2.0 * V[1] + X[0];
        fx[4] += -2.0 * V[0] + X[1];
    }
};

// 9 inputs, 6 outputs.  Body lifted from
// extensions/Tycho_Extensions.h::ModifiedDynamicsAD::compute_impl verbatim
// (mu -> mu_, sqm -> sqm_).
template <DenseDerivativeMode Jm, DenseDerivativeMode Hm>
struct MEEModed : VectorFunction<MEEModed<Jm, Hm>, 9, 6, Jm, Hm> {
    using Base = VectorFunction<MEEModed<Jm, Hm>, 9, 6, Jm, Hm>;
    VF_TYPE_ALIASES(Base)

    double mu_;
    double sqm_;
    MEEModed(double mu = 1.0) : mu_(mu), sqm_(std::sqrt(mu)) {}

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        using std::cos;
        using std::sin;
        using std::sqrt;

        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        Scalar x0 = x[0];
        Scalar x1 = x[1];
        Scalar x2 = x[2];
        Scalar x3 = x[3];
        Scalar x4 = x[4];
        Scalar x5 = x[5];
        Scalar x6 = x[6];
        Scalar x7 = x[7];
        Scalar x8 = x[8];

        Scalar sqx0 = sqrt(x0);

        Scalar x9 = Scalar(1.0 / sqm_);
        Scalar x10 = cos(x5);
        Scalar x11 = sin(x5);
        Scalar x12 = x1 * x10 + x11 * x2;
        Scalar x13 = x12 + 1.0;
        Scalar x14 = 1.0 / x13;
        Scalar x15 = x14 * x7;
        Scalar x16 = x10 * x4;
        Scalar x17 = x11 * x3;
        Scalar x18 = x14 * x8;
        Scalar x19 = x12 + 2.0;
        Scalar x20 = sqx0 * x9;
        Scalar x21 = x18 * (-x16 + x17);
        Scalar x22 = 0.5 * x18 * x20 * ((x3 * x3) + (x4 * x4) + 1.0);

        fx[0] = 2.0 * (x0 * sqx0) * x15 * x9;
        fx[1] = x20 * (x11 * x6 + x15 * (x1 + x10 * x19) + x18 * x2 * (x16 - x17));
        fx[2] = x20 * (x1 * x21 - x10 * x6 + x15 * (x11 * x19 + x2));
        fx[3] = x10 * x22;
        fx[4] = x11 * x22;
        fx[5] = x20 * (mu_ * x13 * x13 / (x0 * x0) + 1.0 * x21);
    }
};

// Phase 4: non-templated compute_impl variant. Same brachistochrone dynamics
// as BrachODEModed<EnzymeAD, EnzymeAD>, but compute_impl drops the
// `<class InType, class OutType>` template header and uses concrete-typed
// arguments matching the EnzymeAD wrapper's `Eigen::Map<...>` exactly.
//
// This is the Phase 4 ergonomic win: when paired with <EnzymeAD, EnzymeAD>,
// the user's compute_impl uses ordinary `double` arithmetic instead of
// dual-number-friendly templated arithmetic.  AutodiffFwd cannot do this —
// see tests/cpp/enzyme/compile_fail/non_template_autodiff.cpp.
struct BrachEnzymeNonTemplate
    : tycho::vf::VectorFunction<BrachEnzymeNonTemplate, 5, 3,
                                tycho::vf::DenseDerivativeMode::EnzymeAD,
                                tycho::vf::DenseDerivativeMode::EnzymeAD> {
    double g_;
    BrachEnzymeNonTemplate(double g = 32.2) : g_{g} {}

    inline void compute_impl(
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> fx) const {
        const double v = x[2];
        const double theta = x[4];
        fx[0] = std::sin(theta) * v;
        fx[1] = -std::cos(theta) * v;
        fx[2] = g_ * std::cos(theta);
    }
};

// Convenience aliases used by tests/benchmarks.
//   *Autodiff   = <AutodiffFwd, AutodiffFwd> (reference path)
//   *EnzymeAD   = <EnzymeAD, AutodiffFwd>    (Phase 1 mixed pairing)
//   *EnzymeFull = <EnzymeAD, EnzymeAD>       (Phase 2 full Enzyme pipeline)
using BrachAutodiff   = BrachODEModed<DenseDerivativeMode::AutodiffFwd, DenseDerivativeMode::AutodiffFwd>;
using BrachEnzymeAD   = BrachODEModed<DenseDerivativeMode::EnzymeAD,    DenseDerivativeMode::AutodiffFwd>;
using BrachEnzymeFull = BrachODEModed<DenseDerivativeMode::EnzymeAD,    DenseDerivativeMode::EnzymeAD>;
using CR3BPAutodiff   = CR3BPModed<DenseDerivativeMode::AutodiffFwd,    DenseDerivativeMode::AutodiffFwd>;
using CR3BPEnzymeAD   = CR3BPModed<DenseDerivativeMode::EnzymeAD,       DenseDerivativeMode::AutodiffFwd>;
using CR3BPEnzymeFull = CR3BPModed<DenseDerivativeMode::EnzymeAD,       DenseDerivativeMode::EnzymeAD>;
using MEEAutodiff     = MEEModed<DenseDerivativeMode::AutodiffFwd,      DenseDerivativeMode::AutodiffFwd>;
using MEEEnzymeAD     = MEEModed<DenseDerivativeMode::EnzymeAD,         DenseDerivativeMode::AutodiffFwd>;
using MEEEnzymeFull   = MEEModed<DenseDerivativeMode::EnzymeAD,         DenseDerivativeMode::EnzymeAD>;

} // namespace tycho_enzyme_test
