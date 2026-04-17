# CR3BP normalized-units ODE — matches Tycho's CR3BP_Substitute (Kepler mu=1.0)
# used in regression test Case03.
#
# Out-of-place formulation with StaticArrays.

using OrdinaryDiffEq, StaticArrays, LinearAlgebra

function crtbp_substitute(u::SVector{6, Float64}, p, t)
    # Same as two-body with mu=1.0 (stand-in per SP1 regression_utils.h)
    mu = p[1]
    r = SVector(u[1], u[2], u[3])
    v = SVector(u[4], u[5], u[6])
    r3 = (r[1]*r[1] + r[2]*r[2] + r[3]*r[3])^1.5
    a = -mu/r3 .* r
    return SVector{6, Float64}(v[1], v[2], v[3], a[1], a[2], a[3])
end

function cr3bp_initial()
    u0 = SVector{6, Float64}(1.0, 0.0, 0.05, 0.0, 1.0, 0.0)
    tf = 2.0
    return u0, tf, SVector{1, Float64}(1.0)
end
