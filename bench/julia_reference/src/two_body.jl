# Two-body Kepler ODE — same as Tycho's Kepler model.
# State: [x, y, z, vx, vy, vz], μ = 398600.4418 km³/s²
# Matches tests/cpp/integrators/regression/regression_utils.h::leo_circular_x0
#
# Out-of-place formulation with StaticArrays for best performance on small
# fixed-size state vectors (see OrdinaryDiffEq.jl FAQ recommendations).

using OrdinaryDiffEq, StaticArrays, LinearAlgebra

const MU_EARTH = 398600.4418

function twobody(u::SVector{6, Float64}, p, t)
    mu = p[1]
    r = SVector(u[1], u[2], u[3])
    v = SVector(u[4], u[5], u[6])
    r3 = (r[1]*r[1] + r[2]*r[2] + r[3]*r[3])^1.5
    a = -mu/r3 .* r
    return SVector{6, Float64}(v[1], v[2], v[3], a[1], a[2], a[3])
end

function leo_circular_initial()
    r0 = 7000.0
    v0 = sqrt(MU_EARTH / r0)
    u0 = SVector{6, Float64}(r0, 0.0, 0.0, 0.0, v0, 0.0)
    T = 2π * sqrt(r0^3 / MU_EARTH)
    tf = T / 4.0
    return u0, tf, SVector{1, Float64}(MU_EARTH)
end
