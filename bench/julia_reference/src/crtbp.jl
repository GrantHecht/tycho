# Classical circular restricted three-body problem (CR3BP) in the rotating
# frame, normalized units. Matches Tycho's CR3BP_ODE wrapper around
# CRTBPDynamics (extra-accel channel held at zero).
#
# State: u = [x, y, z, vx, vy, vz] in rotating-frame coordinates, scaled so
# that the primary-secondary distance is 1 and the synodic rotation rate is 1.
# Parameter: p = [mu] where mu = m2 / (m1 + m2).
#
# Primary (m1) is located at (-mu, 0, 0); secondary (m2) at (1 - mu, 0, 0).
# Out-of-place formulation with StaticArrays.

using OrdinaryDiffEq, StaticArrays, LinearAlgebra

function crtbp(u::SVector{6, Float64}, p, t)
    mu = p[1]
    x, y, z, vx, vy, vz = u[1], u[2], u[3], u[4], u[5], u[6]

    d1 = (x + mu)
    d2 = (x - 1.0 + mu)
    r1 = sqrt(d1 * d1 + y * y + z * z)
    r2 = sqrt(d2 * d2 + y * y + z * z)
    r1_3 = r1 * r1 * r1
    r2_3 = r2 * r2 * r2

    ax = 2.0 * vy + x - (1.0 - mu) * d1 / r1_3 - mu * d2 / r2_3
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1_3 - mu * y / r2_3
    az = -(1.0 - mu) * z / r1_3 - mu * z / r2_3

    return SVector{6, Float64}(vx, vy, vz, ax, ay, az)
end

function cr3bp_initial()
    # Planar halo seed near Earth-Moon L1 (tycho regression_utils.h::cr3bp_l1_x0).
    u0 = SVector{6, Float64}(0.84, 0.0, 0.05, 0.0, 0.15, 0.0)
    tf = 2.0
    mu = 0.012150585609624  # Earth-Moon mass ratio
    return u0, tf, SVector{1, Float64}(mu)
end
