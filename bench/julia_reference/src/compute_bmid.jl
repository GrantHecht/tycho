# Derive Bmid coefficients for Tsit5 at θ=0.5 from the Tsit5Interp polynomial.
# Tsit5 interp: b_i(Θ) = r_i1·Θ + r_i2·Θ² + r_i3·Θ³ + r_i4·Θ⁴
using Printf

# From OrdinaryDiffEqTsit5/.../tsit_tableaus.jl Tsit5Interp (CompiledFloats):
r = [
    [1.0,                -2.763706197274826,  2.9132554618219126, -1.0530884977290216],
    [0.0,                 0.13169999999999998, -0.2234,             0.1017],
    [0.0,                 3.9302962368947516, -5.941033872131505,   2.490627285651253],
    [0.0,               -12.411077166933676,  30.33818863028232,  -16.548102889244902],
    [0.0,                37.50931341651104,  -88.1789048947664,    47.37952196281928],
    [0.0,               -27.896526289197286,  65.09189467479366,  -34.87065786149661],
    [0.0,                 1.5,                -4.0,                 2.5],
]

θ = 0.5
# Tycho convention: Bmid[i] = 2·b_i(Θ=0.5), because Stepper::step applies a
# factor of (1/2) when summing (x_mid = x + Σ (Bmid[i]/2) · k_i, with k_i = h·f_i).
# Julia convention: sol(Θ) = x + h · Σ b_i(Θ) · f_i.
# Match: Σ Bmid[i] = 2 · Σ b_i(0.5) = 2 · 0.5 = 1.0 (sanity).
let totalsum = 0.0
    for (i, ri) in enumerate(r)
        bi = ri[1]*θ + ri[2]*θ^2 + ri[3]*θ^3 + ri[4]*θ^4
        @printf("  Bmid[%d] = %.17e\n", i-1, 2*bi)
        totalsum += 2*bi
    end
    @printf("# Σ Bmid = %.17e (expected: 1.0)\n", totalsum)
end
