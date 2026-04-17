# Evaluate Julia's BS5 interpolation polynomial at Θ=0.5 and emit Bmid = 2·b_i(0.5).
#
# BS5 interp polynomial from OrdinaryDiffEqLowOrderRK BS5Interp_polyweights:
#   b_i(Θ) = r_{i,1}·Θ + r_{i,2}·Θ² + r_{i,3}·Θ³ + r_{i,4}·Θ⁴ + r_{i,5}·Θ⁵ + r_{i,6}·Θ⁶
# with
#   r_{1,1} = 1, r_{i,1} = 0 for i ≠ 1
#   b_2(Θ) ≡ 0 (BS5's b_2 = 0)
#   b_9(Θ) starts at Θ³ (r_{9,2} = 0)
# Stages 1..11 (8 main + 3 interpolation extras).
#
# Tycho convention: Bmid[i] = 2·b_i(Θ=0.5), so Σ Bmid = 1.0 (sanity).

using Printf

# Float64 (CompiledFloats) interp weights from BS5Interp_polyweights.
r = Dict{Int, NTuple{5, Float64}}(
    1  => (-8.103191118438032, 27.298136302807084, -45.470343197183475, 36.89579791207878, -11.547607240534195),
    3  => (-3.4362921012159933, 28.38602193195626, -72.2759709558427, 74.85879078710211, -27.245925284262768),
    4  => (-1.3276772735189397, 13.060399314592578, -35.469854845413806, 38.240038148817945, -14.307769126529058),
    5  => (-0.16393430643430643, 0.8847786781120115, -1.964240395021645, 1.9817123385873385, -0.7296779223862557),
    6  => (3.9253203306051474, -34.163041154825144, 77.4690381021765, -65.9928405785889, 19.121088881250603),
    7  => (-1.853826530612245, 16.83892431972789, -42.4062818877551, 42.17455357142857, -14.676126700680273),
    8  => (2.3125, -21.078125, 53.671875, -54.359375, 19.453125),
    9  => (0.0, -2.3333333333333335, 23.0, -39.0, 18.333333333333332),
    10 => (-3.3528990003856314, 30.106238940962648, -73.55422182095978, 70.20132282057416, -23.400440940191388),
    11 => (12.0, -59.0, 117.0, -105.0, 35.0),
)
# linear Θ¹ term: only stage 1 has r_{1,1} = 1; all others = 0.
lin = Dict{Int, Float64}(1 => 1.0)

θ = 0.5
θpow = (θ, θ^2, θ^3, θ^4, θ^5, θ^6)

totalsum = 0.0
for i in 1:11
    b_i = 0.0
    if haskey(lin, i)
        b_i += lin[i] * θpow[1]
    end
    if haskey(r, i)
        for k in 1:5  # r[i] stores (r_i2, r_i3, r_i4, r_i5, r_i6)
            b_i += r[i][k] * θpow[k + 1]
        end
    end
    bmid = 2 * b_i
    @printf("  Bmid[%2d] = %.17e  // stage %d\n", i - 1, bmid, i)
    global totalsum += bmid
end
@printf("# Σ Bmid = %.17e (expected: 1.0)\n", totalsum)
