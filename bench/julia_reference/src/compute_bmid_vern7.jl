# Evaluate Julia's Vern7 interpolation polynomial at Θ=0.5 and emit
# Bmid = 2·b_i(Θ=0.5) for the Tycho dense-output schema.
#
# Vern7 interpolation polynomial (see verner_tableaus.jl Vern7InterpolationCoefficients
# and interpolants.jl @vern7pre0):
#   b_1(Θ)  = Θ · @evalpoly(Θ, r011, r012, ..., r017)
#   b_i(Θ)  = Θ² · @evalpoly(Θ, r_i2, r_i3, ..., r_i7)     for i ∈ {4,5,6,7,8,9,11,12,13,14,15,16}
#   b_i(Θ)  = 0                                             for i ∈ {2, 3, 10}
#
# Stages 1..10 are main integration stages (k_1..k_10); 11..16 are lazy
# interpolation extras (k_11..k_16, from Vern7ExtraStages).
#
# Tycho storage (with LastStageIsFxf=false, ExtraOffset=1, BmidStages=17):
#   Bmid[0..9]   — main stages (indices 1,4,5,6,7,8,9 populated; 2,3,10 = 0)
#   Bmid[10]     — f(xf) slot (0; Julia's polynomial does not reference f(xf))
#   Bmid[11..16] — extra interpolation stages k_11..k_16

using Printf

# CompiledFloats variant (Float64 literals). Column order: (r_i2, r_i3, r_i4, r_i5, r_i6, r_i7).
# For i=1, the leading r_{1,1} = 1 is handled as the linear Θ term.
r = Dict{Int, NTuple{6, Float64}}(
    1  => (-8.413387198332767, 33.675508884490895, -70.80159089484886, 80.64695108301298, -47.19413969837522, 11.133813442539243),
    4  => (8.754921980674396, -88.4596828699771, 346.9017638429916, -629.2580030059837, 529.6773755604193, -167.35886986514018),
    5  => (8.913387586637922, -90.06081846893218, 353.1807459217058, -640.6476819744374, 539.2646279047156, -170.38809442991547),
    6  => (5.1733120298478, -52.271115900055385, 204.9853867374073, -371.8306118563603, 312.9880934374529, -98.89290352172495),
    7  => (16.79537744079696, -169.70040000059728, 665.4937727009246, -1207.1638892336007, 1016.1291515818546, -321.06001557237494),
    8  => (-10.005997536098665, 101.1005433052275, -396.47391512378437, 719.1787707014183, -605.3681033918824, 191.27439892797935),
    9  => (2.764708833638599, -27.934602637390462, 109.54779186137893, -198.7128113064482, 167.26633571640318, -52.85010499525706),
    11 => (-2.1696320280163506, 22.016696037569876, -86.90152427798948, 159.22388973861476, -135.9618306534588, 43.792401183280006),
    12 => (-4.890070188793804, 22.75407737425176, -30.78034218537731, -2.797194317207249, 31.369456637508403, -15.655927320381801),
    13 => (10.862170929551967, -50.542971417827104, 68.37148040407511, 6.213326521632409, -69.68006323194157, 34.776056794509195),
    14 => (-11.37286691922923, 130.79058078246717, -488.65113677785604, 832.2148793276441, -664.7743368554426, 201.79288044241662),
    15 => (-5.919778732715007, 63.27679965889219, -265.432682088738, 520.1009254140611, -467.412109533902, 155.3868452824017),
    16 => (-10.492146197961823, 105.35538525188011, -409.43975011988937, 732.831448907654, -606.3044574733512, 188.0495196316683),
)

θ = 0.5
θpow = (θ, θ^2, θ^3, θ^4, θ^5, θ^6, θ^7)  # Θ¹, Θ², ..., Θ⁷

# b_i(Θ) per Vern7's form:
# i=1: b_1(Θ) = r_{1,1}·Θ + r_{1,2}·Θ² + r_{1,3}·Θ³ + ... + r_{1,7}·Θ⁷
#              where r_{1,1} = 1.
# i∈{4..9,11..16}: b_i(Θ) = r_{i,2}·Θ² + r_{i,3}·Θ³ + ... + r_{i,7}·Θ⁷
function b(i, θpow)
    (haskey(r, i) || i == 1) || return 0.0
    total = 0.0
    if i == 1
        total += θpow[1]  # r_{1,1} = 1
    end
    if haskey(r, i)
        # r[i] stores (r_i2, r_i3, r_i4, r_i5, r_i6, r_i7): k=2..7.
        for k in 1:6
            total += r[i][k] * θpow[k + 1]
        end
    end
    return total
end

println("// Bmid = 2·b_i(Θ=0.5) via Vern7 interpolation polynomial (CompiledFloats).")
println("// Layout: Bmid[0..9] main k_1..k_10, Bmid[10] f(xf) slot (=0), Bmid[11..16] extras k_11..k_16.")
totalsum = 0.0
for idx in 0:16
    if idx < 10
        # main stages: Tycho index = Julia stage - 1
        bmid = 2 * b(idx + 1, θpow)
        @printf("  Bmid[%2d] = %.17e  // k_%d (main)\n", idx, bmid, idx + 1)
    elseif idx == 10
        bmid = 0.0  # f(xf) slot — Vern7 polynomial does not reference f(xf)
        @printf("  Bmid[%2d] = %.17e  // f(xf) slot (unused)\n", idx, bmid)
    else
        # extra stages: Tycho index (11..16) = Julia stage (11..16)
        bmid = 2 * b(idx, θpow)
        @printf("  Bmid[%2d] = %.17e  // k_%d (extra)\n", idx, bmid, idx)
    end
    global totalsum += bmid
end
@printf("# Σ Bmid = %.17e (expected: 1.0)\n", totalsum)
