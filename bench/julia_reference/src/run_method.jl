# Usage: julia --project=. src/run_method.jl <method> <problem> <abstol> <reltol> <outfile>
# method  ∈ {Tsit5, BS3, BS5, Vern7, Vern8, Vern9, DP5, DP8}
# problem ∈ {two_body, crtbp}
# outfile: binary file receiving [int32 n | float64*n state | int32 nsteps | float64 walltime_sec]
#
# Uses out-of-place ODE formulation with StaticArrays for best performance on
# small fixed-size state vectors.

include("two_body.jl")
include("crtbp.jl")

using OrdinaryDiffEq, StaticArrays, Printf

function method_from_string(s::AbstractString)
    s == "Tsit5" && return Tsit5()
    s == "BS3"   && return BS3()
    s == "BS5"   && return BS5()
    s == "Vern7" && return Vern7()
    s == "Vern8" && return Vern8()
    s == "Vern9" && return Vern9()
    s == "DP5"   && return DP5()
    s == "DP8"   && return DP8()
    error("Unknown method: $s")
end

function problem_from_string(s::AbstractString)
    if s == "two_body"
        u0, tf, p = leo_circular_initial()
        return (twobody, u0, tf, p)
    elseif s == "crtbp"
        u0, tf, p = cr3bp_initial()
        return (crtbp_substitute, u0, tf, p)
    else
        error("Unknown problem: $s")
    end
end

function main(args)
    length(args) == 5 || error("Usage: run_method.jl <method> <problem> <abstol> <reltol> <outfile>")
    method = method_from_string(args[1])
    f, u0, tf, p = problem_from_string(args[2])
    abstol = parse(Float64, args[3])
    reltol = parse(Float64, args[4])
    outfile = args[5]

    prob = ODEProblem{false}(f, u0, (0.0, tf), p)
    # Warm up JIT
    solve(prob, method; abstol=abstol, reltol=reltol, save_everystep=false)
    t0 = time_ns()
    sol = solve(prob, method; abstol=abstol, reltol=reltol, save_everystep=false)
    t1 = time_ns()
    walltime = (t1 - t0) / 1e9

    uf = sol.u[end]
    # save_everystep=false truncates sol.t; pull step count from destats instead.
    nsteps = sol.stats.naccept

    open(outfile, "w") do io
        n = Int32(length(uf))
        write(io, n)
        for x in uf; write(io, Float64(x)); end
        write(io, Int32(nsteps))
        write(io, Float64(walltime))
    end
    @printf("%s %s: uf=%s nsteps=%d walltime=%.3es\n",
            args[1], args[2], collect(uf), nsteps, walltime)
end

main(ARGS)
