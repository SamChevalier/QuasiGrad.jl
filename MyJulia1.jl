# Note -- all package calling and loading is done in the warmup file!
#
# using quasiGrad

# ============
using Pkg
Pkg.activate(DEPOT_PATH[1])

include("./src/quasiGrad.jl")

    # => add quasiGrad
    # @time using quasiGrad
    # => using Pkg
    # => Pkg.activate(".")
    # => Pkg.status()

function MyJulia1(InFile1::String, TimeLimitInSeconds::Int64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")

    # precompile -- only for testing!!!
    quasiGrad.pc("./src/precompile_37bus.json", NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

    # begin
    t0 = time()

    # how long did package loading take? Give it 1 sec for now..
    @info "remove this!!"
    TimeLimitInSeconds = 600.0
    NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 1.0
    
    # in this case, solve the system -- which division are we solving?
    if Division == 1
        quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
    elseif Division == 2
        quasiGrad.compute_quasiGrad_solution_d2(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
    elseif Division == 3
        quasiGrad.compute_quasiGrad_solution_d3(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
    else
        println("Division not recognized!")
    end

    # how long did that take?
    tf = time() - t0
    println("final time: $tf")
end
