using Pkg
Pkg.activate(DEPOT_PATH[1])

# add quasiGrad
# using quasiGrad
# include("./src/quasiGrad.jl")
using quasiGrad

function MyJulia1(InFile1::String, TimeLimitInSeconds::Any, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")
    println(JuMP.Model)

    # how long did package loading take? Give it 5 sec for now..
    NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 5.0

    # compute the solution
    quasiGrad.compute_triage_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_timed(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_feas(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
end
