# using Pkg
# Pkg.activate(DEPOT_PATH[1])
# add quasiGrad
# using quasiGrad
# include("./src/quasiGrad.jl")
using Pkg
Pkg.activate(".")
Pkg.status()
Pkg.instantiate()

using quasiGrad

function MyJulia1(InFile1::String, TimeLimitInSeconds::Int64, Division::Int64, NetworkModel::String, AllowSwitching::Int64; precompile_minisolver::Bool = false)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")

    VERSION

    ## how long did package loading take? Give it 1 sec for now..
    #NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 1.0
#
    ## which division are we solving?
    #if Division == 1
    #    quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching, precompile_minisolver == precompile_minisolver)
    #elseif Division == 2
    #    quasiGrad.compute_quasiGrad_solution_d2(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching, precompile_minisolver == precompile_minisolver)
    #elseif Division == 3
    #    quasiGrad.compute_quasiGrad_solution_d3(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching, precompile_minisolver == precompile_minisolver)
    #else
    #    println("Division not recognized!")
    #end
#
    ## compute the solution
    #quasiGrad.compute_quasiGrad_solution_23k_pf(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_practice(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_TIME_23643(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_ed_timing(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_diagnostics_loop(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_diagnostics(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_triage_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_timed(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_feas(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
end

function MyJulia1_test(InFile1::String, TimeLimitInSeconds::Int64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")

    # how long did package loading take? Give it 5 sec for now..
    NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 5.0

    # compute the solution
    quasiGrad.compute_quasiGrad_solution_ed_timing(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_load_solve_project_write_23643(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_diagnostics_loop(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_diagnostics(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_triage_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_timed(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
        # => quasiGrad.compute_quasiGrad_solution_feas(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
end

# => MyJulia1_test("C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json", 1, 1, "test", 1)