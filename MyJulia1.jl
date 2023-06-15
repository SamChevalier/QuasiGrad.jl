using Pkg
Pkg.activate(DEPOT_PATH[1])

# add quasiGrad
#using quasiGrad
include("./src/quasiGrad.jl")

function MyJulia1(InFile1::String, TimeLimitInSeconds::Any, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")

    # how long did package loading take? Give it 3 sec for now..
    NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 3.0

    println("Threads:")
    println(Threads.nthreads())

    println("total!!")
    println(Sys.CPU_THREADS)

    # compute the solution
    quasiGrad.compute_quasiGrad_solution_timed(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
    #quasiGrad.compute_quasiGrad_solution_feas(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
    #quasiGrad.compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

    println("Threads:")
    println(Threads.nthreads())

    println("total!!")
    println(Sys.CPU_THREADS)
end