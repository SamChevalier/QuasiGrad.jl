# Note -- all package calling and loading is done in the warmup file!
#
# using quasiGrad

# %%
# using Pkg
# Pkg.activate(DEPOT_PATH[1])
    # => add quasiGrad
    # => using quasiGrad
    # => include("./src/quasiGrad.jl")
    # => using Pkg
    # => Pkg.activate(".")
    # => Pkg.status()

function MyJulia1(InFile1::String, TimeLimitInSeconds::Int64, Division::Int64, NetworkModel::String, AllowSwitching::Int64; precompile_minisolver::Bool = false)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")

    # how long did package loading take? Give it 1 sec for now..
    NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 1.0

    if precompile_minisolver == false
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

    else
        jsn = quasiGrad.load_json(InFile1)
        adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
            quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);

        # assign a short run-time
        qG.adam_max_time = 3.0

        # in this case, run a minisolve with the 14 bus system
        quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
        quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
        quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
        quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
        quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
        quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
        quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
        quasiGrad.count_active_binaries!(prm, upd)
        quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
        quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
        quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
        quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
        quasiGrad.cleanup_constrained_pf_with_Gurobi!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
        quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
        
        qG.write_location == "local"
        quasiGrad.write_solution("junk.json", prm, qG, stt, sys)
    end
end