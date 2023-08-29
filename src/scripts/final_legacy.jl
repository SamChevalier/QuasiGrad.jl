
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

        

#if precompile_minisolver == true
#    qG.adam_max_time = 3.0
#    # in this case, run a minisolve with the 14 bus system
#    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
#    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
#    quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
#    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
#    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
#    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
#    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
#    quasiGrad.count_active_binaries!(prm, upd)
#    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
#    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
#    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
#    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
#    quasiGrad.cleanup_constrained_pf_with_Gurobi!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
#    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
#    qG.write_location == "local"
#    quasiGrad.write_solution("junk.json", prm, qG, stt, sys)
#else
