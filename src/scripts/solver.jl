function compute_quasiGrad_solution(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    # this is the master function which executes quasiGrad.
    # 
    # 
    # =====================================================\\
    # TT: start time
    start_time = time()

    # I1. load the system data
    jsn = quasiGrad.load_json(InFile1)
    
    # I2. initialize the system
    adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
    stt, sys, upd, wct = quasiGrad.base_initialization(jsn, Div=Division, hpc_params=true);

    # I3. run an economic dispatch and update the states
    quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

    # TT: time
    time_spent_before_loop = time() - start_time

    # TT: how much time is left?
    time_left = NewTimeLimitInSeconds - time_spent_before_loop

    # TT: time management:
    quasiGrad.manage_time!(time_left, qG)

    # loop and solve: adam -> projection -> IBR
    n_its = length(qG.pcts_to_round)
    for (solver_itr, pct_round) in enumerate(qG.pcts_to_round)

        # TT: set an adam solve time
        qG.adam_max_time = qG.adam_solve_times[solver_itr]

        # L1. run power flow
        quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

        # L2. clean-up reserves by solving softly constrained LP
        quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

        # L3. run adam
        quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

        # L4. solve and apply projection
        quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

        # L5. on the second-to-last iteration, fix the shunts; otherwise, just snap them
        fix = solver_itr == (n_its-1)
        quasiGrad.snap_shunts!(fix, prm, stt, upd)
    end
    ##############################################################
    ##############################################################

    # Now, we're in the End Game.
    #
    # with all binaries and shunts fixed and power flow solved, we..
    #   E1. solve power flow one more time
    #   E2. (softly) cleanup the reserves
    #   E3. run adam one more time, with very tight constraint tolerances
    #   E4. solve the (~MI)LP projection with very tight p/q
    #   E5. cleanup constrained powerflow with LP solver
    #   E6. cleanup the reserves
    #   E7. prepare (and clip) and write solution
    #   E8. post process (print stats)
    #   
    # ensure there are no more binaries/discrete variables:
    quasiGrad.count_active_binaries!(prm, upd)

    # E1. run power flow, one more time
    quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

    # E2. clean-up reserves by solving softly constrained LP
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # E3. run adam
    qG.adam_max_time = qG.adam_solve_times[end]
    quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

    # E4. LP projection
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    
    # E5. cleanup constrained powerflow
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

    # E6. cleanup reserves
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # E7. write the final solution
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

    # E8. post process
    quasiGrad.post_process_stats(false, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
end

function compute_quasiGrad_solution_feas(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    jsn = quasiGrad.load_json(InFile1)

    # initialize
    adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn)

    # solve
    fix       = true
    pct_round = 100.0
    quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.snap_shunts!(true, prm, stt, upd)
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
end

function compute_quasiGrad_solution_timed(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    # this is the master function which executes quasiGrad.
    # 
    # 
    # =====================================================\\
    # TT: start time
    start_time = time()

    # I1. load the system data
    jsn = quasiGrad.load_json(InFile1)

    time_elapsed = start_time - time()
    println("I1: $(time_elapsed)")

    # I2. initialize the system
    adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
    stt, sys, upd, wct = quasiGrad.base_initialization(jsn, Div=Division, hpc_params=true);

    time_elapsed = start_time - time()
    println("I2: $(time_elapsed)")

    qG.skip_ctg_eval = true

    # I3. run an economic dispatch and update the states
    quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

    time_elapsed = start_time - time()
    println("I3: $(time_elapsed)")

    # TT: time
    time_spent_before_loop = time() - start_time

    # TT: how much time is left?
    time_left = NewTimeLimitInSeconds - time_spent_before_loop

    # TT: time management:
    quasiGrad.manage_time!(time_left, qG)

    # loop and solve: adam -> projection -> IBR
    n_its = length(qG.pcts_to_round)
    for (solver_itr, pct_round) in enumerate(qG.pcts_to_round)

        # TT: set an adam solve time
        qG.adam_max_time = qG.adam_solve_times[solver_itr]

        # L1. run power flow
        quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

        time_elapsed = start_time - time()
        println("L1: $(time_elapsed)")

        # L2. clean-up reserves by solving softly constrained LP
        quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

        time_elapsed = start_time - time()
        println("L2: $(time_elapsed)")

        # L3. run adam
        quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

        time_elapsed = start_time - time()
        println("L3: $(time_elapsed)")

        # L4. solve and apply projection
        quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

        time_elapsed = start_time - time()
        println("L4: $(time_elapsed)")

        # L5. on the second-to-last iteration, fix the shunts; otherwise, just snap them
        fix = solver_itr == (n_its-1)
        quasiGrad.snap_shunts!(fix, prm, stt, upd)
    end
    ##############################################################
    ##############################################################

    # Now, we're in the End Game.
    #
    # with all binaries and shunts fixed and power flow solved, we..
    #   E1. solve power flow one more time
    #   E2. (softly) cleanup the reserves
    #   E3. run adam one more time, with very tight constraint tolerances
    #   E4. solve the (~MI)LP projection with very tight p/q
    #   E5. cleanup constrained powerflow with LP solver
    #   E6. cleanup the reserves
    #   E7. prepare (and clip) and write solution
    #   E8. post process (print stats)
    #   
    # ensure there are no more binaries/discrete variables:
    quasiGrad.count_active_binaries!(prm, upd)

    # E1. run power flow, one more time
    quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

    time_elapsed = start_time - time()
    println("E1: $(time_elapsed)")

    # E2. clean-up reserves by solving softly constrained LP
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    time_elapsed = start_time - time()
    println("E2: $(time_elapsed)")

    # E3. run adam
    qG.adam_max_time = qG.adam_solve_times[end]
    quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

    time_elapsed = start_time - time()
    println("E3: $(time_elapsed)")

    # E4. LP projection
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)

    time_elapsed = start_time - time()
    println("E4: $(time_elapsed)")
    
    # E5. cleanup constrained powerflow
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

    time_elapsed = start_time - time()
    println("E5: $(time_elapsed)")

    # E6. cleanup reserves
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    time_elapsed = start_time - time()
    println("E6: $(time_elapsed)")

    # E7. write the final solution
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

    time_elapsed = start_time - time()
    println("E7: $(time_elapsed)")

    # E8. post process
    quasiGrad.post_process_stats(false, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
end