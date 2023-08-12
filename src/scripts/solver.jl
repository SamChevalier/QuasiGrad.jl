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
    
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, Div=Division, hpc_params=true);

    # I3. run an economic dispatch and update the states
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

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
        quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)

        # L2. clean-up reserves by solving softly constrained LP
        quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

        # L3. run adam
        quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

        # L4. solve and apply projection
        quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

        # L5. on the second-to-last iteration, fix the shunts; otherwise, just snap them
        fix = solver_itr == (n_its-1)
        quasiGrad.snap_shunts!(fix, prm, qG, stt, upd)
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
    quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)

    # E2. clean-up reserves by solving softly constrained LP
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # E3. run adam
    qG.adam_max_time = qG.adam_solve_times[end]
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

    # E4. LP projection
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    
    # E5. cleanup constrained powerflow
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

    # E6. cleanup reserves
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # E7. write the final solution
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

    # E8. post process
    quasiGrad.post_process_stats(false, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
end

function compute_triage_quasiGrad_solution(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    # this is the master function which executes quasiGrad.
    # 
    # 
    # =====================================================\\
    # TT: start time
    t_buff     = 30.0
    start_time = time()
    jsn        = quasiGrad.load_json(InFile1)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, Div=Division, hpc_params=true);
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    
    t_before_pf = time()
    quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    t_pf      = time() - t_before_pf
    time_left = NewTimeLimitInSeconds - (time() - start_time)

    # of the time we have left, we have 2x adam, 2x pf, and 2x projection -- 
    # assume t_projection = 0.25*pf, and adam = 10% overall. Is there enough?
    if time_left < 2*t_pf + 2*0.25*t_pf + 2*0.1*time_left + t_buff
        run_pf2 = false

        # skip something else?
        if time_left < t_pf + 2*0.25*t_pf + 2*0.1*time_left + t_buff
            run_adam1 = false

            if time_left < t_pf + 2*0.25*t_pf + 0.1*time_left + t_buff
                # just project and write solution -- no adam
                quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
                quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
                quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
                quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
                quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
            else
                # time for 2nd adam only (no 1st projection)
                qG.adam_max_time = time_left - t_pf - 0.25*t_pf - t_buff
            end
        else
            run_adam1        = true
            qG.adam_max_time = (time_left - t_pf - 2*0.25*t_pf - t_buff)/2
        end
    else
        # seems we have the time
        run_pf2          = true
        run_adam1        = true
        qG.adam_max_time = (time_left - 2*t_pf - 2*0.25*t_pf - t_buff)/2
    end

    if run_adam1
        quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
        quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
        quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
        quasiGrad.count_active_binaries!(prm, upd)
        if run_pf2
            quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)
            quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
        end
    end
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
    quasiGrad.post_process_stats(false, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    tf = time() - start_time
    println("final time: $(tf)")
end

function compute_quasiGrad_solution_feas(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    jsn = quasiGrad.load_json(InFile1)

    # initialize
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, hpc_params=false)

    # solve
    fix       = true
    pct_round = 100.0
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    sleep(5.0)
    quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)
    sleep(5.0)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    sleep(5.0)

    qG.adam_max_time = 30.0
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    sleep(5.0)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
    sleep(5.0)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)
    sleep(5.0)
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
    sleep(5.0)
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
    quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

    edscr = scr[:ed_obj]
    zms   = scr[:zms]
    println()
    println("Economic dispatch upper bound: $edscr")
    println("Final market suplus(zms): $zms")
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

    time_elapsed = time() - start_time
    println("I1: $(time_elapsed)")

    # I2. initialize the system
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, Div=Division, hpc_params=true);

    time_elapsed = time() - start_time
    println("I2: $(time_elapsed)")

    # I3. run an economic dispatch and update the states
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

    time_elapsed = time() - start_time
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
        quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)

        time_elapsed = time() - start_time
        println("L1: $(time_elapsed)")

        # L2. clean-up reserves by solving softly constrained LP
        quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

        time_elapsed = time() - start_time
        println("L2: $(time_elapsed)")

        # L3. run adam
        quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

        time_elapsed = time() - start_time
        println("L3: $(time_elapsed)")

        # L4. solve and apply projection
        quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

        time_elapsed = time() - start_time
        println("L4: $(time_elapsed)")

        # L5. on the second-to-last iteration, fix the shunts; otherwise, just snap them
        fix = solver_itr == (n_its-1)
        quasiGrad.snap_shunts!(fix, prm, qG, stt, upd)
    end
    ###############################################################
    ###############################################################
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
    quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)

    time_elapsed = time() - start_time
    println("E1: $(time_elapsed)")

    # E2. clean-up reserves by solving softly constrained LP
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    time_elapsed = time() - start_time
    println("E2: $(time_elapsed)")

    # E3. run adam
    qG.adam_max_time = qG.adam_solve_times[end]
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

    time_elapsed = time() - start_time
    println("E3: $(time_elapsed)")

    # E4. LP projection
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)

    time_elapsed = time() - start_time
    println("E4: $(time_elapsed)")

    # E5. cleanup constrained powerflow
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

    time_elapsed = time() - start_time
    println("E5: $(time_elapsed)")

    # E6. cleanup reserves
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    time_elapsed = time() - start_time
    println("E6: $(time_elapsed)")

    # E7. write the final solution
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

    time_elapsed = time() - start_time
    println("E7: $(time_elapsed)")

    # E8. post process
    quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
end

function compute_quasiGrad_solution_diagnostics(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)

    v = VERSION
    println(v)

    st = Sys.CPU_THREADS
    nt = Threads.nthreads()  

    println("system threads: $st")
    println("number threads: $nt")

    # print all thread ids :)
    Threads.@threads for i = 1:200
        tt = Threads.threadid()
        print("$tt, ")
        sleep(0.25)
    end

    println()

    # load
    t = time()
    jsn = quasiGrad.load_json(InFile1)
    load_time = time() - t
    println("loadtime1: $load_time")

    t = time()
    jsn = quasiGrad.load_json(InFile1)
    load_time = time() - t
    println("loadtime2: $load_time")

    t = time()
    jsn = quasiGrad.load_json(InFile1)
    load_time = time() - t
    println("loadtime3: $load_time")

    # initialize
    t = time()
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
    init_time = time() - t
    println("init time1: $init_time")

    t = time()
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
    init_time = time() - t
    println("init time2: $init_time")

    t = time()
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
    init_time = time() - t
    println("init time3: $init_time")

    # ed
    t = time()
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    ed_time = time() - t
    println("ed time1: $ed_time")

    t = time()
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    ed_time = time() - t
    println("ed time2: $ed_time")

    t = time()
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    ed_time = time() - t
    println("ed time3: $ed_time")

    # update
    t = time()
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time1: $up_time")

    t = time()
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time2: $up_time")

    t = time()
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time3: $up_time")

    # ctg
    qG.skip_ctg_eval    = false
    qG.always_solve_ctg = true
    t = time()
    quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    ctg_time = time() - t
    println("ctg time1: $ctg_time")

    t = time()
    quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    ctg_time = time() - t
    println("ctg time2: $ctg_time")

    t = time()
    quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    ctg_time = time() - t
    println("ctg time3: $ctg_time")
end