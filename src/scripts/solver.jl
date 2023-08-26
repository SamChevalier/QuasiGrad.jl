function compute_quasiGrad_solution(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    # this is the master function which executes quasiGrad.
    # 
    # 
    # =====================================================\\
    # TT: start time
    start_time = time()
    jsn = quasiGrad.load_json(InFile1)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
        quasiGrad.base_initialization(jsn, Div=Division, hpc_params=true);
    
    qG.adam_max_time  = 60.0
    
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
    quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    # =...
    quasiGrad.count_active_binaries!(prm, upd)
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys, upd)
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
    quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)



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
        quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

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
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

    # E2. clean-up reserves by solving softly constrained LP
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # E3. run adam
    qG.adam_max_time = qG.adam_solve_times[end]
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

    # E4. LP projection
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    
    # E5. cleanup constrained powerflow
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys, upd)

    # E6. cleanup reserves
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # E7. write the final solution
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

    # E8. post process
    quasiGrad.post_process_stats(false, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
end

function compute_quasiGrad_solution_practice(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    # this is the master function which executes quasiGrad.
    # 
    # 
    # =====================================================\\
    # TT: start time
    start_time = time()

    jsn = quasiGrad.load_json(InFile1)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
        quasiGrad.base_initialization(jsn, Div=Division, hpc_params=true, line_switching=AllowSwitching);
    
    qG.adam_max_time  = 90.0
    
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
    quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    # =...
    quasiGrad.count_active_binaries!(prm, upd)
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys, upd)
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
    
    total_time = time() - start_time

    # post process
    quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

    # final print
    println("grand time: $(total_time)")

    println("ed upper bounds: $(scr[:ed_obj])")
    println("zms: $(scr[:zms])")
    println("zms_p: $(scr[:zms_penalized])")

    println("zp: $(scr[:zp])")
    println("zq: $(scr[:zq])")

    println("zs_acline: $(scr[:acl])")
    println("zs_xfm: $(scr[:xfm])")

    println("z_enpr: $(scr[:enpr])")
    println("z_encs: $(scr[:encs])")

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
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
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
                quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys, upd)
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
            quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
        end
    end
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys, upd)
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
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
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
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys, upd)
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
        quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

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
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

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
    quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys, upd)

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

function compute_quasiGrad_solution_ed_timing(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    v = VERSION
    println(v)

    st = Sys.CPU_THREADS
    nt = Threads.nthreads()  

    println("system threads: $st")
    println("number threads: $nt")

    # print all thread ids :)
    #Threads.@threads for i = 1:200
    #    tt = Threads.threadid()
    #    print("$tt, ")
    #    sleep(0.25)
    #end

    println()

    # load
    t = time()
    jsn = quasiGrad.load_json(InFile1)
    load_time = time() - t
    println("loadtime1: $load_time")

    #t = time()
    #jsn = quasiGrad.load_json(InFile1)
    #load_time = time() - t
    #println("loadtime2: $load_time")

    # initialize
    t = time()
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
    init_time = time() - t
    println("init time1: $init_time")

    #t = time()
    #adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
    #init_time = time() - t
    #println("init time2: $init_time")

    # ed
    t = time()
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    ed_time = time() - t
    println("ed time1: $ed_time")

    #t = time()
    #quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    #ed_time = time() - t
    #println("ed time2: $ed_time")

    # update
    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    #t = time()
    #qG.skip_ctg_eval = true
    #quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    #up_time = time() - t
    #println("up time2: $up_time")

    #t = time()
    #qG.skip_ctg_eval = true
    #quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    #up_time = time() - t
    #println("up time3: $up_time")

end

function compute_quasiGrad_solution_diagnostics_loop(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)

    v = VERSION
    println(v)

    st = Sys.CPU_THREADS
    nt = Threads.nthreads()  

    println("system threads: $st")
    println("number threads: $nt")

    println()
    jsn = quasiGrad.load_json(InFile1)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states = true)

    ntk.s_max          .= 0.25
    qG.skip_ctg_eval    = false
    qG.always_solve_ctg = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

    # loop!
    t1 = time()
    for ii in 1:1000
        quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    end

    lt = time() - t1
    println("loop time: $lt")
end

function compute_quasiGrad_load_solve_project_write_23643(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)

    # load!
    jsn = quasiGrad.load_json(InFile1)

    # initialize
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

    # write locally
    qG.write_location   = "local"
    qG.eval_grad        = true
    qG.always_solve_ctg = true
    qG.skip_ctg_eval    = false

    # turn off all printing
    qG.print_zms                     = false # print zms at every adam iteration?
    qG.print_final_stats             = false # print stats at the end?
    qG.print_lbfgs_iterations        = false
    qG.print_projection_success      = false
    qG.print_linear_pf_iterations    = false
    qG.print_reserve_cleanup_success = false
    
    # solve
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    #=
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

    solution_file = "C3E3N23643D1_scenario_003_solution.json"
    quasiGrad.write_solution(solution_file, prm, qG, stt, sys)

    # write results to file
    m1 = "zms = $(scr[:zms])"
    m2 = "penalized zms = $(scr[:zms_penalized])"
    m3 = "ctg avg = $(scr[:zctg_avg])"
    m4 = "ctg min = $(scr[:zctg_min])"
    m5 = "e-min = $(scr[:z_enmin])"
    m6 = "e-max = $(scr[:z_enmax])"

    solution_file = replace(solution_file, ".json" => "")
    txt_file = solution_file*".txt"
    open(txt_file, "w") do file
        write(file, m1)
        write(file, '\n')
        write(file, m2)
        write(file, '\n')
        write(file, '\n')
        write(file, m3)
        write(file, '\n')
        write(file, m4)
        write(file, '\n')
        write(file, '\n')
        write(file, m5)
        write(file, '\n')
        write(file, m6)
    end
    =#
end

function compute_quasiGrad_TIME_23643(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)

    # load!
    jsn = quasiGrad.load_json(InFile1)

    # initialize
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

    # write locally
    qG.write_location   = "local"
    qG.eval_grad        = true
    qG.always_solve_ctg = true
    qG.skip_ctg_eval    = false

    # turn off all printing
    qG.print_zms                     = false # print zms at every adam iteration?
    qG.print_final_stats             = false # print stats at the end?
    qG.print_lbfgs_iterations        = false
    qG.print_projection_success      = false
    qG.print_linear_pf_iterations    = false
    qG.print_reserve_cleanup_success = false
    
    # solve
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    
    # now, time everything
    print("t1: ")
    @time quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
    @time quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
    @time quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    print("t2: ")
    @time quasiGrad.clip_all!(prm, qG, stt)
    @time quasiGrad.clip_all!(prm, qG, stt)
    @time quasiGrad.clip_all!(prm, qG, stt)

    print("t3: ")
    @time quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
    @time quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
    @time quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)

    print("t4: ")
    @time quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)
    @time quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)
    @time quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)

    print("t5: ")
    @time quasiGrad.shunts!(grd, idx, prm, qG, stt)
    @time quasiGrad.shunts!(grd, idx, prm, qG, stt)
    @time quasiGrad.shunts!(grd, idx, prm, qG, stt)

    print("t6: ")
    @time quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
    @time quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
    @time quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)

    print("t7: ")
    @time quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
    @time quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
    @time quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)

    print("t8a: ")
    @time quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    @time quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    @time quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

    print("t8b: ")
    @time quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
    @time quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
    @time quasiGrad.device_reactive_powers!(idx, prm, qG, stt)

    print("t9: ")
    @time quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    @time quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    @time quasiGrad.energy_costs!(grd, prm, qG, stt, sys)

    print("t10: ")
    @time quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
    @time quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
    @time quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)

    print("t11: ")
    @time quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
    @time quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
    @time quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

    print("t12: ")
    @time quasiGrad.device_reserve_costs!(prm, qG, stt)
    @time quasiGrad.device_reserve_costs!(prm, qG, stt)
    @time quasiGrad.device_reserve_costs!(prm, qG, stt)

    print("t13: ")
    @time quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)
    @time quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)
    @time quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

    print("t14: ")
    @time quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)
    @time quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)
    @time quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

    print("t15: ")
    @time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    @time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    @time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

    print("t16: ")
    @time quasiGrad.score_zt!(idx, prm, qG, scr, stt)
    @time quasiGrad.score_zt!(idx, prm, qG, scr, stt)
    @time quasiGrad.score_zt!(idx, prm, qG, scr, stt)

    print("t17: ")
    @time quasiGrad.score_zbase!(qG, scr)
    @time quasiGrad.score_zbase!(qG, scr)
    @time quasiGrad.score_zbase!(qG, scr)

    print("t18: ")
    @time quasiGrad.score_zms!(scr)
    @time quasiGrad.score_zms!(scr)
    @time quasiGrad.score_zms!(scr)

    print("t19: ")
    @time quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
    @time quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
    @time quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

    print("t20: ")
    @time quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
    @time quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
    @time quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
end

function compute_quasiGrad_solution_one_sweep(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    v = VERSION
    println(v)

    st = Sys.CPU_THREADS
    nt = Threads.nthreads()  

    println("system threads: $st")
    println("number threads: $nt")

    println()

    # load
    t = time()
    jsn = quasiGrad.load_json(InFile1)
    load_time = time() - t
    println("loadtime1: $load_time")

    #t = time()
    #jsn = quasiGrad.load_json(InFile1)
    #load_time = time() - t
    #println("loadtime2: $load_time")

    # initialize
    t = time()
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
    init_time = time() - t
    println("init time1: $init_time")

    #t = time()
    #adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
    #init_time = time() - t
    #println("init time2: $init_time")

    # ed
    t = time()
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    ed_time = time() - t
    println("ed time1: $ed_time")

    #t = time()
    #quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    #ed_time = time() - t
    #println("ed time2: $ed_time")

    # update
    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    t = time()
    qG.skip_ctg_eval = true
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    up_time = time() - t
    println("up time: $up_time")

    #t = time()
    #qG.skip_ctg_eval = true
    #quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    #up_time = time() - t
    #println("up time2: $up_time")

    #t = time()
    #qG.skip_ctg_eval = true
    #quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    #up_time = time() - t
    #println("up time3: $up_time")

end