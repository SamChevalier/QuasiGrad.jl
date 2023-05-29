function compute_quasiGrad_solution(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    # this is the master function which executes quasiGrad.
    #
    # =====================================================\\
    # start time
    start_time = time()

    # load the system data
    jsn = quasiGrad.load_json(InFile1)

    # initialize the system
    adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
    stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

    # run an economic dispatch and update the states
    quasiGrad.economic_dispatch_initialization!(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

    # get a power flow solution
    qG.num_lbfgs_steps = 250
    qG.initial_pf_lbfgs_step = 0.01

    quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

    # time
    time_spent_before_loop = time() - start_time

    # how much time is left?
    time_left = NewTimeLimitInSeconds - time_spent_before_loop

    # time management:
    quasiGrad.manage_time!(time_left, qG)

    # plot
    plt = Dict(:plot         => false,
            :first_plot      => true,
            :N_its           => 150,
            :global_adm_step => 0,
            :disp_freq       => 5)

    # loop and solve: adam -> projection -> IBR
    n_its = length(qG.pcts_to_round)
    for (solver_itr, pct_round) in enumerate(qG.pcts_to_round)

        # 0. set an adam solve time
        qG.adam_max_time = qG.adam_solve_times[solver_itr]

        # 1. run adam
        if plt[:plot]
            if plt[:first_plot] ax, fig, z_plt  = quasiGrad.initialize_plot(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, wct) end
            quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctb, ctd, fig, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, wct, z_plt)
        else
            quasiGrad.run_adam!(adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
        end

        # 2. solve Gurobi projection
        quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)

        # 3. fix binaries which are closest to their Gurobi solutions
        quasiGrad.batch_fix!(pct_round, prm, stt, sys, upd)

        # 4. using the previous solution, now update the state (i.e., apply the projection)
        quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)

        # 5. on the second-to-last iteration, fix the shunts; otherwise, just snap them
        fix = solver_itr == (n_its-1)
        quasiGrad.snap_shunts!(fix, prm, stt, upd)

        # 6. run power flow
        quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)
    end
    ##############################################################
    ##############################################################

    # Now, we're in the End Game.
    #
    # with all binaries and shunts fixed and power flow solved, we..
    #   E1. run adam one more time, with very tight constraint tolerances
    #   E2. solve the (~MI)LP projection with very tight p/q
    #   E3. p/q fixed: cleanup powerflow with adam: just v, theta, tau
    #           - goal: minimize power balance + line flow penalties
    #   E4. p/q fixed: cleanup with the final LP solver
    #           - guatenteed feasible, since E3 didn't touch device variables
    #   E5. prepare (and clip) and write solution
    #   E6. post process (print stats)
    #   
    # ensure there are no more binaries/discrete variables:
    quasiGrad.count_active_binaries!(prm, upd)

    # E1. run adam
    qG.adam_max_time = qG.adam_solve_times[end]
    if plt[:plot]
        quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctb, ctd, flw, fig, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, wct, z_plt)
    else
        quasiGrad.run_adam!(adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
    end

    # E2. project
    quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd, final_projection = true)

    # E3. adam with just pf variables
    #------------

    # E4. cleanup
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # E5. write the final solution
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

    # E6. post process
    quasiGrad.post_process_stats(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
end