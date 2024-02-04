function compute_quasiGrad_solution_d1(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64; post_process::Bool = false, run::Bool=true)
    # this is the master function which executes QuasiGrad (d1)
    # 
    # three network-size rounding schemes: 
    #   - fewer than 2500  buses: 75, 90, 99 100, 100
    #   - fewer than 10000 buses: 95, 100, 100
    #   - more than 10000 buses: 100, 100
    start_time = time()

    if run == true
        # =====================================================\\
        jsn = QuasiGrad.load_json(InFile1)
        adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
            QuasiGrad.base_initialization(jsn, Div=Division, hpc_params=true, line_switching=AllowSwitching);
        QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

        if sys.nb < 2500
            # baby systems
            qG.max_linear_pfs = 3
            qG.adam_max_time  = 20.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
            QuasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 55.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(75.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 5.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 50.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(90.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 5.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 55.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(99.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 5.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 55.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(true, prm, qG, stt, upd)

            # ====================================== #
            # => QuasiGrad.count_active_binaries!(prm, upd)
            # => QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
            # ====================================== #

            # time left? save 50 seconds for ramp_constrained solve
            time_for_pf               = 5.0
            time_for_final_activities = 50.0
            time_spent = time() - start_time
            time_left  = NewTimeLimitInSeconds - time_spent - time_for_final_activities - time_for_pf
            if time_left > 15.0
                time_for_final_pf   = time_left*0.05
                time_for_final_adam = time_left*0.7

                qG.adam_max_time  = time_for_final_pf
                qG.max_linear_pfs = 1
                QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
                QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
                qG.adam_max_time  = time_for_final_adam
                QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; clip_pq_based_on_bins=true)
                QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
            else
                # just run a final projection -- this needs to be here!!
                QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
            end
            
            # final activities
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

        elseif sys.nb < 12000
            # larger systems
            qG.max_linear_pfs = 3
            qG.adam_max_time  = 30.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
            QuasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 55.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(95.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 10.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 50.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(true, prm, qG, stt, upd)

            # ====================================== #
            # => QuasiGrad.count_active_binaries!(prm, upd)
            # => QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
            # ====================================== #

            # time left? save 100 seconds for ramp_constrained solve
            time_for_pf               = 10.0
            time_for_final_activities = 60.0
            time_spent = time() - start_time
            time_left  = NewTimeLimitInSeconds - time_spent - time_for_final_activities - time_for_pf
            if time_left > 30.0
                time_for_final_pf   = time_left*0.05
                time_for_final_adam = time_left*0.7

                qG.adam_max_time  = time_for_final_pf
                qG.max_linear_pfs = 1
                QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
                QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
                qG.adam_max_time  = time_for_final_adam
                QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; clip_pq_based_on_bins=true)
                QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
            else
                # just run a final projection -- this needs to be here!!
                QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
            end
            
            # final activities
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
        else
            # monster system
            # => qG.print_linear_pf_iterations = true

            qG.adam_max_time  = 28.0
            QuasiGrad.solve_power_flow_23k!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true, last_solve=false)
            # => QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 30.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            
            qG.adam_max_time  = 22.0
            QuasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true, clip_pq_based_on_bins=true)
            qG.max_linear_pfs = 1
            QuasiGrad.solve_parallel_linear_pf_with_Gurobi_23k!(flw, grd, idx, ntk, prm, qG, stt, sys; first_solve=false)
            
            QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
            QuasiGrad.snap_shunts!(true, prm, qG, stt, upd)
            
            # take 1 penalzied pf iteration, and take 2 true iterations
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized_23kd1!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            qG.max_linear_pfs_final_solve = 2
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; flip_groups=true)
            QuasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
        end

        tf = time() - start_time
        println("final time (internal): $tf")

        # post process?
        if post_process == true
            QuasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
        end
    end
end

function compute_quasiGrad_solution_d23(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64; post_process::Bool = false, run::Bool=true)
    # this is the master function which executes QuasiGrad (d2/3)
    # 
    # three network-size rounding schemes: 
    #   - fewer than 10000  buses: 50, 75, 90, 99 100, 100
    #   - more than 10000 buses:   95, 100, 100
    start_time = time()

    if run == true
        # =====================================================\\
        jsn = QuasiGrad.load_json(InFile1)
        adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
            QuasiGrad.base_initialization(jsn, Div=Division, hpc_params=true, line_switching=AllowSwitching);
        QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

        if sys.nb < 12000
            # baby systems
            qG.max_linear_pfs = 3
            qG.adam_max_time  = 5.0*30.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
            QuasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 10.0*60.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(50.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 5.0*5.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 10.0*60.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(75.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 5.0*5.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 10.0*60.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(90.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)  

            qG.adam_max_time  = 5.0*5.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 10.0*60.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(99.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 5.0*5.0
            QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 10.0*60.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(true, prm, qG, stt, upd)

            # ====================================== #
            QuasiGrad.count_active_binaries!(prm, upd)
            QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
            # ====================================== #

            # time left? save 50 seconds for ramp_constrained solve
            time_for_pf               = 3.0*10.0
            time_for_final_activities = 3.0*50.0
            time_spent = time() - start_time
            time_left  = NewTimeLimitInSeconds - time_spent - time_for_final_activities - time_for_pf
            if time_left > 30.0
                time_for_final_pf   = time_left*0.10
                time_for_final_adam = time_left*0.80

                qG.adam_max_time  = time_for_final_pf
                qG.max_linear_pfs = 2
                QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
                QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
                qG.adam_max_time  = time_for_final_adam
                QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; clip_pq_based_on_bins=true)
                QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
            end
            
            # final activities -- let the first power flow solve try to maintain reserve margins
            qG.max_linear_pfs_final_solve = 1
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized_reserve_penalized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

            qG.max_linear_pfs_final_solve = 3
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

        else
            # monster system
            qG.adam_max_time = 150.0
            QuasiGrad.solve_power_flow_23k!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true, last_solve=false)
            QuasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 1200.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(90.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

            qG.adam_max_time  = 100.0
            QuasiGrad.solve_power_flow_23k!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            qG.adam_max_time  = 1200.0
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
            QuasiGrad.snap_shunts!(true, prm, qG, stt, upd)   
            QuasiGrad.count_active_binaries!(prm, upd)

            qG.adam_max_time  = 100.0
            QuasiGrad.solve_power_flow_23k!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false, last_solve=true)
            QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            time_for_pf               = 150.0
            time_for_final_activities = 150.0
            time_spent = time() - start_time
            time_left  = NewTimeLimitInSeconds - time_spent - time_for_final_activities - time_for_pf
            qG.adam_max_time  = time_left*0.80 # just let it run..
            QuasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; clip_pq_based_on_bins=true)
            QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)

            # final activities -- let the first power flow solve try to maintain reserve margins
            qG.max_linear_pfs_final_solve = 1
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized_reserve_penalized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

            qG.max_linear_pfs_final_solve = 3
            QuasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
            QuasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
            QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
        end

        # post process?
        if post_process == true
            QuasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
        end
    end
end