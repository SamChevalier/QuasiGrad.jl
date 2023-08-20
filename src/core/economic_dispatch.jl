# in this file, we design the function which solves economic dispatch
function solve_economic_dispatch!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}; include_sus_in_ed::Bool=true)
    # note: all binaries are LP relaxed (so there is no BaB-ing): 0 < b < 1

    # first, give BLAS access to ALL threads -- this can accelerate the barrier solver!
    LinearAlgebra.BLAS.set_num_threads(qG.num_threads)

    # build and empty the model!
    tstart = time()
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => qG.num_threads); add_bridges = false)
    set_optimizer_attribute(model, "Method", 3) # force a concurrent solver
    set_string_names_on_creation(model, false)

    # set model properties => let this run until it finishes
        # model = Model(Gurobi.Optimizer; add_bridges = false)
        # set_silent(model)
        # set_optimizer_attribute(model, "Threads", qG.num_threads)
        # quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
        # quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
        # quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)
        # quasiGrad.set_optimizer_attribute(model, "Crossover", 0)
    #quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", 1e-3)
    #quasiGrad.set_optimizer_attribute(model, "OptimalityTol",  1e-2)

    # define the minimum set of variables we will need to solve the constraints
    u_on_dev  = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.u_on_dev[tii][dev],  lower_bound = 0.0, upper_bound = 1.0) for tii in prm.ts.time_keys) # => base_name = "u_on_dev_t$(ii)",  
    p_on      = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_on[tii][dev])                                            for tii in prm.ts.time_keys) # => base_name = "p_on_t$(ii)",      
    dev_q     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.dev_q[tii][dev],     lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "dev_q_t$(ii)",     
    p_rgu     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_rgu[tii][dev],     lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_rgu_t$(ii)",     
    p_rgd     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_rgd[tii][dev],     lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_rgd_t$(ii)",     
    p_scr     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_scr[tii][dev],     lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_scr_t$(ii)",     
    p_nsc     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_nsc[tii][dev],     lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_nsc_t$(ii)",     
    p_rru_on  = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_rru_on[tii][dev],  lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_rru_on_t$(ii)",  
    p_rru_off = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_rru_off[tii][dev], lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_rru_off_t$(ii)", 
    p_rrd_on  = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_rrd_on[tii][dev],  lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_rrd_on_t$(ii)",  
    p_rrd_off = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.p_rrd_off[tii][dev], lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "p_rrd_off_t$(ii)", 
    q_qru     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.q_qru[tii][dev],     lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "q_qru_t$(ii)",     
    q_qrd     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.q_qrd[tii][dev],     lower_bound = 0.0)                    for tii in prm.ts.time_keys) # => base_name = "q_qrd_t$(ii)",     

    # add a few more (implicit) variables which are necessary for solving this system
    u_su_dev = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.u_su_dev[tii][dev], lower_bound = 0.0, upper_bound = 1.0) for tii in prm.ts.time_keys) # => base_name = "u_su_dev_t$(ii)", 
    u_sd_dev = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [dev = 1:sys.ndev], start=stt.u_sd_dev[tii][dev], lower_bound = 0.0, upper_bound = 1.0) for tii in prm.ts.time_keys) # => base_name = "u_sd_dev_t$(ii)", 
    
    # we have the affine "AffExpr" expressions (whose values are specified)
    dev_p   = Dict{Int32, Vector{AffExpr}}(tii => Vector{AffExpr}(undef, sys.ndev) for tii in prm.ts.time_keys)
    p_su    = Dict{Int32, Vector{AffExpr}}(tii => Vector{AffExpr}(undef, sys.ndev) for tii in prm.ts.time_keys)
    p_sd    = Dict{Int32, Vector{AffExpr}}(tii => Vector{AffExpr}(undef, sys.ndev) for tii in prm.ts.time_keys)
    zen_dev = Dict{Int32, Vector{AffExpr}}(tii => Vector{AffExpr}(undef, sys.ndev) for tii in prm.ts.time_keys)

    # now, we need to loop and set the affine expressions to 0
    #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
    for tii in prm.ts.time_keys
        for dev in prm.dev.dev_keys
            dev_p[tii][dev]   = AffExpr(0.0)
            p_su[tii][dev]    = AffExpr(0.0)
            p_sd[tii][dev]    = AffExpr(0.0)
            zen_dev[tii][dev] = AffExpr(0.0)
        end
    end

    # add scoring variables and affine terms
    p_rgu_zonal_REQ     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_rgu_zonal_REQ_t$(ii)",    
    p_rgd_zonal_REQ     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_rgd_zonal_REQ_t$(ii)",    
    p_scr_zonal_REQ     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_scr_zonal_REQ_t$(ii)",    
    p_nsc_zonal_REQ     = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_nsc_zonal_REQ_t$(ii)",    
    p_rgu_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_rgu_zonal_penalty_t$(ii)",
    p_rgd_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_rgd_zonal_penalty_t$(ii)",
    p_scr_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_scr_zonal_penalty_t$(ii)",
    p_nsc_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_nsc_zonal_penalty_t$(ii)",
    p_rru_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_rru_zonal_penalty_t$(ii)",
    p_rrd_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzP], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "p_rrd_zonal_penalty_t$(ii)",
    q_qru_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzQ], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "q_qru_zonal_penalty_t$(ii)",
    q_qrd_zonal_penalty = Dict{Int32, Vector{quasiGrad.VariableRef}}(tii => @variable(model, [1:sys.nzQ], lower_bound = 0.0) for tii in prm.ts.time_keys) # => base_name = "q_qrd_zonal_penalty_t$(ii)",

    # shall we also include startup states?
    if include_sus_in_ed == true
        u_sus = Dict{Int32, Vector{Vector{quasiGrad.VariableRef}}}(tii => [@variable(model, [sus = 1:prm.dev.num_sus[dev]], lower_bound = 0.0, upper_bound = 1.0) for dev in 1:sys.ndev] for tii in prm.ts.time_keys)  
    end

    # affine aggregation terms
    zms     = AffExpr(0.0)
    z_sus   = Vector{AffExpr}(undef, sys.ndev)
    z_enmax = Vector{AffExpr}(undef, sys.ndev)
    z_enmin = Vector{AffExpr}(undef, sys.ndev)

    # we define these as vectors so we can parallelize safely
    for dev in prm.dev.dev_keys
        z_sus[dev]   = AffExpr(0.0)
        z_enmax[dev] = AffExpr(0.0)
        z_enmin[dev] = AffExpr(0.0)
    end

    # loop over all devices => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for parallel which is NOT safe
    for dev in prm.dev.dev_keys
        # == define active power constraints ==
        for tii in prm.ts.time_keys
            # first, get the startup power
            T_supc     = idx.Ts_supc[dev][tii]     # T_set, p_supc_set = get_supc(tii, dev, prm)
            p_supc_set = idx.ps_supc_set[dev][tii] # T_set, p_supc_set = get_supc(tii, dev, prm)
            add_to_expression!(p_su[tii][dev], sum(p_supc_set[ii]*u_su_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_supc); init=0.0))

            # second, get the shutdown power
            T_sdpc     = idx.Ts_sdpc[dev][tii]     # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
            p_sdpc_set = idx.ps_sdpc_set[dev][tii] # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
            add_to_expression!(p_sd[tii][dev], sum(p_sdpc_set[ii]*u_sd_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_sdpc); init=0.0))

            # finally, get the total power balance
            dev_p[tii][dev] = p_on[tii][dev] + p_su[tii][dev] + p_sd[tii][dev]
        end

        # == define reactive power constraints ==
        for tii in prm.ts.time_keys
            # only a subset of devices will have a reactive power equality constraint
            if dev in idx.J_pqe

                # the following (pr vs cs) are equivalent
                if dev in idx.pr_devs
                    # producer?
                    T_supc = idx.Ts_supc[dev][tii] # T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][tii] # T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)
                    
                    # compute q -- this might be the only equality constraint (and below)
                    @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
                else
                    # the device must be a consumer :)
                    T_supc = idx.Ts_supc[dev][tii] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][tii] #T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][tii] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                    # compute q -- this might be the only equality constraint (and above)
                    @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
                end
            end
        end

        # loop over each time period and define the hard constraints
        for tii in prm.ts.time_keys
            # duration
            dt = prm.ts.duration[tii]

            # 0. startup states
            if include_sus_in_ed == true
                if prm.dev.num_sus[dev] > 0
                    # 1. here is the cost:
                    add_to_expression!(z_sus[dev], sum(u_sus[tii][dev][ii]*prm.dev.startup_states[dev][ii][1] for ii in 1:prm.dev.num_sus[dev]))

                    # 2. the device cannot be in a startup state unless it is starting up!
                    @constraint(model, sum(u_sus[tii][dev]; init=0.0) <= u_su_dev[tii][dev])

                    # 3. make sure the device was "on" in a sufficiently recent time period
                    for ii in 1:prm.dev.num_sus[dev] # these are the sus indices
                        if tii in idx.Ts_sus_jf[dev][tii][ii] # do we need the constraint?
                            @constraint(model, u_sus[tii][dev][ii] <= sum(u_on_dev[tij][dev] for tij in idx.Ts_sus_jft[dev][tii][ii]))
                        end
                    end
                end
            end

            # 1. Minimum downtime: zhat_mndn
            T_mndn = idx.Ts_mndn[dev][tii] # t_set = get_tmindn(tii, dev, prm)
            @constraint(model, u_su_dev[tii][dev] + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_mndn; init=0.0) - 1.0 <= 0.0)

            # 2. Minimum uptime: zhat_mnup
            T_mnup = idx.Ts_mnup[dev][tii] # t_set = get_tminup(tii, dev, prm)
            @constraint(model, u_sd_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_mnup; init=0.0) - 1.0 <= 0.0)

            # define the previous power value (used by both up and down ramping!)
            if tii == 1
                # note: p0 = prm.dev.init_p[dev]
                dev_p_previous = prm.dev.init_p[dev]
            else
                # grab previous time
                tii_m1 = prm.ts.time_keys[tii-1]
                dev_p_previous = dev_p[tii_m1][dev]
            end

            # 3. Ramping limits (up): zhat_rup
            @constraint(model, dev_p[tii][dev] - dev_p_previous
                    - dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii][dev] - u_su_dev[tii][dev])
                    +     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii][dev] + 1.0 - u_on_dev[tii][dev])) <= 0.0)

            # 4. Ramping limits (down): zhat_rd
            @constraint(model,  dev_p_previous - dev_p[tii][dev]
                    - dt*(prm.dev.p_ramp_down_ub[dev]*u_on_dev[tii][dev]
                    +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-u_on_dev[tii][dev])) <= 0.0)

            # 5. Regulation up: zhat_rgu
            @constraint(model, p_rgu[tii][dev] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[tii][dev] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] - prm.dev.p_syn_res_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - u_on_dev[tii][dev]) <= 0.0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[tii][dev]) <= 0.0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, p_on[tii][dev] + p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ub[dev][tii]*u_on_dev[tii][dev] <= 0.0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][tii]*u_on_dev[tii][dev] + p_rrd_on[tii][dev] + p_rgd[tii][dev] - p_on[tii][dev] <= 0.0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ub[dev][tii]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][tii] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][tii] # T_supc, ~ = get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][tii] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, dev_q[tii][dev] + q_qru[tii][dev] - prm.dev.q_ub[dev][tii]*u_sum <= 0.0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[tii][dev] + prm.dev.q_lb[dev][tii]*u_sum - dev_q[tii][dev] <= 0.0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[tii][dev] + q_qru[tii][dev] - prm.dev.q_0_ub[dev]*u_sum
                    - prm.dev.beta_ub[dev]*dev_p[tii][dev] <= 0.0)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_sum + 
                        prm.dev.beta_lb[dev]*dev_p[tii][dev] + 
                        q_qrd[tii][dev] - dev_q[tii][dev] <= 0.0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, p_on[tii][dev] + p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ub[dev][tii]*u_on_dev[tii][dev] <= 0.0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][tii]*u_on_dev[tii][dev] + p_rru_on[tii][dev] + p_scr[tii][dev] + p_rgu[tii][dev] - p_on[tii][dev] <= 0.0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_rrd_off[tii][dev] - prm.dev.p_ub[dev][tii]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][tii] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][tii] #T_supc, ~ = get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][tii] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, dev_q[tii][dev] + q_qrd[tii][dev] - prm.dev.q_ub[dev][tii]*u_sum <= 0.0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[tii][dev] + prm.dev.q_lb[dev][tii]*u_sum - dev_q[tii][dev] <= 0.0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[tii][dev] + q_qrd[tii][dev] - prm.dev.q_0_ub[dev]*u_sum
                    - prm.dev.beta_ub[dev]*dev_p[tii][dev] <= 0.0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                    + prm.dev.beta_lb[dev]*dev_p[tii][dev]
                    + q_qru[tii][dev] - dev_q[tii][dev] <= 0.0)
                end
            end
        end

        # misc penalty: maximum starts over multiple periods
        for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
            # get the time periods: zhat_mxst
            T_su_max = idx.Ts_su_max[dev][w_ind] #get_tsumax(w_params, prm)
            @constraint(model, sum(u_su_dev[tii][dev] for tii in T_su_max; init=0.0) - w_params[3] <= 0.0)
        end

        # now, we need to add two other sorts of constraints:
        # 1. "evolutionary" constraints which link startup and shutdown variables
        for tii in prm.ts.time_keys
            if tii == 1
                @constraint(model, u_on_dev[tii][dev] - prm.dev.init_on_status[dev] == u_su_dev[tii][dev] - u_sd_dev[tii][dev])
            else
                tii_m1 = prm.ts.time_keys[tii-1]
                @constraint(model, u_on_dev[tii][dev] - u_on_dev[tii_m1][dev] == u_su_dev[tii][dev] - u_sd_dev[tii][dev])
            end
            # only one can be nonzero
            @constraint(model, u_su_dev[tii][dev] + u_sd_dev[tii][dev] <= 1.0)
        end

        # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "upd"
        # upd = update states
        for tii in prm.ts.time_keys
            # if a device is *not* in the set of variables,
            # then it must be held constant! -- otherwise, try to hold it
            # close to its initial value
            if dev ∉ upd[:u_on_dev][tii]
                @constraint(model, u_on_dev[tii][dev] == stt.u_on_dev[tii][dev])
            end

            if dev ∉ upd[:p_rrd_off][tii]
                @constraint(model, p_rrd_off[tii][dev] == stt.p_rrd_off[tii][dev])
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[tii][dev] == stt.p_nsc[tii][dev])
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[tii][dev] == stt.p_rru_off[tii][dev])
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[tii][dev] == stt.q_qru[tii][dev])
            end

            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[tii][dev] == stt.q_qrd[tii][dev])
            end

            # now, deal with reactive powers, some of which are specified with equality
            # only a subset of devices will have a reactive power equality constraint
            #
            # nothing here :)
        end

        # ========== costs! ============= #
        for tii in prm.ts.time_keys
            # duration
            dt = prm.ts.duration[tii]

            # active power costs -- these were sorted previously!
            cst = prm.dev.cum_cost_blocks[dev][tii][1][2:end]  # cost for each block (trim leading 0)
            pbk = prm.dev.cum_cost_blocks[dev][tii][2][2:end]  # power in each block (trim leading 0)
            nbk = length(pbk)

            # define a set of intermediate vars "p_jtm"
            p_jtm = @variable(model, [1:nbk], lower_bound = 0.0)
            @constraint(model, p_jtm .<= pbk)

            # have the blocks sum to the output power
            @constraint(model, sum(p_jtm) == dev_p[tii][dev])

            # compute the cost!
            zen_dev[tii][dev] = dt*sum(cst.*p_jtm)
        end

        # compute the costs associated with device reserve offers --> computed directly in the objective!!
        # 
        # min/max energy requirements
        Wub = prm.dev.energy_req_ub[dev]
        Wlb = prm.dev.energy_req_lb[dev]

        # upper bounds
        for (w_ind, w_params) in enumerate(Wub)
            T_en_max = idx.Ts_en_max[dev][w_ind]
            zw_enmax = @variable(model, lower_bound = 0.0)
            @constraint(model, prm.vio.e_dev*(sum(prm.ts.duration[tii]*dev_p[tii][dev] for tii in T_en_max; init=0.0) - w_params[3]) <= zw_enmax)
            add_to_expression!(z_enmax[dev], -1.0, zw_enmax)
        end

        # lower bounds
        for (w_ind, w_params) in enumerate(Wlb)
            T_en_min = idx.Ts_en_min[dev][w_ind]
            zw_enmin = @variable(model, lower_bound = 0.0)
            @constraint(model, prm.vio.e_dev*(w_params[3] - sum(prm.ts.duration[tii]*dev_p[tii][dev] for tii in T_en_min; init=0.0)) <= zw_enmin)
            add_to_expression!(z_enmin[dev], -1.0, zw_enmin)
        end
    end

    # now, include a "copper plate" power balance constraint
    # loop over each time period and compute the power balance
    for tii in prm.ts.time_keys
        # power must balance at each time!
        sum_p   = AffExpr(0.0)
        sum_q   = AffExpr(0.0)

        # loop over each bus
        for bus in 1:sys.nb
            # active power balance:
            bus_p = +sum(dev_p[tii][dev] for dev in idx.cs[bus]; init=0.0) +
                    -sum(dev_p[tii][dev] for dev in idx.pr[bus]; init=0.0)
            add_to_expression!(sum_p, bus_p)

            # reactive power balance:
            bus_q = +sum(dev_q[tii][dev] for dev in idx.cs[bus]; init=0.0) +
                    -sum(dev_q[tii][dev] for dev in idx.pr[bus]; init=0.0)
            add_to_expression!(sum_q, bus_q)
        end

        # sum of active and reactive powers is 0
        @constraint(model, sum_p == 0.0)
        @constraint(model, sum_q == 0.0)
    end

    # loop over reserves
    for tii in prm.ts.time_keys
        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == prm.reserve.rgu_sigma[zone]*sum(dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == prm.reserve.rgd_sigma[zone]*sum(dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, prm.reserve.scr_sigma[zone]*[dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[tii][zone])
                @constraint(model, prm.reserve.nsc_sigma[zone]*[dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[tii][zone])
            end

            # balance equations -- compute the shortfall values
            #
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_pzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] <= p_rgu_zonal_penalty[tii][zone])
                
                @constraint(model, p_rgd_zonal_REQ[tii][zone] <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][tii] <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][tii] <= p_rrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] - 
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgu_zonal_penalty[tii][zone])

                @constraint(model, p_rgd_zonal_REQ[tii][zone] - 
                                sum(p_rgd[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) - 
                                sum(p_nsc[tii][dev] for dev in idx.dev_pzone[zone]) <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][tii] -
                                sum(p_rru_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][tii] -
                                sum(p_rrd_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[tii][zone])
            end
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_qzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, prm.reserve.qru_min[zone][tii] <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][tii] <= q_qrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, prm.reserve.qru_min[zone][tii] -
                                sum(q_qru[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][tii] -
                                sum(q_qrd[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[tii][zone])
            end
        end
        # shortfall penalties -- NOT needed explicitly (see objective)
    end

    # loop and compute costs!
    for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]
        
        # add up
        zt_temp = @expression(model,
            # consumer revenues (POSITIVE)
            sum(zen_dev[tii][dev] for dev in idx.cs_devs) - 
            # producer costs
            sum(zen_dev[tii][dev] for dev in idx.pr_devs) - 
            # startup costs
            sum(prm.dev.startup_cost.*u_su_dev[tii]) - 
            # shutdown costs
            sum(prm.dev.shutdown_cost.*u_sd_dev[tii]) - 
            # on-costs
            sum(dt*prm.dev.on_cost.*u_on_dev[tii]) - 
            # time-dependent su costs
            # => **** don't include this here: sum(stt.zsus_dev[tii]) - ****
            # local reserve penalties
            sum(dt*prm.dev.p_reg_res_up_cost_tmdv[tii].*p_rgu[tii]) -   # zrgu
            sum(dt*prm.dev.p_reg_res_down_cost_tmdv[tii].*p_rgd[tii]) - # zrgd
            sum(dt*prm.dev.p_syn_res_cost_tmdv[tii].*p_scr[tii]) -      # zscr
            sum(dt*prm.dev.p_nsyn_res_cost_tmdv[tii].*p_nsc[tii]) -     # znsc
            sum(dt*(prm.dev.p_ramp_res_up_online_cost_tmdv[tii].*p_rru_on[tii] .+
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[tii].*p_rru_off[tii])) -   # zrru
            sum(dt*(prm.dev.p_ramp_res_down_online_cost_tmdv[tii].*p_rrd_on[tii] .+
                    prm.dev.p_ramp_res_down_offline_cost_tmdv[tii].*p_rrd_off[tii])) - # zrrd
            sum(dt*prm.dev.q_res_up_cost_tmdv[tii].*q_qru[tii]) -   # zqru      
            sum(dt*prm.dev.q_res_down_cost_tmdv[tii].*q_qrd[tii]) - # zqrd
            # zonal reserve penalties (P)
            sum(dt*prm.vio.rgu_zonal.*p_rgu_zonal_penalty[tii]) -
            sum(dt*prm.vio.rgd_zonal.*p_rgd_zonal_penalty[tii]) -
            sum(dt*prm.vio.scr_zonal.*p_scr_zonal_penalty[tii]) -
            sum(dt*prm.vio.nsc_zonal.*p_nsc_zonal_penalty[tii]) -
            sum(dt*prm.vio.rru_zonal.*p_rru_zonal_penalty[tii]) -
            sum(dt*prm.vio.rrd_zonal.*p_rrd_zonal_penalty[tii]) -
            # zonal reserve penalties (Q)
            sum(dt*prm.vio.qru_zonal.*q_qru_zonal_penalty[tii]) -
            sum(dt*prm.vio.qrd_zonal.*q_qrd_zonal_penalty[tii]))

        # update zms
        add_to_expression!(zms, zt_temp)
    end

    # add in the min and max energy terms
    add_to_expression!(zms, sum(z_enmax))
    add_to_expression!(zms, sum(z_enmin))

    # include startup states?
    if include_sus_in_ed == true
        add_to_expression!(zms, -sum(z_sus))
    end

    # set the final objective
    @objective(model, Max, zms)

    # solve
    tbuild = round(time() - tstart, sigdigits = 5)
    println("ED build time: $tbuild")
    tnew = time()
    optimize!(model)
    tsolve = round(time() - tnew, sigdigits = 5)
    println("ED solve time: $tsolve")

    # test solution!
    soln_valid = solution_status(model)

    # did Gurobi find something valid?
    if soln_valid == true
        println("Economic dispatch. ", termination_status(model),". ","objective value: ", objective_value(model))

        # solve, and then return the solution
        for tii in prm.ts.time_keys
            stt.u_on_dev[tii]  .= value.(u_on_dev[tii])
            stt.p_on[tii]      .= value.(p_on[tii])
            stt.dev_q[tii]     .= value.(dev_q[tii])
            stt.p_rgu[tii]     .= value.(p_rgu[tii])
            stt.p_rgd[tii]     .= value.(p_rgd[tii])
            stt.p_scr[tii]     .= value.(p_scr[tii])
            stt.p_nsc[tii]     .= value.(p_nsc[tii])
            stt.p_rru_on[tii]  .= value.(p_rru_on[tii])
            stt.p_rru_off[tii] .= value.(p_rru_off[tii])
            stt.p_rrd_on[tii]  .= value.(p_rrd_on[tii])
            stt.p_rrd_off[tii] .= value.(p_rrd_off[tii])
            stt.q_qru[tii]     .= value.(q_qru[tii])
            stt.q_qrd[tii]     .= value.(q_qrd[tii])
        end

        # update the u_sum and powers (used in clipping, so must be correct!)
        qG.run_susd_updates = true
        quasiGrad.simple_device_statuses!(idx, prm, qG, stt)
        quasiGrad.transpose_binaries!(prm, qG, stt)             # huge error: was missing this!!
        quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

        # update the objective value score
        scr[:ed_obj] = objective_value(model)
    else
        # warn!
        @warn "Copper plate economic dispatch (LP) failed -- skip initialization!"
    end

    # finally, demote BLAS -- single thread usage!!
    LinearAlgebra.BLAS.set_num_threads(1)
end

function solve_parallel_economic_dispatch!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # note: all binaries are LP relaxed (so there is no BaB-ing): 0 < b < 1
    t_ed0 = time()
    
    # first, give BLAS access to one thread
    LinearAlgebra.BLAS.set_num_threads(1)

    # parallelize over single time periods
    Threads.@threads for tii in prm.ts.time_keys

        # build and empty the model!
        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => qG.num_threads); add_bridges = false)
        set_optimizer_attribute(model, "Method", 3) # force a concurrent solver
        set_string_names_on_creation(model, false)

        # define the minimum set of variables we will need to solve the constraints
        u_on_dev  = @variable(model, [dev = 1:sys.ndev], start=stt.u_on_dev[tii][dev],  lower_bound = 0.0, upper_bound = 1.0)
        p_on      = @variable(model, [dev = 1:sys.ndev], start=stt.p_on[tii][dev])                                           
        dev_q     = @variable(model, [dev = 1:sys.ndev], start=stt.dev_q[tii][dev],     lower_bound = 0.0)                   
        p_rgu     = @variable(model, [dev = 1:sys.ndev], start=stt.p_rgu[tii][dev],     lower_bound = 0.0)                   
        p_rgd     = @variable(model, [dev = 1:sys.ndev], start=stt.p_rgd[tii][dev],     lower_bound = 0.0)                   
        p_scr     = @variable(model, [dev = 1:sys.ndev], start=stt.p_scr[tii][dev],     lower_bound = 0.0)                   
        p_nsc     = @variable(model, [dev = 1:sys.ndev], start=stt.p_nsc[tii][dev],     lower_bound = 0.0)                   
        p_rru_on  = @variable(model, [dev = 1:sys.ndev], start=stt.p_rru_on[tii][dev],  lower_bound = 0.0)                   
        p_rru_off = @variable(model, [dev = 1:sys.ndev], start=stt.p_rru_off[tii][dev], lower_bound = 0.0)                   
        p_rrd_on  = @variable(model, [dev = 1:sys.ndev], start=stt.p_rrd_on[tii][dev],  lower_bound = 0.0)                   
        p_rrd_off = @variable(model, [dev = 1:sys.ndev], start=stt.p_rrd_off[tii][dev], lower_bound = 0.0)                   
        q_qru     = @variable(model, [dev = 1:sys.ndev], start=stt.q_qru[tii][dev],     lower_bound = 0.0)                   
        q_qrd     = @variable(model, [dev = 1:sys.ndev], start=stt.q_qrd[tii][dev],     lower_bound = 0.0)                   

        # we have the affine "AffExpr" expressions (whose values are specified)
        dev_p   = Vector{AffExpr}(undef, sys.ndev)
        zen_dev = Vector{AffExpr}(undef, sys.ndev)

        # now, we need to loop and set the affine expressions to 0
        #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
        for dev in prm.dev.dev_keys
            dev_p[dev]   = AffExpr(0.0)
            zen_dev[dev] = AffExpr(0.0)
        end

        # define startup and shutdown power to be 0
        p_su = zeros(sys.ndev)
        p_sd = zeros(sys.ndev)

        # add scoring variables and affine terms
        p_rgu_zonal_REQ     = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_rgd_zonal_REQ     = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_scr_zonal_REQ     = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_nsc_zonal_REQ     = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_rgu_zonal_penalty = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_rgd_zonal_penalty = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_scr_zonal_penalty = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_nsc_zonal_penalty = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_rru_zonal_penalty = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        p_rrd_zonal_penalty = @variable(model, [1:sys.nzP], lower_bound = 0.0)
        q_qru_zonal_penalty = @variable(model, [1:sys.nzQ], lower_bound = 0.0)
        q_qrd_zonal_penalty = @variable(model, [1:sys.nzQ], lower_bound = 0.0)

        # affine aggregation terms
        zms = AffExpr(0.0)

        # loop over all devices => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for parallel which is NOT safe
        for dev in prm.dev.dev_keys
            # == define active power constraints ==
            dev_p[dev] = p_on[dev]

            # == define reactive power constraints ==
            #
            # only a subset of devices will have a reactive power equality constraint
            if dev in idx.J_pqe

                # the following (pr vs cs) are equivalent
                if dev in idx.pr_devs
                    # compute q -- this might be the only equality constraint (and below)
                    @constraint(model, dev_q[dev] == prm.dev.q_0[dev]*u_on_dev[dev] + prm.dev.beta[dev]*dev_p[dev])
                else
                    # compute q -- this might be the only equality constraint (and above)
                    @constraint(model, dev_q[dev] == prm.dev.q_0[dev]*u_on_dev[dev] + prm.dev.beta[dev]*dev_p[dev])
                end
            end

            # duration
            dt = prm.ts.duration[tii]

            # =========  no ramping constraints :)))  =========

            # 5. Regulation up: zhat_rgu
            @constraint(model, p_rgu[dev] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[dev] <= 0.0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[dev] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[dev] <= 0.0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[dev] + p_scr[dev] - prm.dev.p_syn_res_ub[dev]*u_on_dev[dev] <= 0.0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[dev]) <= 0.0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[dev] <= 0.0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - u_on_dev[dev]) <= 0.0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[dev] <= 0.0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[dev]) <= 0.0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, p_on[dev] + p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ub[dev][tii]*u_on_dev[dev] <= 0.0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][tii]*u_on_dev[dev] + p_rrd_on[dev] + p_rgd[dev] - p_on[dev] <= 0.0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, p_su[dev] + p_sd[dev] + p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ub[dev][tii]*(1.0 - u_on_dev[dev]) <= 0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, dev_q[dev] + q_qru[dev] - prm.dev.q_ub[dev][tii]*u_on_dev[dev] <= 0.0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[dev] + prm.dev.q_lb[dev][tii]*u_on_dev[dev] - dev_q[dev] <= 0.0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[dev] + q_qru[dev] - prm.dev.q_0_ub[dev]*u_on_dev[dev]
                    - prm.dev.beta_ub[dev]*dev_p[dev] <= 0.0)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_on_dev[dev] + 
                        prm.dev.beta_lb[dev]*dev_p[dev] + 
                        q_qrd[dev] - dev_q[dev] <= 0.0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, p_on[dev] + p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ub[dev][tii]*u_on_dev[dev] <= 0.0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][tii]*u_on_dev[dev] + p_rru_on[dev] + p_scr[dev] + p_rgu[dev] - p_on[dev] <= 0.0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, p_su[dev] + p_sd[dev] + p_rrd_off[dev] - prm.dev.p_ub[dev][tii]*(1.0 - u_on_dev[dev]) <= 0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, dev_q[dev] + q_qrd[dev] - prm.dev.q_ub[dev][tii]*u_on_dev[dev] <= 0.0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[dev] + prm.dev.q_lb[dev][tii]*u_on_dev[dev] - dev_q[dev] <= 0.0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[dev] + q_qrd[dev] - prm.dev.q_0_ub[dev]*u_on_dev[dev]
                    - prm.dev.beta_ub[dev]*dev_p[dev] <= 0.0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_on_dev[dev]
                    + prm.dev.beta_lb[dev]*dev_p[dev]
                    + q_qru[dev] - dev_q[dev] <= 0.0)
                end
            end

            # 2. constraints which hold constant variables from moving
                # a. must run
                # b. planned outages
                # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
                # d. other states which are fixed from previous IBR rounds
                #       note: all of these are relfected in "upd"
            # upd = update states
            #
            # if a device is *not* in the set of variables,
            # then it must be held constant! -- otherwise, try to hold it
            # close to its initial value
            if dev ∉ upd[:u_on_dev][tii]
                @constraint(model, u_on_dev[dev] == stt.u_on_dev[tii][dev])
            end

            if dev ∉ upd[:p_rrd_off][tii]
                @constraint(model, p_rrd_off[dev] == stt.p_rrd_off[tii][dev])
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[dev] == stt.p_nsc[tii][dev])
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[dev] == stt.p_rru_off[tii][dev])
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[dev] == stt.q_qru[tii][dev])
            end

            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[dev] == stt.q_qrd[tii][dev])
            end

            # now, deal with reactive powers, some of which are specified with equality
            # only a subset of devices will have a reactive power equality constraint
            #
            # nothing here :)


            # ========== costs! ============= #
            # duration
            dt = prm.ts.duration[tii]

            # active power costs -- these were sorted previously!
            cst = prm.dev.cum_cost_blocks[dev][tii][1][2:end]  # cost for each block (trim leading 0)
            pbk = prm.dev.cum_cost_blocks[dev][tii][2][2:end]  # power in each block (trim leading 0)
            nbk = length(pbk)

            # define a set of intermediate vars "p_jtm"
            p_jtm = @variable(model, [1:nbk], lower_bound = 0.0)
            @constraint(model, p_jtm .<= pbk)

            # have the blocks sum to the output power
            @constraint(model, sum(p_jtm) == dev_p[dev])

            # compute the cost!
            zen_dev[dev] = dt*sum(cst.*p_jtm)

            # compute the costs associated with device reserve offers -- computed directly in the objective
            # 
            # min/max energy requirements
        end

        # now, include a "copper plate" power balance constraint
        # loop over each time period and compute the power balance

        # power must balance at each time!
        sum_p   = AffExpr(0.0)
        sum_q   = AffExpr(0.0)

        # loop over each bus
        for bus in 1:sys.nb
            # active power balance:
            bus_p = +sum(dev_p[dev] for dev in idx.cs[bus]; init=0.0) +
                    -sum(dev_p[dev] for dev in idx.pr[bus]; init=0.0)
            add_to_expression!(sum_p, bus_p)

            # reactive power balance:
            bus_q = +sum(dev_q[dev] for dev in idx.cs[bus]; init=0.0) +
                    -sum(dev_q[dev] for dev in idx.pr[bus]; init=0.0)
            add_to_expression!(sum_q, bus_q)
        end

        # sum of active and reactive powers is 0
        @constraint(model, sum_p == 0.0)
        @constraint(model, sum_q == 0.0)

        # loop over reserves
        #
        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[zone] == prm.reserve.rgu_sigma[zone]*sum(dev_p[dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[zone] == prm.reserve.rgd_sigma[zone]*sum(dev_p[dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[zone] == 0.0)
            else
                @constraint(model, prm.reserve.scr_sigma[zone]*[dev_p[dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[zone])
                @constraint(model, prm.reserve.nsc_sigma[zone]*[dev_p[dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[zone])
            end

            # balance equations -- compute the shortfall values
            #
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_pzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, p_rgu_zonal_REQ[zone] <= p_rgu_zonal_penalty[zone])
                
                @constraint(model, p_rgd_zonal_REQ[zone] <= p_rgd_zonal_penalty[zone])

                @constraint(model, p_rgu_zonal_REQ[zone] + 
                                p_scr_zonal_REQ[zone] <= p_scr_zonal_penalty[zone])

                @constraint(model, p_rgu_zonal_REQ[zone] + 
                                p_scr_zonal_REQ[zone] +
                                p_nsc_zonal_REQ[zone] <= p_nsc_zonal_penalty[zone])

                @constraint(model, prm.reserve.rru_min[zone][tii] <= p_rru_zonal_penalty[zone])

                @constraint(model, prm.reserve.rrd_min[zone][tii] <= p_rrd_zonal_penalty[zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, p_rgu_zonal_REQ[zone] - 
                                sum(p_rgu[dev] for dev in idx.dev_pzone[zone]) <= p_rgu_zonal_penalty[zone])

                @constraint(model, p_rgd_zonal_REQ[zone] - 
                                sum(p_rgd[dev] for dev in idx.dev_pzone[zone]) <= p_rgd_zonal_penalty[zone])

                @constraint(model, p_rgu_zonal_REQ[zone] + 
                                p_scr_zonal_REQ[zone] -
                                sum(p_rgu[dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[dev] for dev in idx.dev_pzone[zone]) <= p_scr_zonal_penalty[zone])

                @constraint(model, p_rgu_zonal_REQ[zone] + 
                                p_scr_zonal_REQ[zone] +
                                p_nsc_zonal_REQ[zone] -
                                sum(p_rgu[dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[dev] for dev in idx.dev_pzone[zone]) - 
                                sum(p_nsc[dev] for dev in idx.dev_pzone[zone]) <= p_nsc_zonal_penalty[zone])

                @constraint(model, prm.reserve.rru_min[zone][tii] -
                                sum(p_rru_on[dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[zone])

                @constraint(model, prm.reserve.rrd_min[zone][tii] -
                                sum(p_rrd_on[dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[zone])
            end
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_qzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, prm.reserve.qru_min[zone][tii] <= q_qru_zonal_penalty[zone])

                @constraint(model, prm.reserve.qrd_min[zone][tii] <= q_qrd_zonal_penalty[zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, prm.reserve.qru_min[zone][tii] -
                                sum(q_qru[dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[zone])

                @constraint(model, prm.reserve.qrd_min[zone][tii] -
                                sum(q_qrd[dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[zone])
            end
        end
        # shortfall penalties -- NOT needed explicitly (see objective)

        # compute costs!
        #
        # duration
        dt = prm.ts.duration[tii]
        
        # add up
        zt_temp = @expression(model,
            # consumer revenues (POSITIVE)
            sum(zen_dev[dev] for dev in idx.cs_devs) - 
            # producer costs
            sum(zen_dev[dev] for dev in idx.pr_devs) - 
            # startup costs
             # => sum(prm.dev.startup_cost.*u_su_dev) - 
            # shutdown costs
             # => sum(prm.dev.shutdown_cost.*u_sd_dev) - 
            # on-costs
            sum(dt*prm.dev.on_cost.*u_on_dev) - 
            # time-dependent su costs
            # => **** don't include this here: sum(stt.zsus_dev) - ****
            # local reserve penalties
            sum(dt*prm.dev.p_reg_res_up_cost_tmdv[tii].*p_rgu) -   # zrgu
            sum(dt*prm.dev.p_reg_res_down_cost_tmdv[tii].*p_rgd) - # zrgd
            sum(dt*prm.dev.p_syn_res_cost_tmdv[tii].*p_scr) -      # zscr
            sum(dt*prm.dev.p_nsyn_res_cost_tmdv[tii].*p_nsc) -     # znsc
            sum(dt*(prm.dev.p_ramp_res_up_online_cost_tmdv[tii].*p_rru_on .+
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[tii].*p_rru_off)) -   # zrru
            sum(dt*(prm.dev.p_ramp_res_down_online_cost_tmdv[tii].*p_rrd_on .+
                    prm.dev.p_ramp_res_down_offline_cost_tmdv[tii].*p_rrd_off)) - # zrrd
            sum(dt*prm.dev.q_res_up_cost_tmdv[tii].*q_qru) -   # zqru      
            sum(dt*prm.dev.q_res_down_cost_tmdv[tii].*q_qrd) - # zqrd
            # zonal reserve penalties (P)
            sum(dt*prm.vio.rgu_zonal.*p_rgu_zonal_penalty) -
            sum(dt*prm.vio.rgd_zonal.*p_rgd_zonal_penalty) -
            sum(dt*prm.vio.scr_zonal.*p_scr_zonal_penalty) -
            sum(dt*prm.vio.nsc_zonal.*p_nsc_zonal_penalty) -
            sum(dt*prm.vio.rru_zonal.*p_rru_zonal_penalty) -
            sum(dt*prm.vio.rrd_zonal.*p_rrd_zonal_penalty) -
            # zonal reserve penalties (Q)
            sum(dt*prm.vio.qru_zonal.*q_qru_zonal_penalty) -
            sum(dt*prm.vio.qrd_zonal.*q_qrd_zonal_penalty))

        # update zms
        add_to_expression!(zms, zt_temp)

        # set the final objective
        @objective(model, Max, zms)

        # solve
        optimize!(model)

        # test solution!
        soln_valid = solution_status(model)

        # did Gurobi find something valid?
        if soln_valid == true

            # solve, and then return the solution
            stt.u_on_dev[tii]  .= value.(u_on_dev)
            stt.p_on[tii]      .= value.(p_on)
            stt.dev_q[tii]     .= value.(dev_q)
            stt.p_rgu[tii]     .= value.(p_rgu)
            stt.p_rgd[tii]     .= value.(p_rgd)
            stt.p_scr[tii]     .= value.(p_scr)
            stt.p_nsc[tii]     .= value.(p_nsc)
            stt.p_rru_on[tii]  .= value.(p_rru_on)
            stt.p_rru_off[tii] .= value.(p_rru_off)
            stt.p_rrd_on[tii]  .= value.(p_rrd_on)
            stt.p_rrd_off[tii] .= value.(p_rrd_off)
            stt.q_qru[tii]     .= value.(q_qru)
            stt.q_qrd[tii]     .= value.(q_qrd)

            # update the objective value score
            stt.parallel_ed_obj[tii] = objective_value(model)
        else
            # warn!
            @warn "Copper plate economic dispatch (LP) failed -- skip initialization!"
        end
    end

    # now, we may compute the actual score!
    scr[:ed_obj] = sum(stt.parallel_ed_obj)

    # print solution
    t_ed = time() - t_ed0
    println("Parallel ED finished. Objective value: ", scr[:ed_obj], ". Total time: $t_ed.")
end


function dcpf_initialization!(flw::quasiGrad.Flow, idx::quasiGrad.Index, ntk::quasiGrad.Network, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System; balanced::Bool = false)
    # apply dcpf to the economic dispatch solution (see previous versions for a linearized voltage solver -- it doesn't really work)
    #
    # use "flw.theta[tii]" as the phase angle buffer (also used in ctg analysis)
    Threads.@threads for tii in prm.ts.time_keys
        # first, update the xfm phase shifters (whatever they may be..)
        flw.ac_phi[tii][idx.ac_phi] .= copy.(stt.phi[tii])

        # loop over each bus
        for bus in 1:sys.nb
            # active power balance -- just devices
            # !! don't include shunt or dc constributions, 
            #    since power might not balance !!
            stt.pinj_dc[tii][bus] = 
                sum(stt.dev_p[tii][pr] for pr in idx.pr[bus]; init=0.0) - 
                sum(stt.dev_p[tii][cs] for cs in idx.cs[bus]; init=0.0)
        end

        # are we dealing with a balanced dcpf? (i.e., does power balance?)
        if balanced == false
            # get the slack at this time
            @fastmath p_slack = 
                sum(@inbounds stt.dev_p[tii][pr] for pr in idx.pr_devs) -
                sum(@inbounds stt.dev_p[tii][cs] for cs in idx.cs_devs)

            # now, apply this slack power everywhere
            stt.pinj_dc[tii] .= stt.pinj_dc[tii] .- p_slack/sys.nb
        end

        # now, we need to solve Yb*theta = pinj, but we need to 
        # take phase shifters into account first:
        bt = -flw.ac_phi[tii].*ntk.b
        c  = stt.pinj_dc[tii][2:end] - ntk.Er'*bt
        # now, we need to solve Yb_r*theta_r = c via pcg

        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- just use LU
            stt.va[tii][2:end] .= ntk.Ybr\c
            stt.va[tii][1]      = 0.0 # make sure

        else
            # solve with pcg -- va
            _, ch = quasiGrad.cg!(flw.theta[tii], ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)

            # test the krylov solution
            if ~(ch.isconverged)
                # LU backup
                @info "Krylov failed -- using LU backup (dcpf)!"
                flw.theta[tii] .= ntk.Ybr\c
            end

            # update -- before updating, make sure that the largest 
            # phase angle differences are smaller than pi/3! if they are not
            # then scale the entire thing :)
            max_delta = maximum(abs.((@view stt.va[tii][idx.ac_fr_bus]) .- (@view stt.va[tii][idx.ac_fr_bus]) .- flw.ac_phi[tii]))

            if max_delta > pi/3
                # downscale! otherwise, you could have a really bad linearization!
                stt.va[tii][2:end] .= flw.theta[tii].*((pi/3)/max_delta)
                stt.va[tii][1]      = 0.0 # make sure
            else
                stt.va[tii][2:end] .= copy.(flw.theta[tii])
                stt.va[tii][1]      = 0.0 # make sure
            end
        end
    end
end

function economic_dispatch_initialization!(cgd::quasiGrad.ConstantGrad, ctg::quasiGrad.Contingency, flw::quasiGrad.Flow, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, ntk::quasiGrad.Network, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}; include_sus::Bool=true)
    # 1. run ED (global upper bound) -- we can only safely solve this for 5.5k devices over 18 time periods
    #    so, devs*tps = 5.5k*18 =~ 100k (80k) is our computational upper limit -- if we exceed this, we chop up 
    #    the ED problem and solve it over chunks -- then project
    if (sys.ndev * sys.nT) < 80000
        # 1. we're good -- just solve the entire thing all at once!
        quasiGrad.solve_economic_dispatch!(idx, prm, qG, scr, stt, sys, upd; include_sus_in_ed=include_sus)

        # 2. solve a dcpf for initializtion -- active power is garaunteed balanced
        quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys; balanced=true)
    else
        #  **** ===== ****  Large-Scale-OPT  **** ===== **** 
        #
        # 1. in this case, we are going to chop the ED problem up into chunks and do our best..
        quasiGrad.solve_parallel_economic_dispatch!(idx, prm, qG, scr, stt, sys, upd)

        # 2. next, hit all devices with an LP projection!
        quasiGrad.solve_LP_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
        
        # 3. update the u_sum and powers (used in clipping, so must be correct!)
        qG.run_susd_updates = true
        quasiGrad.simple_device_statuses!(idx, prm, qG, stt)
        quasiGrad.transpose_binaries!(prm, qG, stt)
        quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

        # now, solve an "unbalanced" dc power flow
        quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys; balanced = false)
    end

    # finally, update states
    qG.skip_ctg_eval = true
    qG.eval_grad     = false
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    qG.skip_ctg_eval = false
    qG.eval_grad     = true

    # nice idea in theory, but it throws off the network-wide balance way too much!
        # => # 4. take a guess at q injections, based on given voltages -- this is fairly arbitrary, but fast!!
        # => quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)

        # => # 5. update the states again
        # => quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
        # => qG.eval_grad     = true
        # => qG.skip_ctg_eval = tmp
end