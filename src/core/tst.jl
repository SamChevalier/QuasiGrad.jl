# in this file, we design the function which solves economic dispatch
function solve_economic_dispatch!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}; include_sus::Bool=false)
    # note: all binaries are LP relaxed (so there is not BaB-ing): 0 < b < 1
    #
    # NOTE -- we are not including start-up-state discounts -- not worth it :)

    # build and empty the model!
    tstart = time()
    model = Model(Gurobi.Optimizer; add_bridges = false)
    set_optimizer_attribute(model, "Threads", qG.num_threads)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # set model properties => let this run until it finishes
        # quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
        # quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
        # quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)
        # quasiGrad.set_optimizer_attribute(model, "Crossover", 0)

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
    if include_sus == true
        u_sus = Dict{Int32, Vector{Vector{quasiGrad.VariableRef}}}(tii => [@variable(model, [sus = 1:prm.dev.num_sus[dev]], lower_bound = 0.0, upper_bound = 1.0) for dev in 1:sys.ndev] for tii in prm.ts.time_keys)  
    else

    # affine aggregation terms
    zms     = AffExpr(0.0)
    z_sus   = AffExpr(0.0)
    z_enmax = AffExpr(0.0)
    z_enmin = AffExpr(0.0)

    # loop over all devices
    @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for dev in prm.dev.dev_keys
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
            if include_sus == true
                if prm.dev.num_sus[dev] > 0
                    # 1. here is the cost:
                    add_to_expression!(z_sus, sum(u_sus[tii][dev][ii]*prm.dev.startup_states[dev][ii][1] for ii in 1:prm.dev.num_sus[dev]))

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
    end
end