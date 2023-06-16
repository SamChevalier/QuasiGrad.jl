# in this file, we prepare the hard device constraints, which we pass to Gurobi
#
# note -- this is ALWAYS run after clipping
function solve_Gurobi_projection_not_parallel!(final_projection::Bool, idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # loop over each device and solve individually -- not clear if this is faster
    # than solving one big optimization problem all at once. see legacy code for
    # a(n unfinished) version where all devices are solved at once!
    model = Model(Gurobi.Optimizer; add_bridges = false)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # status update
    @info "Running MILP projection across $(sys.ndev) devices."

    # set model properties
    quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
    quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)
    quasiGrad.set_optimizer_attribute(model, "IntFeasTol",     qG.IntFeasTol)
    quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
    #quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), qG.FeasibilityTol)
    #quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), qG.FeasibilityTol)

    # loop over all devices
    for dev in 1:sys.ndev
        # try projecting a second time if the first one fails!
        solve_projection = true
        first_solve      = true

        # loop twice, potentially
        while solve_projection == true
        
            # empty the model!
            empty!(model)

            # define local time keys
            tkeys = prm.ts.time_keys

            # define the minimum set of variables we will need to solve the constraints  
            if first_solve == true
                # in this case, use a hot start***  
                if final_projection == true
                    u_on_dev = Dict(tkeys[ii] => round(stt[:u_on_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_su_dev = Dict(tkeys[ii] => round(stt[:u_su_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_sd_dev = Dict(tkeys[ii] => round(stt[:u_sd_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                else
                    # define binary on/su/sd states
                    u_on_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_on_dev_t$(ii)", 
                    u_su_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_su_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_su_dev_t$(ii)", 
                    u_sd_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_sd_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_sd_dev_t$(ii)",  
                end
                p_on      = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_on][tkeys[ii]][dev])                         for ii in 1:(sys.nT)) # => base_name = "p_on_t$(ii)",      
                dev_q     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:dev_q][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "dev_q_t$(ii)",     
                p_rgu     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rgu][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgu_t$(ii)",     
                p_rgd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rgd][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgd_t$(ii)",     
                p_scr     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_scr][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_scr_t$(ii)",     
                p_nsc     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_nsc][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_nsc_t$(ii)",     
                p_rru_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rru_on][tkeys[ii]][dev],  lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_on_t$(ii)",  
                p_rru_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rru_off][tkeys[ii]][dev], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_off_t$(ii)", 
                p_rrd_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rrd_on][tkeys[ii]][dev],  lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_on_t$(ii)",  
                p_rrd_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rrd_off][tkeys[ii]][dev], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_off_t$(ii)", 
                q_qru     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:q_qru][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qru_t$(ii)",     
                q_qrd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:q_qrd][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qrd_t$(ii)",     
            else
                # in this case (i.e., we failed once), use a flat start! It seems to help :)    
                if final_projection == true
                    u_on_dev = Dict(tkeys[ii] => round(stt[:u_on_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_su_dev = Dict(tkeys[ii] => round(stt[:u_su_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_sd_dev = Dict(tkeys[ii] => round(stt[:u_sd_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                else
                    # define binary on/su/sd states
                    u_on_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_on_dev_t$(ii)", 
                    u_su_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_su_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_su_dev_t$(ii)", 
                    u_sd_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_sd_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_sd_dev_t$(ii)",  
                end
                p_on      = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_on_t$(ii)",      
                dev_q     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "dev_q_t$(ii)",     
                p_rgu     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgu_t$(ii)",     
                p_rgd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgd_t$(ii)",     
                p_scr     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_scr_t$(ii)",     
                p_nsc     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_nsc_t$(ii)",     
                p_rru_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_on_t$(ii)",  
                p_rru_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_off_t$(ii)", 
                p_rrd_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_on_t$(ii)",  
                p_rrd_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_off_t$(ii)", 
                q_qru     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qru_t$(ii)",     
                q_qrd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qrd_t$(ii)",     
            end

            # we have the affine "AffExpr" expressions (whose values are specified)
            dev_p = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))
            p_su  = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))
            p_sd  = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))

            # == define active power constraints ==
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # first, get the startup power
                T_supc     = idx.Ts_supc[dev][t_ind]     # T_set, p_supc_set = get_supc(tii, dev, prm)
                p_supc_set = idx.ps_supc_set[dev][t_ind] # T_set, p_supc_set = get_supc(tii, dev, prm)
                add_to_expression!(p_su[tii], sum(p_supc_set[ii]*u_su_dev[tii_inst] for (ii,tii_inst) in enumerate(T_supc); init=0.0))

                # second, get the shutdown power
                T_sdpc     = idx.Ts_sdpc[dev][t_ind]     # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
                p_sdpc_set = idx.ps_sdpc_set[dev][t_ind] # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
                add_to_expression!(p_sd[tii], sum(p_sdpc_set[ii]*u_sd_dev[tii_inst] for (ii,tii_inst) in enumerate(T_sdpc); init=0.0))

                # finally, get the total power balance
                dev_p[tii] = p_on[tii] + p_su[tii] + p_sd[tii]
            end

            # == define reactive power constraints ==
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # only a subset of devices will have a reactive power equality constraint
                if dev in idx.J_pqe

                    # the following (pr vs cs) are equivalent
                    if dev in idx.pr_devs
                        # producer?
                        T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
                        T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm)
                        u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)
                        
                        # compute q -- this might be the only equality constraint (and below)
                        @constraint(model, dev_q[tii] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii])
                    else
                        # the device must be a consumer :)
                        T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                        T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                        u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                        # compute q -- this might be the only equality constraint (and above)
                        @constraint(model, dev_q[tii] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii])
                    end
                end
            end

            # loop over each time period and define the hard constraints
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # duration
                dt = prm.ts.duration[tii]

                # 1. Minimum downtime: zhat_mndn
                T_mndn = idx.Ts_mndn[dev][t_ind] # t_set = get_tmindn(tii, dev, prm)
                @constraint(model, u_su_dev[tii] + sum(u_sd_dev[tii_inst] for tii_inst in T_mndn; init=0.0) - 1.0 <= 0.0)

                # 2. Minimum uptime: zhat_mnup
                T_mnup = idx.Ts_mnup[dev][t_ind] # t_set = get_tminup(tii, dev, prm)
                @constraint(model, u_sd_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_mnup; init=0.0) - 1.0 <= 0.0)

                # define the previous power value (used by both up and down ramping!)
                if tii == :t1
                    # note: p0 = prm.dev.init_p[dev]
                    dev_p_previous = prm.dev.init_p[dev]
                else
                    # grab previous time
                    tii_m1 = prm.ts.time_keys[t_ind-1]
                    dev_p_previous = dev_p[tii_m1]
                end

                # 3. Ramping limits (up): zhat_rup
                @constraint(model, dev_p[tii] - dev_p_previous
                        - dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii] - u_su_dev[tii])
                        +     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii] + 1.0 - u_on_dev[tii])) <= 0.0)

                # 4. Ramping limits (down): zhat_rd
                @constraint(model,  dev_p_previous - dev_p[tii]
                        - dt*(prm.dev.p_ramp_down_ub[dev]*u_on_dev[tii]
                        +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-u_on_dev[tii])) <= 0.0)

                # 5. Regulation up: zhat_rgu
                @constraint(model, p_rgu[tii] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[tii] <= 0.0)

                # 6. Regulation down: zhat_rgd
                @constraint(model, p_rgd[tii] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[tii] <= 0.0)

                # 7. Synchronized reserve: zhat_scr
                @constraint(model, p_rgu[tii] + p_scr[tii] - prm.dev.p_syn_res_ub[dev]*u_on_dev[tii] <= 0.0)

                # 8. Synchronized reserve: zhat_nsc
                @constraint(model, p_nsc[tii] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[tii]) <= 0.0)

                # 9. Ramping reserve up (on): zhat_rruon
                @constraint(model, p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[tii] <= 0.0)

                # 10. Ramping reserve up (off): zhat_rruoff
                @constraint(model, p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0-u_on_dev[tii]) <= 0.0)
                
                # 11. Ramping reserve down (on): zhat_rrdon
                @constraint(model, p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[tii] <= 0.0)

                # 12. Ramping reserve down (off): zhat_rrdoff
                @constraint(model, p_rrd_off[tii] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[tii]) <= 0.0)
                
                # Now, we must separate: producers vs consumers
                if dev in idx.pr_devs
                    # 13p. Maximum reserve limits (producers): zhat_pmax
                    @constraint(model, p_on[tii] + p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0.0)
                
                    # 14p. Minimum reserve limits (producers): zhat_pmin
                    @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rrd_on[tii] + p_rgd[tii] - p_on[tii] <= 0.0)
                    
                    # 15p. Off reserve limits (producers): zhat_pmaxoff
                    @constraint(model, p_su[tii] + p_sd[tii] + p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0.0)

                    # get common "u_sum" terms that will be used in the subsequent four equations 
                    T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                    # 16p. Maximum reactive power reserves (producers): zhat_qmax
                    @constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

                    # 17p. Minimum reactive power reserves (producers): zhat_qmin
                    @constraint(model, q_qrd[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0.0)

                    # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                    if dev in idx.J_pqmax
                        @constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_0_ub[dev]*u_sum
                        - prm.dev.beta_ub[dev]*dev_p[tii] <= 0.0)
                    end 
                    
                    # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                    if dev in idx.J_pqmin
                        @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                        + prm.dev.beta_lb[dev]*dev_p[tii]
                        + q_qrd[tii] - dev_q[tii] <= 0.0)
                    end

                # consumers
                else  # => dev in idx.cs_devs
                    # 13c. Maximum reserve limits (consumers): zhat_pmax
                    @constraint(model, p_on[tii] + p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0.0)

                    # 14c. Minimum reserve limits (consumers): zhat_pmin
                    @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rru_on[tii] + p_scr[tii] + p_rgu[tii] - p_on[tii] <= 0.0)
                    
                    # 15c. Off reserve limits (consumers): zhat_pmaxoff
                    @constraint(model, p_su[tii] + p_sd[tii] + p_rrd_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0.0)

                    # get common "u_sum" terms that will be used in the subsequent four equations 
                    T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                    # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                    @constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

                    # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                    @constraint(model, q_qru[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0.0)
                    
                    # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                    if dev in idx.J_pqmax
                        @constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_0_ub[dev]*u_sum
                        - prm.dev.beta_ub[dev]*dev_p[tii] <= 0.0)
                    end 

                    # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                    if dev in idx.J_pqmin
                        @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                        + prm.dev.beta_lb[dev]*dev_p[tii]
                        + q_qru[tii] - dev_q[tii] <= 0.0)
                    end
                end
            end

            # misc penalty: maximum starts over multiple periods
            for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
                # get the time periods: zhat_mxst
                T_su_max = idx.Ts_su_max[dev][w_ind] #get_tsumax(w_params, prm)
                @constraint(model, sum(u_su_dev[tii] for tii in T_su_max; init=0.0) - w_params[3] <= 0.0)
            end

            # now, we need to add two other sorts of constraints:
            # 1. "evolutionary" constraints which link startup and shutdown variables
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                if tii == :t1
                    @constraint(model, u_on_dev[tii] - prm.dev.init_on_status[dev] == u_su_dev[tii] - u_sd_dev[tii])
                else
                    tii_m1 = prm.ts.time_keys[t_ind-1]
                    @constraint(model, u_on_dev[tii] - u_on_dev[tii_m1] == u_su_dev[tii] - u_sd_dev[tii])
                end
                # only one can be nonzero
                @constraint(model, u_su_dev[tii] + u_sd_dev[tii] <= 1)
            end

            # 2. constraints which hold constant variables from moving
                # a. must run
                # b. planned outages
                # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
                # d. other states which are fixed from previous IBR rounds
                #       note: all of these are relfected in "upd"
            # upd = update states
            #
            # note -- in this loop, we also build the objective function!
            # now, let's define an objective function and solve this mf.
            # our overall objective is to round and fix some subset of 
            # integer variables. Here is our approach: find a feasible
            # solution which is as close to our Adam solution as possible.
            # next, we process the results: we identify the x% of variables
            # which had to move "the least". We fix these values and remove
            # their associated indices from upd. the end.
            #
            # afterwards, we initialize adam with the closest feasible
            # solution variable values.
            obj = AffExpr(0.0)

            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # if a device is *not* in the set of variables,
                # then it must be held constant! -- otherwise, try to hold it
                # close to its initial value
                if dev ∉ upd[:u_on_dev][tii]
                    @constraint(model, u_on_dev[tii] == stt[:u_on_dev][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, u_on_dev[tii]  - stt[:u_on_dev][tii][dev] <= tmp)
                    @constraint(model, stt[:u_on_dev][tii][dev] - u_on_dev[tii]  <= tmp)
                    add_to_expression!(obj, tmp, qG.binary_projection_weight)
                end

                if dev ∉ upd[:p_rrd_off][tii]
                    @constraint(model, p_rrd_off[tii] == stt[:p_rrd_off][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, p_rrd_off[tii] - stt[:p_rrd_off][tii][dev] <= tmp)
                    @constraint(model, stt[:p_rrd_off][tii][dev] - p_rrd_off[tii] <= tmp)
                    add_to_expression!(obj, tmp)
                end

                if dev ∉ upd[:p_nsc][tii]
                    @constraint(model, p_nsc[tii] == stt[:p_nsc][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, p_nsc[tii]  - stt[:p_nsc][tii][dev] <= tmp)
                    @constraint(model, stt[:p_nsc][tii][dev] - p_nsc[tii] <= tmp)
                    add_to_expression!(obj, tmp)
                end

                if dev ∉ upd[:p_rru_off][tii]
                    @constraint(model, p_rru_off[tii] == stt[:p_rru_off][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, p_rru_off[tii]  - stt[:p_rru_off][tii][dev] <= tmp)
                    @constraint(model, stt[:p_rru_off][tii][dev] - p_rru_off[tii]  <= tmp)
                    add_to_expression!(obj, tmp)
                end

                if dev ∉ upd[:q_qru][tii]
                    @constraint(model, q_qru[tii] == stt[:q_qru][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, q_qru[tii]  - stt[:q_qru][tii][dev] <= tmp)
                    @constraint(model, stt[:q_qru][tii][dev] - q_qru[tii]  <= tmp)
                    add_to_expression!(obj, tmp)
                end
                if dev ∉ upd[:q_qrd][tii]
                    @constraint(model, q_qrd[tii] == stt[:q_qrd][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, q_qrd[tii]  - stt[:q_qrd][tii][dev] <= tmp)
                    @constraint(model, stt[:q_qrd][tii][dev] - q_qrd[tii]  <= tmp)
                    add_to_expression!(obj, tmp)
                end

                # now, deal with reactive powers, some of which are specified with equality
                # only a subset of devices will have a reactive power equality constraint
                if dev ∉ idx.J_pqe

                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, dev_q[tii]  - stt[:dev_q][tii][dev] <= tmp)
                    @constraint(model, stt[:dev_q][tii][dev] - dev_q[tii]  <= tmp)
                    add_to_expression!(obj, tmp, qG.dev_q_projection_weight)
                end

                # and now the rest -- none of which are in fixed sets
                #
                # p_on
                tmp = @variable(model)
                @constraint(model, p_on[tii]  - stt[:p_on][tii][dev] <= tmp)
                @constraint(model, stt[:p_on][tii][dev] - p_on[tii]  <= tmp)
                add_to_expression!(obj, tmp, qG.p_on_projection_weight)
                
                # p_rgu 
                tmp = @variable(model)
                @constraint(model, p_rgu[tii]  - stt[:p_rgu][tii][dev] <= tmp)
                @constraint(model, stt[:p_rgu][tii][dev] - p_rgu[tii]  <= tmp)
                add_to_expression!(obj, tmp)
                
                # p_rgd
                tmp = @variable(model)
                @constraint(model, p_rgd[tii]  - stt[:p_rgd][tii][dev] <= tmp)
                @constraint(model, stt[:p_rgd][tii][dev] - p_rgd[tii]  <= tmp)
                add_to_expression!(obj, tmp)

                # p_scr
                tmp = @variable(model)
                @constraint(model, p_scr[tii]  - stt[:p_scr][tii][dev] <= tmp)
                @constraint(model, stt[:p_scr][tii][dev] - p_scr[tii]  <= tmp)
                add_to_expression!(obj, tmp)

                # p_rru_on
                tmp = @variable(model)
                @constraint(model, p_rru_on[tii]  - stt[:p_rru_on][tii][dev] <= tmp)
                @constraint(model, stt[:p_rru_on][tii][dev] - p_rru_on[tii]  <= tmp)
                add_to_expression!(obj, tmp)

                # p_rrd_on
                tmp = @variable(model)
                @constraint(model, p_rrd_on[tii]  - stt[:p_rrd_on][tii][dev] <= tmp)
                @constraint(model, stt[:p_rrd_on][tii][dev] - p_rrd_on[tii]  <= tmp)
                add_to_expression!(obj, tmp)
            end

            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # test solution!
            soln_valid = solution_status(model)

            # did Gurobi find something valid?
            if soln_valid == true

                # print
                if qG.print_projection_success == true
                    println("Projection for dev $(dev). ", termination_status(model),". objective value: ", objective_value(model))
                end
                # leave the loop
                solve_projection = false

                # return the solution
                for tii in prm.ts.time_keys

                    # copy the binary solution to a temporary location
                    stt[:u_on_dev_GRB][tii][dev]  = copy(round(value(u_on_dev[tii])))

                    # directly update the rest
                    stt[:p_on][tii][dev]      = copy(value(p_on[tii]))
                    stt[:dev_q][tii][dev]     = copy(value(dev_q[tii]))
                    stt[:p_rgu][tii][dev]     = copy(value(p_rgu[tii]))
                    stt[:p_rgd][tii][dev]     = copy(value(p_rgd[tii]))
                    stt[:p_scr][tii][dev]     = copy(value(p_scr[tii]))
                    stt[:p_nsc][tii][dev]     = copy(value(p_nsc[tii]))
                    stt[:p_rru_on][tii][dev]  = copy(value(p_rru_on[tii]))
                    stt[:p_rru_off][tii][dev] = copy(value(p_rru_off[tii]))
                    stt[:p_rrd_on][tii][dev]  = copy(value(p_rrd_on[tii]))
                    stt[:p_rrd_off][tii][dev] = copy(value(p_rrd_off[tii]))
                    stt[:q_qru][tii][dev]     = copy(value(q_qru[tii]))
                    stt[:q_qrd][tii][dev]     = copy(value(q_qrd[tii]))
                end
            else
                if first_solve == true
                    # try again
                    @warn "Gurobi MILP projection failed (dev ($dev)) -- trying with flat start!"
                    solve_projection = true
                    first_solve      = false
                else
                    # that's it -- all done
                    solve_projection = false

                    # warn!
                    @warn "Gurobi MILP projection failed (dev ($dev)) -- skip and try again later!"
                end
            end
        end
    end
end

# note -- this is ALWAYS run after solve_Gurobi_projection!()
function apply_Gurobi_projection_and_states!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # we only need to apply the updated binary variables (all of them)
    # after running the batch_fix function
    for tii in prm.ts.time_keys
        # copy the binary solution to a temporary location
        stt[:u_on_dev][tii] .= copy.(stt[:u_on_dev_GRB][tii])
    end

    # update the u_sum and powers (used in clipping, so must be correct!)
    qG.run_susd_updates = true
    quasiGrad.simple_device_statuses!(idx, prm, qG, stt)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    # reactive powers are all set
end

function project!(pct_round::Float64, idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}}; final_projection::Bool=false)
    # this function 1) projects, 2) batch fixes, and 3) applies the projection
    #
    # 1. solve the projection -- don't treat the final projection as special
    quasiGrad.solve_Gurobi_projection!(final_projection, idx, prm, qG, stt, sys, upd)

    # 2. fix binaries which are closest to their Gurobi solutions
    if final_projection == false
        # only batch fix if this is the last iteration
        quasiGrad.batch_fix!(pct_round, prm, stt, sys, upd)
    end

    # 3. update the state (i.e., apply the projection)
    quasiGrad.apply_Gurobi_projection_and_states!(idx, prm, qG, stt, sys)
end

function solve_Gurobi_projection!(final_projection::Bool, idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # loop over each device and solve individually -- not clear if this is faster
    # than solving one big optimization problem all at once. see legacy code for
    # a(n unfinished) version where all devices are solved at once!

    # status update
    @info "Running MILP projection across $(sys.ndev) devices."
    
    # loop over all devices
    # for dev in 1:sys.ndev
    Threads.@threads for dev in 1:sys.ndev
        # try projecting a second time if the first one fails!
        solve_projection = true
        first_solve      = true

        # loop twice, potentially
        while solve_projection == true
        
            # build an empty model!
            model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => 1))
            set_string_names_on_creation(model, false)
            # set_silent(model)

            # set model properties
            quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
            quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)
            quasiGrad.set_optimizer_attribute(model, "IntFeasTol",     qG.IntFeasTol)
            quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
            # quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), qG.FeasibilityTol)
            # quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), qG.FeasibilityTol)

            # define local time keys
            tkeys = prm.ts.time_keys

            # define the minimum set of variables we will need to solve the constraints  
            if first_solve == true
                # in this case, use a hot start***  
                if final_projection == true
                    u_on_dev = Dict(tkeys[ii] => round(stt[:u_on_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_su_dev = Dict(tkeys[ii] => round(stt[:u_su_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_sd_dev = Dict(tkeys[ii] => round(stt[:u_sd_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                else
                    # define binary on/su/sd states
                    u_on_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_on_dev_t$(ii)", 
                    u_su_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_su_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_su_dev_t$(ii)", 
                    u_sd_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_sd_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_sd_dev_t$(ii)",  
                end
                p_on      = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_on][tkeys[ii]][dev])                         for ii in 1:(sys.nT)) # => base_name = "p_on_t$(ii)",      
                dev_q     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:dev_q][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "dev_q_t$(ii)",     
                p_rgu     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rgu][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgu_t$(ii)",     
                p_rgd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rgd][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgd_t$(ii)",     
                p_scr     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_scr][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_scr_t$(ii)",     
                p_nsc     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_nsc][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_nsc_t$(ii)",     
                p_rru_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rru_on][tkeys[ii]][dev],  lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_on_t$(ii)",  
                p_rru_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rru_off][tkeys[ii]][dev], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_off_t$(ii)", 
                p_rrd_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rrd_on][tkeys[ii]][dev],  lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_on_t$(ii)",  
                p_rrd_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:p_rrd_off][tkeys[ii]][dev], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_off_t$(ii)", 
                q_qru     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:q_qru][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qru_t$(ii)",     
                q_qrd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:q_qrd][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qrd_t$(ii)",     
            else
                # in this case (i.e., we failed once), use a flat start! It seems to help :)    
                if final_projection == true
                    u_on_dev = Dict(tkeys[ii] => round(stt[:u_on_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_su_dev = Dict(tkeys[ii] => round(stt[:u_su_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                    u_sd_dev = Dict(tkeys[ii] => round(stt[:u_sd_dev][tkeys[ii]][dev]) for ii in 1:(sys.nT))
                else
                    # define binary on/su/sd states
                    u_on_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_on_dev_t$(ii)", 
                    u_su_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_su_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_su_dev_t$(ii)", 
                    u_sd_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_sd_dev][tkeys[ii]][dev],  binary=true) for ii in 1:(sys.nT)) # => base_name = "u_sd_dev_t$(ii)",  
                end
                p_on      = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_on_t$(ii)",      
                dev_q     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "dev_q_t$(ii)",     
                p_rgu     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgu_t$(ii)",     
                p_rgd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgd_t$(ii)",     
                p_scr     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_scr_t$(ii)",     
                p_nsc     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_nsc_t$(ii)",     
                p_rru_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_on_t$(ii)",  
                p_rru_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_off_t$(ii)", 
                p_rrd_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_on_t$(ii)",  
                p_rrd_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_off_t$(ii)", 
                q_qru     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qru_t$(ii)",     
                q_qrd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=0.0, lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qrd_t$(ii)",     
            end

            # we have the affine "AffExpr" expressions (whose values are specified)
            dev_p = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))
            p_su  = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))
            p_sd  = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))

            # == define active power constraints ==
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # first, get the startup power
                T_supc     = idx.Ts_supc[dev][t_ind]     # T_set, p_supc_set = get_supc(tii, dev, prm)
                p_supc_set = idx.ps_supc_set[dev][t_ind] # T_set, p_supc_set = get_supc(tii, dev, prm)
                add_to_expression!(p_su[tii], sum(p_supc_set[ii]*u_su_dev[tii_inst] for (ii,tii_inst) in enumerate(T_supc); init=0.0))

                # second, get the shutdown power
                T_sdpc     = idx.Ts_sdpc[dev][t_ind]     # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
                p_sdpc_set = idx.ps_sdpc_set[dev][t_ind] # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
                add_to_expression!(p_sd[tii], sum(p_sdpc_set[ii]*u_sd_dev[tii_inst] for (ii,tii_inst) in enumerate(T_sdpc); init=0.0))

                # finally, get the total power balance
                dev_p[tii] = p_on[tii] + p_su[tii] + p_sd[tii]
            end

            # == define reactive power constraints ==
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # only a subset of devices will have a reactive power equality constraint
                if dev in idx.J_pqe

                    # the following (pr vs cs) are equivalent
                    if dev in idx.pr_devs
                        # producer?
                        T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
                        T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm)
                        u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)
                        
                        # compute q -- this might be the only equality constraint (and below)
                        @constraint(model, dev_q[tii] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii])
                    else
                        # the device must be a consumer :)
                        T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                        T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                        u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                        # compute q -- this might be the only equality constraint (and above)
                        @constraint(model, dev_q[tii] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii])
                    end
                end
            end

            # loop over each time period and define the hard constraints
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # duration
                dt = prm.ts.duration[tii]

                # 1. Minimum downtime: zhat_mndn
                T_mndn = idx.Ts_mndn[dev][t_ind] # t_set = get_tmindn(tii, dev, prm)
                @constraint(model, u_su_dev[tii] + sum(u_sd_dev[tii_inst] for tii_inst in T_mndn; init=0.0) - 1.0 <= 0.0)

                # 2. Minimum uptime: zhat_mnup
                T_mnup = idx.Ts_mnup[dev][t_ind] # t_set = get_tminup(tii, dev, prm)
                @constraint(model, u_sd_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_mnup; init=0.0) - 1.0 <= 0.0)

                # define the previous power value (used by both up and down ramping!)
                if tii == :t1
                    # note: p0 = prm.dev.init_p[dev]
                    dev_p_previous = prm.dev.init_p[dev]
                else
                    # grab previous time
                    tii_m1 = prm.ts.time_keys[t_ind-1]
                    dev_p_previous = dev_p[tii_m1]
                end

                # 3. Ramping limits (up): zhat_rup
                @constraint(model, dev_p[tii] - dev_p_previous
                        - dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii] - u_su_dev[tii])
                        +     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii] + 1.0 - u_on_dev[tii])) <= 0.0)

                # 4. Ramping limits (down): zhat_rd
                @constraint(model,  dev_p_previous - dev_p[tii]
                        - dt*(prm.dev.p_ramp_down_ub[dev]*u_on_dev[tii]
                        +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-u_on_dev[tii])) <= 0.0)

                # 5. Regulation up: zhat_rgu
                @constraint(model, p_rgu[tii] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[tii] <= 0.0)

                # 6. Regulation down: zhat_rgd
                @constraint(model, p_rgd[tii] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[tii] <= 0.0)

                # 7. Synchronized reserve: zhat_scr
                @constraint(model, p_rgu[tii] + p_scr[tii] - prm.dev.p_syn_res_ub[dev]*u_on_dev[tii] <= 0.0)

                # 8. Synchronized reserve: zhat_nsc
                @constraint(model, p_nsc[tii] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[tii]) <= 0.0)

                # 9. Ramping reserve up (on): zhat_rruon
                @constraint(model, p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[tii] <= 0.0)

                # 10. Ramping reserve up (off): zhat_rruoff
                @constraint(model, p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0-u_on_dev[tii]) <= 0.0)
                
                # 11. Ramping reserve down (on): zhat_rrdon
                @constraint(model, p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[tii] <= 0.0)

                # 12. Ramping reserve down (off): zhat_rrdoff
                @constraint(model, p_rrd_off[tii] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[tii]) <= 0.0)
                
                # Now, we must separate: producers vs consumers
                if dev in idx.pr_devs
                    # 13p. Maximum reserve limits (producers): zhat_pmax
                    @constraint(model, p_on[tii] + p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0.0)
                
                    # 14p. Minimum reserve limits (producers): zhat_pmin
                    @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rrd_on[tii] + p_rgd[tii] - p_on[tii] <= 0.0)
                    
                    # 15p. Off reserve limits (producers): zhat_pmaxoff
                    @constraint(model, p_su[tii] + p_sd[tii] + p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0.0)

                    # get common "u_sum" terms that will be used in the subsequent four equations 
                    T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                    # 16p. Maximum reactive power reserves (producers): zhat_qmax
                    @constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

                    # 17p. Minimum reactive power reserves (producers): zhat_qmin
                    @constraint(model, q_qrd[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0.0)

                    # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                    if dev in idx.J_pqmax
                        @constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_0_ub[dev]*u_sum
                        - prm.dev.beta_ub[dev]*dev_p[tii] <= 0.0)
                    end 
                    
                    # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                    if dev in idx.J_pqmin
                        @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                        + prm.dev.beta_lb[dev]*dev_p[tii]
                        + q_qrd[tii] - dev_q[tii] <= 0.0)
                    end

                # consumers
                else  # => dev in idx.cs_devs
                    # 13c. Maximum reserve limits (consumers): zhat_pmax
                    @constraint(model, p_on[tii] + p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0.0)

                    # 14c. Minimum reserve limits (consumers): zhat_pmin
                    @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rru_on[tii] + p_scr[tii] + p_rgu[tii] - p_on[tii] <= 0.0)
                    
                    # 15c. Off reserve limits (consumers): zhat_pmaxoff
                    @constraint(model, p_su[tii] + p_sd[tii] + p_rrd_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0.0)

                    # get common "u_sum" terms that will be used in the subsequent four equations 
                    T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                    # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                    @constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

                    # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                    @constraint(model, q_qru[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0.0)
                    
                    # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                    if dev in idx.J_pqmax
                        @constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_0_ub[dev]*u_sum
                        - prm.dev.beta_ub[dev]*dev_p[tii] <= 0.0)
                    end 

                    # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                    if dev in idx.J_pqmin
                        @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                        + prm.dev.beta_lb[dev]*dev_p[tii]
                        + q_qru[tii] - dev_q[tii] <= 0.0)
                    end
                end
            end

            # misc penalty: maximum starts over multiple periods
            for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
                # get the time periods: zhat_mxst
                T_su_max = idx.Ts_su_max[dev][w_ind] #get_tsumax(w_params, prm)
                @constraint(model, sum(u_su_dev[tii] for tii in T_su_max; init=0.0) - w_params[3] <= 0.0)
            end

            # now, we need to add two other sorts of constraints:
            # 1. "evolutionary" constraints which link startup and shutdown variables
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                if tii == :t1
                    @constraint(model, u_on_dev[tii] - prm.dev.init_on_status[dev] == u_su_dev[tii] - u_sd_dev[tii])
                else
                    tii_m1 = prm.ts.time_keys[t_ind-1]
                    @constraint(model, u_on_dev[tii] - u_on_dev[tii_m1] == u_su_dev[tii] - u_sd_dev[tii])
                end
                # only one can be nonzero
                @constraint(model, u_su_dev[tii] + u_sd_dev[tii] <= 1)
            end

            # 2. constraints which hold constant variables from moving
                # a. must run
                # b. planned outages
                # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
                # d. other states which are fixed from previous IBR rounds
                #       note: all of these are relfected in "upd"
            # upd = update states
            #
            # note -- in this loop, we also build the objective function!
            # now, let's define an objective function and solve this mf.
            # our overall objective is to round and fix some subset of 
            # integer variables. Here is our approach: find a feasible
            # solution which is as close to our Adam solution as possible.
            # next, we process the results: we identify the x% of variables
            # which had to move "the least". We fix these values and remove
            # their associated indices from upd. the end.
            #
            # afterwards, we initialize adam with the closest feasible
            # solution variable values.
            obj = AffExpr(0.0)

            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # if a device is *not* in the set of variables,
                # then it must be held constant! -- otherwise, try to hold it
                # close to its initial value
                if dev ∉ upd[:u_on_dev][tii]
                    @constraint(model, u_on_dev[tii] == stt[:u_on_dev][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, u_on_dev[tii]  - stt[:u_on_dev][tii][dev] <= tmp)
                    @constraint(model, stt[:u_on_dev][tii][dev] - u_on_dev[tii]  <= tmp)
                    add_to_expression!(obj, tmp, qG.binary_projection_weight)
                end

                if dev ∉ upd[:p_rrd_off][tii]
                    @constraint(model, p_rrd_off[tii] == stt[:p_rrd_off][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, p_rrd_off[tii] - stt[:p_rrd_off][tii][dev] <= tmp)
                    @constraint(model, stt[:p_rrd_off][tii][dev] - p_rrd_off[tii] <= tmp)
                    add_to_expression!(obj, tmp)
                end

                if dev ∉ upd[:p_nsc][tii]
                    @constraint(model, p_nsc[tii] == stt[:p_nsc][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, p_nsc[tii]  - stt[:p_nsc][tii][dev] <= tmp)
                    @constraint(model, stt[:p_nsc][tii][dev] - p_nsc[tii] <= tmp)
                    add_to_expression!(obj, tmp)
                end

                if dev ∉ upd[:p_rru_off][tii]
                    @constraint(model, p_rru_off[tii] == stt[:p_rru_off][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, p_rru_off[tii]  - stt[:p_rru_off][tii][dev] <= tmp)
                    @constraint(model, stt[:p_rru_off][tii][dev] - p_rru_off[tii]  <= tmp)
                    add_to_expression!(obj, tmp)
                end

                if dev ∉ upd[:q_qru][tii]
                    @constraint(model, q_qru[tii] == stt[:q_qru][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, q_qru[tii]  - stt[:q_qru][tii][dev] <= tmp)
                    @constraint(model, stt[:q_qru][tii][dev] - q_qru[tii]  <= tmp)
                    add_to_expression!(obj, tmp)
                end
                if dev ∉ upd[:q_qrd][tii]
                    @constraint(model, q_qrd[tii] == stt[:q_qrd][tii][dev])
                else
                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, q_qrd[tii]  - stt[:q_qrd][tii][dev] <= tmp)
                    @constraint(model, stt[:q_qrd][tii][dev] - q_qrd[tii]  <= tmp)
                    add_to_expression!(obj, tmp)
                end

                # now, deal with reactive powers, some of which are specified with equality
                # only a subset of devices will have a reactive power equality constraint
                if dev ∉ idx.J_pqe

                    # add it to the objective function
                    tmp = @variable(model)
                    @constraint(model, dev_q[tii]  - stt[:dev_q][tii][dev] <= tmp)
                    @constraint(model, stt[:dev_q][tii][dev] - dev_q[tii]  <= tmp)
                    add_to_expression!(obj, tmp, qG.dev_q_projection_weight)
                end

                # and now the rest -- none of which are in fixed sets
                #
                # p_on
                tmp = @variable(model)
                @constraint(model, p_on[tii]  - stt[:p_on][tii][dev] <= tmp)
                @constraint(model, stt[:p_on][tii][dev] - p_on[tii]  <= tmp)
                add_to_expression!(obj, tmp, qG.p_on_projection_weight)
                
                # p_rgu 
                tmp = @variable(model)
                @constraint(model, p_rgu[tii]  - stt[:p_rgu][tii][dev] <= tmp)
                @constraint(model, stt[:p_rgu][tii][dev] - p_rgu[tii]  <= tmp)
                add_to_expression!(obj, tmp)
                
                # p_rgd
                tmp = @variable(model)
                @constraint(model, p_rgd[tii]  - stt[:p_rgd][tii][dev] <= tmp)
                @constraint(model, stt[:p_rgd][tii][dev] - p_rgd[tii]  <= tmp)
                add_to_expression!(obj, tmp)

                # p_scr
                tmp = @variable(model)
                @constraint(model, p_scr[tii]  - stt[:p_scr][tii][dev] <= tmp)
                @constraint(model, stt[:p_scr][tii][dev] - p_scr[tii]  <= tmp)
                add_to_expression!(obj, tmp)

                # p_rru_on
                tmp = @variable(model)
                @constraint(model, p_rru_on[tii]  - stt[:p_rru_on][tii][dev] <= tmp)
                @constraint(model, stt[:p_rru_on][tii][dev] - p_rru_on[tii]  <= tmp)
                add_to_expression!(obj, tmp)

                # p_rrd_on
                tmp = @variable(model)
                @constraint(model, p_rrd_on[tii]  - stt[:p_rrd_on][tii][dev] <= tmp)
                @constraint(model, stt[:p_rrd_on][tii][dev] - p_rrd_on[tii]  <= tmp)
                add_to_expression!(obj, tmp)
            end

            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # test solution!
            soln_valid = solution_status(model)

            # did Gurobi find something valid?
            if soln_valid == true

                # print
                if qG.print_projection_success == true
                    println("Projection for dev $(dev). ", termination_status(model),". objective value: ", objective_value(model))
                end
                # leave the loop
                solve_projection = false

                # return the solution
                for tii in prm.ts.time_keys

                    # copy the binary solution to a temporary location
                    stt[:u_on_dev_GRB][tii][dev]  = copy(round(value(u_on_dev[tii])))

                    # directly update the rest
                    stt[:p_on][tii][dev]      = copy(value(p_on[tii]))
                    stt[:dev_q][tii][dev]     = copy(value(dev_q[tii]))
                    stt[:p_rgu][tii][dev]     = copy(value(p_rgu[tii]))
                    stt[:p_rgd][tii][dev]     = copy(value(p_rgd[tii]))
                    stt[:p_scr][tii][dev]     = copy(value(p_scr[tii]))
                    stt[:p_nsc][tii][dev]     = copy(value(p_nsc[tii]))
                    stt[:p_rru_on][tii][dev]  = copy(value(p_rru_on[tii]))
                    stt[:p_rru_off][tii][dev] = copy(value(p_rru_off[tii]))
                    stt[:p_rrd_on][tii][dev]  = copy(value(p_rrd_on[tii]))
                    stt[:p_rrd_off][tii][dev] = copy(value(p_rrd_off[tii]))
                    stt[:q_qru][tii][dev]     = copy(value(q_qru[tii]))
                    stt[:q_qrd][tii][dev]     = copy(value(q_qrd[tii]))
                end
            else
                if first_solve == true
                    # try again
                    @warn "Gurobi MILP projection failed (dev ($dev)) -- trying with flat start!"
                    solve_projection = true
                    first_solve      = false
                else
                    # that's it -- all done
                    solve_projection = false

                    # warn!
                    @warn "Gurobi MILP projection failed (dev ($dev)) -- skip and try again later!"
                end
            end
        end
    end
end
