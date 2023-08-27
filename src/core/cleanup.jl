# cleanup reserve variables, mostly
function reserve_cleanup!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # time limit: not needed -- this is an LP
    # integer tolerance: not needed -- this is an LP
    # FeasibilityTol -- qG.FeasibilityTol
    # 
    # this is, necessarily, a centralized (across devices) optimziation problem.
    #
    # loop over each time period and define the hard constraints
        # => for tii in prm.ts.time_keys
    Threads.@threads for tii in prm.ts.time_keys

        # duration
        dt = prm.ts.duration[tii]

        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => 1))
        set_string_names_on_creation(model, false)

        # set model properties
        quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)

        # affine aggregation terms
        zt = AffExpr(0.0)

        # define the minimum set of variables we will need to solve the constraints
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

        # loop over all devices and apply constaints
        for dev in 1:sys.ndev
            # 1. Minimum downtime: zhat_mndn
            # 2. Minimum uptime: zhat_mnup
            # 3. Ramping limits (up): zhat_rup
            # 4. Ramping limits (down): zhat_rd

            # 5. Regulation up: zhat_rgu
            @constraint(model, p_rgu[dev] - prm.dev.p_reg_res_up_ub[dev]*stt.u_on_dev[tii][dev] <= 0.0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[dev] - prm.dev.p_reg_res_down_ub[dev]*stt.u_on_dev[tii][dev] <= 0.0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[dev] + p_scr[dev] - prm.dev.p_syn_res_ub[dev]*stt.u_on_dev[tii][dev] <= 0.0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt.u_on_dev[tii][dev]) <= 0.0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt.u_on_dev[tii][dev] <= 0.0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - stt.u_on_dev[tii][dev]) <= 0.0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt.u_on_dev[tii][dev] <= 0.0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt.u_on_dev[tii][dev]) <= 0.0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, stt.p_on[tii][dev] + p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ub[dev][tii]*stt.u_on_dev[tii][dev] <= 0.0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][tii]*stt.u_on_dev[tii][dev] + p_rrd_on[dev] + p_rgd[dev] - stt.p_on[tii][dev] <= 0.0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, stt.p_su[tii][dev] + stt.p_sd[tii][dev] + p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ub[dev][tii]*(1.0 - stt.u_on_dev[tii][dev]) <= 0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, stt.dev_q[tii][dev] + q_qru[dev] - prm.dev.q_ub[dev][tii]*stt.u_sum[tii][dev] <= 0.0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[dev] + prm.dev.q_lb[dev][tii]*stt.u_sum[tii][dev] - stt.dev_q[tii][dev] <= 0.0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt.dev_q[tii][dev] + q_qru[dev] - prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev]
                    - prm.dev.beta_ub[dev]*stt.dev_p[tii][dev] <= 0.0)
                end 

                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev] + 
                        prm.dev.beta_lb[dev]*stt.dev_p[tii][dev] + 
                        q_qrd[dev] - stt.dev_q[tii][dev] <= 0.0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, stt.p_on[tii][dev] + p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ub[dev][tii]*stt.u_on_dev[tii][dev] <= 0.0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][tii]*stt.u_on_dev[tii][dev] + p_rru_on[dev] + p_scr[dev] + p_rgu[dev] - stt.p_on[tii][dev] <= 0.0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, stt.p_su[tii][dev] + stt.p_sd[tii][dev] + p_rrd_off[dev] - prm.dev.p_ub[dev][tii]*(1.0 - stt.u_on_dev[tii][dev]) <= 0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, stt.dev_q[tii][dev] + q_qrd[dev] - prm.dev.q_ub[dev][tii]*stt.u_sum[tii][dev] <= 0.0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[dev] + prm.dev.q_lb[dev][tii]*stt.u_sum[tii][dev] - stt.dev_q[tii][dev] <= 0.0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt.dev_q[tii][dev] + q_qrd[dev] - prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev]
                    - prm.dev.beta_ub[dev]*stt.dev_p[tii][dev] <= 0.0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev]
                    + prm.dev.beta_lb[dev]*stt.dev_p[tii][dev]
                    + q_qru[dev] - stt.dev_q[tii][dev] <= 0.0)
                end
            end

            # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "upd"
            #
            # if a device is *not* in the set of variables, then it must be held constant!

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
        end

        # reserves
        #
        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[zone] == prm.reserve.rgu_sigma[zone]*sum(stt.dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[zone] == prm.reserve.rgd_sigma[zone]*sum(stt.dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[zone] == 0.0)
            else
                @constraint(model, prm.reserve.scr_sigma[zone]*[stt.dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[zone])
                @constraint(model, prm.reserve.nsc_sigma[zone]*[stt.dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[zone])
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

        # add up
        add_to_expression!(zt,
            # local reserve penalties
           -sum(dt.*prm.dev.p_reg_res_up_cost_tmdv[tii].*p_rgu) -   # zrgu
            sum(dt.*prm.dev.p_reg_res_down_cost_tmdv[tii].*p_rgd) - # zrgd
            sum(dt.*prm.dev.p_syn_res_cost_tmdv[tii].*p_scr) -      # zscr
            sum(dt.*prm.dev.p_nsyn_res_cost_tmdv[tii].*p_nsc) -     # znsc
            sum(dt.*(prm.dev.p_ramp_res_up_online_cost_tmdv[tii].*p_rru_on +
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[tii].*p_rru_off)) -   # zrru
            sum(dt.*(prm.dev.p_ramp_res_down_online_cost_tmdv[tii].*p_rrd_on +
                    prm.dev.p_ramp_res_down_offline_cost_tmdv[tii].*p_rrd_off)) - # zrrd
            sum(dt.*prm.dev.q_res_up_cost_tmdv[tii].*q_qru) -   # zqru      
            sum(dt.*prm.dev.q_res_down_cost_tmdv[tii].*q_qrd) - # zqrd
            # zonal reserve penalties (P)
            sum(dt.*prm.vio.rgu_zonal.*p_rgu_zonal_penalty) -
            sum(dt.*prm.vio.rgd_zonal.*p_rgd_zonal_penalty) -
            sum(dt.*prm.vio.scr_zonal.*p_scr_zonal_penalty) -
            sum(dt.*prm.vio.nsc_zonal.*p_nsc_zonal_penalty) -
            sum(dt.*prm.vio.rru_zonal.*p_rru_zonal_penalty) -
            sum(dt.*prm.vio.rrd_zonal.*p_rrd_zonal_penalty) -
            # zonal reserve penalties (Q)
            sum(dt.*prm.vio.qru_zonal.*q_qru_zonal_penalty) -
            sum(dt.*prm.vio.qrd_zonal.*q_qrd_zonal_penalty))

        # set the objective
        @objective(model, Max, zt)

        # solve
        optimize!(model)

        # test solution!
        soln_valid = solution_status(model)

        # did Gurobi find something valid?
        if soln_valid == true
            if qG.print_reserve_cleanup_success == true
                println("Reserve cleanup at $(tii). ", termination_status(model),". objective value: ", objective_value(model))
            end

            # return the solution
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
        else
            # warn!
            @warn "Reserve cleanup solver (LP) failed at $(tii) -- skip this cleanup!"
        end
    end
end

# cleanup reserve variables, mostly
function soft_reserve_cleanup!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # this is, necessarily, a centralized optimziation problem (over devices)
    #
    # build the model! default tolerances are fine, because this
    # is a penalized solution (not a final, feasible one)
    #
    # penalization constant -- don't set too small (1e5 seems fine)
    penalty_scalar = 1e4

    # loop over each time period and define the hard constraints
    # => for tii in prm.ts.time_keys
    Threads.@threads for tii in prm.ts.time_keys

        # duration
        dt = prm.ts.duration[tii]

        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => 1))
        set_string_names_on_creation(model, false)

        # set model properties -- no tolerance needed!
            # => quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)

        # affine aggregation terms
        zt        = AffExpr(0.0)
        z_penalty = AffExpr(0.0)

        # duration
        dt = prm.ts.duration[tii]
        
        # define the minimum set of variables we will need to solve the constraints
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
    
        # loop over all devices
        for dev in 1:sys.ndev

            # 1. Minimum downtime: zhat_mndn
            # 2. Minimum uptime: zhat_mnup
            # 3. Ramping limits (up): zhat_rup
            # 4. Ramping limits (down): zhat_rd

            # 5. Regulation up: zhat_rgu
            tmp_penalty_c5 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c5, penalty_scalar)
            @constraint(model, p_rgu[dev] - prm.dev.p_reg_res_up_ub[dev]*stt.u_on_dev[tii][dev] <= tmp_penalty_c5)

            # 6. Regulation down: zhat_rgd
            tmp_penalty_c6 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c6, penalty_scalar)
            @constraint(model, p_rgd[dev] - prm.dev.p_reg_res_down_ub[dev]*stt.u_on_dev[tii][dev] <= tmp_penalty_c6)

            # 7. Synchronized reserve: zhat_scr
            tmp_penalty_c7 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c7, penalty_scalar)
            @constraint(model, p_rgu[dev] + p_scr[dev] - prm.dev.p_syn_res_ub[dev]*stt.u_on_dev[tii][dev] <= tmp_penalty_c7)

            # 8. Synchronized reserve: zhat_nsc
            tmp_penalty_c8 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c8, penalty_scalar)
            @constraint(model, p_nsc[dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt.u_on_dev[tii][dev]) <= tmp_penalty_c8)

            # 9. Ramping reserve up (on): zhat_rruon
            tmp_penalty_c9 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c9, penalty_scalar)
            @constraint(model, p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt.u_on_dev[tii][dev] <= tmp_penalty_c9)

            # 10. Ramping reserve up (off): zhat_rruoff
            tmp_penalty_c10 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c10, penalty_scalar)
            @constraint(model, p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - stt.u_on_dev[tii][dev]) <= tmp_penalty_c10)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            tmp_penalty_c11 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c11, penalty_scalar)
            @constraint(model, p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt.u_on_dev[tii][dev] <= tmp_penalty_c11)

            # 12. Ramping reserve down (off): zhat_rrdoff
            tmp_penalty_c12 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c12, penalty_scalar)
            @constraint(model, p_rrd_off[dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt.u_on_dev[tii][dev]) <= tmp_penalty_c12)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                tmp_penalty_c13pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c13pr, penalty_scalar)
                @constraint(model, stt.p_on[tii][dev] + p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ub[dev][tii]*stt.u_on_dev[tii][dev] <= tmp_penalty_c13pr)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                tmp_penalty_c14pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c14pr, penalty_scalar)
                @constraint(model, prm.dev.p_lb[dev][tii]*stt.u_on_dev[tii][dev] + p_rrd_on[dev] + p_rgd[dev] - stt.p_on[tii][dev] <= tmp_penalty_c14pr)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                tmp_penalty_c15pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c15pr, penalty_scalar)
                @constraint(model, stt.p_su[tii][dev] + stt.p_sd[tii][dev] + p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ub[dev][tii]*(1.0 - stt.u_on_dev[tii][dev]) <= tmp_penalty_c15pr)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                tmp_penalty_c16pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c16pr, penalty_scalar)
                @constraint(model, stt.dev_q[tii][dev] + q_qru[dev] - prm.dev.q_ub[dev][tii]*stt.u_sum[tii][dev] <= tmp_penalty_c16pr)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                tmp_penalty_c17pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c17pr, penalty_scalar)
                @constraint(model, q_qrd[dev] + prm.dev.q_lb[dev][tii]*stt.u_sum[tii][dev] - stt.dev_q[tii][dev] <= tmp_penalty_c17pr)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    tmp_penalty_c18pr = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c18pr, penalty_scalar)
                    @constraint(model, stt.dev_q[tii][dev] + q_qru[dev] - prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev]
                    - prm.dev.beta_ub[dev]*stt.dev_p[tii][dev] <= tmp_penalty_c18pr)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    tmp_penalty_c19pr = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c19pr, penalty_scalar)
                    @constraint(model, prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev] + 
                        prm.dev.beta_lb[dev]*stt.dev_p[tii][dev] + 
                        q_qrd[dev] - stt.dev_q[tii][dev] <= tmp_penalty_c19pr)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                tmp_penalty_c13cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c13cs, penalty_scalar)
                @constraint(model, stt.p_on[tii][dev] + p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ub[dev][tii]*stt.u_on_dev[tii][dev] <= tmp_penalty_c13cs)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                tmp_penalty_c14cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c14cs, penalty_scalar)
                @constraint(model, prm.dev.p_lb[dev][tii]*stt.u_on_dev[tii][dev] + p_rru_on[dev] + p_scr[dev] + p_rgu[dev] - stt.p_on[tii][dev] <= tmp_penalty_c14cs)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                tmp_penalty_c15cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c15cs, penalty_scalar)
                @constraint(model, stt.p_su[tii][dev] + stt.p_sd[tii][dev] + p_rrd_off[dev] - prm.dev.p_ub[dev][tii]*(1.0 - stt.u_on_dev[tii][dev]) <= tmp_penalty_c15cs)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                tmp_penalty_c16cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c16cs, penalty_scalar)
                @constraint(model, stt.dev_q[tii][dev] + q_qrd[dev] - prm.dev.q_ub[dev][tii]*stt.u_sum[tii][dev] <= tmp_penalty_c16cs)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                tmp_penalty_c17cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c17cs, penalty_scalar)
                @constraint(model, q_qru[dev] + prm.dev.q_lb[dev][tii]*stt.u_sum[tii][dev] - stt.dev_q[tii][dev] <= tmp_penalty_c17cs)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    tmp_penalty_c18cs = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c18cs, penalty_scalar)
                    @constraint(model, stt.dev_q[tii][dev] + q_qrd[dev] - prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev]
                    - prm.dev.beta_ub[dev]*stt.dev_p[tii][dev] <= tmp_penalty_c18cs)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    tmp_penalty_c19cs = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c19cs, penalty_scalar)
                    @constraint(model, prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev]
                    + prm.dev.beta_lb[dev]*stt.dev_p[tii][dev]
                    + q_qru[dev] - stt.dev_q[tii][dev] <= tmp_penalty_c19cs)
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
            # if a device is *not* in the set of variables, then it must be held constant!
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
        end

        # loop over reserve zones
        #
        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[zone] == prm.reserve.rgu_sigma[zone]*sum(stt.dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[zone] == prm.reserve.rgd_sigma[zone]*sum(stt.dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[zone] == 0.0)
            else
                @constraint(model, prm.reserve.scr_sigma[zone]*[stt.dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[zone])
                @constraint(model, prm.reserve.nsc_sigma[zone]*[stt.dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[zone])
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

        # objective!
        #
        # add up
        add_to_expression!(zt,
            # local reserve penalties
            -sum(dt*prm.dev.p_reg_res_up_cost_tmdv[tii].*p_rgu) -   # zrgu
            sum(dt*prm.dev.p_reg_res_down_cost_tmdv[tii].*p_rgd) - # zrgd
            sum(dt*prm.dev.p_syn_res_cost_tmdv[tii].*p_scr) -      # zscr
            sum(dt*prm.dev.p_nsyn_res_cost_tmdv[tii].*p_nsc) -     # znsc
            sum(dt*(prm.dev.p_ramp_res_up_online_cost_tmdv[tii].*p_rru_on +
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[tii].*p_rru_off)) -   # zrru
            sum(dt*(prm.dev.p_ramp_res_down_online_cost_tmdv[tii].*p_rrd_on +
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

        # set the objective
        @objective(model, Max, zt - z_penalty)

        # solve
        optimize!(model)

        # test solution!
        soln_valid = solution_status(model)

         # did Gurobi find something valid?
        if soln_valid == true
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
        else
            # warn!
            @warn "(softly constrained) Reserve cleanup solver (LP) failed at $(tii) -- skip this cleanup!"
        end
    end
end

# cleanup power flow (to some degree of accuracy)
function single_shot_pf_cleanup!(idx::quasiGrad.Index, Jac::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, prm::quasiGrad.Param, qG::quasiGrad.QG,  stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # device p/q stay fixed -- just tune v, theta, and dc

    # build and empty the model!
    model = Model(Gurobi.Optimizer)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # set model properties
    quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
    # => quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), qG.FeasibilityTol)
    # => quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), qG.FeasibilityTol)

    @info "Running linearized power flow cleanup at $(tii)."

    # define the variables (single time index)
    @variable(model, x_in[1:(2*sys.nb - 1)])
    set_start_value.(x_in, [stt.vm[tii]; stt.va[tii][2:end]])

    # assign
    dvm  = x_in[1:sys.nb]
    dva  = x_in[(sys.nb+1):end]

    # note:
    # vm   = vm0   + dvm
    # va   = va0   + dva
    # pinj = pinj0 + dpinj
    # qinj = qinj0 + dqinj
    #
    # key equation:
    #                       dPQ .== Jac*dVT
    #                       dPQ + basePQ(v) = devicePQ
    #
    #                       Jac*dVT + basePQ(v) == devicePQ
    #
    # so, we don't actually need to model dPQ explicitly (cool)
    #
    # so, the optimizer asks, how shall we tune dVT in order to produce a power perurbation
    # which, when added to the base point, lives inside the feasible device region?
    #
    # based on the result, we only have to actually update the device set points on the very
    # last power flow iteration, where we have converged.

    # now, model all nodal injections from the device/dc line side, all put in nodal_p/q
    nodal_p = Vector{AffExpr}(undef, sys.nb)
    nodal_q = Vector{AffExpr}(undef, sys.nb)
    for bus in 1:sys.nb
        # now, we need to loop and set the affine expressions to 0, and then add powers
        #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
        nodal_p[bus] = AffExpr(0.0)
        nodal_q[bus] = AffExpr(0.0)
    end

    # create a flow variable for each dc line and sum these into the nodal vectors
    if sys.nldc == 0
        # nothing to see here
    else

        # define dc variables
        @variable(model, pdc_vars[1:sys.nldc])    # oriented so that fr = + !!
        @variable(model, qdc_fr_vars[1:sys.nldc])
        @variable(model, qdc_to_vars[1:sys.nldc])

        set_start_value.(pdc_vars, stt.dc_pfr[tii])
        set_start_value.(qdc_fr_vars, stt.dc_qfr[tii])
        set_start_value.(qdc_to_vars, stt.dc_qto[tii])

        # bound dc power
        @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
        @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
        @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

        # loop and add to the nodal injection vectors
        for dcl in 1:sys.nldc
            add_to_expression!(nodal_p[idx.dc_fr_bus[dcl]], -pdc_vars[dcl])
            add_to_expression!(nodal_p[idx.dc_to_bus[dcl]], +pdc_vars[dcl])
            add_to_expression!(nodal_q[idx.dc_fr_bus[dcl]], -qdc_fr_vars[dcl])
            add_to_expression!(nodal_q[idx.dc_to_bus[dcl]], -qdc_to_vars[dcl])
        end
    end

    # next, deal with devices
    # 
    for dev in 1:sys.ndev
        if dev in idx.pr_devs
            # producers
            add_to_expression!(nodal_p[idx.device_to_bus[dev]], stt.dev_p[tii][dev])
            add_to_expression!(nodal_q[idx.device_to_bus[dev]], stt.dev_q[tii][dev])
        else
            # consumers
            add_to_expression!(nodal_p[idx.device_to_bus[dev]], -stt.dev_p[tii][dev])
            add_to_expression!(nodal_q[idx.device_to_bus[dev]], -stt.dev_q[tii][dev])
        end
    end

    # bound system variables ==============================================
    #
    # bound variables -- voltage
    @constraint(model, prm.bus.vm_lb - stt.vm[tii] .<= dvm .<= prm.bus.vm_ub - stt.vm[tii])
    # alternative: => @constraint(model, prm.bus.vm_lb .<= stt.vm[tii] + dvm .<= prm.bus.vm_ub)

    # mapping
    JacP_noref = @view Jac[1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
    JacQ_noref = @view Jac[(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]

    # objective: find v, theta with minimal mismatch penalty
    obj    = AffExpr(0.0)
    tmp_vm = @variable(model)
    tmp_va = @variable(model)
    @variable(model, slack_p[1:sys.nb])
    @variable(model, slack_q[1:sys.nb])

    for bus in 1:sys.nb
        # penalize mismatch
        @constraint(model, JacP_noref[bus,:]'*x_in + stt.pinj0[tii][bus] - nodal_p[bus] <= slack_p[bus])
        @constraint(model, nodal_p[bus] - JacP_noref[bus,:]'*x_in - stt.pinj0[tii][bus] <= slack_p[bus])
        @constraint(model, JacQ_noref[bus,:]'*x_in + stt.qinj0[tii][bus] - nodal_q[bus] <= slack_q[bus])
        @constraint(model, nodal_q[bus] - JacQ_noref[bus,:]'*x_in - stt.qinj0[tii][bus] <= slack_q[bus])

        # add both to the objective
        add_to_expression!(obj, slack_p[bus], 1e3)
        add_to_expression!(obj, slack_q[bus], 1e3)

        # voltage regularization
        @constraint(model, -dvm[bus] <= tmp_vm)
        @constraint(model,  dvm[bus] <= tmp_vm)

        # phase regularization
        if bus > 1
            @constraint(model, -dva[bus-1] <= tmp_va)
            @constraint(model,  dva[bus-1] <= tmp_va)
        end
    end

    # this adds light regularization and causes convergence
    add_to_expression!(obj, tmp_vm)
    add_to_expression!(obj, tmp_va)

    # set the objective
    @objective(model, Min, obj)

    # solve
    optimize!(model)

    # test solution!
    soln_valid = solution_status(model)

    # test validity
    if soln_valid == true
        # we update the voltage soluion
        stt.vm[tii]        .= stt.vm[tii]        .+ value.(dvm)
        stt.va[tii][2:end] .= stt.va[tii][2:end] .+ value.(dva)

        # update dc
        if sys.nldc > 0
            stt.dc_pfr[tii] .=  value.(pdc_vars)
            stt.dc_pto[tii] .= -value.(pdc_vars)  # also, performed in clipping
            stt.dc_qfr[tii] .= value.(qdc_fr_vars)
            stt.dc_qto[tii] .= value.(qdc_to_vars)
        end

        # take the norm of dv
        max_dx = maximum(abs.(value.(x_in)))
        if qG.print_linear_pf_iterations == true
            println("Single shot at time $(tii): ",primal_status(model),". objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
        end
    else
        # the solution is NOT valid
        @warn "Single shot cleanup failed at $(tii)! Skipping it."
    end
end

function cleanup_constrained_pf_with_Gurobi!(    
    cgd::quasiGrad.ConstantGrad, 
    ctg::quasiGrad.Contingency,
    flw::quasiGrad.Flow, 
    grd::quasiGrad.Grad, 
    idx::quasiGrad.Index, 
    mgd::quasiGrad.MasterGrad,
    ntk::quasiGrad.Network, 
    prm::quasiGrad.Param, 
    qG::quasiGrad.QG, 
    scr::Dict{Symbol, Float64}, 
    stt::quasiGrad.State,
    sys::quasiGrad.System, 
    upd::Dict{Symbol, Vector{Vector{Int64}}})
    
    # **NOTE**: this is necessarily a *serial* solver -- each time period is linked
    qG.skip_ctg_eval = true
    qG.eval_grad     = false
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    qG.skip_ctg_eval = false
    qG.eval_grad     = true

    # ask Gurobi to solve a linearize power flow. two options here:
    #   1) define device variables which are bounded, and then insert them into the power balance expression
    #   2) just define power balance bounds based on device characteristics, and then, at the end, optimally
    #      dispatch the devices via updating.
    # 
    # I am pretty sure 1) is slower, but it is (1) safer, (2) simpler, and (3) more flexible -- and it will
    # require less trouble-shooting. Plus, it will be easy to update/improve in the future! Go with it.
    #
    # here is power balance:
    #
    # p_pr - p_cs - pdc = p_lines/xfm/shunt => this is typical.
    #
    # vm0 = stt.vm[tii]
    # va0 = stt.va[tii][2:end-1]
    #
    # bias point: stt[:pinj0, qinj0] === Y * stt[:vm, va]
    qG.compute_pf_injs_with_Jac = true

    # build and empty the model!
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => qG.num_threads))
    set_string_names_on_creation(model, false)

    # set model properties
    quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
    # => quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), qG.FeasibilityTol)
    # => quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), qG.FeasibilityTol)

    @info "Running constrained, linearized power flow cleanup across $(sys.nT) time periods."

    # defined a "clean_up_failed" vector -- if any is true (1), then we must re-run
    clean_up_failed = zeros(sys.nT)

    # loop over time
    for tii in prm.ts.time_keys

        # duration
        dt = prm.ts.duration[tii]

        # initialize
        run_pf = true
        pf_cnt = 0

        # 1. update the ideal dispatch point (active power) -- we do this just once
            # => quasiGrad.ideal_dispatch!(idx, stt, sys, tii)
            # this is no longer needed, because we penalize device injections directly

        # 2. update y_bus and Jacobian and bias point -- this
        #    only needs to be done once per time, since xfm/shunt
        #    values are not changing between iterations
        quasiGrad.update_Ybus!(idx, ntk, prm, stt, sys, tii)

        # loop over pf solves
        while run_pf == true

            # increment
            pf_cnt += 1

            # first, rebuild the jacobian, and update the base points: stt.pinj0, stt.qinj0
            quasiGrad.build_Jac_and_pq0!(ntk, qG, stt, sys, tii)
            
            # empty model
            empty!(model)

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])

            # assign
            dvm = x_in[1:sys.nb]
            dva = x_in[(sys.nb+1):end]
            set_start_value.(dvm, stt.vm[tii])
            set_start_value.(dva, @view stt.va[tii][2:end])
            # note:
            # vm   = vm0   + dvm
            # va   = va0   + dva
            # pinj = pinj0 + dpinj
            # qinj = qinj0 + dqinj
            #
            # key equation:
            #                       dPQ .== Jac*dVT
            #                       dPQ + basePQ(v) = devicePQ
            #
            #                       Jac*dVT + basePQ(v) == devicePQ
            #
            # so, we don't actually need to model dPQ explicitly (cool)

            # now, model all nodal injections from the device/dc line side, all put in nodal_p/q
            nodal_p = Vector{AffExpr}(undef, sys.nb)
            nodal_q = Vector{AffExpr}(undef, sys.nb)
            for bus in 1:sys.nb
                # now, we need to loop and set the affine expressions to 0, and then add powers
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                nodal_p[bus] = AffExpr(0.0)
                nodal_q[bus] = AffExpr(0.0)
            end

            # create a flow variable for each dc line and sum these into the nodal vectors
            if sys.nldc == 0
                # nothing to see here
            else
                # define dc variables
                @variable(model, pdc_vars[1:sys.nldc])    # oriented so that fr = + !!
                @variable(model, qdc_fr_vars[1:sys.nldc])
                @variable(model, qdc_to_vars[1:sys.nldc])

                set_start_value.(pdc_vars, stt.dc_pfr[tii])
                set_start_value.(qdc_fr_vars, stt.dc_qfr[tii])
                set_start_value.(qdc_to_vars, stt.dc_qto[tii])

                # bound dc power
                @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
                @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
                @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

                # loop and add to the nodal injection vectors
                for dcl in 1:sys.nldc
                    add_to_expression!(nodal_p[idx.dc_fr_bus[dcl]], -pdc_vars[dcl])
                    add_to_expression!(nodal_p[idx.dc_to_bus[dcl]], +pdc_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_fr_bus[dcl]], -qdc_fr_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_to_bus[dcl]], -qdc_to_vars[dcl])
                end
            end
            
            # next, deal with devices
            @variable(model, dev_p_vars[1:sys.ndev])
            @variable(model, dev_q_vars[1:sys.ndev])
            set_start_value.(dev_p_vars, stt.dev_p[tii])
            set_start_value.(dev_q_vars, stt.dev_q[tii])

            # define p_on at this time
            # => p_on = dev_p_vars - stt.p_su[tii] - stt.p_sd[tii]
            p_on = Vector{AffExpr}(undef, sys.ndev)
            for dev in 1:sys.ndev
                # now, we need to loop and set the affine expressions to 0, and then add powers
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                p_on[dev] = AffExpr(0.0)
                add_to_expression!(p_on[dev], dev_p_vars[dev])
                add_to_expression!(p_on[dev], -stt.p_su[tii][dev] - stt.p_sd[tii][dev])
            end

            # constraints: 7 types
            #
            # define the previous power value (used by both up and down ramping!)
            if tii == 1
                # note: p0 = prm.dev.init_p[dev]
                dev_p_previous = prm.dev.init_p
            else
                # grab previous time
                tii_m1 = prm.ts.time_keys[tii-1]
                dev_p_previous = stt.dev_p[tii_m1]
            end

            # 1. ramp up
            @constraint(model, dev_p_vars .- dev_p_previous .- dt.*(prm.dev.p_ramp_up_ub.*(stt.u_on_dev[tii] .- stt.u_su_dev[tii]) .+ prm.dev.p_startup_ramp_ub.*(stt.u_su_dev[tii] .+ 1.0 .- stt.u_on_dev[tii])) .<= 0.0)

            # 2. ramp down
            @constraint(model, dev_p_previous .- dev_p_vars .- dt.*(prm.dev.p_ramp_down_ub.*stt.u_on_dev[tii] .+ prm.dev.p_shutdown_ramp_ub.*(1.0 .- stt.u_on_dev[tii])) .<= 0.0)

            # 3. pmax
            @constraint(model, p_on .<= prm.dev.p_ub_tmdv[tii].*stt.u_on_dev[tii])

            # 4. pmin
            @constraint(model, prm.dev.p_lb_tmdv[tii].*stt.u_on_dev[tii] .<= p_on)

            # 5. qmax
            @constraint(model, dev_q_vars .<= prm.dev.q_ub_tmdv[tii].*stt.u_sum[tii])

            # 6. qmin
            @constraint(model, prm.dev.q_lb_tmdv[tii].*stt.u_sum[tii] .<= dev_q_vars)
            
            # 7. additional reactive power constraints!
            #
            # ~~~ J_pqe, and J_pqmin/max ~~~
            #
            # apply additional bounds: J_pqe (equality constraints)
            for dev in idx.J_pqe
                @constraint(model, dev_q_vars[dev] - prm.dev.beta[dev]*dev_p_vars[dev] == prm.dev.q_0[dev]*stt.u_sum[tii][dev])
                # alternative: @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt.u_sum[tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe])
            end

            # apply additional bounds: J_pqmin/max (inequality constraints)
            #
            # note: when the reserve products are negelected, pr and cs constraints are the same
            #   remember: idx.J_pqmax == idx.J_pqmin
            for dev in idx.J_pqmax
                @constraint(model, dev_q_vars[dev] <= prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev] + prm.dev.beta_ub[dev]*dev_p_vars[dev])
                @constraint(model, prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev] + prm.dev.beta_lb[dev]*dev_p_vars[dev] <= dev_q_vars[dev])
            end

            # great, now just update the nodal injection vectors
            for dev in 1:sys.ndev
                if dev in idx.pr_devs
                    # producers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], dev_p_vars[dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], dev_q_vars[dev])
                else
                    # consumers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], -dev_p_vars[dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], -dev_q_vars[dev])
                end
            end

            # **forward looking constraints**
            if tii != prm.ts.time_keys[end]
                # in this case, we want to make sure that whatever power values are chosen
                # can also be used in the satisfaction of the next time period constraints
                tii_p1    = prm.ts.time_keys[tii+1]
                dt_future = prm.ts.duration[tii_p1]

                @variable(model, dev_p_vars_future[1:sys.ndev])
                @variable(model, dev_q_vars_future[1:sys.ndev])
                set_start_value.(dev_p_vars_future, stt.dev_p[tii_p1])
                set_start_value.(dev_q_vars_future, stt.dev_q[tii_p1])

                # define p_on_future
                p_on_future = Vector{AffExpr}(undef, sys.ndev)
                for dev in 1:sys.ndev
                    p_on_future[dev] = AffExpr(0.0)
                    add_to_expression!(p_on_future[dev], dev_p_vars_future[dev])
                    add_to_expression!(p_on_future[dev], -stt.p_su[tii_p1][dev] - stt.p_sd[tii_p1][dev])
                end

                # mapping:
                    # dt                =>    dt_future
                    # tii               =>    tii_p1
                    # p_on              =>    p_on_future
                    # dev_p_vars        =>    dev_p_vars_future
                    # dev_q_vars        =>    dev_q_vars_future
                    # dev_p_previous    =>    dev_p_vars
                    # dev_q_previous    =>    dev_q_vars

                # 1. ramp up
                @constraint(model, dev_p_vars_future .- dev_p_vars .- dt_future.*(prm.dev.p_ramp_up_ub.*(stt.u_on_dev[tii_p1] .- stt.u_su_dev[tii_p1]) .+ prm.dev.p_startup_ramp_ub.*(stt.u_su_dev[tii_p1] .+ 1.0 .- stt.u_on_dev[tii_p1])) .<= 0.0)
                # 2. ramp down
                @constraint(model, dev_p_vars .- dev_p_vars_future .- dt_future.*(prm.dev.p_ramp_down_ub.*stt.u_on_dev[tii_p1] .+ prm.dev.p_shutdown_ramp_ub.*(1.0 .- stt.u_on_dev[tii_p1])) .<= 0.0)
                # 3. pmax
                @constraint(model, p_on_future .<= prm.dev.p_ub_tmdv[tii_p1].*stt.u_on_dev[tii_p1])
                # 4. pmin
                @constraint(model, prm.dev.p_lb_tmdv[tii_p1].*stt.u_on_dev[tii_p1] .<= p_on_future)
                # 5. qmax
                @constraint(model, dev_q_vars_future .<= prm.dev.q_ub_tmdv[tii_p1].*stt.u_sum[tii_p1])
                # 6. qmin
                @constraint(model, prm.dev.q_lb_tmdv[tii_p1].*stt.u_sum[tii_p1] .<= dev_q_vars_future)
                # 7. apply additional bounds: J_pqe (equality constraints)
                for dev in idx.J_pqe
                    @constraint(model, dev_q_vars_future[dev] - prm.dev.beta[dev]*dev_p_vars_future[dev] == prm.dev.q_0[dev]*stt.u_sum[tii_p1][dev])
                end
                # apply additional bounds: J_pqmin/max (inequality constraints)
                for dev in idx.J_pqmax
                    @constraint(model, dev_q_vars_future[dev] <= prm.dev.q_0_ub[dev]*stt.u_sum[tii_p1][dev] + prm.dev.beta_ub[dev]*dev_p_vars_future[dev])
                    @constraint(model, prm.dev.q_0_lb[dev]*stt.u_sum[tii_p1][dev] + prm.dev.beta_lb[dev]*dev_p_vars_future[dev] <= dev_p_vars_future[dev])
                end
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, prm.bus.vm_lb .- stt.vm[tii] .<= dvm .<= prm.bus.vm_ub .- stt.vm[tii])
            # alternative: => @constraint(model, prm.bus.vm_lb .<= stt.vm[tii] + dvm .<= prm.bus.vm_ub)

            # mapping
            JacP_noref = ntk.Jac[tii][1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
            JacQ_noref = ntk.Jac[tii][(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]
            println("off!!!")
            @constraint(model, JacP_noref*x_in .+ stt.pinj0[tii] .== nodal_p)
            @constraint(model, JacQ_noref*x_in .+ stt.qinj0[tii] .== nodal_q)

            # objective: hold p and q close to their initial values (does not converge without v,a regularization)
                # => || stt.pinj_ideal - (p0 + dp) || + regularization
            obj = @expression(model,
                x_in'*x_in +
                (stt.pinj0[tii] .- nodal_p)'*(stt.pinj0[tii] .- nodal_p) + 
                (stt.qinj0[tii] .- nodal_q)'*(stt.qinj0[tii] .- nodal_q) + 
                5.0*(stt.dev_q[tii] .- dev_q_vars)'*(stt.dev_q[tii] .- dev_q_vars) + 
                25.0*(stt.dev_p[tii] .- dev_p_vars)'*(stt.dev_p[tii] .- dev_p_vars))

            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # test solution!
            soln_valid = solution_status(model)

            # test validity
            if soln_valid == true
                # no matter what, we update the voltage soluion
                stt.vm[tii]        .= stt.vm[tii]        .+ value.(dvm)
                stt.va[tii][2:end] .= stt.va[tii][2:end] .+ value.(dva)

                # now, apply the updated injections to the devices
                stt.dev_p[tii] .= value.(dev_p_vars)
                stt.p_on[tii]  .= stt.dev_p[tii] .- stt.p_su[tii] .- stt.p_sd[tii]
                stt.dev_q[tii] .= value.(dev_q_vars)
                if sys.nldc > 0
                    stt.dc_pfr[tii] .=  value.(pdc_vars)
                    stt.dc_pto[tii] .= .-value.(pdc_vars)  # also, performed in clipping
                    stt.dc_qfr[tii] .= value.(qdc_fr_vars)
                    stt.dc_qto[tii] .= value.(qdc_to_vars)
                end

                # take the norm of dv
                max_dx = maximum(abs.(value.(x_in)))
                if qG.print_linear_pf_iterations == true
                    println(termination_status(model),". time: $(tii). objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
                end
                #
                # shall we terminate?
                run_pf = false
                if (max_dx < qG.max_pf_dx_final_solve) || (pf_cnt == qG.max_linear_pfs_final_solve)
                    run_pf = false
                end
            else
                # the solution is NOT valid, so we should increase bounds and try again
                @warn "Constrained power flow cleanup failed at $(tii)! What a shame.."
                clean_up_failed[tii] = 1.0
                # => quasiGrad.single_shot_pf_cleanup!(idx, Jac, prm, qG, stt, sys, tii)

                # all done with this time period -- move on
                run_pf = false
            end
        end
    end

    # was there a failure? If so, re-project
    if sum(clean_up_failed) > 0.0
        quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    end
end