# useful
# command for running from the bash terminal:
    # julia-1.10 --threads 10 MyJulia_test.jl
    # julia-1.10 --threads 10 MyJulia1.jl

# struct_fields:
    fieldnames(typeof(input_struct))

@batch per=core for tii in prm.ts.time_keys
    ...
end

quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()

Threads.@threads for tii in prm.ts.time_keys
    ...
end

# in this file, we design the function which solves economic dispatch
function solve_parallel_economic_dispatch!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}; include_sus_in_ed::Bool=true)
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

        # now, we need to loop and set the affine expressions to 0
        #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
        for dev in prm.dev.dev_keys
            dev_p[dev]   = AffExpr(0.0)
            zen_dev[dev] = AffExpr(0.0)
        end

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
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == prm.reserve.rgu_sigma[zone]*sum(dev_p[dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == prm.reserve.rgd_sigma[zone]*sum(dev_p[dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, prm.reserve.scr_sigma[zone]*[dev_p[dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[tii][zone])
                @constraint(model, prm.reserve.nsc_sigma[zone]*[dev_p[dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[tii][zone])
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
                                sum(p_rgu[dev] for dev in idx.dev_pzone[zone]) <= p_rgu_zonal_penalty[tii][zone])

                @constraint(model, p_rgd_zonal_REQ[tii][zone] - 
                                sum(p_rgd[dev] for dev in idx.dev_pzone[zone]) <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] -
                                sum(p_rgu[dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[dev] for dev in idx.dev_pzone[zone]) <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] -
                                sum(p_rgu[dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[dev] for dev in idx.dev_pzone[zone]) - 
                                sum(p_nsc[dev] for dev in idx.dev_pzone[zone]) <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][tii] -
                                sum(p_rru_on[dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][tii] -
                                sum(p_rrd_on[dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[tii][zone])
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
                                sum(q_qru[dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][tii] -
                                sum(q_qrd[dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[tii][zone])
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

    # update the u_sum and powers (used in clipping, so must be correct!)
    qG.run_susd_updates = true
    quasiGrad.simple_device_statuses_and_transposition!(idx, prm, qG, stt)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

    # now, we may compute the actual score!
    scr[:ed_obj] = sum(stt.parallel_ed_obj)

    # print solution
    t_ed = time() - t_ed0
    println("Parallel ED finished. Objective value: ", scr[:ed_obj], ". Total time: $t_ed.")
end

