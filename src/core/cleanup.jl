# cleanup reserve variables, mostly
function reserve_cleanup!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # this is, necessarily, a centralized optimziation problem.
    #
    # build and empty the model!
    model = Model(Gurobi.Optimizer; add_bridges = false)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # set model properties
    set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)

    # define local time keys
    tkeys = prm.ts.time_keys

    # define the minimum set of variables we will need to solve the constraints
    p_rgu     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_rgu][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rgd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_rgd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_scr     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_scr][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_nsc     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_nsc][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rru_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_on_t$(ii)",  [dev = 1:sys.ndev], start=stt[:p_rru_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rru_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_off_t$(ii)", [dev = 1:sys.ndev], start=stt[:p_rru_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rrd_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_on_t$(ii)",  [dev = 1:sys.ndev], start=stt[:p_rrd_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rrd_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_off_t$(ii)", [dev = 1:sys.ndev], start=stt[:p_rrd_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT))
    q_qru     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qru_t$(ii)",     [dev = 1:sys.ndev], start=stt[:q_qru][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    q_qrd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qrd_t$(ii)",     [dev = 1:sys.ndev], start=stt[:q_qrd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))

    # add scoring variables and affine terms
    p_rgu_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgd_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_scr_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_nsc_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgu_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_scr_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_nsc_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    q_qru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qru_zonal_penalty_t$(ii)", [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT))
    q_qrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qrd_zonal_penalty_t$(ii)", [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT))
    
    # affine aggregation terms
    zt = AffExpr(0.0)

    # loop over all devices
    for dev in 1:sys.ndev

        # loop over each time period and define the hard constraints
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # duration
            dt = prm.ts.duration[tii]

            # 1. Minimum downtime: zhat_mndn
            # 2. Minimum uptime: zhat_mnup
            # 3. Ramping limits (up): zhat_rup
            # 4. Ramping limits (down): zhat_rd

            # 5. Regulation up: zhat_rgu
            @constraint(model, p_rgu[tii][dev] - prm.dev.p_reg_res_up_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[tii][dev] - prm.dev.p_reg_res_down_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] - prm.dev.p_syn_res_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt[:u_on_dev][tii][dev]) <= 0.0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, stt[:p_on][tii][dev] + p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= 0.0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rrd_on[tii][dev] + p_rgd[tii][dev] - stt[:p_on][tii][dev] <= 0.0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, stt[:dev_q][tii][dev] + q_qru[tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= 0.0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= 0.0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt[:dev_q][tii][dev] + q_qru[tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= 0.0)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev] + 
                        prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev] + 
                        q_qrd[tii][dev] - stt[:dev_q][tii][dev] <= 0.0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, stt[:p_on][tii][dev] + p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= 0.0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rru_on[tii][dev] + p_scr[tii][dev] + p_rgu[tii][dev] - stt[:p_on][tii][dev] <= 0.0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_rrd_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, stt[:dev_q][tii][dev] + q_qrd[tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= 0.0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= 0.0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt[:dev_q][tii][dev] + q_qrd[tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= 0.0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev]
                    + prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev]
                    + q_qru[tii][dev] - stt[:dev_q][tii][dev] <= 0.0)
                end
            end
        end

        # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "upd"
        # upd = update states
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # if a device is *not* in the set of variables, then it must be held constant!

            if dev ∉ upd[:p_rrd_off][tii]
                @constraint(model, p_rrd_off[tii][dev] == stt[:p_rrd_off][tii][dev])
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[tii][dev] == stt[:p_nsc][tii][dev])
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[tii][dev] == stt[:p_rru_off][tii][dev])
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[tii][dev] == stt[:q_qru][tii][dev])
            end

            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[tii][dev] == stt[:q_qrd][tii][dev])
            end
        end
    end

    # loop over reserves
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # for the "endogenous" reserve requirements
        rgu_sigma = prm.reserve.rgu_sigma
        rgd_sigma = prm.reserve.rgd_sigma 
        scr_sigma = prm.reserve.scr_sigma 
        nsc_sigma = prm.reserve.nsc_sigma  

        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == rgu_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == rgd_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, scr_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[tii][zone])
                @constraint(model, nsc_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[tii][zone])
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

                @constraint(model, prm.reserve.rru_min[zone][t_ind] <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] <= p_rrd_zonal_penalty[tii][zone])
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

                @constraint(model, prm.reserve.rru_min[zone][t_ind] -
                                sum(p_rru_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] -
                                sum(p_rrd_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[tii][zone])
            end
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_qzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] <= q_qrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] -
                                sum(q_qru[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] -
                                sum(q_qrd[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[tii][zone])
            end
        end
    end

    # loop -- NOTE -- we are not including start-up-state discounts -- not worth it
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]
        
        # add up
        zt_temp = 
            # local reserve penalties
            sum(dt*getindex.(prm.dev.p_reg_res_up_cost,t_ind).*p_rgu[tii]) -   # zrgu
            sum(dt*getindex.(prm.dev.p_reg_res_down_cost,t_ind).*p_rgd[tii]) - # zrgd
            sum(dt*getindex.(prm.dev.p_syn_res_cost,t_ind).*p_scr[tii]) -      # zscr
            sum(dt*getindex.(prm.dev.p_nsyn_res_cost,t_ind).*p_nsc[tii]) -     # znsc
            sum(dt*(getindex.(prm.dev.p_ramp_res_up_online_cost,t_ind).*p_rru_on[tii] +
                    getindex.(prm.dev.p_ramp_res_up_offline_cost,t_ind).*p_rru_off[tii])) -   # zrru
            sum(dt*(getindex.(prm.dev.p_ramp_res_down_online_cost,t_ind).*p_rrd_on[tii] +
                    getindex.(prm.dev.p_ramp_res_down_offline_cost,t_ind).*p_rrd_off[tii])) - # zrrd
            sum(dt*getindex.(prm.dev.q_res_up_cost,t_ind).*q_qru[tii]) -   # zqru      
            sum(dt*getindex.(prm.dev.q_res_down_cost,t_ind).*q_qrd[tii]) - # zqrd
            # zonal reserve penalties (P)
            sum(dt*prm.vio.rgu_zonal.*p_rgu_zonal_penalty[tii]) -
            sum(dt*prm.vio.rgd_zonal.*p_rgd_zonal_penalty[tii]) -
            sum(dt*prm.vio.scr_zonal.*p_scr_zonal_penalty[tii]) -
            sum(dt*prm.vio.nsc_zonal.*p_nsc_zonal_penalty[tii]) -
            sum(dt*prm.vio.rru_zonal.*p_rru_zonal_penalty[tii]) -
            sum(dt*prm.vio.rrd_zonal.*p_rrd_zonal_penalty[tii]) -
            # zonal reserve penalties (Q)
            sum(dt*prm.vio.qru_zonal.*q_qru_zonal_penalty[tii]) -
            sum(dt*prm.vio.qrd_zonal.*q_qrd_zonal_penalty[tii])

        # update zt
        add_to_expression!(zt, zt_temp)
    end

    # set the objective
    @objective(model, Max, zt)

    # solve
    optimize!(model)

    # test solution!
    soln_valid = solution_status(model)

    # did Gurobi find something valid?
    if soln_valid == true
        println("========================================================")
        println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
        println("========================================================")

        # solve, and then return the solution
        for tii in prm.ts.time_keys
            stt[:p_rgu][tii]     = copy(value.(p_rgu[tii]))
            stt[:p_rgd][tii]     = copy(value.(p_rgd[tii]))
            stt[:p_scr][tii]     = copy(value.(p_scr[tii]))
            stt[:p_nsc][tii]     = copy(value.(p_nsc[tii]))
            stt[:p_rru_on][tii]  = copy(value.(p_rru_on[tii]))
            stt[:p_rru_off][tii] = copy(value.(p_rru_off[tii]))
            stt[:p_rrd_on][tii]  = copy(value.(p_rrd_on[tii]))
            stt[:p_rrd_off][tii] = copy(value.(p_rrd_off[tii]))
            stt[:q_qru][tii]     = copy(value.(q_qru[tii]))
            stt[:q_qrd][tii]     = copy(value.(q_qrd[tii]))
        end
    else
        # warn!
        @warn "Reserve cleanup solver (LP) failed -- skip cleanup!"
    end
end