using quasiGrad
using GLMakie
using Revise
using Plots
using Makie

# call the plotting tools
# include("../core/plotting.jl")

# %% ===============
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D2/C3S1N00600/scenario_001.json"

path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E1_20230214/D1/C3E1N01576D1/scenario_117.json"

InFile1               = path
TimeLimitInSeconds    = 1500.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# this is the master function which executes quasiGrad.
# 
#
# =====================================================\\
# TT: start time
start_time = time()

# I1. load the system data
jsn = quasiGrad.load_json(InFile1)

# I2. initialize the system
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

qG.apply_grad_weight_homotopy = false

# I3. run an economic dispatch and update the states
quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd, final_projection = true)

quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# %% 
qG.max_pf_dx = 1e-2
quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# %%
qG.IntFeasTol = 1e-9
qG.FeasibilityTol = 1e-9
qG.time_lim = 15.0
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd, final_projection = true)

quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)


# %%
dev = 130
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    # duration
    dt = prm.ts.duration[tii]
    
    # define the previous power value (used by both up and down ramping!)
    if tii == :t1
        # note: p0 = prm.dev.init_p[dev]
        dev_p_previous = prm.dev.init_p[dev]
    else
        # grab previous time
        dev_p_previous = stt[:dev_p][prm.ts.tmin1[tii]][dev] 
    end

    println(stt[:dev_p][tii][dev] - dev_p_previous
    - dt*(prm.dev.p_ramp_up_ub[dev]     *(stt[:u_on_dev][tii][dev] - stt[:u_su_dev][tii][dev])
    +     prm.dev.p_startup_ramp_ub[dev]*(stt[:u_su_dev][tii][dev] + 1.0 - stt[:u_on_dev][tii][dev])))

    #println(            max(dev_p_previous - stt[:dev_p][tii][dev]
    #- dt*(prm.dev.p_ramp_down_ub[dev]*stt[:u_on_dev][tii][dev]
    #+     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-stt[:u_on_dev][tii][dev])),0.0))

end

# %%
dev = 130

model = Model(Gurobi.Optimizer; add_bridges = false)
set_string_names_on_creation(model, false)
set_silent(model)

# empty the model!
empty!(model)

# quiet down!!!
# set_silent(model)
# alternative => quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

# set model properties
#quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
quasiGrad.set_optimizer_attribute(model, "IntFeasTol",     qG.IntFeasTol)
#quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
#quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)

# MOI tolerances
quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), 1e-3)
quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), 1e-3)

# define local time keys
tkeys = prm.ts.time_keys

# define the minimum set of variables we will need to solve the constraints                                                       -- round() the int?
u_on_dev  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true)       for ii in 1:(sys.nT)) # => base_name = "u_on_dev_t$(ii)",  
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

# add a few more (implicit) variables which are necessary for solving this system
u_su_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_su_dev][tkeys[ii]][dev], binary=true) for ii in 1:(sys.nT)) # => base_name = "u_su_dev_t$(ii)", 
u_sd_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_sd_dev][tkeys[ii]][dev], binary=true) for ii in 1:(sys.nT)) # => base_name = "u_sd_dev_t$(ii)", 

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
        u_sum     = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

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

quasiGrad.solution_status(model)
# %%


if tii == :t1
    # note: p0 = prm.dev.init_p[dev]
    dev_p_previous = prm.dev.init_p[dev]
else
    # grab previous time
    tii_m1 = prm.ts.time_keys[t_ind-1]
    dev_p_previous = dev_p[tii_m1]
end

value(    dev_p[tii] - dev_p_previous
- dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii] - u_su_dev[tii])
+     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii] + 1.0 - u_on_dev[tii])))

# %%
@btime quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
sum(sum(stt[:zsus_dev][tii]    for tii in prm.ts.time_keys))

# %% timing
qG.eval_grad = false

@btime sus!(grd, idx, mgd, prm, qG, stt, sys)

# %%

function sus!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # loop over each time period
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        for dev in 1:sys.ndev
            # first, we bound ("bnd") the startup state ("sus"):
            # the startup state can only be active if the device
            # has been on within some recent time period.
            #
            # flush the sus
            stt[:zsus_dev][tii][dev] = 0.0

            # loop over sus (i.e., f in F)
            for ii in 1:prm.dev.num_sus[dev]
                # grab the sets of T_sus
                T_sus_jft = idx.Ts_sus_jft[dev][t_ind][ii] # T_sus_jft, T_sus_jf = get_tsus_sets(tii, dev, prm, ii)
                T_sus_jf  = idx.Ts_sus_jf[dev][t_ind][ii]  # T_sus_jft, T_sus_jf = get_tsus_sets(tii, dev, prm, ii)

                if tii in T_sus_jf
                    if tii == :t1
                        # this is an edge case, where there are no previous states which
                        # could be "on" (since we can't turn on the generator in the fixed
                        # past, and it wasn't on)
                        # ** stt[:u_sus_bnd][tii][dev][ii] = 0.0
                        u_sus_bnd = 0.0
                    else
                        u_on_max_ind = argmax([stt[:u_on_dev][tii_inst][dev] for tii_inst in T_sus_jft])
                        u_sus_bnd    = stt[:u_on_dev][T_sus_jft[u_on_max_ind]][dev]
                        #u_sus_bnd = maximum([stt[:u_on_dev][tii_inst][dev] for tii_inst in T_sus_jft])
                        # ** stt[:u_sus_bnd][tii][dev][ii] = stt[:u_on_dev][T_sus_jft[u_on_max_ind]][dev]
                    end
                    #
                    # note: u_on_max == stt[:u_on_dev][T_sus_jft[u_on_max_ind]][dev]
                    #
                    # previous bound based on directly taking the max:
                        # stt[:u_sus_bnd][tii][dev][ii] = max.([stt[:u_on_dev][tii_inst][dev] for tii_inst in T_sus_jft])
                    # previous bound based on the sum (rather than max)
                        # stt[:u_sus_bnd][tii][dev][ii] = max.(sum(stt[:u_on_dev][tii_inst][dev] for tii_inst in T_sus_jft; init=0.0), 1.0)
                else
                    # ok, in this case the device was on in a sufficiently recent time (based on
                    # startup conditions), so we don't need to compute a bound
                    u_sus_bnd = 1.0
                    # ** stt[:u_sus_bnd][tii][dev][ii] = 1.0
                end

                # now, compute the discount/cost ==> this is "+=", since it is over all (f in F) states
                if u_sus_bnd > 0.0
                    stt[:zsus_dev][tii][dev] += prm.dev.startup_states[dev][ii][1]*min(stt[:u_su_dev][tii][dev],u_sus_bnd)
                end
                # ** stt[:zsus_dev][tii][dev] += prm.dev.startup_states[dev][ii][1]*min(stt[:u_su_dev][tii][dev],stt[:u_sus_bnd][tii][dev][ii])

                # this is all pretty expensive, so let's take the gradient right here
                #
                # evaluate gradient?
                if qG.eval_grad
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsus_dev] * prm.dev.startup_states[dev][ii][1]
                    gc = prm.dev.startup_states[dev][ii][1]

                    # test which was smaller: u_su, or the su_bound?
                    #
                    # we want "<=" so that we never end up in a case where 
                    # we try to take the gradient of u_sus_bnd == 1 (see else case above)
                    if stt[:u_su_dev][tii][dev] <= u_sus_bnd # ** stt[:u_sus_bnd][tii][dev][ii]
                        # in this case, there is an available discount, so we want u_su
                        # to feel a bit less downward pressure and rise up (potentially)
                        mgd[:u_on_dev][tii][dev] += gc*grd[:u_su_dev][:u_on_dev][tii][dev]
                        if tii != :t1
                            # previous time?
                            mgd[:u_on_dev][prm.ts.tmin1[tii]][dev] += gc*grd[:u_su_dev][:u_on_dev_prev][tii][dev]
                        end
                    else
                        # in this case, sus bound is lower than u_su, so we'll put some pressure on the
                        # previous largest u_on, trying to push it up, in order to extract a little value
                        # from this sus.. :)
                        #
                        # what time is associated with this derivative? it is the time associated with the max u_on
                        if tii != :t1
                            # skip the gradient if tii == :t1, since stt[:u_sus_bnd] == 0 and no gradient exists
                            # -- this is a weird edge case, but it does make sense if you think about it for
                            # long enough.....
                            tt_max = T_sus_jft[u_on_max_ind]
                            mgd[:u_on_dev][tt_max][dev] += gc*grd[:u_su_dev][:u_on_dev][tt_max][dev]
                            if tt_max != :t1
                                # previous time?
                                mgd[:u_on_dev][prm.ts.tmin1[tt_max]][dev] += gc*grd[:u_su_dev][:u_on_dev_prev][tt_max][dev]
                            end
                        end
                    end
                end
            end
        end
    end
end

# %% timing tests

tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

time_dict = Dict(
    :var1              => Dict(tkeys[ii] => zeros(sys.nb)          for ii in 1:(sys.nT)),
    :var2              => Dict(tkeys[ii] => copy(prm.bus.init_va) for ii in 1:(sys.nT)))

arr_dict = Dict(
    :var1              => [zeros(sys.nb) for _ in 1:(sys.nT)],
    :var2              => [zeros(sys.nb) for _ in 1:(sys.nT)])

arr = [ones(sys.nb) for _ in 1:(sys.nT)]

dev = 10
T_sus_jft = idx.Ts_sus_jft[dev][18][1]
T_sus_jf  = idx.Ts_sus_jf[dev][18][1] 

# %%
f1() = argmax([time_dict[:var1][tii_inst][dev] for tii_inst in T_sus_jft])
f2() = argmax(@inbounds getindex.(arr_dict[:var1], dev))
f3() = @inbounds argmax(first.(arr))
# %%

@btime f1()
@btime f2()
# %%

@btime f3()

# %%
argmax(getindex.(time_dict[:var1],dev))

# %%
dev = 95
t_ind = 1
T_supc     = idx.Ts_supc[dev][t_ind]     # => T_supc, p_supc_set   = get_supc(tii, dev, prm)
p_supc_set = idx.ps_supc_set[dev][t_ind] # => T_supc, p_supc_set   = get_supc(tii, dev, prm)

# %%

f4() = sum(p_supc_set[ii]*stt[:u_su_dev][tii_inst][dev] for (ii,tii_inst) in enumerate(T_supc); init=0.0)
f5() = sum(p_supc_set[ii]*arr_dict[:var1][tii_inst][dev] for (ii,tii_inst) in enumerate(2:10); init=0.0)

# %%
@btime f4()
@btime f5()

# %%
@btime stt[:vm][tii][idx.acline_fr_bus]
@btime @view stt[:vm][tii][idx.acline_fr_bus]

# %%
@btime t = prm.xfm.g_sr;

# %%
@btime @view prm.xfm.g_sr

# %%
@btime msc[:acline_sto] .= sqrt.(stt[:acline_pto][tii].^2 + stt[:acline_qto][tii].^2);