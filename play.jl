# %%
using BenchmarkTools

# %%
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %%

t_ind = 26
tii   = prm.ts.time_keys[t_ind]

residual = zeros(2*sys.nb)
quasiGrad.power_flow_residual!(idx, residual, stt, sys, tii)

# %% ========= %%%%%%%%%%%%%%%%%
model = Model(Gurobi.Optimizer)

t_ind = 26
tii   = prm.ts.time_keys[t_ind]

# initialize
run_pf  = true
pf_cnt  = 0

# 1. update the ideal dispatch point (active power) -- we do this just once
quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

# 2. update y_bus and Jacobian and bias point -- this
#    only needs to be done once per time, since xfm/shunt
#    values are not changing between iterations
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

# loop over pf solves
while run_pf == true

    # increment
    pf_cnt += 1

    # first, rebuild the jacobian, and update the
    # base points: msc[:pinj0], msc[:qinj0]
    Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
    
    # quiet down!!!
    empty!(model)
    set_silent(model)

    # define the variables (single time index)
    @variable(model, x_in[1:(2*sys.nb - 1)])
    set_start_value.(x_in, [stt[:vm][tii]; stt[:va][tii][2:end]])

    # assign
    dvm   = x_in[1:sys.nb]
    dva   = x_in[(sys.nb+1):end]

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

        set_start_value.(pdc_vars, stt[:dc_pfr][tii])
        set_start_value.(qdc_fr_vars, stt[:dc_qfr][tii])
        set_start_value.(qdc_to_vars, stt[:dc_qto][tii])

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

    set_start_value.(dev_p_vars, stt[:dev_p][tii])
    set_start_value.(dev_q_vars, stt[:dev_q][tii])

    # call the bounds
    dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
    dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
    dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
    dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]

    # first, define p_on at this time
    # => p_on = dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii]

    # bound
    @constraint(model, (dev_plb + stt[:p_su][tii] + stt[:p_sd][tii] .- 0*1.5) .<= dev_p_vars .<= (dev_pub + stt[:p_su][tii] + stt[:p_sd][tii]  .+ 0*1.5))
    # alternative: => @constraint(model, dev_plb .<= dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii] .<= dev_pub)
    @constraint(model, (dev_qlb .- 0*0.05) .<= dev_q_vars .<= (dev_qub .+ 0*0.05))

    # apply additional bounds: J_pqe (equality constraints)
    if ~isempty(idx.J_pqe)
        @constraint(model, dev_q_vars[idx.J_pqe] - prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt[:u_sum][tii][idx.J_pqe])
        # alternative: @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt[:u_sum][tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe])
    end

    # apply additional bounds: J_pqmin/max (inequality constraints)
    #
    # note: when the reserve products are negelected, pr and cs constraints are the same
    #   remember: idx.J_pqmax == idx.J_pqmin
    if ~isempty(idx.J_pqmax)
        @constraint(model, dev_q_vars[idx.J_pqmax] .<= prm.dev.q_0_ub[idx.J_pqmax]*stt[:u_sum][tii][idx.J_pqmax] + prm.dev.beta_ub[idx.J_pqmax]*dev_p_vars[idx.J_pqmax])
        @constraint(model, prm.dev.q_0_lb[idx.J_pqmax]*stt[:u_sum][tii][idx.J_pqmax] + prm.dev.beta_lb[idx.J_pqmax]*dev_p_vars[idx.J_pqmax] .<= dev_q_vars[idx.J_pqmax])
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

    # bound system variables ==============================================
    #
    # bound variables -- voltage
    @constraint(model, (prm.bus.vm_lb .- 0*0.1) - stt[:vm][tii] .<= dvm .<= (prm.bus.vm_ub .+ 0*0.1) - stt[:vm][tii])
    # alternative: => @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm .<= prm.bus.vm_ub)

    # mapping
    JacP_noref = @view Jac[1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
    JacQ_noref = @view Jac[(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]

    # balance p and q
    #=
    add_to_expression!.(nodal_p, -msc[:pinj0])
    add_to_expression!.(nodal_q, -msc[:qinj0])
    # constrain
    @constraint(model, JacP_noref*x_in .== nodal_p)
    @constraint(model, JacQ_noref*x_in .== nodal_q)
    # remove these values!
    add_to_expression!.(nodal_p, msc[:pinj0])
    add_to_expression!.(nodal_q, msc[:qinj0])
    =#
    
    # alternative: => @constraint(model, JacP_noref*x_in + msc[:pinj0] .== nodal_p)
    # alternative: => @constraint(model, JacQ_noref*x_in + msc[:qinj0] .== nodal_q)
    @constraint(model, JacP_noref*x_in + msc[:pinj0] .== nodal_p)
    @constraint(model, JacQ_noref*x_in + msc[:qinj0] .== nodal_q)

    # objective: hold p (and v?) close to its initial value
    # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
    if qG.Gurobi_pf_obj == "min_dispatch_distance"
        # this finds a solution close to the dispatch point -- does not converge without v,a regularization
        obj    = AffExpr(0.0)
        tmp_vm = @variable(model)
        tmp_va = @variable(model)
        for bus in 1:sys.nb
            tmp = @variable(model)
            # => @constraint(model, msc[:pinj_ideal][bus] - nodal_p[bus] <= tmp)
            # => @constraint(model, nodal_p[bus] - msc[:pinj_ideal][bus] <= tmp)
            #
            @constraint(model, msc[:pinj_ideal][bus] - nodal_p[bus] <= tmp)
            @constraint(model, nodal_p[bus] - msc[:pinj_ideal][bus] <= tmp)
            # slightly faster:
            #=
            add_to_expression!(nodal_p[bus], -msc[:pinj_ideal][bus])
            @constraint(model,  nodal_p[bus] <= tmp)
            @constraint(model, -nodal_p[bus] <= tmp)
            add_to_expression!(obj, tmp)
            =#

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

    elseif qG.Gurobi_pf_obj == "min_dispatch_perturbation"
        # this finds a solution with minimum movement -- not really needed
        # now that "min_dispatch_distance" converges
        tmp_p  = @variable(model)
        tmp_vm = @variable(model)
        tmp_va = @variable(model)
        for bus in 1:sys.nb
            #tmp = @variable(model)
            @constraint(model, -JacP_noref[bus,:]*x_in <= tmp_p)
            @constraint(model,  JacP_noref[bus,:]*x_in <= tmp_p)

            @constraint(model, -dvm[bus] <= tmp_vm)
            @constraint(model,  dvm[bus] <= tmp_vm)

            if bus > 1
                @constraint(model, -dva[bus-1] <= tmp_va)
                @constraint(model,  dva[bus-1] <= tmp_va)
            end
            # for l1 norm: add_to_expression!(obj, tmp)
        end
        obj = tmp_p + tmp_vm + tmp_va
    else
        @warn "pf solver objective not recognized!"
    end

    # set the objective
    @objective(model, Min, obj)

    # solve
    optimize!(model)

    # take the norm of dv
    max_dx = maximum(value.(x_in))
    
    # println("========================================================")
    println(termination_status(model),". ",primal_status(model),". objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
    # println("========================================================")

    # now, update the state vector with the soluion
    stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
    stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

    # shall we terminate?
    if (maximum(value.(x_in)) < qG.max_pf_dx) || (pf_cnt == qG.max_linear_pfs)
        run_pf = false

        # now, apply the updated injections to the devices
        stt[:dev_p][tii]  = value.(dev_p_vars)
        stt[:p_on][tii]   = stt[:dev_p][tii] - stt[:p_su][tii] - stt[:p_sd][tii]
        stt[:dev_q][tii]  = value.(dev_q_vars)
        if sys.nldc > 0
            stt[:dc_pfr][tii] = value.(pdc_vars)
            stt[:dc_pfr][tii] = value.(pdc_vars)
            stt[:dc_qfr][tii] = value.(qdc_fr_vars)
            stt[:dc_qto][tii] = value.(qdc_to_vars)
        end
    end
end

# %%
# 1. run ED (global upper bound)
ED = quasiGrad.solve_economic_dispatch(GRB, idx, prm, qG, scr, stt, sys, upd)

# 2. apply solution
quasiGrad.apply_economic_dispatch_projection!(ED, idx, prm, qG, stt, sys)

# 3. solve a dc power flow
quasiGrad.dcpf_initialization!(flw, idx, msc, ntk, prm, qG, stt, sys)

# %% 4. update the states -- this is needed for power flow to converge
qG.eval_grad = false
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
qG.eval_grad = true

# %%
for tii in prm.ts.time_keys
    println(stt[:va][tii][2])
end

# %%

thetar = zeros(sys.nb-1)
tii = :t7

# first, update the xfm phase shifters (whatever they may be..)
flw[:ac_phi][idx.ac_phi] = copy(stt[:phi][tii])

# loop over each bus
for bus in 1:sys.nb
    # active power balance
    msc[:pinj_dc][bus] = 
        sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0) - 
        sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) # - 
        # don't include shunt or dc constributions, since power might not balance!
        #sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) - 
        #sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
        #sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0)
    # qinj[bus] = 
    #    sum(stt[:dev_q][tii][idx.pr[bus]]; init=0.0) - 
    #    sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) - 
    #    sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) - 
    #    sum(stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
    #    sum(stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]]; init=0.0)
    #    println(qinj[bus])
end

# now, we need to solve Yb*theta = pinj, but we need to 
# take phase shifters into account first:
bt = -flw[:ac_phi].*ntk.b
c  = msc[:pinj_dc][2:end] - ntk.Er'*bt
# now, we need to solve Yb_r*theta_r = c via pcg

err = ntk.Ybr*stt[:va][tii][2:end] - c

println(maximum(abs.(stt[:va][tii][2:end])))
println(quasiGrad.norm(abs.(c)))

# %%
if sys.nb <= qG.min_buses_for_krylov
    # too few buses -- just use LU
    stt[:va][tii][2:end] = ntk.Ybr\c
    stt[:va][tii][1]     = 0.0 # make sure

    # voltage magnitude
    # stt[:vm][tii][2:end] = ntk.Ybr\qinj[2:end] .+ 1.0
    # stt[:vm][tii][1]     = 1.0 # make sure
else
    # solve with pcg -- va
    quasiGrad.cg!(thetar, ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)
    
    # cg has failed in the past -- not sure why -- test for NaN
    if isnan(sum(thetar)) || maximum(thetar) > 1e7  # => faster than any(isnan, t)
        # LU backup
        @info "Krylov failed -- using LU backup (dcpf)!"
        thetar = ntk.Ybr\c
    end

    # update
    stt[:va][tii][2:end] = copy(thetar)
    stt[:va][tii][1]     = 0.0 # make sure

    # solve with pcg -- vm
    # quasiGrad.cg!(vmr, ntk.Ybr, qinj[2:end], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)
    
    # cg has failed in the past -- not sure why -- test for NaN
    # if isnan(sum(vmr)) || maximum(vmr) > 1e7  # => faster than any(isnan, t)
        # LU backup
    #    @info "Krylov failed -- using LU backup (dcpf)!"
    #    vmr = ntk.Ybr\qinj[2:end]
    # end

    # voltage magnitude
    # stt[:vm][tii][2:end] = vmr .+ 1.0
    # stt[:vm][tii] .= 1.0 # make sure
end

# %%
# note: all binaries are LP relaxed (so there is not BaB-ing): 0 < b < 1
#
# NOTE -- we are not including start-up-state discounts -- not worth it :)
#
# initialize by just copying GRB to ED
#ED = deepcopy(GRB)

# build and empty the model!
model = quasiGrad.Model(quasiGrad.Gurobi.Optimizer; add_bridges = false)
quasiGrad.set_string_names_on_creation(model, false)

# quiet down!!!
quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

# set model properties => let this run until it finishes
    # quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
    # quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
    # quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)

# define local time keys
tkeys = prm.ts.time_keys

# define the minimum set of variables we will need to solve the constraints
u_on_dev  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "u_on_dev_t$(ii)",  [dev = 1:sys.ndev], start=stt[:u_on_dev][tkeys[ii]][dev],  lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT))
p_on      = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_on_t$(ii)",      [dev = 1:sys.ndev], start=stt[:p_on][tkeys[ii]][dev])                                            for ii in 1:(sys.nT))
dev_q     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "dev_q_t$(ii)",     [dev = 1:sys.ndev], start=stt[:dev_q][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
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

# add a few more (implicit) variables which are necessary for solving this system
u_su_dev = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "u_su_dev_t$(ii)", [dev = 1:sys.ndev], start=stt[:u_su_dev][tkeys[ii]][dev], lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT))
u_sd_dev = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "u_sd_dev_t$(ii)", [dev = 1:sys.ndev], start=stt[:u_sd_dev][tkeys[ii]][dev], lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT))

# we have the affine "AffExpr" expressions (whose values are specified)
dev_p   = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
p_su    = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
p_sd    = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
zen_dev = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))

# now, we need to loop and set the affine expressions to 0
#   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
for tii in prm.ts.time_keys
    for dev in 1:sys.ndev
        dev_p[tii][dev]   = AffExpr(0.0)
        p_su[tii][dev]    = AffExpr(0.0)
        p_sd[tii][dev]    = AffExpr(0.0)
        zen_dev[tii][dev] = AffExpr(0.0)
    end
end

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
zt      = AffExpr(0.0)
z_enmax = AffExpr(0.0)
z_enmin = AffExpr(0.0)

# loop over all devices
for dev in 1:sys.ndev

    # == define active power constraints ==
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # first, get the startup power
        T_supc     = idx.Ts_supc[dev][t_ind]     # T_set, p_supc_set = get_supc(tii, dev, prm)
        p_supc_set = idx.ps_supc_set[dev][t_ind] # T_set, p_supc_set = get_supc(tii, dev, prm)
        add_to_expression!(p_su[tii][dev], sum(p_supc_set[ii]*u_su_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_supc); init=0.0))

        # second, get the shutdown power
        T_sdpc     = idx.Ts_sdpc[dev][t_ind]     # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
        p_sdpc_set = idx.ps_sdpc_set[dev][t_ind] # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
        add_to_expression!(p_sd[tii][dev], sum(p_sdpc_set[ii]*u_sd_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_sdpc); init=0.0))

        # finally, get the total power balance
        dev_p[tii][dev] = p_on[tii][dev] + p_su[tii][dev] + p_sd[tii][dev]
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
                u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)
                
                # compute q -- this might be the only equality constraint (and below)
                @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
            else
                # the device must be a consumer :)
                T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                # compute q -- this might be the only equality constraint (and above)
                @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
            end
        end
    end

    # loop over each time period and define the hard constraints
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # 1. Minimum downtime: zhat_mndn
        T_mndn = idx.Ts_mndn[dev][t_ind] # t_set = get_tmindn(tii, dev, prm)
        @constraint(model, u_su_dev[tii][dev] + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_mndn; init=0.0) - 1.0 <= 0.0)

        # 2. Minimum uptime: zhat_mnup
        T_mnup = idx.Ts_mnup[dev][t_ind] # t_set = get_tminup(tii, dev, prm)
        @constraint(model, u_sd_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_mnup; init=0.0) - 1.0 <= 0.0)

        # define the previous power value (used by both up and down ramping!)
        if tii == :t1
            # note: p0 = prm.dev.init_p[dev]
            dev_p_previous = prm.dev.init_p[dev]
        else
            # grab previous time
            tii_m1 = prm.ts.time_keys[t_ind-1]
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
            @constraint(model, p_on[tii][dev] + p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii][dev] <= 0.0)
        
            # 14p. Minimum reserve limits (producers): zhat_pmin
            @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii][dev] + p_rrd_on[tii][dev] + p_rgd[tii][dev] - p_on[tii][dev] <= 0.0)
            
            # 15p. Off reserve limits (producers): zhat_pmaxoff
            @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

            # get common "u_sum" terms that will be used in the subsequent four equations 
            T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
            T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
            u_sum     = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

            # 16p. Maximum reactive power reserves (producers): zhat_qmax
            @constraint(model, dev_q[tii][dev] + q_qru[tii][dev] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

            # 17p. Minimum reactive power reserves (producers): zhat_qmin
            @constraint(model, q_qrd[tii][dev] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii][dev] <= 0.0)

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
            @constraint(model, p_on[tii][dev] + p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii][dev] <= 0.0)

            # 14c. Minimum reserve limits (consumers): zhat_pmin
            @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii][dev] + p_rru_on[tii][dev] + p_scr[tii][dev] + p_rgu[tii][dev] - p_on[tii][dev] <= 0.0)
            
            # 15c. Off reserve limits (consumers): zhat_pmaxoff
            @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_rrd_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

            # get common "u_sum" terms that will be used in the subsequent four equations 
            T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
            T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
            u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

            # 16c. Maximum reactive power reserves (consumers): zhat_qmax
            @constraint(model, dev_q[tii][dev] + q_qrd[tii][dev] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

            # 17c. Minimum reactive power reserves (consumers): zhat_qmin
            @constraint(model, q_qru[tii][dev] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii][dev] <= 0.0)
            
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
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        if tii == :t1
            @constraint(model, u_on_dev[tii][dev] - prm.dev.init_on_status[dev] == u_su_dev[tii][dev] - u_sd_dev[tii][dev])
        else
            tii_m1 = prm.ts.time_keys[t_ind-1]
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
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # if a device is *not* in the set of variables,
        # then it must be held constant! -- otherwise, try to hold it
        # close to its initial value
        if dev ∉ upd[:u_on_dev][tii]
            @constraint(model, u_on_dev[tii][dev] == stt[:u_on_dev][tii][dev])
        end

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

        # now, deal with reactive powers, some of which are specified with equality
        # only a subset of devices will have a reactive power equality constraint
        #
        # nothing here :)
    end
end

# now, include a "copper plate" power balance constraint
# loop over each time period and compute the power balance
for tii in prm.ts.time_keys
    # duration
    dt = prm.ts.duration[tii]

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

# ========== costs! ============= #
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    # duration
    dt = prm.ts.duration[tii]

    # active power costs
    for dev in 1:sys.ndev
        # note -- these were sorted previously!
        cst = prm.dev.cum_cost_blocks[dev][t_ind][1][2:end]  # cost for each block (trim leading 0)
        pbk = prm.dev.cum_cost_blocks[dev][t_ind][2][2:end]  # power in each block (trim leading 0)
        nbk = length(pbk)

        # define a set of intermediate vars "p_jtm"
        p_jtm = @variable(model, [1:nbk], lower_bound = 0.0)
        @constraint(model, p_jtm .<= pbk)

        # have the blocks sum to the output power
        @constraint(model, sum(p_jtm) == dev_p[tii][dev])

        # compute the cost!
        zen_dev[tii][dev] = dt*sum(cst.*p_jtm)
    end
end
        
# compute the costs associated with device reserve offers -- computed directly in the objective
# 
# min/max energy requirements
for dev in 1:sys.ndev
    Wub = prm.dev.energy_req_ub[dev]
    Wlb = prm.dev.energy_req_lb[dev]

    # upper bounds
    for (w_ind, w_params) in enumerate(Wub)
        T_en_max = idx.Ts_en_max[dev][w_ind]
        zw_enmax = @variable(model, lower_bound = 0.0)
        @constraint(model, prm.vio.e_dev*(sum(prm.ts.duration[tii]*dev_p[tii][dev] for tii in T_en_max; init=0.0) - w_params[3]) <= zw_enmax)
        add_to_expression!(z_enmax, -1.0, zw_enmax)
    end

    # lower bounds
    for (w_ind, w_params) in enumerate(Wlb)
        T_en_min = idx.Ts_en_min[dev][w_ind]
        zw_enmin = @variable(model, lower_bound = 0.0)
        @constraint(model, prm.vio.e_dev*(w_params[3] - sum(prm.ts.duration[tii]*dev_p[tii][dev] for tii in T_en_min; init=0.0)) <= zw_enmin)
        add_to_expression!(z_enmin, -1.0, zw_enmin)
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
            @constraint(model, p_rgu_zonal_REQ[tii][zone] == rgu_sigma[zone]*sum(dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
            @constraint(model, p_rgd_zonal_REQ[tii][zone] == rgd_sigma[zone]*sum(dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
        end

        # endogenous max
        if idx.pr_pzone[zone] == []
            # in the case there are NO producers in a zone
            @constraint(model, p_scr_zonal_REQ[tii][zone] == 0.0)
            @constraint(model, p_nsc_zonal_REQ[tii][zone] == 0.0)
        else
            @constraint(model, scr_sigma[zone]*[dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[tii][zone])
            @constraint(model, nsc_sigma[zone]*[dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[tii][zone])
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

    # shortfall penalties -- NOT needed explicitly (see objective)
end

# loop -- NOTE -- we are not including start-up-state discounts -- not worth it
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    # duration
    dt = prm.ts.duration[tii]
    
    # add up
    zt_temp = 
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
        # => **** don't include for now: sum(stt[:zsus_dev][tii]) - ****
        # local reserve penalties
        sum(dt*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*p_rgu[tii]) -   # zrgu
        sum(dt*prm.dev.p_reg_res_down_cost_tmdv[t_ind].*p_rgd[tii]) - # zrgd
        sum(dt*prm.dev.p_syn_res_cost_tmdv[t_ind].*p_scr[tii]) -      # zscr
        sum(dt*prm.dev.p_nsyn_res_cost_tmdv[t_ind].*p_nsc[tii]) -     # znsc
        sum(dt*(prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind].*p_rru_on[tii] +
                prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind].*p_rru_off[tii])) -   # zrru
        sum(dt*(prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind].*p_rrd_on[tii] +
                prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind].*p_rrd_off[tii])) - # zrrd
        sum(dt*prm.dev.q_res_up_cost_tmdv[t_ind].*q_qru[tii]) -   # zqru      
        sum(dt*prm.dev.q_res_down_cost_tmdv[t_ind].*q_qrd[tii]) - # zqrd
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

# define the objective
zms_partial = zt + z_enmax + z_enmin

# set the objective
@objective(model, Max, zms_partial)

# solve
optimize!(model)
# println("========================================================")
println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
# println("========================================================")

# solve, and then return the solution
for tii in prm.ts.time_keys
    ED[:u_on_dev][tii]  = copy(value.(u_on_dev[tii]))
    ED[:p_on][tii]      = copy(value.(p_on[tii]))
    ED[:dev_q][tii]     = copy(value.(dev_q[tii]))
    ED[:p_rgu][tii]     = copy(value.(p_rgu[tii]))
    ED[:p_rgd][tii]     = copy(value.(p_rgd[tii]))
    ED[:p_scr][tii]     = copy(value.(p_scr[tii]))
    ED[:p_nsc][tii]     = copy(value.(p_nsc[tii]))
    ED[:p_rru_on][tii]  = copy(value.(p_rru_on[tii]))
    ED[:p_rru_off][tii] = copy(value.(p_rru_off[tii]))
    ED[:p_rrd_on][tii]  = copy(value.(p_rrd_on[tii]))
    ED[:p_rrd_off][tii] = copy(value.(p_rrd_off[tii]))
    ED[:q_qru][tii]     = copy(value.(q_qru[tii]))
    ED[:q_qrd][tii]     = copy(value.(q_qrd[tii]))
end

# update the objective value score
scr[:ed_obj] = objective_value(model)

# %%
@time ED[:u_on_dev][tii]  = copy(value.(u_on_dev[tii]));
@time ED[:u_on_dev][tii]  = value.(u_on_dev[tii]);

# %%
function cpy(GRB::Dict{Symbol, Dict{Symbol, Vector{Float64}}},prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}},)
    for tii in prm.ts.time_keys
        stt[:u_on_dev][tii] = copy(GRB[:u_on_dev][tii])
    end
end

function dpcpy(GRB::Dict{Symbol, Dict{Symbol, Vector{Float64}}},prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}},)
    stt[:u_on_dev] = deepcopy(GRB[:u_on_dev])
end

# %% ==================
@btime t = dpcpy(GRB, prm, stt);
@btime t = cpy(GRB, prm, stt);

# %%
for j = 1:1000
    println(j)
    if j >= 3
        break
    end
    println(j)
end

# %%
A = randn(250,250)
A = A + A' + 200*quasiGrad.I
b = randn(250)
x = zeros(250)

# %%

svs = quasiGrad.CGStateVariables(zero(x), similar(x), similar(x))
@btime quasiGrad.cg!(x, A, b, abstol = 0.000001, statevars = svs, log = true, maxiter = 50, verbose = false);

@btime quasiGrad.cg!(x, A, b, abstol = 0.000001, log = false, maxiter = 50, verbose = false);

# %%
A = randn(250,250)
A = A + A' + 200*quasiGrad.I
b = randn(250)
x = zeros(250)
#println(ch.:isconverged)
@time _, ch = quasiGrad.cg!(x, A, b, abstol = 0.000001, statevars = svs, log = true, maxiter = 25, verbose = false);
# %%
A = randn(250,250)
A = A + A' + 200*quasiGrad.I
b = randn(250)
x = zeros(250)

@time quasiGrad.cg!(x, A, b, abstol = 0.000001, statevars = svs, log = false, maxiter = 25, verbose = false);

# %%

thetar = zeros(sys.nb-1) # this will be overwritten -- we leave it, since cg writes to it!!
# => vmr    = zeros(sys.nb-1) # this will be overwritten
tii = :t1
# first, update the xfm phase shifters (whatever they may be..)
flw[:ac_phi][idx.ac_phi] = copy(stt[:phi][tii])

# loop over each bus
for bus in 1:sys.nb
    # active power balance -- just devices
    # !! don't include shunt or dc constributions, 
    #    since power might not balance !!
    msc[:pinj_dc][bus] = 
        sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0) - 
        sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) # - 

    # qinj[bus] = 
    #    sum(stt[:dev_q][tii][idx.pr[bus]]; init=0.0) - 
    #    sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) - 
    #    sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) - 
    #    sum(stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
    #    sum(stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]]; init=0.0)
    #    println(qinj[bus])
end

# now, we need to solve Yb*theta = pinj, but we need to 
# take phase shifters into account first:
bt = -flw[:ac_phi].*ntk.b
c  = msc[:pinj_dc][2:end] - ntk.Er'*bt
# now, we need to solve Yb_r*theta_r = c via pcg

if sys.nb <= qG.min_buses_for_krylov
    # too few buses -- just use LU
    stt[:va][tii][2:end] = ntk.Ybr\c
    stt[:va][tii][1]     = 0.0 # make sure

    # voltage magnitude
    # stt[:vm][tii][2:end] = ntk.Ybr\qinj[2:end] .+ 1.0
    # stt[:vm][tii][1]     = 1.0 # make sure
else
    # solve with pcg -- va
    _, ch = quasiGrad.cg!(thetar, ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)

    # test the krylov solution
    if ~(ch.isconverged)
        # LU backup
        @info "Krylov failed -- using LU backup (dcpf)!"
        thetar = ntk.Ybr\c
    end

    # update
    stt[:va][tii][2:end] = copy(thetar)
    stt[:va][tii][1]     = 0.0 # make sure

    # solve with pcg -- vm
    # quasiGrad.cg!(vmr, ntk.Ybr, qinj[2:end], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)
    
    # cg has failed in the past -- not sure why -- test for NaN
    # if isnan(sum(vmr)) || maximum(vmr) > 1e7  # => faster than any(isnan, t)
        # LU backup
    #    @info "Krylov failed -- using LU backup (dcpf)!"
    #    vmr = ntk.Ybr\qinj[2:end]
    # end

    # voltage magnitude
    # stt[:vm][tii][2:end] = vmr .+ 1.0
    # stt[:vm][tii] .= 1.0 # make sure
end

# %%
scr[:zctg_min] = 0.0
scr[:zctg_avg] = 0.0

# how many ctgs 
num_wrst = Int64(round(qG.frac_ctg_keep*sys.nctg/2))
num_rnd  = Int64(round(qG.frac_ctg_keep*sys.nctg/2))
num_ctg  = num_wrst + num_rnd

# loop over time
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    # duration
    dt = prm.ts.duration[tii]

    # compute the shared, common gradient terms (time dependent)
    #
    # gc_avg = grd[:nzms][:zctg_avg] * grd[:zctg_avg][:zctg_avg_t] * grd[:zctg_avg_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
    #        = (-1)                  *                (1)          *       (1/sys.nctg)        *          (-1)       *   dt*prm.vio.s_flow
    #        = dt*prm.vio.s_flow/sys.nctg
    gc_avg   = cgd.ctg_avg[tii] * qG.scale_c_sflow_testing
    
    # gc_min = (-1)                  *                (1)          *            (1)          *          (-1)       *   dt*prm.vio.s_flow
    #        = grd[:nzms][:zctg_min] * grd[:zctg_min][:zctg_min_t] * grd[:zctg_min_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
    gc_min   = cgd.ctg_min[tii] * qG.scale_c_sflow_testing

    # get the slack at this time
    p_slack = 
        sum(stt[:dev_p][tii][idx.pr_devs]) -
        sum(stt[:dev_p][tii][idx.cs_devs]) - 
        sum(stt[:sh_p][tii])

    # loop over each bus
    for bus in 1:sys.nb
        # active power balance
        flw[:p_inj][bus] = 
            sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0) - 
            sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) - 
            sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) - 
            sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
            sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) - 
            ntk.alpha*p_slack
    end

    # also, we need to update the flows on all lines! and the phase shift
    flw[:ac_qfr][idx.ac_line_flows] = stt[:acline_qfr][tii]
    flw[:ac_qfr][idx.ac_xfm_flows]  = stt[:xfm_qfr][tii]
    flw[:ac_qto][idx.ac_line_flows] = stt[:acline_qto][tii]
    flw[:ac_qto][idx.ac_xfm_flows]  = stt[:xfm_qto][tii]
    flw[:ac_phi][idx.ac_phi]        = stt[:phi][tii]

    # compute square flows
    qfr2 = flw[:ac_qfr].^2
    qto2 = flw[:ac_qto].^2

    # solve for the flows across each ctg
    #   p  =  @view flw[:p_inj][2:end]
    bt = -flw[:ac_phi].*ntk.b
    # now, we have flw[:p_inj] = Yb*theta + E'*bt
    #   c = p - ntk.Er'*bt
    #
    # simplified:
    c = (@view flw[:p_inj][2:end]) - ntk.Er'*bt

    # solve the base case with pcg
    if qG.base_solver == "lu"
        ctb[t_ind]  = ntk.Ybr\c

    # error with this type !!!
    # elseif qG.base_solver == "cholesky"
    #    ctb[t_ind]  = ntk.Ybr_Ch\c
    
    elseif qG.base_solver == "pcg"
        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- just use LU
            ctb[t_ind] = ntk.Ybr\c
        else
            # solve with a hot start!
            #
            # note: ctg[:ctb][tii][end] is modified in place,
            # and it represents the base case solution
            _, ch = quasiGrad.cg!(ctb[t_ind], ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
            
            # test the krylov solution
            if ch.isconverged == false
                @info "Krylov failed -- using LU backup (ctg flows)!"
                ctb[t_ind] = ntk.Ybr\c
            end
        end
    else
        println("base case solve type not recognized :)")
    end

    # set all ctg scores to 0:
    stt[:zctg][tii] .= 0.0

    # zero out the gradients, which will be collected and applied all at once!
    flw[:dz_dpinj_all] .= 0.0

    # do we want to score all ctgs? for testing/post processing
    if qG.score_all_ctgs == true
        ###########################################################
        @info "Warning -- scoring all contingencies! No gradients."
        ###########################################################
        for ctg_ii in 1:sys.nctg
            # see the "else" case for comments and details
            theta_k = quasiGrad.special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], c)
            pflow_k = ntk.Yfr*theta_k  + bt
            sfr     = sqrt.(qfr2 + pflow_k.^2)
            sto     = sqrt.(qto2 + pflow_k.^2)
            sfr_vio = sfr - ntk.s_max
            sto_vio = sto - ntk.s_max
            println(maximum(sto_vio))
            sfr_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
            sto_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
            smax_vio = max.(sfr_vio, sto_vio, 0.0)
            zctg_s = dt*prm.vio.s_flow*smax_vio * qG.scale_c_sflow_testing
            stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)
            println(stt[:zctg][tii][ctg_ii])
            println("==============")
        end
    end
end

# %%
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    println(sum(stt[:zctg][tii]))
end

# %%
using JuMP
using Gurobi

function f1(A::Matrix{Float64}, b::Vector{Float64})
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, x[1:50])
    @constraint(model, -10.0 .<= x .<= 10.0)
    @constraint(model, A*x <= b)
    @objective(model, Min, sum(x))
    optimize!(model)
end

function f2(A::Matrix{Float64}, b::Vector{Float64}, z::Vector{Float64})
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, x[1:50])
    @constraint(model, -10.0 .<= x .<= 10.0)
    @constraint(model, A*x - b <= z)
    @objective(model, Min, sum(x))
    optimize!(model)
end

# %% ===========
A = randn(50,50)
b = randn(50)
z = zeros(50)
# %%
@btime f1(A,b);
# %%
@btime f2(A,b,z);

# %%

ff1(x,y) = x*y

@btime ff1(1.2,-1.0)
@btime ff1(1.2,-1)

# %%
tii   = :t14
t_ind = 14
dev   = 182
dev   = 1
dt    = prm.ts.duration[tii]

cst = prm.dev.cum_cost_blocks[dev][t_ind][1]  # cost for each block (leading with 0)
pbk = prm.dev.cum_cost_blocks[dev][t_ind][2]  # power in each block (leading with 0)
pcm = prm.dev.cum_cost_blocks[dev][t_ind][3]  # accumulated power for each block!
nbk = length(pbk)

# %% get the cost!
stt[:zen_dev][tii][dev] = dt*sum(cst[ii]*max(min(stt[:dev_p][tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)

# evaluate the grd? 
#
# WARNING -- this will break if stt[:dev_p] > pcm[end]! It will
#            mean the device power is out of bounds, and this will
#            call a price curve which does not exist.
#                  ~ clipping will fix ~
if qG.eval_grad
    # what is the index of the "active" block?
    del = stt[:dev_p][tii][dev] .- pcm
    active_block_ind = argmin(del[del .>= 0.0])
    grd[:zen_dev][:dev_p][tii][dev] = dt*cst[active_block_ind + 1] # where + 1 is due to the leading 0
end

# %%
scr[:zt_penalty] = 0.0
tii = :t18
scr[:zt_penalty] += -qG.constraint_grad_weight*(
            sum(stt[:zhat_mndn][tii]) +
            sum(stt[:zhat_mnup][tii]) + 
            sum(stt[:zhat_rup][tii]) + 
            sum(stt[:zhat_rd][tii])  + 
            sum(stt[:zhat_rgu][tii]) + 
            sum(stt[:zhat_rgd][tii]) + 
            sum(stt[:zhat_scr][tii]) + 
            sum(stt[:zhat_nsc][tii]) + 
            sum(stt[:zhat_rruon][tii])  + 
            sum(stt[:zhat_rruoff][tii]) +
            sum(stt[:zhat_rrdon][tii])  +
            sum(stt[:zhat_rrdoff][tii]) +
            # common set of pr and cs constraint variables (see below)
            sum(stt[:zhat_pmax][tii])      + 
            sum(stt[:zhat_pmin][tii])      + 
            sum(stt[:zhat_pmaxoff][tii])   + 
            sum(stt[:zhat_qmax][tii])      + 
            sum(stt[:zhat_qmin][tii])      + 
            sum(stt[:zhat_qmax_beta][tii]) + 
            sum(stt[:zhat_qmin_beta][tii]))

# %%

println(sum(sum(stt[:zhat_mndn][tii] for tii in prm.ts.time_keys)))
println(sum(sum(stt[:zhat_mndn][tii] for tii in prm.ts.time_keys)))
println(sum(sum(stt[:zhat_mnup][tii] for tii in prm.ts.time_keys))) 
println(sum(sum(stt[:zhat_rup][tii] for tii in prm.ts.time_keys))) 
println(sum(sum(stt[:zhat_rd][tii] for tii in prm.ts.time_keys)))  
println(sum(sum(stt[:zhat_rgu][tii] for tii in prm.ts.time_keys))) 
println(sum(sum(stt[:zhat_rgd][tii] for tii in prm.ts.time_keys))) 
println(sum(sum(stt[:zhat_scr][tii] for tii in prm.ts.time_keys))) 
println(sum(sum(stt[:zhat_nsc][tii] for tii in prm.ts.time_keys))) 
println(sum(sum(stt[:zhat_rruon][tii] for tii in prm.ts.time_keys)))  
println(sum(sum(stt[:zhat_rruoff][tii] for tii in prm.ts.time_keys)))
println(sum(sum(stt[:zhat_rrdon][tii] for tii in prm.ts.time_keys))) 
println(sum(sum(stt[:zhat_rrdoff][tii] for tii in prm.ts.time_keys)))
println(sum(sum(stt[:zhat_pmax][tii] for tii in prm.ts.time_keys)))       
println(sum(sum(stt[:zhat_pmin][tii] for tii in prm.ts.time_keys)))       
println(sum(sum(stt[:zhat_pmaxoff][tii] for tii in prm.ts.time_keys)))    
println(sum(sum(stt[:zhat_qmax][tii] for tii in prm.ts.time_keys)))       
println(sum(sum(stt[:zhat_qmin][tii] for tii in prm.ts.time_keys)))       
println(sum(sum(stt[:zhat_qmax_beta][tii] for tii in prm.ts.time_keys)))  
println(sum(sum(stt[:zhat_qmin_beta][tii] for tii in prm.ts.time_keys)))


# %% ===========
v = randn(10000)
p = 1:1000
ci = CartesianIndices(p)


@btime view(v, p);
@btime view(v, ci);
# %%

@btime v'*v;
@btime (v[p])'*(v[p])
@btime quasiGrad.dot(v[p],v[p])
@btime (@view v[p])'*(@view v[p])
@btime (@view v[ci])'*(@view v[ci])
@btime view(v, ci)'*view(v, ci)

# %%
@btime v'*v;
@btime (@view v[p])'*(@view v[p])
@btime (@view v[ci])'*(@view v[ci])
@btime view(v, ci)'*view(v, ci)

# %%
tii = :t1
c1 = idx.acline_fr_bus
c2 = CartesianIndices(idx.acline_fr_bus)
c3 = eachindex(stt[:vm][tii])
@btime stt[:vm][tii][c1];
@btime stt[:vm][tii][c3];
# %%
@btime quasiGrad.dot(view(v, ci),view(v, ci))

# %% clean-up the reserve values by solving a softly constrained LP
@time quasiGrad.centralized_soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% clean-up the reserve values by solving a softly constrained LP
@time quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %%
x = -10:0.01:10
@btime sign.(x);
@btime x./(sqrt.(x.^2 .+ eps2));s

# %%

function MyJulia1(InFile1::String, TimeLimitInSeconds::Any, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")

    # how long did package loading take? Give it 20 sec for now..
    NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 20.0

    # compute the solution
    quasiGrad.compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
end

# %% =================
path                  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
InFile1               = path
TimeLimitInSeconds    = 600.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

MyJulia1(InFile1, TimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

# %%
vals = Float64[]
for dev in 1:sys.ndev
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        #push!(vals, stt[:q_qru][tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev])
        push!(vals, prm.dev.q_lb[dev][t_ind])
    end
end
# %%

dev = 1
t_ind = 6
@btime idx.Ts_mndn[dev][t_ind]
@btime @view idx.Ts_mndn[dev][t_ind]
dt = prm.ts.duration[tii]

# %%
smallest = 0.0
for 

# %%
tii = :t18
t_ind = 18

# call device bounds
dev_plb = prm.dev.p_lb_tmdv[t_ind]
dev_pub = prm.dev.p_ub_tmdv[t_ind]
dev_qlb = prm.dev.q_lb_tmdv[t_ind]
dev_qub = prm.dev.q_ub_tmdv[t_ind]

# define p_on at this time

# constraints: 7 types
#
# define the previous power value (used by both up and down ramping!)

# grab previous time
tii_m1 = prm.ts.time_keys[t_ind-1]
dev_p_previous = stt[:dev_p][tii_m1]
dt = prm.ts.duration[tii]

# 1. ramp up
c1 = stt[:dev_p][tii] - dev_p_previous - dt*(prm.dev.p_ramp_up_ub.*(stt[:u_on_dev][tii] - stt[:u_su_dev][tii]) + prm.dev.p_startup_ramp_ub.*(stt[:u_su_dev][tii] .+ 1.0 .- stt[:u_on_dev][tii]))

# 2. ramp down
c2 = dev_p_previous - stt[:dev_p][tii] - dt*(prm.dev.p_ramp_down_ub.*stt[:u_on_dev][tii] + prm.dev.p_shutdown_ramp_ub.*(1.0 .- stt[:u_on_dev][tii]))

# 3. pmax
c3 = stt[:p_on][tii] - dev_pub.*stt[:u_on_dev][tii]

# 4. pmin
c4 = dev_plb.*stt[:u_on_dev][tii] - stt[:p_on][tii]

# 5. qmax
c5 = stt[:dev_q][tii] - dev_qub.*stt[:u_sum][tii]

# 6. qmin
c6 = dev_qlb.*stt[:u_sum][tii] - stt[:dev_q][tii]

println(maximum(c1))
println(maximum(c2))
println(maximum(c3))
println(maximum(c4))
println(maximum(c5))
println(maximum(c6))

# %%
dev = 306
stt[:dev_p][tii][dev] - dev_p_previous[dev] - dt*(prm.dev.p_ramp_up_ub[dev].*(stt[:u_on_dev][tii][dev] - stt[:u_su_dev][tii][dev]) + prm.dev.p_startup_ramp_ub[dev].*(stt[:u_su_dev][tii][dev] .+ 1.0 .- stt[:u_on_dev][tii][dev]))




# %% ======================
dev = 58
final_projection = false
# loop over each device and solve individually -- not clear if this is faster
# than solving one big optimization problem all at once. see legacy code for
# a(n unfinished) version where all devices are solved at once!
model = Model(Gurobi.Optimizer; add_bridges = false)
set_string_names_on_creation(model, false)
set_silent(model)

# MOI tolerances
quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), 1e-10)
quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), 1e-10)

# define local time keys
tkeys = prm.ts.time_keys

# define the minimum set of variables we will need to solve the constraints           
if final_projection
    # treat u_on_dev as a fixed binary value
    u_on_dev = Dict(tkeys[ii] => stt[:u_on_dev][tkeys[ii]][dev] for ii in 1:(sys.nT))
else
    # treat u_on_dev it as a variable
    u_on_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true)       for ii in 1:(sys.nT)) # => base_name = "u_on_dev_t$(ii)",  
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
    #=
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
    =#
end

# loop over each time period and define the hard constraints
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    # duration
    dt = prm.ts.duration[tii]

    # 1. Minimum downtime: zhat_mndn
    T_mndn = idx.Ts_mndn[dev][t_ind] # t_set = get_tmindn(tii, dev, prm)
    #@constraint(model, u_su_dev[tii] + sum(u_sd_dev[tii_inst] for tii_inst in T_mndn; init=0.0) - 1.0 <= 0.0)

    # 2. Minimum uptime: zhat_mnup
    T_mnup = idx.Ts_mnup[dev][t_ind] # t_set = get_tminup(tii, dev, prm)
    #@constraint(model, u_sd_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_mnup; init=0.0) - 1.0 <= 0.0)

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
    #@constraint(model, dev_p[tii] - dev_p_previous
    #        - dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii] - u_su_dev[tii])
    #        +     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii] + 1.0 - u_on_dev[tii])) <= 0.0)

    # 4. Ramping limits (down): zhat_rd
    #@constraint(model,  dev_p_previous - dev_p[tii]
    #        - dt*(prm.dev.p_ramp_down_ub[dev]*u_on_dev[tii]
    #        +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-u_on_dev[tii])) <= 0.0)

    # 5. Regulation up: zhat_rgu
    #@constraint(model, p_rgu[tii] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[tii] <= 0.0)

    # 6. Regulation down: zhat_rgd
    #@constraint(model, p_rgd[tii] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[tii] <= 0.0)

    # 7. Synchronized reserve: zhat_scr
    #@constraint(model, p_rgu[tii] + p_scr[tii] - prm.dev.p_syn_res_ub[dev]*u_on_dev[tii] <= 0.0)

    # 8. Synchronized reserve: zhat_nsc
    #@constraint(model, p_nsc[tii] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[tii]) <= 0.0)

    # 9. Ramping reserve up (on): zhat_rruon
    #@constraint(model, p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[tii] <= 0.0)

    # 10. Ramping reserve up (off): zhat_rruoff
    #@constraint(model, p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0-u_on_dev[tii]) <= 0.0)
    
    # 11. Ramping reserve down (on): zhat_rrdon
    #@constraint(model, p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[tii] <= 0.0)

    # 12. Ramping reserve down (off): zhat_rrdoff
    #@constraint(model, p_rrd_off[tii] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[tii]) <= 0.0)
    
    # Now, we must separate: producers vs consumers
    if dev in idx.pr_devs
        # 13p. Maximum reserve limits (producers): zhat_pmax
        #@constraint(model, p_on[tii] + p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0.0)
    
        # 14p. Minimum reserve limits (producers): zhat_pmin
        #@constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rrd_on[tii] + p_rgd[tii] - p_on[tii] <= 0.0)
        
        # 15p. Off reserve limits (producers): zhat_pmaxoff
        #@constraint(model, p_su[tii] + p_sd[tii] + p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0.0)

        # get common "u_sum" terms that will be used in the subsequent four equations 
        T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
        T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
        u_sum     = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

        # 16p. Maximum reactive power reserves (producers): zhat_qmax
        #@constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

        # 17p. Minimum reactive power reserves (producers): zhat_qmin
        #@constraint(model, q_qrd[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0.0)

        # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
        if dev in idx.J_pqmax
            #@constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_0_ub[dev]*u_sum
            #- prm.dev.beta_ub[dev]*dev_p[tii] <= 0.0)
        end 
        
        # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
        if dev in idx.J_pqmin
            #@constraint(model, prm.dev.q_0_lb[dev]*u_sum
            #+ prm.dev.beta_lb[dev]*dev_p[tii]
            #+ q_qrd[tii] - dev_q[tii] <= 0.0)
        end

    # consumers
    else  # => dev in idx.cs_devs
        # 13c. Maximum reserve limits (consumers): zhat_pmax
        #@constraint(model, p_on[tii] + p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0.0)

        # 14c. Minimum reserve limits (consumers): zhat_pmin
        #@constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rru_on[tii] + p_scr[tii] + p_rgu[tii] - p_on[tii] <= 0.0)
        
        # 15c. Off reserve limits (consumers): zhat_pmaxoff
        #@constraint(model, p_su[tii] + p_sd[tii] + p_rrd_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0.0)

        # get common "u_sum" terms that will be used in the subsequent four equations 
        T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
        T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
        u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

        # 16c. Maximum reactive power reserves (consumers): zhat_qmax
        #@constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

        # 17c. Minimum reactive power reserves (consumers): zhat_qmin
        #@constraint(model, q_qru[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0.0)
        
        # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
        if dev in idx.J_pqmax
            #@constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_0_ub[dev]*u_sum
            #- prm.dev.beta_ub[dev]*dev_p[tii] <= 0.0)
        end 

        # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
        if dev in idx.J_pqmin
            #@constraint(model, prm.dev.q_0_lb[dev]*u_sum
            #+ prm.dev.beta_lb[dev]*dev_p[tii]
            #+ q_qru[tii] - dev_q[tii] <= 0.0)
        end
    end
end

# misc penalty: maximum starts over multiple periods
for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
    # get the time periods: zhat_mxst
    T_su_max = idx.Ts_su_max[dev][w_ind] #get_tsumax(w_params, prm)
    #@constraint(model, sum(u_su_dev[tii] for tii in T_su_max; init=0.0) - w_params[3] <= 0.0)
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
soln_valid = quasiGrad.solution_status(model)

# did Gurobi find something valid?
if soln_valid == true

    # print
    println("Projection. ", termination_status(model),". objective value: ", objective_value(model))

    # return the solution
    for tii in prm.ts.time_keys
        if final_projection == true
            # no need to copy the binaries -- they are static
            #   ...but just in case:
            stt[:u_on_dev][tii][dev]  = copy(value(u_on_dev[tii]))
        else
            # copy the binary solution to a temporary location
            stt[:u_on_dev_GRB][tii][dev]  = copy(value(u_on_dev[tii]))
        end

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

    # clip, to help ensure feasibility -- (17c) sometimes causes "epsilon" infeasibility
    clip_for_feasibility!(idx, prm, qG, stt, sys)

    # if this is the final projection, update the update 
    # the u_sum and powers right here (used in clipping, so must be correct!)
    if final_projection == true
        qG.run_susd_updates = true
        quasiGrad.simple_device_statuses!(idx, prm, stt)
        quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    end
else
    # warn!
    @warn "Gurobi MILP projection failed (dev ($dev)) -- skip and try again later!"
    @assert 1 == 2
end

# %%
for tii in prm.ts.time_keys
println(stt[:u_on_dev][tii][58])
end