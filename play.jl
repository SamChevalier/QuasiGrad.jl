# %%
using BenchmarkTools

# %%
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

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
    dev_plb = stt[:u_on_dev][tii].*getindex.(prm.dev.p_lb,t_ind)
    dev_pub = stt[:u_on_dev][tii].*getindex.(prm.dev.p_ub,t_ind)
    dev_qlb = stt[:u_sum][tii].*getindex.(prm.dev.q_lb,t_ind)
    dev_qub = stt[:u_sum][tii].*getindex.(prm.dev.q_ub,t_ind)

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