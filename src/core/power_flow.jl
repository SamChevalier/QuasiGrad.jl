function solve_power_flow!(cgd::quasiGrad.ConstantGrad, grd::quasiGrad.Grad, idx::quasiGrad.Index, lbf::quasiGrad.LBFGS, mgd::quasiGrad.MasterGrad, ntk::quasiGrad.Network, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}; first_solve::Bool=true)
    # Note -- this is run *after* DCPF and q/v corrections (maybe..)
    # 
    if first_solve == true
        # 1. fire up lbfgs, as controlled by adam, WITH regularization + OPF
        # 2. after a short period, use Gurobi to solve successive power flows
        # 
        # step 1: lbfgs power flow
        #
        # turn on extra influence
        qG.eval_grad                       = true
        qG.include_energy_costs_lbfgs      = true
        qG.include_lbfgs_p0_regularization = true

        # set the loss function
        qG.pqbal_grad_type = "soft_abs"
        qG.pqbal_grad_eps2 = 1e-3

        # loop -- lbfgs
        run_lbfgs = true
        lbfgs_cnt = 0
        zt0       = 0.0

        # make a few copies, just in case
        stt.vm_copy    .= deepcopy.(stt.vm)
        stt.va_copy    .= deepcopy.(stt.va)
        stt.p_on_copy  .= deepcopy.(stt.p_on)
        stt.dev_q_copy .= deepcopy.(stt.dev_q)

        # re-initialize the lbf(gs) struct
        quasiGrad.flush_lbfgs!(lbf, prm, qG, stt)

        # initialize: compute all states and grads
        quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)

        # store the first value
        zt0 = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys) + sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)

        # loop -- lbfgs
        while run_lbfgs == true
            # take an lbfgs step
            emergency_stop = quasiGrad.solve_pf_lbfgs!(lbf, mgd, prm, qG, stt, upd)

            # save zpf BEFORE updating with the new state -- don't track bias terms
            for tii in prm.ts.time_keys
                lbf.step[:zpf_prev][tii] = (lbf.zpf[:zp][tii]+lbf.zpf[:zq][tii]) 
            end

            # compute all states and grads
            quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)

            # store the first value
            zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
            zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
            zt = zp + zq

            # print
            if qG.print_lbfgs_iterations == true
                ztr = round(zt; sigdigits = 3)
                zpr = round(zp; sigdigits = 3)
                zqr = round(zq; sigdigits = 3)
                stp = round(sum(lbf.step[:step][tii] for tii in prm.ts.time_keys)/sys.nT; sigdigits = 3)
                println("Total: $(ztr), P penalty: $(zpr), Q penalty: $(zqr), avg adam step: $(stp)!")
            end

            # increment
            lbfgs_cnt += 1

            # quit if the error gets too large relative to the first error
            if (lbfgs_cnt > qG.num_lbfgs_steps) || (zt > 1.25*zt0) || (emergency_stop == true)
                run_lbfgs = false
                if (emergency_stop == true) || (zt > 1.25*zt0)
                    @info "LBFGS failed -- error too high. Snapping state back!"

                    # update the copies of vm, va, p, and q
                    stt.vm    .= deepcopy.(stt.vm_copy)
                    stt.va    .= deepcopy.(stt.va_copy)
                    stt.p_on  .= deepcopy.(stt.p_on_copy)
                    stt.dev_q .= deepcopy.(stt.dev_q_copy)
                end
            end
        end
    end

    # step 2: Gurobi linear solve (projection)
    quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

end

# correct the reactive power injections into the network
function apply_q_injections!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # note -- this is a fairly approximate function
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
    Threads.@threads for tii in prm.ts.time_keys
        # at this time, compute the pr and cs upper and lower bounds across all devices
        stt.dev_qlb[tii] .= stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]
        stt.dev_qub[tii] .= stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]

        # for devices with reactive power equality constraints, just
        # set the associated upper and lower bounds to the given production
        for dev in idx.J_pqe
            stt.dev_qlb[tii][dev] = copy(stt.dev_q[tii][dev])
            stt.dev_qub[tii][dev] = copy(stt.dev_q[tii][dev])
        end

        # note: clipping is based on the upper/lower bounds, and not
        # based on the beta linking equations -- so, we just treat
        # that as a penalty, and not as a power balance factor
        # 
        # also, compute the dc line upper and lower bounds
        dcfr_qlb = prm.dc.qdc_fr_lb
        dcfr_qub = prm.dc.qdc_fr_ub
        dcto_qlb = prm.dc.qdc_to_lb
        dcto_qub = prm.dc.qdc_to_ub

        # how does balance work? for reactive power,
        # 0 = qb_slack + (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
        #
        # so, we take want to set:
        # -qb_slack = (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
        for bus in 1:sys.nb
            # reactive power balance
            qb_slack = 
                    # shunt        
                    sum(stt.sh_q[tii][sh] for sh in idx.sh[bus]; init=0.0) +
                    # acline
                    sum(stt.acline_qfr[tii][ln] for ln in idx.bus_is_acline_frs[bus]; init=0.0) + 
                    sum(stt.acline_qto[tii][ln] for ln in idx.bus_is_acline_tos[bus]; init=0.0) +
                    # xfm
                    sum(stt.xfm_qfr[tii][xfm] for xfm in idx.bus_is_xfm_frs[bus]; init=0.0) + 
                    sum(stt.xfm_qto[tii][xfm] for xfm in idx.bus_is_xfm_tos[bus]; init=0.0)
                    # dcline -- not included
                    # consumers (positive) -- not included
                    # producer (negative) -- not included

            # get limits -- Q
            pr_Qlb   = sum(stt.dev_qlb[tii][pr] for pr  in idx.pr[bus]; init=0.0)
            cs_Qlb   = sum(stt.dev_qlb[tii][cs] for cs  in idx.cs[bus]; init=0.0)
            pr_Qub   = sum(stt.dev_qub[tii][pr] for pr  in idx.pr[bus]; init=0.0) 
            cs_Qub   = sum(stt.dev_qub[tii][cs] for cs  in idx.cs[bus]; init=0.0)
            dcfr_Qlb = sum(dcfr_qlb[dcl]        for dcl in idx.bus_is_dc_frs[bus]; init=0.0)
            dcfr_Qub = sum(dcfr_qub[dcl]        for dcl in idx.bus_is_dc_frs[bus]; init=0.0)
            dcto_Qlb = sum(dcto_qlb[dcl]        for dcl in idx.bus_is_dc_tos[bus]; init=0.0)
            dcto_Qub = sum(dcto_qub[dcl]        for dcl in idx.bus_is_dc_tos[bus]; init=0.0) 
            
            # total: lb < -qb_slack < ub
            qub = cs_Qub + dcfr_Qub + dcto_Qub - pr_Qlb
            qlb = cs_Qlb + dcfr_Qlb + dcto_Qlb - pr_Qub

            # now, apply Q
            if -qb_slack >= qub
                # => println("ub limit")
                # max everything out
                for cs in idx.cs[bus]
                    stt.dev_q[tii][cs] = copy(stt.dev_qub[tii][cs])
                end
                for pr in idx.pr[bus]
                    stt.dev_q[tii][pr] = copy(stt.dev_qlb[tii][pr])
                end
                for dcl in idx.bus_is_dc_frs[bus]
                    stt.dc_qfr[tii][dcl] = copy(dcfr_qub[dcl])
                end
                for dcl in idx.bus_is_dc_tos[bus]
                    stt.dc_qto[tii][dcl] = copy(dcfr_qub[dcl])
                end
            elseif -qb_slack < qlb
                # => println("lb limit")

                # min everything out
                for cs in idx.cs[bus]
                    stt.dev_q[tii][cs] = copy(stt.dev_qlb[tii][cs])
                end
                for pr in idx.pr[bus]
                    stt.dev_q[tii][pr] = copy(stt.dev_qub[tii][pr])
                end
                for dcl in idx.bus_is_dc_frs[bus]
                    stt.dc_qfr[tii][dcl] = copy(dcfr_qlb[dcl])
                end
                for dcl in idx.bus_is_dc_tos[bus]
                    stt.dc_qto[tii][dcl] = copy(dcfr_qlb[dcl])
                end
            else # in the middle -- all good -- no need to copy
                # => println("middle")
                lb_dist  = -qb_slack - qlb
                bnd_dist = qub - qlb
                scale    = lb_dist/bnd_dist

                # apply
                for cs in idx.cs[bus]
                    stt.dev_q[tii][cs] = stt.dev_qlb[tii][cs] + scale*(stt.dev_qub[tii][cs] - stt.dev_qlb[tii][cs])
                end
                for pr in idx.pr[bus]
                    stt.dev_q[tii][pr] = stt.dev_qub[tii][pr] - scale*(stt.dev_qub[tii][pr] - stt.dev_qlb[tii][pr])
                end
                for dcl in idx.bus_is_dc_frs[bus]
                    stt.dc_qfr[tii][dcl] = dcfr_qlb[dcl] + scale*(dcfr_qub[dcl] - dcfr_qlb[dcl])
                end
                for dcl in idx.bus_is_dc_tos[bus]
                    stt.dc_qto[tii][dcl] = dcfr_qlb[dcl] + scale*(dcfr_qub[dcl] - dcfr_qlb[dcl])
                end
            end
        end
    end
end

function flush_lbfgs!(lbf::quasiGrad.LBFGS, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # no need to touch the map -- not changing between powerflow solves!
    for tii in prm.ts.time_keys
        lbf.step[:zpf_prev][tii]    = 0.0
        lbf.step[:beta1_decay][tii] = 1.0
        lbf.step[:beta2_decay][tii] = 1.0
        lbf.step[:m][tii]           = 0.0
        lbf.step[:v][tii]           = 0.0      
        lbf.step[:mhat][tii]        = 0.0      
        lbf.step[:vhat][tii]        = 0.0      
        lbf.step[:step][tii]        = 0.0 # this is reset at the first itertation!   
        lbf.step[:alpha_0][tii]     = qG.lbfgs_adam_alpha_0
    end

    # indices to track where previous differential vectors are stored --
    # lbfgs_idx[1] is always the most recent data, and lbfgs_idx[end] is the oldest
    lbf.idx .= Int64(0)

    # update the dict for regularizing the solution
    for tii in prm.ts.time_keys
        lbf.p0[:p_on][tii] .= copy.(stt.p_on[tii])
    end
end

function solve_parallel_linear_pf_with_Gurobi!(idx::quasiGrad.Index, ntk::quasiGrad.Network, prm::quasiGrad.Param, qG::quasiGrad.QG,  stt::quasiGrad.State, sys::quasiGrad.System)
    # Solve linearized power flow with Gurobi -- use margin tinkering to guarentee convergence. Only consinder upper 
    # and lower bounds on the p/q production (no other limits).
    #
    # ask Gurobi to solve a linearized power flow. two options here:
    #   1) define device variables which are bounded, and then insert them into the power balance expression
    #   2) just define power balance bounds based on device characteristics, and then, at the end, optimally
    #      dispatch the devices via updating.
    # 
    # I am pretty sure 1) is slower, but it is (1) safer, (2) simpler, and (3) more flexible -- and it will
    # require less trouble-shooting. Plus, it will be easy to update/improve in the future! Go with it.
    #
    # here is power balance: p_pr - p_cs - pdc = p_lines/xfm/shunt => this is typical.
    #
    # vm0 = stt.vm[tii]
    # va0 = stt.va[tii][2:end-1]
    #
    # bias point: stt[:pinj0, qinj0] === Y * stt[:vm, va]
    qG.compute_pf_injs_with_Jac = true

    @info "Running parallel linearized power flows across $(sys.nT) time periods."
    
    # loop over time
    Threads.@threads for tii in prm.ts.time_keys

        # initialize
        first_run = true
        run_pf    = true
        pf_cnt    = 0  # successes

        # update y_bus -- this only needs to be done once per time, 
        # since xfm/shunt values are not changing between iterations
        quasiGrad.update_Ybus!(idx, ntk, prm, stt, sys, tii)

        # update the line flow admittance matrices (only "fr" used)
        quasiGrad.update_Yflow!(idx, ntk, prm, stt, sys, tii)

        # loop over pf solves
        while run_pf == true
            t1 = time()

            # build an empty model!
            model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => 1))
            set_string_names_on_creation(model, false)

            # lower the tolerance a bit.. => doesn't help!
                # => quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", 1e-3)
                # => quasiGrad.set_optimizer_attribute(model, "OptimalityTol",  1e-3)

            # increment
            pf_cnt += 1

            # first, rebuild jacobian, and update base points: stt.pinj0, stt.qinj0
            quasiGrad.build_Jac_and_pq0!(ntk, qG, stt, sys, tii)
            quasiGrad.build_Jac_sfr_and_sfr0!(idx, ntk, prm, stt, sys, tii)

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])

            # assign
            dvm = @view x_in[1:sys.nb]
            dva = @view x_in[(sys.nb+1):end]
            set_start_value.(dvm, stt.vm[tii])
            set_start_value.(dva, @view stt.va[tii][2:end])

            # voltage penalty -- penalizes voltages out-of-bounds
            @variable(model, vm_penalty[1:sys.nb])

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
            @variable(model, dev_p_vars[1:sys.ndev])
            @variable(model, dev_q_vars[1:sys.ndev])
            set_start_value.(dev_p_vars, stt.dev_p[tii])
            set_start_value.(dev_q_vars, stt.dev_q[tii])

            # call the bounds -- note: this is fairly approximate,
            # since these bounds do not include, e.g., ramp rate constraints
            # between the various time windows -- this will be addressed in the
            # final, constrained power flow solve                            # => ignore binaries?
                # => stt.dev_plb[tii] .= stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii]    # => stt.dev_plb[tii] .= prm.dev.p_lb_tmdv[tii]
                # => stt.dev_pub[tii] .= stt.u_on_dev[tii].*prm.dev.p_ub_tmdv[tii]    # => stt.dev_pub[tii] .= prm.dev.p_ub_tmdv[tii]
                # => stt.dev_qlb[tii] .= stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]       # => stt.dev_qlb[tii] .= prm.dev.q_lb_tmdv[tii]
                # => stt.dev_qub[tii] .= stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]       # => stt.dev_qub[tii] .= prm.dev.q_ub_tmdv[tii]

            # ignore binaries
            stt.dev_plb[tii] .= 0.0
            stt.dev_pub[tii] .= prm.dev.p_ub_tmdv[tii]
            stt.dev_qlb[tii] .= min.(0.0, prm.dev.q_lb_tmdv[tii])
            stt.dev_qub[tii] .= max.(0.0, prm.dev.q_ub_tmdv[tii])

            # first, define p_on at this time
                # => p_on = dev_p_vars - stt.p_su[tii] - stt.p_sd[tii]
            @constraint(model, stt.dev_plb[tii] .+ stt.p_su[tii] .+ stt.p_sd[tii] .<= dev_p_vars .<= stt.dev_pub[tii] .+ stt.p_su[tii] .+ stt.p_sd[tii])
            # alternative: => @constraint(model, dev_plb .<= dev_p_vars - stt.p_su[tii] - stt.p_sd[tii] .<= dev_pub)
            @constraint(model, stt.dev_qlb[tii] .<= dev_q_vars .<= stt.dev_qub[tii])

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
            for dev in idx.pr_devs # producers
                add_to_expression!(nodal_p[idx.device_to_bus[dev]], dev_p_vars[dev])
                add_to_expression!(nodal_q[idx.device_to_bus[dev]], dev_q_vars[dev])
            end
            for dev in idx.cs_devs # consumers
                add_to_expression!(nodal_p[idx.device_to_bus[dev]], -dev_p_vars[dev])
                add_to_expression!(nodal_q[idx.device_to_bus[dev]], -dev_q_vars[dev])
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, prm.bus.vm_lb .- vm_penalty .<= stt.vm[tii] .+ dvm)
            @constraint(model,                                 stt.vm[tii] .+ dvm .<= prm.bus.vm_ub .+ vm_penalty)

            # mapping
            JacP_noref      = @view ntk.Jac[tii][1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
            JacQ_noref      = @view ntk.Jac[tii][(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]
            @constraint(model, JacP_noref*x_in .+ stt.pinj0[tii] .== nodal_p)
            @constraint(model, JacQ_noref*x_in .+ stt.qinj0[tii] .== nodal_q)

            # finally, bound the apparent power flow in the lines and transformers
            JacSfr_acl_noref = @view ntk.Jac_sflow_fr[tii][1:sys.nl,       [1:sys.nb; (sys.nb+2):end]]
            JacSfr_xfm_noref = @view ntk.Jac_sflow_fr[tii][(sys.nl+1):end, [1:sys.nb; (sys.nb+2):end]]
            @constraint(model, JacSfr_acl_noref*x_in + stt.acline_sfr[tii] .<= 1.2 .* prm.acline.mva_ub_nom)
            @constraint(model, JacSfr_xfm_noref*x_in + stt.xfm_sfr[tii]    .<= 1.2 .* prm.xfm.mva_ub_nom)
                        
            if first_run == true
                first_run = false # toggle
                # on the first run, just find a pf solution for active power, and
                # penalize any voltage violation
                obj = @expression(model, 
                        1e3*(vm_penalty'*vm_penalty) + 
                        x_in'*x_in +
                        # => (stt.pinj0[tii] .- nodal_p)'*(stt.pinj0[tii] .- nodal_p) + 
                        # => (stt.qinj0[tii] .- nodal_q)'*(stt.qinj0[tii] .- nodal_q) + 
                        (stt.dev_q[tii] .- dev_q_vars)'*(stt.dev_q[tii] .- dev_q_vars) +
                        10.0*(stt.dev_p[tii] .- dev_p_vars)'*(stt.dev_p[tii] .- dev_p_vars))
            else 
                obj = @expression(model, 
                        50.0*(vm_penalty'*vm_penalty) + 
                        x_in'*x_in +
                        # => (stt.pinj0[tii] .- nodal_p)'*(stt.pinj0[tii] .- nodal_p) + 
                        # => (stt.qinj0[tii] .- nodal_q)'*(stt.qinj0[tii] .- nodal_q) + 
                        (stt.dev_q[tii] .- dev_q_vars)'*(stt.dev_q[tii] .- dev_q_vars) +
                        10.0*(stt.dev_p[tii] .- dev_p_vars)'*(stt.dev_p[tii] .- dev_p_vars))
            end

            # set the objective
            @objective(model, Min, obj)

            # solve --- 
            build_time = time() - t1
            t2 = time()
            optimize!(model)
            solve_time = time() - t2

            # print time stats
            if qG.print_linear_pf_iterations == true
                println("build time: ", round(build_time, sigdigits = 4), ". solve time: ", round(solve_time, sigdigits = 4))
            end

            # test solution!
            soln_valid = solution_status(model)

            # test validity
            if soln_valid == true
                # no matter what, we update the voltage soluion
                stt.vm[tii]        .= value.(dvm) .+ stt.vm[tii]
                stt.va[tii][2:end] .= value.(dva) .+ @view stt.va[tii][2:end]

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

                # take the norm of dv and da
                max_dx = maximum(abs.(value.(x_in)))
                
                # println("========================================================")
                if qG.print_linear_pf_iterations == true
                    println(termination_status(model), ". time: $(tii). ","objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
                end
                # println("========================================================")
                #
                # shall we terminate?
                if (max_dx < qG.max_pf_dx) || (pf_cnt == qG.max_linear_pfs)
                    run_pf = false
                end
            else
                # the solution is NOT valid, so we should increase bounds and try again
                @warn "Linearized power flow failed at time ($tii) -- nothing we can do!"
                @assert 1 == 2
                run_pf = false
            end
        end
    end
end

function update_states_and_grads_for_solve_pf_lbfgs!(cgd::quasiGrad.ConstantGrad, grd::quasiGrad.Grad, idx::quasiGrad.Index, lbf::quasiGrad.LBFGS, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # in this function, we only update the states and gradients needed
    # to solve a single-time-period ACOPF with lbfgs:
    # 1) flush
    # 2) clip (including both p_on, based on binary values and dev_q)
    # 3) line, xfm, and shunt
    # 4a) device power (p/q) 
    # 4b) optional: power costs
    # 5) power balance
    # 6) score quadratic distance metric 
    # 7) run the master grad

    # if we are here, we want to make sure we are NOT running su/sd updates
    qG.run_susd_updates = false

    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    # clip all basic states (i.e., the states which are iterated on)
    qG.clip_pq_based_on_bins = false
    quasiGrad.clip_all!(prm, qG, stt)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, prm, qG, stt)

    # device powers
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, qG, stt)

    # include OPF costs? this regularizes/biases the solution
    if qG.include_energy_costs_lbfgs
        quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    end

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

    # take quadratic distance and directly take/apply the master gradient
    if qG.include_lbfgs_p0_regularization
        quasiGrad.quadratic_distance!(lbf, mgd, prm, qG, stt)
    end

    # score
    quasiGrad.score_solve_pf!(lbf, prm, stt)

    # compute the master grad
    quasiGrad.master_grad_solve_pf!(cgd, grd, idx, mgd, prm, qG, stt, sys)
end

# lbfgs
function solve_pf_lbfgs!(lbf::quasiGrad.LBFGS, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # note: lbf.idx is a set of ordered indices, where the first is the most
    #       recent step information, and the last is the oldest step information
    #       in the following order: (k-1), (k-2)
    #
    # prepare the lbfgs structures -- x and gradf
    emergency_stop = false
    for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
        state = getfield(stt,var_key)
        grad  = getfield(mgd,var_key)
        for tii in prm.ts.time_keys
            # states to update
            if var_key in keys(upd)
                lbf.state[:x_now][tii][lbf.map[var_key]]     .= copy.(@view state[tii][upd[var_key][tii]]) # no update_subset needed on lbfgs side
                lbf.state[:gradf_now][tii][lbf.map[var_key]] .= copy.(@view grad[tii][upd[var_key][tii]]) # no update_subset needed on lbfgs side
            else
                lbf.state[:x_now][tii][lbf.map[var_key]]     .= copy.(state[tii])
                lbf.state[:gradf_now][tii][lbf.map[var_key]] .= copy.(grad[tii])
            end
        end
    end

    # if this is the very first iteration, just take a gradient step
    if sum(lbf.idx) == 0
        # we solve pf at each instant, so loop over all time!
        for tii in prm.ts.time_keys
            # if this is the very first iteration, just take a tiny gradient step
            #
            # want: maximum(grad)*step_size = 1e-4, to, step_size = 1e-4/maximum(grad)
            step_size = (1e-5)/maximum(lbf.state[:gradf_now][tii])
            lbf.state[:x_new][tii] .= lbf.state[:x_now][tii] .- step_size*lbf.state[:gradf_now][tii]
        end

        # pass x back into the state vector
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            state = getfield(stt,var_key)
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    state[tii][upd[var_key][tii]] .= @view lbf.state[:x_new][tii][lbf.map[var_key]] # no update_subset needed
                else
                    state[tii]                    .= @view lbf.state[:x_new][tii][lbf.map[var_key]]
                end
            end
        end

        # update the lbfgs states and grads
        for tii in prm.ts.time_keys
            lbf.state[:x_prev][tii]     .= copy.(lbf.state[:x_now][tii])
            lbf.state[:gradf_prev][tii] .= copy.(lbf.state[:gradf_now][tii])
        end

        # now, let's initialize lbfgs_idx
        lbf.idx[1] = 1
    else
        # we solve pf at each instant, so loop over all time!
        # => for tii in prm.ts.time_keys
        @batch per=core for tii in prm.ts.time_keys
        # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys

            # udpdate the state difference
                      idx_km1                     = lbf.idx[1]
            @turbo    lbf.diff[:s][tii][idx_km1] .= lbf.state[:x_now][tii]     .- lbf.state[:x_prev][tii]
            @turbo    lbf.diff[:y][tii][idx_km1] .= lbf.state[:gradf_now][tii] .- lbf.state[:gradf_prev][tii]
            @fastmath rho                         = quasiGrad.dot(lbf.diff[:s][tii][idx_km1], lbf.diff[:y][tii][idx_km1])
            if abs(rho) < 1e-7
                # in this case, lbfgs is stalling out and might return a NaN if we're not careful
                emergency_stop = true
                @info "Breaking out of lbfgs loop! s'*y too small. NaN possible."
                break # this breaks from everything 
            end
            
            # if not y^T * s is not too small, go and ahead and take its inverse: rho === 1/(y^T * s)
            lbf.state[:rho][tii][idx_km1] = 1.0/rho

            # now, double-loop and compute lbfgs values
            lbf.state[:q][tii] .= copy.(lbf.state[:gradf_now][tii])
            for ii in lbf.idx[lbf.idx .!= 0] # k-1, k-2, ..., k-m
                @fastmath lbf.state[:alpha][tii][ii] = lbf.state[:rho][tii][ii]*quasiGrad.dot(lbf.diff[:s][tii][ii], lbf.state[:q][tii])
                @fastmath lbf.state[:q][tii]       .-= lbf.state[:alpha][tii][ii].*lbf.diff[:y][tii][ii]
            end
            
            # set "r", which will be H*grad
            @fastmath lbf.state[:r][tii] .= lbf.state[:q][tii].*(quasiGrad.dot(lbf.diff[:s][tii][idx_km1], lbf.diff[:y][tii][idx_km1])/quasiGrad.dot(lbf.diff[:y][tii][idx_km1], lbf.diff[:y][tii][idx_km1]))
            
            # compute H*grad
            for ii in reverse(lbf.idx[lbf.idx .!= 0]) # k-m, k-m+1, ..., k-1
                # skip beta -- defined implicitly below
                @fastmath lbf.state[:r][tii] .+= lbf.diff[:s][tii][ii].*(lbf.state[:alpha][tii][ii] - lbf.state[:rho][tii][ii]*quasiGrad.dot(lbf.diff[:y][tii][ii], lbf.state[:r][tii]))
            end

            # step size: let adam control?
            if sum(lbf.idx) == 1
                # this is the first step, so just use qG.initial_pf_lbfgs_step (about ~0.1)
                lbf.step[:step][tii] = qG.initial_pf_lbfgs_step
            else
                # decay beta
                lbf.step[:beta1_decay][tii] = lbf.step[:beta1_decay][tii]*qG.beta1
                lbf.step[:beta2_decay][tii] = lbf.step[:beta2_decay][tii]*qG.beta2

                # have the STEP take a step with adam!
                grad                 = ((lbf.zpf[:zp][tii]+lbf.zpf[:zq][tii]) - lbf.step[:zpf_prev][tii])/lbf.step[:step][tii]
                lbf.step[:m][tii]    = qG.beta1.*lbf.step[:m][tii] + (1.0-qG.beta1).*grad
                lbf.step[:v][tii]    = qG.beta2.*lbf.step[:v][tii] + (1.0-qG.beta2).*grad.^2.0
                lbf.step[:mhat][tii] = lbf.step[:m][tii]/(1.0-lbf.step[:beta1_decay][tii])
                lbf.step[:vhat][tii] = lbf.step[:v][tii]/(1.0-lbf.step[:beta2_decay][tii])
                lbf.step[:step][tii] = lbf.step[:step][tii] - lbf.step[:alpha_0][tii]*lbf.step[:mhat][tii]/(sqrt.(lbf.step[:vhat][tii]) .+ qG.eps)
            end

            # lbfgs step
            @turbo lbf.state[:x_new][tii] .= lbf.state[:x_now][tii] .- lbf.step[:step][tii].*lbf.state[:r][tii]
        end

        # pass x back into the state vector
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            state = getfield(stt,var_key)
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    state[tii][upd[var_key][tii]] .= copy.(@view lbf.state[:x_new][tii][lbf.map[var_key]]) # no update_subset needed
                else
                    state[tii]                    .= copy.(@view lbf.state[:x_new][tii][lbf.map[var_key]])
                end
            end
        end

        # update the lbfgs states and grads
        for tii in prm.ts.time_keys
            lbf.state[:x_prev][tii]     .= copy.(lbf.state[:x_now][tii])
            lbf.state[:gradf_prev][tii] .= copy.(lbf.state[:gradf_now][tii])
        end

        # finally, update the lbfgs indices -- rule: lbfgs_idx[1] is where 
        # we write the newest data, and every next index is successively
        # older data -- oldest data gets bumped when the dataset if full.
        #
        # v = [data(0), -, -]  => lbfgs_idx = [1,0,0]
        #
        # v = [data(0), data(1), -]  => lbfgs_idx = [2,1,0]
        #
        # v = [data(0), data(1), data(2)]  => lbfgs_idx = [3,2,1]
        # 
        # v = [data(3), data(1), data(2)]  => lbfgs_idx = [1,3,2]
        #
        # v = [data(3), data(4), data(2)]  => lbfgs_idx = [2,1,3]
        #
        # ....
        #
        # so, 1 becomes 2, 2 becomes 3, etc. :
        if 0 ∈ lbf.idx
            circshift!(lbf.idx, 1)
            lbf.idx[1] = lbf.idx[2] + 1
        else
            circshift!(lbf.idx, 1)
        end
    end

    # output
    return emergency_stop
end

function quadratic_distance!(lbf::quasiGrad.LBFGS, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=core for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # grab the distance between p_on and its initial value -- this is something we 
        # minimize, so the regularization function is positive valued
        # => value not needed: zdist      =   qG.cdist_psolve*(stt.p_on[tii] - lbf.p0[:p_on][tii]).^2
            # => zdist_grad = 2.0.*qG.cdist_psolve*(stt.p_on[tii] .- lbf.p0[:p_on][tii])

        # now, apply the gradients directly (no need to use dp_alpha!())
        mgd.p_on[tii] .+= (2.0*qG.cdist_psolve).*(stt.p_on[tii] .- lbf.p0[:p_on][tii])
    end
end

function build_Jac_and_pq0!(ntk::quasiGrad.Network, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)

    # build the admittance structure
    NY  = [ntk.Ybus_real[tii] -ntk.Ybus_imag[tii];
          -ntk.Ybus_imag[tii] -ntk.Ybus_real[tii]]

    # complex voltage
    stt.cva[tii] .= cos.(stt.va[tii])
    stt.sva[tii] .= sin.(stt.va[tii])
    stt.vr[tii]  .= stt.vm[tii].*stt.cva[tii]
    stt.vi[tii]  .= stt.vm[tii].*stt.sva[tii]

    # complex current:
        # => ic = Yc*vc = (Ycr + j*Yci) * (vr + j*vi)
        # => Ir = Ycr*vr - Yci*vi
        # => Ii = Yci*vr + Ycr*vi
    stt.Ir[tii] .= ntk.Ybus_real[tii]*stt.vr[tii] .- ntk.Ybus_imag[tii]*stt.vi[tii]
    stt.Ii[tii] .= ntk.Ybus_imag[tii]*stt.vr[tii] .+ ntk.Ybus_real[tii]*stt.vi[tii]

    # Populate MI
    MIr = quasiGrad.spdiagm(sys.nb, sys.nb, stt.Ir[tii])
    MIi = quasiGrad.spdiagm(sys.nb, sys.nb, stt.Ii[tii])
    MI  = [MIr  MIi;
          -MIi  MIr]

    # Populate MV
    MVr = quasiGrad.spdiagm(sys.nb, sys.nb, stt.vr[tii])
    MVi = quasiGrad.spdiagm(sys.nb, sys.nb, stt.vi[tii])
    MV  = [MVr -MVi;
           MVi  MVr]

    # Populate RV
    RV = [quasiGrad.spdiagm(sys.nb, sys.nb, stt.cva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, .-stt.vi[tii]); 
          quasiGrad.spdiagm(sys.nb, sys.nb, stt.sva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, stt.vr[tii])];

    # build full Jacobian
    ntk.Jac[tii] .= (MI + MV*NY)*RV

    # also compute injections?
    if qG.compute_pf_injs_with_Jac
        # complex coordinates -- don't actually do this :)
        # => Yb = Ybus_real + im*Ybus_imag
        # => vc = stt.vm[tii].*(exp.(im*stt.va[tii]))
        # => ic = Yb*vc
        # => sc = vc.*conj.(ic)
        # => pinj = real(sc)
        # => qinj = imag(sc)
        stt.pinj0[tii] .= stt.vr[tii].*stt.Ir[tii] .+ stt.vi[tii].*stt.Ii[tii]
        stt.qinj0[tii] .= stt.vi[tii].*stt.Ir[tii] .- stt.vr[tii].*stt.Ii[tii]
    end
end

function build_Jac_sfr_and_sfr0!(idx::quasiGrad.Index, ntk::quasiGrad.Network, prm::quasiGrad.Param, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # NOTE: build_Jac_and_pq0! must be called first (to update vr/vi/cva/sva)
    #
    # first, update the line flows
    quasiGrad.update_acline_sfr_flows!(idx, prm, stt, tii)
    quasiGrad.update_xfm_sfr_flows!(idx, prm, stt, tii)

    # build the admittance structure
    NYf = [ntk.Yflow_fr_real[tii] -ntk.Yflow_fr_imag[tii];
          -ntk.Yflow_fr_imag[tii] -ntk.Yflow_fr_real[tii]]

    # get current flows
    stt.Ir_flow_fr[tii] .= ntk.Yflow_fr_real[tii]*stt.vr[tii] .- ntk.Yflow_fr_imag[tii]*stt.vi[tii]
    stt.Ii_flow_fr[tii] .= ntk.Yflow_fr_imag[tii]*stt.vr[tii] .+ ntk.Yflow_fr_real[tii]*stt.vi[tii]

    # Populate MI
    MIr = quasiGrad.spdiagm(sys.nac, sys.nac, stt.Ir_flow_fr[tii])*ntk.Efr
    MIi = quasiGrad.spdiagm(sys.nac, sys.nac, stt.Ii_flow_fr[tii])*ntk.Efr
    MI  = [MIr MIi;
          -MIi MIr]

    # Populate MV
    MVr = quasiGrad.spdiagm(sys.nac, sys.nac, ntk.Efr*stt.vr[tii])
    MVi = quasiGrad.spdiagm(sys.nac, sys.nac, ntk.Efr*stt.vi[tii])
    MV  = [MVr -MVi;
           MVi  MVr]

    # Populate RV
    RV = [quasiGrad.spdiagm(sys.nb, sys.nb, stt.cva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, .-stt.vi[tii]); 
          quasiGrad.spdiagm(sys.nb, sys.nb, stt.sva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, stt.vr[tii])];
    
    # build Jacobian
    ntk.Jac_pq_flow_fr[tii] = (MI + MV*NYf)*RV

    # now, populate flow vectors
    for ln in 1:sys.nl
        if stt.acline_sfr[tii][ln] < 1e-2
            # in this case, just skip -- flow is so small that it probably doesn't matter anyways
            stt.pflow_over_sflow_fr[tii][ln] = stt.acline_pfr[tii][ln]./1e-2
            stt.qflow_over_sflow_fr[tii][ln] = stt.acline_qfr[tii][ln]./1e-2
        else
            stt.pflow_over_sflow_fr[tii][ln] = stt.acline_pfr[tii][ln]./stt.acline_sfr[tii][ln]
            stt.qflow_over_sflow_fr[tii][ln] = stt.acline_qfr[tii][ln]./stt.acline_sfr[tii][ln] 
        end
    end

    # now, populate flow vectors
    for xfm in 1:sys.nx
        if stt.xfm_sfr[tii][xfm] < 1e-2
            # in this case, just skip -- flow is so small that it probably doesn't matter anyways
            stt.pflow_over_sflow_fr[tii][xfm+sys.nl] = stt.xfm_pfr[tii][xfm]./1e-2
            stt.qflow_over_sflow_fr[tii][xfm+sys.nl] = stt.xfm_qfr[tii][xfm]./1e-2
        else
            stt.pflow_over_sflow_fr[tii][xfm+sys.nl] = stt.xfm_pfr[tii][xfm]./stt.xfm_sfr[tii][xfm]
            stt.qflow_over_sflow_fr[tii][xfm+sys.nl] = stt.xfm_qfr[tii][xfm]./stt.xfm_sfr[tii][xfm] 
        end
    end

    # first, populate V -> S
    ntk.Jac_sflow_fr[tii][:,1:sys.nb]       = stt.pflow_over_sflow_fr[tii].*(@view ntk.Jac_pq_flow_fr[tii][1:sys.nac      , 1:sys.nb]) .+ 
                                              stt.qflow_over_sflow_fr[tii].*(@view ntk.Jac_pq_flow_fr[tii][(sys.nac+1):end, 1:sys.nb])
    # second, populate Th -> S
    ntk.Jac_sflow_fr[tii][:,(sys.nb+1):end] = stt.pflow_over_sflow_fr[tii].*(@view ntk.Jac_pq_flow_fr[tii][1:sys.nac      , (sys.nb+1):end]) .+ 
                                              stt.qflow_over_sflow_fr[tii].*(@view ntk.Jac_pq_flow_fr[tii][(sys.nac+1):end, (sys.nb+1):end])
end

function build_Jac_sto!(ntk::quasiGrad.Network, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # NOTE: build_Jac_and_pq0! must be called first (to update vr/vi/cva/sva)
    #
    # build the admittance structure
    NYf = [ntk.Yflow_to_real[tii] -ntk.Yflow_to_imag[tii];
          -ntk.Yflow_to_imag[tii] -ntk.Yflow_to_real[tii]]

    # get current flows
    stt.Ir_flow_to[tii] .= ntk.Yflow_to_real[tii]*stt.vr[tii] .- ntk.Yflow_to_imag[tii]*stt.vi[tii]
    stt.Ii_flow_to[tii] .= ntk.Yflow_to_imag[tii]*stt.vr[tii] .+ ntk.Yflow_to_real[tii]*stt.vi[tii]

    # Populate MI
    MIr = quasiGrad.spdiagm(sys.nac, sys.nac, stt.Ir_flow_to[tii])*ntk.Eto
    MIi = quasiGrad.spdiagm(sys.nac, sys.nac, stt.Ii_flow_to[tii])*ntk.Eto
    MI  = [MIr MIi;
          -MIi MIr]

    # Populate MV
    MVr = quasiGrad.spdiagm(sys.nac, sys.nac, ntk.Eto*stt.vr[tii])
    MVi = quasiGrad.spdiagm(sys.nac, sys.nac, ntk.Eto*stt.vi[tii])
    MV  = [MVr -MVi;
           MVi  MVr]

    # Populate RV
    RV = [quasiGrad.spdiagm(sys.nb, sys.nb, stt.cva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, .-stt.vi[tii]); 
          quasiGrad.spdiagm(sys.nb, sys.nb, stt.sva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, stt.vr[tii])];
    
    # build Jacobian
    ntk.Jac_pq_flow_to[tii] = (MI + MV*NYf)*RV

    # now, populate flow vectors
    for ln in 1:sys.nl
        if stt.acline_sto[tii][ln] < 1e-2
            # in this case, just skip -- flow is so small that it probably doesn't matter anyways
            stt.pflow_over_sflow_to[tii][ln] = stt.acline_pto[tii][ln]./1e-2
            stt.qflow_over_sflow_to[tii][ln] = stt.acline_qto[tii][ln]./1e-2
        else
            stt.pflow_over_sflow_to[tii][ln] = stt.acline_pto[tii][ln]./stt.acline_sto[tii][ln]
            stt.qflow_over_sflow_to[tii][ln] = stt.acline_qto[tii][ln]./stt.acline_sto[tii][ln] 
        end
    end

    # now, populate flow vectors
    for xfm in 1:sys.nx
        if stt.xfm_sto[tii][xfm] < 1e-2
            # in this case, just skip -- flow is so small that it probably doesn't matter anyways
            stt.pflow_over_sflow_to[tii][xfm+sys.nl] = stt.xfm_pto[tii][xfm]./1e-2
            stt.qflow_over_sflow_to[tii][xfm+sys.nl] = stt.xfm_qto[tii][xfm]./1e-2
        else
            stt.pflow_over_sflow_to[tii][xfm+sys.nl] = stt.xfm_pto[tii][xfm]./stt.xfm_sto[tii][xfm]
            stt.qflow_over_sflow_to[tii][xfm+sys.nl] = stt.xfm_qto[tii][xfm]./stt.xfm_sto[tii][xfm] 
        end
    end

    # first, populate V -> S
    ntk.Jac_sflow_to[tii][:,1:sys.nb]       = stt.pflow_over_sflow_to[tii].*(@view ntk.Jac_pq_flow_to[tii][1:sys.nac      , 1:sys.nb]) .+ 
                                              stt.qflow_over_sflow_to[tii].*(@view ntk.Jac_pq_flow_to[tii][(sys.nac+1):end, 1:sys.nb])
    # second, populate Th -> S
    ntk.Jac_sflow_to[tii][:,(sys.nb+1):end] = stt.pflow_over_sflow_to[tii].*(@view ntk.Jac_pq_flow_to[tii][1:sys.nac      , (sys.nb+1):end]) .+ 
                                              stt.qflow_over_sflow_to[tii].*(@view ntk.Jac_pq_flow_to[tii][(sys.nac+1):end, (sys.nb+1):end])
end