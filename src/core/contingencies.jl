function solve_ctgs!(
    bit::quasiGrad.Bit,
    cgd::quasiGrad.Cgd,
    ctb::Vector{Vector{Float64}},        
    ctd::Vector{Vector{Float64}},   
    flw::quasiGrad.Flow,
    grd::quasiGrad.Grad, 
    idx::quasiGrad.Idx, 
    mgd::quasiGrad.Mgd, 
    ntk::quasiGrad.Ntk, 
    prm::quasiGrad.Param, 
    qG::quasiGrad.QG,
    scr::Dict{Symbol, Float64},
    stt::quasiGrad.State, 
    sys::quasiGrad.System,                                          
    wct::Vector{Vector{Int64}})
    # this script solves AND scores 
    #
    # loop over each time period and compute the power injections
    #
    # This step is contingency invariant -- i.e., each ctg will use this information

    # ===========================
    # ctb = base theta solutions, across time (then rank-1 corrected)
    # ctd = contingency gradient solutions (across time.. (used to be across ctgs)) solved
    #       on the base case (then rank-1 corrected)
    # ===========================
    
    # reset ===
        # => scr[:zctg_min] = 0.0
        # => scr[:zctg_avg] = 0.0
    zctg_min = 0.0
    zctg_avg = 0.0

    # how many ctgs 
    num_wrst = Int64(ceil(qG.frac_ctg_keep*sys.nctg/2))  # in case n_ctg is odd, and we want to keep all!
    num_rnd  = Int64(floor(qG.frac_ctg_keep*sys.nctg/2)) # in case n_ctg is odd, and we want to keep all!
    num_ctg  = num_wrst + num_rnd

    if qG.score_all_ctgs == true
        ###########################################################
        @info "Warning -- scoring all contingencies! No gradients."
        ###########################################################
    end

    # loop over time
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
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
            sum(stt.dev_p[tii][pr] for pr in idx.pr_devs) -
            sum(stt.dev_p[tii][cs] for cs in idx.cs_devs) - 
            sum(stt.sh_p[tii])

        # loop over each bus
        for bus in 1:sys.nb
            # active power balance
            flw.p_inj[tii][bus] = 
                sum(stt.dev_p[tii][pr] for pr in idx.pr[bus]; init=0.0) - 
                sum(stt.dev_p[tii][cs] for cs in idx.cs[bus]; init=0.0) - 
                sum(stt.sh_p[tii][sh] for sh in idx.sh[bus]; init=0.0) - 
                sum(stt.dc_pfr[tii][dc_fr] for dc_fr in idx.bus_is_dc_frs[bus]; init=0.0) - 
                sum(stt.dc_pto[tii][dc_to] for dc_to in idx.bus_is_dc_tos[bus]; init=0.0) - 
                ntk.alpha*p_slack
        end

        # also, we need to update the flows on all lines! and the phase shift
        flw.ac_qfr[tii][idx.ac_line_flows] .= stt.acline_qfr[tii]
        flw.ac_qfr[tii][idx.ac_xfm_flows]  .= stt.xfm_qfr[tii]
        flw.ac_qto[tii][idx.ac_line_flows] .= stt.acline_qto[tii]
        flw.ac_qto[tii][idx.ac_xfm_flows]  .= stt.xfm_qto[tii]
        flw.ac_phi[tii][idx.ac_phi]        .= stt.phi[tii]

        # compute square flows
        flw.qfr2[tii] .= flw.ac_qfr[tii].^2
        flw.qto2[tii] .= flw.ac_qto[tii].^2

        # solve for the flows across each ctg
        #   p  =  @view flw.p_inj[2:end]
        flw.bt[tii] .= .-flw.ac_phi[tii].*ntk.b
        # now, we have flw.p_inj = Yb*theta + E'*bt
        #   c = p - ntk.ErT*bt
        #
        # simplified:
        # => flw.c .= (@view flw.p_inj[2:end]) .- ntk.ErT*flw.bt
            # this is a little **odd**, but it's fine (the first use of flw.c is just for storage!)
        quasiGrad.mul!(flw.c[tii], ntk.ErT, flw.bt[tii])
        flw.c[tii] .= (@view flw.p_inj[tii][2:end]) .- flw.c[tii]

        # solve the base case with pcg
        if qG.base_solver == "lu"
            ctb[tii] .= ntk.Ybr\flw.c[tii]

        # error with this type !!!
        # elseif qG.base_solver == "cholesky"
        #    ctb[tii]  = ntk.Ybr_Ch\c

        elseif qG.base_solver == "pcg"
            if sys.nb <= qG.min_buses_for_krylov
                # too few buses -- just use LU
                ctb[tii] .= ntk.Ybr\flw.c[tii]
            else
                # solve with a hot start!
                #
                # note: ctg[:ctb][tii][end] is modified in place,
                # and it represents the base case solution
                _, ch = quasiGrad.cg!(ctb[tii], ntk.Ybr, flw.c[tii], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
                
                # test the krylov solution
                if ~(ch.isconverged)
                    println("Krylov failed -- using LU backup (ctg flows)!")
                    ctb[tii] = ntk.Ybr\flw.c[tii]
                end
            end
        else
            println("base case solve type not recognized :)")
        end


        # set all ctg scores to 0:
        stt.zctg[tii] .= 0.0

        # zero out the gradients, which will be collected and applied all at once!
        flw.dz_dpinj_all[tii] .= 0.0

        # define the ctg 
        cs = dt*prm.vio.s_flow*qG.scale_c_sflow_testing

        # do we want to score all ctgs? for testing/post processing
        if qG.score_all_ctgs == true
            ###########################################################
            # => up above: @info "Warning -- scoring all contingencies! No gradients."
            ###########################################################
            for ctg_ii in 1:sys.nctg
                # see the "else" case for comments and details
                flw.theta_k[tii] .= wmi_update(ctb[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])
                # => slow: flw.pflow_k .= ntk.Yfr*flw.theta_k .+ flw.bt
                quasiGrad.mul!(flw.pflow_k[tii], ntk.Yfr, flw.theta_k[tii])
                flw.pflow_k[tii] .+= flw.bt[tii]
                flw.sfr[tii]     .= sqrt.(flw.qfr2[tii] .+ flw.pflow_k[tii].^2)
                flw.sto[tii]     .= sqrt.(flw.qto2[tii] .+ flw.pflow_k[tii].^2)
                flw.sfr_vio[tii] .= flw.sfr[tii] .- ntk.s_max
                flw.sto_vio[tii] .= flw.sto[tii] .- ntk.s_max
                flw.sfr_vio[tii][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                flw.sto_vio[tii][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                    # => flw[:smax_vio] .= max.(flw.sfr_vio, flw.sto_vio, 0.0)
                    # => if helpful: zctg_s = cs*flw[:smax_vio]
                    # => if helpful: stt.zctg[tii][ctg_ii] = -sum(zctg_s, init=0.0)
                stt.zctg[tii][ctg_ii] = -cs*sum(max.(flw.sfr_vio[tii], flw.sto_vio[tii], 0.0))
            end

            # score
                # => reduce scr[:zctg_min] += minimum(stt.zctg[tii])
                # => reduce scr[:zctg_avg] += sum(stt.zctg[tii])/sys.nctg
            @reduce(zctg_min += minimum(stt.zctg[tii]))
            @reduce(zctg_avg += sum(stt.zctg[tii])/sys.nctg)
        else
            # loop over contingency subset
            for ctg_ii in wct[tii][1:num_ctg] # first is worst!! sys.nctg
                # Here, we must solve theta_k = Ybr_k\c -- assume qG.ctg_solver == "wmi"
                #
                # now, we need to solve the following:
                # (Yb + v*b*v')x = c
                #
                # we already know x0 = Yb\c, so let's use it!
                #
                # wmi :)
                # explicit version => theta_k = ctb[tii] - Vector(ntk.u_k[ctg_ii]*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii],c)))
                flw.theta_k[tii] .= wmi_update(ctb[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])
                # compute flows
                #
                # NOTE: ctg[:pflow_k][tii][ctg_ii] contains the flow on the outaged line --
                #       -- this will be dealt with when computing the flows and gradients
                # => slow: flw.pflow_k .= ntk.Yfr*flw.theta_k .+ flw.bt
                quasiGrad.mul!(flw.pflow_k[tii], ntk.Yfr, flw.theta_k[tii])
                flw.pflow_k[tii] .+= flw.bt[tii]
                flw.sfr[tii]     .= sqrt.(flw.qfr2[tii] .+ flw.pflow_k[tii].^2)
                flw.sto[tii]     .= sqrt.(flw.qto2[tii] .+ flw.pflow_k[tii].^2)
                flw.sfr_vio[tii] .= flw.sfr[tii] .- ntk.s_max
                flw.sto_vio[tii] .= flw.sto[tii] .- ntk.s_max

                # make sure there are no penalties on lines that are out-aged!
                flw.sfr_vio[tii][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                flw.sto_vio[tii][ntk.ctg_out_ind[ctg_ii]] .= 0.0

                # compute the penalties: "stt[:zctg_s][tii][ctg_ii]" -- if want to keep
                    # => if helpful: smax_vio = max.(sfr_vio, sto_vio, 0.0)
                    # => if helpful: zctg_s = cs*flw[:smax_vio]
                    # => if helpful: stt.zctg[tii][ctg_ii] = -sum(zctg_s, init=0.0)

                # each contingency, at each time, gets a score:
                stt.zctg[tii][ctg_ii] = -cs*sum(max.(flw.sfr_vio[tii], flw.sto_vio[tii], 0.0), init=0.0)

                # great -- now, do we take the gradient?
                if qG.eval_grad
                    # only take the gradient if the ctg violation is sufficiently large!!!
                    if stt.zctg[tii][ctg_ii] < qG.ctg_grad_cutoff
                        # game on :)
                        #
                        # in this code, we assume we take the gradient of all scored
                        # contingencies -- this can be updated!

                        # What are the gradients? build indicators with some tolerance
                        # slower => get_largest_ctg_indices(bit, flw, qG, :sfr_vio, :sto_vio)
                        bit.sfr_vio[tii] .= (flw.sfr_vio[tii] .> qG.grad_ctg_tol) .&& (flw.sfr_vio[tii] .> flw.sto_vio[tii])
                        bit.sto_vio[tii] .= (flw.sto_vio[tii] .> qG.grad_ctg_tol) .&& (flw.sto_vio[tii] .> flw.sfr_vio[tii])

                        # build the grads
                        flw.dsmax_dqfr_flow[tii]                     .= 0.0
                        flw.dsmax_dqto_flow[tii]                     .= 0.0
                        flw.dsmax_dp_flow[tii]                       .= 0.0
                        flw.dsmax_dp_flow[tii][bit.sfr_vio[tii]]   .= flw.pflow_k[tii][bit.sfr_vio[tii]]./flw.sfr[tii][bit.sfr_vio[tii]]
                        flw.dsmax_dp_flow[tii][bit.sto_vio[tii]]   .= flw.pflow_k[tii][bit.sto_vio[tii]]./flw.sto[tii][bit.sto_vio[tii]]
                        flw.dsmax_dqfr_flow[tii][bit.sfr_vio[tii]] .= flw.ac_qfr[tii][bit.sfr_vio[tii]]./flw.sfr[tii][bit.sfr_vio[tii]]
                        flw.dsmax_dqto_flow[tii][bit.sto_vio[tii]] .= flw.ac_qto[tii][bit.sto_vio[tii]]./flw.sto[tii][bit.sto_vio[tii]]

                        # "was" this the worst ctg of the lot? (most negative!)
                        if ctg_ii == wct[tii][1]
                            gc = copy(gc_avg) + copy(gc_min)
                        else
                            gc = copy(gc_avg)
                        end

                        # first, deal with the reactive power flows -- these are functions
                        # of line variables (v, theta, phi, tau, u_on)
                        #
                        # acline
                        if 1 in bit.sfr_vio[tii][1:sys.nl]
                            # deal with the fr line
                            aclfr_alpha = gc*(flw.dsmax_dqfr_flow[tii][1:sys.nl][bit.sfr_vio[tii][1:sys.nl]])
                            zctgs_grad_qfr_acline!(aclfr_alpha, bit, grd, idx, mgd, prm, qG, sys, tii)
                        end

                        if 1 in bit.sto_vio[tii][1:sys.nl]
                            # deal with the to line
                            aclto_alpha = gc*(flw.dsmax_dqto_flow[tii][1:sys.nl][bit.sto_vio[tii][1:sys.nl]])
                            zctgs_grad_qto_acline!(aclto_alpha, bit, grd, idx, mgd, prm, qG, sys, tii)
                        end

                        # slower:
                            # => aclfr_inds  = findall(!iszero,bit.sfr_vio[1:sys.nl])
                            # => aclto_inds  = findall(!iszero,bit.sto_vio[1:sys.nl])
                            # => aclfr_alpha = gc*(flw.dsmax_dqfr_flow[1:sys.nl][aclfr_inds])
                            # => aclto_alpha = gc*(flw.dsmax_dqto_flow[1:sys.nl][aclto_inds])
                            # => zctgs_grad_q_acline!(tii, idx, grd, mgd, aclfr_inds, aclto_inds, aclfr_alpha, aclto_alpha)

                        # xfm
                        if 1 in bit.sfr_vio[tii][(sys.nl+1):sys.nac]
                            xfr_alpha = gc*(flw.dsmax_dqfr_flow[tii][(sys.nl+1):sys.nac][bit.sfr_vio[tii][(sys.nl+1):sys.nac]])
                            zctgs_grad_qfr_xfm!(bit, grd, idx, mgd, prm, qG, sys, tii, xfr_alpha)
                        end
                        if 1 in bit.sto_vio[tii][(sys.nl+1):sys.nac]
                            xto_alpha = gc*(flw.dsmax_dqto_flow[tii][(sys.nl+1):sys.nac][bit.sto_vio[tii][(sys.nl+1):sys.nac]])
                            zctgs_grad_qto_xfm!(bit, grd, idx, mgd, prm, qG, sys, tii, xto_alpha)
                        end

                        # slower:
                            # => xfr_inds  = findall(!iszero,bit.sfr_vio[(sys.nl+1):sys.nac])
                            # => xto_inds  = findall(!iszero,bit.sto_vio[(sys.nl+1):sys.nac])
                            # => xfr_alpha = gc*(flw.dsmax_dqfr_flow[(sys.nl+1):sys.nac][xfr_inds])
                            # => xto_alpha = gc*(flw.dsmax_dqto_flow[(sys.nl+1):sys.nac][xto_inds])
                            # => zctgs_grad_q_xfm!(tii, idx, grd, mgd, xfr_inds, xto_inds, xfr_alpha, xto_alpha)

                        # now, the fun one: active power injection + xfm phase shift!!
                        # **** => alpha_p_flow_phi = gc*flw.dsmax_dp_flow
                        # **** => rhs = ntk.YfrT*alpha_p_flow_phi
                        #
                        # 
                        # alpha_p_flow_phi is the derivative of znms with repsect to the
                        # active power flow vector in a given contingency at a given time

                        # get the derivative of znms wrt active power injection
                        # NOTE: ntk.Yfr = Ybs*Er, so ntk.Yfr^T = Er^T*Ybs
                        #   -> techincally, we need Yfr_k, where the admittance
                        #      at the outaged line has been drive to 0, but we
                        #      can safely use Yfr, since alpha_p_flow_phi["k"] = 0
                        #      (this was enforced ~50 or so lines above)
                        # NOTE #2 -- this does NOT include the reference bus!
                        #            we skip this gradient :)
                        # => flw.rhs .= ntk.YfrT*(gc.*flw.dsmax_dp_flow)
                        flw.dsmax_dp_flow[tii] .= gc.*flw.dsmax_dp_flow[tii]
                        quasiGrad.mul!(flw.rhs[tii], ntk.YfrT, flw.dsmax_dp_flow[tii]);
                        # time to solve for dz_dpinj -- two options here:
                        #   1. solve with ntk.Ybr_k, but we didn't actually build this,
                        #      and we didn't build its preconditioner either..
                        #   2. solve with ntk.Ybr, and then use a rank 1 update! Let's do
                        #      this instead :) we'll do this in-loop for each ctg at each time.
                        # => flw.dz_dpinj .= lowrank_update_single_ctg_gradient(ctd, ctg_ii, ntk, qG, flw.rhs, sys)
                        lowrank_update_single_ctg_gradient!(ctd, ctg_ii, flw, ntk, qG, sys, tii)
                        
                        # now, we have the gradient of znms wrt all nodal injections/xfm phase shifts!!!
                        # except the slack bus... time to apply these gradients into 
                        # the master grad at all buses except the slack bus.
                        #
                        # update the injection gradient to account for slack!
                        #   alternative direct solution: 
                        #       -> ctg[:dz_dpinj][tii][ctg_ii] = (quasiGrad.I-ones(sys.nb-1)*ones(sys.nb-1)'/(sys.nb))*(ntk.Ybr_k[ctg_ii]\(ntk.YfrT*alpha_p_flow))
                        flw.dz_dpinj_all[tii] .+= flw.dz_dpinj[tii] .- sum(flw.dz_dpinj[tii])/Float64(sys.nb)

                        # legacy option: apply device gradients -- super slow!!
                            # => zctgs_grad_pinj!(dz_dpinj, grd, idx, mgd, ntk, prm, sys, tii)
                    end
                end
            end

            # now, actually apply the active power gradients! In the reactive power case, we just apply as we go
            if qG.eval_grad
                zctgs_grad_pinj!(flw.dz_dpinj_all[tii], grd, idx, mgd, ntk, prm, sys, tii)
            end
            # across each contingency, we get the average, and we get the min
                # => scr[:zctg_min] += minimum(stt.zctg[tii])
                # => scr[:zctg_avg] += sum(stt.zctg[tii])/sys.nctg
            @reduce(zctg_min += minimum(stt.zctg[tii]))
            @reduce(zctg_avg += sum(stt.zctg[tii])/sys.nctg)

            # now that we have scored all contingencies at this given time,
            # rank them from most negative to least (worst is first)
            wct[tii][1:num_ctg] .= sortperm(stt.zctg[tii][wct[tii][1:num_ctg]])

            # however, only keep half!
            wct[tii][1:num_ctg] .= union(wct[tii][1:num_wrst],quasiGrad.shuffle(setdiff(1:sys.nctg, wct[tii][1:num_wrst]))[1:num_rnd])
        end
    end

    # assign at the very end
    scr[:zctg_min] = zctg_min
    scr[:zctg_avg] = zctg_avg
end

function lowrank_update_single_ctg_gradient!(ctd::Vector{Vector{Float64}}, ctg_ii::Int64, flw::quasiGrad.Flow, ntk::quasiGrad.Ntk, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8)
    # step 1: solve the contingency on the base-case
    # step 2: low rank update the solution
    # 
    # step 1:
    # solve with the previous base-case solution: ctg[:ctd][tii][ctg_ii::Int64ctg_ii]
    # note: this keeps getting overwritten at each time!
    #
    # i.e., the previous time solutions implicitly hot-starts the solution
    # solve the base case with pcg
    if qG.base_solver == "lu"
        ctd[tii] .= ntk.Ybr\flw.rhs[tii]

        # error with this type !!!
    # elseif qG.base_solver == "cholesky"
    #    ctd[tii] = ntk.Ybr_Ch\rhs

    elseif qG.base_solver == "pcg"
        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- just use LU
            ctd[tii] .= ntk.Ybr\flw.rhs[tii]
            
        else
            # solve with a hot start!
            _, ch = quasiGrad.cg!(ctd[tii], ntk.Ybr, flw.rhs[tii], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
        
            # test the krylov solution
            if ~(ch.isconverged)
                # LU backup
                println("Krylov failed -- using LU backup (ctg gradient)")
                ctd[tii] .= ntk.Ybr\flw.rhs[tii]
            end
        end
    end

    # step 2:
    # now, apply a low-rank update!
    # explicit version => dz_dpinj = ctd[tii] - Vector(ntk.u_k[ctg_ii]*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], rhs)))
    flw.dz_dpinj[tii] .= wmi_update(ctd[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.rhs[tii])
end

function zctgs_grad_qfr_acline!(aclfr_alpha::Vector{Float64}, bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Idx, mgd::quasiGrad.Mgd, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8)
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
    vmfrqfr = aclfr_alpha.*grd.acline_qfr.vmfr[tii][bit.sfr_vio[tii][1:sys.nl]]
    vmtoqfr = aclfr_alpha.*grd.acline_qfr.vmto[tii][bit.sfr_vio[tii][1:sys.nl]]
    vafrqfr = aclfr_alpha.*grd.acline_qfr.vafr[tii][bit.sfr_vio[tii][1:sys.nl]]
    vatoqfr = aclfr_alpha.*grd.acline_qfr.vato[tii][bit.sfr_vio[tii][1:sys.nl]]

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(prm.acline.line_inds[bit.sfr_vio[tii][1:sys.nl]])
        # update the master grad -- qfr
        mgd.vm[tii][idx.acline_fr_bus[ln]] += vmfrqfr[ii]
        mgd.vm[tii][idx.acline_to_bus[ln]] += vmtoqfr[ii]
        mgd.va[tii][idx.acline_fr_bus[ln]] += vafrqfr[ii]
        mgd.va[tii][idx.acline_to_bus[ln]] += vatoqfr[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        uonqfr  = aclfr_alpha.*grd.acline_qfr.uon[tii][bit.sfr_vio[tii][1:sys.nl]]
        for (ii,ln) in enumerate(prm.acline.line_inds[bit.sfr_vio[tii][1:sys.nl]])
            mgd.u_on_acline[tii][ln]           += uonqfr[ii]
        end
    end
end

function zctgs_grad_qto_acline!(aclto_alpha::Vector{Float64}, bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Idx, mgd::quasiGrad.Mgd, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8)
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
    vmfrqto = aclto_alpha.*grd.acline_qto.vmfr[tii][bit.sto_vio[tii][1:sys.nl]]
    vmtoqto = aclto_alpha.*grd.acline_qto.vmto[tii][bit.sto_vio[tii][1:sys.nl]]
    vafrqto = aclto_alpha.*grd.acline_qto.vafr[tii][bit.sto_vio[tii][1:sys.nl]]
    vatoqto = aclto_alpha.*grd.acline_qto.vato[tii][bit.sto_vio[tii][1:sys.nl]]

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(prm.acline.line_inds[bit.sto_vio[tii][1:sys.nl]])
        # update the master grad -- qto
        mgd.vm[tii][idx.acline_fr_bus[ln]] += vmfrqto[ii]
        mgd.vm[tii][idx.acline_to_bus[ln]] += vmtoqto[ii]
        mgd.va[tii][idx.acline_fr_bus[ln]] += vafrqto[ii]
        mgd.va[tii][idx.acline_to_bus[ln]] += vatoqto[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        uonqto  = aclto_alpha.*grd.acline_qto.uon[tii][bit.sto_vio[tii][1:sys.nl]]
        for (ii,ln) in enumerate(prm.acline.line_inds[bit.sto_vio[tii][1:sys.nl]])
            mgd.u_on_acline[tii][ln]           += uonqto[ii]
        end
    end
end

function zctgs_grad_qfr_xfm!(bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Idx, mgd::quasiGrad.Mgd, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8, xfr_alpha::Vector{Float64})
    # so, this function takes and applies the gradient of
    # zctgs (at transformers) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # xfr_inds = xfms which are violated on the from side!
    # xto_inds = xfms which are violated on the to side!
    # xfr_alpha = associated partial
    # xto_alpha = associated partial
    #   example: xfms 1 (max overload on to), 2 and 3 (max overload on frm)
    #   xfr_inds = [3]
    #   xto_inds = [1]

    vmfrqfr = xfr_alpha.*grd.xfm_qfr.vmfr[tii][bit.sfr_vio[tii][(sys.nl+1):sys.nac]]
    vmtoqfr = xfr_alpha.*grd.xfm_qfr.vmto[tii][bit.sfr_vio[tii][(sys.nl+1):sys.nac]]
    vafrqfr = xfr_alpha.*grd.xfm_qfr.vafr[tii][bit.sfr_vio[tii][(sys.nl+1):sys.nac]]
    vatoqfr = xfr_alpha.*grd.xfm_qfr.vato[tii][bit.sfr_vio[tii][(sys.nl+1):sys.nac]]
    tauqfr  = xfr_alpha.*grd.xfm_qfr.tau[tii][bit.sfr_vio[tii][(sys.nl+1):sys.nac]]
    phiqfr  = xfr_alpha.*grd.xfm_qfr.phi[tii][bit.sfr_vio[tii][(sys.nl+1):sys.nac]]

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit.sfr_vio[tii][(sys.nl+1):sys.nac]])
        # update the master grad -- qfr
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += vmfrqfr[ii]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += vmtoqfr[ii]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += vafrqfr[ii]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += vatoqfr[ii]
        mgd.tau[tii][xfm]                += tauqfr[ii]
        mgd.phi[tii][xfm]                += phiqfr[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        uonqfr  = xfr_alpha.*grd.xfm_qfr.uon[tii][bit.sfr_vio[tii][(sys.nl+1):sys.nac]]
        for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit.sfr_vio[tii][(sys.nl+1):sys.nac]])
            mgd.u_on_xfm[tii][xfm] += uonqfr[ii]
        end
    end
end

function zctgs_grad_qto_xfm!(bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Idx, mgd::quasiGrad.Mgd, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8, xto_alpha::Vector{Float64})
    # so, this function takes and applies the gradient of
    # zctgs (at transformers) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # xfr_inds = xfms which are violated on the from side!
    # xto_inds = xfms which are violated on the to side!
    # xfr_alpha = associated partial
    # xto_alpha = associated partial
    #   example: xfms 1 (max overload on to), 2 and 3 (max overload on frm)
    #   xfr_inds = [3]
    #   xto_inds = [1]
    vmfrqto = xto_alpha.*grd.xfm_qto.vmfr[tii][bit.sto_vio[tii][(sys.nl+1):sys.nac]]
    vmtoqto = xto_alpha.*grd.xfm_qto.vmto[tii][bit.sto_vio[tii][(sys.nl+1):sys.nac]]
    vafrqto = xto_alpha.*grd.xfm_qto.vafr[tii][bit.sto_vio[tii][(sys.nl+1):sys.nac]]
    vatoqto = xto_alpha.*grd.xfm_qto.vato[tii][bit.sto_vio[tii][(sys.nl+1):sys.nac]]
    tauqto  = xto_alpha.*grd.xfm_qto.tau[tii][bit.sto_vio[tii][(sys.nl+1):sys.nac]]
    phiqto  = xto_alpha.*grd.xfm_qto.phi[tii][bit.sto_vio[tii][(sys.nl+1):sys.nac]]

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit.sto_vio[tii][(sys.nl+1):sys.nac]])
        # update the master grad -- qto
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += vmfrqto[ii]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += vmtoqto[ii]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += vafrqto[ii]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += vatoqto[ii]
        mgd.tau[tii][xfm]                += tauqto[ii]
        mgd.phi[tii][xfm]                += phiqto[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        uonqto  = xto_alpha.*grd.xfm_qto.uon[tii][bit.sto_vio[tii][(sys.nl+1):sys.nac]]
        for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit.sto_vio[tii][(sys.nl+1):sys.nac]])
            mgd.u_on_xfm[tii][xfm]           += uonqto[ii]
        end
    end
end

function zctgs_grad_pinj!(alpha::Vector{Float64}, grd::quasiGrad.Grad, idx::quasiGrad.Idx, mgd::quasiGrad.Mgd, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, sys::quasiGrad.System, tii::Int8)
    # note: the influcence of alpha/slack power is completely neglected when
    # applying gradients (of course, we use if to computes flows, etc.). If we
    # did consider it, then every device would show up at every single bus! yuck!!!
    #
    # note the following convention used for power injections:
    # pinj = p_pr - p_cs - p_sh - p_fr_dc - p_to_dc - alpha*slack
    #
    # note: ac flows are, obviously, NOT consiered here!!!
    #
    # so, +alpha[bus-1] is applied at the producers, and 
    # then -alpha[bus-1] is applied everywhere else!
    #
    # here, we loop over all non-reference buses, but we
    # call "alpha[bus-1]", since alpha has length nb-1
    # but corresponds to buses 2:nb
    #
    # note: pinj actually also includes the xfm phase shift:
    #   pinj = (p_pr - p_cs - p_sh - p_fr_dc - p_to_dc - alpha*slack) + Er^T*phi*b
    for bus in 2:sys.nb
        # consumer injections
        for dev in idx.cs[bus]
            quasiGrad.dp_alpha!(grd, dev, tii, -alpha[bus-1])
        end

        # producer injections
        for dev in idx.pr[bus]
            quasiGrad.dp_alpha!(grd, dev, tii, alpha[bus-1])
        end

        # shunt injections
        mgd.vm[tii][bus] += sum(-alpha[bus-1]*grd.sh_p.vm[tii][sh] for sh in idx.sh[bus]; init=0.0)

        # shunt injections -- shunt steps # grd[:pb_slack][bus][:sh_p][tii][idx.sh[bus]].*
        for sh in idx.sh[bus]
            mgd.u_step_shunt[tii][sh] += -alpha[bus-1]*
            grd.sh_p.g_tv_shunt[tii][sh]*
            prm.shunt.gs[sh] # => grd[:g_tv_shunt][:u_step_shunt][idx.sh[bus]]
        end
        
        # skip dc lines if there are none
        if !isempty(idx.bus_is_dc_frs[bus])
            # dc injections -- "pfr" contributions # grd[:pb_slack][bus][:dc_pfr][tii][idx.bus_is_dc_frs[bus]]
            mgd.dc_pfr[tii][idx.bus_is_dc_frs[bus]] .+= -alpha[bus-1]
        end

        if !isempty(idx.bus_is_dc_tos[bus])
            # dc injections -- "pto" contributions
            #
            # note: "dc_pto" does not exist, as a "mgd" variable, so
            # we just leverage that dc_pto = -dc_pfr   ->  d(dc_pto)_d(dc_pfr) = -1 grd[:pb_slack][bus][:dc_pto][tii][idx.bus_is_dc_tos[bus]]
            mgd.dc_pfr[tii][idx.bus_is_dc_tos[bus]] .+= - -alpha[bus-1]
        end
        
        # phase shift derivatives -- apply gradients
        #   => pinj = (p_pr - p_cs - p_sh - p_fr_dc - p_to_dc - alpha*slack) + Er^T*phi*b
        if ~isempty(ntk.xfm_at_bus[bus]) # make sure there is an xfm here!
            mgd.phi[tii][ntk.xfm_at_bus[bus]] .+= alpha[bus-1].*ntk.xfm_phi_scalars[bus]
        end
    end
end

# optimally compute the wmi update
function wmi_update(y0::Vector{Float64}, u::Vector{Float64}, g::Float64, x::Vector{Float64})
    # this special function to speedily compute y = y0 - u*g*(u'*x), where u can be sparse-ish,
    # AND, importantly, we don't want to convert from sparse to dense via "Vector"
    #
    # ~~ sadly, this is slow ~~~ depricated for now..
    #
    #y = copy(y0)
    #s = 0.0
    # loop once for the dot
    #for nzu_idx in quasiGrad.rowvals(u)
    #    s += u[nzu_idx] * x[nzu_idx]
    #end

    # loop again for subtraction
    #gs = g*s
    #for nzu_idx in quasiGrad.rowvals(u)
    #    y[nzu_idx] = y0[nzu_idx] - gs*u[nzu_idx]
    #end

    # output
    return y0 .- u.*(g*quasiGrad.dot(u, x))
end

function get_largest_ctg_indices(bit::quasiGrad.Bit, flw::quasiGrad.Flow, qG::quasiGrad.QG, s1::Symbol, s2::Symbol)
    for ii in 1:length(flw[s1])
        if (flw[s1][ii] >= flw[s2][ii]) && (flw[s1][ii] > qG.grad_ctg_tol)
            bit[s1][ii] = 1
            bit[s2][ii] = 0
        elseif flw[s2][ii] > qG.grad_ctg_tol # no need to check v2[ii] > v1[ii]
            bit[s2][ii] = 1
            bit[s1][ii] = 0
        else
            bit[s1][ii] = 0  
            bit[s2][ii] = 0
        end
    end
end