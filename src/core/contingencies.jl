function solve_ctgs!(
    cgd::quasiGrad.Cgd,
    ctb::Vector{Vector{Float64}},        
    ctd::Vector{Vector{Float64}},   
    flw::Dict{Symbol, Vector{Float64}},
    grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, 
    idx::quasiGrad.Idx, 
    mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    ntk::quasiGrad.Ntk, 
    prm::quasiGrad.Param, 
    qG::quasiGrad.QG,
    scr::Dict{Symbol, Float64},
    stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    sys::quasiGrad.System,                                          
    wct::Vector{Vector{Int64}})
    # this script solves AND scores 
    #
    # loop over each time period and compute the power injections
    #
    # This step is contingency invariant -- i.e., each ctg will use this information

    # ===========================
    # ctb = base theta solutions, across time (then rank-1 corrected)
    # ctd = contingency gradient solutions (across gradients) solved
    #       on the base case (then rank-1 corrected)
    # ===========================
    
    # reset 
    scr[:zctg_min] = 0.0
    scr[:zctg_avg] = 0.0

    # how many ctgs 
    num_wrst = Int64(round(qG.frac_ctg_keep*sys.nctg/2))
    num_rnd  = Int64(round(qG.frac_ctg_keep*sys.nctg/2))
    num_ctg  = num_wrst + num_rnd

    if qG.score_all_ctgs == true
        ###########################################################
        @info "Warning -- scoring all contingencies! No gradients."
        ###########################################################
    end

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
            sum(stt[:dev_p][tii][pr] for pr in idx.pr_devs) -
            sum(stt[:dev_p][tii][cs] for cs in idx.cs_devs) - 
            sum(stt[:sh_p][tii])

        # loop over each bus
        for bus in 1:sys.nb
            # active power balance
            flw[:p_inj][bus] = 
                sum(stt[:dev_p][tii][pr] for pr in idx.pr[bus]; init=0.0) - 
                sum(stt[:dev_p][tii][cs] for cs in idx.cs[bus]; init=0.0) - 
                sum(stt[:sh_p][tii][sh] for sh in idx.sh[bus]; init=0.0) - 
                sum(stt[:dc_pfr][tii][dc_fr] for dc_fr in idx.bus_is_dc_frs[bus]; init=0.0) - 
                sum(stt[:dc_pto][tii][dc_to] for dc_to in idx.bus_is_dc_tos[bus]; init=0.0) - 
                ntk.alpha*p_slack
        end

        # also, we need to update the flows on all lines! and the phase shift
        flw[:ac_qfr][idx.ac_line_flows] .= stt[:acline_qfr][tii]
        flw[:ac_qfr][idx.ac_xfm_flows]  .= stt[:xfm_qfr][tii]
        flw[:ac_qto][idx.ac_line_flows] .= stt[:acline_qto][tii]
        flw[:ac_qto][idx.ac_xfm_flows]  .= stt[:xfm_qto][tii]
        flw[:ac_phi][idx.ac_phi]        .= stt[:phi][tii]

        # compute square flows
        flw[:qfr2] .= flw[:ac_qfr].^2
        flw[:qto2] .= flw[:ac_qto].^2

        # solve for the flows across each ctg
        #   p  =  @view flw[:p_inj][2:end]
        flw[:bt] .= .-flw[:ac_phi].*ntk.b
        # now, we have flw[:p_inj] = Yb*theta + E'*bt
        #   c = p - ntk.Er'*bt
        #
        # simplified:
        flw[:c] .= (@view flw[:p_inj][2:end]) .- ntk.Er'*flw[:bt]

        # solve the base case with pcg
        if qG.base_solver == "lu"
            ctb[t_ind]  .= ntk.Ybr\flw[:c]

        # error with this type !!!
        # elseif qG.base_solver == "cholesky"
        #    ctb[t_ind]  = ntk.Ybr_Ch\c
        
        elseif qG.base_solver == "pcg"
            if sys.nb <= qG.min_buses_for_krylov
                # too few buses -- just use LU
                ctb[t_ind] .= ntk.Ybr\flw[:c]
            else
                # solve with a hot start!
                #
                # note: ctg[:ctb][tii][end] is modified in place,
                # and it represents the base case solution
                _, ch = quasiGrad.cg!(ctb[t_ind], ntk.Ybr, flw[:c], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
                
                # test the krylov solution
                if ~(ch.isconverged)
                    @info "Krylov failed -- using LU backup (ctg flows)!"
                    ctb[t_ind] = ntk.Ybr\flw[:c]
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
            # => up above: @info "Warning -- scoring all contingencies! No gradients."
            ###########################################################
            for ctg_ii in 1:sys.nctg
                # see the "else" case for comments and details
                theta_k = special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
                pflow_k = ntk.Yfr*theta_k  + flw[:bt]
                sfr     = sqrt.(flw[:qfr2] + pflow_k.^2)
                sto     = sqrt.(flw[:qto2] + pflow_k.^2)
                sfr_vio = sfr - ntk.s_max
                sto_vio = sto - ntk.s_max
                sfr_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
                sto_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
                smax_vio = max.(sfr_vio, sto_vio, 0.0)
                zctg_s = dt*prm.vio.s_flow*smax_vio * qG.scale_c_sflow_testing
                stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)
            end

            # score
            scr[:zctg_min] += minimum(stt[:zctg][tii])
            scr[:zctg_avg] += sum(stt[:zctg][tii])/sys.nctg
        else
            # loop over contingency subset
            for ctg_ii in wct[t_ind][1:num_ctg] # first is worst!! sys.nctg
                # Here, we must solve theta_k = Ybr_k\c -- assume qG.ctg_solver == "wmi"
                #
                # now, we need to solve the following:
                # (Yb + v*b*v')x = c
                #
                # we already know x0 = Yb\c, so let's use it!
                #
                # wmi :)
                # explicit version => theta_k = ctb[t_ind] - Vector(ntk.u_k[ctg_ii]*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii],c)))
                theta_k = special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
                # compute flows
                #
                # NOTE: ctg[:pflow_k][tii][ctg_ii] contains the flow on the outaged line --
                #       -- this will be dealt with when computing the flows and gradients
                pflow_k = ntk.Yfr*theta_k  + flw[:bt]
                sfr     = sqrt.(flw[:qfr2] + pflow_k.^2)
                sto     = sqrt.(flw[:qto2] + pflow_k.^2)
                sfr_vio = sfr - ntk.s_max
                sto_vio = sto - ntk.s_max

                # make sure there are no penalties on lines that are out-aged!
                sfr_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
                sto_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
                smax_vio = max.(sfr_vio, sto_vio, 0.0)

                # compute the penalties: "stt[:zctg_s][tii][ctg_ii]" -- if want to keep
                zctg_s = dt*prm.vio.s_flow*smax_vio * qG.scale_c_sflow_testing

                # each contingency, at each time, gets a score:
                stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)

                # great -- now, do we take the gradient? look at the previous adam step
                if qG.eval_grad
                    # game on :)
                    #
                    # in this code, we assume we take the gradient of all scored
                    # contingencies -- this can be updated!

                    # What are the gradients? build indicators with some tolerance
                    gamma_fr   = (sfr_vio .> qG.grad_ctg_tol) .&& (sfr_vio .> sto_vio)
                    gamma_to   = (sto_vio .> qG.grad_ctg_tol) .&& (sto_vio .> sfr_vio)

                    # build the grads
                    flw[:dsmax_dqfr_flow]          .= 0.0
                    flw[:dsmax_dqto_flow]          .= 0.0
                    flw[:dsmax_dp_flow]            .= 0.0
                    flw[:dsmax_dp_flow][gamma_fr]   = pflow_k[gamma_fr]./sfr[gamma_fr]
                    flw[:dsmax_dp_flow][gamma_to]   = pflow_k[gamma_to]./sto[gamma_to]
                    flw[:dsmax_dqfr_flow][gamma_fr] = flw[:ac_qfr][gamma_fr]./sfr[gamma_fr]
                    flw[:dsmax_dqto_flow][gamma_to] = flw[:ac_qto][gamma_to]./sto[gamma_to]

                    # "was" this the worst ctg of the lot? (most negative!)
                    if ctg_ii == wct[t_ind][1]
                        gc = copy(gc_avg) + copy(gc_min)
                    else
                        gc = copy(gc_avg)
                    end

                    # first, deal with the reactive power flows -- these are functions
                    # of line variables (v, theta, phi, tau, u_on)
                    #
                    # acline
                    aclfr_inds  = findall(!iszero,gamma_fr[1:sys.nl])
                    aclto_inds  = findall(!iszero,gamma_to[1:sys.nl])
                    aclfr_alpha = gc*(flw[:dsmax_dqfr_flow][1:sys.nl][aclfr_inds])
                    aclto_alpha = gc*(flw[:dsmax_dqto_flow][1:sys.nl][aclto_inds])
                    zctgs_grad_q_acline!(tii, idx, grd, mgd, aclfr_inds, aclto_inds, aclfr_alpha, aclto_alpha)
                    # xfm
                    xfr_inds  = findall(!iszero,gamma_fr[(sys.nl+1):sys.nac])
                    xto_inds  = findall(!iszero,gamma_to[(sys.nl+1):sys.nac])
                    xfr_alpha = gc*(flw[:dsmax_dqfr_flow][(sys.nl+1):sys.nac][xfr_inds])
                    xto_alpha = gc*(flw[:dsmax_dqto_flow][(sys.nl+1):sys.nac][xto_inds])
                    zctgs_grad_q_xfm!(tii, idx, grd, mgd, xfr_inds, xto_inds, xfr_alpha, xto_alpha)

                    # now, the fun one: active power injection + xfm phase shift!!
                    alpha_p_flow_phi = gc*flw[:dsmax_dp_flow]
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
                    rhs = ntk.Yfr'*alpha_p_flow_phi

                    # time to solve for dz_dpinj -- two options here:
                    #   1. solve with ntk.Ybr_k, but we didn't actually build this,
                    #      and we didn't build its preconditioner either..
                    #   2. solve with ntk.Ybr, and then use a rank 1 update! Let's do
                    #      this instead :) we'll do this in-loop for each ctg at each time.
                    dz_dpinj = lowrank_update_single_ctg_gradient(ctd, ctg_ii, ntk, qG, rhs, sys)
                    
                    # now, we have the gradient of znms wrt all nodal injections/xfm phase shifts!!!
                    # except the slack bus... time to apply these gradients into 
                    # the master grad at all buses except the slack bus.
                    #
                    # update the injection gradient to account for slack!
                    #   alternative direct solution: 
                    #       -> ctg[:dz_dpinj][tii][ctg_ii] = (quasiGrad.I-ones(sys.nb-1)*ones(sys.nb-1)'/(sys.nb))*(ntk.Ybr_k[ctg_ii]\(ntk.Yfr'*alpha_p_flow))
                    flw[:dz_dpinj_all] += dz_dpinj .- sum(dz_dpinj)/Float64(sys.nb)

                    # legacy option: apply device gradients -- super slow!!
                        # => zctgs_grad_pinj!(dz_dpinj, grd, idx, mgd, ntk, prm, sys, tii)
                end
            end

            # now, actually apply the gradients!
            if qG.eval_grad
                zctgs_grad_pinj!(flw[:dz_dpinj_all], grd, idx, mgd, ntk, prm, sys, tii)
            end
            # across each contingency, we get the average, and we get the min
            scr[:zctg_min] += minimum(stt[:zctg][tii])
            scr[:zctg_avg] += sum(stt[:zctg][tii])/sys.nctg

            # now that we have scored all contingencies at this given time,
            # rank them from most negative to least (worst is first)
            wct[t_ind][1:num_ctg] = sortperm(stt[:zctg][tii][wct[t_ind][1:num_ctg]])

            # however, only keep half!
            wct[t_ind][1:num_ctg] = union(wct[t_ind][1:num_wrst],quasiGrad.shuffle(setdiff(1:sys.nctg, wct[t_ind][1:num_wrst]))[1:num_rnd])
        end
    end
end


function lowrank_update_single_ctg_gradient(ctd::Vector{Vector{Float64}}, ctg_ii::Int64, ntk::quasiGrad.Ntk, qG::quasiGrad.QG, rhs::Vector{Float64}, sys::quasiGrad.System)
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
        ctd[ctg_ii] .= ntk.Ybr\rhs

        # error with this type !!!
    # elseif qG.base_solver == "cholesky"
    #    ctd[ctg_ii] = ntk.Ybr_Ch\rhs

    elseif qG.base_solver == "pcg"
        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- just use LU
            ctd[ctg_ii] .= ntk.Ybr\rhs
            
        else
            # solve with a hot start!
            _, ch = quasiGrad.cg!(ctd[ctg_ii], ntk.Ybr, rhs, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
        
            # test the krylov solution
            if ~(ch.isconverged)
                # LU backup
                @info "Krylov failed -- using LU backup (ctg gradient)"
                ctd[ctg_ii] = ntk.Ybr\rhs
            end
        end
    end

    # step 2:
    # now, apply a low-rank update!
    # explicit version => dz_dpinj = ctd[ctg_ii] - Vector(ntk.u_k[ctg_ii]*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], rhs)))
    dz_dpinj = special_wmi_update(ctd[ctg_ii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], rhs)

    # output
    return dz_dpinj
end

function zctgs_grad_q_acline!(tii::Symbol, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, aclfr_inds::Vector{Int64}, aclto_inds::Vector{Int64}, aclfr_alpha::Vector{Float64}, aclto_alpha::Vector{Float64})
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
    vmfrqfr = aclfr_alpha.*grd[:acline_qfr][:vmfr][tii][aclfr_inds]
    vmtoqfr = aclfr_alpha.*grd[:acline_qfr][:vmto][tii][aclfr_inds]
    vafrqfr = aclfr_alpha.*grd[:acline_qfr][:vafr][tii][aclfr_inds]
    vatoqfr = aclfr_alpha.*grd[:acline_qfr][:vato][tii][aclfr_inds]
    uonqfr  = aclfr_alpha.*grd[:acline_qfr][:uon][tii][aclfr_inds]

    # final qfr gradients
    vmfrqto = aclto_alpha.*grd[:acline_qto][:vmfr][tii][aclto_inds]
    vmtoqto = aclto_alpha.*grd[:acline_qto][:vmto][tii][aclto_inds]
    vafrqto = aclto_alpha.*grd[:acline_qto][:vafr][tii][aclto_inds]
    vatoqto = aclto_alpha.*grd[:acline_qto][:vato][tii][aclto_inds]
    uonqto  = aclto_alpha.*grd[:acline_qto][:uon][tii][aclto_inds]

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(aclfr_inds)
        # update the master grad -- qfr
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqfr[ii]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqfr[ii]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqfr[ii]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqfr[ii]
        mgd[:u_on_acline][tii][ln]           += uonqfr[ii]
    end

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(aclto_inds)
        # update the master grad -- qto
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqto[ii]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqto[ii]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqto[ii]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqto[ii]
        mgd[:u_on_acline][tii][ln]           += uonqto[ii]
    end
end

function zctgs_grad_q_xfm!(tii::Symbol, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, xfr_inds::Vector{Int64}, xto_inds::Vector{Int64}, xfr_alpha::Vector{Float64}, xto_alpha::Vector{Float64})
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

    vmfrqfr = xfr_alpha.*grd[:xfm_qfr][:vmfr][tii][xfr_inds]
    vmtoqfr = xfr_alpha.*grd[:xfm_qfr][:vmto][tii][xfr_inds]
    vafrqfr = xfr_alpha.*grd[:xfm_qfr][:vafr][tii][xfr_inds]
    vatoqfr = xfr_alpha.*grd[:xfm_qfr][:vato][tii][xfr_inds]
    tauqfr  = xfr_alpha.*grd[:xfm_qfr][:tau][tii][xfr_inds]
    phiqfr  = xfr_alpha.*grd[:xfm_qfr][:phi][tii][xfr_inds]
    uonqfr  = xfr_alpha.*grd[:xfm_qfr][:uon][tii][xfr_inds]

    # final qfr gradients
    vmfrqto = xto_alpha.*grd[:xfm_qto][:vmfr][tii][xto_inds]
    vmtoqto = xto_alpha.*grd[:xfm_qto][:vmto][tii][xto_inds]
    vafrqto = xto_alpha.*grd[:xfm_qto][:vafr][tii][xto_inds]
    vatoqto = xto_alpha.*grd[:xfm_qto][:vato][tii][xto_inds]
    tauqto  = xto_alpha.*grd[:xfm_qto][:tau][tii][xto_inds]
    phiqto  = xto_alpha.*grd[:xfm_qto][:phi][tii][xto_inds]
    uonqto  = xto_alpha.*grd[:xfm_qto][:uon][tii][xto_inds]

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(xfr_inds)
        # update the master grad -- qfr
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqfr[ii]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqfr[ii]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqfr[ii]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqfr[ii]
        mgd[:tau][tii][xfm]                += tauqfr[ii]
        mgd[:phi][tii][xfm]                += phiqfr[ii]
        mgd[:u_on_xfm][tii][xfm]           += uonqfr[ii]
    end

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(xto_inds)
        # update the master grad -- qto
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqto[ii]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqto[ii]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqto[ii]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqto[ii]
        mgd[:tau][tii][xfm]                += tauqto[ii]
        mgd[:phi][tii][xfm]                += phiqto[ii]
        mgd[:u_on_xfm][tii][xfm]           += uonqto[ii]
    end
end

function zctgs_grad_pinj!(alpha::Vector{Float64}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, sys::quasiGrad.System, tii::Symbol)
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
        mgd[:vm][tii][bus] += sum(-alpha[bus-1]*grd[:sh_p][:vm][tii][sh] for sh in idx.sh[bus]; init=0.0)

        # shunt injections -- shunt steps # grd[:pb_slack][bus][:sh_p][tii][idx.sh[bus]].*
        for sh in idx.sh[bus]
            mgd[:u_step_shunt][tii][sh] += -alpha[bus-1]*
            grd[:sh_p][:g_tv_shunt][tii][sh]*
            prm.shunt.gs[sh] # => grd[:g_tv_shunt][:u_step_shunt][idx.sh[bus]]
        end
        
        # skip dc lines if there are none
        if !isempty(idx.bus_is_dc_frs[bus])
            # dc injections -- "pfr" contributions # grd[:pb_slack][bus][:dc_pfr][tii][idx.bus_is_dc_frs[bus]]
            mgd[:dc_pfr][tii][idx.bus_is_dc_frs[bus]] .+= -alpha[bus-1]
        end

        if !isempty(idx.bus_is_dc_tos[bus])
            # dc injections -- "pto" contributions
            #
            # note: "dc_pto" does not exist, as a "mgd" variable, so
            # we just leverage that dc_pto = -dc_pfr   ->  d(dc_pto)_d(dc_pfr) = -1 grd[:pb_slack][bus][:dc_pto][tii][idx.bus_is_dc_tos[bus]]
            mgd[:dc_pfr][tii][idx.bus_is_dc_tos[bus]] .+= - -alpha[bus-1]
        end
        
        # phase shift derivatives -- apply gradients
        #   => pinj = (p_pr - p_cs - p_sh - p_fr_dc - p_to_dc - alpha*slack) + Er^T*phi*b
        if ~isempty(ntk.xfm_at_bus[bus]) # make sure there is an xfm here!
            mgd[:phi][tii][ntk.xfm_at_bus[bus]] += alpha[bus-1]*ntk.xfm_phi_scalars[bus]
        end
    end
end

# optimally compute the wmi update
function special_wmi_update(y0::Vector{Float64}, u::Vector{Float64}, g::Float64, x::Vector{Float64})
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
    return y0 - u*(g*quasiGrad.dot(u, x))
end