function solve_ctgs!(
    bit::Dict{Symbol, Dict{Symbol, BitVector}},
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
    num_wrst = Int64(ceil(qG.frac_ctg_keep*sys.nctg/2))  # in case n_ctg is odd, and we want to keep all!
    num_rnd  = Int64(floor(qG.frac_ctg_keep*sys.nctg/2)) # in case n_ctg is odd, and we want to keep all!
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
        #   c = p - ntk.ErT*bt
        #
        # simplified:
        # => flw[:c] .= (@view flw[:p_inj][2:end]) .- ntk.ErT*flw[:bt]
            # this is a little odd, but it's fine (the first use of flw[:c] is just for storage!)
        quasiGrad.mul!(flw[:c], ntk.ErT, flw[:bt])
        flw[:c] .= (@view flw[:p_inj][2:end]) .- flw[:c]

        # solve the base case with pcg
        if qG.base_solver == "lu"
            ctb[t_ind] .= ntk.Ybr\flw[:c]

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

        # define the ctg 
        cs = dt*prm.vio.s_flow*qG.scale_c_sflow_testing

        # do we want to score all ctgs? for testing/post processing
        if qG.score_all_ctgs == true
            ###########################################################
            # => up above: @info "Warning -- scoring all contingencies! No gradients."
            ###########################################################
            for ctg_ii in 1:sys.nctg
                # see the "else" case for comments and details
                flw[:theta_k] .= special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
                # => slow: flw[:pflow_k] .= ntk.Yfr*flw[:theta_k] .+ flw[:bt]
                quasiGrad.mul!(flw[:pflow_k], ntk.Yfr, flw[:theta_k])
                flw[:pflow_k] .+= flw[:bt]
                flw[:sfr]     .= sqrt.(flw[:qfr2] .+ flw[:pflow_k].^2)
                flw[:sto]     .= sqrt.(flw[:qto2] .+ flw[:pflow_k].^2)
                flw[:sfr_vio] .= flw[:sfr] .- ntk.s_max
                flw[:sto_vio] .= flw[:sto] .- ntk.s_max
                flw[:sfr_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                flw[:sto_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                    # => flw[:smax_vio] .= max.(flw[:sfr_vio], flw[:sto_vio], 0.0)
                    # => if helpful: zctg_s = cs*flw[:smax_vio]
                    # => if helpful: stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)
                stt[:zctg][tii][ctg_ii] = -cs*sum(max.(flw[:sfr_vio], flw[:sto_vio], 0.0))
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
                flw[:theta_k] .= special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
                # compute flows
                #
                # NOTE: ctg[:pflow_k][tii][ctg_ii] contains the flow on the outaged line --
                #       -- this will be dealt with when computing the flows and gradients
                # => slow: flw[:pflow_k] .= ntk.Yfr*flw[:theta_k] .+ flw[:bt]
                quasiGrad.mul!(flw[:pflow_k], ntk.Yfr, flw[:theta_k])
                flw[:pflow_k] .+= flw[:bt]
                flw[:sfr]     .= sqrt.(flw[:qfr2] .+ flw[:pflow_k].^2)
                flw[:sto]     .= sqrt.(flw[:qto2] .+ flw[:pflow_k].^2)
                flw[:sfr_vio] .= flw[:sfr] .- ntk.s_max
                flw[:sto_vio] .= flw[:sto] .- ntk.s_max

                # make sure there are no penalties on lines that are out-aged!
                flw[:sfr_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                flw[:sto_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0

                # compute the penalties: "stt[:zctg_s][tii][ctg_ii]" -- if want to keep
                    # => if helpful: smax_vio = max.(sfr_vio, sto_vio, 0.0)
                    # => if helpful: zctg_s = cs*flw[:smax_vio]
                    # => if helpful: stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)

                # each contingency, at each time, gets a score:
                stt[:zctg][tii][ctg_ii] = -cs*sum(max.(flw[:sfr_vio], flw[:sto_vio], 0.0), init=0.0)

                # great -- now, do we take the gradient?
                if qG.eval_grad
                    # only take the gradient if the ctg violation is sufficiently large!!!
                    if stt[:zctg][tii][ctg_ii] < qG.ctg_grad_cutoff
                        # game on :)
                        #
                        # in this code, we assume we take the gradient of all scored
                        # contingencies -- this can be updated!

                        # What are the gradients? build indicators with some tolerance
                        # slower => get_largest_ctg_indices(bit, flw, qG, :sfr_vio, :sto_vio)
                        bit[:sfr_vio] .= (flw[:sfr_vio] .> qG.grad_ctg_tol) .&& (flw[:sfr_vio] .> flw[:sto_vio])
                        bit[:sto_vio] .= (flw[:sto_vio] .> qG.grad_ctg_tol) .&& (flw[:sto_vio] .> flw[:sfr_vio])

                        # build the grads
                        flw[:dsmax_dqfr_flow]                .= 0.0
                        flw[:dsmax_dqto_flow]                .= 0.0
                        flw[:dsmax_dp_flow]                  .= 0.0
                        flw[:dsmax_dp_flow][bit[:sfr_vio]]   .= flw[:pflow_k][bit[:sfr_vio]]./flw[:sfr][bit[:sfr_vio]]
                        flw[:dsmax_dp_flow][bit[:sto_vio]]   .= flw[:pflow_k][bit[:sto_vio]]./flw[:sto][bit[:sto_vio]]
                        flw[:dsmax_dqfr_flow][bit[:sfr_vio]] .= flw[:ac_qfr][bit[:sfr_vio]]./flw[:sfr][bit[:sfr_vio]]
                        flw[:dsmax_dqto_flow][bit[:sto_vio]] .= flw[:ac_qto][bit[:sto_vio]]./flw[:sto][bit[:sto_vio]]

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
                        if 1 in bit[:sfr_vio][1:sys.nl]
                            # deal with the fr line
                            aclfr_alpha = gc*(flw[:dsmax_dqfr_flow][1:sys.nl][bit[:sfr_vio][1:sys.nl]])
                            zctgs_grad_qfr_acline!(aclfr_alpha, bit, grd, idx, mgd, prm, qG, sys, tii)
                        end

                        if 1 in bit[:sto_vio][1:sys.nl]
                            # deal with the to line
                            aclto_alpha = gc*(flw[:dsmax_dqto_flow][1:sys.nl][bit[:sto_vio][1:sys.nl]])
                            zctgs_grad_qto_acline!(aclto_alpha, bit, grd, idx, mgd, prm, qG, sys, tii)
                        end

                        # slower:
                            # => aclfr_inds  = findall(!iszero,bit[:sfr_vio][1:sys.nl])
                            # => aclto_inds  = findall(!iszero,bit[:sto_vio][1:sys.nl])
                            # => aclfr_alpha = gc*(flw[:dsmax_dqfr_flow][1:sys.nl][aclfr_inds])
                            # => aclto_alpha = gc*(flw[:dsmax_dqto_flow][1:sys.nl][aclto_inds])
                            # => zctgs_grad_q_acline!(tii, idx, grd, mgd, aclfr_inds, aclto_inds, aclfr_alpha, aclto_alpha)

                        # xfm
                        if 1 in bit[:sfr_vio][(sys.nl+1):sys.nac]
                            xfr_alpha = gc*(flw[:dsmax_dqfr_flow][(sys.nl+1):sys.nac][bit[:sfr_vio][(sys.nl+1):sys.nac]])
                            zctgs_grad_qfr_xfm!(bit, grd, idx, mgd, prm, qG, sys, tii, xfr_alpha)
                        end
                        if 1 in bit[:sto_vio][(sys.nl+1):sys.nac]
                            xfr_alpha = gc*(flw[:dsmax_dqfr_flow][(sys.nl+1):sys.nac][bit[:sto_vio][(sys.nl+1):sys.nac]])
                            zctgs_grad_qto_xfm!(bit, grd, idx, mgd, prm, qG, sys, tii, xto_alpha)
                        end

                        # slower:
                            # => xfr_inds  = findall(!iszero,bit[:sfr_vio][(sys.nl+1):sys.nac])
                            # => xto_inds  = findall(!iszero,bit[:sto_vio][(sys.nl+1):sys.nac])
                            # => xfr_alpha = gc*(flw[:dsmax_dqfr_flow][(sys.nl+1):sys.nac][xfr_inds])
                            # => xto_alpha = gc*(flw[:dsmax_dqto_flow][(sys.nl+1):sys.nac][xto_inds])
                            # => zctgs_grad_q_xfm!(tii, idx, grd, mgd, xfr_inds, xto_inds, xfr_alpha, xto_alpha)

                        # now, the fun one: active power injection + xfm phase shift!!
                        # **** => alpha_p_flow_phi = gc*flw[:dsmax_dp_flow]
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
                        # => flw[:rhs] .= ntk.YfrT*(gc.*flw[:dsmax_dp_flow])
                        flw[:dsmax_dp_flow] .= gc.*flw[:dsmax_dp_flow]
                        quasiGrad.mul!(flw[:rhs], ntk.YfrT, flw[:dsmax_dp_flow]);
                        # time to solve for dz_dpinj -- two options here:
                        #   1. solve with ntk.Ybr_k, but we didn't actually build this,
                        #      and we didn't build its preconditioner either..
                        #   2. solve with ntk.Ybr, and then use a rank 1 update! Let's do
                        #      this instead :) we'll do this in-loop for each ctg at each time.
                        # => flw[:dz_dpinj] .= lowrank_update_single_ctg_gradient(ctd, ctg_ii, ntk, qG, flw[:rhs], sys)
                        lowrank_update_single_ctg_gradient!(ctd, ctg_ii, flw, ntk, qG, sys)
                        
                        # now, we have the gradient of znms wrt all nodal injections/xfm phase shifts!!!
                        # except the slack bus... time to apply these gradients into 
                        # the master grad at all buses except the slack bus.
                        #
                        # update the injection gradient to account for slack!
                        #   alternative direct solution: 
                        #       -> ctg[:dz_dpinj][tii][ctg_ii] = (quasiGrad.I-ones(sys.nb-1)*ones(sys.nb-1)'/(sys.nb))*(ntk.Ybr_k[ctg_ii]\(ntk.YfrT*alpha_p_flow))
                        flw[:dz_dpinj_all] .+= flw[:dz_dpinj] .- sum(flw[:dz_dpinj])/Float64(sys.nb)

                        # legacy option: apply device gradients -- super slow!!
                            # => zctgs_grad_pinj!(dz_dpinj, grd, idx, mgd, ntk, prm, sys, tii)
                    end
                end
            end

            # now, actually apply the active power gradients! In the reactive power case, we just apply as we go
            if qG.eval_grad
                zctgs_grad_pinj!(flw[:dz_dpinj_all], grd, idx, mgd, ntk, prm, sys, tii)
            end
            # across each contingency, we get the average, and we get the min
            scr[:zctg_min] += minimum(stt[:zctg][tii])
            scr[:zctg_avg] += sum(stt[:zctg][tii])/sys.nctg

            # now that we have scored all contingencies at this given time,
            # rank them from most negative to least (worst is first)
            wct[t_ind][1:num_ctg] .= sortperm(stt[:zctg][tii][wct[t_ind][1:num_ctg]])

            # however, only keep half!
            wct[t_ind][1:num_ctg] .= union(wct[t_ind][1:num_wrst],quasiGrad.shuffle(setdiff(1:sys.nctg, wct[t_ind][1:num_wrst]))[1:num_rnd])
        end
    end
end

function lowrank_update_single_ctg_gradient!(ctd::Vector{Vector{Float64}}, ctg_ii::Int64, flw::Dict{Symbol, Vector{Float64}}, ntk::quasiGrad.Ntk, qG::quasiGrad.QG, sys::quasiGrad.System)
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
        ctd[ctg_ii] .= ntk.Ybr\flw[:rhs]

        # error with this type !!!
    # elseif qG.base_solver == "cholesky"
    #    ctd[ctg_ii] = ntk.Ybr_Ch\rhs

    elseif qG.base_solver == "pcg"
        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- just use LU
            ctd[ctg_ii] .= ntk.Ybr\flw[:rhs]
            
        else
            # solve with a hot start!
            _, ch = quasiGrad.cg!(ctd[ctg_ii], ntk.Ybr, flw[:rhs], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
        
            # test the krylov solution
            if ~(ch.isconverged)
                # LU backup
                @info "Krylov failed -- using LU backup (ctg gradient)"
                ctd[ctg_ii] .= ntk.Ybr\flw[:rhs]
            end
        end
    end

    # step 2:
    # now, apply a low-rank update!
    # explicit version => dz_dpinj = ctd[ctg_ii] - Vector(ntk.u_k[ctg_ii]*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], rhs)))
    flw[:dz_dpinj] .= special_wmi_update(ctd[ctg_ii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:rhs])
end

function zctgs_grad_qfr_acline!(aclfr_alpha::Vector{Float64}, bit::Dict{Symbol, Dict{Symbol, BitVector}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Symbol)
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
    vmfrqfr = aclfr_alpha.*grd[:acline_qfr][:vmfr][tii][bit[:sfr_vio][1:sys.nl]]
    vmtoqfr = aclfr_alpha.*grd[:acline_qfr][:vmto][tii][bit[:sfr_vio][1:sys.nl]]
    vafrqfr = aclfr_alpha.*grd[:acline_qfr][:vafr][tii][bit[:sfr_vio][1:sys.nl]]
    vatoqfr = aclfr_alpha.*grd[:acline_qfr][:vato][tii][bit[:sfr_vio][1:sys.nl]]

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(prm.acline.line_inds[bit[:sfr_vio][1:sys.nl]])
        # update the master grad -- qfr
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqfr[ii]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqfr[ii]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqfr[ii]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqfr[ii]
    end

    # NOT efficient
    if qG.change_ac_device_bins
        uonqfr  = aclfr_alpha.*grd[:acline_qfr][:uon][tii][bit[:sfr_vio][1:sys.nl]]
        for (ii,ln) in enumerate(prm.acline.line_inds[bit[:sfr_vio][1:sys.nl]])
            mgd[:u_on_acline][tii][ln]           += uonqfr[ii]
        end
    end
end

function zctgs_grad_qto_acline!(aclto_alpha::Vector{Float64}, bit::Dict{Symbol, Dict{Symbol, BitVector}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Symbol)
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
    vmfrqto = aclto_alpha.*grd[:acline_qto][:vmfr][tii][bit[:sto_vio][1:sys.nl]]
    vmtoqto = aclto_alpha.*grd[:acline_qto][:vmto][tii][bit[:sto_vio][1:sys.nl]]
    vafrqto = aclto_alpha.*grd[:acline_qto][:vafr][tii][bit[:sto_vio][1:sys.nl]]
    vatoqto = aclto_alpha.*grd[:acline_qto][:vato][tii][bit[:sto_vio][1:sys.nl]]

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(prm.acline.line_inds[bit[:sto_vio][1:sys.nl]])
        # update the master grad -- qto
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqto[ii]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqto[ii]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqto[ii]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqto[ii]
    end

    # NOT efficient
    if qG.change_ac_device_bins
        uonqto  = aclto_alpha.*grd[:acline_qto][:uon][tii][bit[:sto_vio][1:sys.nl]]
        for (ii,ln) in enumerate(prm.acline.line_inds[bit[:sto_vio][1:sys.nl]])
            mgd[:u_on_acline][tii][ln]           += uonqto[ii]
        end
    end
end

function zctgs_grad_qfr_xfm!(bit::Dict{Symbol, Dict{Symbol, BitVector}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Symbol, xfr_alpha::Vector{Float64})
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

    vmfrqfr = xfr_alpha.*grd[:xfm_qfr][:vmfr][tii][bit[:sfr_vio][(sys.nl+1):sys.nac]]
    vmtoqfr = xfr_alpha.*grd[:xfm_qfr][:vmto][tii][bit[:sfr_vio][(sys.nl+1):sys.nac]]
    vafrqfr = xfr_alpha.*grd[:xfm_qfr][:vafr][tii][bit[:sfr_vio][(sys.nl+1):sys.nac]]
    vatoqfr = xfr_alpha.*grd[:xfm_qfr][:vato][tii][bit[:sfr_vio][(sys.nl+1):sys.nac]]
    tauqfr  = xfr_alpha.*grd[:xfm_qfr][:tau][tii][bit[:sfr_vio][(sys.nl+1):sys.nac]]
    phiqfr  = xfr_alpha.*grd[:xfm_qfr][:phi][tii][bit[:sfr_vio][(sys.nl+1):sys.nac]]

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit[:sfr_vio][(sys.nl+1):sys.nac]])
        # update the master grad -- qfr
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqfr[ii]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqfr[ii]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqfr[ii]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqfr[ii]
        mgd[:tau][tii][xfm]                += tauqfr[ii]
        mgd[:phi][tii][xfm]                += phiqfr[ii]
    end

    # NOT efficient
    if qG.change_ac_device_bins
        uonqfr  = xfr_alpha.*grd[:xfm_qfr][:uon][tii][bit[:sfr_vio][(sys.nl+1):sys.nac]]
        for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit[:sfr_vio][(sys.nl+1):sys.nac]])
            mgd[:u_on_xfm][tii][xfm] += uonqfr[ii]
        end
    end
end

function zctgs_grad_qto_xfm!(bit::Dict{Symbol, Dict{Symbol, BitVector}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Symbol, xto_alpha::Vector{Float64})
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
    vmfrqto = xto_alpha.*grd[:xfm_qto][:vmfr][tii][bit[:sto_vio][(sys.nl+1):sys.nac]]
    vmtoqto = xto_alpha.*grd[:xfm_qto][:vmto][tii][bit[:sto_vio][(sys.nl+1):sys.nac]]
    vafrqto = xto_alpha.*grd[:xfm_qto][:vafr][tii][bit[:sto_vio][(sys.nl+1):sys.nac]]
    vatoqto = xto_alpha.*grd[:xfm_qto][:vato][tii][bit[:sto_vio][(sys.nl+1):sys.nac]]
    tauqto  = xto_alpha.*grd[:xfm_qto][:tau][tii][bit[:sto_vio][(sys.nl+1):sys.nac]]
    phiqto  = xto_alpha.*grd[:xfm_qto][:phi][tii][bit[:sto_vio][(sys.nl+1):sys.nac]]

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit[:sto_vio][(sys.nl+1):sys.nac]])
        # update the master grad -- qto
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqto[ii]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqto[ii]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqto[ii]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqto[ii]
        mgd[:tau][tii][xfm]                += tauqto[ii]
        mgd[:phi][tii][xfm]                += phiqto[ii]
    end

    # NOT efficient
    if qG.change_ac_device_bins
        uonqto  = xto_alpha.*grd[:xfm_qto][:uon][tii][bit[:sto_vio][(sys.nl+1):sys.nac]]
        for (ii,xfm) in enumerate(prm.xfm.xfm_inds[bit[:sto_vio][(sys.nl+1):sys.nac]])
            mgd[:u_on_xfm][tii][xfm]           += uonqto[ii]
        end
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
            mgd[:phi][tii][ntk.xfm_at_bus[bus]] .+= alpha[bus-1].*ntk.xfm_phi_scalars[bus]
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
    return y0 .- u.*(g*quasiGrad.dot(u, x))
end

function get_largest_ctg_indices(bit::Dict{Symbol, Dict{Symbol, BitVector}}, flw::Dict{Symbol, Vector{Float64}}, qG::quasiGrad.QG, s1::Symbol, s2::Symbol)
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

#=
function solve_ctgs_parallel!(
    bit::Dict{Symbol, Dict{Symbol, BitVector}},
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
    num_wrst = Int64(ceil(qG.frac_ctg_keep*sys.nctg/2))  # in case n_ctg is odd, and we want to keep all!
    num_rnd  = Int64(floor(qG.frac_ctg_keep*sys.nctg/2)) # in case n_ctg is odd, and we want to keep all!
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
        #   c = p - ntk.ErT*bt
        #
        # simplified:
        # => flw[:c] .= (@view flw[:p_inj][2:end]) .- ntk.ErT*flw[:bt]
            # this is a little odd, but it's fine (the first use of flw[:c] is just for storage!)
        quasiGrad.mul!(flw[:c], ntk.ErT, flw[:bt])
        flw[:c] .= (@view flw[:p_inj][2:end]) .- flw[:c]

        # solve the base case with pcg
        if qG.base_solver == "lu"
            ctb[t_ind] .= ntk.Ybr\flw[:c]

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

        # define the ctg 
        cs = dt*prm.vio.s_flow*qG.scale_c_sflow_testing

        # do we want to score all ctgs? for testing/post processing
        if qG.score_all_ctgs == true
            ###########################################################
            # => up above: @info "Warning -- scoring all contingencies! No gradients."
            ###########################################################
            for ctg_ii in 1:sys.nctg
                # see the "else" case for comments and details
                flw[:theta_k] .= special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
                # => slow: flw[:pflow_k] .= ntk.Yfr*flw[:theta_k] .+ flw[:bt]
                quasiGrad.mul!(flw[:pflow_k], ntk.Yfr, flw[:theta_k])
                flw[:pflow_k] .+= flw[:bt]
                flw[:sfr]     .= sqrt.(flw[:qfr2] .+ flw[:pflow_k].^2)
                flw[:sto]     .= sqrt.(flw[:qto2] .+ flw[:pflow_k].^2)
                flw[:sfr_vio] .= flw[:sfr] .- ntk.s_max
                flw[:sto_vio] .= flw[:sto] .- ntk.s_max
                flw[:sfr_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                flw[:sto_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                    # => flw[:smax_vio] .= max.(flw[:sfr_vio], flw[:sto_vio], 0.0)
                    # => if helpful: zctg_s = cs*flw[:smax_vio]
                    # => if helpful: stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)
                stt[:zctg][tii][ctg_ii] = -cs*sum(max.(flw[:sfr_vio], flw[:sto_vio], 0.0))
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
                flw[:theta_k] .= special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
                # compute flows
                #
                # NOTE: ctg[:pflow_k][tii][ctg_ii] contains the flow on the outaged line --
                #       -- this will be dealt with when computing the flows and gradients
                # => slow: flw[:pflow_k] .= ntk.Yfr*flw[:theta_k] .+ flw[:bt]
                quasiGrad.mul!(flw[:pflow_k], ntk.Yfr, flw[:theta_k])
                flw[:pflow_k] .+= flw[:bt]
                flw[:sfr]     .= sqrt.(flw[:qfr2] .+ flw[:pflow_k].^2)
                flw[:sto]     .= sqrt.(flw[:qto2] .+ flw[:pflow_k].^2)
                flw[:sfr_vio] .= flw[:sfr] .- ntk.s_max
                flw[:sto_vio] .= flw[:sto] .- ntk.s_max

                # make sure there are no penalties on lines that are out-aged!
                flw[:sfr_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                flw[:sto_vio][ntk.ctg_out_ind[ctg_ii]] .= 0.0

                # compute the penalties: "stt[:zctg_s][tii][ctg_ii]" -- if want to keep
                    # => if helpful: smax_vio = max.(sfr_vio, sto_vio, 0.0)
                    # => if helpful: zctg_s = cs*flw[:smax_vio]
                    # => if helpful: stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)

                # each contingency, at each time, gets a score:
                stt[:zctg][tii][ctg_ii] = -cs*sum(max.(flw[:sfr_vio], flw[:sto_vio], 0.0), init=0.0)

                # great -- now, do we take the gradient?
                if qG.eval_grad
                    # only take the gradient if the ctg violation is sufficiently large!!!
                    if stt[:zctg][tii][ctg_ii] < qG.ctg_grad_cutoff
                        # game on :)
                        #
                        # in this code, we assume we take the gradient of all scored
                        # contingencies -- this can be updated!

                        # What are the gradients? build indicators with some tolerance
                        # slower => get_largest_ctg_indices(bit, flw, qG, :sfr_vio, :sto_vio)
                        bit[:sfr_vio] .= (flw[:sfr_vio] .> qG.grad_ctg_tol) .&& (flw[:sfr_vio] .> flw[:sto_vio])
                        bit[:sto_vio] .= (flw[:sto_vio] .> qG.grad_ctg_tol) .&& (flw[:sto_vio] .> flw[:sfr_vio])

                        # build the grads
                        flw[:dsmax_dqfr_flow]                .= 0.0
                        flw[:dsmax_dqto_flow]                .= 0.0
                        flw[:dsmax_dp_flow]                  .= 0.0
                        flw[:dsmax_dp_flow][bit[:sfr_vio]]   .= flw[:pflow_k][bit[:sfr_vio]]./flw[:sfr][bit[:sfr_vio]]
                        flw[:dsmax_dp_flow][bit[:sto_vio]]   .= flw[:pflow_k][bit[:sto_vio]]./flw[:sto][bit[:sto_vio]]
                        flw[:dsmax_dqfr_flow][bit[:sfr_vio]] .= flw[:ac_qfr][bit[:sfr_vio]]./flw[:sfr][bit[:sfr_vio]]
                        flw[:dsmax_dqto_flow][bit[:sto_vio]] .= flw[:ac_qto][bit[:sto_vio]]./flw[:sto][bit[:sto_vio]]

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
                        if 1 in bit[:sfr_vio][1:sys.nl]
                            # deal with the fr line
                            aclfr_alpha = gc*(flw[:dsmax_dqfr_flow][1:sys.nl][bit[:sfr_vio][1:sys.nl]])
                            zctgs_grad_qfr_acline!(aclfr_alpha, bit, grd, idx, mgd, prm, qG, sys, tii)
                        end

                        if 1 in bit[:sto_vio][1:sys.nl]
                            # deal with the to line
                            aclto_alpha = gc*(flw[:dsmax_dqto_flow][1:sys.nl][bit[:sto_vio][1:sys.nl]])
                            zctgs_grad_qto_acline!(aclto_alpha, bit, grd, idx, mgd, prm, qG, sys, tii)
                        end

                        # slower:
                            # => aclfr_inds  = findall(!iszero,bit[:sfr_vio][1:sys.nl])
                            # => aclto_inds  = findall(!iszero,bit[:sto_vio][1:sys.nl])
                            # => aclfr_alpha = gc*(flw[:dsmax_dqfr_flow][1:sys.nl][aclfr_inds])
                            # => aclto_alpha = gc*(flw[:dsmax_dqto_flow][1:sys.nl][aclto_inds])
                            # => zctgs_grad_q_acline!(tii, idx, grd, mgd, aclfr_inds, aclto_inds, aclfr_alpha, aclto_alpha)

                        # xfm
                        if 1 in bit[:sfr_vio][(sys.nl+1):sys.nac]
                            xfr_alpha = gc*(flw[:dsmax_dqfr_flow][(sys.nl+1):sys.nac][bit[:sfr_vio][(sys.nl+1):sys.nac]])
                            zctgs_grad_qfr_xfm!(bit, grd, idx, mgd, prm, qG, sys, tii, xfr_alpha)
                        end
                        if 1 in bit[:sto_vio][(sys.nl+1):sys.nac]
                            xfr_alpha = gc*(flw[:dsmax_dqfr_flow][(sys.nl+1):sys.nac][bit[:sto_vio][(sys.nl+1):sys.nac]])
                            zctgs_grad_qto_xfm!(bit, grd, idx, mgd, prm, qG, sys, tii, xto_alpha)
                        end

                        # slower:
                            # => xfr_inds  = findall(!iszero,bit[:sfr_vio][(sys.nl+1):sys.nac])
                            # => xto_inds  = findall(!iszero,bit[:sto_vio][(sys.nl+1):sys.nac])
                            # => xfr_alpha = gc*(flw[:dsmax_dqfr_flow][(sys.nl+1):sys.nac][xfr_inds])
                            # => xto_alpha = gc*(flw[:dsmax_dqto_flow][(sys.nl+1):sys.nac][xto_inds])
                            # => zctgs_grad_q_xfm!(tii, idx, grd, mgd, xfr_inds, xto_inds, xfr_alpha, xto_alpha)

                        # now, the fun one: active power injection + xfm phase shift!!
                        # **** => alpha_p_flow_phi = gc*flw[:dsmax_dp_flow]
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
                        # => flw[:rhs] .= ntk.YfrT*(gc.*flw[:dsmax_dp_flow])
                        flw[:dsmax_dp_flow] .= gc.*flw[:dsmax_dp_flow]
                        quasiGrad.mul!(flw[:rhs], ntk.YfrT, flw[:dsmax_dp_flow]);
                        # time to solve for dz_dpinj -- two options here:
                        #   1. solve with ntk.Ybr_k, but we didn't actually build this,
                        #      and we didn't build its preconditioner either..
                        #   2. solve with ntk.Ybr, and then use a rank 1 update! Let's do
                        #      this instead :) we'll do this in-loop for each ctg at each time.
                        # => flw[:dz_dpinj] .= lowrank_update_single_ctg_gradient(ctd, ctg_ii, ntk, qG, flw[:rhs], sys)
                        lowrank_update_single_ctg_gradient!(ctd, ctg_ii, flw, ntk, qG, sys)
                        
                        # now, we have the gradient of znms wrt all nodal injections/xfm phase shifts!!!
                        # except the slack bus... time to apply these gradients into 
                        # the master grad at all buses except the slack bus.
                        #
                        # update the injection gradient to account for slack!
                        #   alternative direct solution: 
                        #       -> ctg[:dz_dpinj][tii][ctg_ii] = (quasiGrad.I-ones(sys.nb-1)*ones(sys.nb-1)'/(sys.nb))*(ntk.Ybr_k[ctg_ii]\(ntk.YfrT*alpha_p_flow))
                        flw[:dz_dpinj_all] .+= flw[:dz_dpinj] .- sum(flw[:dz_dpinj])/Float64(sys.nb)

                        # legacy option: apply device gradients -- super slow!!
                            # => zctgs_grad_pinj!(dz_dpinj, grd, idx, mgd, ntk, prm, sys, tii)
                    end
                end
            end

            # now, actually apply the active power gradients! In the reactive power case, we just apply as we go
            if qG.eval_grad
                zctgs_grad_pinj!(flw[:dz_dpinj_all], grd, idx, mgd, ntk, prm, sys, tii)
            end
            # across each contingency, we get the average, and we get the min
            scr[:zctg_min] += minimum(stt[:zctg][tii])
            scr[:zctg_avg] += sum(stt[:zctg][tii])/sys.nctg

            # now that we have scored all contingencies at this given time,
            # rank them from most negative to least (worst is first)
            wct[t_ind][1:num_ctg] .= sortperm(stt[:zctg][tii][wct[t_ind][1:num_ctg]])

            # however, only keep half!
            wct[t_ind][1:num_ctg] .= union(wct[t_ind][1:num_wrst],quasiGrad.shuffle(setdiff(1:sys.nctg, wct[t_ind][1:num_wrst]))[1:num_rnd])
        end
    end
end

function acline_flows_parallel!(bit::Dict{Symbol, Dict{Symbol, BitVector}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # line parameters
    g_sr = prm.acline.g_sr
    b_sr = prm.acline.b_sr
    b_ch = prm.acline.b_ch
    g_fr = prm.acline.g_fr
    b_fr = prm.acline.b_fr
    g_to = prm.acline.g_to
    b_to = prm.acline.b_to

    # call penalty costs
    cs = prm.vio.s_flow * qG.scale_c_sflow_testing

    # loop over time
    quasiGrad.@floop ThreadedEx(basesize = sys.nT ÷ qG.num_threads ) for tii in prm.ts.time_keys

        # duration
        dt = prm.ts.duration[tii]

        # organize relevant line values
        vm_fr = stt[:vm][tii][idx.acline_fr_bus]
        va_fr = stt[:va][tii][idx.acline_fr_bus]
        vm_to = stt[:vm][tii][idx.acline_to_bus]
        va_to = stt[:va][tii][idx.acline_to_bus]
        
        # tools
        msc[:cos_ftp]  .= cos.(va_fr .- va_to)
        msc[:sin_ftp]  .= sin.(va_fr .- va_to)
        msc[:vff]      .= vm_fr.^2
        msc[:vtt]      .= vm_to.^2
        msc[:vft]      .= vm_fr.*vm_to
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        msc[:pfr] .= (g_sr.+g_fr).*msc[:vff] .+ (.-g_sr.*msc[:cos_ftp] .- b_sr.*msc[:sin_ftp]).*msc[:vft]
        stt[:acline_pfr][tii] .= stt[:u_on_acline][tii].*msc[:pfr]
        
        # reactive power flow -- from -> to
        msc[:qfr] .= (.-b_sr.-b_fr.-b_ch./2.0).*msc[:vff] .+ (b_sr.*msc[:cos_ftp] .- g_sr.*msc[:sin_ftp]).*msc[:vft]
        stt[:acline_qfr][tii] .= stt[:u_on_acline][tii].*msc[:qfr]
        
        # apparent power flow -- to -> from
        msc[:acline_sfr] .= sqrt.(stt[:acline_pfr][tii].^2 .+ stt[:acline_qfr][tii].^2)
        
        # active power flow -- to -> from
        msc[:pto] .= (g_sr.+g_to).*msc[:vtt] .+ (.-g_sr.*msc[:cos_ftp] .+ b_sr.*msc[:sin_ftp]).*msc[:vft]
        stt[:acline_pto][tii] .= stt[:u_on_acline][tii].*msc[:pto]
        
        # reactive power flow -- to -> from
        msc[:qto] .= (.-b_sr.-b_to.-b_ch./2.0).*msc[:vtt] .+ (b_sr.*msc[:cos_ftp] .+ g_sr.*msc[:sin_ftp]).*msc[:vft]
        stt[:acline_qto][tii] .= stt[:u_on_acline][tii].*msc[:qto]

        # apparent power flow -- to -> from
        msc[:acline_sto] .= sqrt.(stt[:acline_pto][tii].^2 .+ stt[:acline_qto][tii].^2)
        
        # penalty functions and scores
        msc[:acline_sfr_plus] .= msc[:acline_sfr] .- prm.acline.mva_ub_nom
        msc[:acline_sto_plus] .= msc[:acline_sto] .- prm.acline.mva_ub_nom
        stt[:zs_acline][tii]  .= (dt*cs).*max.(msc[:acline_sfr_plus], msc[:acline_sto_plus], 0.0)

        # ====================================================== #
        # ====================================================== #
        #
        # evaluate the grd?
        if qG.eval_grad

            # Gradients: active power flow -- from -> to
            grd[:acline_pfr][:vmfr][tii] .= stt[:u_on_acline][tii].*(2.0.*(g_sr.+g_fr).*vm_fr .+ 
                    (.-g_sr.*msc[:cos_ftp] .- b_sr.*msc[:sin_ftp]).*vm_to)
            grd[:acline_pfr][:vmto][tii] .= stt[:u_on_acline][tii].*(
                    (.-g_sr.*msc[:cos_ftp] .- b_sr.*msc[:sin_ftp]).*vm_fr)
            grd[:acline_pfr][:vafr][tii] .= stt[:u_on_acline][tii].*(
                    (g_sr.*msc[:sin_ftp] .- b_sr.*msc[:cos_ftp]).*msc[:vft])
            grd[:acline_pfr][:vato][tii] .= stt[:u_on_acline][tii].*(
                    (.-g_sr.*msc[:sin_ftp] .+ b_sr.*msc[:cos_ftp]).*msc[:vft])
            if qG.change_ac_device_bins
                grd[:acline_pfr][:uon][tii] .= msc[:pfr]   
            end
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            
            grd[:acline_qfr][:vmfr][tii] .= stt[:u_on_acline][tii].*(2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*vm_fr .+
                    (b_sr.*msc[:cos_ftp] .- g_sr.*msc[:sin_ftp]).*vm_to)
            grd[:acline_qfr][:vmto][tii] .= stt[:u_on_acline][tii].*(
                    (b_sr.*msc[:cos_ftp] .- g_sr.*msc[:sin_ftp]).*vm_fr)
            grd[:acline_qfr][:vafr][tii] .= stt[:u_on_acline][tii].*(
                    (.-b_sr.*msc[:sin_ftp] .- g_sr.*msc[:cos_ftp]).*msc[:vft])
            grd[:acline_qfr][:vato][tii] .= stt[:u_on_acline][tii].*(
                    (b_sr.*msc[:sin_ftp] .+ g_sr.*msc[:cos_ftp]).*msc[:vft])
            if qG.change_ac_device_bins
                grd[:acline_qfr][:uon][tii] .= msc[:qfr] 
            end
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
                           
            grd[:acline_pto][:vmfr][tii] .= stt[:u_on_acline][tii].*( 
                    (.-g_sr.*msc[:cos_ftp] .+ b_sr.*msc[:sin_ftp]).*vm_to)
            grd[:acline_pto][:vmto][tii] .= stt[:u_on_acline][tii].*(2.0.*(g_sr.+g_to).*vm_to .+
                    (.-g_sr.*msc[:cos_ftp] .+ b_sr.*msc[:sin_ftp]).*vm_fr)
            grd[:acline_pto][:vafr][tii] .= stt[:u_on_acline][tii].*(
                    (g_sr.*msc[:sin_ftp] .+ b_sr.*msc[:cos_ftp]).*msc[:vft])
            grd[:acline_pto][:vato][tii] .= stt[:u_on_acline][tii].*(
                    (.-g_sr.*msc[:sin_ftp] .- b_sr.*msc[:cos_ftp]).*msc[:vft])
            if qG.change_ac_device_bins
                grd[:acline_pto][:uon][tii] .= msc[:pto]
            end
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            
            grd[:acline_qto][:vmfr][tii] .= stt[:u_on_acline][tii].*(
                    (b_sr.*msc[:cos_ftp] .+ g_sr.*msc[:sin_ftp]).*vm_to)
            grd[:acline_qto][:vmto][tii] .= stt[:u_on_acline][tii].*(2.0.*(.-b_sr.-b_to.-b_ch./2.0).*vm_to .+
                    (b_sr.*msc[:cos_ftp] .+ g_sr.*msc[:sin_ftp]).*vm_fr)
            grd[:acline_qto][:vafr][tii] .= stt[:u_on_acline][tii].*(
                    (.-b_sr.*msc[:sin_ftp] .+ g_sr.*msc[:cos_ftp]).*msc[:vft])
            grd[:acline_qto][:vato][tii] .= stt[:u_on_acline][tii].*(
                    (b_sr.*msc[:sin_ftp] .- g_sr.*msc[:cos_ftp]).*msc[:vft])
            if qG.change_ac_device_bins
                grd[:acline_qto][:uon][tii] .= msc[:qto] 
            end

            # apply gradients
            grd[:zs_acline][:acline_pfr][tii] .= 0.0
            grd[:zs_acline][:acline_qfr][tii] .= 0.0
            grd[:zs_acline][:acline_pto][tii] .= 0.0
            grd[:zs_acline][:acline_qto][tii] .= 0.0  

            # indicators
            # => slower :( quasiGrad.get_largest_indices(msc, bit, :acline_sfr_plus, :acline_sto_plus)
            bit[:acline_sfr_plus] .= (msc[:acline_sfr_plus] .> 0.0) .&& (msc[:acline_sfr_plus] .> msc[:acline_sto_plus]);
            bit[:acline_sto_plus] .= (msc[:acline_sto_plus] .> 0.0) .&& (msc[:acline_sto_plus] .> msc[:acline_sfr_plus]); 
            #
            # slower alternative
                # => max_sfst0 = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(msc[:acline_sfr_plus], msc[:acline_sto_plus])]
                # => ind_fr = max_sfst0 .== 1
                # => ind_to = max_sfst0 .== 2

            # flush -- set to 0 => [Not(ind_fr)] and [Not(ind_to)] was slow/memory inefficient
            grd[:zs_acline][:acline_pfr][tii] .= 0.0
            grd[:zs_acline][:acline_qfr][tii] .= 0.0
            grd[:zs_acline][:acline_pto][tii] .= 0.0
            grd[:zs_acline][:acline_qto][tii] .= 0.0

            if qG.acflow_grad_is_soft_abs
                # compute the scaled gradients
                if sum(bit[:acline_sfr_plus]) > 0
                    msc[:acline_scale_fr][bit[:acline_sfr_plus]]             .= msc[:acline_sfr_plus][bit[:acline_sfr_plus]]./sqrt.(msc[:acline_sfr_plus][bit[:acline_sfr_plus]].^2 .+ qG.acflow_grad_eps2);
                    grd[:zs_acline][:acline_pfr][tii][bit[:acline_sfr_plus]] .= msc[:acline_scale_fr][bit[:acline_sfr_plus]].*((dt*qG.acflow_grad_weight).*stt[:acline_pfr][tii][bit[:acline_sfr_plus]]./msc[:acline_sfr][bit[:acline_sfr_plus]])
                    grd[:zs_acline][:acline_qfr][tii][bit[:acline_sfr_plus]] .= msc[:acline_scale_fr][bit[:acline_sfr_plus]].*((dt*qG.acflow_grad_weight).*stt[:acline_qfr][tii][bit[:acline_sfr_plus]]./msc[:acline_sfr][bit[:acline_sfr_plus]])
                end
                # compute the scaled gradients
                if sum(bit[:acline_sto_plus]) > 0
                    msc[:acline_scale_to][bit[:acline_sto_plus]]             .= msc[:acline_sto_plus][bit[:acline_sto_plus]]./sqrt.(msc[:acline_sto_plus][bit[:acline_sto_plus]].^2 .+ qG.acflow_grad_eps2);
                    grd[:zs_acline][:acline_pto][tii][bit[:acline_sto_plus]] .= msc[:acline_scale_to][bit[:acline_sto_plus]].*((dt*qG.acflow_grad_weight).*stt[:acline_pto][tii][bit[:acline_sto_plus]]./msc[:acline_sto][bit[:acline_sto_plus]])
                    grd[:zs_acline][:acline_qto][tii][bit[:acline_sto_plus]] .= msc[:acline_scale_to][bit[:acline_sto_plus]].*((dt*qG.acflow_grad_weight).*stt[:acline_qto][tii][bit[:acline_sto_plus]]./msc[:acline_sto][bit[:acline_sto_plus]])
                end
            else
                # gradients
                grd[:zs_acline][:acline_pfr][tii][bit[:acline_sfr_plus]] .= (dt*cs).*stt[:acline_pfr][tii][bit[:acline_sfr_plus]]./msc[:acline_sfr][bit[:acline_sfr_plus]]
                grd[:zs_acline][:acline_qfr][tii][bit[:acline_sfr_plus]] .= (dt*cs).*stt[:acline_qfr][tii][bit[:acline_sfr_plus]]./msc[:acline_sfr][bit[:acline_sfr_plus]]
                grd[:zs_acline][:acline_pto][tii][bit[:acline_sto_plus]] .= (dt*cs).*stt[:acline_pto][tii][bit[:acline_sto_plus]]./msc[:acline_sto][bit[:acline_sto_plus]]
                grd[:zs_acline][:acline_qto][tii][bit[:acline_sto_plus]] .= (dt*cs).*stt[:acline_qto][tii][bit[:acline_sto_plus]]./msc[:acline_sto][bit[:acline_sto_plus]]
            end

            #= Previous gradient junk
            # loop
            if qG.acflow_grad_is_soft_abs
                dtgw = dt*qG.acflow_grad_weight
                for xx in 1:sys.nl
                    if (msc[:acline_sfr_plus][xx] >= msc[:acline_sto_plus][xx]) && (msc[:acline_sfr_plus][xx] > 0.0)
                        #msc[:pub][1] = quasiGrad.soft_abs_grad_ac(xx, msc, qG, :acline_sfr_plus)
                        grd[:zs_acline][:acline_pfr][tii][xx] = msc[:pub][1]#*dtgw#*stt[:acline_pfr][tii][xx]/msc[:acline_sfr][xx]
                        grd[:zs_acline][:acline_qfr][tii][xx] = msc[:pub][1]#*dtgw#*stt[:acline_qfr][tii][xx]/msc[:acline_sfr][xx]
                    elseif (msc[:acline_sto_plus][xx] > 0.0)
                        #msc[:pub][1] = quasiGrad.soft_abs_grad_ac(xx, msc, qG, :acline_sto_plus)
                        grd[:zs_acline][:acline_pto][tii][xx] = msc[:pub][1]#*dtgw#*stt[:acline_pto][tii][xx]/msc[:acline_sto][xx]
                        grd[:zs_acline][:acline_qto][tii][xx] = msc[:pub][1]#*dtgw#*stt[:acline_qto][tii][xx]/msc[:acline_sto][xx]
                    end
                end
            # no softabs -- use standard
            else
                dtcs = dt*cs
                for xx in 1:sys.nl
                    if (msc[:acline_sfr_plus][xx] >= msc[:acline_sto_plus][xx]) && (msc[:acline_sfr_plus][xx] > 0.0)
                        grd[:zs_acline][:acline_pfr][tii][xx] = dtcs*stt[:acline_pfr][tii][xx]/msc[:acline_sfr][xx]
                        grd[:zs_acline][:acline_qfr][tii][xx] = dtcs*stt[:acline_qfr][tii][xx]/msc[:acline_sfr][xx]
                    elseif (msc[:acline_sto_plus][xx] > 0.0)
                        grd[:zs_acline][:acline_pto][tii][xx] = dtcs*stt[:acline_pto][tii][xx]/msc[:acline_sto][xx]
                        grd[:zs_acline][:acline_qto][tii][xx] = dtcs*stt[:acline_qto][tii][xx]/msc[:acline_sto][xx]
                    end
                end
            end
            =#

            # ====================================================== #
            # Gradients: apparent power flow -- to -> from
            #
            # penalty function derivatives
            #=
            grd[:acline_sfr_plus][:acline_pfr][tii] = stt[:acline_pfr][tii]./acline_sfr
            grd[:acline_sfr_plus][:acline_qfr][tii] = stt[:acline_qfr][tii]./acline_sfr
            grd[:acline_sto_plus][:acline_pto][tii] = stt[:acline_pto][tii]./acline_sto
            grd[:acline_sto_plus][:acline_qto][tii] = stt[:acline_qto][tii]./acline_sto 
            max_sfst0  = [argmax([spfr,spto,0]) for (spfr,spto) in zip(acline_sfr_plus,acline_sto_plus)]
            grd[:zs_acline][:acline_sfr_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sfr_plus][tii][max_sfst0 .== 1] .= dt*cs
            grd[:zs_acline][:acline_sto_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sto_plus][tii][max_sfst0 .== 2] .= dt*cs
            =#
        end
    end
end

# cleanup reserve variables, mostly
function reserve_cleanup_parallel!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # time limit: not needed -- this is an LP
    # integer tolerance: not needed -- this is an LP
    # FeasibilityTol -- qG.FeasibilityTol
    # 
    # this is, necessarily, a centralized (across devices) optimziation problem.
    #
    #
    # build and empty the model!
    model = Model(Gurobi.Optimizer; add_bridges = false)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # set model properties
    quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
    #quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), qG.FeasibilityTol)
    #quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), qG.FeasibilityTol)

    # loop over each time period and define the hard constraints
    quasiGrad.@floop ThreadedEx(basesize = sys.nT ÷ 12) for (t_ind, tii) in enumerate(prm.ts.time_keys)

        # duration
        dt = prm.ts.duration[tii]

        # empty the model!
        empty!(model)

        # affine aggregation terms
        zt = AffExpr(0.0)

        # define the minimum set of variables we will need to solve the constraints
        p_rgu     = @variable(model, [dev = 1:sys.ndev], start=stt[:p_rgu][tii][dev],     lower_bound = 0.0)
        p_rgd     = @variable(model, [dev = 1:sys.ndev], start=stt[:p_rgd][tii][dev],     lower_bound = 0.0)
        p_scr     = @variable(model, [dev = 1:sys.ndev], start=stt[:p_scr][tii][dev],     lower_bound = 0.0)
        p_nsc     = @variable(model, [dev = 1:sys.ndev], start=stt[:p_nsc][tii][dev],     lower_bound = 0.0)
        p_rru_on  = @variable(model, [dev = 1:sys.ndev], start=stt[:p_rru_on][tii][dev],  lower_bound = 0.0)
        p_rru_off = @variable(model, [dev = 1:sys.ndev], start=stt[:p_rru_off][tii][dev], lower_bound = 0.0)
        p_rrd_on  = @variable(model, [dev = 1:sys.ndev], start=stt[:p_rrd_on][tii][dev],  lower_bound = 0.0)
        p_rrd_off = @variable(model, [dev = 1:sys.ndev], start=stt[:p_rrd_off][tii][dev], lower_bound = 0.0)
        q_qru     = @variable(model, [dev = 1:sys.ndev], start=stt[:q_qru][tii][dev],     lower_bound = 0.0)
        q_qrd     = @variable(model, [dev = 1:sys.ndev], start=stt[:q_qrd][tii][dev],     lower_bound = 0.0)

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
            @constraint(model, p_rgu[dev] - prm.dev.p_reg_res_up_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[dev] - prm.dev.p_reg_res_down_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[dev] + p_scr[dev] - prm.dev.p_syn_res_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt[:u_on_dev][tii][dev]) <= 0.0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, stt[:p_on][tii][dev] + p_rgu[dev] + p_scr[dev] + p_rru_on[dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= 0.0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rrd_on[dev] + p_rgd[dev] - stt[:p_on][tii][dev] <= 0.0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_nsc[dev] + p_rru_off[dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, stt[:dev_q][tii][dev] + q_qru[dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= 0.0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= 0.0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt[:dev_q][tii][dev] + q_qru[dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= 0.0)
                end 

                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev] + 
                        prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev] + 
                        q_qrd[dev] - stt[:dev_q][tii][dev] <= 0.0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, stt[:p_on][tii][dev] + p_rgd[dev] + p_rrd_on[dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= 0.0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rru_on[dev] + p_scr[dev] + p_rgu[dev] - stt[:p_on][tii][dev] <= 0.0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_rrd_off[dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, stt[:dev_q][tii][dev] + q_qrd[dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= 0.0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= 0.0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt[:dev_q][tii][dev] + q_qrd[dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= 0.0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev]
                    + prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev]
                    + q_qru[dev] - stt[:dev_q][tii][dev] <= 0.0)
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
                @constraint(model, p_rrd_off[dev] == stt[:p_rrd_off][tii][dev])
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[dev] == stt[:p_nsc][tii][dev])
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[dev] == stt[:p_rru_off][tii][dev])
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[dev] == stt[:q_qru][tii][dev])
            end

            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[dev] == stt[:q_qrd][tii][dev])
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
                @constraint(model, p_rgu_zonal_REQ[zone] == prm.reserve.rgu_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[zone] == prm.reserve.rgd_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[zone] == 0.0)
            else
                @constraint(model, prm.reserve.scr_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[zone])
                @constraint(model, prm.reserve.nsc_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[zone])
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

                @constraint(model, prm.reserve.rru_min[zone][t_ind] <= p_rru_zonal_penalty[zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] <= p_rrd_zonal_penalty[zone])
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

                @constraint(model, prm.reserve.rru_min[zone][t_ind] -
                                sum(p_rru_on[dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] -
                                sum(p_rrd_on[dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[zone])
            end
        end
        
        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_qzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] <= q_qru_zonal_penalty[zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] <= q_qrd_zonal_penalty[zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] -
                                sum(q_qru[dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] -
                                sum(q_qrd[dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[zone])
            end
        end

        # add up
        zt_temp = 
            # local reserve penalties
           -sum(dt.*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*p_rgu) -   # zrgu
            sum(dt.*prm.dev.p_reg_res_down_cost_tmdv[t_ind].*p_rgd) - # zrgd
            sum(dt.*prm.dev.p_syn_res_cost_tmdv[t_ind].*p_scr) -      # zscr
            sum(dt.*prm.dev.p_nsyn_res_cost_tmdv[t_ind].*p_nsc) -     # znsc
            sum(dt.*(prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind].*p_rru_on +
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind].*p_rru_off)) -   # zrru
            sum(dt.*(prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind].*p_rrd_on +
                    prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind].*p_rrd_off)) - # zrrd
            sum(dt.*prm.dev.q_res_up_cost_tmdv[t_ind].*q_qru) -   # zqru      
            sum(dt.*prm.dev.q_res_down_cost_tmdv[t_ind].*q_qrd) - # zqrd
            # zonal reserve penalties (P)
            sum(dt.*prm.vio.rgu_zonal.*p_rgu_zonal_penalty) -
            sum(dt.*prm.vio.rgd_zonal.*p_rgd_zonal_penalty) -
            sum(dt.*prm.vio.scr_zonal.*p_scr_zonal_penalty) -
            sum(dt.*prm.vio.nsc_zonal.*p_nsc_zonal_penalty) -
            sum(dt.*prm.vio.rru_zonal.*p_rru_zonal_penalty) -
            sum(dt.*prm.vio.rrd_zonal.*p_rrd_zonal_penalty) -
            # zonal reserve penalties (Q)
            sum(dt.*prm.vio.qru_zonal.*q_qru_zonal_penalty) -
            sum(dt.*prm.vio.qrd_zonal.*q_qrd_zonal_penalty)

        # update zt
        add_to_expression!(zt, zt_temp)

        # set the objective
        @objective(model, Max, zt)

        # solve
        optimize!(model)
    end
end
=#