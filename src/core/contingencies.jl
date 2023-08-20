function solve_ctgs!(
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
    sys::quasiGrad.System)

    # first, increment, regardless
    qG.ctg_adam_counter += 1

    # should we skip ctg solving?
    if qG.skip_ctg_eval
        # skipping
    else
        # should we solve, based on where we are in the adam solve?
        if (qG.ctg_adam_counter == qG.ctg_solve_frequency) || qG.always_solve_ctg
            # now, reset
            qG.ctg_adam_counter = 0

            # how many ctgs do we consider?
            num_wrst = Int64(ceil(qG.frac_ctg_keep*sys.nctg/2))  # in case n_ctg is odd, and we want to keep all!
            num_rnd  = Int64(floor(qG.frac_ctg_keep*sys.nctg/2)) # in case n_ctg is odd, and we want to keep all!
            num_ctg  = num_wrst + num_rnd

            if qG.score_all_ctgs == true
                ###########################################################
                println("Warning -- scoring all contingencies! No gradients.")
                ###########################################################
            end

            ############################################################
            #  Step 1: Parellelize loop over time to solve base case!  #
            ############################################################
            # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
            Threads.@threads for tii in prm.ts.time_keys
                # get the slack at this time
                @fastmath p_slack = 
                    sum(@inbounds stt.dev_p[tii][pr] for pr in idx.pr_devs) -
                    sum(@inbounds stt.dev_p[tii][cs] for cs in idx.cs_devs) - 
                    sum(stt.sh_p[tii])

                # loop over each bus
                @fastmath @inbounds for bus in 1:sys.nb
                    # active power balance
                    flw.p_inj[tii][bus] = 
                        sum(@inbounds stt.dev_p[tii][pr] for pr in idx.pr[bus]; init=0.0) - 
                        sum(@inbounds stt.dev_p[tii][cs] for cs in idx.cs[bus]; init=0.0) - 
                        sum(@inbounds stt.sh_p[tii][sh] for sh in idx.sh[bus]; init=0.0) - 
                        sum(@inbounds stt.dc_pfr[tii][dc_fr] for dc_fr in idx.bus_is_dc_frs[bus]; init=0.0) - 
                        sum(@inbounds stt.dc_pto[tii][dc_to] for dc_to in idx.bus_is_dc_tos[bus]; init=0.0) - 
                        ntk.alpha*p_slack
                end

                # while we're here, let's reset these (used later)
                flw.dz_dpinj_all[tii]        .= 0.0
                flw.dsmax_dqfr_flow_all[tii] .= 0.0
                flw.dsmax_dqto_flow_all[tii] .= 0.0

                # also, we need to update the flows on all lines! and the phase shift
                flw.ac_qfr[tii][idx.ac_line_flows] .= stt.acline_qfr[tii]
                flw.ac_qfr[tii][idx.ac_xfm_flows]  .= stt.xfm_qfr[tii]
                flw.ac_qto[tii][idx.ac_line_flows] .= stt.acline_qto[tii]
                flw.ac_qto[tii][idx.ac_xfm_flows]  .= stt.xfm_qto[tii]
                flw.ac_phi[tii][idx.ac_phi]        .= stt.phi[tii]

                # compute square flows
                @turbo flw.qfr2[tii] .= quasiGrad.LoopVectorization.pow_fast.(flw.ac_qfr[tii],2)
                @turbo flw.qto2[tii] .= quasiGrad.LoopVectorization.pow_fast.(flw.ac_qto[tii],2)

                # solve for the flows across each ctg
                #   p  =  @view flw.p_inj[2:end]
                @turbo flw.bt[tii] .= .-flw.ac_phi[tii].*ntk.b
                # now, we have flw.p_inj = Yb*theta + E'*bt
                #   c = p - ntk.ErT*bt
                #
                # simplified:
                # => flw.c .= (@view flw.p_inj[2:end]) .- ntk.ErT*flw.bt
                    # this is a little **odd**, but it's fine (the first use of flw.c is just for storage!)
                @turbo quasiGrad.mul!(flw.c[tii], ntk.ErT, flw.bt[tii])
                flw.c[tii] .= (@view flw.p_inj[tii][2:end]) .- flw.c[tii]
                
                # solve the base case with pcg
                if qG.base_solver == "lu"
                    flw.theta[tii] .= ntk.Ybr\flw.c[tii]
                # elseif qG.base_solver == "cholesky" -- error here
                    # flw.theta[tii]  = ntk.Ybr_Ch\c
                elseif qG.base_solver == "pcg"
                    if sys.nb <= qG.min_buses_for_krylov
                        # too few buses -- just use LU
                        flw.theta[tii] .= ntk.Ybr\flw.c[tii]
                    else
                        # solve with a hot start!
                        quasiGrad.cg!(flw.theta[tii], ntk.Ybr, flw.c[tii], statevars = flw.pf_cg_statevars[tii], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its)
                        
                        # if we wanted to test for cg convergence, we would do the following:
                            # => _, ch = quasiGrad.cg!(flw.theta[tii], ntk.Ybr, flw.c[tii], statevars = flw.pf_cg_statevars[tii], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
                            # => # test the krylov solution
                            # => if ~(ch.isconverged)
                            # =>     println("Krylov failed -- using LU backup (ctg flows)!")
                            # =>     flw.theta[tii] = ntk.Ybr\flw.c[tii]
                            # => end
                    end
                else
                    println("base case solve type not recognized :)")
                end

                # get the base-case flow (no phase shifter, yet! not needed here)
                @turbo quasiGrad.mul!(flw.pflow[tii], ntk.Yfr, flw.theta[tii])
            end

            ########################################################
            #  Step 2: Loop over time and parellelize across ctgs  #
            ########################################################
            for tii in prm.ts.time_keys
                # set all ctg scores to 0 (we need this, because not all ctgs are necessarily scored!)
                stt.zctg[tii] .= 0.0

                # compute the shared, common gradient terms (time dependent)
                #
                # gc_avg = grd[:nzms][:zctg_avg] * grd[:zctg_avg][:zctg_avg_t] * grd[:zctg_avg_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
                #        = (-1)                  *                (1)          *       (1/sys.nctg)        *          (-1)       *   dt*prm.vio.s_flow
                #        = dt*prm.vio.s_flow/sys.nctg
                gc_avg   = cgd.ctg_avg[tii] * qG.scale_c_sflow_testing
                
                # gc_min = (-1)                  *                (1)          *            (1)          *          (-1)       *   dt*prm.vio.s_flow
                #        = grd[:nzms][:zctg_min] * grd[:zctg_min][:zctg_min_t] * grd[:zctg_min_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
                gc_min   = cgd.ctg_min[tii] * qG.scale_c_sflow_testing

                # define the ctg
                dt = prm.ts.duration[tii]
                cs = dt*prm.vio.s_flow*qG.scale_c_sflow_testing

                lck = Threads.SpinLock() # => Threads.ReentrantLock(); SpinLock slower, but safer, than ReentrantLock ..?

                # define a vector of bools acciated with multithread storage:
                ctg.ready_to_use .= true

                # Note: - if ctg.ready_to_use[ii] = true, we can use this index to save things!
                #       - if ctg.ready_to_use[ii] = false, then a thread is busy working with this
                #         storage and the associated ID cannot be used. This method if safe against
                #         e.g., thread migration -- if a new thread takes over mid-task.
                #         
                if qG.score_all_ctgs == true
                    ###########################################################
                    # => up above: @info "Warning -- scoring all contingencies! No gradients."
                    ###########################################################
                    # => @floop ThreadedEx(basesize = sys.nctg ÷ qG.num_threads) for ctg_ii in 1:sys.nctg
                    Threads.@threads for ctg_ii in 1:sys.nctg
                        # use a custom "thread ID" -- three indices: tii, ctg_ii, and thrID
                        thrID = Int16(1)
                        Threads.lock(lck)
                            thrIdx = findfirst(ctg.ready_to_use)
                            if thrIdx != Nothing
                                thrID = Int16(thrIdx)
                            end
                            ctg.ready_to_use[thrID] = false # now in use :)
                        Threads.unlock(lck)

                        # apply WMI update! don't use function -- it allocates: ctg.theta_k[thrID] .= wmi_update(flw.theta[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])
                        @turbo ctg.theta_k[thrID] .= flw.theta[tii] .- ntk.u_k[ctg_ii].*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], flw.c[tii]))
                        @turbo quasiGrad.mul!(ctg.pflow_k[thrID], ntk.Yfr, ctg.theta_k[thrID])
                        @turbo ctg.pflow_k[thrID] .+= flw.bt[tii]
                        @turbo ctg.sfr[thrID]     .= quasiGrad.LoopVectorization.sqrt_fast.(flw.qfr2[tii] .+ quasiGrad.LoopVectorization.pow_fast.(ctg.pflow_k[thrID],2))
                        @turbo ctg.sto[thrID]     .= quasiGrad.LoopVectorization.sqrt_fast.(flw.qto2[tii] .+ quasiGrad.LoopVectorization.pow_fast.(ctg.pflow_k[thrID],2))
                        @turbo ctg.sfr_vio[thrID] .= ctg.sfr[thrID] .- ntk.s_max
                        @turbo ctg.sto_vio[thrID] .= ctg.sto[thrID] .- ntk.s_max
                        ctg.sfr_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                        ctg.sto_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0

                        # The following is quite slow, for some reason
                        @inbounds for ac_dev in 1:sys.nac
                            stt.zctg[tii][ctg_ii] -= cs*quasiGrad.LoopVectorization.max_fast(ctg.sfr_vio[thrID][ac_dev], ctg.sto_vio[thrID][ac_dev], 0.0)
                        end
                        # single shot alternative => stt.zctg[tii][ctg_ii] = -cs*sum(max.(ctg.sfr_vio[thrID], ctg.sto_vio[thrID], 0.0))

                        # all done!!
                        Threads.lock(lck)
                            ctg.ready_to_use[thrID] = true
                        Threads.unlock(lck)
                    end
                else
                    # initialize -- redone at each time!
                    @batch per=core for thrID in 1:(qG.num_threads+2) # two extra added, for safety
                        ctg.dz_dpinj_all_threadsum[thrID]    .= 0.0
                        ctg.dsmax_dqfr_flow_threadsum[thrID] .= 0.0
                        ctg.dsmax_dqto_flow_threadsum[thrID] .= 0.0
                    end

                    # sleep tasks
                    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()

                    # loop over contingency subset
                    # => @floop ThreadedEx(basesize = num_ctg ÷ qG.num_threads) for ctg_ii in @view flw.worst_ctg_ids[tii][1:num_ctg] # first is worst!!
                    Threads.@threads for ctg_ii in @view flw.worst_ctg_ids[tii][1:num_ctg] # first is worst!!
                        # use a custom "thread ID"
                        thrID = Int16(1)
                        Threads.lock(lck)
                            thrIdx = findfirst(ctg.ready_to_use)
                            if thrIdx != Nothing
                                thrID = Int16(thrIdx)
                            end
                            ctg.ready_to_use[thrID] = false # now in use :)
                        Threads.unlock(lck)
                        # Here, we must solve theta_k = Ybr_k\c -- assume qG.ctg_solver == "wmi".
                        # now, we need to solve the following: (Yb + v*b*v')x = c
                        #
                        # we already know x0 = Yb\c, so let's use it!
                        #
                        
                        # *************** option one *************** (implicit -- slower)
                        # 1) correct theta (base) via WMI
                        # explicit version => theta_k = flw.theta[tii] - Vector(ntk.u_k[ctg_ii]*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii],c)))
                        # function version (don't use -- allocates!) => wmi_update(flw.theta[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])
                            # => @turbo ctg.theta_k[thrID] .= flw.theta[tii] .- ntk.u_k[ctg_ii].*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], flw.c[tii]))
                        # 2) compute flows -- NOTE: ctg[:pflow_k][tii][ctg_ii] contains outaged line flow -- fixed later --
                            # => @turbo @fastmath quasiGrad.mul!(ctg.pflow_k[thrID], ntk.Yfr, ctg.theta_k[thrID])

                        # *************** option two *************** (WMI-update the flows directly)
                        @turbo ctg.pflow_k[thrID] .= flw.pflow[tii] .- ntk.z_k[ctg_ii].*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], flw.c[tii]))

                        # continue
                        @turbo ctg.pflow_k[thrID] .+= flw.bt[tii]
                        @turbo ctg.sfr[thrID]     .= quasiGrad.LoopVectorization.sqrt_fast.(flw.qfr2[tii] .+ quasiGrad.LoopVectorization.pow_fast.(ctg.pflow_k[thrID],2))
                        @turbo ctg.sto[thrID]     .= quasiGrad.LoopVectorization.sqrt_fast.(flw.qto2[tii] .+ quasiGrad.LoopVectorization.pow_fast.(ctg.pflow_k[thrID],2))
                        @turbo ctg.sfr_vio[thrID] .= ctg.sfr[thrID] .- ntk.s_max
                        @turbo ctg.sto_vio[thrID] .= ctg.sto[thrID] .- ntk.s_max

                        # make sure there are no penalties on lines that are out-aged!
                        ctg.sfr_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0
                        ctg.sto_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0

                        # compute the penalties
                            # => if helpful: smax_vio = max.(sfr_vio, sto_vio, 0.0)
                            # => if helpful: zctg_s = cs*flw[:smax_vio]
                            # => if helpful: stt.zctg[tii][ctg_ii] = -sum(zctg_s, init=0.0)
                        # each contingency, at each time, gets a score:
                        @inbounds for ac_dev in 1:sys.nac
                            stt.zctg[tii][ctg_ii] -= cs*quasiGrad.LoopVectorization.max_fast(ctg.sfr_vio[thrID][ac_dev], ctg.sto_vio[thrID][ac_dev], 0.0)
                        end
                        # single shot alternative => stt.zctg[tii][ctg_ii] = -cs*sum(max.(ctg.sfr_vio[thrID], ctg.sto_vio[thrID], 0.0))

                        # great -- now, do we take the gradient?
                        if qG.eval_grad
                            # only take the gradient if the ctg violation is sufficiently large!!!
                            if stt.zctg[tii][ctg_ii] < qG.ctg_grad_cutoff
                                # game on :)
                                #
                                # first, "was" this (backwards looking -- from last step) the 
                                # worst ctg of the lot? (most negative!) -- this is approximate!
                                if ctg_ii == flw.worst_ctg_ids[tii][1]
                                    gc = copy(gc_avg) + copy(gc_min)
                                else
                                    gc = copy(gc_avg)
                                end

                                # loop over lines and xfms
                                @inbounds @fastmath for ac_dev in 1:sys.nac
                                    if (ctg.sfr_vio[thrID][ac_dev] > qG.grad_ctg_tol) && (ctg.sfr_vio[thrID][ac_dev] > ctg.sto_vio[thrID][ac_dev])
                                        # scaling: if "sfr_vio" or "sto_vio" is even slightly positive, the gradient applies 
                                        #          a really big pressure. Let's soften this with softabs:
                                        grad_scalar                                   = quasiGrad.soft_abs_ctg_grad(ctg.sfr_vio[thrID][ac_dev], qG)
                                        # P is "=", since it assigns (its threadsum come a few lines later!!), and Q is "+=", since it accumulates 
                                        ctg.dsmax_dp_flow[thrID][ac_dev]              = grad_scalar*gc*ctg.pflow_k[thrID][ac_dev]/ctg.sfr[thrID][ac_dev]
                                        ctg.dsmax_dqfr_flow_threadsum[thrID][ac_dev] += grad_scalar*gc*flw.ac_qfr[tii][ac_dev]/ctg.sfr[thrID][ac_dev]
                                    elseif (ctg.sto_vio[thrID][ac_dev] > qG.grad_ctg_tol) && (ctg.sto_vio[thrID][ac_dev] > ctg.sfr_vio[thrID][ac_dev])
                                        grad_scalar                                   = quasiGrad.soft_abs_ctg_grad(ctg.sfr_vio[thrID][ac_dev], qG)
                                        ctg.dsmax_dp_flow[thrID][ac_dev]              = grad_scalar*gc*ctg.pflow_k[thrID][ac_dev]/ctg.sfr[thrID][ac_dev]
                                        ctg.dsmax_dqto_flow_threadsum[thrID][ac_dev] += grad_scalar*gc*flw.ac_qto[tii][ac_dev]/ctg.sto[thrID][ac_dev]
                                    end
                                end

                                # now, the fun gradient: active power injection + xfm phase shift!!
                                # **** => alpha_p_flow_phi = gc*flw.dsmax_dp_flow
                                # **** => rhs = ntk.YfrT*alpha_p_flow_phi
                                #
                                # alpha_p_flow_phi is the derivative of znms with repsect to the
                                # active power flow vector in a given contingency at a given time
                                # 
                                # get the derivative of znms wrt active power injection
                                # NOTE: ntk.Yfr = Ybs*Er, so ntk.Yfr^T = Er^T*Ybs
                                #   -> techincally, we need Yfr_k, where the admittance
                                #      at the outaged line has been drive to 0, but we
                                #      can safely use Yfr, since alpha_p_flow_phi["k"] = 0
                                #      (this was enforced ~50 or so lines above)
                                # NOTE #2 -- this does NOT include the reference bus!
                                #            we skip this gradient :)
                                # => flw.rhs .= ntk.YfrT*(gc.*flw.dsmax_dp_flow)
                                @turbo quasiGrad.mul!(ctg.rhs[thrID], ntk.YfrT, ctg.dsmax_dp_flow[thrID]);
                                # time to solve for dz_dpinj -- two options here:
                                #   1. solve with ntk.Ybr_k, but we didn't actually build this,
                                #      and we didn't build its preconditioner either..
                                #   2. solve with ntk.Ybr, and then use a rank 1 update! Let's do
                                #      this instead :) we'll do this in-loop for each ctg at each time.
                                quasiGrad.solve_and_lowrank_update_single_ctg_gradient!(ctg, ctg_ii, ntk, qG, sys, thrID)
                                
                                # now, we have the gradient of znms wrt all nodal injections/xfm phase shifts!!!
                                # except the slack bus... time to apply these gradients into 
                                # the master grad at all buses except the slack bus.
                                #
                                # update the injection gradient to account for slack!
                                #   alternative direct solution: 
                                #       => ctg[:dz_dpinj][tii][ctg_ii] = (quasiGrad.I-ones(sys.nb-1)*ones(sys.nb-1)'/(sys.nb))*(ntk.Ybr_k[ctg_ii]\(ntk.YfrT*alpha_p_flow))
                                @turbo ctg.dz_dpinj_all_threadsum[thrID] .+= ctg.dz_dpinj[thrID] .- sum(ctg.dz_dpinj[thrID])/sys.nb
                            end
                        end

                        # all done!!
                        Threads.lock(lck)
                            ctg.ready_to_use[thrID] = true
                        Threads.unlock(lck)
                    end

                    # now that we have backpropogated across all ctgs (in this time), we sum!
                    # note -- these were 0'd out in the previous time loop (step 1)
                    if qG.eval_grad
                        @inbounds for thrID in 1:(qG.num_threads+2) # two extra added, for safety
                            @turbo flw.dz_dpinj_all[tii]        .+= ctg.dz_dpinj_all_threadsum[thrID]
                            @turbo flw.dsmax_dqfr_flow_all[tii] .+= ctg.dsmax_dqfr_flow_threadsum[thrID]
                            @turbo flw.dsmax_dqto_flow_all[tii] .+= ctg.dsmax_dqto_flow_threadsum[thrID]
                        end
                    end
                end
                
                # finally, the mixer :) these are the gradients we actually use!!
                @turbo flw.dz_dpinj_rolling[tii]        .= qG.ctg_memory.*flw.dz_dpinj_rolling[tii]        .+ qG.one_min_ctg_memory.*flw.dz_dpinj_all[tii]   
                @turbo flw.dsmax_dqfr_flow_rolling[tii] .= qG.ctg_memory.*flw.dsmax_dqfr_flow_rolling[tii] .+ qG.one_min_ctg_memory.*flw.dsmax_dqfr_flow_all[tii]
                @turbo flw.dsmax_dqto_flow_rolling[tii] .= qG.ctg_memory.*flw.dsmax_dqto_flow_rolling[tii] .+ qG.one_min_ctg_memory.*flw.dsmax_dqto_flow_all[tii]
            end

            ########################################################
            #  Step 3: Loop over parallel time and apply gradients #
            ########################################################
            if qG.eval_grad && (~qG.score_all_ctgs)
                # parallel loop over time
                # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
                Threads.@threads for tii in prm.ts.time_keys
                    # A. active power injections
                    quasiGrad.zctgs_grad_pinj!(flw, grd, idx, mgd, ntk, prm, sys, tii)

                    # B. reactive power injections (fr and to lines)
                    quasiGrad.zctgs_grad_qfr_acline!(flw, grd, idx, mgd, prm, qG, sys, tii)
                    quasiGrad.zctgs_grad_qto_acline!(flw, grd, idx, mgd, prm, qG, sys, tii)

                    # C. reactive power injections (fr and to xfms)
                    quasiGrad.zctgs_grad_qfr_xfm!(flw, grd, idx, mgd, prm, qG, sys, tii)
                    quasiGrad.zctgs_grad_qto_xfm!(flw, grd, idx, mgd, prm, qG, sys, tii)

                    # score -- note: zctg_scored assumes and unchanging number of ctgs 
                    stt.zctg_scored[tii]                         .= @view stt.zctg[tii][flw.worst_ctg_ids[tii][1:num_ctg]]
                    flw.worst_ctg_ids[tii][1:num_wrst]           .= @view flw.worst_ctg_ids[tii][partialsortperm(stt.zctg_scored[tii], 1:num_wrst)]
                    flw.worst_ctg_ids[tii][(num_wrst+1:num_ctg)] .= @view quasiGrad.shuffle!(deleteat!(collect(1:sys.nctg), sort(flw.worst_ctg_ids[tii][1:num_wrst])))[1:num_rnd]
                    # OG scoring and ranking:
                        # =>  # since we have scored all contingencies at this given time,
                        # =>  # rank them from most negative to least (worst is first)
                        # =>  flw.worst_ctg_ids[tii][1:num_ctg] .= sortperm(@view stt.zctg[tii][@view flw.worst_ctg_ids[tii][1:num_ctg]])
                        # =>  # however, only keep half!
                        # =>  flw.worst_ctg_ids[tii][1:num_ctg] .= union(flw.worst_ctg_ids[tii][1:num_wrst],quasiGrad.shuffle(setdiff(1:sys.nctg, flw.worst_ctg_ids[tii][1:num_wrst]))[1:num_rnd])
                end
            end
        else
            # in this case, just apply the previous gradients!
            if qG.eval_grad && (~qG.score_all_ctgs)
                # parallel loop over time
                # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
                Threads.@threads for tii in prm.ts.time_keys
                    # A. active power injections
                    quasiGrad.zctgs_grad_pinj!(flw, grd, idx, mgd, ntk, prm, sys, tii)

                    # B. reactive power injections (fr and to lines)
                    quasiGrad.zctgs_grad_qfr_acline!(flw, grd, idx, mgd, prm, qG, sys, tii)
                    quasiGrad.zctgs_grad_qto_acline!(flw, grd, idx, mgd, prm, qG, sys, tii)

                    # C. reactive power injections (fr and to xfms)
                    quasiGrad.zctgs_grad_qfr_xfm!(flw, grd, idx, mgd, prm, qG, sys, tii)
                    quasiGrad.zctgs_grad_qto_xfm!(flw, grd, idx, mgd, prm, qG, sys, tii)
                end
            end
        end

        # final score -- these might be old or new scores!
        scr[:zctg_min] = sum(minimum(stt.zctg[tii]) for tii in prm.ts.time_keys)
        scr[:zctg_avg] = sum(sum(stt.zctg[tii]) for tii in prm.ts.time_keys)/sys.nctg
    end
end

function solve_and_lowrank_update_single_ctg_gradient!(ctg::quasiGrad.Contingency, ctg_ii::Int64, ntk::quasiGrad.Network, qG::quasiGrad.QG, sys::quasiGrad.System, thrID::Int16)
    # step 1: solve the contingency on the base-case
    # step 2: low rank update the solution
    if qG.base_solver == "lu"
        ctg.dz_dpinj[thrID] .= ntk.Ybr\ctg.rhs[thrID]
        # error with this type !!!
            # => elseif qG.base_solver == "cholesky"
            # =>    ctg.dz_dpinj[thrID] = ntk.Ybr_Ch\rhs
    elseif qG.base_solver == "pcg"
        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- just use LU
            ctg.dz_dpinj[thrID] .= ntk.Ybr\ctg.rhs[thrID]
        else
            # solve with a hot start!
            quasiGrad.cg!(ctg.dz_dpinj[thrID], ntk.Ybr, ctg.rhs[thrID], statevars = ctg.grad_cg_statevars[thrID],  abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its)
            # if we wanted to test for cg convergence, we would do the following:
                # _, ch = quasiGrad.cg!(ctg.dz_dpinj[thrID], ntk.Ybr, ctg.rhs[thrID], statevars = ctg.grad_cg_statevars[thrID],  abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
                # test the krylov solution
                # if ~(ch.isconverged)
                #     # LU backup
                #     println("Krylov failed -- using LU backup (ctg gradient)")
                #     ctg.dz_dpinj[thrID] .= ntk.Ybr\ctg.rhs[thrID]
                # end
        end
    end

    # step 2: now, apply a low-rank update!
        # explicit version => dz_dpinj = ctg.dz_dpinj[thrID] - Vector(ntk.u_k[ctg_ii]*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], rhs)))
        # function version (don't use -- allocates) => wmi_update(ctg.dz_dpinj[thrID], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], ctg.rhs[thrID])
    @turbo ctg.dz_dpinj[thrID] .= ctg.dz_dpinj[thrID] .- ntk.u_k[ctg_ii].*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], ctg.rhs[thrID]))
end

function zctgs_grad_qfr_acline!(flw::quasiGrad.Flow, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8)
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
        # => vmfrqfr = flw.dsmax_dqfr_flow_all[tii][1:sys.nl].*grd.acline_qfr.vmfr[tii]
        # => vmtoqfr = flw.dsmax_dqfr_flow_all[tii][1:sys.nl].*grd.acline_qfr.vmto[tii]
        # => vafrqfr = flw.dsmax_dqfr_flow_all[tii][1:sys.nl].*grd.acline_qfr.vafr[tii]
        # => vatoqfr = flw.dsmax_dqfr_flow_all[tii][1:sys.nl].*grd.acline_qfr.vato[tii]

    # note: we must loop over these assignments! techincally, no need to enumerate
    @turbo for ln in 1:sys.nl # => prm.acline.line_inds
        # update the master grad -- qfr
        mgd.vm[tii][idx.acline_fr_bus[ln]] += flw.dsmax_dqfr_flow_rolling[tii][ln]*grd.acline_qfr.vmfr[tii][ln] # => vmfrqfr[ii]
        mgd.vm[tii][idx.acline_to_bus[ln]] += flw.dsmax_dqfr_flow_rolling[tii][ln]*grd.acline_qfr.vmto[tii][ln] # => vmtoqfr[ii]
        mgd.va[tii][idx.acline_fr_bus[ln]] += flw.dsmax_dqfr_flow_rolling[tii][ln]*grd.acline_qfr.vafr[tii][ln] # => vafrqfr[ii]
        mgd.va[tii][idx.acline_to_bus[ln]] += flw.dsmax_dqfr_flow_rolling[tii][ln]*grd.acline_qfr.vato[tii][ln] # => vatoqfr[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        # => uonqfr = flw.dsmax_dqfr_flow_all[tii][1:sys.nl].*grd.acline_qfr.uon[tii]
        for ln in prm.acline.line_inds
            mgd.u_on_acline[tii][ln] += flw.dsmax_dqfr_flow_rolling[tii][ln]*grd.acline_qfr.uon[tii][ln] # => uonqfr[ii]
        end
    end
end

function zctgs_grad_qto_acline!(flw::quasiGrad.Flow, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8)
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
    # => vmfrqto = flw.dsmax_dqto_flow_all[tii][1:sys.nl].*grd.acline_qto.vmfr[tii]
    # => vmtoqto = flw.dsmax_dqto_flow_all[tii][1:sys.nl].*grd.acline_qto.vmto[tii]
    # => vafrqto = flw.dsmax_dqto_flow_all[tii][1:sys.nl].*grd.acline_qto.vafr[tii]
    # => vatoqto = flw.dsmax_dqto_flow_all[tii][1:sys.nl].*grd.acline_qto.vato[tii]

    # note: we must loop over these assignments!
    @turbo for ln in 1:sys.nl # => prm.acline.line_inds
        # update the master grad -- qto
        mgd.vm[tii][idx.acline_fr_bus[ln]] += flw.dsmax_dqto_flow_rolling[tii][ln]*grd.acline_qto.vmfr[tii][ln] # => vmfrqto[ii]
        mgd.vm[tii][idx.acline_to_bus[ln]] += flw.dsmax_dqto_flow_rolling[tii][ln]*grd.acline_qto.vmto[tii][ln] # => vmtoqto[ii]
        mgd.va[tii][idx.acline_fr_bus[ln]] += flw.dsmax_dqto_flow_rolling[tii][ln]*grd.acline_qto.vafr[tii][ln] # => vafrqto[ii]
        mgd.va[tii][idx.acline_to_bus[ln]] += flw.dsmax_dqto_flow_rolling[tii][ln]*grd.acline_qto.vato[tii][ln] # => vatoqto[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        # => uonqto = flw.dsmax_dqto_flow_all[tii][1:sys.nl].*grd.acline_qto.uon[tii]
        for ln in prm.acline.line_inds
            mgd.u_on_acline[tii][ln] += flw.dsmax_dqto_flow_rolling[tii][ln]*grd.acline_qto.uon[tii][ln] # => uonqto[ii]
        end
    end
end

function zctgs_grad_qfr_xfm!(flw::quasiGrad.Flow, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8)
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
    # => vmfrqfr = flw.dsmax_dqfr_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qfr.vmfr[tii]
    # => vmtoqfr = flw.dsmax_dqfr_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qfr.vmto[tii]
    # => vafrqfr = flw.dsmax_dqfr_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qfr.vafr[tii]
    # => vatoqfr = flw.dsmax_dqfr_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qfr.vato[tii]
    # => tauqfr  = flw.dsmax_dqfr_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qfr.tau[tii]
    # => phiqfr  = flw.dsmax_dqfr_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qfr.phi[tii]

    # note: we must loop over these assignments!
    @turbo for xfm in 1:sys.nx # => prm.xfm.xfm_inds
        # update the master grad -- qfr
        xfm_idx = sys.nl+xfm
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += flw.dsmax_dqfr_flow_rolling[tii][xfm_idx]*grd.xfm_qfr.vmfr[tii][xfm] # => vmfrqfr[ii]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += flw.dsmax_dqfr_flow_rolling[tii][xfm_idx]*grd.xfm_qfr.vmto[tii][xfm] # => vmtoqfr[ii]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += flw.dsmax_dqfr_flow_rolling[tii][xfm_idx]*grd.xfm_qfr.vafr[tii][xfm] # => vafrqfr[ii]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += flw.dsmax_dqfr_flow_rolling[tii][xfm_idx]*grd.xfm_qfr.vato[tii][xfm] # => vatoqfr[ii]
        mgd.tau[tii][xfm]                += flw.dsmax_dqfr_flow_rolling[tii][xfm_idx]*grd.xfm_qfr.tau[tii][xfm]  # => tauqfr[ii]
        mgd.phi[tii][xfm]                += flw.dsmax_dqfr_flow_rolling[tii][xfm_idx]*grd.xfm_qfr.phi[tii][xfm]  # => phiqfr[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        # => uonqfr  = flw.dsmax_dqfr_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qfr.uon[tii]
        for xfm in prm.xfm.xfm_inds
            mgd.u_on_xfm[tii][xfm] += flw.dsmax_dqfr_flow_rolling[tii][sys.nl+xfm]*grd.xfm_qfr.uon[tii][xfm] # => uonqfr[ii]
        end
    end
end

function zctgs_grad_qto_xfm!(flw::quasiGrad.Flow, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System, tii::Int8)
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
    # => vmfrqto = flw.dsmax_dqto_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qto.vmfr[tii]
    # => vmtoqto = flw.dsmax_dqto_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qto.vmto[tii]
    # => vafrqto = flw.dsmax_dqto_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qto.vafr[tii]
    # => vatoqto = flw.dsmax_dqto_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qto.vato[tii]
    # => tauqto  = flw.dsmax_dqto_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qto.tau[tii]
    # => phiqto  = flw.dsmax_dqto_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qto.phi[tii]

    # note: we must loop over these assignments!
    @turbo for xfm in 1:sys.nx # => prm.xfm.xfm_inds
        # update the master grad -- qto
        xfm_idx = sys.nl+xfm
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += flw.dsmax_dqto_flow_rolling[tii][xfm_idx]*grd.xfm_qto.vmfr[tii][xfm] # => vmfrqto[ii]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += flw.dsmax_dqto_flow_rolling[tii][xfm_idx]*grd.xfm_qto.vmto[tii][xfm] # => vmtoqto[ii]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += flw.dsmax_dqto_flow_rolling[tii][xfm_idx]*grd.xfm_qto.vafr[tii][xfm] # => vafrqto[ii]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += flw.dsmax_dqto_flow_rolling[tii][xfm_idx]*grd.xfm_qto.vato[tii][xfm] # => vatoqto[ii]
        mgd.tau[tii][xfm]                += flw.dsmax_dqto_flow_rolling[tii][xfm_idx]*grd.xfm_qto.tau[tii][xfm]  # => tauqto[ii]
        mgd.phi[tii][xfm]                += flw.dsmax_dqto_flow_rolling[tii][xfm_idx]*grd.xfm_qto.phi[tii][xfm]  # => phiqto[ii]
    end

    # NOT efficient
    if qG.update_acline_xfm_bins
        # => uonqto = flw.dsmax_dqto_flow_all[tii][(sys.nl+1):sys.nac].*grd.xfm_qto.uon[tii]
        for xfm in prm.xfm.xfm_inds
            mgd.u_on_xfm[tii][xfm] += flw.dsmax_dqto_flow_rolling[tii][sys.nl+xfm]*grd.xfm_qto.uon[tii][xfm] # => uonqto[ii]
        end
    end
end

function zctgs_grad_pinj!(flw::quasiGrad.Flow, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, ntk::quasiGrad.Network, prm::quasiGrad.Param, sys::quasiGrad.System, tii::Int8)
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
    
    # define alpha locally
    alpha = flw.dz_dpinj_rolling[tii]

    # loop and apply at each bus
    @inbounds for bus in 2:sys.nb
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
            grd.sh_p.g_tv_shunt[tii][sh]*prm.shunt.gs[sh] # => grd[:g_tv_shunt][:u_step_shunt][idx.sh[bus]]
        end

        # dc injections -- "pfr" contributions # grd[:pb_slack][bus][:dc_pfr][tii][idx.bus_is_dc_frs[bus]]
        for dc_fr in idx.bus_is_dc_frs[bus]
            mgd.dc_pfr[tii][dc_fr] += -alpha[bus-1]
        end

        # dc injections -- "pto" contributions
        #
        # note: "dc_pto" does not exist, as a "mgd" variable, so we just leverage
        # that dc_pto = -dc_pfr   ->  d(dc_pto)_d(dc_pfr) = -1 grd[:pb_slack][bus][:dc_pto][tii][idx.bus_is_dc_tos[bus]]
        for dc_to in idx.bus_is_dc_tos[bus]
            mgd.dc_pfr[tii][dc_to] -= -alpha[bus-1]
        end
        
        # phase shift derivatives -- apply gradients
        #   => pinj = (p_pr - p_cs - p_sh - p_fr_dc - p_to_dc - alpha*slack) + Er^T*phi*b
        for (ii,xfm) in enumerate(ntk.xfm_at_bus[bus])
            mgd.phi[tii][xfm] += alpha[bus-1]*ntk.xfm_phi_scalars[bus][ii]
        end
    end
end

# optimally compute the wmi update
function wmi_update(y0::Vector{Float64}, u::Vector{Float64}, g::Float64, x::Vector{Float64})
    # this special function to speedily compute y = y0 - u*g*(u'*x), where u can be sparse-ish,
    # AND, importantly, we don't want to convert from sparse to dense via "Vector"
    #
    # ~~ sadly, this is slow ~~~ depricated for now..
        # y = copy(y0)
        # s = 0.0
        #  loop once for the dot
        # for nzu_idx in quasiGrad.rowvals(u)
        #     s += u[nzu_idx] * x[nzu_idx]
        # end
    # loop again for subtraction
        # gs = g*s
        # for nzu_idx in quasiGrad.rowvals(u)
        #     y[nzu_idx] = y0[nzu_idx] - gs*u[nzu_idx]
        # end

    # output
    return y0 .- u.*(g*quasiGrad.dot(u, x))
end