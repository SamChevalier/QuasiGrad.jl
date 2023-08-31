function penalized_device_constraints!(grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System)
    # loop over each time period
    #
    # Note -- delta penalty (qG.constraint_grad_weight) applied later in the scoring
    #         function, but it is applied to the gradient here!
    # => @batch per=core for dev in prm.dev.dev_keys
    # => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) 
    Threads.@threads for dev in prm.dev.dev_keys
        # for now, we use the "del" thing in the scoring function to penalize all
        # constraint violations -- thus, don't call the "c_hat" constants
        @fastmath @simd for tii in prm.ts.time_keys
            # duration
            dt = prm.ts.duration[tii]
            # in the following constraints, we need to sum over previous constraints
            # in time -- so, say t = 10, and d = 5, then we need to sum over all time
            # state between t < 10 and and t = 5.0
            #
            # so, we need a way to get the variables associated with previous time
            # instances. Thus, we write a function which takes a given time instance
            # and a time interval (e.g., min downtime) and returns the set of relevant
            # time instances: t_set = get_mintimes(tii,interval)

            # careful here -- if we're in, for example, the second "tii", then the
            # start time is 1.

            # 1. Minimum downtime
            T_mndn = idx.Ts_mndn[dev][tii] # => get_tmindn(tii, dev, prm)
            if isempty(T_mndn)
                cvio = max(stt.u_su_dev[tii][dev] + 0.0 - 1.0, 0.0)
            else
                cvio = max(stt.u_su_dev[tii][dev] + sum(@view stt.u_sd_dev_Trx[dev][T_mndn]) - 1.0, 0.0)
            end
            stt.zhat_mndn[tii][dev] = dt*cvio

            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # first, su
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_mndn] * dt * 2.0 * cvio # sign(stt.zhat_mndn[tii][dev])
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG) # sign(stt.zhat_mndn[tii][dev])
                    mgd.u_on_dev[tii][dev] += gc .* grd.u_su_dev.u_on_dev[tii][dev]
                    if tii != 1
                        mgd.u_on_dev[prm.ts.tmin1[tii]][dev] += gc .* grd.u_su_dev.u_on_dev_prev[tii][dev]
                    end

                    # next, sd
                    for tii_inst in T_mndn
                        mgd.u_on_dev[tii_inst][dev] += gc .* grd.u_sd_dev.u_on_dev[tii_inst][dev]
                        if tii_inst != 1
                            mgd.u_on_dev[prm.ts.tmin1[tii_inst]][dev] += gc .* grd.u_sd_dev.u_on_dev_prev[tii_inst][dev]
                        end
                    end
                end
            end

            # 2. Minimum uptime
            T_mnup = idx.Ts_mnup[dev][tii] # => get_tminup(tii, dev, prm)
            if isempty(T_mnup)
                cvio = max(stt.u_sd_dev[tii][dev] + 0.0 - 1.0 , 0.0)
            else
                cvio = max(stt.u_sd_dev[tii][dev] + sum(@view stt.u_su_dev_Trx[dev][T_mnup]) - 1.0 , 0.0)
            end
            stt.zhat_mnup[tii][dev] = dt* cvio

            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # first, sd
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_mnup] * dt * 2.0 * cvio # * sign(stt.zhat_mnup[tii][dev])
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG) # * sign(stt.zhat_mnup[tii][dev])
                    mgd.u_on_dev[tii][dev] += gc .* grd.u_sd_dev.u_on_dev[tii][dev]
                    if tii != 1
                        mgd.u_on_dev[prm.ts.tmin1[tii]][dev] += gc .* grd.u_sd_dev.u_on_dev_prev[tii][dev]
                    end

                    # next, su
                    for tii_inst in T_mnup
                        mgd.u_on_dev[tii_inst][dev] += gc .* grd.u_su_dev.u_on_dev[tii_inst][dev]
                        if tii_inst != 1
                            mgd.u_on_dev[prm.ts.tmin1[tii_inst]][dev] += gc .* grd.u_su_dev.u_on_dev_prev[tii_inst][dev]
                        end
                    end
                end
            end

            # define the previous power value (used by both up and down ramping!)
            if tii == 1
                # note: p0 = prm.dev.init_p[dev]
                dev_p_previous = prm.dev.init_p[dev]
            else
                # grab previous time
                dev_p_previous = stt.dev_p[prm.ts.tmin1[tii]][dev] 
            end

            # 3. Ramping limits (up)
            cvio = max(stt.dev_p[tii][dev] - dev_p_previous
                    - dt*(prm.dev.p_ramp_up_ub[dev]     *(stt.u_on_dev[tii][dev] - stt.u_su_dev[tii][dev])
                    +     prm.dev.p_startup_ramp_ub[dev]*(stt.u_su_dev[tii][dev] + 1.0 - stt.u_on_dev[tii][dev])),0.0)
            stt.zhat_rup[tii][dev] = dt* cvio
            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rup] * dt * 2.0 * cvio # *sign(stt.zhat_rup[tii][dev])
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG) # *sign(stt.zhat_rup[tii][dev])
                    # gradients
                    dp_alpha!(grd, dev, tii, gc) # pjt
                    if tii != 1
                        dp_alpha!(grd, dev, prm.ts.tmin1[tii], -gc) # pjt-1
                    end
                    #dp_alpha!(grd, dev, prm.ts.tmin1[tii], -gc) # pjt-1
                    mgd.u_on_dev[tii][dev] += -gc*dt*prm.dev.p_ramp_up_ub[dev] # uon
                    mgd.u_on_dev[tii][dev] +=  gc*dt*prm.dev.p_ramp_up_ub[dev]*grd.u_su_dev.u_on_dev[tii][dev] # usu
                    mgd.u_on_dev[tii][dev] += -gc*dt*prm.dev.p_startup_ramp_ub[dev]*grd.u_su_dev.u_on_dev[tii][dev] # 2nd usu
                    mgd.u_on_dev[tii][dev] +=  gc*dt*prm.dev.p_startup_ramp_ub[dev] # 2nd uon
                    if tii != 1
                        mgd.u_on_dev[prm.ts.tmin1[tii]][dev] +=  gc*dt*prm.dev.p_ramp_up_ub[dev]*grd.u_su_dev.u_on_dev_prev[tii][dev] # usu
                        mgd.u_on_dev[prm.ts.tmin1[tii]][dev] += -gc*dt*prm.dev.p_startup_ramp_ub[dev]*grd.u_su_dev.u_on_dev_prev[tii][dev] # 2nd usu
                    end
                end
            end

            # 4. Ramping limits (down)
            cvio = max(dev_p_previous - stt.dev_p[tii][dev]
                    - dt*(prm.dev.p_ramp_down_ub[dev]*stt.u_on_dev[tii][dev]
                    +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-stt.u_on_dev[tii][dev])),0.0)
            stt.zhat_rd[tii][dev] = dt* cvio
            
            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rd] * dt * 2.0 * cvio #* sign(stt.zhat_rd[tii][dev])
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG) #* sign(stt.zhat_rd[tii][dev])
                    # gradients
                    dp_alpha!(grd, dev, tii, -gc) # pjt
                    if tii != 1
                        dp_alpha!(grd, dev, prm.ts.tmin1[tii], gc) # pjt-1
                    end
                    mgd.u_on_dev[tii][dev] += -gc*dt*prm.dev.p_ramp_down_ub[dev]     # uon
                    mgd.u_on_dev[tii][dev] +=  gc*dt*prm.dev.p_shutdown_ramp_ub[dev] # 2nd uon
                end
            end

            # 5. Regulation up
            cvio                   = max(stt.p_rgu[tii][dev] - prm.dev.p_reg_res_up_ub[dev]*stt.u_on_dev[tii][dev], 0.0)
            stt.zhat_rgu[tii][dev] = dt* cvio
            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rgu] * dt * 2.0 * cvio # * sign(stt.zhat_rgu[tii][dev])
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG) # * sign(stt.zhat_rgu[tii][dev])
                    mgd.p_rgu[tii][dev]    += gc                                       # prgu
                    mgd.u_on_dev[tii][dev] += -gc*prm.dev.p_reg_res_up_ub[dev]     # uon
                end
            end

            # 6. Regulation down
            cvio                   = max(stt.p_rgd[tii][dev] - prm.dev.p_reg_res_down_ub[dev]*stt.u_on_dev[tii][dev], 0.0)
            stt.zhat_rgd[tii][dev] = dt* cvio

            # evaluate gradient?
            if qG.eval_grad
                if stt.zhat_rgd[tii][dev] > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rgd] * dt * 2.0 * cvio # * sign(stt.zhat_rgd[tii][dev])
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG) # * sign(stt.zhat_rgd[tii][dev])
                    mgd.p_rgd[tii][dev]    += gc                                       # prgu
                    mgd.u_on_dev[tii][dev] += -gc*prm.dev.p_reg_res_down_ub[dev]   # uon
                end
            end

            # 7. Synchronized reserve
            cvio                   = max(stt.p_rgu[tii][dev] + stt.p_scr[tii][dev] - prm.dev.p_syn_res_ub[dev]*stt.u_on_dev[tii][dev], 0.0)
            stt.zhat_scr[tii][dev] = dt* cvio
            
            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_scr] * dt * 2.0 * cvio
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                    mgd.p_rgu[tii][dev] += gc
                    mgd.p_scr[tii][dev] += gc
                    mgd.u_on_dev[tii][dev] += -gc*prm.dev.p_syn_res_ub[dev]
                end
            end

            # 8. Synchronized reserve
            cvio                   = max(stt.p_nsc[tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt.u_on_dev[tii][dev]), 0.0)
            stt.zhat_nsc[tii][dev] = dt* cvio

            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_nsc] * dt * 2.0 * cvio
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                    mgd.p_nsc[tii][dev]    += gc
                    mgd.u_on_dev[tii][dev] += gc*prm.dev.p_nsyn_res_ub[dev]
                end
            end

            # 9. Ramping reserve up (on)
            cvio                     = max(stt.p_rgu[tii][dev] + stt.p_scr[tii][dev] + stt.p_rru_on[tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt.u_on_dev[tii][dev], 0.0)
            stt.zhat_rruon[tii][dev] = dt* cvio

            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rruon] * dt * 2.0 * cvio
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                    mgd.p_rgu[tii][dev]    += gc
                    mgd.p_scr[tii][dev]    += gc
                    mgd.p_rru_on[tii][dev] += gc
                    mgd.u_on_dev[tii][dev] += -gc*prm.dev.p_ramp_res_up_online_ub[dev]
                end
            end

            # 10. Ramping reserve up (off)
            cvio                      = max(stt.p_nsc[tii][dev] + stt.p_rru_off[tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0-stt.u_on_dev[tii][dev]), 0.0)
            stt.zhat_rruoff[tii][dev] = dt* cvio

            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rruoff] * dt * 2.0 * cvio
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                    mgd.p_nsc[tii][dev]     += gc
                    mgd.p_rru_off[tii][dev] += gc
                    mgd.u_on_dev[tii][dev]  += gc*prm.dev.p_ramp_res_up_offline_ub[dev]
                end
            end

            # 11. Ramping reserve down (on)
            cvio                     = max(stt.p_rgd[tii][dev] + stt.p_rrd_on[tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt.u_on_dev[tii][dev], 0.0)
            stt.zhat_rrdon[tii][dev] = dt* cvio

            # evaluate gradient?
            if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rrdon] * dt * 2.0 * cvio
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                    mgd.p_rgd[tii][dev]    += gc
                    mgd.p_rrd_on[tii][dev] += gc
                    mgd.u_on_dev[tii][dev] += -gc*prm.dev.p_ramp_res_down_online_ub[dev]
                end
            end

            # 12. Ramping reserve down (off)
            cvio                      = max(stt.p_rrd_off[tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt.u_on_dev[tii][dev]), 0.0)
            stt.zhat_rrdoff[tii][dev] = dt* cvio

             # evaluate gradient?
             if qG.eval_grad
                if cvio > qG.pg_tol
                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_rrdoff] * dt * 2.0 * cvio
                    gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                    mgd.p_rrd_off[tii][dev] += gc
                    mgd.u_on_dev[tii][dev]  += gc*prm.dev.p_ramp_res_down_offline_ub[dev]
                end
            end

            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers)
                cvio                    = max(stt.p_on[tii][dev] + stt.p_rgu[tii][dev] + stt.p_scr[tii][dev] + stt.p_rru_on[tii][dev] - prm.dev.p_ub[dev][tii]*stt.u_on_dev[tii][dev], 0.0)
                stt.zhat_pmax[tii][dev] = dt* cvio
                
                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_pmax] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        mgd.p_on[tii][dev]     += gc
                        mgd.p_rgu[tii][dev]    += gc
                        mgd.p_scr[tii][dev]    += gc
                        mgd.p_rru_on[tii][dev] += gc
                        mgd.u_on_dev[tii][dev] += -gc*prm.dev.p_ub[dev][tii]
                    end
                end

                # 14p. Minimum reserve limits (producers)
                cvio                      = max(prm.dev.p_lb[dev][tii]*stt.u_on_dev[tii][dev] + stt.p_rrd_on[tii][dev] + stt.p_rgd[tii][dev] - stt.p_on[tii][dev], 0.0)
                stt.zhat_pmin[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_pmin] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        mgd.p_on[tii][dev]     += -gc
                        mgd.p_rgd[tii][dev]    += gc
                        mgd.p_rrd_on[tii][dev] += gc
                        mgd.u_on_dev[tii][dev] += gc*prm.dev.p_lb[dev][tii]
                    end
                end
                
                # 15p. Off reserve limits (producers)
                cvio                         = max(stt.p_su[tii][dev] + stt.p_sd[tii][dev] + stt.p_nsc[tii][dev] + stt.p_rru_off[tii][dev] - prm.dev.p_ub[dev][tii]*(1.0 - stt.u_on_dev[tii][dev]), 0.0)
                stt.zhat_pmaxoff[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_pmaxoff] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        mgd.p_nsc[tii][dev]     += gc
                        mgd.p_rru_off[tii][dev] += gc
                        mgd.u_on_dev[tii][dev]  += gc*prm.dev.p_ub[dev][tii]
                        apply_p_su_grad!(idx, tii, dev, gc, prm, grd, mgd)
                        apply_p_sd_grad!(idx, tii, dev, gc, prm, grd, mgd)
                    end
                end

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][tii] # => get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][tii] # => get_sdpc(tii, dev, prm)
                stt.u_sum[tii][dev] = stt.u_on_dev[tii][dev] + sum(stt.u_su_dev_Trx[dev][tii_inst] for tii_inst in T_supc; init=0.0) + sum(stt.u_sd_dev_Trx[dev][tii_inst] for tii_inst in T_sdpc; init=0.0)

                # 16p. Maximum reactive power reserves (producers)
                cvio                      = max(stt.dev_q[tii][dev] + stt.q_qru[tii][dev] - prm.dev.q_ub[dev][tii]*stt.u_sum[tii][dev], 0.0)
                stt.zhat_qmax[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmax] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        # 1. reactive power derivative
                        dq_alpha!(grd, dev, tii, gc)
                        # 2. qru
                        mgd.q_qru[tii][dev] += gc
                        # 3. u_sum derivative
                        du_sum!(tii, prm, stt, mgd, dev, -gc*prm.dev.q_ub[dev][tii], T_supc, T_sdpc)
                    end
                end

                # 17p. Minimum reactive power reserves (producers)
                cvio                    = max(stt.q_qrd[tii][dev] + prm.dev.q_lb[dev][tii]*stt.u_sum[tii][dev] - stt.dev_q[tii][dev], 0.0)
                stt.zhat_qmin[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmin] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        # 1. qrd
                        mgd.q_qrd[tii][dev] += gc
                        # 2. u_sum derivative
                        du_sum!(tii, prm, stt, mgd, dev, gc*prm.dev.q_lb[dev][tii], T_supc, T_sdpc)
                        # 3. reactive power derivative
                        dq_alpha!(grd, dev, tii, -gc)
                    end
                end

                # 18p. Linked maximum reactive power reserves (producers)
                if dev in idx.J_pqmax
                    cvio                           = max(stt.dev_q[tii][dev] + stt.q_qru[tii][dev] - prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev] - prm.dev.beta_ub[dev]*stt.dev_p[tii][dev], 0.0)
                    stt.zhat_qmax_beta[tii][dev] = dt* cvio

                    # evaluate gradient?
                    if qG.eval_grad
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmax_beta] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        if cvio > qG.pg_tol
                            # 1. reactive power derivative
                            dq_alpha!(grd, dev, tii, gc)
                            # 2. qru
                            mgd.q_qru[tii][dev] += gc
                            # 3. u_sum derivative
                            du_sum!(tii, prm, stt, mgd, dev, -gc*prm.dev.q_0_ub[dev], T_supc, T_sdpc)
                            # 4. active power derivative
                            dp_alpha!(grd, dev, tii, -gc*prm.dev.beta_ub[dev])
                        end
                    end
                end 
                
                # 19p. Linked minimum reactive power reserves (producers)
                if dev in idx.J_pqmin
                    cvio                         = max(prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev] + prm.dev.beta_lb[dev]*stt.dev_p[tii][dev] + stt.q_qrd[tii][dev] - stt.dev_q[tii][dev], 0.0)
                    stt.zhat_qmin_beta[tii][dev] = dt* cvio

                    # evaluate gradient?
                    if qG.eval_grad
                        if cvio > qG.pg_tol
                            # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmin_beta] * dt * 2.0 * cvio
                            gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                            # 1. u_sum derivative
                            du_sum!(tii, prm, stt, mgd, dev, gc*prm.dev.q_0_lb[dev], T_supc, T_sdpc)
                            # 2. active power derivative
                            dp_alpha!(grd, dev, tii, gc*prm.dev.beta_lb[dev])
                            # 3. qrd
                            mgd.q_qrd[tii][dev] += gc
                            # 4. reactive power derivative
                            dq_alpha!(grd, dev, tii, -gc)
                        end
                    end
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers)
                cvio                    = max(stt.p_on[tii][dev] + stt.p_rgd[tii][dev] + stt.p_rrd_on[tii][dev] - prm.dev.p_ub[dev][tii]*stt.u_on_dev[tii][dev], 0.0)
                stt.zhat_pmax[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_pmax] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        mgd.p_on[tii][dev]     += gc
                        mgd.p_rgd[tii][dev]    += gc
                        mgd.p_rrd_on[tii][dev] += gc
                        mgd.u_on_dev[tii][dev] += -gc*prm.dev.p_ub[dev][tii]
                    end
                end

                # 14c. Minimum reserve limits (consumers)
                cvio                    = max(prm.dev.p_lb[dev][tii]*stt.u_on_dev[tii][dev] + stt.p_rru_on[tii][dev] + stt.p_scr[tii][dev] + stt.p_rgu[tii][dev] - stt.p_on[tii][dev], 0.0)
                stt.zhat_pmin[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_pmin] * dt * 2 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        mgd.p_on[tii][dev]     += -gc
                        mgd.p_rgu[tii][dev]    += gc
                        mgd.p_scr[tii][dev]    += gc
                        mgd.p_rru_on[tii][dev] += gc
                        mgd.u_on_dev[tii][dev] += gc*prm.dev.p_lb[dev][tii]
                    end
                end

                # 15c. Off reserve limits (consumers)
                cvio                         = max(stt.p_su[tii][dev] + stt.p_sd[tii][dev] + stt.p_rrd_off[tii][dev] - prm.dev.p_ub[dev][tii]*(1.0 - stt.u_on_dev[tii][dev]), 0.0)
                stt.zhat_pmaxoff[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_pmaxoff] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        mgd.p_rrd_off[tii][dev] += gc
                        mgd.u_on_dev[tii][dev]  += gc*prm.dev.p_ub[dev][tii]
                        apply_p_su_grad!(idx, tii, dev, gc, prm, grd, mgd)
                        apply_p_sd_grad!(idx, tii, dev, gc, prm, grd, mgd)
                    end
                end

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][tii] # => get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][tii] # => get_sdpc(tii, dev, prm)
                stt.u_sum[tii][dev] = stt.u_on_dev[tii][dev] + sum(stt.u_su_dev_Trx[dev][tii_inst] for tii_inst in T_supc; init=0.0) + sum(stt.u_sd_dev_Trx[dev][tii_inst] for tii_inst in T_sdpc; init=0.0)

                # 16c. Maximum reactive power reserves (consumers)
                cvio                      = max(stt.dev_q[tii][dev] + stt.q_qrd[tii][dev] - prm.dev.q_ub[dev][tii]*stt.u_sum[tii][dev], 0.0)
                stt.zhat_qmax[tii][dev] = dt* cvio

                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmax] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        # 1. reactive power derivative
                        dq_alpha!(grd, dev, tii, gc)
                        # 2. qrd
                        mgd.q_qrd[tii][dev] += gc
                        # 3. u_sum derivative
                        du_sum!(tii, prm, stt, mgd, dev, -gc*prm.dev.q_ub[dev][tii], T_supc, T_sdpc)
                    end
                end

                # 17c. Minimum reactive power reserves (consumers)
                cvio                      = max(stt.q_qru[tii][dev] + prm.dev.q_lb[dev][tii]*stt.u_sum[tii][dev] - stt.dev_q[tii][dev], 0.0)
                stt.zhat_qmin[tii][dev] = dt* cvio
                
                # evaluate gradient?
                if qG.eval_grad
                    if cvio > qG.pg_tol
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmin] * dt * 2.0 * cvio
                        gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                        # 1. qru
                        mgd.q_qru[tii][dev] += gc
                        # 2. u_sum derivative
                        du_sum!(tii, prm, stt, mgd, dev, gc*prm.dev.q_lb[dev][tii], T_supc, T_sdpc)
                        # 3. reactive power derivative
                        dq_alpha!(grd, dev, tii, -gc)
                    end
                end

                # 18c. Linked maximum reactive power reserves (consumers)
                if dev in idx.J_pqmax
                    cvio                           = max(stt.dev_q[tii][dev] + stt.q_qrd[tii][dev] - prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev] - prm.dev.beta_ub[dev]*stt.dev_p[tii][dev], 0.0)
                    stt.zhat_qmax_beta[tii][dev] = dt* cvio

                    # evaluate gradient?
                    if qG.eval_grad
                        if cvio > qG.pg_tol
                            # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmax_beta] * dt * 2.0 * cvio
                            gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                            # 1. reactive power derivative
                            dq_alpha!(grd, dev, tii, gc)
                            # 2. qrd
                            mgd.q_qrd[tii][dev] += gc
                            # 3. u_sum derivative
                            du_sum!(tii, prm, stt, mgd, dev, -gc*prm.dev.q_0_ub[dev], T_supc, T_sdpc)
                            # 4. active power derivative
                            dp_alpha!(grd, dev, tii, -gc*prm.dev.beta_ub[dev])
                        end
                    end
                end 

                # 19c. Linked minimum reactive power reserves (consumers)
                if dev in idx.J_pqmin
                    cvio                           = max(prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev] + prm.dev.beta_lb[dev]*stt.dev_p[tii][dev] + stt.q_qru[tii][dev] - stt.dev_q[tii][dev], 0.0)
                    stt.zhat_qmin_beta[tii][dev] = dt* cvio

                    # evaluate gradient?
                    if qG.eval_grad
                        if cvio > qG.pg_tol
                            # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zhat_qmin_beta] * dt * 2.0 * cvio
                            gc = qG.constraint_grad_weight * dt * soft_abs_constraint_grad(cvio, qG)
                            # 1. u_sum derivative
                            du_sum!(tii, prm, stt, mgd, dev, gc*prm.dev.q_0_lb[dev], T_supc, T_sdpc)
                            # 2. active power derivative
                            dp_alpha!(grd, dev, tii, gc*prm.dev.beta_lb[dev])
                            # 3. qru
                            mgd.q_qru[tii][dev] += gc
                            # 4. reactive power derivative
                            dq_alpha!(grd, dev, tii, -gc)
                        end
                    end
                end
            end
        end

        # *** misc penalty: maximum starts over multiple periods ***
            # NOTE => "zhat_mxst" does not need initialization, but it might allocate
        # 
        # now, loop over the startup constraints
        stt.zhat_mxst_scr[dev] = 0.0
        for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
            # get the time periods
            T_su_max            = idx.Ts_su_max[dev][w_ind] # => get_tsumax(w_params, prm)
            zhat_mxst_ii        = max(sum(stt.u_su_dev[tii][dev] for tii in T_su_max; init=0.0) - w_params[3], 0.0)
            stt.zhat_mxst_scr[dev] += zhat_mxst_ii
            # evaluate the gradient? -- make sure mgd has been flushed first!
            if qG.eval_grad
                for tii in T_su_max
                    # OG => mgd.u_on_dev[tii][dev] += grd[:nzms][:zbase] * grd[:zbase][:zhat_mxst] * sign(zhat_mxst_ii) * grd.u_su_dev.u_on_dev[tii][dev]
                    mgd.u_on_dev[tii][dev] += qG.constraint_grad_weight * sign(zhat_mxst_ii) * grd.u_su_dev.u_on_dev[tii][dev]

                    if tii != 1
                        # also update the previous time
                        # OG => mgd.u_on_dev[prm.ts.tmin1[tii]][dev] += grd[:nzms][:zbase] * grd[:zbase][:zhat_mxst] * sign(zhat_mxst_ii) * grd.u_su_dev.u_on_dev_prev[tii][dev]
                        mgd.u_on_dev[prm.ts.tmin1[tii]][dev] += qG.constraint_grad_weight * sign(zhat_mxst_ii) * grd.u_su_dev.u_on_dev_prev[tii][dev]
                    end
                end
            end
        end
    end

    # record the max start score
    scr[:zhat_mxst] = sum(stt.zhat_mxst_scr)
end

function device_reserve_costs!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # compute the costs associated with device reserve offers
    Threads.@threads for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]
        
        # costs
        stt.zrgu[tii] .= dt.*prm.dev.p_reg_res_up_cost_tmdv[tii].*stt.p_rgu[tii]
        stt.zrgd[tii] .= dt.*prm.dev.p_reg_res_down_cost_tmdv[tii].*stt.p_rgd[tii]
        stt.zscr[tii] .= dt.*prm.dev.p_syn_res_cost_tmdv[tii].*stt.p_scr[tii]
        stt.znsc[tii] .= dt.*prm.dev.p_nsyn_res_cost_tmdv[tii].*stt.p_nsc[tii]
        stt.zrru[tii] .= dt.*(prm.dev.p_ramp_res_up_online_cost_tmdv[tii].*stt.p_rru_on[tii] .+
                              prm.dev.p_ramp_res_up_offline_cost_tmdv[tii].*stt.p_rru_off[tii])
        stt.zrrd[tii] .= dt.*(prm.dev.p_ramp_res_down_online_cost_tmdv[tii].*stt.p_rrd_on[tii] .+
                              prm.dev.p_ramp_res_down_offline_cost_tmdv[tii].*stt.p_rrd_off[tii]) 
        stt.zqru[tii] .= dt.*prm.dev.q_res_up_cost_tmdv[tii].*stt.q_qru[tii]      
        stt.zqrd[tii] .= dt.*prm.dev.q_res_down_cost_tmdv[tii].*stt.q_qrd[tii]
    end
end

function energy_costs!(grd::quasiGrad.Grad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # loop over each time period
    for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]

        # devices
        Threads.@threads for dev in prm.dev.dev_keys
        # => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for dev in prm.dev.dev_keys
            cst = prm.dev.cum_cost_blocks[dev][tii][1]  # cost for each block (leading with 0)
            pbk = prm.dev.cum_cost_blocks[dev][tii][2]  # power in each block (leading with 0)
            pcm = prm.dev.cum_cost_blocks[dev][tii][3]  # accumulated power for each block!
            nbk = length(pbk)

            # get the cost directly:
                # => stt.zen_dev[tii][dev] = dt*sum(cst[ii]*max(min(stt.dev_p[tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)
                # => fancy alternative => stt.zen_dev[tii][dev] = dt*sum(cst_i*max(min(stt.dev_p[tii][dev] - pcm_i, pbk_i), 0.0)  for (cst_i,pcm_i,pbk_i) in zip(cst[2:end],pcm[1:end-1],pbk[2:end]); init=0.0)
            # loop and compute costs
            stt.zen_dev[tii][dev] = 0.0
            if stt.dev_p[tii][dev] == 0.0
                # nothing to do: stt.zen_dev[tii][dev] = 0.0
                gradient_block = 2
            else
                for ii in 2:nbk
                    if stt.dev_p[tii][dev] > pcm[ii]
                        stt.zen_dev[tii][dev] += pbk[ii]*cst[ii]
                    else
                        stt.zen_dev[tii][dev] += (stt.dev_p[tii][dev] - pcm[ii-1])*cst[ii]
                        gradient_block = ii
                        break
                    end
                end

                # add dt
                stt.zen_dev[tii][dev] = stt.zen_dev[tii][dev]*dt
            end
            # evaluate the grd? 
            #
            # WARNING -- this will break if stt.dev_p > pcm[end]! It will
            #            mean the device power is out of bounds, and this will
            #            call a price curve which does not exist.
            #                  ~ clipping will fix ~
            if qG.eval_grad
                # what is the index of the "active" block?
                grd.zen_dev.dev_p[tii][dev] = dt*cst[gradient_block]

                # easier to understand:
                    # => del = stt.dev_p[tii][dev] .- pcm
                    # => active_block_ind = argmin(del[del .>= 0.0])
                    # => grd.zen_dev.dev_p[tii][dev] = dt*cst[active_block_ind + 1] # where + 1 is due to the leading 0
                    # faster (than directly above):
                        # if stt.dev_p[tii][dev] == 0.0
                        #     grd.zen_dev.dev_p[tii][dev] = dt*cst[2]
                        # else
                        #     grd.zen_dev.dev_p[tii][dev] = dt*cst[findfirst(stt.dev_p[tii][dev] .< pcm)] # no +1 needed, because we find the upper block
                        # end
            end
        end
    end
end

function energy_penalties!(grd::quasiGrad.Grad, idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System)
    # loop over devices, not time
    # => @batch per=core for dev in prm.dev.dev_keys
    # => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for dev in prm.dev.dev_keys
    Threads.@threads for dev in prm.dev.dev_keys
        Wub = prm.dev.energy_req_ub[dev]
        Wlb = prm.dev.energy_req_lb[dev]
        stt.z_enmax_scr[dev] = 0
        stt.z_enmin_scr[dev] = 0

        # upper bounds
        for (w_ind, w_params) in enumerate(Wub)
            T_en_max = idx.Ts_en_max[dev][w_ind] # = get_tenmax(w_params, prm)
            # => stt[:zw_enmax][dev][w_ind] = prm.vio.e_dev*max(sum(prm.ts.duration[tii]*stt.dev_p[tii][dev] for tii in T_en_max; init=0.0) - w_params[3], 0.0)
            zw_enmax = prm.vio.e_dev*max(sum(prm.ts.duration[tii]*stt.dev_p[tii][dev] for tii in T_en_max; init=0.0) - w_params[3], 0.0)
            stt.z_enmax_scr[dev] -= zw_enmax

            # evaluate the gradient of ep_max? 
            if qG.eval_grad
                for tii in T_en_max
                    dt = prm.ts.duration[tii]
                    # => alpha = grd[:nzms][:zbase] .* grd[:zbase][:z_enmax] * grd[:z_enmax][:zw_enmax] * prm.vio.e_dev * sign(stt[:zw_enmax][dev][w_ind]) * dt
                    # OG => alpha = grd[:nzms][:zbase] .* grd[:zbase][:z_enmax] * grd[:z_enmax][:zw_enmax] * prm.vio.e_dev * sign(zw_enmax) * dt
                    alpha = prm.vio.e_dev * sign(zw_enmax) * dt
                    
                    # update dp_alpha!, which accounts for the partial coefficients
                    dp_alpha!(grd,dev,tii,alpha)
                end
            end
        end

        # lower bounds
        for (w_ind, w_params) in enumerate(Wlb)
            T_en_min = idx.Ts_en_min[dev][w_ind] # get_tenmin(w_params, prm)
            # => stt[:zw_enmin][dev][w_ind] = prm.vio.e_dev*max(w_params[3] - sum(prm.ts.duration[tii]*stt.dev_p[tii][dev] for tii in T_en_min; init=0.0), 0.0)
            zw_enmin = prm.vio.e_dev*max(w_params[3] - sum(prm.ts.duration[tii]*stt.dev_p[tii][dev] for tii in T_en_min; init=0.0), 0.0)
            stt.z_enmin_scr[dev] -= zw_enmin

            # evaluate the gradient of ep_min?
            if qG.eval_grad
                # loop
                for tii in T_en_min
                    dt = prm.ts.duration[tii]
                    # => alpha = grd[:nzms][:zbase] .* grd[:zbase][:z_enmin] * grd[:z_enmin][:zw_enmin] * prm.vio.e_dev * sign(stt[:zw_enmin][dev][w_ind]) * dt
                    # OG => alpha = grd[:nzms][:zbase] .* grd[:zbase][:z_enmin] * grd[:z_enmin][:zw_enmin] * prm.vio.e_dev * sign(zw_enmin) * dt
                    alpha = prm.vio.e_dev * sign(zw_enmin) * dt
                    
                    # update dp_alpha!, which accounts for the partial coefficients
                    dp_alpha!(grd,dev,tii,alpha)
                end
            end
        end
    end

    # get the scores
    scr[:z_enmax] = sum(stt.z_enmax_scr) 
    scr[:z_enmin] = sum(stt.z_enmin_scr)
end

function all_device_statuses_and_costs!(grd::quasiGrad.Grad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # loop over each time period
    Threads.@threads for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]
        
        # start up and shutdown costs
        if tii == 1
            # devices
            stt.u_su_dev[tii] .=   max.(stt.u_on_dev[tii] .- prm.dev.init_on_status, 0.0)
            stt.u_sd_dev[tii] .= .-min.(stt.u_on_dev[tii] .- prm.dev.init_on_status, 0.0)

            if qG.update_acline_xfm_bins
                # aclines
                stt.u_su_acline[tii] .=   max.(stt.u_on_acline[tii] .- prm.acline.init_on_status, 0.0)
                stt.u_sd_acline[tii] .= .-min.(stt.u_on_acline[tii] .- prm.acline.init_on_status, 0.0)
                # xfms
                stt.u_su_xfm[tii] .=   max.(stt.u_on_xfm[tii] .- prm.xfm.init_on_status, 0.0)
                stt.u_sd_xfm[tii] .= .-min.(stt.u_on_xfm[tii] .- prm.xfm.init_on_status, 0.0)
            end

            # evaluate the gradient?
            if qG.eval_grad
                # devices
                grd.u_su_dev.u_on_dev[tii] .=   sign.(stt.u_su_dev[tii])
                grd.u_sd_dev.u_on_dev[tii] .= .-sign.(stt.u_sd_dev[tii])

                if qG.update_acline_xfm_bins
                    # aclines
                    grd.u_su_acline.u_on_acline[tii] .=   sign.(stt.u_su_acline[tii])
                    grd.u_sd_acline.u_on_acline[tii] .= .-sign.(stt.u_sd_acline[tii])
                    # xfms
                    grd.u_su_xfm.u_on_xfm[tii] .=   sign.(stt.u_su_xfm[tii])
                    grd.u_sd_xfm.u_on_xfm[tii] .= .-sign.(stt.u_sd_xfm[tii])
                end
            end
        else
            # devices
            stt.u_su_dev[tii] .=   max.(stt.u_on_dev[tii] .- stt.u_on_dev[prm.ts.tmin1[tii]], 0.0)
            stt.u_sd_dev[tii] .= .-min.(stt.u_on_dev[tii] .- stt.u_on_dev[prm.ts.tmin1[tii]], 0.0)

            if qG.update_acline_xfm_bins
                # aclines
                stt.u_su_acline[tii] .=   max.(stt.u_on_acline[tii] .- stt.u_on_acline[prm.ts.tmin1[tii]], 0.0)
                stt.u_sd_acline[tii] .= .-min.(stt.u_on_acline[tii] .- stt.u_on_acline[prm.ts.tmin1[tii]], 0.0)
                # xfms
                stt.u_su_xfm[tii] .=   max.(stt.u_on_xfm[tii] .- stt.u_on_xfm[prm.ts.tmin1[tii]], 0.0)
                stt.u_sd_xfm[tii] .= .-min.(stt.u_on_xfm[tii] .- stt.u_on_xfm[prm.ts.tmin1[tii]], 0.0)
            end

            # evaluate the gradient?
            if qG.eval_grad
                # current time:
                #
                # devices
                grd.u_su_dev.u_on_dev[tii] .=   sign.(stt.u_su_dev[tii])
                grd.u_sd_dev.u_on_dev[tii] .= .-sign.(stt.u_sd_dev[tii])

                if qG.update_acline_xfm_bins
                    # aclines
                    grd.u_su_acline.u_on_acline[tii] .=   sign.(stt.u_su_acline[tii])
                    grd.u_sd_acline.u_on_acline[tii] .= .-sign.(stt.u_sd_acline[tii])
                    # xfms
                    grd.u_su_xfm.u_on_xfm[tii] .=   sign.(stt.u_su_xfm[tii])
                    grd.u_sd_xfm.u_on_xfm[tii] .= .-sign.(stt.u_sd_xfm[tii])
                end

                # previous time:
                #
                # devices
                grd.u_su_dev.u_on_dev_prev[tii] .= .-sign.(stt.u_su_dev[tii])
                grd.u_sd_dev.u_on_dev_prev[tii] .=   sign.(stt.u_sd_dev[tii])

                if qG.update_acline_xfm_bins
                    # aclines
                    grd.u_su_acline.u_on_acline_prev[tii] .= .-sign.(stt.u_su_acline[tii])
                    grd.u_sd_acline.u_on_acline_prev[tii] .=   sign.(stt.u_sd_acline[tii])
                    # xfms
                    grd.u_su_xfm.u_on_xfm_prev[tii] .= .-sign.(stt.u_su_xfm[tii])
                    grd.u_sd_xfm.u_on_xfm_prev[tii] .=   sign.(stt.u_sd_xfm[tii])
                end
            end
        end

        # get these costs -- devices
        stt.zon_dev[tii] .= dt.*prm.dev.on_cost.*stt.u_on_dev[tii]
        stt.zsu_dev[tii] .= prm.dev.startup_cost.*stt.u_su_dev[tii]
        stt.zsd_dev[tii] .= prm.dev.shutdown_cost.*stt.u_sd_dev[tii]
        if qG.update_acline_xfm_bins
            # aclines
                # stt[:zon_acline][tii] ---> this does not exist
            stt.zsu_acline[tii] .= prm.acline.connection_cost.*stt.u_su_acline[tii]
            stt.zsd_acline[tii] .= prm.acline.disconnection_cost.*stt.u_sd_acline[tii]
            # xfms
                # stt[:zon_xfm][tii] ---> this does not exist
            stt.zsu_xfm[tii] .= prm.xfm.connection_cost.*stt.u_su_xfm[tii]
            stt.zsd_xfm[tii] .= prm.xfm.disconnection_cost.*stt.u_sd_xfm[tii]
        end
    end
end

function simple_device_statuses_and_transposition!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # this function has caused a lot of bugs -- make sure update 
    # and transposition ordering are repsected!
    #
    # loop over each time period
    Threads.@threads for tii in prm.ts.time_keys
        # start up and shutdown costs
        if tii == 1
            # devices
            stt.u_su_dev[tii] .=   max.(stt.u_on_dev[tii] .- prm.dev.init_on_status, 0.0)
            stt.u_sd_dev[tii] .= .-min.(stt.u_on_dev[tii] .- prm.dev.init_on_status, 0.0)
        else
            # devices
            stt.u_su_dev[tii] .=   max.(stt.u_on_dev[tii] .- stt.u_on_dev[prm.ts.tmin1[tii]], 0.0)
            stt.u_sd_dev[tii] .= .-min.(stt.u_on_dev[tii] .- stt.u_on_dev[prm.ts.tmin1[tii]], 0.0)
        end
    end

    # across all devices and all times, compute the transposed on, su, sd and u_sum variables
    Threads.@threads for tii in prm.ts.time_keys # don't multithread here -- not worth it
        @inbounds @simd for dev in prm.dev.dev_keys
            stt.u_on_dev_Trx[dev][tii] = stt.u_on_dev[tii][dev]
            stt.u_su_dev_Trx[dev][tii] = stt.u_su_dev[tii][dev]
            stt.u_sd_dev_Trx[dev][tii] = stt.u_sd_dev[tii][dev]
        end
    end

    # now, compute the u_sums
    Threads.@threads for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        for dev in prm.dev.dev_keys
            # => T_supc = idx.Ts_supc[dev][tii] # => get_supc(tii, dev, prm)
            # => T_sdpc = idx.Ts_sdpc[dev][tii] # => get_sdpc(tii, dev, prm)
            # => stt.u_sum[tii][dev] = stt.u_on_dev[tii][dev] + sum(stt.u_su_dev_Trx[dev][tii_inst] for tii_inst in T_supc; init=0.0) + sum(stt.u_sd_dev_Trx[dev][tii_inst] for tii_inst in T_sdpc; init=0.0)
            stt.u_sum[tii][dev] = stt.u_on_dev[tii][dev] + quasiGrad.u_sum_supc(dev, idx, stt, tii) + quasiGrad.u_sum_sdpc(dev, idx, stt, tii)
        end
    end

    # note -- this must be called AFTER u_sum is populated -- ugly bug!!
    Threads.@threads for tii in prm.ts.time_keys
        @inbounds @simd for dev in prm.dev.dev_keys
            stt.u_sum_Trx[dev][tii] = stt.u_sum[tii][dev]
        end
    end
end

# active power computation
function device_active_powers!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # loop over each time period
    for tii in prm.ts.time_keys
        # the following is expensive, so we skip it during power flow solves
        # (and we don't update p_su/p_sd anyways!)
        if qG.run_susd_updates
            Threads.@threads for dev in prm.dev.dev_keys
            # => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for dev in prm.dev.dev_keys
                # first, get the startup power
                # => T_supc     = idx.Ts_supc[dev][tii]     # => T_supc, p_supc_set   = get_supc(tii, dev, prm)
                # => p_supc_set = idx.ps_supc_set[dev][tii] # => T_supc, p_supc_set   = get_supc(tii, dev, prm)
                if isempty(idx.Ts_supc[dev][tii])
                    stt.p_su[tii][dev] = 0.0
                else
                    stt.p_su[tii][dev] = quasiGrad.dot(idx.ps_supc_set[dev][tii],(@view stt.u_su_dev_Trx[dev][idx.Ts_supc[dev][tii]]))
                end
                # => stt.p_su[tii][dev] = sum(idx.ps_supc_set[dev][tii][ii]*stt.u_su_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(idx.Ts_supc[dev][tii]); init=0.0)

                # second, get the shutdown power
                # => T_sdpc     = idx.Ts_sdpc[dev][tii]     # => get_sdpc(tii, dev, prm)
                # => p_sdpc_set = idx.ps_sdpc_set[dev][tii] # => get_sdpc(tii, dev, prm)
                if isempty(idx.Ts_sdpc[dev][tii])
                    stt.p_sd[tii][dev] = 0.0
                else
                    stt.p_sd[tii][dev] = quasiGrad.dot(idx.ps_sdpc_set[dev][tii],(@view stt.u_sd_dev_Trx[dev][idx.Ts_sdpc[dev][tii]]))
                end
                # => stt.p_sd[tii][dev] = sum(idx.ps_sdpc_set[dev][tii][ii]*stt.u_sd_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(idx.Ts_sdpc[dev][tii]); init=0.0)
            end
        end

        # finally, get the total power balance
        stt.dev_p[tii] .= stt.p_on[tii] .+ stt.p_su[tii] .+ stt.p_sd[tii]
    end
end

# reactive power computation
function device_reactive_powers!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # loop over each time period
    Threads.@threads for tii in prm.ts.time_keys
        # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # => stt.dev_q[tii][idx.J_pqe] .= prm.dev.q_0[idx.J_pqe].*stt.u_sum[tii][idx.J_pqe] .+ prm.dev.beta[idx.J_pqe].*stt.dev_p[tii][idx.J_pqe]
        for dev in idx.J_pqe
            stt.dev_q[tii][dev] = prm.dev.q_0[dev]*stt.u_sum[tii][dev] + prm.dev.beta[dev]*stt.dev_p[tii][dev]
        end
    end
end

function device_startup_states!(grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # before looping over the startup states, flush stt
    Threads.@threads for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        for dev in prm.dev.dev_keys
            stt.zsus_dev_local[tii][dev]  .= 0.0
            stt.u_sus_bnd[tii][dev] .= 0.0
        end
    end
    
    # parallel loop over devices
    Threads.@threads for dev in prm.dev.dev_keys
    # => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for dev in prm.dev.dev_keys
        # loop over each time period
        for tii in prm.ts.time_keys
            # first, we bound ("bnd") the startup state ("sus"):
            # the startup state can only be active if the device
            # has been on within some recent time period.
            #
            # flush the sus (state)
            stt.zsus_dev[tii][dev] = 0.0

            # loop over sus (i.e., f in F)
            if prm.dev.num_sus[dev] > 0
                for ii in 1:prm.dev.num_sus[dev]
                    if prm.dev.startup_states[dev][ii][1] < 0.0 # skip if 0! why are these even here?
                        if tii in idx.Ts_sus_jf[dev][tii][ii]
                            if tii == 1
                                # this is an edge case, where there are no previous states which
                                # could be "on" (since we can't turn on the generator in the fixed
                                # past, and it wasn't on)
                                stt.u_sus_bnd[tii][dev][ii] = 0.0
                            else
                                # grab the largest
                                stt.u_sus_bnd[tii][dev][ii] = quasiGrad.max_binary(dev, idx, ii, stt, tii) 
                                    # => stt.u_sus_bnd[tii][dev][ii] = maximum(stt.u_on_dev_Trx[dev][tij] for tij in idx.Ts_sus_jft[dev][tii][ii])
                            end
                        else
                            # ok, in this case the device was on in a sufficiently recent time (based on
                            # startup conditions), so we don't need to compute a bound
                            stt.u_sus_bnd[tii][dev][ii] = 1.0
                        end

                        # now, compute the discount/cost
                        if stt.u_sus_bnd[tii][dev][ii] > 0.0
                            stt.zsus_dev_local[tii][dev][ii] = prm.dev.startup_states[dev][ii][1]*min(stt.u_su_dev_Trx[dev][tii], stt.u_sus_bnd[tii][dev][ii])
                        end
                    end
                end

                # now, we score, and then take a gradient
                ii = argmin(stt.zsus_dev_local[tii][dev])
                if stt.zsus_dev_local[tii][dev][ii] < 0.0
                    # update the score and take the gradient  ==> this is "+=", since it is over all (f in F) states
                    stt.zsus_dev[tii][dev] += stt.zsus_dev_local[tii][dev][ii]
                    # this is all pretty expensive, so let's take the gradient right here
                    #
                    # evaluate gradient?
                    if qG.eval_grad
                        # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsus_dev] * prm.dev.startup_states[dev][ii][1]
                        # test which was smaller: u_su, or the su_bound?
                        #
                        # we want "<=" so that we never end up in a case where 
                        # we try to take the gradient of u_sus_bnd == 1 (see else case above)
                        if stt.u_su_dev[tii][dev] <= stt.u_sus_bnd[tii][dev][ii] # ** stt[:u_sus_bnd][tii][dev][ii]
                            # in this case, there is an available discount, so we want u_su
                            # to feel a bit less downward pressure and rise up (potentially)
                            mgd.u_on_dev[tii][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev[tii][dev]
                            if tii != 1
                                # previous time?
                                mgd.u_on_dev[prm.ts.tmin1[tii]][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev_prev[tii][dev]
                            end
                        else
                            # in this case, sus bound is lower than u_su, so we'll put some pressure on the
                            # previous largest u_on, trying to push it up, in order to extract a little value
                            # from this sus.. :)
                            #
                            # what time is associated with this derivative? it is the time associated with the max u_on
                            if tii != 1
                                # skip the gradient if tii == 1, since stt[:u_sus_bnd] == 0 and no gradient exists
                                # -- this is a weird edge case, but it does make sense if you think about it for long enough.....
                                u_on_max_ind = argmax(@view stt.u_on_dev_Trx[dev][idx.Ts_sus_jft[dev][tii][ii]])
                                mgd.u_on_dev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev]
                                if idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind] != 1
                                    # previous time?
                                    mgd.u_on_dev[prm.ts.tmin1[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]]][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev_prev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# min downtimes:
    # as an example, assume each TP is 0.25h
    # T1_start = 0          T1_end = 0.25
    # T2_start = 0.25       T2_end = 0.5
    # T3_start = 0.5        T3_end = 0.75..
function get_tmindn(tii::Int8, dev::Int64, prm::quasiGrad.Param)
    # get current start time
    current_start_time = prm.ts.start_time[tii]

    # all other times minus d_min -- note: d_dn_min = prm.dev.down_time_lb[dev]
    valid_times = (current_start_time - prm.dev.down_time_lb[dev] + quasiGrad.eps_time .< prm.ts.start_time) .&& (prm.ts.start_time .< current_start_time)
    t_set       = prm.ts.time_keys[valid_times]

    # output
    return t_set 
end

# min uptimes
function get_tminup(tii::Int8, dev::Int64, prm::quasiGrad.Param)
    # get current start time
    current_start_time = prm.ts.start_time[tii]

    # all other times minus d_min -- note: d_up_min = prm.dev.in_service_time_lb[dev]
    valid_times = (current_start_time - prm.dev.in_service_time_lb[dev] + quasiGrad.eps_time .< prm.ts.start_time) .&& (prm.ts.start_time .< current_start_time)
    t_set       = prm.ts.time_keys[valid_times]

    # output
    return t_set 
end

# get the startup production curve power values
function get_supc(tii::Int8, dev::Int64, prm::quasiGrad.Param)
    # get current end time
    current_end_time = prm.ts.end_time[tii]

    # the following is only defined for t < t', but we will filter those out
    # note: p_rusu = prm.dev.p_startup_ramp_ub[dev]
    p_supc      = prm.dev.p_lb[dev] - prm.dev.p_startup_ramp_ub[dev]*(prm.ts.end_time .- current_end_time)
    valid_times = (p_supc .> 0.0) .&& (current_end_time .< prm.ts.end_time)
    T_set       = prm.ts.time_keys[valid_times]
    p_supc_set  = p_supc[valid_times]

    # output
    return T_set, p_supc_set
end

# get the shutdown production curve power values
function get_sdpc(tii::Int8, dev::Int64, prm::quasiGrad.Param)
    # get current end time
    current_end_time = prm.ts.end_time[tii]

    # deal with edge case -- note: p0 = prm.dev.init_p[dev]
    p_min = [prm.dev.init_p[dev]; prm.dev.p_lb[dev][1:end-1]]

    # compute p_sdpc -- note: p_rdsd = prm.dev.p_shutdown_ramp_ub[dev]
    p_sdpc = p_min - prm.dev.p_shutdown_ramp_ub[dev]*(current_end_time .- prm.ts.start_time)

    # the following is only defined for t' <= t, but we will filter those out
    valid_times = (p_sdpc .> 0.0) .&& (prm.ts.end_time .<= current_end_time)
    T_set       = prm.ts.time_keys[valid_times]
    p_sdpc_set  = p_sdpc[valid_times]

    # output
    return T_set, p_sdpc_set
end

# startup state time periods
function get_tsus_sets(tii::Int8, dev::Int64, prm::quasiGrad.Param, ii::Int64)
    # get current start time
    current_start_time = prm.ts.start_time[tii]

    # sus parameters -- ii is the considered sus (i.e., ii => f in F)
    sus_params = prm.dev.startup_states[dev][ii]
    dn_max     = sus_params[2]

    # compute the sets -- first, T_sus_jft
    valid_times = (current_start_time .- prm.ts.start_time .- quasiGrad.eps_time .- dn_max .<= 0.0) .&& (current_start_time .> prm.ts.start_time)
    T_sus_jft   = prm.ts.time_keys[valid_times]

    # second, T_sus_jf -- this is where we must enforce upper bound on uon
    d_dn0       = prm.dev.init_accu_down_time[dev]
    valid_times = (d_dn0 .+ prm.ts.start_time .- dn_max .- quasiGrad.eps_time .> 0.0)
    T_sus_jf    = prm.ts.time_keys[valid_times]

    # so, what are these?
        # T_sus_jf    -> boundary condition which takes the accumulated downtime
        #                into account -- if we're NOT in this set, then the
        #                device was definitely on in a sufficiently recent time,
        #                and no bounding constraint is needed.
        # T_sus_jft   -> this is the set of times we need to sum over to enure
        #                the device was on recenlty enough to be in a sus.
    
    # output
    return T_sus_jft, T_sus_jf
end

function get_tenmin(w_params::Vector{Float64}, prm::quasiGrad.Param)
    # define start and end
    a_min_start = w_params[1]
    a_min_end   = w_params[2]

    # get the middle of each time period
    a_mid = 0.5*(prm.ts.start_time + prm.ts.end_time)

    # test
    valid_times = (a_min_start + quasiGrad.eps_time .< a_mid) .&& (a_mid .<= a_min_end + quasiGrad.eps_time)
    T_en_min    = prm.ts.time_keys[valid_times]

    # output
    return T_en_min
end

function get_tenmax(w_params::Vector{Float64}, prm::quasiGrad.Param)
    # define start and end
    a_max_start = w_params[1]
    a_max_end   = w_params[2]

    # get the middle of each time period
    a_mid   = 0.5*(prm.ts.start_time + prm.ts.end_time)

    # test
    valid_times = (a_max_start + quasiGrad.eps_time .< a_mid) .&& (a_mid .<= a_max_end + quasiGrad.eps_time)
    T_en_max    = prm.ts.time_keys[valid_times]

    # output
    return T_en_max
end

function get_tsumax(w_params::Vector{Float64}, prm::quasiGrad.Param)
    # define start and end
    a_su_max_start = w_params[1]
    a_su_max_end   = w_params[2]

    # get the starts of each time period
    a_tstart    = prm.ts.start_time
    valid_times = (a_su_max_start .<= a_tstart .+ quasiGrad.eps_time) .&& (a_tstart .+ quasiGrad.eps_time .< a_su_max_end)
    T_su_max    = prm.ts.time_keys[valid_times]
    
    # output
    return T_su_max
end

function apply_p_su_grad!(idx::quasiGrad.Index, tii::Int8, dev::Int64, alpha::Float64, prm::quasiGrad.Param, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad)
    # for a given time and device, call the supc
    T_supc     = idx.Ts_supc[dev][tii]     # => T_supc, p_supc_set = get_supc(tii, dev, prm)
    p_supc_set = idx.ps_supc_set[dev][tii] # => T_supc, p_supc_set = get_supc(tii, dev, prm)

    # loop over the supc and take gradients wrt u_on_dev times some incoming factor alpha
    for (ii,tii_inst) in enumerate(T_supc)
        mgd.u_on_dev[tii_inst][dev] += alpha*p_supc_set[ii]*grd.u_su_dev.u_on_dev[tii_inst][dev]
        # previous time
        if tii_inst != 1
            mgd.u_on_dev[prm.ts.tmin1[tii_inst]][dev] += alpha*p_supc_set[ii]*grd.u_su_dev.u_on_dev_prev[tii_inst][dev]
        end
    end
end

function apply_p_sd_grad!(idx::quasiGrad.Index, tii::Int8, dev::Int64, alpha::Float64, prm::quasiGrad.Param, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad)
    # for a given time and device, call the supc
    T_sdpc     = idx.Ts_sdpc[dev][tii]     # => T_sdpc, p_sdpc_set = get_sdpc(tii, dev, prm)
    p_sdpc_set = idx.ps_sdpc_set[dev][tii] # => T_sdpc, p_sdpc_set = get_sdpc(tii, dev, prm)

    # loop over the supc and take gradients wrt u_on_dev times some incoming factor alpha
    for (ii,tii_inst) in enumerate(T_sdpc)
        mgd.u_on_dev[tii_inst][dev] += alpha*p_sdpc_set[ii]*grd.u_sd_dev.u_on_dev[tii_inst][dev]
        # previous time
        if tii_inst != 1
            mgd.u_on_dev[prm.ts.tmin1[tii_inst]][dev] += alpha*p_sdpc_set[ii]*grd.u_sd_dev.u_on_dev_prev[tii_inst][dev]
        end
    end
end

function max_binary(dev::Int64, idx::quasiGrad.Index, ii::Int64, stt::quasiGrad.State, tii::Int8) 
    # this function is needed because polyester can't deal with internal for loops (of the following form)
    return maximum(stt.u_on_dev_Trx[dev][tij] for tij in idx.Ts_sus_jft[dev][tii][ii])
end

function u_sum_supc(dev::Int64, idx::quasiGrad.Index, stt::quasiGrad.State, tii::Int8)
    return sum(stt.u_su_dev_Trx[dev][tii_inst] for tii_inst in idx.Ts_supc[dev][tii]; init=0.0)
end

function u_sum_sdpc(dev::Int64, idx::quasiGrad.Index, stt::quasiGrad.State, tii::Int8)
    return sum(stt.u_sd_dev_Trx[dev][tii_inst] for tii_inst in idx.Ts_sdpc[dev][tii]; init=0.0)
end
