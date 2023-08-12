function master_grad!(cgd::quasiGrad.ConstantGrad, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # ...follow each z...
    #
    # NOTE: mgd should have been flushed prior to evaluating all gradients,
    #       since some gradient updates directly update mgd :)
    #
    # 1. zctg ==========================================
    #
    # 2. enmin and enmax (in zbase) ====================
    #
    # in this case, we have already computed the
    # necessary derivative terms and placed them in
    # grd.dx.dp -- see the functions energy_penalties!()
    # and dp_alpha!() for more details
    #
    # 3. max starts (in zbase) =========================
    #
    # in this case, we have already computed the
    # necessary derivative terms and placed them in
    # the mgd -- see penalized_device_constraints!()
    #
    # 4. zt (in zbase) =================================
    #
    # note: in many of the folling, we simplify the derivatives
    # by hardcoding the leading terms in the gradient backprop.
    #   => OG is the original!
    #
    if qG.eval_grad # skip the master_grad..?
        ######################### ################### ###########################
        ######################### Parallel Time Loops ###########################
        ######################### ################### ###########################
        # start by looping over the terms which can be safely looped over in time
        # => @batch per=thread for tii in prm.ts.time_keys
        # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        Threads.@threads for tii in prm.ts.time_keys
            # g1 (zen): nzms => zbase => zt => => zen => (dev_p, u_on_dev)
            #
            # all devices
            @inbounds @simd for dev in prm.dev.dev_keys
                # OG=> alpha = grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zen_dev][dev] .* grd.zen_dev.dev_p[tii][dev]
                # => alpha = -cgd.dzt_dzen[dev] .* grd.zen_dev.dev_p[tii][dev]
                dp_alpha!(grd, dev, tii, -cgd.dzt_dzen[dev] * grd.zen_dev.dev_p[tii][dev])
            end

            # g2 (zsu): nzms => zbase => zt => => zsu => u_su_dev => u_on_dev
            #
            # devices -- OG => gc_d = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_dev] * grd[:zsu_dev][:u_su_dev]
            @turbo mgd.u_on_dev[tii] .+= prm.dev.startup_cost .* grd.u_su_dev.u_on_dev[tii]
            
            if qG.update_acline_xfm_bins
                # acline -- OG => gc_l = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_acline] * grd[:zsu_acline][:u_su_acline]
                @turbo mgd.u_on_acline[tii] .+= prm.acline.connection_cost .* grd.u_su_acline.u_on_acline[tii]
                # xfm -- OG => gc_x = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_xfm] * grd[:zsu_xfm][:u_su_xfm]
                @turbo mgd.u_on_xfm[tii] .+= prm.xfm.connection_cost .* grd.u_su_xfm.u_on_xfm[tii]
            end
            # ***NOTE*** -- see secondary time loop for 2nd part of the gradient (previous time)
            
            # g3 (zsd): nzms => zbase => zt => => zsd => u_sd_dev => u_on_dev
            #
            # devices -- OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_dev] * grd[:zsd_dev][:u_sd_dev]
            @turbo mgd.u_on_dev[tii] .+= prm.dev.shutdown_cost .* grd.u_sd_dev.u_on_dev[tii]

            if qG.update_acline_xfm_bins
                # acline -- OG => gc_l = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_acline] * grd[:zsd_acline][:u_sd_acline]
                @turbo mgd.u_on_acline[tii] .+= prm.acline.disconnection_cost .* grd.u_sd_acline.u_on_acline[tii]
                # xfm -- OG => gc_x = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_xfm] * grd[:zsd_xfm][:u_sd_xfm]
                @turbo mgd.u_on_xfm[tii] .+= prm.xfm.disconnection_cost .* grd.u_sd_xfm.u_on_xfm[tii]
            end
            # NOTE -- see secondary time loop for 2nd part of the gradient (previous time)

            # g4 (zon_dev): nzms => zbase => zt => => zon_dev => u_on_dev
            # OG => mgd.u_on_dev[tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zon_dev] * grd[:zon_dev][:u_on_dev][tii]
            @turbo mgd.u_on_dev[tii] .+= cgd.dzon_dev_du_on_dev[tii]

            # g5 (zsus_dev): nzms => zbase => zt => => zsus_dev => u_on_dev ... ?
                # => taken in device_startup_states!() ==========================

            # g6 (zs): nzms => zbase => zt => => zs => (all line and xfm variables)
            quasiGrad.master_grad_zs_acline!(tii, idx, grd, mgd, qG, stt, sys)
            quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, qG, stt, sys)

            # g7 (zrgu):  nzms => zbase => zt => zrgu => p_rgu
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgu] * cgd.dzrgu_dp_rgu[tii] #grd[:zrgu][:p_rgu][tii]
            @turbo mgd.p_rgu[tii] .+= cgd.dzrgu_dp_rgu[tii]

            # g8 (zrgd):  nzms => zbase => zt => => zrgd => p_rgd
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgd] * cgd.dzrgd_dp_rgd[tii] #grd[:zrgd][:p_rgd][tii]
            @turbo mgd.p_rgd[tii] .+= cgd.dzrgd_dp_rgd[tii]

            # g9 (zscr):  nzms => zbase => zt => => zscr => p_scr
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zscr] * cgd.dzscr_dp_scr[tii] #grd[:zscr][:p_scr][tii]
            @turbo mgd.p_scr[tii] .+= cgd.dzscr_dp_scr[tii]
            
            # g10 (znsc): nzms => zbase => zt => => znsc => p_nsc
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:znsc] * cgd.dznsc_dp_nsc[tii] #grd[:znsc][:p_nsc][tii]
            @turbo mgd.p_nsc[tii] .+= cgd.dznsc_dp_nsc[tii]

            # g11 (zrru): nzms => zbase => zt => => zrru => (p_rru_on,p_rru_off)
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru] * cgd.dzrru_dp_rru_on[tii]  #grd[:zrru][:p_rru_on][tii]
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru] * cgd.dzrru_dp_rru_off[tii] #grd[:zrru][:p_rru_off][tii]
            @turbo mgd.p_rru_on[tii]  .+= cgd.dzrru_dp_rru_on[tii]
            @turbo mgd.p_rru_off[tii] .+= cgd.dzrru_dp_rru_off[tii]

            # g12 (zrrd): nzms => zbase => zt => => zrrd => (p_rrd_on,p_rrd_off)
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd] * cgd.dzrrd_dp_rrd_on[tii]  #grd[:zrrd][:p_rrd_on][tii]
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd] * cgd.dzrrd_dp_rrd_off[tii] #grd[:zrrd][:p_rrd_off][tii]
            @turbo mgd.p_rrd_on[tii]  .+= cgd.dzrrd_dp_rrd_on[tii]
            @turbo mgd.p_rrd_off[tii] .+= cgd.dzrrd_dp_rrd_off[tii]

            # g13 (zqru): nzms => zbase => zt => => zqru => q_qru
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqru] * cgd.dzqru_dq_qru[tii] #grd[:zqru][:q_qru][tii]
            @turbo mgd.q_qru[tii] .+= cgd.dzqru_dq_qru[tii]

            # g14 (zqrd): nzms => zbase => zt => => zqrd => q_qrd
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqrd] * cgd.dzqrd_dq_qrd[tii] #grd[:zqrd][:q_qrd][tii]
            @turbo mgd.q_qrd[tii] .+= cgd.dzqrd_dq_qrd[tii]

            # NOTE -- I have lazily left the ac binaries in the following functions -- you can easily remove
            #
            # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
            quasiGrad.master_grad_zp!(tii, prm, idx, grd, mgd, sys)
                    
            # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
            quasiGrad.master_grad_zq!(tii, prm, idx, grd, mgd, sys)

            # reserve zones -- p
            #
            # note: clipping MUST be called before sign() returns a useful result!!!
            @inbounds @simd for zone in 1:sys.nzP
                # g17 (zrgu_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgu_zonal] * cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone] #grd[:zrgu_zonal][:p_rgu_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rgu_zonal_penalty[tii][zone], qG)
                else
                    mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*sign(stt.p_rgu_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_pzone[zone]
                    mgd.p_rgu[tii][dev] -= mgd_com
                end
                # ===> requirements -- depend on active power consumption
                @inbounds @simd for dev in idx.cs_pzone[zone]
                    # => alpha = mgd_com*prm.reserve.rgu_sigma[zone]
                    dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.rgu_sigma[zone])
                end

                # g18 (zrgd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgd_zonal] * cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone] #grd[:zrgd_zonal][:p_rgd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(sign(stt.p_rgd_zonal_penalty[tii][zone]), qG)
                else
                    mgd_com = cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone]*sign(stt.p_rgd_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_pzone[zone]
                    mgd.p_rgd[tii][dev] -= mgd_com
                end
                # ===> requirements -- depend on active power consumption
                @inbounds @simd for dev in idx.cs_pzone[zone]
                    # => alpha = mgd_com*prm.reserve.rgd_sigma[zone]
                    dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.rgd_sigma[zone])
                end

                # g19 (zscr_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zscr_zonal] * cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone] #grd[:zscr_zonal][:p_scr_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone]*soft_abs_reserve_grad(sign(stt.p_scr_zonal_penalty[tii][zone]), qG)
                else
                    mgd_com = cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone]*sign(stt.p_scr_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_pzone[zone]
                    mgd.p_rgu[tii][dev] -= mgd_com
                    mgd.p_scr[tii][dev] -= mgd_com
                end
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(@view stt.dev_p[tii][idx.pr_pzone[zone]])]
                    # ===> requirements -- depend on active power production/consumption!
                    for dev = i_pmax # we only take the derivative of the device which has the highest production
                        # => alpha = mgd_com*prm.reserve.scr_sigma[zone]
                        dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.scr_sigma[zone])
                    end
                end
                if ~isempty(idx.cs_pzone[zone])
                    # only do the following if there are consumers here -- overly cautious
                    @inbounds @simd for dev in idx.cs_pzone[zone]
                        # => alpha = mgd_com*prm.reserve.rgu_sigma[zone]
                        dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.rgu_sigma[zone])
                    end
                end

                # g20 (znsc_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:znsc_zonal] * cgd.dznsc_zonal_dp_nsc_zonal_penalty[tii][zone] #grd[:znsc_zonal][:p_nsc_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dznsc_zonal_dp_nsc_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_nsc_zonal_penalty[tii][zone], qG)
                else
                    mgd_com = cgd.dznsc_zonal_dp_nsc_zonal_penalty[tii][zone]*sign(stt.p_nsc_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_pzone[zone]
                    mgd.p_rgu[tii][dev] -= mgd_com
                    mgd.p_scr[tii][dev] -= mgd_com
                    mgd.p_nsc[tii][dev] -= mgd_com
                end
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(@view stt.dev_p[tii][idx.pr_pzone[zone]])]
                    # ===> requirements -- depend on active power production/consumption!
                    for dev in i_pmax # we only take the derivative of the device which has the highest production
                        # => alpha = mgd_com*prm.reserve.scr_sigma[zone]
                        dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.scr_sigma[zone])
                    end
                    for dev in i_pmax # we only take the derivative of the device which has the highest production
                        # => alpha = mgd_com*prm.reserve.nsc_sigma[zone]
                        dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.nsc_sigma[zone])
                    end
                end
                if ~isempty(idx.cs_pzone[zone])
                    # only do the following if there are consumers here -- overly cautious
                    for dev in idx.cs_pzone[zone]
                        # => alpha = mgd_com*prm.reserve.rgu_sigma[zone]
                        dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.rgu_sigma[zone])
                    end
                end

                # g21 (zrru_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru_zonal] * cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone] #grd[:zrru_zonal][:p_rru_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rru_zonal_penalty[tii][zone], qG)
                else
                    mgd_com = cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*sign(stt.p_rru_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_pzone[zone]
                    mgd.p_rru_on[tii][dev]  -= mgd_com
                    mgd.p_rru_off[tii][dev] -= mgd_com
                end

                # g22 (zrrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd_zonal] * cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone] #grd[:zrrd_zonal][:p_rrd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rrd_zonal_penalty[tii][zone], qG)
                else
                    mgd_com = cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*sign(stt.p_rrd_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_pzone[zone]
                    mgd.p_rrd_on[tii][dev]  -= mgd_com
                    mgd.p_rrd_off[tii][dev] -= mgd_com
                end
            end

            # reserve zones -- q
            @inbounds @simd for zone in 1:sys.nzQ
                # g23 (zqru_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqru_zonal] * cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone] #grd[:zqru_zonal][:q_qru_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.q_qru_zonal_penalty[tii][zone], qG)
                else
                    mgd_com = cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone]*sign(stt.q_qru_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_qzone[zone]
                    mgd.q_qru[tii][dev] -= mgd_com
                end
                # g24 (zqrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqrd_zonal] * cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone] #grd[:zqrd_zonal][:q_qrd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.q_qrd_zonal_penalty[tii][zone], qG)
                else
                    mgd_com = cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone]*sign(stt.q_qrd_zonal_penalty[tii][zone])
                end
                @inbounds @simd for dev in idx.dev_qzone[zone]
                    mgd.q_qrd[tii][dev] -= mgd_com
                end
            end
        end

        #################### ############################# ######################
        #################### Secondary Parallel Time Loops ######################
        #################### ############################# ######################
        #
        # we do this to take the gradients of the su/sd costs! Part of this gradient
        # considers the previous time (t = t_prev), but updating this gradient isn't 
        # "safe" if we're also updating the t = t_now gradient!
        # => @batch per=thread for tii in prm.ts.time_keys
        # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        Threads.@threads for tii in @view prm.ts.time_keys[2:end]
            # include previous time only if tii != 1
            #
            # g2 (zsu)
            @turbo mgd.u_on_dev[prm.ts.tmin1[tii]]        .+= prm.dev.startup_cost       .* grd.u_su_dev.u_on_dev_prev[tii]
            if qG.update_acline_xfm_bins
                @turbo mgd.u_on_acline[prm.ts.tmin1[tii]] .+= prm.acline.connection_cost .* grd.u_su_acline.u_on_acline_prev[tii]
                @turbo mgd.u_on_xfm[prm.ts.tmin1[tii]]    .+= prm.xfm.connection_cost    .* grd.u_su_xfm.u_on_xfm_prev[tii]
            end

            # g3 (zsd) 
            @turbo mgd.u_on_dev[prm.ts.tmin1[tii]]        .+= prm.dev.shutdown_cost         .* grd.u_sd_dev.u_on_dev_prev[tii]
            if qG.update_acline_xfm_bins
                @turbo mgd.u_on_acline[prm.ts.tmin1[tii]] .+= prm.acline.disconnection_cost .* grd.u_sd_acline.u_on_acline_prev[tii]
                @turbo mgd.u_on_xfm[prm.ts.tmin1[tii]]    .+= prm.xfm.disconnection_cost    .* grd.u_sd_xfm.u_on_xfm_prev[tii]
            end
        end

        ######################### ################### ###########################
        ######################### Parallel Dev Loops  ###########################
        ######################### ################### ###########################
        # 
        # compute the final partial derivative contributions -- we need to parallel
        # loop over devices because of the su/sd index sets (among other things..)
        # 
        # => @batch per=thread for dev in prm.dev.dev_keys
        # => @floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for dev in prm.dev.dev_keys
        Threads.@threads for dev in prm.dev.dev_keys
            # => # loop over time -- compute the partial derivative contributions
            # => for tii in prm.ts.time_keys
            # loop over devices -- compute the partial derivative contributions
            @inbounds @simd for tii in prm.ts.time_keys
                quasiGrad.apply_dev_q_grads!(tii, prm, qG, idx, stt, grd, mgd, dev, grd.dx.dq[tii][dev])

                # NOTE -- apply_dev_q_grads!() must be called first! some reactive power
                #         terms also call active power terms, which will add to their derivatives
                quasiGrad.apply_dev_p_grads!(tii, prm, qG, idx, stt, grd, mgd, dev, grd.dx.dp[tii][dev])
            end
        end
    end
end

function master_grad_solve_pf!(cgd::quasiGrad.ConstantGrad, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # this function takes the gradient of zms with respect to the
    # variables which will help to resolve power flow.
    # Notably, even though we solve time-independent power flow
    # problems, we can still use this case exactly, since:
    # dzms_dv => dpftii_dv maps without any issue
    # => turbo :) @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
    Threads.@threads for tii in prm.ts.time_keys
        # g1 (zen): nzms => zbase => zt => => zen => (dev_p, u_on_dev)
        #
        # all devices
        if qG.include_energy_costs_lbfgs == true
            for dev in prm.dev.dev_keys
                # OG=> alpha = grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zen_dev][dev] .* grd.zen_dev.dev_p[tii][dev]
                alpha = -cgd.dzt_dzen[dev] .* grd.zen_dev.dev_p[tii][dev]
                dp_alpha!(grd, dev, tii, alpha)
            end
        end

        # g6 (zs): nzms => zbase => zt => => zs => (all line and xfm variables)
        quasiGrad.master_grad_zs_acline!(tii, idx, grd, mgd, qG, stt, sys)
        quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, qG, stt, sys)

        # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
        quasiGrad.master_grad_zp!(tii, prm, idx, grd, mgd, sys)
        # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
        quasiGrad.master_grad_zq!(tii, prm, idx, grd, mgd, sys)

        # loop over devices
        for dev in prm.dev.dev_keys
            quasiGrad.apply_dev_q_grads!(tii, prm, qG, idx, stt, grd, mgd, dev, grd.dx.dq[tii][dev])

            # NOTE -- apply_dev_q_grads!() must be called first! some reactive power
            #         terms also call active power terms, which will add to their derivatives
            quasiGrad.apply_dev_p_grads!(tii, prm, qG, idx, stt, grd, mgd, dev, grd.dx.dp[tii][dev])
        end
    end
end

function master_grad_adam_pf!(grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, sys::quasiGrad.System)
    # this function takes the gradient of zms with respect to the
    # variables which will help to resolve power flow.
    # Notably, even though we solve time-independent power flow
    # problems, we can still use this case exactly, since:
    # dzms_dv => dpftii_dv maps without any issue
    # => turbo :) @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
    Threads.@threads for tii in prm.ts.time_keys

        # g6 (zs): nzms => zbase => zt => => zs => (all line and xfm variables)
        quasiGrad.master_grad_zs_acline!(tii, idx, grd, mgd, qG, stt, sys)
        quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, qG, stt, sys)

        # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
        quasiGrad.master_grad_zp!(tii, prm, idx, grd, mgd, sys, run_devs = false)

        # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
        quasiGrad.master_grad_zq!(tii, prm, idx, grd, mgd, sys, run_devs = false)
    end
end

function master_grad_zs_acline!(tii::Int8, idx::quasiGrad.Index, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # =========== =========== =========== #
                # zs (acline flows)
    # =========== =========== =========== #
    #
    # common master grad
    # OG => mg_com =  grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zs_acline]
    
    # common flow grads
    pfr_com = grd.zs_acline.acline_pfr[tii]
    qfr_com = grd.zs_acline.acline_qfr[tii]
    pto_com = grd.zs_acline.acline_pto[tii]
    qto_com = grd.zs_acline.acline_qto[tii]

    # final pfr gradients
    # OG => mgpfr   = mg_com.*pfr_com
    # mgpfr * everything below:
    @turbo stt.vmfrpfr[tii] .= pfr_com.*grd.acline_pfr.vmfr[tii]
    @turbo stt.vmtopfr[tii] .= pfr_com.*grd.acline_pfr.vmto[tii]
    @turbo stt.vafrpfr[tii] .= pfr_com.*grd.acline_pfr.vafr[tii]
    @turbo stt.vatopfr[tii] .= pfr_com.*grd.acline_pfr.vato[tii]

    # final qfr gradients
    # OG => mgqfr   = mg_com.*qfr_com
    # mgqfr * everything below:
    @turbo stt.vmfrqfr[tii] .= qfr_com.*grd.acline_qfr.vmfr[tii]
    @turbo stt.vmtoqfr[tii] .= qfr_com.*grd.acline_qfr.vmto[tii]
    @turbo stt.vafrqfr[tii] .= qfr_com.*grd.acline_qfr.vafr[tii]
    @turbo stt.vatoqfr[tii] .= qfr_com.*grd.acline_qfr.vato[tii]

    # final pto gradients
    # OG => mgpto   = mg_com.*pto_com
    # mgpto * everything below:
    @turbo stt.vmfrpto[tii] .= pto_com.*grd.acline_pto.vmfr[tii]
    @turbo stt.vmtopto[tii] .= pto_com.*grd.acline_pto.vmto[tii]
    @turbo stt.vafrpto[tii] .= pto_com.*grd.acline_pto.vafr[tii]
    @turbo stt.vatopto[tii] .= pto_com.*grd.acline_pto.vato[tii]

    # final qfr gradients
    # OG => mgqto   = mg_com.*qto_com
    # mgqto * everything below:
    @turbo stt.vmfrqto[tii] .= qto_com.*grd.acline_qto.vmfr[tii]
    @turbo stt.vmtoqto[tii] .= qto_com.*grd.acline_qto.vmto[tii]
    @turbo stt.vafrqto[tii] .= qto_com.*grd.acline_qto.vafr[tii]
    @turbo stt.vatoqto[tii] .= qto_com.*grd.acline_qto.vato[tii]

    if qG.update_acline_xfm_bins
        @turbo stt.uonpfr[tii] .= pfr_com.*grd.acline_pfr.uon[tii]
        @turbo stt.uonqfr[tii] .= qfr_com.*grd.acline_qfr.uon[tii]
        @turbo stt.uonpto[tii] .= pto_com.*grd.acline_pto.uon[tii]
        @turbo stt.uonqto[tii] .= qto_com.*grd.acline_qto.uon[tii]
    end

    # note: we MUST loop over these assignments! otherwise, += gets confused
    @turbo for ln in 1:sys.nl
        # see binaries at the bottom
        #
        # update the master grad -- pfr
        mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrpfr[tii][ln]
        mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtopfr[tii][ln]
        mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrpfr[tii][ln]
        mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatopfr[tii][ln]

        # update the master grad -- qfr
        mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrqfr[tii][ln]
        mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtoqfr[tii][ln]
        mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrqfr[tii][ln]
        mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatoqfr[tii][ln]

        # update the master grad -- pto
        mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrpto[tii][ln]
        mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtopto[tii][ln]
        mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrpto[tii][ln]
        mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatopto[tii][ln]

        # update the master grad -- qto
        mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrqto[tii][ln]
        mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtoqto[tii][ln]
        mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrqto[tii][ln]
        mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatoqto[tii][ln]
    end

    if qG.update_acline_xfm_bins
        @turbo for ln in 1:sys.nl
            mgd.u_on_acline[tii][ln] += stt.uonpfr[tii][ln]
            mgd.u_on_acline[tii][ln] += stt.uonqfr[tii][ln]
            mgd.u_on_acline[tii][ln] += stt.uonpto[tii][ln]
            mgd.u_on_acline[tii][ln] += stt.uonqto[tii][ln]
        end
    end
end

function master_grad_zs_xfm!(tii::Int8, idx::quasiGrad.Index, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # =========== =========== =========== #
                # zs (xfm)
    # =========== =========== =========== #
    #
    # common master grad
    # OG => mg_com =  grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zs_xfm]
    
    # common flow grads
    pfr_com = grd.zs_xfm.xfm_pfr[tii]
    qfr_com = grd.zs_xfm.xfm_qfr[tii]
    pto_com = grd.zs_xfm.xfm_pto[tii]
    qto_com = grd.zs_xfm.xfm_qto[tii]

    # final pfr gradients
    # OG => mgpfr   = mg_com.*pfr_com
    # ... * everything below:
    @turbo stt.vmfrpfr_x[tii] .= pfr_com.*grd.xfm_pfr.vmfr[tii]
    @turbo stt.vmtopfr_x[tii] .= pfr_com.*grd.xfm_pfr.vmto[tii]
    @turbo stt.vafrpfr_x[tii] .= pfr_com.*grd.xfm_pfr.vafr[tii]
    @turbo stt.vatopfr_x[tii] .= pfr_com.*grd.xfm_pfr.vato[tii]
    @turbo stt.taupfr_x[tii]  .= pfr_com.*grd.xfm_pfr.tau[tii]
    @turbo stt.phipfr_x[tii]  .= pfr_com.*grd.xfm_pfr.phi[tii]

    # final qfr gradients
    # OG => mgqfr   = mg_com.*qfr_com
    # ... * everything below:
    @turbo stt.vmfrqfr_x[tii] .= qfr_com.*grd.xfm_qfr.vmfr[tii]
    @turbo stt.vmtoqfr_x[tii] .= qfr_com.*grd.xfm_qfr.vmto[tii]
    @turbo stt.vafrqfr_x[tii] .= qfr_com.*grd.xfm_qfr.vafr[tii]
    @turbo stt.vatoqfr_x[tii] .= qfr_com.*grd.xfm_qfr.vato[tii]
    @turbo stt.tauqfr_x[tii]  .= qfr_com.*grd.xfm_qfr.tau[tii]
    @turbo stt.phiqfr_x[tii]  .= qfr_com.*grd.xfm_qfr.phi[tii]

    # final pto gradients
    # OG => mgpto   = mg_com.*pto_com
    # ... * everything below:
    @turbo stt.vmfrpto_x[tii] .= pto_com.*grd.xfm_pto.vmfr[tii]
    @turbo stt.vmtopto_x[tii] .= pto_com.*grd.xfm_pto.vmto[tii]
    @turbo stt.vafrpto_x[tii] .= pto_com.*grd.xfm_pto.vafr[tii]
    @turbo stt.vatopto_x[tii] .= pto_com.*grd.xfm_pto.vato[tii]
    @turbo stt.taupto_x[tii]  .= pto_com.*grd.xfm_pto.tau[tii]
    @turbo stt.phipto_x[tii]  .= pto_com.*grd.xfm_pto.phi[tii]

    # final qfr gradients
    # OG => mgqto   = mg_com.*qto_com
    # ... * everything below:
    @turbo stt.vmfrqto_x[tii] .= qto_com.*grd.xfm_qto.vmfr[tii]
    @turbo stt.vmtoqto_x[tii] .= qto_com.*grd.xfm_qto.vmto[tii]
    @turbo stt.vafrqto_x[tii] .= qto_com.*grd.xfm_qto.vafr[tii]
    @turbo stt.vatoqto_x[tii] .= qto_com.*grd.xfm_qto.vato[tii]
    @turbo stt.tauqto_x[tii]  .= qto_com.*grd.xfm_qto.tau[tii]
    @turbo stt.phiqto_x[tii]  .= qto_com.*grd.xfm_qto.phi[tii]

    if qG.update_acline_xfm_bins
        @turbo stt.uonpfr_x[tii] .= pfr_com.*grd.xfm_pfr.uon[tii]
        @turbo stt.uonqfr_x[tii] .= qfr_com.*grd.xfm_qfr.uon[tii]
        @turbo stt.uonpto_x[tii] .= pto_com.*grd.xfm_pto.uon[tii]
        @turbo stt.uonqto_x[tii] .= qto_com.*grd.xfm_qto.uon[tii]
    end

    # note: we must loop over these assignments!
    @turbo for xfm in 1:sys.nx
        # see binaries at the bottom
        #
        # update the master grad -- pfr
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += stt.vmfrpfr_x[tii][xfm]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += stt.vmtopfr_x[tii][xfm]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += stt.vafrpfr_x[tii][xfm]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += stt.vatopfr_x[tii][xfm]
        mgd.tau[tii][xfm]                += stt.taupfr_x[tii][xfm]
        mgd.phi[tii][xfm]                += stt.phipfr_x[tii][xfm]

        # update the master grad -- qfr
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += stt.vmfrqfr_x[tii][xfm]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += stt.vmtoqfr_x[tii][xfm]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += stt.vafrqfr_x[tii][xfm]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += stt.vatoqfr_x[tii][xfm]
        mgd.tau[tii][xfm]                += stt.tauqfr_x[tii][xfm]
        mgd.phi[tii][xfm]                += stt.phiqfr_x[tii][xfm]

        # update the master grad -- pto
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += stt.vmfrpto_x[tii][xfm]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += stt.vmtopto_x[tii][xfm]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += stt.vafrpto_x[tii][xfm]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += stt.vatopto_x[tii][xfm]
        mgd.tau[tii][xfm]                += stt.taupto_x[tii][xfm]
        mgd.phi[tii][xfm]                += stt.phipto_x[tii][xfm]

        # update the master grad -- qto
        mgd.vm[tii][idx.xfm_fr_bus[xfm]] += stt.vmfrqto_x[tii][xfm]
        mgd.vm[tii][idx.xfm_to_bus[xfm]] += stt.vmtoqto_x[tii][xfm]
        mgd.va[tii][idx.xfm_fr_bus[xfm]] += stt.vafrqto_x[tii][xfm]
        mgd.va[tii][idx.xfm_to_bus[xfm]] += stt.vatoqto_x[tii][xfm]
        mgd.tau[tii][xfm]                += stt.tauqto_x[tii][xfm]
        mgd.phi[tii][xfm]                += stt.phiqto_x[tii][xfm]
    end

    if qG.update_acline_xfm_bins
        @turbo for xfm in 1:sys.nx
            mgd.u_on_xfm[tii][xfm] += stt.uonpfr_x[tii][xfm]
            mgd.u_on_xfm[tii][xfm] += stt.uonqfr_x[tii][xfm]
            mgd.u_on_xfm[tii][xfm] += stt.uonpto_x[tii][xfm]
            mgd.u_on_xfm[tii][xfm] += stt.uonqto_x[tii][xfm]
        end
    end
end

function master_grad_zp!(tii::Int8, prm::quasiGrad.Param, idx::quasiGrad.Index, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, sys::quasiGrad.System; run_devs = true)
    # gradient chain: nzms => zbase => zt => zp => (all p injection variables)
    #
    # note: grd[:pb_slack][...] is negelected here, and all terms are trivially
    # hardcoded based on the conservation equations
    
    @inbounds @fastmath @simd for bus in 1:sys.nb
        # common gradient term
        # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zp]
        mgd_com = grd.zp.pb_slack[tii][bus]

        # run devices?
        if run_devs
            # consumer injections
            for dev in idx.cs[bus]
                # => alpha = mgd_com # grd[:pb_slack][:dev_p_cs] = +1
                dp_alpha!(grd, dev, tii, mgd_com)
            end

            # producer injections
            for dev in idx.pr[bus]
                # => alpha = -mgd_com # grd[:pb_slack][:dev_p_pr] = -1
                dp_alpha!(grd, dev, tii, -mgd_com)
            end
        end

        # shunt injections -- bus voltage # grd[:pb_slack][bus][:sh_p][tii][idx.sh[bus]].*
        mgd.vm[tii][bus] += sum(mgd_com*grd.sh_p.vm[tii][sh] for sh in idx.sh[bus]; init=0.0)

        # shunt injections -- shunt steps # grd[:pb_slack][bus][:sh_p][tii][idx.sh[bus]].*
        for sh in idx.sh[bus]
            mgd.u_step_shunt[tii][sh] += mgd_com*
            grd.sh_p.g_tv_shunt[tii][sh]*
            prm.shunt.gs[sh] # => grd[:g_tv_shunt][:u_step_shunt][idx.sh[bus]]
        end
        
        # line injections -- "from" flows
        #
        # loop over each line, and compute the gradient of flows
        for line in idx.bus_is_acline_frs[bus]
            bus_fr = idx.acline_fr_bus[line]
            bus_to = idx.acline_to_bus[line]

            # make sure :)
            @assert bus_fr == bus

            # gradients
            # => vmfrpfr = grd.acline_pfr.vmfr[tii][line]
            # => vafrpfr = grd.acline_pfr.vafr[tii][line]
            # => uonpfr  = grd.acline_pfr.uon[tii][line]

            # "pfr" is also a function of the to bus voltages, so we need these
            # => vmtopfr = grd.acline_pfr.vmto[tii][line]
            # => vatopfr = grd.acline_pfr.vato[tii][line]

            # update the master grad -- pfr, at this bus and its corresponding "to" bus
            mgd.vm[tii][bus_fr]        += mgd_com*grd.acline_pfr.vmfr[tii][line]
            mgd.va[tii][bus_fr]        += mgd_com*grd.acline_pfr.vafr[tii][line]
            mgd.vm[tii][bus_to]        += mgd_com*grd.acline_pfr.vmto[tii][line]
            mgd.va[tii][bus_to]        += mgd_com*grd.acline_pfr.vato[tii][line]
            mgd.u_on_acline[tii][line] += mgd_com*grd.acline_pfr.uon[tii][line]
        end
        # line injections -- "to" flows
        #
        # loop over each line, and compute the gradient of flows
        for line in idx.bus_is_acline_tos[bus]
            bus_to = idx.acline_to_bus[line]
            bus_fr = idx.acline_fr_bus[line]

            # make sure :)
            @assert bus_to == bus

            # gradients
            # => vmtopto = grd.acline_pto.vmto[tii][line]
            # => vatopto = grd.acline_pto.vato[tii][line]
            # => uonpto  = grd.acline_pto.uon[tii][line]

            # "pto" is also a function of the fr bus voltages, so we need these
            # => vmfrpto = grd.acline_pto.vmfr[tii][line]
            # => vafrpto = grd.acline_pto.vafr[tii][line]

            # update the master grad -- pto, at this bus and its corresponding "fr" bus
            mgd.vm[tii][bus_to]        += mgd_com*grd.acline_pto.vmto[tii][line]
            mgd.va[tii][bus_to]        += mgd_com*grd.acline_pto.vato[tii][line]
            mgd.vm[tii][bus_fr]        += mgd_com*grd.acline_pto.vmfr[tii][line]
            mgd.va[tii][bus_fr]        += mgd_com*grd.acline_pto.vafr[tii][line]
            mgd.u_on_acline[tii][line] += mgd_com*grd.acline_pto.uon[tii][line]
        end
        
        # xfm injections -- "from" flows
        #
        # loop over each xfm, and compute the gradient of flows
        for xfm in idx.bus_is_xfm_frs[bus]
            bus_fr = idx.xfm_fr_bus[xfm]
            bus_to = idx.xfm_to_bus[xfm]

            # make sure :)
            @assert bus_fr == bus

            # gradients
            # => vmfrpfr = grd.xfm_pfr.vmfr[tii][xfm]
            # => vafrpfr = grd.xfm_pfr.vafr[tii][xfm]
            # => uonpfr  = grd.xfm_pfr.uon[tii][xfm]

            # "pfr" is also a function of the to bus voltages, so we need these
            # => vmtopfr = grd.xfm_pfr.vmto[tii][xfm]
            # => vatopfr = grd.xfm_pfr.vato[tii][xfm]

            # xfm ratios
            # => taupfr  = grd.xfm_pfr.tau[tii][xfm]
            # => phipfr  = grd.xfm_pfr.phi[tii][xfm]

            # update the master grad -- pfr, at this bus and its corresponding "to" bus
            mgd.vm[tii][bus_fr]    += mgd_com*grd.xfm_pfr.vmfr[tii][xfm]
            mgd.va[tii][bus_fr]    += mgd_com*grd.xfm_pfr.vafr[tii][xfm]
            mgd.vm[tii][bus_to]    += mgd_com*grd.xfm_pfr.vmto[tii][xfm]
            mgd.va[tii][bus_to]    += mgd_com*grd.xfm_pfr.vato[tii][xfm]
            mgd.tau[tii][xfm]      += mgd_com*grd.xfm_pfr.tau[tii][xfm]
            mgd.phi[tii][xfm]      += mgd_com*grd.xfm_pfr.phi[tii][xfm]
            mgd.u_on_xfm[tii][xfm] += mgd_com*grd.xfm_pfr.uon[tii][xfm]
        end
        # xfm injections -- "to" flows
        #
        # loop over each xfm, and compute the gradient of flows
        for xfm in idx.bus_is_xfm_tos[bus]
            bus_to = idx.xfm_to_bus[xfm]
            bus_fr = idx.xfm_fr_bus[xfm]

            # make sure :)
            @assert bus_to == bus

            # gradients
            # => vmtopto = grd.xfm_pto.vmto[tii][xfm]
            # => vatopto = grd.xfm_pto.vato[tii][xfm]
            # => uonpto  = grd.xfm_pto.uon[tii][xfm]

            # "pto" is also a function of the fr bus voltages, so we need these
            # => vmfrpto = grd.xfm_pto.vmfr[tii][xfm]
            # => vafrpto = grd.xfm_pto.vafr[tii][xfm]

            # xfm ratios
            # => taupto  = grd.xfm_pto.tau[tii][xfm]
            # => phipto  = grd.xfm_pto.phi[tii][xfm]

            # update the master grad -- pto, at this bus and its corresponding "fr" bus
            mgd.vm[tii][bus_to]    += mgd_com*grd.xfm_pto.vmto[tii][xfm]
            mgd.va[tii][bus_to]    += mgd_com*grd.xfm_pto.vato[tii][xfm]
            mgd.vm[tii][bus_fr]    += mgd_com*grd.xfm_pto.vmfr[tii][xfm]
            mgd.va[tii][bus_fr]    += mgd_com*grd.xfm_pto.vafr[tii][xfm]
            mgd.tau[tii][xfm]      += mgd_com*grd.xfm_pto.tau[tii][xfm]
            mgd.phi[tii][xfm]      += mgd_com*grd.xfm_pto.phi[tii][xfm]
            mgd.u_on_xfm[tii][xfm] += mgd_com*grd.xfm_pto.uon[tii][xfm]
        end

        # skip dc lines if there are none
        for dc_fr in idx.bus_is_dc_frs[bus]
            # dc injections -- "pfr" contributions # grd[:pb_slack][bus][:dc_pfr][tii][idx.bus_is_dc_frs[bus]]
            mgd.dc_pfr[tii][dc_fr] += mgd_com
        end

        for dc_to in idx.bus_is_dc_tos[bus]
            # dc injections -- "pto" contributions
            #
            # note: "dc_pto" does not exist, as a "mgd" variable, so
            # we just leverage that dc_pto = -dc_pfr   ->  d(dc_pto)_d(dc_pfr) = -1 grd[:pb_slack][bus][:dc_pto][tii][idx.bus_is_dc_tos[bus]]
            mgd.dc_pfr[tii][dc_to] += -mgd_com
        end
    end
end

function master_grad_zq!(tii::Int8, prm::quasiGrad.Param, idx::quasiGrad.Index, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, sys::quasiGrad.System; run_devs = true)
    # gradient chain: nzms => zbase => zt => zq => (all q injection variables)
    #
    # note: grd[:qb_slack][...] is negelected here, and all terms are trivially
    # hardcoded based on the conservation equations
    @inbounds @fastmath @simd for bus in 1:sys.nb
        # common gradient term
        # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zq] * grd.zq.qb_slack[tii][bus]
        mgd_com = grd.zq.qb_slack[tii][bus]

        # device injections -- there are two types of device injections:
        #   1) those for which "dev_q" is a free variable (easy)! 
        #   2) those for which "dev_q" is constrained by equality (hard..)
        #
        # regardless, we record "alpha" for both and deal with the partials later.
        #    
        # run devices?
        if run_devs
            # consumers  
            for dev in idx.cs[bus]
                # => alpha = mgd_com
                dq_alpha!(grd, dev, tii, mgd_com)
            end

            # producers
            for dev in idx.pr[bus]
                # => alpha = -mgd_com
                dq_alpha!(grd, dev, tii, -mgd_com)
            end
        end

        # shunt injections -- bus voltage # grd[:qb_slack][bus][:sh_q][tii][idx.sh[bus]]
        mgd.vm[tii][bus] += sum(mgd_com*grd.sh_q.vm[tii][sh] for sh in idx.sh[bus]; init=0.0)

        # shunt injections -- shunt steps # grd[:qb_slack][bus][:sh_q][tii][idx.sh[bus]].*
        for sh in idx.sh[bus]
            mgd.u_step_shunt[tii][sh] += mgd_com*
            grd.sh_q.b_tv_shunt[tii][sh].*
            prm.shunt.bs[sh] # => grd[:b_tv_shunt][:u_step_shunt][idx.sh[bus]]
        end

        # line injections -- "from" flows
        #
        # loop over each line, and compute the gradient of flows
        for line in idx.bus_is_acline_frs[bus]
            bus_fr = idx.acline_fr_bus[line]
            bus_to = idx.acline_to_bus[line]

            # make sure :)
            @assert bus_fr == bus

            # update the master grad -- qfr, at this bus and its corresponding "to" bus (below)
            mgd.vm[tii][bus_fr]        += mgd_com*grd.acline_qfr.vmfr[tii][line]
            mgd.va[tii][bus_fr]        += mgd_com*grd.acline_qfr.vafr[tii][line]
            mgd.vm[tii][bus_to]        += mgd_com*grd.acline_qfr.vmto[tii][line]
            mgd.va[tii][bus_to]        += mgd_com*grd.acline_qfr.vato[tii][line]
            mgd.u_on_acline[tii][line] += mgd_com*grd.acline_qfr.uon[tii][line]
        end
        # line injections -- "to" flows
        #
        # loop over each line, and compute the gradient of flows
        for line in idx.bus_is_acline_tos[bus]
            bus_to = idx.acline_to_bus[line]
            bus_fr = idx.acline_fr_bus[line]

            # make sure :)
            @assert bus_to == bus

            # update the master grad -- qto, at this bus and its corresponding "fr" bus (above)
            mgd.vm[tii][bus_to]        += mgd_com*grd.acline_qto.vmto[tii][line]
            mgd.va[tii][bus_to]        += mgd_com*grd.acline_qto.vato[tii][line]
            mgd.vm[tii][bus_fr]        += mgd_com*grd.acline_qto.vmfr[tii][line]
            mgd.va[tii][bus_fr]        += mgd_com*grd.acline_qto.vafr[tii][line]
            mgd.u_on_acline[tii][line] += mgd_com*grd.acline_qto.uon[tii][line]
        end
        
        # xfm injections -- "from" flows
        #
        # loop over each xfm, and compute the gradient of flows
        for xfm in idx.bus_is_xfm_frs[bus]
            bus_fr = idx.xfm_fr_bus[xfm]
            bus_to = idx.xfm_to_bus[xfm]

            # make sure :)
            @assert bus_fr == bus

            # update the master grad -- qfr, at this bus and its corresponding "to" bus
            mgd.vm[tii][bus_fr]    += mgd_com*grd.xfm_qfr.vmfr[tii][xfm]
            mgd.va[tii][bus_fr]    += mgd_com*grd.xfm_qfr.vafr[tii][xfm]
            mgd.vm[tii][bus_to]    += mgd_com*grd.xfm_qfr.vmto[tii][xfm]
            mgd.va[tii][bus_to]    += mgd_com*grd.xfm_qfr.vato[tii][xfm]
            mgd.tau[tii][xfm]      += mgd_com*grd.xfm_qfr.tau[tii][xfm]
            mgd.phi[tii][xfm]      += mgd_com*grd.xfm_qfr.phi[tii][xfm]
            mgd.u_on_xfm[tii][xfm] += mgd_com*grd.xfm_qfr.uon[tii][xfm]
        end
        # xfm injections -- "to" flows
        #
        # loop over each xfm, and compute the gradient of flows
        for xfm in idx.bus_is_xfm_tos[bus]
            bus_to = idx.xfm_to_bus[xfm]
            bus_fr = idx.xfm_fr_bus[xfm]

            # make sure :)
            @assert bus_to == bus

            # update the master grad -- qto, at this bus and its corresponding "fr" bus
            mgd.vm[tii][bus_to]    += mgd_com*grd.xfm_qto.vmto[tii][xfm]
            mgd.va[tii][bus_to]    += mgd_com*grd.xfm_qto.vato[tii][xfm]
            mgd.vm[tii][bus_fr]    += mgd_com*grd.xfm_qto.vmfr[tii][xfm]
            mgd.va[tii][bus_fr]    += mgd_com*grd.xfm_qto.vafr[tii][xfm]
            mgd.tau[tii][xfm]      += mgd_com*grd.xfm_qto.tau[tii][xfm]
            mgd.phi[tii][xfm]      += mgd_com*grd.xfm_qto.phi[tii][xfm]
            mgd.u_on_xfm[tii][xfm] += mgd_com*grd.xfm_qto.uon[tii][xfm]
        end

        for dc_fr in idx.bus_is_dc_frs[bus]
            # dc injections -- "qfr" contributions # grd[:qb_slack][bus][:dc_qfr][tii][idx.bus_is_dc_frs[bus]]
            mgd.dc_qfr[tii][dc_fr] += mgd_com
        end

        for dc_to in idx.bus_is_dc_tos[bus]
            # dc injections -- "qto" contributions # grd[:qb_slack][bus][:dc_qto][tii][idx.bus_is_dc_tos[bus]]
            mgd.dc_qto[tii][dc_to] += mgd_com
        end
    end
end

function apply_dev_p_grads!(tii::Int8, prm::quasiGrad.Param, qG::quasiGrad.QG, idx::quasiGrad.Index, stt::quasiGrad.State, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, dev::Union{Int32,Int64}, alpha::Float64)
    # this function computes the partial derivative of dev_p:
    # stt.dev_p[tii] = stt.p_on[tii] + stt.p_su[tii] + stt.p_sd[tii]
    #
    # it then takes the partial derivarives, multiplies by the incoming
    # alpha factor, and then updates the master grad accordingly
    #
    # p_on -- simple
    mgd.p_on[tii][dev] += alpha  # grd[:dev_p][:p_on] == 1 :)

    # the following is expensive, so we skip it during power flow solves
    # (and we don't update binaries anyways!)
    if qG.run_susd_updates
        # to get the derivatives wrt p_su and p_sd, we need T_supc and T_sdpc
        T_supc     = idx.Ts_supc[dev][tii]     # => T_supc, p_supc_set = get_supc(tii, dev, prm)
        p_supc_set = idx.ps_supc_set[dev][tii] # => T_supc, p_supc_set = get_supc(tii, dev, prm)
        T_sdpc     = idx.Ts_sdpc[dev][tii]     # => T_sdpc, p_sdpc_set = get_sdpc(tii, dev, prm)
        p_sdpc_set = idx.ps_sdpc_set[dev][tii] # => T_sdpc, p_sdpc_set = get_sdpc(tii, dev, prm)

        # 2. p_su
        @inbounds for (ii, tii_p) in enumerate(T_supc)
            if tii_p == 1
                mgd.u_on_dev[tii_p][dev]               += alpha*p_supc_set[ii]*sign.(stt.u_su_dev[tii_p][dev])
            else
                mgd.u_on_dev[tii_p][dev]               += alpha*p_supc_set[ii]*  sign.(stt.u_su_dev[tii_p][dev])
                mgd.u_on_dev[prm.ts.tmin1[tii_p]][dev] += alpha*p_supc_set[ii]*(-sign.(stt.u_su_dev[tii_p][dev]))
            end
        end

        # 3. u_sd
        @inbounds for (ii, tii_p) in enumerate(T_sdpc)
            if tii_p == 1
                # we take the derivative of a shutdown variable wrt to just :u_on_dev[tii_p]
                mgd.u_on_dev[tii_p][dev]               += alpha*p_sdpc_set[ii]*-(sign.(stt.u_sd_dev[tii_p][dev]))
            else
                mgd.u_on_dev[tii_p][dev]               += alpha*p_sdpc_set[ii]*-(sign.(stt.u_sd_dev[tii_p][dev]))
                mgd.u_on_dev[prm.ts.tmin1[tii_p]][dev] += alpha*p_sdpc_set[ii]*  sign.(stt.u_sd_dev[tii_p][dev])
            end
        end
    end
end

function apply_dev_q_grads!(tii::Int8, prm::quasiGrad.Param, qG::quasiGrad.QG, idx::quasiGrad.Index, stt::quasiGrad.State, grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, dev::Union{Int32,Int64}, alpha::Float64)
    # if the device is not in idx.J_pqe, then "dev_q" is just a state variable!
    if dev in idx.J_pqe
        # in this case, we take the derivatives of "dev_q" wrt
        #   a) u_on_dev (a la u_sum)
        #
        # the following is expensive, so we skip it during power flow solves
        # (and we don't update binaries anyways!)
        if qG.run_susd_updates
            alpha_usum = alpha*prm.dev.q_0[dev]  # => grd[:dev_q][:u_sum][dev] == prm.dev.q_0[dev]
            T_supc     = idx.Ts_supc[dev][tii] # => get_supc(tii, dev, prm)
            T_sdpc     = idx.Ts_sdpc[dev][tii] # => get_sdpc(tii, dev, prm)
            du_sum!(tii, prm, stt, mgd, dev, alpha_usum, T_supc, T_sdpc)
        end

        #   b) dev_p (and its fellows)
        alpha_p = alpha*prm.dev.beta[dev]  # => grd[:dev_q][:dev_p][dev]
        dp_alpha!(grd, dev, tii, alpha_p)
    else
        mgd.dev_q[tii][dev] += alpha
    end
end

function du_sum!(tii::Int8, prm::quasiGrad.Param, stt::quasiGrad.State, mgd::quasiGrad.MasterGrad, dev::Union{Int32,Int64}, alpha::Float64, T_supc::Vector{Int8}, T_sdpc::Vector{Int8})
    # this function takes the derivative of the commonly used "u_sum"
    # term, and it applies the derivatives across mgd
    #
    # note: this is done for a single device at a time
    #
    # 1. u_on
    mgd.u_on_dev[tii][dev] += alpha

    # 2. u_su
    @inbounds @simd for tii_p in T_supc
        if tii_p == 1
            # we take the derivative of a startup variable wrt to just :u_on_dev[tii_p]
            # note: dsu_duon =  sign.(stt.u_su_dev[tii_p][dev])
            mgd.u_on_dev[tii_p][dev]               += alpha*  sign.(stt.u_su_dev[tii_p][dev])
        else
            mgd.u_on_dev[tii_p][dev]               += alpha*  sign.(stt.u_su_dev[tii_p][dev])
            mgd.u_on_dev[prm.ts.tmin1[tii_p]][dev] += alpha*(-sign.(stt.u_su_dev[tii_p][dev]))
        end
    end

    # 3. u_sd
    @inbounds @simd for tii_p in T_sdpc
        if tii_p == 1
            # we take the derivative of a shutdown variable wrt to just :u_on_dev[tii_p]
            mgd.u_on_dev[tii_p][dev]               += alpha*-(sign.(stt.u_sd_dev[tii_p][dev]))
        else
            mgd.u_on_dev[tii_p][dev]               += alpha*-(sign.(stt.u_sd_dev[tii_p][dev]))
            mgd.u_on_dev[prm.ts.tmin1[tii_p]][dev] += alpha*  sign.(stt.u_sd_dev[tii_p][dev])
        end
    end
end

# flush the master grad and other key gradients terms :)
function flush_gradients!(grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # set all to 0
        mgd.vm[tii]           .= 0.0    
        mgd.va[tii]           .= 0.0        
        mgd.tau[tii]          .= 0.0         
        mgd.phi[tii]          .= 0.0
        mgd.dc_pfr[tii]       .= 0.0
        mgd.dc_qfr[tii]       .= 0.0
        mgd.dc_qto[tii]       .= 0.0
        mgd.u_on_acline[tii]  .= 0.0
        mgd.u_on_xfm[tii]     .= 0.0
        mgd.u_step_shunt[tii] .= 0.0
        mgd.u_on_dev[tii]     .= 0.0
        mgd.p_on[tii]         .= 0.0
        mgd.dev_q[tii]        .= 0.0
        mgd.p_rgu[tii]        .= 0.0
        mgd.p_rgd[tii]        .= 0.0
        mgd.p_scr[tii]        .= 0.0
        mgd.p_nsc[tii]        .= 0.0
        mgd.p_rru_on[tii]     .= 0.0
        mgd.p_rrd_on[tii]     .= 0.0
        mgd.p_rru_off[tii]    .= 0.0
        mgd.p_rrd_off[tii]    .= 0.0
        mgd.q_qru[tii]        .= 0.0
        mgd.q_qrd[tii]        .= 0.0
        # device active and reactive power gradients
        grd.dx.dp[tii]      .= 0.0
        grd.dx.dq[tii]      .= 0.0
    end
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
end

function dp_alpha!(grd::quasiGrad.Grad, dev::Union{Int32,Int64}, tii::Int8, alpha::Float64)
    # this is an intersting function: it collects the partial
    # derivative coefficients which scale dx/dp terms:
    #   alpha1 = (dz/dzb)(dzb/dzF)(dF/dp)
    #   alpha2 = (dz/dzb)(dzb/dzG)(dG/dp)
    # then, when we need to take dp/duon, dp/dpon, etc.,
    # we simply scale all of the needed derivatives by
    # the sum of all alphas
    grd.dx.dp[tii][dev] += alpha
end

function dq_alpha!(grd::quasiGrad.Grad, dev::Union{Int32,Int64}, tii::Int8, alpha::Float64)
    # this is an intersting function: it collects the partial
    # derivative coefficients which scale dx/dq terms:
    #   alpha1 = (dz/dzb)(dzb/dzF)(dF/dq)
    #   alpha2 = (dz/dzb)(dzb/dzG)(dG/dq)
    # then, when we need to take dq/duon, dq/dpon, etc.,
    # we simply scale all of the needed derivatives by
    # the sum of all alphas
    grd.dx.dq[tii][dev] += alpha
end