function master_grad!(cgd::quasiGrad.ConstantGrad, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # ...follow each z...
    #
    # NOTE: mgd should have been flushed prior to evaluating all gradients,
    #       since some gradient updates directly update mgd :)
    #
    # 1. zctg ==========================================

    # 2. enmin and enmax (in zbase) ====================
    #
    # in this case, we have already computed the
    # necessary derivative terms and placed them in
    # grd.dx.dp -- see the functions energy_penalties!()
    # and dp_alpha!() for more details

    # 3. max starts (in zbase) =========================
    #
    # in this case, we have already computed the
    # necessary derivative terms and placed them in
    # the mgd -- see penalized_device_constraints!()

    # 4. zt (in zbase) =================================
    #
    # note: in many of hte folling, we simplify the derivatives
    # by hardcoding the leading terms in the gradient backprop.
    # OG is the original!
    #
    if qG.eval_grad # this is here because sometimes we want to skip
                    # the master grad when evaluating update_states_and_grads!()
                    
        # ** The following can be used to turn off gradient computation for
        #    devices, if, e.g., we want to focus just on solving power flow
        #    (notably, from testing in early May, this wasn't helpful -- why?
        #     because you can't get a good power flow solution when p and q
        #     are fixed -- they need to move around)

        # loop over time
        for tii in prm.ts.time_keys
            # g1 (zen): nzms => zbase => zt => => zen => (dev_p, u_on_dev)
            #
            # all devices
            for dev in prm.dev.dev_keys
                # OG=> alpha = grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zen_dev][dev] .* grd.zen_dev.dev_p[tii][dev]
                # => alpha = -cgd.dzt_dzen[dev] .* grd.zen_dev.dev_p[tii][dev]
                dp_alpha!(grd, dev, tii, -cgd.dzt_dzen[dev] .* grd.zen_dev.dev_p[tii][dev])
            end

            # g2 (zsu): nzms => zbase => zt => => zsu => u_su_dev => u_on_dev
            #
            # devices
            # OG=> gc_d = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_dev] * grd[:zsu_dev][:u_su_dev]
            mgd.u_on_dev[tii] .+= prm.dev.startup_cost .* grd.u_su_dev.u_on_dev[tii]
            
            if qG.update_acline_xfm_bins
                # acline
                # OG => gc_l = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_acline] * grd[:zsu_acline][:u_su_acline]
                mgd.u_on_acline[tii] .+= prm.acline.connection_cost .* grd.u_su_acline.u_on_acline[tii]

                # xfm
                # OG => gc_x = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_xfm] * grd[:zsu_xfm][:u_su_xfm]
                mgd.u_on_xfm[tii] .+= prm.xfm.connection_cost .* grd.u_su_xfm.u_on_xfm[tii]
            end

            # include previous times?
            if tii != 1
                mgd.u_on_dev[prm.ts.tmin1[tii]]        .+= prm.dev.startup_cost       .* grd.u_su_dev.u_on_dev_prev[tii]
                if qG.update_acline_xfm_bins
                    mgd.u_on_acline[prm.ts.tmin1[tii]] .+= prm.acline.connection_cost .* grd.u_su_acline.u_on_acline_prev[tii]
                    mgd.u_on_xfm[prm.ts.tmin1[tii]]    .+= prm.xfm.connection_cost    .* grd.u_su_xfm.u_on_xfm_prev[tii]
                end
            end
            
            # g3 (zsd): nzms => zbase => zt => => zsd => u_sd_dev => u_on_dev
            #
            # devices
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_dev] * grd[:zsd_dev][:u_sd_dev]
            mgd.u_on_dev[tii] .+= prm.dev.shutdown_cost .* grd.u_sd_dev.u_on_dev[tii]

            if qG.update_acline_xfm_bins
                # acline
                # OG => gc_l = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_acline] * grd[:zsd_acline][:u_sd_acline]
                mgd.u_on_acline[tii] .+= prm.acline.disconnection_cost .* grd.u_sd_acline.u_on_acline[tii]

                # xfm
                # OG => gc_x = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_xfm] * grd[:zsd_xfm][:u_sd_xfm]
                mgd.u_on_xfm[tii] .+= prm.xfm.disconnection_cost .* grd.u_sd_xfm.u_on_xfm[tii]
            end
            # include previous times?
            if tii != 1
                mgd.u_on_dev[prm.ts.tmin1[tii]]    .+= prm.dev.shutdown_cost         .* grd.u_sd_dev.u_on_dev_prev[tii]
                if qG.update_acline_xfm_bins
                    mgd.u_on_acline[prm.ts.tmin1[tii]] .+= prm.acline.disconnection_cost .* grd.u_sd_acline.u_on_acline_prev[tii]
                    mgd.u_on_xfm[prm.ts.tmin1[tii]]    .+= prm.xfm.disconnection_cost    .* grd.u_sd_xfm.u_on_xfm_prev[tii]
                end
            end

            # g4 (zon_dev): nzms => zbase => zt => => zon_dev => u_on_dev
            # OG => mgd.u_on_dev[tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zon_dev] * grd[:zon_dev][:u_on_dev][tii]
            mgd.u_on_dev[tii] .+= cgd.dzon_dev_du_on_dev[tii] #grd[:zon_dev][:u_on_dev][tii]

            # g5 (zsus_dev): nzms => zbase => zt => => zsus_dev => u_on_dev ... ?
                # => taken in device_startup_states!()

            # g6 (zs): nzms => zbase => zt => => zs => (all line and xfm variables)
            master_grad_zs_acline!(tii, idx, grd, mgd, qG, stt, sys)
            master_grad_zs_xfm!(tii, idx, grd, mgd, qG, stt, sys)

            # g7 (zrgu):  nzms => zbase => zt => zrgu => p_rgu
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgu] * cgd.dzrgu_dp_rgu[tii] #grd[:zrgu][:p_rgu][tii]
            mgd.p_rgu[tii] .+= cgd.dzrgu_dp_rgu[tii]

            # g8 (zrgd):  nzms => zbase => zt => => zrgd => p_rgd
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgd] * cgd.dzrgd_dp_rgd[tii] #grd[:zrgd][:p_rgd][tii]
            mgd.p_rgd[tii] .+= cgd.dzrgd_dp_rgd[tii]

            # g9 (zscr):  nzms => zbase => zt => => zscr => p_scr
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zscr] * cgd.dzscr_dp_scr[tii] #grd[:zscr][:p_scr][tii]
            mgd.p_scr[tii] .+= cgd.dzscr_dp_scr[tii]
            
            # g10 (znsc): nzms => zbase => zt => => znsc => p_nsc
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:znsc] * cgd.dznsc_dp_nsc[tii] #grd[:znsc][:p_nsc][tii]
            mgd.p_nsc[tii] .+= cgd.dznsc_dp_nsc[tii]

            # g11 (zrru): nzms => zbase => zt => => zrru => (p_rru_on,p_rru_off)
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru] * cgd.dzrru_dp_rru_on[tii]  #grd[:zrru][:p_rru_on][tii]
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru] * cgd.dzrru_dp_rru_off[tii] #grd[:zrru][:p_rru_off][tii]
            mgd.p_rru_on[tii]  .+= cgd.dzrru_dp_rru_on[tii]
            mgd.p_rru_off[tii] .+= cgd.dzrru_dp_rru_off[tii]

            # g12 (zrrd): nzms => zbase => zt => => zrrd => (p_rrd_on,p_rrd_off)
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd] * cgd.dzrrd_dp_rrd_on[tii]  #grd[:zrrd][:p_rrd_on][tii]
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd] * cgd.dzrrd_dp_rrd_off[tii] #grd[:zrrd][:p_rrd_off][tii]
            mgd.p_rrd_on[tii]  .+= cgd.dzrrd_dp_rrd_on[tii]
            mgd.p_rrd_off[tii] .+= cgd.dzrrd_dp_rrd_off[tii]

            # g13 (zqru): nzms => zbase => zt => => zqru => q_qru
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqru] * cgd.dzqru_dq_qru[tii] #grd[:zqru][:q_qru][tii]
            mgd.q_qru[tii] .+= cgd.dzqru_dq_qru[tii]

            # g14 (zqrd): nzms => zbase => zt => => zqrd => q_qrd
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqrd] * cgd.dzqrd_dq_qrd[tii] #grd[:zqrd][:q_qrd][tii]
            mgd.q_qrd[tii] .+= cgd.dzqrd_dq_qrd[tii]

            # NOTE -- I have lazily left the ac binaries in the following functions -- you can easily remove
            #
            # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
            master_grad_zp!(tii, prm, idx, grd, mgd, sys)
                    
            # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
            master_grad_zq!(tii, prm, idx, grd, mgd, sys)

            # reserve zones -- p
            #
            # note: clipping MUST be called before sign() returns a useful result!!!
            for zone in 1:sys.nzP
                # g17 (zrgu_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgu_zonal] * cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone] #grd[:zrgu_zonal][:p_rgu_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rgu_zonal_penalty[tii][zone], qG)
                else
                    mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*sign(stt.p_rgu_zonal_penalty[tii][zone])
                end
                mgd.p_rgu[tii][idx.dev_pzone[zone]] .-= mgd_com
                # ===> requirements -- depend on active power consumption
                for dev in idx.cs_pzone[zone]
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
                mgd.p_rgd[tii][idx.dev_pzone[zone]] .-= mgd_com
                # ===> requirements -- depend on active power consumption
                for dev in idx.cs_pzone[zone]
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
                mgd.p_rgu[tii][idx.dev_pzone[zone]] .-= mgd_com
                mgd.p_scr[tii][idx.dev_pzone[zone]] .-= mgd_com
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(stt.dev_p[tii][idx.pr_pzone[zone]])]
                    # ===> requirements -- depend on active power production/consumption!
                    for dev = i_pmax # we only take the derivative of the device which has the highest production
                        # => alpha = mgd_com*prm.reserve.scr_sigma[zone]
                        dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.scr_sigma[zone])
                    end
                end
                if ~isempty(idx.cs_pzone[zone])
                    # only do the following if there are consumers here -- overly cautious
                    for dev in idx.cs_pzone[zone]
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
                mgd.p_rgu[tii][idx.dev_pzone[zone]] .-= mgd_com
                mgd.p_scr[tii][idx.dev_pzone[zone]] .-= mgd_com
                mgd.p_nsc[tii][idx.dev_pzone[zone]] .-= mgd_com
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(stt.dev_p[tii][idx.pr_pzone[zone]])]
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
                    mgd.p_rru_on[tii][idx.dev_pzone[zone]]  .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rru_zonal_penalty[tii][zone], qG)
                    mgd.p_rru_off[tii][idx.dev_pzone[zone]] .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rru_zonal_penalty[tii][zone], qG)
                else
                    mgd.p_rru_on[tii][idx.dev_pzone[zone]]  .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*sign(stt.p_rru_zonal_penalty[tii][zone])
                    mgd.p_rru_off[tii][idx.dev_pzone[zone]] .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*sign(stt.p_rru_zonal_penalty[tii][zone])
                end

                # g22 (zrrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd_zonal] * cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone] #grd[:zrrd_zonal][:p_rrd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd.p_rrd_on[tii][idx.dev_pzone[zone]]  .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rrd_zonal_penalty[tii][zone], qG)
                    mgd.p_rrd_off[tii][idx.dev_pzone[zone]] .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.p_rrd_zonal_penalty[tii][zone], qG)
                else
                    mgd.p_rrd_on[tii][idx.dev_pzone[zone]]  .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*sign(stt.p_rrd_zonal_penalty[tii][zone])
                    mgd.p_rrd_off[tii][idx.dev_pzone[zone]] .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*sign(stt.p_rrd_zonal_penalty[tii][zone])
                end
            end

            # reserve zones -- q
            for zone in 1:sys.nzQ
                # g23 (zqru_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqru_zonal] * cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone] #grd[:zqru_zonal][:q_qru_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd.q_qru[tii][idx.dev_qzone[zone]] .-= cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.q_qru_zonal_penalty[tii][zone], qG)
                else
                    mgd.q_qru[tii][idx.dev_qzone[zone]] .-= cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone]*sign(stt.q_qru_zonal_penalty[tii][zone])
                end
                # g24 (zqrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqrd_zonal] * cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone] #grd[:zqrd_zonal][:q_qrd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd.q_qrd[tii][idx.dev_qzone[zone]] .-= cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt.q_qrd_zonal_penalty[tii][zone], qG)
                else
                    mgd.q_qrd[tii][idx.dev_qzone[zone]] .-= cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone]*sign(stt.q_qrd_zonal_penalty[tii][zone])
                end
            end

            # => # loop over time -- compute the partial derivative contributions
            # => for tii in prm.ts.time_keys
            # loop over devices -- compute the partial derivative contributions
            for dev in prm.dev.dev_keys
                apply_dev_q_grads!(tii, prm, qG, idx, stt, grd, mgd, dev, grd.dx.dq[tii][dev])

                # NOTE -- apply_dev_q_grads!() must be called first! some reactive power
                #         terms also call active power terms, which will add to their derivatives
                apply_dev_p_grads!(tii, prm, qG, idx, stt, grd, mgd, dev, grd.dx.dp[tii][dev])
            end
        end
    end
end