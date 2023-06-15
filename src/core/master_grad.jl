function master_grad!(cgd::quasiGrad.Cgd, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
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
    # grd[:dx][:dp] -- see the functions energy_penalties!()
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

        # if qG.solve_pf == true
        #    # loop over time
        #    for tii in prm.ts.time_keys
        #        # in this case, we just need to evaluate a small number of gradients
        #        # (i.e., we don't need ANY device or ctg gradients)
        #        #
        #        # g6 (zs): nzms => zbase => zt => => zs => (all line and xfm variables)
        #        master_grad_zs_acline!(tii, idx, stt, grd, mgd, sys)
        #        master_grad_zs_xfm!(tii, idx, stt, grd, mgd, sys)
        #        # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
        #        master_grad_zp!(tii, prm, idx, stt, grd, mgd, sys)
        #        # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
        #        master_grad_zq!(tii, prm, idx, stt, grd, mgd, sys)
        #    end
    
        # loop over time
        @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # g1 (zen): nzms => zbase => zt => => zen => (dev_p, u_on_dev)
            #
            # all devices
            for dev in 1:sys.ndev
                # OG=> alpha = grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zen_dev][dev] .* grd[:zen_dev][:dev_p][tii][dev]
                # => alpha = -cgd.dzt_dzen[dev] .* grd[:zen_dev][:dev_p][tii][dev]
                dp_alpha!(grd, dev, tii, -cgd.dzt_dzen[dev] .* grd[:zen_dev][:dev_p][tii][dev])
            end

            # g2 (zsu): nzms => zbase => zt => => zsu => u_su_dev => u_on_dev
            #
            # devices
            # OG=> gc_d = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_dev] * grd[:zsu_dev][:u_su_dev]
            mgd[:u_on_dev][tii] .+= prm.dev.startup_cost .* grd[:u_su_dev][:u_on_dev][tii]
            
            if qG.change_ac_device_bins
                # acline
                # OG => gc_l = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_acline] * grd[:zsu_acline][:u_su_acline]
                mgd[:u_on_acline][tii] .+= prm.acline.connection_cost .* grd[:u_su_acline][:u_on_acline][tii]

                # xfm
                # OG => gc_x = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_xfm] * grd[:zsu_xfm][:u_su_xfm]
                mgd[:u_on_xfm][tii] .+= prm.xfm.connection_cost .* grd[:u_su_xfm][:u_on_xfm][tii]
            end

            # include previous times?
            if tii != :t1
                mgd[:u_on_dev][prm.ts.tmin1[tii]]    .+= prm.dev.startup_cost       .* grd[:u_su_dev][:u_on_dev_prev][tii]
                if qG.change_ac_device_bins
                    mgd[:u_on_acline][prm.ts.tmin1[tii]] .+= prm.acline.connection_cost .* grd[:u_su_acline][:u_on_acline_prev][tii]
                    mgd[:u_on_xfm][prm.ts.tmin1[tii]]    .+= prm.xfm.connection_cost    .* grd[:u_su_xfm][:u_on_xfm_prev][tii]
                end
            end
            
            # g3 (zsd): nzms => zbase => zt => => zsd => u_sd_dev => u_on_dev
            #
            # devices
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_dev] * grd[:zsd_dev][:u_sd_dev]
            mgd[:u_on_dev][tii] .+= prm.dev.shutdown_cost .* grd[:u_sd_dev][:u_on_dev][tii]

            if qG.change_ac_device_bins
                # acline
                # OG => gc_l = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_acline] * grd[:zsd_acline][:u_sd_acline]
                mgd[:u_on_acline][tii] .+= prm.acline.disconnection_cost .* grd[:u_sd_acline][:u_on_acline][tii]

                # xfm
                # OG => gc_x = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_xfm] * grd[:zsd_xfm][:u_sd_xfm]
                mgd[:u_on_xfm][tii] .+= prm.xfm.disconnection_cost .* grd[:u_sd_xfm][:u_on_xfm][tii]
            end
            # include previous times?
            if tii != :t1
                mgd[:u_on_dev][prm.ts.tmin1[tii]]    .+= prm.dev.shutdown_cost         .* grd[:u_sd_dev][:u_on_dev_prev][tii]
                if qG.change_ac_device_bins
                    mgd[:u_on_acline][prm.ts.tmin1[tii]] .+= prm.acline.disconnection_cost .* grd[:u_sd_acline][:u_on_acline_prev][tii]
                    mgd[:u_on_xfm][prm.ts.tmin1[tii]]    .+= prm.xfm.disconnection_cost    .* grd[:u_sd_xfm][:u_on_xfm_prev][tii]
                end
            end

            # g4 (zon_dev): nzms => zbase => zt => => zon_dev => u_on_dev
            # OG => mgd[:u_on_dev][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zon_dev] * grd[:zon_dev][:u_on_dev][tii]
            mgd[:u_on_dev][tii] .+= cgd.dzon_dev_du_on_dev[tii] #grd[:zon_dev][:u_on_dev][tii]

            # g5 (zsus_dev): nzms => zbase => zt => => zsus_dev => u_on_dev ... ?
                # => taken in device_startup_states!()

            # g6 (zs): nzms => zbase => zt => => zs => (all line and xfm variables)
            master_grad_zs_acline!(tii, idx, grd, mgd, qG, sys)
            master_grad_zs_xfm!(tii, idx, grd, mgd, qG, sys)

            # g7 (zrgu):  nzms => zbase => zt => zrgu => p_rgu
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgu] * cgd.dzrgu_dp_rgu[tii] #grd[:zrgu][:p_rgu][tii]
            mgd[:p_rgu][tii] .+= cgd.dzrgu_dp_rgu[tii]

            # g8 (zrgd):  nzms => zbase => zt => => zrgd => p_rgd
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgd] * cgd.dzrgd_dp_rgd[tii] #grd[:zrgd][:p_rgd][tii]
            mgd[:p_rgd][tii] .+= cgd.dzrgd_dp_rgd[tii]

            # g9 (zscr):  nzms => zbase => zt => => zscr => p_scr
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zscr] * cgd.dzscr_dp_scr[tii] #grd[:zscr][:p_scr][tii]
            mgd[:p_scr][tii] .+= cgd.dzscr_dp_scr[tii]
            
            # g10 (znsc): nzms => zbase => zt => => znsc => p_nsc
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:znsc] * cgd.dznsc_dp_nsc[tii] #grd[:znsc][:p_nsc][tii]
            mgd[:p_nsc][tii] .+= cgd.dznsc_dp_nsc[tii]

            # g11 (zrru): nzms => zbase => zt => => zrru => (p_rru_on,p_rru_off)
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru] * cgd.dzrru_dp_rru_on[tii]  #grd[:zrru][:p_rru_on][tii]
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru] * cgd.dzrru_dp_rru_off[tii] #grd[:zrru][:p_rru_off][tii]
            mgd[:p_rru_on][tii]  .+= cgd.dzrru_dp_rru_on[tii]
            mgd[:p_rru_off][tii] .+= cgd.dzrru_dp_rru_off[tii]

            # g12 (zrrd): nzms => zbase => zt => => zrrd => (p_rrd_on,p_rrd_off)
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd] * cgd.dzrrd_dp_rrd_on[tii]  #grd[:zrrd][:p_rrd_on][tii]
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd] * cgd.dzrrd_dp_rrd_off[tii] #grd[:zrrd][:p_rrd_off][tii]
            mgd[:p_rrd_on][tii]  .+= cgd.dzrrd_dp_rrd_on[tii]
            mgd[:p_rrd_off][tii] .+= cgd.dzrrd_dp_rrd_off[tii]

            # g13 (zqru): nzms => zbase => zt => => zqru => q_qru
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqru] * cgd.dzqru_dq_qru[tii] #grd[:zqru][:q_qru][tii]
            mgd[:q_qru][tii] .+= cgd.dzqru_dq_qru[tii]

            # g14 (zqrd): nzms => zbase => zt => => zqrd => q_qrd
            # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqrd] * cgd.dzqrd_dq_qrd[tii] #grd[:zqrd][:q_qrd][tii]
            mgd[:q_qrd][tii] .+= cgd.dzqrd_dq_qrd[tii]

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
                    mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:p_rgu_zonal_penalty][tii][zone], qG)
                else
                    mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*sign(stt[:p_rgu_zonal_penalty][tii][zone])
                end
                mgd[:p_rgu][tii][idx.dev_pzone[zone]] .-= mgd_com
                # ===> requirements -- depend on active power consumption
                for dev in idx.cs_pzone[zone]
                    # => alpha = mgd_com*prm.reserve.rgu_sigma[zone]
                    dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.rgu_sigma[zone])
                end

                # g18 (zrgd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgd_zonal] * cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone] #grd[:zrgd_zonal][:p_rgd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(sign(stt[:p_rgd_zonal_penalty][tii][zone]), qG)
                else
                    mgd_com = cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone]*sign(stt[:p_rgd_zonal_penalty][tii][zone])
                end
                mgd[:p_rgd][tii][idx.dev_pzone[zone]] .-= mgd_com
                # ===> requirements -- depend on active power consumption
                for dev in idx.cs_pzone[zone]
                    # => alpha = mgd_com*prm.reserve.rgd_sigma[zone]
                    dp_alpha!(grd, dev, tii, mgd_com*prm.reserve.rgd_sigma[zone])
                end

                # g19 (zscr_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zscr_zonal] * cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone] #grd[:zscr_zonal][:p_scr_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd_com = cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone]*soft_abs_reserve_grad(sign(stt[:p_scr_zonal_penalty][tii][zone]), qG)
                else
                    mgd_com = cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone]*sign(stt[:p_scr_zonal_penalty][tii][zone])
                end
                mgd[:p_rgu][tii][idx.dev_pzone[zone]] .-= mgd_com
                mgd[:p_scr][tii][idx.dev_pzone[zone]] .-= mgd_com
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(stt[:dev_p][tii][idx.pr_pzone[zone]])]
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
                    mgd_com = cgd.dznsc_zonal_dp_nsc_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:p_nsc_zonal_penalty][tii][zone], qG)
                else
                    mgd_com = cgd.dznsc_zonal_dp_nsc_zonal_penalty[tii][zone]*sign(stt[:p_nsc_zonal_penalty][tii][zone])
                end
                mgd[:p_rgu][tii][idx.dev_pzone[zone]] .-= mgd_com
                mgd[:p_scr][tii][idx.dev_pzone[zone]] .-= mgd_com
                mgd[:p_nsc][tii][idx.dev_pzone[zone]] .-= mgd_com
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(stt[:dev_p][tii][idx.pr_pzone[zone]])]
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
                    mgd[:p_rru_on][tii][idx.dev_pzone[zone]]  .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:p_rru_zonal_penalty][tii][zone], qG)
                    mgd[:p_rru_off][tii][idx.dev_pzone[zone]] .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:p_rru_zonal_penalty][tii][zone], qG)
                else
                    mgd[:p_rru_on][tii][idx.dev_pzone[zone]]  .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*sign(stt[:p_rru_zonal_penalty][tii][zone])
                    mgd[:p_rru_off][tii][idx.dev_pzone[zone]] .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*sign(stt[:p_rru_zonal_penalty][tii][zone])
                end

                # g22 (zrrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd_zonal] * cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone] #grd[:zrrd_zonal][:p_rrd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd[:p_rrd_on][tii][idx.dev_pzone[zone]]  .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:p_rrd_zonal_penalty][tii][zone], qG)
                    mgd[:p_rrd_off][tii][idx.dev_pzone[zone]] .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:p_rrd_zonal_penalty][tii][zone], qG)
                else
                    mgd[:p_rrd_on][tii][idx.dev_pzone[zone]]  .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*sign(stt[:p_rrd_zonal_penalty][tii][zone])
                    mgd[:p_rrd_off][tii][idx.dev_pzone[zone]] .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*sign(stt[:p_rrd_zonal_penalty][tii][zone])
                end
            end

            # reserve zones -- q
            for zone in 1:sys.nzQ
                # g23 (zqru_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqru_zonal] * cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone] #grd[:zqru_zonal][:q_qru_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd[:q_qru][tii][idx.dev_qzone[zone]] .-= cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:q_qru_zonal_penalty][tii][zone], qG)
                else
                    mgd[:q_qru][tii][idx.dev_qzone[zone]] .-= cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone]*sign(stt[:q_qru_zonal_penalty][tii][zone])
                end
                # g24 (zqrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqrd_zonal] * cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone] #grd[:zqrd_zonal][:q_qrd_zonal_penalty][tii][zone]
                if qG.reserve_grad_is_soft_abs
                    mgd[:q_qrd][tii][idx.dev_qzone[zone]] .-= cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone]*soft_abs_reserve_grad(stt[:q_qrd_zonal_penalty][tii][zone], qG)
                else
                    mgd[:q_qrd][tii][idx.dev_qzone[zone]] .-= cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone]*sign(stt[:q_qrd_zonal_penalty][tii][zone])
                end
            end

            # => # loop over time -- compute the partial derivative contributions
            # => for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # loop over devices -- compute the partial derivative contributions
            for dev in 1:sys.ndev
                apply_dev_q_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dq][tii][dev])

                # NOTE -- apply_dev_q_grads!() must be called first! some reactive power
                #         terms also call active power terms, which will add to their derivatives
                apply_dev_p_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dp][tii][dev])
            end
        end
    end
end

function master_grad_solve_pf!(cgd::quasiGrad.Cgd, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # this function takes the gradient of zms with respect to the
    # variables which will help to resolve power flow.
    # Notably, even though we solve time-independent power flow
    # problems, we can still use this case exactly, since:
    # dzms_dv => dpftii_dv maps without any issue

    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # g1 (zen): nzms => zbase => zt => => zen => (dev_p, u_on_dev)
        #
        # all devices
        if qG.include_energy_costs_lbfgs == true
            for dev in 1:sys.ndev
                # OG=> alpha = grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zen_dev][dev] .* grd[:zen_dev][:dev_p][tii][dev]
                alpha = -cgd.dzt_dzen[dev] .* grd[:zen_dev][:dev_p][tii][dev]
                dp_alpha!(grd, dev, tii, alpha)
            end
        end

        # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
        master_grad_zp!(tii, prm, idx, grd, mgd, sys)
        # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
        master_grad_zq!(tii, prm, idx, grd, mgd, sys)

        # => # loop over time -- compute the partial derivative contributions
        # => for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # loop over devices
        for dev in 1:sys.ndev
            apply_dev_q_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dq][tii][dev])

            # NOTE -- apply_dev_q_grads!() must be called first! some reactive power
            #         terms also call active power terms, which will add to their derivatives
            apply_dev_p_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dp][tii][dev])
        end
    end
end

function master_grad_adam_pf!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, sys::quasiGrad.System)
    # this function takes the gradient of zms with respect to the
    # variables which will help to resolve power flow.
    # Notably, even though we solve time-independent power flow
    # problems, we can still use this case exactly, since:
    # dzms_dv => dpftii_dv maps without any issue

    for tii in prm.ts.time_keys
        # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
        master_grad_zp!(tii, prm, idx, grd, mgd, sys, run_devs = false)

        # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
        master_grad_zq!(tii, prm, idx, grd, mgd, sys, run_devs = false)
    end
end

function master_grad_zs_acline!(tii::Symbol, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, qG::quasiGrad.QG, sys::quasiGrad.System)
    # =========== =========== =========== #
                # zs (acline flows)
    # =========== =========== =========== #
    #
    # common master grad
    # OG => mg_com =  grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zs_acline]
    
    # common flow grads
    pfr_com = grd[:zs_acline][:acline_pfr][tii]
    qfr_com = grd[:zs_acline][:acline_qfr][tii]
    pto_com = grd[:zs_acline][:acline_pto][tii]
    qto_com = grd[:zs_acline][:acline_qto][tii]

    # final pfr gradients
    # OG => mgpfr   = mg_com.*pfr_com
    # mgpfr * everything below:
    vmfrpfr = pfr_com.*grd[:acline_pfr][:vmfr][tii]
    vmtopfr = pfr_com.*grd[:acline_pfr][:vmto][tii]
    vafrpfr = pfr_com.*grd[:acline_pfr][:vafr][tii]
    vatopfr = pfr_com.*grd[:acline_pfr][:vato][tii]

    # final qfr gradients
    # OG => mgqfr   = mg_com.*qfr_com
    # mgqfr * everything below:
    vmfrqfr = qfr_com.*grd[:acline_qfr][:vmfr][tii]
    vmtoqfr = qfr_com.*grd[:acline_qfr][:vmto][tii]
    vafrqfr = qfr_com.*grd[:acline_qfr][:vafr][tii]
    vatoqfr = qfr_com.*grd[:acline_qfr][:vato][tii]

    # final pto gradients
    # OG => mgpto   = mg_com.*pto_com
    # mgpto * everything below:
    vmfrpto = pto_com.*grd[:acline_pto][:vmfr][tii]
    vmtopto = pto_com.*grd[:acline_pto][:vmto][tii]
    vafrpto = pto_com.*grd[:acline_pto][:vafr][tii]
    vatopto = pto_com.*grd[:acline_pto][:vato][tii]

    # final qfr gradients
    # OG => mgqto   = mg_com.*qto_com
    # mgqto * everything below:
    vmfrqto = qto_com.*grd[:acline_qto][:vmfr][tii]
    vmtoqto = qto_com.*grd[:acline_qto][:vmto][tii]
    vafrqto = qto_com.*grd[:acline_qto][:vafr][tii]
    vatoqto = qto_com.*grd[:acline_qto][:vato][tii]

    if qG.change_ac_device_bins
        uonpfr  = pfr_com.*grd[:acline_pfr][:uon][tii]
        uonqfr  = qfr_com.*grd[:acline_qfr][:uon][tii]
        uonpto  = pto_com.*grd[:acline_pto][:uon][tii]
        uonqto  = qto_com.*grd[:acline_qto][:uon][tii]
    end

    # note: we MUST loop over these assignments! otherwise, += gets confused
    for ln in 1:sys.nl
        # see binaries at the bottom
        #
        # update the master grad -- pfr
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrpfr[ln]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtopfr[ln]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrpfr[ln]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatopfr[ln]

        # update the master grad -- qfr
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqfr[ln]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqfr[ln]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqfr[ln]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqfr[ln]

        # update the master grad -- pto
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrpto[ln]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtopto[ln]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrpto[ln]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatopto[ln]

        # update the master grad -- qto
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqto[ln]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqto[ln]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqto[ln]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqto[ln]

        if qG.change_ac_device_bins
            mgd[:u_on_acline][tii][ln]           += uonpfr[ln]
            mgd[:u_on_acline][tii][ln]           += uonqfr[ln]
            mgd[:u_on_acline][tii][ln]           += uonpto[ln]
            mgd[:u_on_acline][tii][ln]           += uonqto[ln]
        end
    end
end

function master_grad_zs_xfm!(tii::Symbol, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, qG::quasiGrad.QG, sys::quasiGrad.System)
    # =========== =========== =========== #
                # zs (xfm)
    # =========== =========== =========== #
    #
    # common master grad
    # OG => mg_com =  grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zs_xfm]
    
    # common flow grads
    pfr_com = grd[:zs_xfm][:xfm_pfr][tii]
    qfr_com = grd[:zs_xfm][:xfm_qfr][tii]
    pto_com = grd[:zs_xfm][:xfm_pto][tii]
    qto_com = grd[:zs_xfm][:xfm_qto][tii]

    # final pfr gradients
    # OG => mgpfr   = mg_com.*pfr_com
    # ... * everything below:
    vmfrpfr = pfr_com.*grd[:xfm_pfr][:vmfr][tii]
    vmtopfr = pfr_com.*grd[:xfm_pfr][:vmto][tii]
    vafrpfr = pfr_com.*grd[:xfm_pfr][:vafr][tii]
    vatopfr = pfr_com.*grd[:xfm_pfr][:vato][tii]
    taupfr  = pfr_com.*grd[:xfm_pfr][:tau][tii]
    phipfr  = pfr_com.*grd[:xfm_pfr][:phi][tii]

    # final qfr gradients
    # OG => mgqfr   = mg_com.*qfr_com
    # ... * everything below:
    vmfrqfr = qfr_com.*grd[:xfm_qfr][:vmfr][tii]
    vmtoqfr = qfr_com.*grd[:xfm_qfr][:vmto][tii]
    vafrqfr = qfr_com.*grd[:xfm_qfr][:vafr][tii]
    vatoqfr = qfr_com.*grd[:xfm_qfr][:vato][tii]
    tauqfr  = qfr_com.*grd[:xfm_qfr][:tau][tii]
    phiqfr  = qfr_com.*grd[:xfm_qfr][:phi][tii]

    # final pto gradients
    # OG => mgpto   = mg_com.*pto_com
    # ... * everything below:
    vmfrpto = pto_com.*grd[:xfm_pto][:vmfr][tii]
    vmtopto = pto_com.*grd[:xfm_pto][:vmto][tii]
    vafrpto = pto_com.*grd[:xfm_pto][:vafr][tii]
    vatopto = pto_com.*grd[:xfm_pto][:vato][tii]
    taupto  = pto_com.*grd[:xfm_pto][:tau][tii]
    phipto  = pto_com.*grd[:xfm_pto][:phi][tii]

    # final qfr gradients
    # OG => mgqto   = mg_com.*qto_com
    # ... * everything below:
    vmfrqto = qto_com.*grd[:xfm_qto][:vmfr][tii]
    vmtoqto = qto_com.*grd[:xfm_qto][:vmto][tii]
    vafrqto = qto_com.*grd[:xfm_qto][:vafr][tii]
    vatoqto = qto_com.*grd[:xfm_qto][:vato][tii]
    tauqto  = qto_com.*grd[:xfm_qto][:tau][tii]
    phiqto  = qto_com.*grd[:xfm_qto][:phi][tii]

    if qG.change_ac_device_bins
        uonpfr  = pfr_com.*grd[:xfm_pfr][:uon][tii]
        uonqfr  = qfr_com.*grd[:xfm_qfr][:uon][tii]
        uonpto  = pto_com.*grd[:xfm_pto][:uon][tii]
        uonqto  = qto_com.*grd[:xfm_qto][:uon][tii]
    end

    # note: we must loop over these assignments!
    for xfm in 1:sys.nx
        # see binaries at the bottom
        #
        # update the master grad -- pfr
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrpfr[xfm]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtopfr[xfm]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrpfr[xfm]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatopfr[xfm]
        mgd[:tau][tii][xfm]                += taupfr[xfm]
        mgd[:phi][tii][xfm]                += phipfr[xfm]

        # update the master grad -- qfr
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqfr[xfm]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqfr[xfm]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqfr[xfm]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqfr[xfm]
        mgd[:tau][tii][xfm]                += tauqfr[xfm]
        mgd[:phi][tii][xfm]                += phiqfr[xfm]

        # update the master grad -- pto
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrpto[xfm]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtopto[xfm]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrpto[xfm]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatopto[xfm]
        mgd[:tau][tii][xfm]                += taupto[xfm]
        mgd[:phi][tii][xfm]                += phipto[xfm]

        # update the master grad -- qto
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqto[xfm]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqto[xfm]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqto[xfm]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqto[xfm]
        mgd[:tau][tii][xfm]                += tauqto[xfm]
        mgd[:phi][tii][xfm]                += phiqto[xfm]

        if qG.change_ac_device_bins
            mgd[:u_on_xfm][tii][xfm]  += uonpfr[xfm]
            mgd[:u_on_xfm][tii][xfm]  += uonqfr[xfm]
            mgd[:u_on_xfm][tii][xfm]  += uonpto[xfm]
            mgd[:u_on_xfm][tii][xfm]  += uonqto[xfm]
        end
    end
end

function master_grad_zp!(tii::Symbol, prm::quasiGrad.Param, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System; run_devs = true)
    # gradient chain: nzms => zbase => zt => zp => (all p injection variables)
    #
    # note: grd[:pb_slack][...] is negelected here, and all terms are trivially
    # hardcoded based on the conservation equations
    
    for bus in 1:sys.nb
        # common gradient term
        # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zp]
        mgd_com = grd[:zp][:pb_slack][tii][bus]

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
        mgd[:vm][tii][bus] += sum(mgd_com*grd[:sh_p][:vm][tii][sh] for sh in idx.sh[bus]; init=0.0)

        # shunt injections -- shunt steps # grd[:pb_slack][bus][:sh_p][tii][idx.sh[bus]].*
        for sh in idx.sh[bus]
            mgd[:u_step_shunt][tii][sh] += mgd_com*
            grd[:sh_p][:g_tv_shunt][tii][sh]*
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
            # => vmfrpfr = grd[:acline_pfr][:vmfr][tii][line]
            # => vafrpfr = grd[:acline_pfr][:vafr][tii][line]
            # => uonpfr  = grd[:acline_pfr][:uon][tii][line]

            # "pfr" is also a function of the to bus voltages, so we need these
            # => vmtopfr = grd[:acline_pfr][:vmto][tii][line]
            # => vatopfr = grd[:acline_pfr][:vato][tii][line]

            # update the master grad -- pfr, at this bus and its corresponding "to" bus
            mgd[:vm][tii][bus_fr]        += mgd_com*grd[:acline_pfr][:vmfr][tii][line]
            mgd[:va][tii][bus_fr]        += mgd_com*grd[:acline_pfr][:vafr][tii][line]
            mgd[:vm][tii][bus_to]        += mgd_com*grd[:acline_pfr][:vmto][tii][line]
            mgd[:va][tii][bus_to]        += mgd_com*grd[:acline_pfr][:vato][tii][line]
            mgd[:u_on_acline][tii][line] += mgd_com*grd[:acline_pfr][:uon][tii][line]
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
            # => vmtopto = grd[:acline_pto][:vmto][tii][line]
            # => vatopto = grd[:acline_pto][:vato][tii][line]
            # => uonpto  = grd[:acline_pto][:uon][tii][line]

            # "pto" is also a function of the fr bus voltages, so we need these
            # => vmfrpto = grd[:acline_pto][:vmfr][tii][line]
            # => vafrpto = grd[:acline_pto][:vafr][tii][line]

            # update the master grad -- pto, at this bus and its corresponding "fr" bus
            mgd[:vm][tii][bus_to]        += mgd_com*grd[:acline_pto][:vmto][tii][line]
            mgd[:va][tii][bus_to]        += mgd_com*grd[:acline_pto][:vato][tii][line]
            mgd[:vm][tii][bus_fr]        += mgd_com*grd[:acline_pto][:vmfr][tii][line]
            mgd[:va][tii][bus_fr]        += mgd_com*grd[:acline_pto][:vafr][tii][line]
            mgd[:u_on_acline][tii][line] += mgd_com*grd[:acline_pto][:uon][tii][line]
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
            # => vmfrpfr = grd[:xfm_pfr][:vmfr][tii][xfm]
            # => vafrpfr = grd[:xfm_pfr][:vafr][tii][xfm]
            # => uonpfr  = grd[:xfm_pfr][:uon][tii][xfm]

            # "pfr" is also a function of the to bus voltages, so we need these
            # => vmtopfr = grd[:xfm_pfr][:vmto][tii][xfm]
            # => vatopfr = grd[:xfm_pfr][:vato][tii][xfm]

            # xfm ratios
            # => taupfr  = grd[:xfm_pfr][:tau][tii][xfm]
            # => phipfr  = grd[:xfm_pfr][:phi][tii][xfm]

            # update the master grad -- pfr, at this bus and its corresponding "to" bus
            mgd[:vm][tii][bus_fr]    += mgd_com*grd[:xfm_pfr][:vmfr][tii][xfm]
            mgd[:va][tii][bus_fr]    += mgd_com*grd[:xfm_pfr][:vafr][tii][xfm]
            mgd[:vm][tii][bus_to]    += mgd_com*grd[:xfm_pfr][:vmto][tii][xfm]
            mgd[:va][tii][bus_to]    += mgd_com*grd[:xfm_pfr][:vato][tii][xfm]
            mgd[:tau][tii][xfm]      += mgd_com*grd[:xfm_pfr][:tau][tii][xfm]
            mgd[:phi][tii][xfm]      += mgd_com*grd[:xfm_pfr][:phi][tii][xfm]
            mgd[:u_on_xfm][tii][xfm] += mgd_com*grd[:xfm_pfr][:uon][tii][xfm]
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
            # => vmtopto = grd[:xfm_pto][:vmto][tii][xfm]
            # => vatopto = grd[:xfm_pto][:vato][tii][xfm]
            # => uonpto  = grd[:xfm_pto][:uon][tii][xfm]

            # "pto" is also a function of the fr bus voltages, so we need these
            # => vmfrpto = grd[:xfm_pto][:vmfr][tii][xfm]
            # => vafrpto = grd[:xfm_pto][:vafr][tii][xfm]

            # xfm ratios
            # => taupto  = grd[:xfm_pto][:tau][tii][xfm]
            # => phipto  = grd[:xfm_pto][:phi][tii][xfm]

            # update the master grad -- pto, at this bus and its corresponding "fr" bus
            mgd[:vm][tii][bus_to]    += mgd_com*grd[:xfm_pto][:vmto][tii][xfm]
            mgd[:va][tii][bus_to]    += mgd_com*grd[:xfm_pto][:vato][tii][xfm]
            mgd[:vm][tii][bus_fr]    += mgd_com*grd[:xfm_pto][:vmfr][tii][xfm]
            mgd[:va][tii][bus_fr]    += mgd_com*grd[:xfm_pto][:vafr][tii][xfm]
            mgd[:tau][tii][xfm]      += mgd_com*grd[:xfm_pto][:tau][tii][xfm]
            mgd[:phi][tii][xfm]      += mgd_com*grd[:xfm_pto][:phi][tii][xfm]
            mgd[:u_on_xfm][tii][xfm] += mgd_com*grd[:xfm_pto][:uon][tii][xfm]
        end

        # skip dc lines if there are none
        for dc_fr in idx.bus_is_dc_frs[bus]
            # dc injections -- "pfr" contributions # grd[:pb_slack][bus][:dc_pfr][tii][idx.bus_is_dc_frs[bus]]
            mgd[:dc_pfr][tii][dc_fr] += mgd_com
        end

        for dc_to in idx.bus_is_dc_tos[bus]
            # dc injections -- "pto" contributions
            #
            # note: "dc_pto" does not exist, as a "mgd" variable, so
            # we just leverage that dc_pto = -dc_pfr   ->  d(dc_pto)_d(dc_pfr) = -1 grd[:pb_slack][bus][:dc_pto][tii][idx.bus_is_dc_tos[bus]]
            mgd[:dc_pfr][tii][dc_to] += -mgd_com
        end
    end
end

function master_grad_zq!(tii::Symbol, prm::quasiGrad.Param, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System; run_devs = true)
    # gradient chain: nzms => zbase => zt => zq => (all q injection variables)
    #
    # note: grd[:qb_slack][...] is negelected here, and all terms are trivially
    # hardcoded based on the conservation equations
    for bus in 1:sys.nb
        # common gradient term
        # OG => grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zq] * grd[:zq][:qb_slack][tii][bus]
        mgd_com = grd[:zq][:qb_slack][tii][bus]

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
        mgd[:vm][tii][bus] += sum(mgd_com*grd[:sh_q][:vm][tii][sh] for sh in idx.sh[bus]; init=0.0)

        # shunt injections -- shunt steps # grd[:qb_slack][bus][:sh_q][tii][idx.sh[bus]].*
        for sh in idx.sh[bus]
            mgd[:u_step_shunt][tii][sh] += mgd_com*
            grd[:sh_q][:b_tv_shunt][tii][sh].*
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

            # gradients
            # => vmfrqfr = grd[:acline_qfr][:vmfr][tii][line]
            # => vafrqfr = grd[:acline_qfr][:vafr][tii][line]
            # => uonqfr  = grd[:acline_qfr][:uon][tii][line]

            # "qfr" is also a function of the to bus voltages, so we need these
            # => vmtoqfr = grd[:acline_qfr][:vmto][tii][line]
            # => vatoqfr = grd[:acline_qfr][:vato][tii][line]

            # update the master grad -- qfr, at this bus and its corresponding "to" bus
            mgd[:vm][tii][bus_fr]        += mgd_com*grd[:acline_qfr][:vmfr][tii][line]
            mgd[:va][tii][bus_fr]        += mgd_com*grd[:acline_qfr][:vafr][tii][line]
            mgd[:vm][tii][bus_to]        += mgd_com*grd[:acline_qfr][:vmto][tii][line]
            mgd[:va][tii][bus_to]        += mgd_com*grd[:acline_qfr][:vato][tii][line]
            mgd[:u_on_acline][tii][line] += mgd_com*grd[:acline_qfr][:uon][tii][line]
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
            # => vmtoqto = grd[:acline_qto][:vmto][tii][line]
            # => vatoqto = grd[:acline_qto][:vato][tii][line]
            # => uonqto  = grd[:acline_qto][:uon][tii][line]

            # "qto" is also a function of the fr bus voltages, so we need these
            # => vmfrqto = grd[:acline_qto][:vmfr][tii][line]
            # => vafrqto = grd[:acline_qto][:vafr][tii][line]

            # update the master grad -- qto, at this bus and its corresponding "fr" bus
            mgd[:vm][tii][bus_to]        += mgd_com*grd[:acline_qto][:vmto][tii][line]
            mgd[:va][tii][bus_to]        += mgd_com*grd[:acline_qto][:vato][tii][line]
            mgd[:vm][tii][bus_fr]        += mgd_com*grd[:acline_qto][:vmfr][tii][line]
            mgd[:va][tii][bus_fr]        += mgd_com*grd[:acline_qto][:vafr][tii][line]
            mgd[:u_on_acline][tii][line] += mgd_com*grd[:acline_qto][:uon][tii][line]
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
            # => vmfrqfr = grd[:xfm_qfr][:vmfr][tii][xfm]
            # => vafrqfr = grd[:xfm_qfr][:vafr][tii][xfm]
            # => uonqfr  = grd[:xfm_qfr][:uon][tii][xfm]

            # "qfr" is also a function of the to bus voltages, so we need these
            # => vmtoqfr = grd[:xfm_qfr][:vmto][tii][xfm]
            # => vatoqfr = grd[:xfm_qfr][:vato][tii][xfm]

            # xfm ratios
            # => tauqfr  = grd[:xfm_qfr][:tau][tii][xfm]
            # => phiqfr  = grd[:xfm_qfr][:phi][tii][xfm]

            # update the master grad -- qfr, at this bus and its corresponding "to" bus
            mgd[:vm][tii][bus_fr]    += mgd_com*grd[:xfm_qfr][:vmfr][tii][xfm]
            mgd[:va][tii][bus_fr]    += mgd_com*grd[:xfm_qfr][:vafr][tii][xfm]
            mgd[:vm][tii][bus_to]    += mgd_com*grd[:xfm_qfr][:vmto][tii][xfm]
            mgd[:va][tii][bus_to]    += mgd_com*grd[:xfm_qfr][:vato][tii][xfm]
            mgd[:tau][tii][xfm]      += mgd_com*grd[:xfm_qfr][:tau][tii][xfm]
            mgd[:phi][tii][xfm]      += mgd_com*grd[:xfm_qfr][:phi][tii][xfm]
            mgd[:u_on_xfm][tii][xfm] += mgd_com*grd[:xfm_qfr][:uon][tii][xfm]
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
            # => vmtoqto = grd[:xfm_qto][:vmto][tii][xfm]
            # => vatoqto = grd[:xfm_qto][:vato][tii][xfm]
            # => uonqto  = grd[:xfm_qto][:uon][tii][xfm]

            # "qto" is also a function of the fr bus voltages, so we need these
            # => vmfrqto = grd[:xfm_qto][:vmfr][tii][xfm]
            # => vafrqto = grd[:xfm_qto][:vafr][tii][xfm]

            # xfm ratios
            # => tauqto  = grd[:xfm_qto][:tau][tii][xfm]
            # => phiqto  = grd[:xfm_qto][:phi][tii][xfm]

            # update the master grad -- qto, at this bus and its corresponding "fr" bus
            mgd[:vm][tii][bus_to]    += mgd_com*grd[:xfm_qto][:vmto][tii][xfm]
            mgd[:va][tii][bus_to]    += mgd_com*grd[:xfm_qto][:vato][tii][xfm]
            mgd[:vm][tii][bus_fr]    += mgd_com*grd[:xfm_qto][:vmfr][tii][xfm]
            mgd[:va][tii][bus_fr]    += mgd_com*grd[:xfm_qto][:vafr][tii][xfm]
            mgd[:tau][tii][xfm]      += mgd_com*grd[:xfm_qto][:tau][tii][xfm]
            mgd[:phi][tii][xfm]      += mgd_com*grd[:xfm_qto][:phi][tii][xfm]
            mgd[:u_on_xfm][tii][xfm] += mgd_com*grd[:xfm_qto][:uon][tii][xfm]
        end

        for dc_fr in idx.bus_is_dc_frs[bus]
            # dc injections -- "qfr" contributions # grd[:qb_slack][bus][:dc_qfr][tii][idx.bus_is_dc_frs[bus]]
            mgd[:dc_qfr][tii][dc_fr] += mgd_com
        end

        for dc_to in idx.bus_is_dc_tos[bus]
            # dc injections -- "qto" contributions # grd[:qb_slack][bus][:dc_qto][tii][idx.bus_is_dc_tos[bus]]
            mgd[:dc_qto][tii][dc_to] += mgd_com
        end
    end
end

function apply_dev_p_grads!(tii::Symbol, t_ind::Int64, prm::quasiGrad.Param, qG::quasiGrad.QG, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, dev::Int64, alpha::Float64)
    # this function computes the partial derivative of dev_p:
    # stt[:dev_p][tii] = stt[:p_on][tii] + stt[:p_su][tii] + stt[:p_sd][tii]
    #
    # it then takes the partial derivarives, multiplies by the incoming
    # alpha factor, and then updates the master grad accordingly
    #
    # p_on -- simple
    mgd[:p_on][tii][dev] += alpha  # grd[:dev_p][:p_on] == 1 :)

    # the following is expensive, so we skip it during power flow solves
    # (and we don't update binaries anyways!)
    if qG.run_susd_updates
        # to get the derivatives wrt p_su and p_sd, we need T_supc and T_sdpc
        T_supc     = idx.Ts_supc[dev][t_ind]     # => T_supc, p_supc_set = get_supc(tii, dev, prm)
        p_supc_set = idx.ps_supc_set[dev][t_ind] # => T_supc, p_supc_set = get_supc(tii, dev, prm)
        T_sdpc     = idx.Ts_sdpc[dev][t_ind]     # => T_sdpc, p_sdpc_set = get_sdpc(tii, dev, prm)
        p_sdpc_set = idx.ps_sdpc_set[dev][t_ind] # => T_sdpc, p_sdpc_set = get_sdpc(tii, dev, prm)

        # 2. p_su
        for (ii, tii_p) in enumerate(T_supc)
            if tii_p == :t1
                mgd[:u_on_dev][tii_p][dev]               += alpha*p_supc_set[ii]*sign.(stt[:u_su_dev][tii_p][dev])
            else
                mgd[:u_on_dev][tii_p][dev]               += alpha*p_supc_set[ii]*  sign.(stt[:u_su_dev][tii_p][dev])
                mgd[:u_on_dev][prm.ts.tmin1[tii_p]][dev] += alpha*p_supc_set[ii]*(-sign.(stt[:u_su_dev][tii_p][dev]))
            end
        end

        # 3. u_sd
        for (ii, tii_p) in enumerate(T_sdpc)
            if tii_p == :t1
                # we take the derivative of a shutdown variable wrt to just :u_on_dev[tii_p]
                mgd[:u_on_dev][tii_p][dev]               += alpha*p_sdpc_set[ii]*-(sign.(stt[:u_sd_dev][tii_p][dev]))
            else
                mgd[:u_on_dev][tii_p][dev]               += alpha*p_sdpc_set[ii]*-(sign.(stt[:u_sd_dev][tii_p][dev]))
                mgd[:u_on_dev][prm.ts.tmin1[tii_p]][dev] += alpha*p_sdpc_set[ii]*  sign.(stt[:u_sd_dev][tii_p][dev])
            end
        end
    end
end

function apply_dev_q_grads!(tii::Symbol, t_ind::Int64, prm::quasiGrad.Param, qG::quasiGrad.QG, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, dev::Int64, alpha::Float64)
    # if the device is not in idx.J_pqe, then "dev_q" is just a state variable!
    if dev in idx.J_pqe
        # in this case, we take the derivatives of "dev_q" wrt
        #   a) u_on_dev (a la u_sum)
        #
        # the following is expensive, so we skip it during power flow solves
        # (and we don't update binaries anyways!)
        if qG.run_susd_updates
            alpha_usum = alpha*prm.dev.q_0[dev]  # => grd[:dev_q][:u_sum][dev] == prm.dev.q_0[dev]
            T_supc     = idx.Ts_supc[dev][t_ind] # => get_supc(tii, dev, prm)
            T_sdpc     = idx.Ts_sdpc[dev][t_ind] # => get_sdpc(tii, dev, prm)
            du_sum!(tii, prm, stt, mgd, dev, alpha_usum, T_supc, T_sdpc)
        end

        #   b) dev_p (and its fellows)
        alpha_p = alpha*prm.dev.beta[dev]  # => grd[:dev_q][:dev_p][dev]
        dp_alpha!(grd, dev, tii, alpha_p)
    else
        mgd[:dev_q][tii][dev] += alpha
    end
end

function du_sum!(tii::Symbol, prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, dev::Int64, alpha::Float64, T_supc::Vector{Symbol}, T_sdpc::Vector{Symbol})
    # this function takes the derivative of the commonly used "u_sum"
    # term, and it applies the derivatives across mgd
    #
    # note: this is done for a single device at a time
    #
    # 1. u_on
    mgd[:u_on_dev][tii][dev] += alpha

    # 2. u_su
    for tii_p in T_supc
        if tii_p == :t1
            # we take the derivative of a startup variable wrt to just :u_on_dev[tii_p]
            # note: dsu_duon =  sign.(stt[:u_su_dev][tii_p][dev])
            mgd[:u_on_dev][tii_p][dev]               += alpha*  sign.(stt[:u_su_dev][tii_p][dev])
        else
            mgd[:u_on_dev][tii_p][dev]               += alpha*  sign.(stt[:u_su_dev][tii_p][dev])
            mgd[:u_on_dev][prm.ts.tmin1[tii_p]][dev] += alpha*(-sign.(stt[:u_su_dev][tii_p][dev]))
        end
    end

    # 3. u_sd
    for tii_p in T_sdpc
        if tii_p == :t1
            # we take the derivative of a shutdown variable wrt to just :u_on_dev[tii_p]
            mgd[:u_on_dev][tii_p][dev]               += alpha*-(sign.(stt[:u_sd_dev][tii_p][dev]))
        else
            mgd[:u_on_dev][tii_p][dev]               += alpha*-(sign.(stt[:u_sd_dev][tii_p][dev]))
            mgd[:u_on_dev][prm.ts.tmin1[tii_p]][dev] += alpha*  sign.(stt[:u_sd_dev][tii_p][dev])
        end
    end
end

# flush the master grad and other key gradients terms :)
function flush_gradients!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, sys::quasiGrad.System)
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # set all to 0
        mgd[:vm][tii]           .= 0.0    
        mgd[:va][tii]           .= 0.0        
        mgd[:tau][tii]          .= 0.0         
        mgd[:phi][tii]          .= 0.0
        mgd[:dc_pfr][tii]       .= 0.0
        mgd[:dc_qfr][tii]       .= 0.0
        mgd[:dc_qto][tii]       .= 0.0
        mgd[:u_on_acline][tii]  .= 0.0
        mgd[:u_on_xfm][tii]     .= 0.0
        mgd[:u_step_shunt][tii] .= 0.0
        mgd[:u_on_dev][tii]     .= 0.0
        mgd[:p_on][tii]         .= 0.0
        mgd[:dev_q][tii]        .= 0.0
        mgd[:p_rgu][tii]        .= 0.0
        mgd[:p_rgd][tii]        .= 0.0
        mgd[:p_scr][tii]        .= 0.0
        mgd[:p_nsc][tii]        .= 0.0
        mgd[:p_rru_on][tii]     .= 0.0
        mgd[:p_rrd_on][tii]     .= 0.0
        mgd[:p_rru_off][tii]    .= 0.0
        mgd[:p_rrd_off][tii]    .= 0.0
        mgd[:q_qru][tii]        .= 0.0
        mgd[:q_qrd][tii]        .= 0.0
        # device active and reactive power gradients
        grd[:dx][:dp][tii]      .= 0.0
        grd[:dx][:dq][tii]      .= 0.0
    end
end

function dp_alpha!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, dev::Int64, tii::Symbol, alpha::Float64)
    # this is an intersting function: it collects the partial
    # derivative coefficients which scale dx/dp terms:
    #   alpha1 = (dz/dzb)(dzb/dzF)(dF/dp)
    #   alpha2 = (dz/dzb)(dzb/dzG)(dG/dp)
    # then, when we need to take dp/duon, dp/dpon, etc.,
    # we simply scale all of the needed derivatives by
    # the sum of all alphas
    grd[:dx][:dp][tii][dev] += alpha
end

function dq_alpha!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, dev::Int64, tii::Symbol, alpha::Float64)
    # this is an intersting function: it collects the partial
    # derivative coefficients which scale dx/dq terms:
    #   alpha1 = (dz/dzb)(dzb/dzF)(dF/dq)
    #   alpha2 = (dz/dzb)(dzb/dzG)(dG/dq)
    # then, when we need to take dq/duon, dq/dpon, etc.,
    # we simply scale all of the needed derivatives by
    # the sum of all alphas
    grd[:dx][:dq][tii][dev] += alpha
end