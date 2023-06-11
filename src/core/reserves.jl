function reserve_balance!(idx::quasiGrad.Idx, prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # for the "endogenous" reserve requirements
    rgu_sigma = prm.reserve.rgu_sigma
    rgd_sigma = prm.reserve.rgd_sigma 
    scr_sigma = prm.reserve.scr_sigma 
    nsc_sigma = prm.reserve.nsc_sigma  

    # finally, call the penalty costt
    crgu = prm.vio.rgu_zonal
    crgd = prm.vio.rgd_zonal
    cscr = prm.vio.scr_zonal
    cnsc = prm.vio.nsc_zonal
    crru = prm.vio.rru_zonal
    crrd = prm.vio.rrd_zonal
    cqru = prm.vio.qru_zonal
    cqrd = prm.vio.qrd_zonal
    
    # we need access to the time index itself
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if isempty(idx.cs_pzone[zone])
                # in the case there are NO consumers in a zone
                stt[:p_rgu_zonal_REQ][tii][zone] = 0.0
                stt[:p_rgd_zonal_REQ][tii][zone] = 0.0
            else
                stt[:p_rgu_zonal_REQ][tii][zone] = rgu_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]; init=0.0)
                stt[:p_rgd_zonal_REQ][tii][zone] = rgd_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]; init=0.0)
            end

            # endogenous max
            if isempty(idx.pr_pzone[zone])
                # in the case there are NO producers in a zone
                stt[:p_scr_zonal_REQ][tii][zone] = 0.0
                stt[:p_scr_zonal_REQ][tii][zone] = 0.0
            else
                stt[:p_scr_zonal_REQ][tii][zone] = scr_sigma[zone]*maximum([stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]])
                stt[:p_nsc_zonal_REQ][tii][zone] = nsc_sigma[zone]*maximum([stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]])
            end

            # balance equations -- compute the shortfall values
            stt[:p_rgu_zonal_penalty][tii][zone] = max(stt[:p_rgu_zonal_REQ][tii][zone] - 
                        sum(stt[:p_rgu][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0),0.0)
            
            stt[:p_rgd_zonal_penalty][tii][zone] = max(stt[:p_rgd_zonal_REQ][tii][zone] - 
                        sum(stt[:p_rgd][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0),0.0)

            stt[:p_scr_zonal_penalty][tii][zone] = max(stt[:p_rgu_zonal_REQ][tii][zone] + 
                        stt[:p_scr_zonal_REQ][tii][zone] -
                        sum(stt[:p_rgu][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0) -
                        sum(stt[:p_scr][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0),0.0)

            stt[:p_nsc_zonal_penalty][tii][zone] = max(stt[:p_rgu_zonal_REQ][tii][zone] + 
                        stt[:p_scr_zonal_REQ][tii][zone] +
                        stt[:p_nsc_zonal_REQ][tii][zone] -
                        sum(stt[:p_rgu][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0) -
                        sum(stt[:p_scr][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0) - 
                        sum(stt[:p_nsc][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0),0.0)

            stt[:p_rru_zonal_penalty][tii][zone] = max(prm.reserve.rru_min[zone][t_ind] -
                        sum(stt[:p_rru_on][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0) - 
                        sum(stt[:p_rru_off][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0),0.0)

            stt[:p_rrd_zonal_penalty][tii][zone] = max(prm.reserve.rrd_min[zone][t_ind] -
                        sum(stt[:p_rrd_on][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0) - 
                        sum(stt[:p_rrd_off][tii][dev] for dev in idx.dev_pzone[zone]; init=0.0),0.0)
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            stt[:q_qru_zonal_penalty][tii][zone] = max(prm.reserve.qru_min[zone][t_ind] -
                        sum(stt[:q_qru][tii][dev] for dev in idx.dev_qzone[zone]; init=0.0),0.0)

            stt[:q_qrd_zonal_penalty][tii][zone] = max(prm.reserve.qrd_min[zone][t_ind] -
                        sum(stt[:q_qrd][tii][dev] for dev in idx.dev_qzone[zone]; init=0.0),0.0)
        end

        # shortfall penalties -- gradients are static and taken when initialized
        stt[:zrgu_zonal][tii] .= (dt*crgu).*stt[:p_rgu_zonal_penalty][tii]
        stt[:zrgd_zonal][tii] .= (dt*crgd).*stt[:p_rgd_zonal_penalty][tii]
        stt[:zscr_zonal][tii] .= (dt*cscr).*stt[:p_scr_zonal_penalty][tii]
        stt[:znsc_zonal][tii] .= (dt*cnsc).*stt[:p_nsc_zonal_penalty][tii]
        stt[:zrru_zonal][tii] .= (dt*crru).*stt[:p_rru_zonal_penalty][tii]
        stt[:zrrd_zonal][tii] .= (dt*crrd).*stt[:p_rrd_zonal_penalty][tii]
        stt[:zqru_zonal][tii] .= (dt*cqru).*stt[:q_qru_zonal_penalty][tii]
        stt[:zqrd_zonal][tii] .= (dt*cqrd).*stt[:q_qrd_zonal_penalty][tii]
    end
end
