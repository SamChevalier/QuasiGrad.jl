function reserve_balance!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
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
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]

        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if isempty(idx.cs_pzone[zone])
                # in the case there are NO consumers in a zone
                stt.p_rgu_zonal_REQ[tii][zone] = 0.0
                stt.p_rgd_zonal_REQ[tii][zone] = 0.0
            else
                psum = quasiGrad.sum_power(idx, stt, tii, zone) 
                stt.p_rgu_zonal_REQ[tii][zone] = rgu_sigma[zone]*psum
                stt.p_rgd_zonal_REQ[tii][zone] = rgd_sigma[zone]*psum
            end

            # endogenous max
            if isempty(idx.pr_pzone[zone])
                # in the case there are NO producers in a zone
                stt.p_scr_zonal_REQ[tii][zone] = 0.0
                stt.p_nsc_zonal_REQ[tii][zone] = 0.0
            else
                pmax = quasiGrad.max_power(idx, stt, tii, zone) 
                stt.p_scr_zonal_REQ[tii][zone] = scr_sigma[zone]*pmax
                stt.p_nsc_zonal_REQ[tii][zone] = nsc_sigma[zone]*pmax
            end

            # precompute sums
            rgu     = sum_p_rgu(idx, stt, tii, zone)
            rgd     = sum_p_rgd(idx, stt, tii, zone)
            scr     = sum_p_scr(idx, stt, tii, zone)
            nsc     = sum_p_nsc(idx, stt, tii, zone)
            rru_on  = sum_p_rru_on(idx, stt, tii, zone)
            rru_off = sum_p_rru_off(idx, stt, tii, zone)
            rrd_on  = sum_p_rrd_on(idx, stt, tii, zone)
            rrd_off = sum_p_rrd_off(idx, stt, tii, zone)

            # balance equations -- compute the shortfall values
            stt.p_rgu_zonal_penalty[tii][zone] = max(stt.p_rgu_zonal_REQ[tii][zone] - rgu,0.0)
            stt.p_rgd_zonal_penalty[tii][zone] = max(stt.p_rgd_zonal_REQ[tii][zone] - rgd,0.0)
            stt.p_scr_zonal_penalty[tii][zone] = max(stt.p_rgu_zonal_REQ[tii][zone] + stt.p_scr_zonal_REQ[tii][zone] - rgu - scr,0.0)
            stt.p_nsc_zonal_penalty[tii][zone] = max(stt.p_rgu_zonal_REQ[tii][zone] + stt.p_scr_zonal_REQ[tii][zone] + stt.p_nsc_zonal_REQ[tii][zone] - rgu - scr - nsc,0.0)
            stt.p_rru_zonal_penalty[tii][zone] = max(prm.reserve.rru_min[zone][tii] - rru_on - rru_off,0.0)
            stt.p_rrd_zonal_penalty[tii][zone] = max(prm.reserve.rrd_min[zone][tii] - rrd_on - rrd_off,0.0)
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # precompute sums
            q_qru = sum_q_qru(idx, stt, tii, zone)
            q_qrd = sum_q_qrd(idx, stt, tii, zone)

            # balance equations -- compute the shortfall values
            stt.q_qru_zonal_penalty[tii][zone] = max(prm.reserve.qru_min[zone][tii] - q_qru, 0.0)
            stt.q_qrd_zonal_penalty[tii][zone] = max(prm.reserve.qrd_min[zone][tii] - q_qrd,0.0)
        end

        # shortfall penalties -- gradients are static and taken when initialized
        stt.zrgu_zonal[tii] .= (dt.*crgu).*stt.p_rgu_zonal_penalty[tii]
        stt.zrgd_zonal[tii] .= (dt.*crgd).*stt.p_rgd_zonal_penalty[tii]
        stt.zscr_zonal[tii] .= (dt.*cscr).*stt.p_scr_zonal_penalty[tii]
        stt.znsc_zonal[tii] .= (dt.*cnsc).*stt.p_nsc_zonal_penalty[tii]
        stt.zrru_zonal[tii] .= (dt.*crru).*stt.p_rru_zonal_penalty[tii]
        stt.zrrd_zonal[tii] .= (dt.*crrd).*stt.p_rrd_zonal_penalty[tii]
        stt.zqru_zonal[tii] .= (dt.*cqru).*stt.q_qru_zonal_penalty[tii]
        stt.zqrd_zonal[tii] .= (dt.*cqrd).*stt.q_qrd_zonal_penalty[tii]
    end

    # sleep tasks
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
end

# functions for using polyester, which has a hard time with internal for loops of this sort
function sum_power(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64) 
    return sum(stt.dev_p[tii][dev] for dev in idx.cs_pzone[zone]; init=0.0)
end

function max_power(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64) 
    return maximum(stt.dev_p[tii][dev] for dev in idx.pr_pzone[zone])
end

function sum_p_rgu(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_rgu[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_p_rgd(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_rgd[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_p_scr(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_scr[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_p_nsc(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_nsc[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_p_rru_on(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_rru_on[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_p_rru_off(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_rru_off[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_p_rrd_on(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_rrd_on[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_p_rrd_off(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.p_rrd_off[tii][dev] for dev in idx.dev_pzone[zone]; init=0.0)
end

function sum_q_qru(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.q_qru[tii][dev] for dev in idx.dev_qzone[zone]; init=0.0)
end

function sum_q_qrd(idx::quasiGrad.Idx, stt::quasiGrad.State, tii::Int8, zone::Int64)
    return sum(stt.q_qrd[tii][dev] for dev in idx.dev_qzone[zone]; init=0.0)
end




