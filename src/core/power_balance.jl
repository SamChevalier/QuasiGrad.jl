function power_balance!(grd::QuasiGrad.Grad, idx::QuasiGrad.Index, prm::QuasiGrad.Param, qG::QuasiGrad.QG, stt::QuasiGrad.State, sys::QuasiGrad.System)
    # call penalty cost
    cp = prm.vio.p_bus * qG.scale_c_pbus_testing
    cq = prm.vio.q_bus * qG.scale_c_qbus_testing

    # loop over each bus and aggregate powers
    Threads.@threads for bus in 1:sys.nb
        # loop over each time period and compute the power balance
        for tii in prm.ts.time_keys
            QuasiGrad.pq_sums!(bus, idx, stt, tii)
        end
    end

    # now, loop over time
    Threads.@threads for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]
        
        # actual mismatch penalty
        @turbo stt.zp[tii] .= abs.(stt.pb_slack[tii]).*(cp*dt)
        @turbo stt.zq[tii] .= abs.(stt.qb_slack[tii]).*(cq*dt)

        # evaluate the grad?
        if qG.eval_grad
            if qG.pqbal_grad_type == "standard"
                @turbo grd.zp.pb_slack[tii] .= (cp*dt).*sign.(stt.pb_slack[tii])
                @turbo grd.zq.qb_slack[tii] .= (cq*dt).*sign.(stt.qb_slack[tii])
            elseif qG.pqbal_grad_type == "soft_abs"
                @turbo grd.zp.pb_slack[tii] .= (qG.pqbal_grad_weight_p*dt).*(stt.pb_slack[tii]./(QuasiGrad.LoopVectorization.sqrt_fast.(QuasiGrad.LoopVectorization.pow_fast.(stt.pb_slack[tii],2) .+ qG.pqbal_grad_eps2)))
                @turbo grd.zq.qb_slack[tii] .= (qG.pqbal_grad_weight_q*dt).*(stt.qb_slack[tii]./(QuasiGrad.LoopVectorization.sqrt_fast.(QuasiGrad.LoopVectorization.pow_fast.(stt.qb_slack[tii],2) .+ qG.pqbal_grad_eps2)))
            elseif qG.pqbal_grad_type == "quadratic_for_lbfgs"
                @turbo grd.zp.pb_slack[tii] .= (cp*dt).*stt.pb_slack[tii]
                @turbo grd.zq.qb_slack[tii] .= (cq*dt).*stt.qb_slack[tii]
            elseif qG.pqbal_grad_type == "scaled_quadratic"
                @turbo grd.zp.pb_slack[tii] .= (2.0*(qG.pqbal_quadratic_grad_weight_p*dt)).*stt.pb_slack[tii]
                @turbo grd.zq.qb_slack[tii] .= (2.0*(qG.pqbal_quadratic_grad_weight_q*dt)).*stt.qb_slack[tii]
            else
                println("Power balance gradient type not recognized!")
            end
        end
    end
end

# fast sum
function pq_sums!(bus::Int64, idx::QuasiGrad.Index, stt::QuasiGrad.State, tii::Int8)
    # loop over devices
    #
    stt.pb_slack[tii][bus] = 0.0
    stt.qb_slack[tii][bus] = 0.0

    # consumers -- positive
    @fastmath @inbounds for cs in idx.cs[bus]
        stt.pb_slack[tii][bus] += stt.dev_p[tii][cs]
        stt.qb_slack[tii][bus] += stt.dev_q[tii][cs]
    end

    # shunts -- positive
    @fastmath @inbounds for sh in idx.sh[bus]
        stt.pb_slack[tii][bus] += stt.sh_p[tii][sh]
        stt.qb_slack[tii][bus] += stt.sh_q[tii][sh]
    end

    # acline -- positive
    @fastmath @inbounds for acl in idx.bus_is_acline_frs[bus]
        stt.pb_slack[tii][bus] += stt.acline_pfr[tii][acl]
        stt.qb_slack[tii][bus] += stt.acline_qfr[tii][acl]
    end
    @fastmath @inbounds for acl in idx.bus_is_acline_tos[bus]
        stt.pb_slack[tii][bus] += stt.acline_pto[tii][acl]
        stt.qb_slack[tii][bus] += stt.acline_qto[tii][acl]
    end

    # xfm -- positive
    @fastmath @inbounds for xfm in idx.bus_is_xfm_frs[bus]
        stt.pb_slack[tii][bus] += stt.xfm_pfr[tii][xfm]
        stt.qb_slack[tii][bus] += stt.xfm_qfr[tii][xfm]
    end
    @fastmath @inbounds for xfm in idx.bus_is_xfm_tos[bus]
        stt.pb_slack[tii][bus] += stt.xfm_pto[tii][xfm]
        stt.qb_slack[tii][bus] += stt.xfm_qto[tii][xfm]
    end

    # dcline -- positive
    @fastmath @inbounds for dc in idx.bus_is_dc_frs[bus]
        stt.pb_slack[tii][bus] += stt.dc_pfr[tii][dc]
        stt.qb_slack[tii][bus] += stt.dc_qfr[tii][dc]
    end
    @fastmath @inbounds for dc in idx.bus_is_dc_tos[bus] 
        stt.pb_slack[tii][bus] += stt.dc_pto[tii][dc]
        stt.qb_slack[tii][bus] += stt.dc_qto[tii][dc]
    end

    # producer -- NEGATIVE
    @fastmath @inbounds for pr in idx.pr[bus]
        stt.pb_slack[tii][bus] -= stt.dev_p[tii][pr]
        stt.qb_slack[tii][bus] -= stt.dev_q[tii][pr]
    end
end