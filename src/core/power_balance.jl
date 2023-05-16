function power_balance!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # call penalty costt
    cp       = prm.vio.p_bus * qG.scale_c_pbus_testing
    cq       = prm.vio.q_bus * qG.scale_c_qbus_testing
    pb_slack = Vector{Float64}(undef,(sys.nb))
    qb_slack = Vector{Float64}(undef,(sys.nb))

    # loop over each time period and compute the power balance
    for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]

        # loop over each bus
        for bus in 1:sys.nb
            # active power balance: stt[:pb_slack][tii][bus] to record with time
            pb_slack[bus] = 
                    # consumers (positive)
                    sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) +
                    # shunt
                    sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) +
                    # acline
                    sum(stt[:acline_pfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                    sum(stt[:acline_pto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                    # xfm
                    sum(stt[:xfm_pfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                    sum(stt[:xfm_pto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
                    # dcline
                    sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
                    sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
                    # producer (negative)
                   -sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0)
            
            # reactive power balance
            qb_slack[bus] = 
                    # consumers (positive)
                    sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) +
                    # shunt        
                    sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) +
                    # acline
                    sum(stt[:acline_qfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                    sum(stt[:acline_qto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                    # xfm
                    sum(stt[:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                    sum(stt[:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
                    # dcline
                    sum(stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
                    sum(stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
                    # producer (negative)
                   -sum(stt[:dev_q][tii][idx.pr[bus]]; init=0.0)
        end

        # actual mismatch penalty
        stt[:zp][tii] = abs.(pb_slack)*cp*dt
        stt[:zq][tii] = abs.(qb_slack)*cq*dt

        # evaluate the grad?
        if qG.eval_grad
            if qG.pqbal_grad_mod_type == "standard"
                grd[:zp][:pb_slack][tii] = cp*dt*sign.(pb_slack)
                grd[:zq][:qb_slack][tii] = cq*dt*sign.(qb_slack)
            elseif qG.pqbal_grad_mod_type == "soft_abs"
                grd[:zp][:pb_slack][tii] = qG.pqbal_grad_mod_weight_p*dt*pb_slack./(sqrt.(pb_slack.^2 .+ qG.pqbal_grad_mod_eps2))
                grd[:zq][:qb_slack][tii] = qG.pqbal_grad_mod_weight_q*dt*qb_slack./(sqrt.(qb_slack.^2 .+ qG.pqbal_grad_mod_eps2))
            else
                println("not recognized!")
            end
        end
    end
end