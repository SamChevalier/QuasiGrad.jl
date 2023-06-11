function power_balance!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, msc::Dict{Symbol, Vector{Float64}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # call penalty cost
    cp = prm.vio.p_bus * qG.scale_c_pbus_testing
    cq = prm.vio.q_bus * qG.scale_c_qbus_testing

    # note: msc[:pb_slack] and stt[:pq][:slack] are just
    #       endlessly overwritten

    # loop over each time period and compute the power balance
    for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]

        # loop over each bus and aggregate powers
        for bus in 1:sys.nb
            quasiGrad.pq_sums!(bus, idx, msc, stt, tii)
        end

        # actual mismatch penalty
        stt[:zp][tii] .= abs.(msc[:pb_slack]).*(cp*dt)
        stt[:zq][tii] .= abs.(msc[:qb_slack]).*(cq*dt)

        # evaluate the grad?
        if qG.eval_grad
            if qG.pqbal_grad_type == "standard"
                grd[:zp][:pb_slack][tii] .= (cp*dt).*sign.(msc[:pb_slack])
                grd[:zq][:qb_slack][tii] .= (cq*dt).*sign.(msc[:qb_slack])
            elseif qG.pqbal_grad_type == "soft_abs"
                grd[:zp][:pb_slack][tii] .= (qG.pqbal_grad_weight_p*dt).*msc[:pb_slack]./(sqrt.(msc[:pb_slack].^2 .+ qG.pqbal_grad_eps2))
                grd[:zq][:qb_slack][tii] .= (qG.pqbal_grad_weight_q*dt).*msc[:qb_slack]./(sqrt.(msc[:qb_slack].^2 .+ qG.pqbal_grad_eps2))
            elseif qG.pqbal_grad_type == "quadratic_for_lbfgs"
                grd[:zp][:pb_slack][tii] .= (cp*dt).*msc[:pb_slack]
                grd[:zq][:qb_slack][tii] .= (cp*dt).*msc[:qb_slack]
            else
                println("not recognized!")
            end
        end
    end
end

# fast sum
function pq_sums!(bus::Int64, idx::quasiGrad.Idx, msc::Dict{Symbol, Vector{Float64}}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, tii::Symbol)
    # loop over devices
    #
    msc[:pb_slack][bus] = 0.0
    msc[:qb_slack][bus] = 0.0

    # consumers -- positive
    for cs in idx.cs[bus]
        msc[:pb_slack][bus] += stt[:dev_p][tii][cs]
        msc[:qb_slack][bus] += stt[:dev_q][tii][cs]
    end

    # shunts -- positive
    for sh in idx.sh[bus]
        msc[:pb_slack][bus] += stt[:sh_p][tii][sh]
        msc[:qb_slack][bus] += stt[:sh_q][tii][sh]
    end

    # acline -- positive
    for acl in idx.bus_is_acline_frs[bus]
        msc[:pb_slack][bus] += stt[:acline_pfr][tii][acl]
        msc[:qb_slack][bus] += stt[:acline_qfr][tii][acl]
    end
    for acl in idx.bus_is_acline_tos[bus]
        msc[:pb_slack][bus] += stt[:acline_pto][tii][acl]
        msc[:qb_slack][bus] += stt[:acline_qto][tii][acl]
    end

    # xfm -- positive
    for xfm in idx.bus_is_xfm_frs[bus]
        msc[:pb_slack][bus] += stt[:xfm_pfr][tii][xfm]
        msc[:qb_slack][bus] += stt[:xfm_qfr][tii][xfm]
    end
    for xfm in idx.bus_is_xfm_tos[bus]
        msc[:pb_slack][bus] += stt[:xfm_pto][tii][xfm]
        msc[:qb_slack][bus] += stt[:xfm_qto][tii][xfm]
    end

    # dcline -- positive
    for dc in idx.bus_is_dc_frs[bus]
        msc[:pb_slack][bus] += stt[:dc_pfr][tii][dc]
        msc[:qb_slack][bus] += stt[:dc_qfr][tii][dc]
    end
    for dc in idx.bus_is_dc_tos[bus] 
        msc[:pb_slack][bus] += stt[:dc_pto][tii][dc]
        msc[:qb_slack][bus] += stt[:dc_qto][tii][dc]
    end

    # producer -- NEGATIVE
    for pr in idx.pr[bus]
        msc[:pb_slack][bus] -= stt[:dev_p][tii][pr]
        msc[:qb_slack][bus] -= stt[:dev_q][tii][pr]
    end
end

# correct the reactive power injections into the network
function correct_reactive_injections!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol)

    # warning
    @info "note: the reactive power correction function does NOT take J pqe into account yet"
    @warn "this was a good idea, but it doesn't really work -- replaced by power flow!"

    # loop over each time period
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # at this time, compute the pr and cs upper and lower bounds across all devices
        dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
        dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]
        # note: clipping is based on the upper/lower bounds, and not
        # based on the beta linking equations -- so, we just treat
        # that as a penalty, and not as a power balance factor
        # 
        # also, compute the dc line upper and lower bounds
        dcfr_qlb = prm.dc.qdc_fr_lb
        dcfr_qub = prm.dc.qdc_fr_ub
        dcto_qlb = prm.dc.qdc_to_lb
        dcto_qub = prm.dc.qdc_to_ub

        # how does balance work?
        # 0 = qb_slack + (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
        #
        # so, we take want to set:
        # -qb_slack = (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)

        for bus in 1:sys.nb
            # reactive power balance
            qb_slack = 
                    # shunt        
                    sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) +
                    # acline
                    sum(stt[:acline_qfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                    sum(stt[:acline_qto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                    # xfm
                    sum(stt[:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                    sum(stt[:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0)
                    # dcline -- not included
                    # consumers (positive) -- not included
                    # producer (negative) -- not included

            # get limits
            pr_lb   = sum(dev_qlb[idx.pr[bus]]; init=0.0)
            cs_lb   = sum(dev_qlb[idx.cs[bus]]; init=0.0)
            pr_ub   = sum(dev_qub[idx.pr[bus]]; init=0.0) 
            cs_ub   = sum(dev_qub[idx.cs[bus]]; init=0.0)
            dcfr_lb = sum(dcfr_qlb[idx.bus_is_dc_frs[bus]]; init=0.0)
            dcfr_ub = sum(dcfr_qub[idx.bus_is_dc_frs[bus]]; init=0.0)
            dcto_lb = sum(dcto_qlb[idx.bus_is_dc_tos[bus]]; init=0.0)
            dcto_ub = sum(dcto_qub[idx.bus_is_dc_tos[bus]]; init=0.0) 

            # total: lb < -qb_slack < ub
            ub = cs_ub + dcfr_ub + dcto_ub - pr_lb
            lb = cs_lb + dcfr_lb + dcto_lb - pr_ub

            # test
            if -qb_slack > ub
                println("ub limit")
                #assign = ub
                # max everything out
                stt[:dev_q][tii][idx.cs[bus]]             = dev_qub[idx.cs[bus]]
                stt[:dev_q][tii][idx.pr[bus]]             = dev_qlb[idx.pr[bus]]
                stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]] = dcfr_qub[idx.bus_is_dc_frs[bus]]
                stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]] = dcfr_qub[idx.bus_is_dc_tos[bus]]

            elseif -qb_slack < lb
                println("lb limit")
                #assign = ub
                # min everything out
                stt[:dev_q][tii][idx.cs[bus]]             = dev_qlb[idx.cs[bus]]
                stt[:dev_q][tii][idx.pr[bus]]             = dev_qub[idx.pr[bus]]
                stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]] = dcfr_qlb[idx.bus_is_dc_frs[bus]]
                stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]] = dcfr_qlb[idx.bus_is_dc_tos[bus]]

            else # in the middle -- all good
                println("middle")
                lb_dist  = -qb_slack - lb
                bnd_dist = ub - lb
                scale    = lb_dist/bnd_dist

                stt[:dev_q][tii][idx.cs[bus]]             = dev_qlb[idx.cs[bus]]             + scale*(dev_qub[idx.cs[bus]] - dev_qlb[idx.cs[bus]])
                stt[:dev_q][tii][idx.pr[bus]]             = dev_qub[idx.pr[bus]]             - scale*(dev_qub[idx.pr[bus]] - dev_qlb[idx.pr[bus]])
                stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]] = dcfr_qlb[idx.bus_is_dc_frs[bus]] + scale*(dcfr_qub[idx.bus_is_dc_frs[bus]] - dcfr_qlb[idx.bus_is_dc_frs[bus]])
                stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]] = dcfr_qlb[idx.bus_is_dc_tos[bus]] + scale*(dcfr_qub[idx.bus_is_dc_tos[bus]] - dcfr_qlb[idx.bus_is_dc_tos[bus]])
            end
        end
    end
end