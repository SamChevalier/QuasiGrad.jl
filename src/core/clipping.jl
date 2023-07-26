function clip_all!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # sequentially clip -- order does not matter
    #
    # note: "clamp" is much faster than the alternatives!
    clip_dc!(prm, qG, stt)
    clip_xfm!(prm, qG, stt)
    clip_shunts!(prm, qG, stt)
    clip_voltage!(prm, qG, stt)
    clip_onoff_binaries!(prm, qG, stt)
    transpose_binaries!(prm, qG, stt)
    clip_reserves!(prm, qG, stt)

    # clip dev_p and dev_q after binaries, since p_on clipping MAY depend on u_on
    clip_pq!(prm, qG, stt)

    # sleep tasks
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
end

function clip_for_adam_pf!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # sequentially clip -- order does not matter
    #
    # note: "clamp" is much faster than the alternatives!
    clip_dc!(prm, qG, stt)
    clip_xfm!(prm, qG, stt)
    clip_shunts!(prm, qG, stt)
    clip_voltage!(prm, qG, stt)
end

function clip_dc!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys

        # clip all flows
            # stt.dc_qfr[tii] = min.(max.(stt.dc_qfr[tii],  prm.dc.qdc_fr_lb), prm.dc.qdc_fr_ub)
            # stt.dc_qto[tii] = min.(max.(stt.dc_qto[tii],  prm.dc.qdc_to_lb), prm.dc.qdc_to_ub)
            # stt.dc_qto[tii] = min.(max.(stt.dc_qto[tii], -prm.dc.pdc_ub ),    prm.dc.pdc_ub )
        stt.dc_qfr[tii] .= clamp.(stt.dc_qfr[tii],  prm.dc.qdc_fr_lb, prm.dc.qdc_fr_ub)
        stt.dc_qto[tii] .= clamp.(stt.dc_qto[tii],  prm.dc.qdc_to_lb, prm.dc.qdc_to_ub)
        stt.dc_pfr[tii] .= clamp.(stt.dc_pfr[tii], .-prm.dc.pdc_ub,    prm.dc.pdc_ub )

        # equate from and to end powers -- dc_pfr is the reference, or default
        stt.dc_pto[tii] .= .-stt.dc_pfr[tii]
    end
end

function clip_xfm!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # stt.phi[tii] = min.(max.(stt.phi[tii], prm.xfm.ta_lb), prm.xfm.ta_ub)
        # stt.tau[tii] = min.(max.(stt.tau[tii], prm.xfm.tm_lb), prm.xfm.tm_ub)
        stt.phi[tii] .= clamp.(stt.phi[tii], prm.xfm.ta_lb, prm.xfm.ta_ub)
        stt.tau[tii] .= clamp.(stt.tau[tii], prm.xfm.tm_lb, prm.xfm.tm_ub)
    end
end

function clip_shunts!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        #stt.u_step_shunt[tii] = min.(max.(stt.u_step_shunt[tii], prm.shunt.step_lb), prm.shunt.step_ub)
        stt.u_step_shunt[tii] .= clamp.(stt.u_step_shunt[tii], prm.shunt.step_lb, prm.shunt.step_ub)
    end
end

function clip_voltage!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # comute the deviation (0 if within bounds!)
        #del = min.(stt.vm[tii] - prm.bus.vm_lb  , 0.0) + max.(stt.vm[tii] - prm.bus.vm_ub, 0.0)
        #stt.vm[tii] = stt.vm[tii] - del
        # save the amount clipped? add to intialization!
            # -> stt.vm[:amount_clipped][tii] = del
        stt.vm[tii] .= clamp.(stt.vm[tii], prm.bus.vm_lb, prm.bus.vm_ub)
    end
end

function clip_reserves!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        stt.p_rgu[tii]     .= max.(stt.p_rgu[tii],0.0)
        stt.p_rgd[tii]     .= max.(stt.p_rgd[tii],0.0)
        stt.p_scr[tii]     .= max.(stt.p_scr[tii],0.0)
        stt.p_nsc[tii]     .= max.(stt.p_nsc[tii],0.0)
        stt.p_rru_on[tii]  .= max.(stt.p_rru_on[tii],0.0)
        stt.p_rru_off[tii] .= max.(stt.p_rru_off[tii],0.0)
        stt.p_rrd_on[tii]  .= max.(stt.p_rrd_on[tii],0.0)
        stt.p_rrd_off[tii] .= max.(stt.p_rrd_off[tii],0.0)
        stt.q_qru[tii]     .= max.(stt.q_qru[tii],0.0)
        stt.q_qrd[tii]     .= max.(stt.q_qrd[tii],0.0)
    end
end

function clip_onoff_binaries!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        stt.u_on_dev[tii]    .= clamp.(stt.u_on_dev[tii],    0.0, 1.0)
        stt.u_on_acline[tii] .= clamp.(stt.u_on_acline[tii], 0.0, 1.0)
        stt.u_on_xfm[tii]    .= clamp.(stt.u_on_xfm[tii],    0.0, 1.0)
        #
        # there is no need to clip startup or shutdown variables, since if
        # the on/off variables are clipped, then su and sd variables will 
        # necessarily be properly bounded.
        #
        # all good.
    end
end

# in the second-to-last iteration, we snap all shunts
function snap_shunts!(fix::Bool, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}})
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # clamp, to be safe
        stt.u_step_shunt[tii] .= clamp.(stt.u_step_shunt[tii], prm.shunt.step_lb, prm.shunt.step_ub)

        # now round -- no need for Int
        stt.u_step_shunt[tii] .= round.(stt.u_step_shunt[tii])

        # don't let adam make any more updates in this case
        if fix == true
            upd[:u_step_shunt][tii] = Int64[]
        end
    end
end

function clip_pq!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # qG.clip_pq_based_on_bins: should we clip p and q based on the current values of the binaries?
    #                           there are pros and cons to both decisions, so it is probably best
    #                           to alternate..
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # we also clip p_on, even though its value isn't explicitly set \ge 0
            # for justification, see (254) and (110)
            # note: stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii] \ge 0, since p_lb \ge 0
            # we also clip p_on to its maximum value (see 109)
        if qG.clip_pq_based_on_bins == true
            stt.p_on[tii] .= max.(stt.p_on[tii], stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii])
            stt.p_on[tii] .= min.(stt.p_on[tii], stt.u_on_dev[tii].*prm.dev.p_ub_tmdv[tii])
        else
            # watch out!! 
            # => the absolute bounds are given by 0 < p < p_ub, so this is how we clip!
            stt.p_on[tii] .= max.(stt.p_on[tii], 0.0)
            stt.p_on[tii] .= min.(stt.p_on[tii], prm.dev.p_ub_tmdv[tii])
        end

        # clip q -- we clip very simply based on (112), (113), (122), (123), where q_qru is negelcted!
        #
        if qG.clip_pq_based_on_bins == true
            stt.dev_q[tii] .= max.(stt.dev_q[tii], stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii])
            stt.dev_q[tii] .= min.(stt.dev_q[tii], stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii])
        else
            # watch out!! 
            # => the absolute bounds are given by q_lb < q < q_ub, but
            # we don't know if q_lb is positive or negative, so to include 0,
            # we take min.(q_lb, 0.0). same for upper bound -- I think :)
            stt.dev_q[tii] .= max.(stt.dev_q[tii], min.(prm.dev.q_lb_tmdv[tii], 0.0))
            stt.dev_q[tii] .= min.(stt.dev_q[tii], max.(prm.dev.q_ub_tmdv[tii], 0.0))
        end
    end
end

function count_active_binaries!(prm::quasiGrad.Param, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # how many binaries are still active?
    num_bin = sum([upd[:u_on_dev][tii]     != Int64[] for tii in prm.ts.time_keys])
    num_sh  = sum([upd[:u_step_shunt][tii] != Int64[] for tii in prm.ts.time_keys])

    # the following will error out if upd has active binaries or discrete values left
    @assert (num_bin+num_sh) == 0 "Some discrete or binary variables are still active!"
end

function transpose_binaries!(prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # across all devices and all times, compute the transposed on, su, sd and u_sum variables
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        for dev in prm.dev.dev_keys
            stt.u_on_dev_Trx[dev][tii] = stt.u_on_dev[tii][dev]
            stt.u_su_dev_Trx[dev][tii] = stt.u_su_dev[tii][dev]
            stt.u_sd_dev_Trx[dev][tii] = stt.u_sd_dev[tii][dev]
            stt.u_sum_Trx[dev][tii]    = stt.u_sum[tii][dev]
        end
    end
end