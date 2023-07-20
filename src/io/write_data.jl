function prepare_solution(prm::quasiGrad.Param, stt::quasiGrad.State, sys::quasiGrad.System, qG::quasiGrad.QG)
    # clip once more, just to be safe
    qG.clip_pq_based_on_bins = true
    quasiGrad.clip_all!(prm, qG, stt)

    # prepare the solution dictionary
    soln_dict = Dict("time_series_output" => Dict("bus"                        => Array{Dict}(undef,sys.nb),
                                                  "shunt"                      => Array{Dict}(undef,sys.nsh),
                                                  "simple_dispatchable_device" => Array{Dict}(undef,sys.ndev),
                                                  "ac_line"                    => Array{Dict}(undef,sys.nl),
                                                  "two_winding_transformer"    => Array{Dict}(undef,sys.nx),
                                                  "dc_line"                    => Array{Dict}(undef,sys.nldc)))

    # now, populate each -- buses
    for ii in 1:sys.nb
        soln_dict["time_series_output"]["bus"][ii] = Dict("uid" => String(prm.bus.id[ii]),
                                                          "vm"  => Float64.([stt.vm[tii][ii] for tii in prm.ts.time_keys]),
                                                          "va"  => Float64.([stt.va[tii][ii] for tii in prm.ts.time_keys]))
    end

    # shunts
    for ii in 1:sys.nsh
        soln_dict["time_series_output"]["shunt"][ii] = Dict("uid"  => String(prm.shunt.id[ii]),
                                                            "step" => Int64.(round.([stt.u_step_shunt[tii][ii] for tii in prm.ts.time_keys])))
    end

    # simple_dispatchable_device
    for ii in prm.dev.dev_keys
        soln_dict["time_series_output"]["simple_dispatchable_device"][ii] = Dict(
        "uid"                     => String(prm.dev.id[ii]),
        "on_status"               => Int64.(round.([stt.u_on_dev[tii][ii]  for tii in prm.ts.time_keys])),
        "p_on"                    => Float64.([stt.p_on[tii][ii]      for tii in prm.ts.time_keys]),
        "q"                       => Float64.([stt.dev_q[tii][ii]     for tii in prm.ts.time_keys]),
        "p_reg_res_up"            => Float64.([stt.p_rgu[tii][ii]     for tii in prm.ts.time_keys]),
        "p_reg_res_down"          => Float64.([stt.p_rgd[tii][ii]     for tii in prm.ts.time_keys]),
        "p_syn_res"               => Float64.([stt.p_scr[tii][ii]     for tii in prm.ts.time_keys]),
        "p_nsyn_res"              => Float64.([stt.p_nsc[tii][ii]     for tii in prm.ts.time_keys]),
        "p_ramp_res_up_online"    => Float64.([stt.p_rru_on[tii][ii]  for tii in prm.ts.time_keys]),
        "p_ramp_res_down_online"  => Float64.([stt.p_rrd_on[tii][ii]  for tii in prm.ts.time_keys]),
        "p_ramp_res_up_offline"   => Float64.([stt.p_rru_off[tii][ii] for tii in prm.ts.time_keys]),
        "p_ramp_res_down_offline" => Float64.([stt.p_rrd_off[tii][ii] for tii in prm.ts.time_keys]),
        "q_res_up"                => Float64.([stt.q_qru[tii][ii]     for tii in prm.ts.time_keys]),
        "q_res_down"              => Float64.([stt.q_qrd[tii][ii]     for tii in prm.ts.time_keys]))
    end

    # ac_line
    for ii in 1:sys.nl
        soln_dict["time_series_output"]["ac_line"][ii] = Dict("uid"       => String(prm.acline.id[ii]),
                                                              "on_status" => Int64.(round.([stt.u_on_acline[tii][ii] for tii in prm.ts.time_keys])))
    end

    # two_winding_transformer
    for ii in 1:sys.nx
        soln_dict["time_series_output"]["two_winding_transformer"][ii] = Dict("uid"       => String(prm.xfm.id[ii]),
                                                                              "tm"        => Float64.([stt.tau[tii][ii] for tii in prm.ts.time_keys]),
                                                                              "ta"        => Float64.([stt.phi[tii][ii] for tii in prm.ts.time_keys]),
                                                                              "on_status" => Int64.(round.([stt.u_on_xfm[tii][ii] for tii in prm.ts.time_keys])))
    end

    # dc_line
    for ii in 1:sys.nldc
        soln_dict["time_series_output"]["dc_line"][ii] = Dict("uid"    => String(prm.dc.id[ii]),
                                                              "pdc_fr" => Float64.([stt.dc_pfr[tii][ii] for tii in prm.ts.time_keys]),
                                                              "qdc_fr" => Float64.([stt.dc_qfr[tii][ii] for tii in prm.ts.time_keys]),
                                                              "qdc_to" => Float64.([stt.dc_qto[tii][ii] for tii in prm.ts.time_keys]))
    end

    # output
    return soln_dict
end

# write the JSON
function write_solution(input_json_path::String, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)

    # prepare the solution dictionary
    soln_dict = quasiGrad.prepare_solution(prm, stt, sys, qG)

    # parse the input and then append
    if qG.write_location == "local"
        input_json       = replace(input_json_path, ".json" => "")
        output_json_path = input_json*"_solution_schev.json"
    else 
        @assert qG.write_location == "GO"
        # just write the folder in the cwd
        output_json_path = "solution.json"
    end

    # write to JSON
    open(output_json_path, "w") do io
        JSON.print(io, soln_dict)
    end

    # ...thank you, and goodnight. - Sam Chevalier
end

# post process
function post_process_stats(
    run::Bool,  
    bit::quasiGrad.Bit,  
    cgd::quasiGrad.Cgd, 
    ctb::Vector{Vector{Float64}},
    ctd::Vector{Vector{Float64}}, 
    flw::quasiGrad.Flow, 
    grd::quasiGrad.Grad, 
    idx::quasiGrad.Idx, 
    mgd::quasiGrad.Mgd, 
    msc::quasiGrad.Msc, 
    ntk::quasiGrad.Ntk, 
    prm::quasiGrad.Param, 
    qG::quasiGrad.QG, 
    scr::Dict{Symbol, Float64}, 
    stt::quasiGrad.State, 
    sys::quasiGrad.System, 
    wct::Vector{Vector{Int64}})

    # shall we actually post-process?
    if run == true
        # update the state vector
        qG.eval_grad      = false
        qG.score_all_ctgs = true
        quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
        
        # flop, just in case
        qG.eval_grad         = true
        qG.score_all_ctgs    = false
        qG.print_final_stats = true
        
        # print some stats?
        if qG.print_final_stats
            zms     = scr[:zms]
            zto     = scr[:zt_original]
            ztp     = scr[:zt_penalty]
            zb      = scr[:zbase]
            zenmax  = scr[:z_enmax]
            zenmin  = scr[:z_enmin]
            zhmxst  = qG.constraint_grad_weight*scr[:zhat_mxst]
            zctgmin = scr[:zctg_min]
            zctgavg = scr[:zctg_avg]

            println()
            println("====== ====== final output stats ====== ======")
            println(" • zms: $zms")
            println(" • zbase: $zb")
            println(" • zt (original): $zto")
            println(" • zt (penalty): $ztp")
            println(" • z (enmax): $zenmax")
            println(" • z (enmin): $zenmin")
            println(" • z (max starts): $zhmxst")
            println(" • z (ctg -- min): $zctgmin")
            println(" • z (ctg -- average): $zctgavg")
        end
    end
end