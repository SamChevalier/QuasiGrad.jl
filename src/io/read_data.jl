# load the json
function load_json(json_file_path::String)
    json_data = Dict{String,Any}()
    open(json_file_path, "r") do io
        json_data = JSON.parse(io)
    end

    # output
    return json_data
end

# parse the json
function parse_json(json_data::Dict{String, Any})

    # build the system struct
    sys = QuasiGrad.build_sys(json_data)

    # parse the network elements
    dc_prm      = QuasiGrad.parse_json_dc(json_data)
    ctg_prm     = QuasiGrad.parse_json_ctg(json_data)
    bus_prm     = QuasiGrad.parse_json_bus(json_data)
    xfm_prm     = QuasiGrad.parse_json_xfm(json_data)
    shunt_prm   = QuasiGrad.parse_json_shunt(json_data)
    acline_prm  = QuasiGrad.parse_json_acline(json_data)
    device_prm  = QuasiGrad.parse_json_device(json_data)

    # parse violation, and then parse the reserve, which updates vio_prm
    reserve_prm, vio_prm = QuasiGrad.parse_json_reserve_and_vio(json_data)

    # read the time series data
    ts_prm = QuasiGrad.parse_json_timeseries(json_data)
    
    # join the parameter dicts
    prm = Param(
            ts_prm,
            dc_prm, 
            ctg_prm, 
            bus_prm, 
            xfm_prm, 
            vio_prm,
            shunt_prm, 
            acline_prm, 
            device_prm, 
            reserve_prm)

    # join the mappings into one idx
    idx = QuasiGrad.initialize_indices(prm, sys)

    # output
    return prm, idx, sys
end

# parse the json into dicts - ac lines
function parse_json_acline(json_data::Dict)
    # Alternative ways to call the dicts:
    #       N_lines = length(json_data["network"]["ac_line"])
    #       r_series = [json_data["network"]["ac_line"][i]["r"] for i in 1:N_lines]
    # or
    #       r_series = Float64.([ac_line["r"] for ac_line in values(json_data["network"]["ac_line"])])
    # the chosen method to idx through the dicts is no the fastest; however, it *safely* preserves order
    acline_inds      = keys(json_data["network"]["ac_line"])
    aclines          = json_data["network"]["ac_line"]
    naclines         = length(acline_inds)
    r_series         = Float64.([aclines[ind]["r"] for ind in acline_inds])
    x_series         = Float64.([aclines[ind]["x"] for ind in acline_inds])
    b_ch             = Float64.([aclines[ind]["b"] for ind in acline_inds])
    fr_bus           = [aclines[ind]["fr_bus"] for ind in acline_inds]
    to_bus           = [aclines[ind]["to_bus"] for ind in acline_inds]
    acline_id        = [aclines[ind]["uid"] for ind in acline_inds]
    additional_shunt = Bool.([aclines[ind]["additional_shunt"] for ind in acline_inds]) # findfirst(x -> x.=="C34",line_id)

    # limits and costs -- skip "mva_ub_sht" --> not required
    mva_ub_em          = Float64.([aclines[ind]["mva_ub_em"] for ind in acline_inds])
    mva_ub_nom         = Float64.([aclines[ind]["mva_ub_nom"] for ind in acline_inds])
    disconnection_cost = Float64.([aclines[ind]["disconnection_cost"] for ind in acline_inds])
    connection_cost    = Float64.([aclines[ind]["connection_cost"] for ind in acline_inds])
    
    # initialization
    init_status = Float64.([aclines[ind]["initial_status"]["on_status"] for ind in acline_inds])

    # add additional shunts
    shunt_lines        = findall(x -> x == true, additional_shunt)
    g_fr               = zeros(Float64,naclines)
    b_fr               = zeros(Float64,naclines)
    g_to               = zeros(Float64,naclines)
    b_to               = zeros(Float64,naclines)
    g_fr_nonzero       = Float64.([aclines[ind]["g_fr"] for ind in shunt_lines])
    b_fr_nonzero       = Float64.([aclines[ind]["b_fr"] for ind in shunt_lines])
    g_to_nonzero       = Float64.([aclines[ind]["g_to"] for ind in shunt_lines])
    b_to_nonzero       = Float64.([aclines[ind]["b_to"] for ind in shunt_lines])
    g_fr[shunt_lines] .= g_fr_nonzero
    b_fr[shunt_lines] .= b_fr_nonzero
    g_to[shunt_lines] .= g_to_nonzero
    b_to[shunt_lines] .= b_to_nonzero

    # compute line conductance and susceptance
    imp2 =  r_series.^2 + x_series.^2
    g_sr =  r_series./imp2
    b_sr = -x_series./imp2

    # setup outputs
    acline_param = Acline(
        acline_id,
        g_sr,
        b_sr,
        b_ch,
        g_fr,
        b_fr,
        g_to,
        b_to,
        init_status,
        mva_ub_em,
        mva_ub_nom,
        disconnection_cost,
        connection_cost,
        acline_inds,
        fr_bus,
        to_bus)

    #= acline_param = Dict(
        :id                 => acline_id,
        :g_sr               => g_sr,
        :b_sr               => b_sr,
        :b_ch               => b_ch,
        :g_fr               => g_fr,
        :b_fr               => b_fr,
        :g_to               => g_to,
        :b_to               => b_to,
        :init_on_status     => init_status,
        :mva_ub_em          => mva_ub_em,
        :mva_ub_nom         => mva_ub_nom,
        :disconnection_cost => disconnection_cost,
        :connection_cost    => connection_cost,
        :line_inds          => acline_inds,
        :fr_bus             => fr_bus,
        :to_bus             => to_bus)
        =#
    
    # output
    return acline_param
end

# parse the json into dicts - xfm
function parse_json_xfm(json_data::Dict)
    # call data
    xfm_inds         = Int64.(keys(json_data["network"]["two_winding_transformer"]))
    xfms             = json_data["network"]["two_winding_transformer"]
    nxfms            = length(xfm_inds)
    r_series         = Float64.([xfms[ind]["r"] for ind in xfm_inds])
    x_series         = Float64.([xfms[ind]["x"] for ind in xfm_inds])
    b_ch             = Float64.([xfms[ind]["b"] for ind in xfm_inds])
    fr_bus           = [xfms[ind]["fr_bus"] for ind in xfm_inds]
    to_bus           = [xfms[ind]["to_bus"] for ind in xfm_inds]
    xfm_id           = [xfms[ind]["uid"] for ind in xfm_inds]
    additional_shunt = Bool.([xfms[ind]["additional_shunt"] for ind in xfm_inds])

    # limits and costs -- skip "mva_ub_sht" --> not required
    tm_lb = Float64.([xfms[ind]["tm_lb"] for ind in xfm_inds])
    tm_ub = Float64.([xfms[ind]["tm_ub"] for ind in xfm_inds])
    ta_lb = Float64.([xfms[ind]["ta_lb"] for ind in xfm_inds])
    ta_ub = Float64.([xfms[ind]["ta_ub"] for ind in xfm_inds])

    mva_ub_em  = Float64.([xfms[ind]["mva_ub_em"] for ind in xfm_inds])
    mva_ub_nom = Float64.([xfms[ind]["mva_ub_nom"] for ind in xfm_inds])
    disconnection_cost = Float64.([xfms[ind]["disconnection_cost"] for ind in xfm_inds])
    connection_cost    = Float64.([xfms[ind]["connection_cost"] for ind in xfm_inds])

    # initialization
    init_status = Float64.([xfms[ind]["initial_status"]["on_status"] for ind in xfm_inds])
    init_tau    = Float64.([xfms[ind]["initial_status"]["tm"] for ind in xfm_inds])
    init_phi    = Float64.([xfms[ind]["initial_status"]["ta"] for ind in xfm_inds])

    # add additional shunts
    shunt_xfms         = findall(x -> x == true, additional_shunt)
    g_fr               = zeros(Float64,nxfms)
    b_fr               = zeros(Float64,nxfms)
    g_to               = zeros(Float64,nxfms)
    b_to               = zeros(Float64,nxfms)
    g_fr_nonzero       = [xfms[ind]["g_fr"] for ind in shunt_xfms]
    b_fr_nonzero       = [xfms[ind]["b_fr"] for ind in shunt_xfms]
    g_to_nonzero       = [xfms[ind]["g_to"] for ind in shunt_xfms]
    b_to_nonzero       = [xfms[ind]["b_to"] for ind in shunt_xfms]
    g_fr[shunt_xfms]  .= g_fr_nonzero
    b_fr[shunt_xfms]  .= b_fr_nonzero
    g_to[shunt_xfms]  .= g_to_nonzero
    b_to[shunt_xfms]  .= b_to_nonzero

    # compute xfm conductance and susceptance
    imp2 =  r_series.^2 + x_series.^2
    g_sr =  r_series./imp2
    b_sr = -x_series./imp2

    # get the xfm sets
    J_fwr = xfm_inds[isapprox.(tm_lb,tm_ub)]
    J_fpd = xfm_inds[QuasiGrad.Not(J_fwr)]

    # setup outputs
    xfm_param = Xfm(
        xfm_id,
        g_sr,
        b_sr,
        b_ch,
        g_fr,
        b_fr,
        g_to,
        b_to,
        tm_lb,
        tm_ub,
        ta_lb,
        ta_ub,
        init_status,
        init_tau,
        init_phi,
        mva_ub_em,
        mva_ub_nom,
        disconnection_cost,
        connection_cost,
        xfm_inds,
        fr_bus,
        to_bus,
        J_fpd,
        J_fwr)
        
    #= xfm_param = Dict(
        :id                 => xfm_id,
        :g_sr               => g_sr,
        :b_sr               => b_sr,
        :b_ch               => b_ch,
        :g_fr               => g_fr,
        :b_fr               => b_fr,
        :g_to               => g_to,
        :b_to               => b_to,
        :tm_lb              => tm_lb,
        :tm_ub              => tm_ub,
        :ta_lb              => ta_lb,
        :ta_ub              => ta_ub,
        :init_on_status     => init_status,
        :init_tau           => init_tau,
        :init_phi           => init_phi,
        :mva_ub_em          => mva_ub_em,
        :mva_ub_nom         => mva_ub_nom,
        :disconnection_cost => disconnection_cost,
        :connection_cost    => connection_cost,
        :xfm_inds           => xfm_inds,
        :fr_bus             => fr_bus,
        :to_bus             => to_bus,
        :J_fpd              => J_fpd,
        :J_fwr              => J_fwr)
        =#
    
    # output
    return xfm_param
end

# parse the json into dicts -- bus
function parse_json_bus(json_data::Dict)
    # call data
    bus_inds       = Int64.(keys(json_data["network"]["bus"]))
    buses          = json_data["network"]["bus"]
    bus_id         = [buses[ind]["uid"] for ind in bus_inds]
    vm_ub          = Float64.([buses[ind]["vm_ub"] for ind in bus_inds])
    vm_lb          = Float64.([buses[ind]["vm_lb"] for ind in bus_inds])
    base_nom_volt  = Float64.([buses[ind]["base_nom_volt"] for ind in bus_inds])
    active_rsvid   = Vector{Vector{String}}([buses[ind]["active_reserve_uids"] for ind in bus_inds])
    reactive_rsvid = Vector{Vector{String}}([buses[ind]["reactive_reserve_uids"] for ind in bus_inds])

    # initialization
    bus_init_vm         = Float64.([buses[ind]["initial_status"]["vm"] for ind in bus_inds])
    bus_init_va         = Float64.([buses[ind]["initial_status"]["va"] for ind in bus_inds])

    # We are now going to make an arbitrary decsion: each bus with "bus_id" is going
    # to be assigned a number. This number corresponds to its "bus_inds" value, i.e.,
    # the order in which it was read in the JSON. Let us now define a mapping between
    # the two via
        # bus_id_to_number(id) = Int64.(indexin(id, bus_id))
    
    # setup outputs
    bus_param = Bus(
        bus_inds,
        bus_id,
        vm_ub,
        vm_lb,
        base_nom_volt,
        bus_init_vm,
        bus_init_va,
        active_rsvid,
        reactive_rsvid)

    #=
    bus_param = Dict(
        :bus_num        => bus_inds,
        :id             => bus_id,
        :vm_ub          => vm_ub,
        :vm_lb          => vm_lb,
        :base_nom_volt  => base_nom_volt,
        :init_vm        => bus_init_vm,
        :init_va        => bus_init_va,
        :active_rsvid   => active_rsvid,
        :reactive_rsvid => reactive_rsvid)
    =#
    
    # output
    return bus_param
end

# parse the json into dicts -- shunt
function parse_json_shunt(json_data::Dict)
    # call data
    shunt_inds       = keys(json_data["network"]["shunt"])
    shunts           = json_data["network"]["shunt"]
    shunt_id         = [shunts[ind]["uid"] for ind in shunt_inds]
    bus              = [shunts[ind]["bus"] for ind in shunt_inds]
    gs               = Float64.([shunts[ind]["gs"] for ind in shunt_inds])
    bs               = Float64.([shunts[ind]["bs"] for ind in shunt_inds])
    step_ub          = Float64.([shunts[ind]["step_ub"] for ind in shunt_inds])
    step_lb          = Float64.([shunts[ind]["step_lb"] for ind in shunt_inds])
    init_step        = Float64.([shunts[ind]["initial_status"]["step"] for ind in shunt_inds])

    # setup outputs
    shunt_param = Shunt(
        shunt_id,
        bus,
        gs,
        bs,
        step_ub,
        step_lb,
        init_step,
        shunt_inds)
    #=
    shunt_param = Dict(
        :id         => shunt_id,
        :bus        => bus,
        :gs         => gs,
        :bs         => bs,
        :step_ub    => step_ub,
        :step_lb    => step_lb,
        :init_step  => init_step,
        :shunt_inds => shunt_inds)
        =#

    # output
    return shunt_param
end

# parse the json into dicts -- contingency
function parse_json_ctg(json_data::Dict)
    # call data
    ctg_inds      = keys(json_data["reliability"]["contingency"])
    ctg           = json_data["reliability"]["contingency"]
    ctg_id        = [ctg[ind]["uid"] for ind in ctg_inds]
    components    = Vector{String}.([ctg[ind]["components"] for ind in ctg_inds])
    alpha         = 1/length(json_data["network"]["bus"])
    
    # setup outputs
    ctg_param = Ctg_Prm(
        alpha,     
        ctg_inds,  
        ctg_id,        
        components)
    #=
    ctg_param = Dict(
        :alpha      => alpha,
        :ctg_inds   => ctg_inds, 
        :id         => ctg_id,
        :components => components)
        =#
    # output
    return ctg_param
end

# parse the json into dicts -- dc
function parse_json_dc(json_data::Dict)
    # call data
    dcline_inds      = keys(json_data["network"]["dc_line"])
    dcline           = json_data["network"]["dc_line"]
    fr_bus           = [dcline[ind]["fr_bus"] for ind in dcline_inds]
    to_bus           = [dcline[ind]["to_bus"] for ind in dcline_inds]
    dc_id            = [dcline[ind]["uid"]    for ind in dcline_inds]
    pdc_ub           = Float64.([dcline[ind]["pdc_ub"]    for ind in dcline_inds])
    qdc_fr_lb        = Float64.([dcline[ind]["qdc_fr_lb"] for ind in dcline_inds])
    qdc_to_lb        = Float64.([dcline[ind]["qdc_to_lb"] for ind in dcline_inds])
    qdc_fr_ub        = Float64.([dcline[ind]["qdc_fr_ub"] for ind in dcline_inds])
    qdc_to_ub        = Float64.([dcline[ind]["qdc_to_ub"] for ind in dcline_inds])
    init_pdc_fr      = Float64.([dcline[ind]["initial_status"]["pdc_fr"] for ind in dcline_inds])
    init_qdc_fr      = Float64.([dcline[ind]["initial_status"]["qdc_fr"] for ind in dcline_inds])
    init_qdc_to      = Float64.([dcline[ind]["initial_status"]["qdc_to"] for ind in dcline_inds])
    
    # setup outputs
    dc_param = Dc(
        dcline_inds,      
        fr_bus,
        to_bus,
        dc_id,
        pdc_ub,
        qdc_fr_lb,
        qdc_to_lb,
        qdc_fr_ub,
        qdc_to_ub,
        init_pdc_fr,
        init_qdc_fr,
        init_qdc_to)
    #=
    dc_param = Dict(
        :dcline_inds => dcline_inds,       
        :fr_bus      => fr_bus,
        :to_bus      => to_bus,
        :id          => dc_id,
        :pdc_ub      => pdc_ub,
        :qdc_fr_lb   => qdc_fr_lb,
        :qdc_to_lb   => qdc_to_lb,
        :qdc_fr_ub   => qdc_fr_ub,
        :qdc_to_ub   => qdc_to_ub,
        :init_pdc_fr => init_pdc_fr, 
        :init_qdc_fr => init_qdc_fr,
        :init_qdc_to => init_qdc_to)
        =#
        
    # output
    return dc_param
end

# parse the json into dicts -- device
function parse_json_device(json_data::Dict)
    # call data
    device_inds      = keys(json_data["network"]["simple_dispatchable_device"])
    device           = json_data["network"]["simple_dispatchable_device"]
    device_id        = [device[ind]["uid"] for ind in device_inds]
    bus              = [device[ind]["bus"] for ind in device_inds]
    device_type      = [device[ind]["device_type"] for ind in device_inds]
    ndev             = length(device_inds)
    dev_keys         = Int64.(1:ndev)

    # costs -- some of the time interval "ints" are set to "float"
    startup_cost       = Float64.([device[ind]["startup_cost"] for ind in device_inds])
    startup_states     = Vector{Vector{Float64}}.([device[ind]["startup_states"] for ind in device_inds])
    shutdown_cost      = Float64.([device[ind]["shutdown_cost"] for ind in device_inds])
    startups_ub        = Vector{Vector{Float64}}.([device[ind]["startups_ub"] for ind in device_inds])
    energy_req_ub      = Vector{Vector{Float64}}.([device[ind]["energy_req_ub"] for ind in device_inds])
    energy_req_lb      = Vector{Vector{Float64}}.([device[ind]["energy_req_lb"] for ind in device_inds])
    on_cost            = Float64.([device[ind]["on_cost"] for ind in device_inds])
    down_time_lb       = Float64.([device[ind]["down_time_lb"] for ind in device_inds])
    in_service_time_lb = Float64.([device[ind]["in_service_time_lb"] for ind in device_inds])

    # Reserve attributes
    p_ramp_up_ub       = Float64.([device[ind]["p_ramp_up_ub"] for ind in device_inds])
    p_ramp_down_ub     = Float64.([device[ind]["p_ramp_down_ub"] for ind in device_inds])
    p_startup_ramp_ub  = Float64.([device[ind]["p_startup_ramp_ub"] for ind in device_inds])
    p_shutdown_ramp_ub = Float64.([device[ind]["p_shutdown_ramp_ub"] for ind in device_inds])

    # initialization
    init_on_status      = Float64.([device[ind]["initial_status"]["on_status"] for ind in device_inds])
    init_p              = Float64.([device[ind]["initial_status"]["p"] for ind in device_inds])
    init_q              = Float64.([device[ind]["initial_status"]["q"] for ind in device_inds])
    init_accu_down_time = Float64.([device[ind]["initial_status"]["accu_down_time"] for ind in device_inds])
    init_accu_up_time   = Float64.([device[ind]["initial_status"]["accu_up_time"] for ind in device_inds])

    # extra parameterts
    q_linear_cap  = Bool.([device[ind]["q_linear_cap"] for ind in device_inds])
    q_bound_cap   = Bool.([device[ind]["q_bound_cap"] for ind in device_inds])
    iq_linear_cap = Int64.(q_linear_cap)
    iq_bound_cap  = Int64.(q_bound_cap)

    # conditional reactive devices
    q_linear_cap_true  = findall(x -> x == true, q_linear_cap)
    q_bound_cap_true   = findall(x -> x == true, q_bound_cap)

    # initialize
    q_0     = zeros(Float64,ndev)
    beta    = zeros(Float64,ndev)
    q_0_ub  = zeros(Float64,ndev)
    q_0_lb  = zeros(Float64,ndev)
    beta_ub = zeros(Float64,ndev)
    beta_lb = zeros(Float64,ndev)

    # populate
    q_0[q_linear_cap_true]    = Float64.([device[ind]["q_0"]     for ind in q_linear_cap_true])
    beta[q_linear_cap_true]   = Float64.([device[ind]["beta"]    for ind in q_linear_cap_true])
    q_0_ub[q_bound_cap_true]  = Float64.([device[ind]["q_0_ub"]  for ind in q_bound_cap_true])
    q_0_lb[q_bound_cap_true]  = Float64.([device[ind]["q_0_lb"]  for ind in q_bound_cap_true])
    beta_ub[q_bound_cap_true] = Float64.([device[ind]["beta_ub"] for ind in q_bound_cap_true])
    beta_lb[q_bound_cap_true] = Float64.([device[ind]["beta_lb"] for ind in q_bound_cap_true])

    # reserve attributes
    p_reg_res_up_ub            = Float64.([device[ind]["p_reg_res_up_ub"] for ind in device_inds]) 
    p_reg_res_down_ub          = Float64.([device[ind]["p_reg_res_down_ub"] for ind in device_inds])
    p_syn_res_ub               = Float64.([device[ind]["p_syn_res_ub"] for ind in device_inds])
    p_nsyn_res_ub              = Float64.([device[ind]["p_nsyn_res_ub"] for ind in device_inds])
    p_ramp_res_up_online_ub    = Float64.([device[ind]["p_ramp_res_up_online_ub"] for ind in device_inds])
    p_ramp_res_down_online_ub  = Float64.([device[ind]["p_ramp_res_down_online_ub"] for ind in device_inds])
    p_ramp_res_up_offline_ub   = Float64.([device[ind]["p_ramp_res_up_offline_ub"] for ind in device_inds])
    p_ramp_res_down_offline_ub = Float64.([device[ind]["p_ramp_res_down_offline_ub"] for ind in device_inds])

    # time series inputs -- device attributes
    ts_device_inds = keys(json_data["time_series_input"]["simple_dispatchable_device"])
    ts_device      = json_data["time_series_input"]["simple_dispatchable_device"]
    ts_device_id   = [ts_device[ind]["uid"] for ind in ts_device_inds]

    # now, for each device in "ts_device_uids", we need to get its index in "device_id"
    # wrong => ts_device_inds_adjusted = Int64.(indexin(ts_device_id, device_id))
    ts_device_inds_adjusted = Int64.(indexin(device_id, ts_device_id))

    on_status_ub = Vector{Float64}.([ts_device[ind]["on_status_ub"] for ind in ts_device_inds_adjusted])
    on_status_lb = Vector{Float64}.([ts_device[ind]["on_status_lb"] for ind in ts_device_inds_adjusted])
    p_ub         = Vector{Float64}.([ts_device[ind]["p_ub"]         for ind in ts_device_inds_adjusted])
    p_lb         = Vector{Float64}.([ts_device[ind]["p_lb"]         for ind in ts_device_inds_adjusted])
    q_ub         = Vector{Float64}.([ts_device[ind]["q_ub"]         for ind in ts_device_inds_adjusted])
    q_lb         = Vector{Float64}.([ts_device[ind]["q_lb"]         for ind in ts_device_inds_adjusted])
    cost         = Vector{Vector{Vector{Float64}}}.([ts_device[ind]["cost"] for ind in ts_device_inds_adjusted])

    # transpose for faster access
    p_ub_tmdv = Vector{Float64}.(collect(eachrow(reduce(hcat, p_ub))))    
    p_lb_tmdv = Vector{Float64}.(collect(eachrow(reduce(hcat, p_lb)))) 
    q_ub_tmdv = Vector{Float64}.(collect(eachrow(reduce(hcat, q_ub)))) 
    q_lb_tmdv = Vector{Float64}.(collect(eachrow(reduce(hcat, q_lb)))) 

    # time series inputs -- time series reserve attributes
    p_reg_res_up_cost            = Vector{Float64}.([ts_device[ind]["p_reg_res_up_cost"] for ind in ts_device_inds_adjusted])
    p_reg_res_down_cost          = Vector{Float64}.([ts_device[ind]["p_reg_res_down_cost"] for ind in ts_device_inds_adjusted])
    p_syn_res_cost               = Vector{Float64}.([ts_device[ind]["p_syn_res_cost"] for ind in ts_device_inds_adjusted]) 
    p_nsyn_res_cost              = Vector{Float64}.([ts_device[ind]["p_nsyn_res_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_up_online_cost    = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_online_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_down_online_cost  = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_online_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_up_offline_cost   = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_offline_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_down_offline_cost = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_offline_cost"] for ind in ts_device_inds_adjusted])
    q_res_up_cost                = Vector{Float64}.([ts_device[ind]["q_res_up_cost"] for ind in ts_device_inds_adjusted])
    q_res_down_cost              = Vector{Float64}.([ts_device[ind]["q_res_down_cost"] for ind in ts_device_inds_adjusted])

    # let's also get the transpose for faster access (tmdv = time; dev)
    p_reg_res_up_cost_tmdv            = Vector{Float64}.(collect(eachrow(reduce(hcat,p_reg_res_up_cost           ))))           
    p_reg_res_down_cost_tmdv          = Vector{Float64}.(collect(eachrow(reduce(hcat,p_reg_res_down_cost         ))))            
    p_syn_res_cost_tmdv               = Vector{Float64}.(collect(eachrow(reduce(hcat,p_syn_res_cost              ))))                 
    p_nsyn_res_cost_tmdv              = Vector{Float64}.(collect(eachrow(reduce(hcat,p_nsyn_res_cost             ))))                
    p_ramp_res_up_online_cost_tmdv    = Vector{Float64}.(collect(eachrow(reduce(hcat,p_ramp_res_up_online_cost   ))))      
    p_ramp_res_down_online_cost_tmdv  = Vector{Float64}.(collect(eachrow(reduce(hcat,p_ramp_res_down_online_cost ))))    
    p_ramp_res_up_offline_cost_tmdv   = Vector{Float64}.(collect(eachrow(reduce(hcat,p_ramp_res_up_offline_cost  ))))     
    p_ramp_res_down_offline_cost_tmdv = Vector{Float64}.(collect(eachrow(reduce(hcat,p_ramp_res_down_offline_cost))))   
    q_res_up_cost_tmdv                = Vector{Float64}.(collect(eachrow(reduce(hcat,q_res_up_cost               ))))                  
    q_res_down_cost_tmdv              = Vector{Float64}.(collect(eachrow(reduce(hcat,q_res_down_cost             ))))                

    # get the number of startup states for each device
    num_dev_sus = [length(dev_sus) for dev_sus in startup_states]

    # get the number of minimum/maximum energy reqs states for each device
    num_W_enmin = [length(dev_reqs) for dev_reqs in energy_req_lb]
    num_W_enmax = [length(dev_reqs) for dev_reqs in energy_req_ub]

    # get the number of maximum startup reqs for each device
    num_mxst = [length(startups) for startups in startups_ub]

    # update cost structures
    cum_cost_blocks = compute_cost_curves(cost,device_type)
    
    # setup outputs
    device_param = Device(
        device_inds,
        device_id,
        bus,
        device_type,
        dev_keys,
        startup_cost,
        startup_states,
        num_dev_sus,
        shutdown_cost,
        startups_ub,
        num_mxst,
        energy_req_ub,
        energy_req_lb,
        num_W_enmax,
        num_W_enmin,
        on_cost,
        down_time_lb,
        in_service_time_lb,
        p_ramp_up_ub,
        p_ramp_down_ub,
        p_startup_ramp_ub,
        p_shutdown_ramp_ub,
        init_on_status, 
        init_p,
        init_q,
        init_accu_down_time,
        init_accu_up_time,
        iq_linear_cap,
        iq_bound_cap,
        q_0,
        beta,
        q_0_ub,
        q_0_lb,
        beta_ub,
        beta_lb,
        q_linear_cap_true,
        q_bound_cap_true,
        p_reg_res_up_ub,
        p_reg_res_down_ub,
        p_syn_res_ub,
        p_nsyn_res_ub,
        p_ramp_res_up_online_ub,
        p_ramp_res_down_online_ub,
        p_ramp_res_up_offline_ub,
        p_ramp_res_down_offline_ub,
        on_status_ub,
        on_status_lb,
        p_ub,
        p_lb,
        q_ub,
        q_lb,
        p_ub_tmdv,
        p_lb_tmdv,
        q_ub_tmdv,
        q_lb_tmdv,
        cost,
        cum_cost_blocks,
        p_reg_res_up_cost,
        p_reg_res_down_cost,
        p_syn_res_cost,
        p_nsyn_res_cost,
        p_ramp_res_up_online_cost,
        p_ramp_res_down_online_cost,
        p_ramp_res_up_offline_cost,
        p_ramp_res_down_offline_cost,
        q_res_up_cost,
        q_res_down_cost,
        p_reg_res_up_cost_tmdv,           
        p_reg_res_down_cost_tmdv,         
        p_syn_res_cost_tmdv,              
        p_nsyn_res_cost_tmdv,             
        p_ramp_res_up_online_cost_tmdv,   
        p_ramp_res_down_online_cost_tmdv, 
        p_ramp_res_up_offline_cost_tmdv,  
        p_ramp_res_down_offline_cost_tmdv,
        q_res_up_cost_tmdv,               
        q_res_down_cost_tmdv)

    #= device_param = Dict(
        :device_inds                    => device_inds,
        :id                             => device_id,
        :bus                            => bus,
        :device_type                    => device_type,
        :startup_cost                   => startup_cost,
        :startup_states                 => startup_states,
        :num_sus                        => num_dev_sus,
        :shutdown_cost                  => shutdown_cost,
        :startups_ub                    => startups_ub,
        :num_mxst                       => num_mxst,
        :energy_req_ub                  => energy_req_ub,
        :energy_req_lb                  => energy_req_lb,
        :num_W_enmax                    => num_W_enmax,
        :num_W_enmin                    => num_W_enmin,
        :on_cost                        => on_cost,
        :down_time_lb                   => down_time_lb,
        :in_service_time_lb             => in_service_time_lb,
        :p_ramp_up_ub                   => p_ramp_up_ub,
        :p_ramp_down_ub                 => p_ramp_down_ub,
        :p_startup_ramp_ub              => p_startup_ramp_ub,
        :p_shutdown_ramp_ub             => p_shutdown_ramp_ub,
        :init_on_status                 => init_on_status, 
        :init_p                         => init_p,
        :init_q                         => init_q,
        :init_accu_down_time            => init_accu_down_time,
        :init_accu_up_time              => init_accu_up_time,
        :q_linear_cap                   => Int64.(q_linear_cap),
        :q_bound_cap                    => Int64.(q_bound_cap),
        :q_0                            => q_0,
        :beta                           => beta,
        :q_0_ub                         => q_0_ub,
        :q_0_lb                         => q_0_lb,
        :beta_ub                        => beta_ub,
        :beta_lb                        => beta_lb,
        :J_pqe                          => q_linear_cap_true,
        :J_pqmax                        => q_bound_cap_true,
        :p_reg_res_up_ub                => p_reg_res_up_ub,
        :p_reg_res_down_ub              => p_reg_res_down_ub,
        :p_syn_res_ub                   => p_syn_res_ub,
        :p_nsyn_res_ub                  => p_nsyn_res_ub,
        :p_ramp_res_up_online_ub        => p_ramp_res_up_online_ub,
        :p_ramp_res_down_online_ub      => p_ramp_res_down_online_ub,
        :p_ramp_res_up_offline_ub       => p_ramp_res_up_offline_ub,
        :p_ramp_res_down_offline_ub     => p_ramp_res_down_offline_ub,
        :on_status_ub                   => on_status_ub,
        :on_status_lb                   => on_status_lb,
        :p_ub                           => p_ub,
        :p_lb                           => p_lb,
        :q_ub                           => q_ub,
        :q_lb                           => q_lb,
        :cost                           => cost,
        :cum_cost_blocks                => cum_cost_blocks,
        :p_reg_res_up_cost              => p_reg_res_up_cost,
        :p_reg_res_down_cost            => p_reg_res_down_cost,
        :p_syn_res_cost                 => p_syn_res_cost,
        :p_nsyn_res_cost                => p_nsyn_res_cost,
        :p_ramp_res_up_online_cost      => p_ramp_res_up_online_cost,
        :p_ramp_res_down_online_cost    => p_ramp_res_down_online_cost,
        :p_ramp_res_up_offline_cost     => p_ramp_res_up_offline_cost,
        :p_ramp_res_down_offline_cost   => p_ramp_res_down_offline_cost,
        :q_res_up_cost                  => q_res_up_cost,
        :q_res_down_cost                => q_res_down_cost)
        =#

    # output
    return device_param
end

# parse the json into dicts -- reserve zones (P and Q)
function parse_json_reserve_and_vio(json_data::Dict)
    # parse the active power zone first
    pz       = json_data["network"]["active_zonal_reserve"]
    pz_inds  = Int64.(keys(pz))
    id_pzone = [pz[ind]["uid"] for ind in pz_inds]

    # parse
    REG_DOWN_vio_cost             = Float64.([pz[ind]["REG_DOWN_vio_cost"] for ind in pz_inds])
    SYN                           = Float64.([pz[ind]["SYN"] for ind in pz_inds])
    REG_UP                        = Float64.([pz[ind]["REG_UP"] for ind in pz_inds])
    RAMPING_RESERVE_DOWN_vio_cost = Float64.([pz[ind]["RAMPING_RESERVE_DOWN_vio_cost"] for ind in pz_inds])
    NSYN                          = Float64.([pz[ind]["NSYN"] for ind in pz_inds])
    RAMPING_RESERVE_UP_vio_cost   = Float64.([pz[ind]["RAMPING_RESERVE_UP_vio_cost"] for ind in pz_inds])
    SYN_vio_cost                  = Float64.([pz[ind]["SYN_vio_cost"] for ind in pz_inds])
    REG_UP_vio_cost               = Float64.([pz[ind]["REG_UP_vio_cost"] for ind in pz_inds])
    NSYN_vio_cost                 = Float64.([pz[ind]["NSYN_vio_cost"] for ind in pz_inds])
    REG_DOWN                      = Float64.([pz[ind]["REG_DOWN"] for ind in pz_inds])

    # parse the reactive power zone second
    qz       = json_data["network"]["reactive_zonal_reserve"]
    qz_inds  = Int64.(keys(qz))
    id_qzone = [qz[ind]["uid"] for ind in qz_inds]

    # parse
    REACT_UP_vio_cost   = Float64.([qz[ind]["REACT_UP_vio_cost"] for ind in qz_inds])
    REACT_DOWN_vio_cost = Float64.([qz[ind]["REACT_DOWN_vio_cost"] for ind in qz_inds])

    # we also need to parse some of the time series data to get
    # qru_min, qrd_min, rru_min, rrd_min:
    #
    # active power zone (time series) ====================================
    ts_pz       = json_data["time_series_input"]["active_zonal_reserve"]
    ts_pz_inds  = Int64.(keys(ts_pz))
    ts_id_pzone = [ts_pz[ind]["uid"] for ind in ts_pz_inds]

    # now, for each device in "ts_id_pzone", we need to get its index in "id_pzone"
    # wrong => ts_pz_inds_sorted = Int64.(indexin(ts_id_pzone, id_pzone))
    ts_pz_inds_sorted = Int64.(indexin(id_pzone, ts_id_pzone))

    # parse
    RAMPING_RESERVE_UP   = Vector{Float64}.([ts_pz[ind]["RAMPING_RESERVE_UP"]   for ind in ts_pz_inds_sorted])
    RAMPING_RESERVE_DOWN = Vector{Float64}.([ts_pz[ind]["RAMPING_RESERVE_DOWN"] for ind in ts_pz_inds_sorted])

    # reactive power zone (time series) ====================================
    ts_qz       = json_data["time_series_input"]["reactive_zonal_reserve"]
    ts_qz_inds  = Int64.(keys(ts_qz))
    ts_id_qzone = [ts_qz[ind]["uid"] for ind in ts_qz_inds]

    # now, for each device in "ts_id_qzone", we need to get its index in "id_qzone"
    # wrong => ts_qz_inds_sorted = Int64.(indexin(ts_id_qzone, id_qzone))
    ts_qz_inds_sorted = Int64.(indexin(id_qzone, ts_id_qzone))

    # parse
    REACT_UP   = Vector{Float64}.([ts_qz[ind]["REACT_UP"]   for ind in ts_qz_inds_sorted])
    REACT_DOWN = Vector{Float64}.([ts_qz[ind]["REACT_DOWN"] for ind in ts_qz_inds_sorted])

    # setup outputs -- reserves
    reserve_param = Reserve(
        pz_inds,
        qz_inds,
        id_pzone,
        id_qzone,
        REG_UP,
        REG_DOWN,
        SYN,
        NSYN,
        RAMPING_RESERVE_UP,
        RAMPING_RESERVE_DOWN,
        REACT_UP,
        REACT_DOWN)

    # violations
    pbv = Float64(json_data["network"]["violation_cost"]["p_bus_vio_cost"])
    qbv = Float64(json_data["network"]["violation_cost"]["q_bus_vio_cost"])
    sv  = Float64(json_data["network"]["violation_cost"]["s_vio_cost"])
    ev  = Float64(json_data["network"]["violation_cost"]["e_vio_cost"])
    
    # setup outputs -- violations    
    violation_param = Violation(
        pbv,
        qbv,
        sv,
        ev,
        REG_UP_vio_cost,
        REG_DOWN_vio_cost,
        SYN_vio_cost,
        NSYN_vio_cost,
        RAMPING_RESERVE_UP_vio_cost,
        RAMPING_RESERVE_DOWN_vio_cost,
        REACT_UP_vio_cost,
        REACT_DOWN_vio_cost)

    #= reserve_param = Dict(
        :pzone_inds  => pz_inds,
        :qzone_inds  => qz_inds,
        :id_pzone    => id_pzone,
        :id_qzone    => id_qzone,
        :rgu_sigma   => REG_UP,
        :rgd_sigma   => REG_DOWN,
        :scr_sigma   => SYN,
        :nsc_sigma   => NSYN,
        :rru_min     => RAMPING_RESERVE_UP,
        :rrd_min     => RAMPING_RESERVE_DOWN,
        :qru_min     => REACT_UP,
        :qrd_min     => REACT_DOWN)
        =# 
    
    # update the violation dict
    #=
    violation_param[:rgu_zonal] = REG_UP_vio_cost
    violation_param[:rgd_zonal] = REG_DOWN_vio_cost
    violation_param[:scr_zonal] = SYN_vio_cost
    violation_param[:nsc_zonal] = NSYN_vio_cost
    violation_param[:rru_zonal] = RAMPING_RESERVE_UP_vio_cost
    violation_param[:rrd_zonal] = RAMPING_RESERVE_DOWN_vio_cost
    violation_param[:qru_zonal] = REACT_UP_vio_cost
    violation_param[:qrd_zonal] = REACT_DOWN_vio_cost
    =#
    
    # output
    return reserve_param, violation_param
end

# parse the json into dicts -- time series
function parse_json_timeseries(json_data::Dict)
    # 1. general ####################### ####################### #######################
    #
    # this is a ORDERED vector of time keys
    nT              = Int64(json_data["time_series_input"]["general"]["time_periods"])
    time_keys       = Int8.(1:nT)
    prev_time_keys  = Int8.(0:nT-1)
    duration        = Float64.(json_data["time_series_input"]["general"]["interval_duration"])
    cum_time        = cumsum(duration)
    start_time      = [0; cum_time[1:end-1]]
    end_time        = cum_time
    #
    # 2. active_zonal_reserve -- dealt with in "reserve"
    #
    # 3. reactive_zonal_reserve -- dealt with in "reserve"
    # 
    # 4. simple_dispatchable_device -- dealt with in "device"
    # 
    # setup outputs direclty
    timeseries_param = Timeseries(
        time_keys,
        prev_time_keys,
        duration,
        start_time,
        end_time)

    # output
    return timeseries_param
end

# cost curves
function compute_cost_curves(cost::Vector{Vector{Vector{Vector{Float64}}}}, device_type::Vector{String})
    # for each device, for each time period, get the pr_p_cum_max
    nD              = length(cost)
    nT              = length(cost[1])
    nV              = 3
    cum_cost_blocks = [[[Vector{Float64}(undef, length(cost[dev][tii]) + 1) for vecs = 1:nV] for tii = 1:nT] for dev = 1:nD]
    # nV = 3, because we want to store 3 vectors (see below)
    for dev in 1:length(cost)
        for tii in 1:length(cost[dev])
            # note: while generally ordered correctly, the blocks
            # should be considered as an unordered set (per an email 
            # from Jesse), so let's make sure costs are ordered from
            # small to big for all devices -- then, for consumers,
            # we flip them around.
            #
            # gen offers: smaller  ->  medium  ->  higher         ***use this one
            # load bids:  higher  ->  medium  ->  smaller
            # 
            # probably, this code never actually runs
            if !issorted(round.(getindex.(cost[dev][tii],1), digits=4))
                # hmmm, not sorted -- let's fix:
                #     smaller  ->  medium  ->  higher
                pv = sortperm(getindex.(cost[dev][tii],1))
                cost[dev][tii] = cost[dev][tii][pv]

                # for permutation alert:
                # @info "device cost blocks permuted" 
                # println("dev: $dev")
                # println("time: $tii")
            end

            if device_type[dev] == "consumer"
                # in this case, we need to flip the cost blocks around!
                blocks  = reverse(cost[dev][tii])
            else
                blocks  = cost[dev][tii]
            end
            nb = length(cost[dev][tii])
            #                              [0 c1    c2    c3    ...]
            cum_cost_blocks[dev][tii][1] = append!([0.0], [blocks[block][1] for block in 1:nb])
                # cum_cost_blocks[dev][tii][1] = append!([0.0], [cost[dev][tii][block][1] for block in 1:nb])
            #                              [0 pmax1 pmax2 pmax3 ...]
            cum_cost_blocks[dev][tii][2] = append!([0.0], [blocks[block][2] for block in 1:nb])
                # cum_cost_blocks[dev][tii][2] = append!([0.0], [cost[dev][tii][block][2] for block in 1:nb])
            #                              [0 pmax1 pmax1+pmax2  pmax1+pmax2+pmax3 ...]
            cum_cost_blocks[dev][tii][3] = cumsum(cum_cost_blocks[dev][tii][2])

            # we pertub the final block upwards to protect against edge cases:
            # if p_dev == cum_cost_blocks[dev][tii][3][end], then the gradient
            # computer pushes us up into the next block, which does not exist!
            cum_cost_blocks[dev][tii][3][end] += 500.0
            # this has no affect on the system -- it just extend the final
            # block up higher than the generator can actually produce (see
            # eq. (312) for proof that this is a safe thing to do).
            #
            # Interpretation: "this would be the marginal cost if the generator
            # /load could produce/consumer up to an extra 100.0 units of power."
        end
    end

    # output
    return cum_cost_blocks
end