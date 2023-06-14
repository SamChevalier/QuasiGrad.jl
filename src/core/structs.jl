# record system size information
struct System
    nb::Int64    # number of buses
    nx::Int64    # number of xfms
    nsh::Int64   # number of shunts
    nl::Int64    # number of ac lines
    nac::Int64   # number of ac lines + xfms
    nldc::Int64  # number of dc lines
    ndev::Int64  # number of devices (produce + consume)
    ncs::Int64   # number of consumers
    npr::Int64   # number of producers
    nT::Int64    # number of time periods
    nvar::Int64  # number of variables in the variable stack
    nzP::Int64   # number of zones (active power)
    nzQ::Int64   # number of zones (reactive power)
    nctg::Int64  # number of contingencies
end

# record system size information
mutable struct QG
    print_projection_success::Bool
    print_reserve_cleanup_success::Bool
    compute_sus_on_each_iteration::Bool
    compute_sus_frequency::Int64
    pcts_to_round::Vector{Float64}
    cdist_psolve::Float64
    run_susd_updates::Bool
    grad_max::Float64
    adam_solve_times::Vector{Float64}
    write_location::String
    pg_tol::Float64
    eval_grad::Bool
    binary_projection_weight::Float64
    p_on_projection_weight::Float64
    dev_q_projection_weight::Float64
    print_final_stats::Bool
    FeasibilityTol::Float64
    IntFeasTol::Float64
    mip_gap::Float64
    time_lim::Float64
    print_zms::Bool
    print_freq::Int64
    scale_c_pbus_testing::Float64
    scale_c_qbus_testing::Float64
    scale_c_sflow_testing::Float64
    ctg_grad_cutoff::Float64
    score_all_ctgs::Bool
    min_buses_for_krylov::Int64
    frac_ctg_keep::Float64
    pcg_tol::Float64
    max_pcg_its::Int64
    grad_ctg_tol::Float64
    cutoff_level::Int64
    build_basecase_cholesky::Bool     
    accuracy_sparsify_lr_updates::Float64
    save_sparse_WMI_updates::Bool
    base_solver::String
    ctg_solver::String
    build_ctg_full::Bool
    build_ctg_lowrank::Bool
    eps::Float64
    beta1::Float64
    beta2::Float64
    alpha_0::Dict{Symbol, Float64}
    alpha_min::Float64
    alpha_max::Float64
    Ti::Int64
    step_decay::Float64
    decay_type::String
    plot_scale_up::Float64 
    plot_scale_dn::Float64
    adam_max_time::Float64 
    adam_max_its::Int64 
    adam_stopper::String  
    apply_grad_weight_homotopy::Bool
    pqbal_grad_type::String
    pqbal_grad_weight_p::Float64
    pqbal_grad_weight_q::Float64
    pqbal_grad_eps2::Float64
    constraint_grad_is_soft_abs::Bool 
    constraint_grad_weight::Float64
    constraint_grad_eps2::Float64
    acflow_grad_is_soft_abs::Bool
    acflow_grad_weight::Float64
    acflow_grad_eps2::Float64
    ctg_grad_is_soft_abs::Bool
    ctg_grad_weight::Float64
    ctg_grad_eps2::Float64
    reserve_grad_is_soft_abs::Bool
    reserve_grad_eps2::Float64
    compute_pf_injs_with_Jac::Bool
    max_pf_dx::Float64   
    max_pf_dx_final_solve::Float64
    max_linear_pfs::Int64
    max_linear_pfs_final_solve::Int64
    max_linear_pfs_total::Int64
    print_linear_pf_iterations::Bool
    Gurobi_pf_obj::String
    initialize_shunt_to_given_value::Bool
    initialize_vm_to_given_value::Bool
    include_energy_costs_lbfgs::Bool
    include_lbfgs_p0_regularization::Bool
    print_lbfgs_iterations::Bool
    initial_pf_lbfgs_step::Float64
    lbfgs_map_over_all_time::Bool
    num_lbfgs_steps::Int64
    clip_pq_based_on_bins::Bool
    first_qG_step::Bool
    first_qG_step_size::Float64
    skip_ctg_eval::Bool
    take_adam_pf_steps::Bool
    num_adam_pf_step::Int64
    adam_pf_variables::Vector{Symbol}
end

struct Shunt
    id::Vector{String}
    bus::Vector{String}
    gs::Vector{Float64}
    bs::Vector{Float64}
    step_ub::Vector{Float64}
    step_lb::Vector{Float64}
    init_step::Vector{Float64}
    shunt_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}
end

struct Dc
    dcline_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}
    fr_bus::Vector{String}
    to_bus::Vector{String}
    id::Vector{String}
    pdc_ub::Vector{Float64}
    qdc_fr_lb::Vector{Float64}
    qdc_to_lb::Vector{Float64}
    qdc_fr_ub::Vector{Float64}
    qdc_to_ub::Vector{Float64}
    init_pdc_fr::Vector{Float64}
    init_qdc_fr::Vector{Float64}
    init_qdc_to::Vector{Float64}
end

struct Ctg
    alpha::Float64      
    ctg_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}  
    id::Vector{String}         
    components::Vector{Vector{String}}
end

struct Bus
    bus_num::Vector{Int64}
    id::Vector{String}
    vm_ub::Vector{Float64}
    vm_lb::Vector{Float64}
    base_nom_volt::Vector{Float64}
    init_vm::Vector{Float64}
    init_va::Vector{Float64}
    active_rsvid::Vector{Vector{String}}
    reactive_rsvid::Vector{Vector{String}}
end

struct Xfm
    id::Vector{String}
    g_sr::Vector{Float64}
    b_sr::Vector{Float64}
    b_ch::Vector{Float64}
    g_fr::Vector{Float64}
    b_fr::Vector{Float64}
    g_to::Vector{Float64}
    b_to::Vector{Float64}
    tm_lb::Vector{Float64}
    tm_ub::Vector{Float64}
    ta_lb::Vector{Float64}
    ta_ub::Vector{Float64}
    init_on_status::Vector{Float64}
    init_tau::Vector{Float64}
    init_phi::Vector{Float64}
    mva_ub_em::Vector{Float64}
    mva_ub_nom::Vector{Float64}
    disconnection_cost::Vector{Float64}
    connection_cost::Vector{Float64}
    xfm_inds::Vector{Int64}
    fr_bus::Vector{String}
    to_bus::Vector{String}
    J_fpd::Vector{Int64}
    J_fwr::Vector{Int64}
end

struct Acline
    id::Vector{String}
    g_sr::Vector{Float64}
    b_sr::Vector{Float64}
    b_ch::Vector{Float64}
    g_fr::Vector{Float64}
    b_fr::Vector{Float64}
    g_to::Vector{Float64}
    b_to::Vector{Float64}
    init_on_status::Vector{Float64}
    mva_ub_em::Vector{Float64}
    mva_ub_nom::Vector{Float64}
    disconnection_cost::Vector{Float64}
    connection_cost::Vector{Float64}
    line_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}
    fr_bus::Vector{String}
    to_bus::Vector{String}
end

struct Reserve
    pzone_inds::Vector{Int64}
    qzone_inds::Vector{Int64}
    id_pzone::Vector{String}
    id_qzone::Vector{String}
    rgu_sigma::Vector{Float64}
    rgd_sigma::Vector{Float64}
    scr_sigma::Vector{Float64}
    nsc_sigma::Vector{Float64}
    rru_min::Vector{Vector{Float64}}
    rrd_min::Vector{Vector{Float64}}
    qru_min::Vector{Vector{Float64}}
    qrd_min::Vector{Vector{Float64}}
end

struct Violation
    p_bus::Float64
    q_bus::Float64
    s_flow::Float64
    e_dev::Float64
    rgu_zonal::Vector{Float64}
    rgd_zonal::Vector{Float64}
    scr_zonal::Vector{Float64}
    nsc_zonal::Vector{Float64}
    rru_zonal::Vector{Float64}
    rrd_zonal::Vector{Float64}
    qru_zonal::Vector{Float64}
    qrd_zonal::Vector{Float64}
end

struct Timeseries
    time_keys::Vector{Symbol}     
    tmin1::Dict{Symbol, Symbol}          
    duration::Dict{Symbol, Float64}       
    start_time::Vector{Float64}     
    end_time::Vector{Float64}
    start_time_dict::Dict{Symbol, Float64}
    end_time_dict::Dict{Symbol, Float64}  
    time_key_ind::Dict{Symbol, Int64}   
end

struct Device
    device_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}
    id::Vector{String}
    bus::Vector{String}
    device_type::Vector{String}
    startup_cost::Vector{Float64}
    startup_states::Vector{Vector{Vector{Float64}}}
    num_sus::Vector{Int64}
    shutdown_cost::Vector{Float64}
    startups_ub::Vector{Vector{Vector{Float64}}}
    num_mxst::Vector{Int64}
    energy_req_ub::Vector{Vector{Vector{Float64}}}
    energy_req_lb::Vector{Vector{Vector{Float64}}}
    num_W_enmax::Vector{Int64}
    num_W_enmin::Vector{Int64}
    on_cost::Vector{Float64}
    down_time_lb::Vector{Float64}
    in_service_time_lb::Vector{Float64}
    p_ramp_up_ub::Vector{Float64}
    p_ramp_down_ub::Vector{Float64}
    p_startup_ramp_ub::Vector{Float64}
    p_shutdown_ramp_ub::Vector{Float64}
    init_on_status::Vector{Float64}
    init_p::Vector{Float64}
    init_q::Vector{Float64}
    init_accu_down_time::Vector{Float64}
    init_accu_up_time::Vector{Float64}
    q_linear_cap::Vector{Int64}
    q_bound_cap::Vector{Int64}
    q_0::Vector{Float64}
    beta::Vector{Float64}
    q_0_ub::Vector{Float64}
    q_0_lb::Vector{Float64}
    beta_ub::Vector{Float64}
    beta_lb::Vector{Float64}
    J_pqe::Vector{Int64}
    J_pqmax::Vector{Int64}
    p_reg_res_up_ub::Vector{Float64}
    p_reg_res_down_ub::Vector{Float64}
    p_syn_res_ub::Vector{Float64}
    p_nsyn_res_ub::Vector{Float64}
    p_ramp_res_up_online_ub::Vector{Float64}
    p_ramp_res_down_online_ub::Vector{Float64}
    p_ramp_res_up_offline_ub::Vector{Float64}
    p_ramp_res_down_offline_ub::Vector{Float64}
    on_status_ub::Vector{Vector{Float64}}
    on_status_lb::Vector{Vector{Float64}}
    p_ub::Vector{Vector{Float64}}
    p_lb::Vector{Vector{Float64}}
    q_ub::Vector{Vector{Float64}}
    q_lb::Vector{Vector{Float64}}
    p_ub_tmdv::Vector{Vector{Float64}}
    p_lb_tmdv::Vector{Vector{Float64}}
    q_ub_tmdv::Vector{Vector{Float64}}
    q_lb_tmdv::Vector{Vector{Float64}}
    cost::Vector{Vector{Vector{Vector{Float64}}}}
    cum_cost_blocks::Vector{Vector{Vector{Vector{Float64}}}}
    p_reg_res_up_cost::Vector{Vector{Float64}}
    p_reg_res_down_cost::Vector{Vector{Float64}}
    p_syn_res_cost::Vector{Vector{Float64}}
    p_nsyn_res_cost::Vector{Vector{Float64}}
    p_ramp_res_up_online_cost::Vector{Vector{Float64}}
    p_ramp_res_down_online_cost::Vector{Vector{Float64}}
    p_ramp_res_up_offline_cost::Vector{Vector{Float64}}
    p_ramp_res_down_offline_cost::Vector{Vector{Float64}}
    q_res_up_cost::Vector{Vector{Float64}}
    q_res_down_cost::Vector{Vector{Float64}}
    p_reg_res_up_cost_tmdv::Vector{Vector{Float64}}           
    p_reg_res_down_cost_tmdv::Vector{Vector{Float64}}         
    p_syn_res_cost_tmdv::Vector{Vector{Float64}}              
    p_nsyn_res_cost_tmdv::Vector{Vector{Float64}}             
    p_ramp_res_up_online_cost_tmdv::Vector{Vector{Float64}}   
    p_ramp_res_down_online_cost_tmdv::Vector{Vector{Float64}} 
    p_ramp_res_up_offline_cost_tmdv::Vector{Vector{Float64}}  
    p_ramp_res_down_offline_cost_tmdv::Vector{Vector{Float64}}
    q_res_up_cost_tmdv::Vector{Vector{Float64}}               
    q_res_down_cost_tmdv::Vector{Vector{Float64}}             
end

# upper level struct
struct Param
    ts::Timeseries     
    dc::Dc     
    ctg::Ctg    
    bus::Bus    
    xfm::Xfm    
    vio::Violation 
    shunt::Shunt  
    acline::Acline 
    dev::Device
    reserve::Reserve
end

struct Idx
    acline_fr_bus::Vector{Int64}
    acline_to_bus::Vector{Int64}
    xfm_fr_bus::Vector{Int64}
    xfm_to_bus::Vector{Int64}
    dc_fr_bus::Vector{Int64}
    dc_to_bus::Vector{Int64}
    ac_line_flows::Vector{Int64}
    ac_xfm_flows::Vector{Int64}
    ac_phi::Vector{Int64}
    bus_is_acline_frs::Dict{Int64, Vector{Int64}}
    bus_is_acline_tos::Dict{Int64, Vector{Int64}}
    bus_is_xfm_frs::Dict{Int64, Vector{Int64}}
    bus_is_xfm_tos::Dict{Int64, Vector{Int64}}
    bus_is_dc_frs::Dict{Int64, Vector{Int64}}
    bus_is_dc_tos::Dict{Int64, Vector{Int64}}
    J_pqe::Vector{Int64}
    J_pqmax::Vector{Int64}
    J_pqmin::Vector{Int64}
    J_fpd::Vector{Int64}
    J_fwr::Vector{Int64}
    pr::Dict{Int64, Vector{Int64}}
    cs::Dict{Int64, Vector{Int64}}
    sh::Dict{Int64, Vector{Int64}}
    bus_to_pr_not_Jpqe::Dict{Int64, Vector{Int64}}
    bus_to_cs_not_Jpqe::Dict{Int64, Vector{Int64}}
    bus_to_pr_and_Jpqe::Dict{Int64, Vector{Int64}}
    bus_to_cs_and_Jpqe::Dict{Int64, Vector{Int64}}
    shunt_bus::Vector{Int64}
    pr_devs::Vector{Int64}
    cs_devs::Vector{Int64}
    pr_and_Jpqe::Vector{Int64}
    cs_and_Jpqe::Vector{Int64}
    pr_not_Jpqe::Vector{Int64}
    cs_not_Jpqe::Vector{Int64}
    device_to_bus::Vector{Int64}
    pr_pzone::Dict{Int64, Vector{Int64}}
    cs_pzone::Dict{Int64, Vector{Int64}}
    dev_pzone::Dict{Int64, Vector{Int64}}
    pr_qzone::Dict{Int64, Vector{Int64}}
    cs_qzone::Dict{Int64, Vector{Int64}}
    dev_qzone::Dict{Int64, Vector{Int64}}
    Ts_mndn::Vector{Vector{Vector{Symbol}}}
    Ts_mnup::Vector{Vector{Vector{Symbol}}} 
    Ts_sdpc::Vector{Vector{Vector{Symbol}}} 
    ps_sdpc_set::Vector{Vector{Vector{Float64}}} 
    Ts_supc::Vector{Vector{Vector{Symbol}}} 
    ps_supc_set::Vector{Vector{Vector{Float64}}} 
    Ts_sus_jft::Vector{Vector{Vector{Vector{Symbol}}}} 
    Ts_sus_jf::Vector{Vector{Vector{Vector{Symbol}}}} 
    Ts_en_max::Vector{Vector{Vector{Symbol}}} 
    Ts_en_min::Vector{Vector{Vector{Symbol}}} 
    Ts_su_max::Vector{Vector{Vector{Symbol}}}
end

# constant gradient terms -- precomputed for speed
struct Cgd
    ctg_avg::Dict{Symbol, Float64}
    ctg_min::Dict{Symbol, Float64}
    dzon_dev_du_on_dev::Dict{Symbol, Vector{Float64}}
    dzt_dzen::Vector{Float64}
    dzrgu_dp_rgu::Dict{Symbol, Vector{Float64}}
    dzrgd_dp_rgd::Dict{Symbol, Vector{Float64}}
    dzscr_dp_scr::Dict{Symbol, Vector{Float64}}
    dznsc_dp_nsc::Dict{Symbol, Vector{Float64}}
    dzrru_dp_rru_on::Dict{Symbol, Vector{Float64}}
    dzrru_dp_rru_off::Dict{Symbol, Vector{Float64}}
    dzrrd_dp_rrd_on::Dict{Symbol, Vector{Float64}}
    dzrrd_dp_rrd_off::Dict{Symbol, Vector{Float64}}
    dzqru_dq_qru::Dict{Symbol, Vector{Float64}}
    dzqrd_dq_qrd::Dict{Symbol, Vector{Float64}}
    dzrgu_zonal_dp_rgu_zonal_penalty::Dict{Symbol, Vector{Float64}}
    dzrgd_zonal_dp_rgd_zonal_penalty::Dict{Symbol, Vector{Float64}}
    dzscr_zonal_dp_scr_zonal_penalty::Dict{Symbol, Vector{Float64}}
    dznsc_zonal_dp_nsc_zonal_penalty::Dict{Symbol, Vector{Float64}}
    dzrru_zonal_dp_rru_zonal_penalty::Dict{Symbol, Vector{Float64}}
    dzrrd_zonal_dp_rrd_zonal_penalty::Dict{Symbol, Vector{Float64}}
    dzqru_zonal_dq_qru_zonal_penalty::Dict{Symbol, Vector{Float64}}
    dzqrd_zonal_dq_qrd_zonal_penalty::Dict{Symbol, Vector{Float64}}
end

struct Ntk
    s_max::Vector{Float64}
    E::SparseMatrixCSC{Int64, Int64}
    Er::SparseMatrixCSC{Int64, Int64}
    ErT::SparseMatrixCSC{Int64, Int64}
    Yb::SparseMatrixCSC{Float64, Int64}
    Ybr::SparseMatrixCSC{Float64, Int64}
    Yfr::SparseMatrixCSC{Float64, Int64}
    YfrT::SparseMatrixCSC{Float64, Int64}
    ctg_out_ind::Dict{Int64, Vector{Int64}}
    ctg_params::Dict{Int64, Vector{Float64}}
    Ybr_k::Dict{Int64, SparseMatrixCSC{Float64, Int64}}
    b::Vector{Float64}
    xfm_at_bus::Dict{Int64, Vector{Int64}}
    xfm_phi_scalars::Dict{Int64, Vector{Float64}}
    alpha::Float64
    components::Vector{Vector{String}}
    id::Vector{String}              
    ctg_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}    
    Ybr_Ch::Any # quasiGrad.SuiteSparse.CHOLMOD.Factor{Float64} #::Any   # quasiGrad.LinearAlgebra.Cholesky{Float64, Matrix{Float64}}   
    Ybr_ChPr::quasiGrad.Preconditioners.LimitedLDLFactorization{Float64, Int64, Vector{Int64}, Vector{Int64}}# Preconditioners.CholeskyPreconditioner{quasiGrad.Preconditioners.LimitedLDLFactorizations.LimitedLDLFactorization{Float64, Int64, Vector{Int64}, Vector{Int64}}} # Any # quasiGrad.Preconditioners.CholeskyPreconditioner{quasiGrad.Preconditioners.LimitedLDLFactorizations.LimitedLDLFactorization{Float64, Int64}}        
    u_k::Dict{Int64, Vector{Float64}} # Dict{Int64, SparseArrays.SparseVector{Float64, Int64}} # Dict{Int64, Vector{Float64}}             
    g_k::Dict{Int64, Float64}
    Ybus_acline_real::SparseArrays.SparseMatrixCSC{Float64, Int64}
    Ybus_acline_imag::SparseArrays.SparseMatrixCSC{Float64, Int64}
end