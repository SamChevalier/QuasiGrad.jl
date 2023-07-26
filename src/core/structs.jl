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
    nT::Int64
    num_threads::Int64
    update_acline_xfm_bins::Bool
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
    time_keys::Vector{Int8}    
    tmin1::Vector{Int8}         
    duration::Vector{Float64}       
    start_time::Vector{Float64}     
    end_time::Vector{Float64}
end

struct Device
    device_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}
    id::Vector{String}
    bus::Vector{String}
    device_type::Vector{String}
    dev_keys::Vector{Int32}
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
    Ts_mndn::Vector{Vector{Vector{Int8}}}
    Ts_mnup::Vector{Vector{Vector{Int8}}}
    Ts_sdpc::Vector{Vector{Vector{Int8}}}
    ps_sdpc_set::Vector{Vector{Vector{Float64}}}
    Ts_supc::Vector{Vector{Vector{Int8}}}
    ps_supc_set::Vector{Vector{Vector{Float64}}}
    Ts_sus_jft::Vector{Vector{Vector{Vector{Int8}}}}
    Ts_sus_jf::Vector{Vector{Vector{Vector{Int8}}}}
    Ts_en_max::Vector{Vector{Vector{Int8}}}
    Ts_en_min::Vector{Vector{Vector{Int8}}}
    Ts_su_max::Vector{Vector{Vector{Int8}}}
end

# constant gradient terms -- precomputed for speed
struct Cgd
    ctg_avg::Vector{Float64}
    ctg_min::Vector{Float64}
    dzon_dev_du_on_dev::Vector{Vector{Float64}}
    dzt_dzen::Vector{Float64}
    dzrgu_dp_rgu::Vector{Vector{Float64}}
    dzrgd_dp_rgd::Vector{Vector{Float64}}
    dzscr_dp_scr::Vector{Vector{Float64}}
    dznsc_dp_nsc::Vector{Vector{Float64}}
    dzrru_dp_rru_on::Vector{Vector{Float64}}
    dzrru_dp_rru_off::Vector{Vector{Float64}}
    dzrrd_dp_rrd_on::Vector{Vector{Float64}}
    dzrrd_dp_rrd_off::Vector{Vector{Float64}}
    dzqru_dq_qru::Vector{Vector{Float64}}
    dzqrd_dq_qrd::Vector{Vector{Float64}}
    dzrgu_zonal_dp_rgu_zonal_penalty::Vector{Vector{Float64}}
    dzrgd_zonal_dp_rgd_zonal_penalty::Vector{Vector{Float64}}
    dzscr_zonal_dp_scr_zonal_penalty::Vector{Vector{Float64}}
    dznsc_zonal_dp_nsc_zonal_penalty::Vector{Vector{Float64}}
    dzrru_zonal_dp_rru_zonal_penalty::Vector{Vector{Float64}}
    dzrrd_zonal_dp_rrd_zonal_penalty::Vector{Vector{Float64}}
    dzqru_zonal_dq_qru_zonal_penalty::Vector{Vector{Float64}}
    dzqrd_zonal_dq_qrd_zonal_penalty::Vector{Vector{Float64}}
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

struct Flow
    ac_phi::Vector{Vector{Float64}}
    ac_qfr::Vector{Vector{Float64}}         
    ac_qto::Vector{Vector{Float64}}         
    qfr2::Vector{Vector{Float64}}           
    qto2::Vector{Vector{Float64}}           
    bt::Vector{Vector{Float64}}             
    dsmax_dp_flow::Vector{Vector{Float64}} 
    dsmax_dqfr_flow::Vector{Vector{Float64}}
    dsmax_dqto_flow::Vector{Vector{Float64}}
    pflow_k::Vector{Vector{Float64}}        
    sfr::Vector{Vector{Float64}}            
    sto::Vector{Vector{Float64}}            
    sfr_vio::Vector{Vector{Float64}}        
    sto_vio::Vector{Vector{Float64}}        
    p_inj::Vector{Vector{Float64}}          
    theta_k::Vector{Vector{Float64}}        
    rhs::Vector{Vector{Float64}}            
    dz_dpinj::Vector{Vector{Float64}}       
    dz_dpinj_all::Vector{Vector{Float64}}   
    c::Vector{Vector{Float64}}
end

struct State
    vm::Vector{Vector{Float64}}            
    va::Vector{Vector{Float64}}              
    acline_pfr::Vector{Vector{Float64}}      
    acline_qfr::Vector{Vector{Float64}}      
    acline_pto::Vector{Vector{Float64}}      
    acline_qto::Vector{Vector{Float64}}      
    u_on_acline::Vector{Vector{Float64}}     
    u_su_acline::Vector{Vector{Float64}}     
    u_sd_acline::Vector{Vector{Float64}}     
    phi::Vector{Vector{Float64}}            
    tau::Vector{Vector{Float64}}            
    xfm_pfr::Vector{Vector{Float64}}        
    xfm_qfr::Vector{Vector{Float64}}        
    xfm_pto::Vector{Vector{Float64}}        
    xfm_qto::Vector{Vector{Float64}}        
    u_on_xfm::Vector{Vector{Float64}}       
    u_su_xfm::Vector{Vector{Float64}}       
    u_sd_xfm::Vector{Vector{Float64}}           
    dc_pfr::Vector{Vector{Float64}}         
    dc_pto::Vector{Vector{Float64}}         
    dc_qfr::Vector{Vector{Float64}}         
    dc_qto::Vector{Vector{Float64}}         
    sh_p::Vector{Vector{Float64}}           
    sh_q::Vector{Vector{Float64}}           
    u_step_shunt::Vector{Vector{Float64}}   
    u_on_dev::Vector{Vector{Float64}}    
    u_on_dev_Trx::Vector{Vector{Float64}}
    dev_p::Vector{Vector{Float64}}       
    dev_q::Vector{Vector{Float64}}       
    u_on_dev_GRB::Vector{Vector{Float64}}  
    u_su_dev::Vector{Vector{Float64}}    
    u_su_dev_Trx::Vector{Vector{Float64}}  
    u_sd_dev::Vector{Vector{Float64}}  
    u_sd_dev_Trx::Vector{Vector{Float64}}  
    u_sum::Vector{Vector{Float64}}    
    u_sum_Trx::Vector{Vector{Float64}}
    p_on::Vector{Vector{Float64}}        
    p_su::Vector{Vector{Float64}}        
    p_sd::Vector{Vector{Float64}}        
    p_rgu::Vector{Vector{Float64}}       
    p_rgd::Vector{Vector{Float64}}
    p_scr::Vector{Vector{Float64}}       
    p_nsc::Vector{Vector{Float64}}       
    p_rru_on::Vector{Vector{Float64}}    
    p_rru_off::Vector{Vector{Float64}}   
    p_rrd_on::Vector{Vector{Float64}}    
    p_rrd_off::Vector{Vector{Float64}}   
    q_qru::Vector{Vector{Float64}}       
    q_qrd::Vector{Vector{Float64}}       
    zctg::Vector{Vector{Float64}}        
    zen_dev::Vector{Vector{Float64}}     
    zsu_dev::Vector{Vector{Float64}}      
    zsu_acline::Vector{Vector{Float64}}   
    zsu_xfm::Vector{Vector{Float64}}      
    zsd_dev::Vector{Vector{Float64}}      
    zsd_acline::Vector{Vector{Float64}}   
    zsd_xfm::Vector{Vector{Float64}}      
    zon_dev::Vector{Vector{Float64}}      
    zsus_dev::Vector{Vector{Float64}}     
    zs_acline::Vector{Vector{Float64}}    
    zs_xfm::Vector{Vector{Float64}}       
    zrgu::Vector{Vector{Float64}}         
    zrgd::Vector{Vector{Float64}}  
    zscr::Vector{Vector{Float64}}  
    znsc::Vector{Vector{Float64}}  
    zrru::Vector{Vector{Float64}}  
    zrrd::Vector{Vector{Float64}}  
    zqru::Vector{Vector{Float64}}  
    zqrd::Vector{Vector{Float64}}  
    zp::Vector{Vector{Float64}}    
    zq::Vector{Vector{Float64}}    
    zrgu_zonal::Vector{Vector{Float64}}  
    zrgd_zonal::Vector{Vector{Float64}}  
    zscr_zonal::Vector{Vector{Float64}}  
    znsc_zonal::Vector{Vector{Float64}}  
    zrru_zonal::Vector{Vector{Float64}}  
    zrrd_zonal::Vector{Vector{Float64}}  
    zqru_zonal::Vector{Vector{Float64}}  
    zqrd_zonal::Vector{Vector{Float64}}  
    p_rgu_zonal_REQ::Vector{Vector{Float64}}  
    p_rgd_zonal_REQ::Vector{Vector{Float64}}  
    p_scr_zonal_REQ::Vector{Vector{Float64}}  
    p_nsc_zonal_REQ::Vector{Vector{Float64}}  
    p_rgu_zonal_penalty::Vector{Vector{Float64}}  
    p_rgd_zonal_penalty::Vector{Vector{Float64}}  
    p_scr_zonal_penalty::Vector{Vector{Float64}}  
    p_nsc_zonal_penalty::Vector{Vector{Float64}}  
    p_rru_zonal_penalty::Vector{Vector{Float64}}  
    p_rrd_zonal_penalty::Vector{Vector{Float64}}  
    q_qru_zonal_penalty::Vector{Vector{Float64}}  
    q_qrd_zonal_penalty::Vector{Vector{Float64}}  
    zhat_mndn::Vector{Vector{Float64}}          
    zhat_mnup::Vector{Vector{Float64}}          
    zhat_rup::Vector{Vector{Float64}}           
    zhat_rd::Vector{Vector{Float64}}            
    zhat_rgu::Vector{Vector{Float64}}           
    zhat_rgd::Vector{Vector{Float64}}           
    zhat_scr::Vector{Vector{Float64}}           
    zhat_nsc::Vector{Vector{Float64}}           
    zhat_rruon::Vector{Vector{Float64}}         
    zhat_rruoff::Vector{Vector{Float64}}        
    zhat_rrdon::Vector{Vector{Float64}}         
    zhat_rrdoff::Vector{Vector{Float64}}        
    zhat_pmax::Vector{Vector{Float64}}          
    zhat_pmin::Vector{Vector{Float64}}          
    zhat_pmaxoff::Vector{Vector{Float64}}       
    zhat_qmax::Vector{Vector{Float64}}          
    zhat_qmin::Vector{Vector{Float64}}          
    zhat_qmax_beta::Vector{Vector{Float64}}     
    zhat_qmin_beta::Vector{Vector{Float64}}                 
end

struct Msc
    pinj_ideal::Vector{Vector{Float64}}      
    qinj_ideal::Vector{Vector{Float64}}      
    pb_slack::Vector{Vector{Float64}}        
    qb_slack::Vector{Vector{Float64}}        
    pub::Vector{Vector{Float64}}             
    plb::Vector{Vector{Float64}}             
    qub::Vector{Vector{Float64}}             
    qlb::Vector{Vector{Float64}}             
    pinj0::Vector{Vector{Float64}}           
    qinj0::Vector{Vector{Float64}}           
    pinj_dc::Vector{Vector{Float64}}         
    cos_ftp::Vector{Vector{Float64}}         
    sin_ftp::Vector{Vector{Float64}}         
    vff::Vector{Vector{Float64}}             
    vtt::Vector{Vector{Float64}}             
    vft::Vector{Vector{Float64}}             
    pfr::Vector{Vector{Float64}}             
    pto::Vector{Vector{Float64}}             
    qfr::Vector{Vector{Float64}}             
    qto::Vector{Vector{Float64}}               
    acline_sfr::Vector{Vector{Float64}}      
    acline_sto::Vector{Vector{Float64}}      
    acline_sfr_plus::Vector{Vector{Float64}} 
    acline_sto_plus::Vector{Vector{Float64}}
    # begin -- acline gradients --
    vmfrpfr::Vector{Vector{Float64}}
    vmtopfr::Vector{Vector{Float64}}
    vafrpfr::Vector{Vector{Float64}}
    vatopfr::Vector{Vector{Float64}}
    uonpfr::Vector{Vector{Float64}}
    vmfrqfr::Vector{Vector{Float64}}
    vmtoqfr::Vector{Vector{Float64}}
    vafrqfr::Vector{Vector{Float64}}
    vatoqfr::Vector{Vector{Float64}}
    uonqfr::Vector{Vector{Float64}}
    vmfrpto::Vector{Vector{Float64}}
    vmtopto::Vector{Vector{Float64}}
    vafrpto::Vector{Vector{Float64}}
    vatopto::Vector{Vector{Float64}}
    uonpto::Vector{Vector{Float64}}
    vmfrqto::Vector{Vector{Float64}}
    vmtoqto::Vector{Vector{Float64}}
    vafrqto::Vector{Vector{Float64}}
    vatoqto::Vector{Vector{Float64}}
    uonqto::Vector{Vector{Float64}}
    # end -- acline gradients --
    cos_ftp_x::Vector{Vector{Float64}}       
    sin_ftp_x::Vector{Vector{Float64}}       
    vff_x::Vector{Vector{Float64}}           
    vtt_x::Vector{Vector{Float64}}           
    vft_x::Vector{Vector{Float64}}           
    vt_tau_x::Vector{Vector{Float64}}        
    vf_tau_x::Vector{Vector{Float64}}        
    vf_tau2_x::Vector{Vector{Float64}}       
    vff_tau2_x::Vector{Vector{Float64}}      
    vft_tau_x::Vector{Vector{Float64}}       
    vft_tau2_x::Vector{Vector{Float64}}      
    vff_tau3_x::Vector{Vector{Float64}}      
    pfr_x::Vector{Vector{Float64}}           
    pto_x::Vector{Vector{Float64}}           
    qfr_x::Vector{Vector{Float64}}           
    qto_x::Vector{Vector{Float64}}           
    xfm_sfr_x::Vector{Vector{Float64}}       
    xfm_sto_x::Vector{Vector{Float64}}       
    xfm_sfr_plus_x::Vector{Vector{Float64}}  
    xfm_sto_plus_x::Vector{Vector{Float64}}
    # begin -- xfm gradients --
    vmfrpfr_x::Vector{Vector{Float64}}
    vmtopfr_x::Vector{Vector{Float64}}
    vafrpfr_x::Vector{Vector{Float64}}
    vatopfr_x::Vector{Vector{Float64}}
    taupfr_x::Vector{Vector{Float64}}
    phipfr_x::Vector{Vector{Float64}}
    uonpfr_x::Vector{Vector{Float64}}
    vmfrqfr_x::Vector{Vector{Float64}}
    vmtoqfr_x::Vector{Vector{Float64}}
    vafrqfr_x::Vector{Vector{Float64}}
    vatoqfr_x::Vector{Vector{Float64}}
    tauqfr_x::Vector{Vector{Float64}}
    phiqfr_x::Vector{Vector{Float64}}
    uonqfr_x::Vector{Vector{Float64}}
    vmfrpto_x::Vector{Vector{Float64}}
    vmtopto_x::Vector{Vector{Float64}}
    vafrpto_x::Vector{Vector{Float64}}
    vatopto_x::Vector{Vector{Float64}}
    taupto_x::Vector{Vector{Float64}}
    phipto_x::Vector{Vector{Float64}}
    uonpto_x::Vector{Vector{Float64}}
    vmfrqto_x::Vector{Vector{Float64}}
    vmtoqto_x::Vector{Vector{Float64}}
    vafrqto_x::Vector{Vector{Float64}}
    vatoqto_x::Vector{Vector{Float64}}
    tauqto_x::Vector{Vector{Float64}}
    phiqto_x::Vector{Vector{Float64}}
    uonqto_x::Vector{Vector{Float64}}
    # end -- xfm gradients -- 
    vm2_sh::Vector{Vector{Float64}}          
    g_tv_shunt::Vector{Vector{Float64}}      
    b_tv_shunt::Vector{Vector{Float64}}      
    u_sus_bnd::Vector{Vector{Vector{Float64}}}
    zsus_dev::Vector{Vector{Vector{Float64}}}
    zhat_mxst_scr::Vector{Float64}
    z_enmax_scr::Vector{Float64}
    z_enmin_scr::Vector{Float64}
end

# mini gradient structs
struct Acline_pfr
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    uon::Vector{Vector{Float64}} 
end

struct Acline_qfr
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    uon::Vector{Vector{Float64}} 
end

struct Acline_pto
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    uon::Vector{Vector{Float64}} 
end

struct Acline_qto
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    uon::Vector{Vector{Float64}} 
end

struct Zs_acline
    acline_pfr::Vector{Vector{Float64}}
    acline_qfr::Vector{Vector{Float64}}
    acline_pto::Vector{Vector{Float64}}
    acline_qto::Vector{Vector{Float64}}
end

struct Xfm_pfr
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    phi::Vector{Vector{Float64}} 
    tau::Vector{Vector{Float64}} 
    uon::Vector{Vector{Float64}} 
end

struct Xfm_qfr
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    phi::Vector{Vector{Float64}} 
    tau::Vector{Vector{Float64}} 
    uon::Vector{Vector{Float64}} 
end

struct Xfm_pto
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    phi::Vector{Vector{Float64}} 
    tau::Vector{Vector{Float64}} 
    uon::Vector{Vector{Float64}} 
end

struct Xfm_qto
    vmfr::Vector{Vector{Float64}}
    vmto::Vector{Vector{Float64}}
    vafr::Vector{Vector{Float64}}
    vato::Vector{Vector{Float64}}
    phi::Vector{Vector{Float64}} 
    tau::Vector{Vector{Float64}} 
    uon::Vector{Vector{Float64}} 
end

struct Zs_xfm
    xfm_pfr::Vector{Vector{Float64}}
    xfm_qfr::Vector{Vector{Float64}}
    xfm_pto::Vector{Vector{Float64}}
    xfm_qto::Vector{Vector{Float64}}
end

struct Sh_p
    vm::Vector{Vector{Float64}}        
    g_tv_shunt::Vector{Vector{Float64}}
end

struct Sh_q
    vm::Vector{Vector{Float64}}        
    b_tv_shunt::Vector{Vector{Float64}}
end

struct Zp
    pb_slack::Vector{Vector{Float64}}  
end

struct Zq
    qb_slack::Vector{Vector{Float64}}  
end

struct Zen_dev
    dev_p::Vector{Vector{Float64}}  
end

struct U_su_dev
    u_on_dev::Vector{Vector{Float64}}  
    u_on_dev_prev::Vector{Vector{Float64}}  
end

struct U_sd_dev
    u_on_dev::Vector{Vector{Float64}}  
    u_on_dev_prev::Vector{Vector{Float64}}  
end

struct U_su_acline
    u_on_acline::Vector{Vector{Float64}}       
    u_on_acline_prev::Vector{Vector{Float64}}  
end

struct U_sd_acline
    u_on_acline::Vector{Vector{Float64}}       
    u_on_acline_prev::Vector{Vector{Float64}}  
end

struct U_su_xfm
    u_on_xfm::Vector{Vector{Float64}}       
    u_on_xfm_prev::Vector{Vector{Float64}}  
end

struct U_sd_xfm
    u_on_xfm::Vector{Vector{Float64}}       
    u_on_xfm_prev::Vector{Vector{Float64}}  
end

struct Dx
    dp::Vector{Vector{Float64}}  
    dq::Vector{Vector{Float64}}  
end

struct Grad
    acline_pfr::Acline_pfr
    acline_qfr::Acline_qfr
    acline_pto::Acline_pto
    acline_qto::Acline_qto
    zs_acline::Zs_acline
    xfm_pfr::Xfm_pfr
    xfm_qfr::Xfm_qfr
    xfm_pto::Xfm_pto
    xfm_qto::Xfm_qto
    zs_xfm::Zs_xfm
    sh_p::Sh_p
    sh_q::Sh_q
    zp::Zp
    zq::Zq
    zen_dev::Zen_dev
    u_su_dev::U_su_dev
    u_sd_dev::U_sd_dev
    u_su_acline::U_su_acline
    u_sd_acline::U_sd_acline
    u_su_xfm::U_su_xfm
    u_sd_xfm::U_sd_xfm
    dx::Dx
end

struct Bit
    sfr_vio::Vector{BitVector}        
    sto_vio::Vector{BitVector}        
end

struct Mgd
    vm::Vector{Vector{Float64}}          
    va::Vector{Vector{Float64}}          
    tau::Vector{Vector{Float64}}         
    phi::Vector{Vector{Float64}}         
    dc_pfr::Vector{Vector{Float64}}      
    dc_qfr::Vector{Vector{Float64}}      
    dc_qto::Vector{Vector{Float64}}      
    u_on_acline::Vector{Vector{Float64}} 
    u_on_xfm::Vector{Vector{Float64}}    
    u_step_shunt::Vector{Vector{Float64}}
    u_on_dev::Vector{Vector{Float64}} 
    p_on::Vector{Vector{Float64}}     
    dev_q::Vector{Vector{Float64}}    
    p_rgu::Vector{Vector{Float64}}    
    p_rgd::Vector{Vector{Float64}}    
    p_scr::Vector{Vector{Float64}}    
    p_nsc::Vector{Vector{Float64}}    
    p_rru_on::Vector{Vector{Float64}} 
    p_rrd_on::Vector{Vector{Float64}} 
    p_rru_off::Vector{Vector{Float64}}
    p_rrd_off::Vector{Vector{Float64}}
    q_qru::Vector{Vector{Float64}}    
    q_qrd::Vector{Vector{Float64}}
end

struct MV
    m::Vector{Vector{Float64}}
    v::Vector{Vector{Float64}}
end

struct Adam
    keys::Vector{Symbol}
    vm::MV
    va::MV
    tau::MV
    phi::MV
    dc_pfr::MV
    dc_qfr::MV
    dc_qto::MV
    u_on_acline::MV
    u_on_xfm::MV
    u_step_shunt::MV
    u_on_dev::MV
    p_on::MV
    dev_q::MV
    p_rgu::MV
    p_rgd::MV
    p_scr::MV
    p_nsc::MV
    p_rru_on::MV
    p_rrd_on::MV
    p_rru_off::MV
    p_rrd_off::MV
    q_qru::MV
    q_qrd::MV
end