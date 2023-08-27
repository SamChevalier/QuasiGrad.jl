module quasiGrad

# don't call Pkg, don't activate, and don't use Plots
    # using Pkg
    # Pkg.activate(".")

using JSON
using JuMP
using Gurobi
using Random
using Statistics
using SparseArrays
using LinearAlgebra
using InvertedIndices
using Preconditioners
using IterativeSolvers
using LoopVectorization

using BenchmarkTools
# don't call => using LimitedLDLFactorizations
# don't call => using Polyester

# call this first
include("./core/structs.jl")

# and the rest :)
include("./core/shunts.jl")
include("./core/devices.jl")
include("./core/cleanup.jl")
include("./io/read_data.jl")
include("./core/ac_flow.jl")
include("./core/scoring.jl")
include("./core/clipping.jl")
include("./io/write_data.jl")
include("./core/reserves.jl")
include("./core/homotopy.jl")
include("./scripts/solver.jl")
include("./core/projection.jl")
include("./core/power_flow.jl")
include("./core/master_grad.jl")
include("./core/optimization.jl")
include("./core/contingencies.jl")
include("./core/power_balance.jl")
include("./core/initializations.jl")
include("./core/economic_dispatch.jl")

# for plotting!
# => using GLMakie
# => using Makie
# => 
# => # include plotting tools
# => include("./core/plotting.jl")

# define static data constants
const eps_int    = 1e-8::Float64
const eps_time   = 1e-6::Float64
const eps_constr = 1e-8::Float64
const eps_beta   = 1e-6::Float64
const eps_susd   = 1e-6::Float64
const d_unit     = 5e-3::Float64

# define a gurobi licence: 
#   => https://github.com/jump-dev/Gurobi.jl/issues/424
const GRB_ENV = Ref{Gurobi.Env}()
function __init__()
    GRB_ENV[] = Gurobi.Env()
    return
end

# use PrecompileTools to precompile the functions which initialize the system --
# use the precompile function for th rest.

# "jsn" is given as type "Any" in many of the initialization functions, so the functions
# are not fully inferable -- oh well.
using PrecompileTools

# set up the workload
@compile_workload begin
    path = "./src/precompile_14bus.json"

    # call the jsn and initialize
    jsn = quasiGrad.load_json(path)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

end

# directly precompile everthing else which is NOT a function of jsn -- move down alphabetically
precompile(acline_flows!,(quasiGrad.Grad, quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(adam!,(quasiGrad.Adam, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(adam_pf!,(quasiGrad.Adam, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(adam_step_decay!,(quasiGrad.QG, Float64, Float64, Float64, Bool, Bool))
precompile(adam_termination,(Float64, quasiGrad.QG, Bool, Float64))
precompile(all_device_statuses_and_costs!,(quasiGrad.Grad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(apply_dev_p_grads!,(Int8, quasiGrad.Param, quasiGrad.QG, quasiGrad.Index, quasiGrad.State, quasiGrad.Grad, quasiGrad.MasterGrad, Union{Int32,Int64}, Float64))
precompile(apply_dev_q_grads!,(Int8, quasiGrad.Param, quasiGrad.QG, quasiGrad.Index, quasiGrad.State, quasiGrad.Grad, quasiGrad.MasterGrad, Union{Int32,Int64}, Float64))
precompile(apply_Gurobi_projection_and_states!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(apply_p_sd_grad!,(quasiGrad.Index, Int8, Union{Int32,Int64}, Float64, quasiGrad.Param, quasiGrad.Grad, quasiGrad.MasterGrad))
precompile(apply_p_su_grad!,(quasiGrad.Index, Int8, Union{Int32,Int64}, Float64, quasiGrad.Param, quasiGrad.Grad, quasiGrad.MasterGrad))
precompile(apply_q_injections!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(batch_fix!,(Float64, quasiGrad.Param, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(build_Jac_and_pq0!,(quasiGrad.Network, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Int8))
precompile(build_Jac_sfr_and_sfr0!,(quasiGrad.Index, quasiGrad.Network, quasiGrad.Param, quasiGrad.State, quasiGrad.System, Int8))
precompile(build_Jac_sto!,(quasiGrad.Network, quasiGrad.State, quasiGrad.System, Int8))
precompile(call_adam_states,(quasiGrad.Adam, quasiGrad.MasterGrad, quasiGrad.State, Symbol))
precompile(cleanup_constrained_pf_with_Gurobi!,(quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(clip_all!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(clip_dc!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(clip_for_adam_pf!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(clip_onoff_binaries!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(clip_pq!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(clip_reserves!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(clip_shunts!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(clip_voltage!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(clip_xfm!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(compute_quasiGrad_solution,(String, Float64, Int64, String, Int64))
precompile(count_active_binaries!,(quasiGrad.Param, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(dcpf_initialization!,(quasiGrad.Flow, quasiGrad.Index, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Bool))
precompile(device_active_powers!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(device_reactive_powers!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(device_startup_states!,(quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(device_reserve_costs!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(dp_alpha!,(quasiGrad.Grad, Union{Int32,Int64}, Int8, Float64))
precompile(dq_alpha!,(quasiGrad.Grad, Union{Int32,Int64}, Int8, Float64))
precompile(du_sum!,(Int8, quasiGrad.Param, quasiGrad.State, quasiGrad.MasterGrad, Union{Int32,Int64}, Float64, Vector{Int8}, Vector{Int8}))
precompile(economic_dispatch_initialization!,(quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
precompile(energy_costs!,(quasiGrad.Grad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(energy_penalties!,(quasiGrad.Grad, quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System))
precompile(flush_adam!,(quasiGrad.Adam, quasiGrad.Flow, quasiGrad.Param, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(flush_gradients!,(quasiGrad.Grad, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.System))
precompile(flush_lbfgs!,(quasiGrad.LBFGS, quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(get_sdpc,(Int8, Union{Int32,Int64}, quasiGrad.Param))
precompile(get_supc,(Int8, Union{Int32,Int64}, quasiGrad.Param))
precompile(get_tenmax,(Vector{Float64}, quasiGrad.Param))
precompile(get_tenmin,(Vector{Float64}, quasiGrad.Param))
precompile(get_tmindn,(Int8, Union{Int32,Int64}, quasiGrad.Param))
precompile(get_tminup,(Int8, Union{Int32,Int64}, quasiGrad.Param))
precompile(get_tsumax,(Vector{Float64}, quasiGrad.Param))
precompile(get_tsus_sets,(Int8, Union{Int32,Int64}, quasiGrad.Param, Int64))
precompile(initialize_ctg_lists!,(quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System))
precompile(manage_time!,(Float64, quasiGrad.QG))
precompile(master_grad!,(quasiGrad.ConstantGrad, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(master_grad_adam_pf!,(quasiGrad.ConstantGrad, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.System))
precompile(master_grad_solve_pf!,(quasiGrad.ConstantGrad, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(master_grad_zp!,(Int8, quasiGrad.Param, quasiGrad.Index, quasiGrad.Grad, quasiGrad.MasterGrad, quasiGrad.System, Bool))
precompile(master_grad_zq!,(Int8, quasiGrad.Param, quasiGrad.Index, quasiGrad.Grad, quasiGrad.MasterGrad, quasiGrad.System, Bool))
precompile(master_grad_zs_acline!,(Int8, quasiGrad.Index, quasiGrad.Grad, quasiGrad.MasterGrad, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(master_grad_zs_xfm!,(Int8, quasiGrad.Index, quasiGrad.Grad, quasiGrad.MasterGrad, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(max_binary,(Int32, quasiGrad.Index, Int64, quasiGrad.State, Int8))
precompile(max_power,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(penalized_device_constraints!,(quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System))
precompile(post_process_stats,(Bool, quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System))
precompile(power_balance!,(quasiGrad.Grad, quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(pq_sums!,(Int64, quasiGrad.Index, quasiGrad.State, Int8))
precompile(prepare_solution,(quasiGrad.Param, quasiGrad.State, quasiGrad.System, quasiGrad.QG))
precompile(print_penalty_breakdown,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State))
precompile(print_zms,(quasiGrad.QG, Dict{Symbol, Float64}))
precompile(project!,(Float64, quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
precompile(quadratic_distance!,(quasiGrad.LBFGS, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(reserve_balance!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(reserve_cleanup!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(run_adam!,(quasiGrad.Adam, quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(run_adam_pf!,(quasiGrad.Adam, quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
precompile(score_solve_pf!,(quasiGrad.LBFGS, quasiGrad.Param, quasiGrad.State))
precompile(score_zbase!,(quasiGrad.QG, Dict{Symbol, Float64}))
precompile(score_zms!,(Dict{Symbol, Float64},))
precompile(score_zt!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State))
precompile(shunts!,(quasiGrad.Grad, quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(soft_abs,(Float64, Float64))
precompile(soft_abs_acflow_grad,(Float64, quasiGrad.QG))
precompile(soft_abs_constraint_grad,(Float64, quasiGrad.QG))
precompile(soft_abs_ctg_grad,(Float64, quasiGrad.QG))
precompile(soft_abs_reserve_grad,(Float64, quasiGrad.QG))
precompile(solution_status,(quasiGrad.Model,))
precompile(solve_Gurobi_projection!,(Bool, quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(solve_parallel_linear_pf_with_Gurobi!,(quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG,  quasiGrad.State, quasiGrad.System, Bool, Bool))
precompile(solve_parallel_linear_pf_with_Gurobi_23k!,(quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG,  quasiGrad.State, quasiGrad.System))
precompile(solve_power_flow!,(quasiGrad.ConstantGrad, quasiGrad.Grad, quasiGrad.Index, quasiGrad.LBFGS, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool, Bool))
precompile(simple_device_statuses_and_transposition!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(single_shot_pf_cleanup!,(quasiGrad.Index, quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, quasiGrad.Param, quasiGrad.QG,  quasiGrad.State, quasiGrad.System, Int8))
precompile(snap_shunts!,(Bool, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(soft_reserve_cleanup!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(solve_and_lowrank_update_single_ctg_gradient!,(quasiGrad.Contingency, Int64, quasiGrad.Network, quasiGrad.QG, quasiGrad.System, Int16))
precompile(solve_ctgs!,(quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System))
precompile(solve_economic_dispatch!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
precompile(solve_LP_Gurobi_projection!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(solve_parallel_economic_dispatch!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(solve_pf_lbfgs!,(quasiGrad.LBFGS, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
precompile(sum_power,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_p_nsc,(quasiGrad.Index, quasiGrad.State, Int8, Int64)) 
precompile(sum_p_rgd,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_p_rgu,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_p_rrd_off,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_p_rrd_on,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_p_rru_off,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_p_rru_on,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_p_scr,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_q_qrd,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(sum_q_qru,(quasiGrad.Index, quasiGrad.State, Int8, Int64))
precompile(transpose_binaries!,(quasiGrad.Param, quasiGrad.QG, quasiGrad.State))
precompile(u_sum_sdpc,(Int32, quasiGrad.Index, quasiGrad.State, Int8))
precompile(u_sum_supc,(Int32, quasiGrad.Index, quasiGrad.State, Int8))
precompile(update_acline_sfr_flows!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.State, Int8))
precompile(update_penalties!,(quasiGrad.Param, quasiGrad.QG, Float64, Float64, Float64))
precompile(update_states_and_grads!,(quasiGrad.ConstantGrad, quasiGrad.Contingency, quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.QG, Dict{Symbol, Float64}, quasiGrad.State, quasiGrad.System))
precompile(update_states_and_grads_for_adam_pf!,(quasiGrad.ConstantGrad, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(update_states_and_grads_for_solve_pf_lbfgs!,(quasiGrad.ConstantGrad, quasiGrad.Grad, quasiGrad.Index, quasiGrad.LBFGS, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(update_xfm_sfr_flows!,(quasiGrad.Index, quasiGrad.Param, quasiGrad.State, Int8))
precompile(update_Ybus!,(quasiGrad.Index, quasiGrad.Network, quasiGrad.Param, quasiGrad.State, quasiGrad.System, Int8))
precompile(update_Yflow!,(quasiGrad.Index, quasiGrad.Network, quasiGrad.Param, quasiGrad.State, quasiGrad.System, Int8))
precompile(wmi_update,(Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}))
precompile(write_solution,(String, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(xfm_flows!,(quasiGrad.Grad, quasiGrad.Index, quasiGrad.Param, quasiGrad.QG, quasiGrad.State, quasiGrad.System))
precompile(zctgs_grad_pinj!,(quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Network, quasiGrad.Param, quasiGrad.System, Int8))
precompile(zctgs_grad_qfr_acline!,(quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.System, Int8))
precompile(zctgs_grad_qto_acline!,(quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.System, Int8))
precompile(zctgs_grad_qfr_xfm!,(quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.System, Int8))
precompile(zctgs_grad_qto_xfm!,(quasiGrad.Flow, quasiGrad.Grad, quasiGrad.Index, quasiGrad.MasterGrad, quasiGrad.Param, quasiGrad.QG, quasiGrad.System, Int8))

end
