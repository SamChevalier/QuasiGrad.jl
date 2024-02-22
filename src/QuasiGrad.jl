module QuasiGrad

# don't call Pkg, don't activate, and don't use Plots
# => using Pkg
# => Pkg.activate(".")

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
# don't call => using LimitedLDLFactorizations
# don't call => using Polyester

# call this first
include("./core/structs.jl")

# and the rest :)
include("./core/test.jl")
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
    try
        GRB_ENV[] = Gurobi.Env()
        return
    catch
        println("No dice with the Gurobi license.")
    end
end

# the following was used for pre-compilation during the GO comp -- not needed.
    #=
    # use PrecompileTools to precompile the functions which initialize the system --
    # use the precompile function for the rest.

    # "jsn" is given as type "Any" in many of the initialization functions, so the functions
    # are not fully inferable -- oh well.
    using PrecompileTools

    # set up the workload
    @compile_workload begin
        path = dirname(@__FILE__)*"\\precompile_617bus.json"
        # => path = "precompile_617bus.json"

        # call the jsn and initialize
        jsn = QuasiGrad.load_json(path)
        adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn)
        QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    end

    # directly precompile everthing else which is NOT a function of jsn -- move down alphabetically
    precompile(acline_flows!,(QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(adam!,(QuasiGrad.Adam, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(adam_pf!,(QuasiGrad.Adam, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(adam_step_decay!,(QuasiGrad.QG, Float64, Float64, Float64, Bool, Bool))
    precompile(adam_termination,(Float64, QuasiGrad.QG, Bool, Float64))
    precompile(all_device_statuses_and_costs!,(QuasiGrad.Grad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(apply_dev_p_grads!,(Int8, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.Index, QuasiGrad.State, QuasiGrad.Grad, QuasiGrad.MasterGrad, Int64, Float64))
    precompile(apply_dev_q_grads!,(Int8, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.Index, QuasiGrad.State, QuasiGrad.Grad, QuasiGrad.MasterGrad, Int64, Float64))
    precompile(apply_Gurobi_projection_and_states!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(apply_p_sd_grad!,(QuasiGrad.Index, Int8, Int64, Float64, QuasiGrad.Param, QuasiGrad.Grad, QuasiGrad.MasterGrad))
    precompile(apply_p_su_grad!,(QuasiGrad.Index, Int8, Int64, Float64, QuasiGrad.Param, QuasiGrad.Grad, QuasiGrad.MasterGrad))
    precompile(apply_q_injections!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(batch_fix!,(Float64, QuasiGrad.Param, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(build_Jac_and_pq0!,(QuasiGrad.Network, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Int8))
    precompile(build_Jac_sfr_and_sfr0!,(QuasiGrad.Index, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.State, QuasiGrad.System, Int8))
    precompile(build_Jac_sto!,(QuasiGrad.Network, QuasiGrad.State, QuasiGrad.System, Int8))
    precompile(call_adam_states,(QuasiGrad.Adam, QuasiGrad.MasterGrad, QuasiGrad.State, Symbol))
    precompile(cleanup_constrained_pf_with_Gurobi!,(QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(cleanup_constrained_pf_with_Gurobi_parallelized!,(QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
    precompile(cleanup_constrained_pf_with_Gurobi_parallelized_reserve_penalized!,(QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(clip_all!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(clip_dc!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(clip_for_adam_pf!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(clip_onoff_binaries!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(clip_pq!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(clip_reserves!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(clip_shunts!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(clip_voltage!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(clip_xfm!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(count_active_binaries!,(QuasiGrad.Param, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(dcpf_initialization!,(QuasiGrad.Flow, QuasiGrad.Index, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Bool))
    precompile(device_active_powers!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(device_reactive_powers!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(device_startup_states!,(QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(device_reserve_costs!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(dp_alpha!,(QuasiGrad.Grad, Int64, Int8, Float64))
    precompile(dq_alpha!,(QuasiGrad.Grad, Int64, Int8, Float64))
    precompile(du_sum!,(Int8, QuasiGrad.Param, QuasiGrad.State, QuasiGrad.MasterGrad, Int64, Float64, Vector{Int8}, Vector{Int8}))
    precompile(economic_dispatch_initialization!,(QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
    precompile(energy_costs!,(QuasiGrad.Grad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(energy_penalties!,(QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System))
    precompile(flush_adam!,(QuasiGrad.Adam, QuasiGrad.Flow, QuasiGrad.Param, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(flush_gradients!,(QuasiGrad.Grad, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.System))
    precompile(flush_lbfgs!,(QuasiGrad.LBFGS, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(get_sdpc,(Int8, Int64, QuasiGrad.Param))
    precompile(get_supc,(Int8, Int64, QuasiGrad.Param))
    precompile(get_tenmax,(Vector{Float64}, QuasiGrad.Param))
    precompile(get_tenmin,(Vector{Float64}, QuasiGrad.Param))
    precompile(get_tmindn,(Int8, Int64, QuasiGrad.Param))
    precompile(get_tminup,(Int8, Int64, QuasiGrad.Param))
    precompile(get_tsumax,(Vector{Float64}, QuasiGrad.Param))
    precompile(get_tsus_sets,(Int8, Int64, QuasiGrad.Param, Int64))
    precompile(initialize_ctg_lists!,(QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System))
    precompile(manage_time!,(Float64, QuasiGrad.QG))
    precompile(master_grad!,(QuasiGrad.ConstantGrad, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(master_grad_adam_pf!,(QuasiGrad.ConstantGrad, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.System))
    precompile(master_grad_solve_pf!,(QuasiGrad.ConstantGrad, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(master_grad_zp!,(Int8, QuasiGrad.Param, QuasiGrad.Index, QuasiGrad.Grad, QuasiGrad.MasterGrad, QuasiGrad.System, Bool))
    precompile(master_grad_zq!,(Int8, QuasiGrad.Param, QuasiGrad.Index, QuasiGrad.Grad, QuasiGrad.MasterGrad, QuasiGrad.System, Bool))
    precompile(master_grad_zs_acline!,(Int8, QuasiGrad.Index, QuasiGrad.Grad, QuasiGrad.MasterGrad, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(master_grad_zs_xfm!,(Int8, QuasiGrad.Index, QuasiGrad.Grad, QuasiGrad.MasterGrad, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(max_binary,(Int64, QuasiGrad.Index, Int64, QuasiGrad.State, Int8))
    precompile(max_power,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(penalized_device_constraints!,(QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System))
    precompile(post_process_stats,(Bool, QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System))
    precompile(power_balance!,(QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(pq_sums!,(Int64, QuasiGrad.Index, QuasiGrad.State, Int8))
    precompile(prepare_solution,(QuasiGrad.Param, QuasiGrad.State, QuasiGrad.System, QuasiGrad.QG))
    precompile(print_penalty_breakdown,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State))
    precompile(print_zms,(QuasiGrad.QG, Dict{Symbol, Float64}))
    precompile(project!,(Float64, QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
    precompile(quadratic_distance!,(QuasiGrad.LBFGS, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(reserve_balance!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(reserve_cleanup!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(run_adam!,(QuasiGrad.Adam, QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
    precompile(run_adam_pf!,(QuasiGrad.Adam, QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool, Bool))
    precompile(score_solve_pf!,(QuasiGrad.LBFGS, QuasiGrad.Param, QuasiGrad.State))
    precompile(score_zbase!,(QuasiGrad.QG, Dict{Symbol, Float64}))
    precompile(score_zms!,(Dict{Symbol, Float64},))
    precompile(score_zt!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State))
    precompile(shunts!,(QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(soft_abs,(Float64, Float64))
    precompile(soft_abs_acflow_grad,(Float64, QuasiGrad.QG))
    precompile(soft_abs_constraint_grad,(Float64, QuasiGrad.QG))
    precompile(soft_abs_ctg_grad,(Float64, QuasiGrad.QG))
    precompile(soft_abs_reserve_grad,(Float64, QuasiGrad.QG))
    precompile(solution_status,(QuasiGrad.Model,))
    precompile(solve_Gurobi_projection!,(Bool, QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(solve_parallel_linear_pf_with_Gurobi!,(QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG,  QuasiGrad.State, QuasiGrad.System, Bool, Bool))
    precompile(solve_parallel_linear_pf_with_Gurobi_23k!,(QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG,  QuasiGrad.State, QuasiGrad.System, Bool))
    precompile(solve_power_flow!,(QuasiGrad.ConstantGrad, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.LBFGS, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool, Bool))
    precompile(solve_power_flow_23k!,(QuasiGrad.Adam, QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.LBFGS, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool, Bool))
    precompile(simple_device_statuses_and_transposition!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(single_shot_pf_cleanup!,(QuasiGrad.Index, QuasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, QuasiGrad.Param, QuasiGrad.QG,  QuasiGrad.State, QuasiGrad.System, Int8))
    precompile(snap_shunts!,(Bool, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(soft_reserve_cleanup!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(solve_and_lowrank_update_single_ctg_gradient!,(QuasiGrad.Contingency, Int64, QuasiGrad.Network, QuasiGrad.QG, QuasiGrad.System, Int16))
    precompile(solve_ctgs!,(QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System))
    precompile(solve_economic_dispatch!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}, Bool))
    precompile(solve_LP_Gurobi_projection!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(solve_parallel_economic_dispatch!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(solve_pf_lbfgs!,(QuasiGrad.LBFGS, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, Dict{Symbol, Vector{Vector{Int64}}}))
    precompile(sum_power,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_p_nsc,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64)) 
    precompile(sum_p_rgd,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_p_rgu,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_p_rrd_off,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_p_rrd_on,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_p_rru_off,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_p_rru_on,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_p_scr,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_q_qrd,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(sum_q_qru,(QuasiGrad.Index, QuasiGrad.State, Int8, Int64))
    precompile(transpose_binaries!,(QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State))
    precompile(u_sum_sdpc,(Int64, QuasiGrad.Index, QuasiGrad.State, Int8))
    precompile(u_sum_supc,(Int64, QuasiGrad.Index, QuasiGrad.State, Int8))
    precompile(update_acline_sfr_flows!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.State, Int8))
    precompile(update_penalties!,(QuasiGrad.Param, QuasiGrad.QG, Float64, Float64, Float64))
    precompile(update_states_and_grads!,(QuasiGrad.ConstantGrad, QuasiGrad.Contingency, QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.QG, Dict{Symbol, Float64}, QuasiGrad.State, QuasiGrad.System, Bool))
    precompile(update_states_and_grads_for_adam_pf!,(QuasiGrad.ConstantGrad, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System, Bool))
    precompile(update_states_and_grads_for_solve_pf_lbfgs!,(QuasiGrad.ConstantGrad, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.LBFGS, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(update_xfm_sfr_flows!,(QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.State, Int8))
    precompile(update_Ybus!,(QuasiGrad.Index, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.State, QuasiGrad.System, Int8))
    precompile(update_Yflow!,(QuasiGrad.Index, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.State, QuasiGrad.System, Int8))
    precompile(wmi_update,(Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}))
    precompile(write_solution,(String, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(xfm_flows!,(QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.State, QuasiGrad.System))
    precompile(zctgs_grad_pinj!,(QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Network, QuasiGrad.Param, QuasiGrad.System, Int8))
    precompile(zctgs_grad_qfr_acline!,(QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.System, Int8))
    precompile(zctgs_grad_qto_acline!,(QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.System, Int8))
    precompile(zctgs_grad_qfr_xfm!,(QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.System, Int8))
    precompile(zctgs_grad_qto_xfm!,(QuasiGrad.Flow, QuasiGrad.Grad, QuasiGrad.Index, QuasiGrad.MasterGrad, QuasiGrad.Param, QuasiGrad.QG, QuasiGrad.System, Int8))
    =#
end
