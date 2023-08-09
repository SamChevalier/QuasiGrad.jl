using quasiGrad
using Revise

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00014/scenario_003.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00073/scenario_002.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
jsn  = quasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);

# run copper plate ED
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# copy 
stt0 = deepcopy(stt)

# %% ===
# stt0 = deepcopy(stt)
stt = deepcopy(stt0)
# => quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
# => quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

#quasiGrad.flush_adam!(adm, prm, upd)
#var_key = :vm
#adam_states = getfield(adm, :vm)     
#state       = getfield(stt, :vm)
#grad        = getfield(mgd, :vm)
#tii = Int8(1)
#t1 = zeros(73)
#t2 = zeros(73)
#t3 = zeros(73)
#t4 = zeros(73)
#t5 = zeros(73)
#t6 = zeros(73)
#
#                 t2 .= qG.beta1.*adam_states.m[tii] .+ qG.one_min_beta1.*grad[tii]
#quasiGrad.@turbo t1 .= qG.beta1.*adam_states.m[tii] .+ qG.one_min_beta1.*grad[tii]
#
#                 t3 .= qG.beta2.*adam_states.v[tii] .+ qG.one_min_beta2.*(grad[tii].^2.0)
#quasiGrad.@turbo t4 .= qG.beta2.*adam_states.v[tii] .+ qG.one_min_beta2.*(grad[tii].^2.0)
#
#                 t5 .= state[tii] .- qG.alpha_tnow[var_key].*(adam_states.m[tii]./qG.one_min_beta1_decay)./(sqrt.(adam_states.v[tii]./qG.one_min_beta2_decay) .+ qG.eps)
#quasiGrad.@turbo t6 .= state[tii] .- qG.alpha_tnow[var_key].*(adam_states.m[tii]./qG.one_min_beta1_decay)./(sqrt.(adam_states.v[tii]./qG.one_min_beta2_decay) .+ qG.eps)
#
#
#@turbo adam_states.m[tii] .= qG.beta1.*adam_states.m[tii] .+ qG.one_min_beta1.*grad[tii]
#@turbo adam_states.v[tii] .= qG.beta2.*adam_states.v[tii] .+ qG.one_min_beta2.*(grad[tii].^2.0)
#@turbo state[tii]         .= state[tii] .- qG.alpha_tnow[var_key].*(adam_states.m[tii]./qG.one_min_beta1_decay)./(sqrt.(adam_states.v[tii]./qG.one_min_beta2_decay) .+ qG.eps)
#
#
#
# initialize plot
plt = Dict(:plot            => false,
           :first_plot      => true,
           :N_its           => 150,
           :global_adm_step => 0,
           :disp_freq       => 5)

# initialize
ax, fig, z_plt  = quasiGrad.initialize_plot(cgd, ctg, flw, grd, idx, mgd, ntk, plt, prm, qG, scr, stt, sys)

qG.print_freq                  = 10
qG.num_threads                 = 10
qG.apply_grad_weight_homotopy  = false
qG.take_adam_pf_steps          = false
#qG.pqbal_grad_type             = "soft_abs" 
#qG.pqbal_grad_eps2             = 1e-1

qG.constraint_grad_is_soft_abs = true
qG.acflow_grad_is_soft_abs     = true
qG.reserve_grad_is_soft_abs    = true
qG.skip_ctg_eval               = true

qG.beta1                       = 0.9
qG.beta2                       = 0.99

#qG.beta1                       = 0.98
#qG.beta2                       = 0.995

qG.pqbal_grad_eps2             = 1e-3
qG.constraint_grad_eps2        = 1e-3
qG.acflow_grad_eps2            = 1e-3
qG.reserve_grad_eps2           = 1e-3
qG.ctg_grad_eps2               = 1e-3
qG.adam_max_time               = 15.0
qG.apply_grad_weight_homotopy  = false
qG.decay_adam_step             = true

vmva_scale_tf    = 1e-2
xfm_scale_tf     = 1e-2
dc_scale_tf      = 1e-2
power_scale_tf   = 1e-2
reserve_scale_tf = 1e-2
bin_scale_tf     = 1e-2 # bullish!!!

qG.alpha_tnow[:vm]     = vmva_scale_tf
qG.alpha_tnow[:va]     = vmva_scale_tf
               # xfm
qG.alpha_tnow[:phi]    = xfm_scale_tf
qG.alpha_tnow[:tau]    = xfm_scale_tf
               # dc
qG.alpha_tnow[:dc_pfr] = dc_scale_tf
qG.alpha_tnow[:dc_qto] = dc_scale_tf
qG.alpha_tnow[:dc_qfr] = dc_scale_tf
               # powers -- decay this!!
qG.alpha_tnow[:dev_q]  = power_scale_tf
qG.alpha_tnow[:p_on]   = power_scale_tf
               # reserves
qG.alpha_tnow[:p_rgu]     = reserve_scale_tf
qG.alpha_tnow[:p_rgd]     = reserve_scale_tf
qG.alpha_tnow[:p_scr]     = reserve_scale_tf
qG.alpha_tnow[:p_nsc]     = reserve_scale_tf
qG.alpha_tnow[:p_rrd_on]  = reserve_scale_tf
qG.alpha_tnow[:p_rrd_off] = reserve_scale_tf
qG.alpha_tnow[:p_rru_on]  = reserve_scale_tf
qG.alpha_tnow[:p_rru_off] = reserve_scale_tf
qG.alpha_tnow[:q_qrd]     = reserve_scale_tf
qG.alpha_tnow[:q_qru]     = reserve_scale_tf
               # bins
qG.alpha_tnow[:u_on_xfm]     = bin_scale_tf
qG.alpha_tnow[:u_on_dev]     = bin_scale_tf
qG.alpha_tnow[:u_step_shunt] = bin_scale_tf
qG.alpha_tnow[:u_on_acline]  = bin_scale_tf

qG.num_threads = 10

quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctg, fig, flw, grd, idx, mgd, 
                                  ntk, plt, prm, qG, scr, stt, sys, upd, z_plt)

# %% ============
pct_round = 0.0
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)