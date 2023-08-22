using quasiGrad
using Revise

# %% files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00014/scenario_003.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00073/scenario_002.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
#tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
#path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_997.json"

path = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"
solution_file = "C3E3N04224D1_scenario_131_solution.json"

path = tfp*"C3E3.1_20230629/D1/C3E3N08316D1/scenario_001.json"
#solution_file = "C3E3N08316D1_scenario_001_solution.json"

# %%
path = tfp*"C3E3.1_20230629/D1/C3E3N23643D1/scenario_003.json"
#solution_file = "C3E3N23643D1_scenario_003_solution.json"
#load_solve_project_write(path, solution_file)

# %%

path = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json" 
#path = tfp*"C3E3.1_20230629/D2/C3E3N00073D2/scenario_231.json"
#path = tfp*"C3E3.1_20230629/D3/C3E3N00073D3/scenario_231.json"

path = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
#path = tfp*"C3E3.1_20230629/D2/C3E3N00617D2/scenario_001.json"
#path = tfp*"C3E3.1_20230629/D3/C3E3N00617D3/scenario_001.json"

jsn  = quasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);

# %% ===========
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %%   
stt = deepcopy(stt0);
# for tii in prm.ts.time_keys
#     stt.va[tii] .= 0.0
# end

# %%


qG.lbfgs_adam_alpha_0 = 0.001
qG.initial_pf_lbfgs_step = 0.01
qG.max_linear_pfs = 3
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)


# %% adam!!!
stt = deepcopy(stt0);

qG.decay_adam_step             = true
qG.apply_grad_weight_homotopy  = true
qG.homotopy_with_cos_decay     = false
qG.num_threads                 = 10
qG.print_zms                   = true
qG.adam_max_time               = 150.0
qG.take_adam_pf_steps          = false
qG.beta1                       = 0.9
qG.beta2                       = 0.99
qG.ctg_solve_frequency         = 3
qG.always_solve_ctg            = false
qG.skip_ctg_eval               = true

qG.ctg_memory         = 0.15            
qG.one_min_ctg_memory = 1.0 - qG.ctg_memory

qG.pqbal_grad_weight_p = prm.vio.p_bus # standard: prm.vio.p_bus
qG.pqbal_grad_weight_q = prm.vio.q_bus # standard: prm.vio.q_bus

# qG.ctg_grad_weight         = prm.vio.s_flow
# qG.scale_c_sflow_testing   = 1.0

quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

@info "fix pbal weight, fix ctg ac and xfm flows!!"

println(qG.adm_step)



# %% ========================
    # choose adam step sizes (initial)
    vmva_scale_t0    = 1e-4
    xfm_scale_t0     = 1e-4
    dc_scale_t0      = 1e-1
    power_scale_t0   = 1e-1
    reserve_scale_t0 = 1e-1
    bin_scale_t0     = 1e-1  # bullish!!!
    qG.alpha_t0 = Dict(:vm    => vmva_scale_t0,
                   :va     => vmva_scale_t0,
                   # xfm
                   :phi    => xfm_scale_t0,
                   :tau    => xfm_scale_t0,
                   # dc
                   :dc_pfr => dc_scale_t0,
                   :dc_qto => dc_scale_t0,
                   :dc_qfr => dc_scale_t0,
                   # powers -- decay this!!
                   :dev_q  => power_scale_t0,
                   :p_on   => power_scale_t0,
                   # reserves
                   :p_rgu     => reserve_scale_t0,
                   :p_rgd     => reserve_scale_t0,
                   :p_scr     => reserve_scale_t0,
                   :p_nsc     => reserve_scale_t0,
                   :p_rrd_on  => reserve_scale_t0,
                   :p_rrd_off => reserve_scale_t0,
                   :p_rru_on  => reserve_scale_t0,
                   :p_rru_off => reserve_scale_t0,
                   :q_qrd     => reserve_scale_t0,
                   :q_qru     => reserve_scale_t0,
                   # bins
                   :u_on_xfm     => bin_scale_t0,
                   :u_on_dev     => bin_scale_t0,
                   :u_step_shunt => bin_scale_t0,
                   :u_on_acline  => bin_scale_t0)

        # choose adam step sizes (final)
        vmva_scale_tf    = 1e-7
        xfm_scale_tf     = 1e-7
        dc_scale_tf      = 1e-5
        power_scale_tf   = 1e-5
        reserve_scale_tf = 1e-5
        bin_scale_tf     = 1e-5 # bullish!!!
        qG.alpha_tf = Dict(:vm    => vmva_scale_tf,
                       :va     => vmva_scale_tf,
                       # xfm
                       :phi    => xfm_scale_tf,
                       :tau    => xfm_scale_tf,
                       # dc
                       :dc_pfr => dc_scale_tf,
                       :dc_qto => dc_scale_tf,
                       :dc_qfr => dc_scale_tf,
                       # powers -- decay this!!
                       :dev_q  => power_scale_tf,
                       :p_on   => power_scale_tf,
                       # reserves
                       :p_rgu     => reserve_scale_tf,
                       :p_rgd     => reserve_scale_tf,
                       :p_scr     => reserve_scale_tf,
                       :p_nsc     => reserve_scale_tf,
                       :p_rrd_on  => reserve_scale_tf,
                       :p_rrd_off => reserve_scale_tf,
                       :p_rru_on  => reserve_scale_tf,
                       :p_rru_off => reserve_scale_tf,
                       :q_qrd     => reserve_scale_tf,
                       :q_qru     => reserve_scale_tf,
                       # bins
                       :u_on_xfm     => bin_scale_tf,
                       :u_on_dev     => bin_scale_tf,
                       :u_step_shunt => bin_scale_tf,
                       :u_on_acline  => bin_scale_tf)


# %% -- test 1
t1 = time()
quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
tfl = time() - t1
println(tfl)

# %% -- test 2
qG.skip_ctg_eval    = false
qG.always_solve_ctg = true

t2 = time()
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
tup = time() - t2
println(tup)

# %%
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% ====================== 
#stt = deepcopy(stt0);
#quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
println(zp+zq)

# %%
qG.max_linear_pfs        = 1
quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

# %%
qG.initial_pf_lbfgs_step = 0.25
qG.num_lbfgs_steps  = 150

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)



# %% =================
# run copper plate ED
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# copy
stt0 = deepcopy(stt);

# %% ===
stt = deepcopy(stt0)
quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
println(zp+zq)

# %% stt0 = deepcopy(stt)
qG.num_threads = 10
stt = deepcopy(stt0)
qG.num_lbfgs_steps  = 15000

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
stt = deepcopy(stt0)
quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)


# %% initialize plot
# plt = Dict(:plot            => false,
#            :first_plot      => true,
#            :N_its           => 150,
#            :global_adm_step => 0,
#            :disp_freq       => 5)
# 
# initialize
# ax, fig, z_plt  = quasiGrad.initialize_plot(cgd, ctg, flw, grd, idx, mgd, ntk, plt, prm, qG, scr, stt, sys)

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

qG.pqbal_grad_eps2             = 1e-3
qG.constraint_grad_eps2        = 1e-3
qG.acflow_grad_eps2            = 1e-3
qG.reserve_grad_eps2           = 1e-3
qG.ctg_grad_eps2               = 1e-3
qG.adam_max_time               = 50.0

qG.decay_adam_step             = false
qG.apply_grad_weight_homotopy  = false
qG.homotopy_with_cos_decay     = false

qG.pqbal_grad_type     = "soft_abs"
qG.pqbal_grad_eps2     = 1e-7

qG.pqbal_quadratic_grad_weight_p = prm.vio.p_bus/(2.0*0.05)
qG.pqbal_quadratic_grad_weight_q = prm.vio.q_bus/(2.0*0.05)

vmva_scale_tf    = 1e-5
xfm_scale_tf     = 1e-5
dc_scale_tf      = 1e-5
power_scale_tf   = 1e-5
reserve_scale_tf = 1e-5
bin_scale_tf     = 1e-5 # bullish!!!

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
qG.print_zms   = true

quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
# ==========
#quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctg, fig, flw, grd, idx, mgd, ntk, plt, prm, qG, scr, stt, sys, upd, z_plt)

# %% ============
#pct_round = 0.0
#quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

# %%
using Plots

# for step size:
stp_0 = 1e-2
stp_f = 1e-7
ds    = stp_0/stp_f
lds   = log10(ds)

# the y-axis will span from 0 to -lds
x = -1:0.01:1
Plots.plot(x, -beta*lds .+ log10(stp_0))

# %%
Plots.plot(x, 10 .^ (-beta*lds .+ log10(stp_0)))

# %%

x = -1:0.01:1
beta = exp.(6.0*x)./(1.0 .+ exp.(6.0*x))
Plots.plot(x, 10.0 .* (1 .- beta) .- 10.0)

# %%
Plots.plot(x, 1e-3.*(1.0.-beta) .+ 1e-7.*beta)

#qG.alpha_t0[stp_key]*(1.0-beta) + qG.alpha_tf[stp_key]*beta

# %%
x = -1:0.01:1
beta  = exp.(5.0.*x)./(0.5 .+ exp.(5.0.*x))

# loop and compute the homotopy based on a log-transformation fall-off
#for stp_key in keys(qG.alpha_tnow)
    log_stp_ratio          = log10(1e-1/1e-4)
    stp = -beta.*log_stp_ratio .+ log10.(1e-1)
#end

Plots.plot(x, stp)

#Plots.plot(x, 10 .^ stp)

x       = -1:0.01:1
beta_p  = exp.(6.0.*x)./(1.0 .+ exp.(6.0.*x)) # penalty
beta_s  = exp.(5.0.*x)./(0.5 .+ exp.(5.0.*x)) # step

Plots.plot(x,exp.(5.0*x)./(1.0 .+ exp.(5.0*x))) # penalty
Plots.plot!(x,exp.(5.0*x)./(0.6 .+ exp.(5.0*x))) # step