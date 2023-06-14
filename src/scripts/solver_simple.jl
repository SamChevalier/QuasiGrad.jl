using quasiGrad
using GLMakie
using Revise
using Plots
using Makie

# call the plotting tools
# include("../core/plotting.jl")

# %% ===============
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D2/C3S1N00600/scenario_001.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D1/C3S0N00073/scenario_002.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E1_20230214/D1/C3E1N01576D1/scenario_117.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E1_20230214/D1/C3E1N01576D1/scenario_117.json"

# parameters
InFile1               = path
TimeLimitInSeconds    = 600.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# this is the master function which executes quasiGrad.
# 
#
# =====================================================\\
# TT: start time
start_time = time()

# I1. load the system data
jsn = quasiGrad.load_json(InFile1)

# %% I2. initialize the system
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, Div=Division);

@warn "homotopy ON"
qG.apply_grad_weight_homotopy = true

# %% I3. run an economic dispatch and update the states
@time quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)



# %% quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

#  E7. write the final solution
#quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# E8. post process
#quasiGrad.post_process_stats(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# TT: time
time_spent_before_loop = time() - start_time

# TT: how much time is left?
time_left = NewTimeLimitInSeconds - time_spent_before_loop

# TT: time management:
quasiGrad.manage_time!(time_left, qG)

# TT: set an adam solve time
qG.adam_max_time = 150.0

# L1. run power flow
quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# %%
quasiGrad.solve_linear_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

# %% L2. clean-up reserves by solving softly constrained LP
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

# L4. solve Gurobi projection
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)

# L5. fix binaries which are closest to their Gurobi solutions
quasiGrad.batch_fix!(pct_round, prm, stt, sys, upd)

# %% L6. update the state (i.e., apply the projection)
quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)

qG.max_pf_dx = 1e-2
quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

# %% save a copy
stt0 = deepcopy(stt)

# %% return!
stt = deepcopy(stt0)

# L3. run adam
qG.adam_max_time      = 25.0
qG.take_adam_pf_steps = false

# choose step sizes
vmva_scale    = 1e-6
xfm_scale     = 1e-5
dc_scale      = 1e-5
power_scale   = 1e-4
reserve_scale = 1e-4
bin_scale     = 1e-2

qG.alpha_0[:vm]           = vmva_scale
qG.alpha_0[:va]           = vmva_scale
qG.alpha_0[:phi]          = xfm_scale
qG.alpha_0[:tau]          = xfm_scale
qG.alpha_0[:dc_pfr]       = dc_scale
qG.alpha_0[:dc_qto]       = dc_scale
qG.alpha_0[:dc_qfr]       = dc_scale
qG.alpha_0[:dev_q]        = power_scale
qG.alpha_0[:p_on]         = power_scale
qG.alpha_0[:p_rgu]        = reserve_scale
qG.alpha_0[:p_rgd]        = reserve_scale
qG.alpha_0[:p_scr]        = reserve_scale
qG.alpha_0[:p_nsc]        = reserve_scale
qG.alpha_0[:p_rrd_on]     = reserve_scale
qG.alpha_0[:p_rrd_off]    = reserve_scale
qG.alpha_0[:p_rru_on]     = reserve_scale
qG.alpha_0[:p_rru_off]    = reserve_scale
qG.alpha_0[:q_qrd]        = reserve_scale
qG.alpha_0[:q_qru]        = reserve_scale
qG.alpha_0[:u_on_xfm]     = bin_scale
qG.alpha_0[:u_on_dev]     = bin_scale
qG.alpha_0[:u_step_shunt] = bin_scale
qG.alpha_0[:u_on_acline]  = bin_scale

quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# %% Plots
sum(stt[:u_on_dev][:t7])



# Plots.plot(stt[:u_on_dev][:t1])

# %% ===================
#=
tii   = :t1
alpha = 2e-6

adm[:u_on_dev][:m][tii][498] = beta1.*adm[:u_on_dev][:m][tii][498] + (1.0-beta1).*mgd[:u_on_dev][:t1][498]
adm[:u_on_dev][:v][tii][498] = beta2.*adm[:u_on_dev][:v][tii][498] + (1.0-beta2).*mgd[:u_on_dev][:t1][498].^2.0
stt[:u_on_dev][tii]          = stt[:u_on_dev][tii][498] - alpha*(adm[:u_on_dev][:m][tii][498])./(sqrt.(adm[:u_on_dev][:v][tii][498]) .+ qG.eps)
=#

# %% ================================
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D2/C3S1N00600/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D1/C3S0N00073/scenario_002.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E1_20230214/D1/C3E1N01576D1/scenario_117.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E1_20230214/D1/C3E1N01576D1/scenario_130.json"

# parameters
InFile1               = path
TimeLimitInSeconds    = 1000.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# this is the master function which executes quasiGrad.
# 
#
# =====================================================\\
# TT: start time
start_time = time()

# I1. load the system data
jsn = quasiGrad.load_json(InFile1)

# I2. initialize the system
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, true, 1.0);

# project
quasiGrad.project!(15.0, idx, prm, qG, stt, sys, upd)

quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% E7. write the final solution
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# E8. post process
quasiGrad.post_process_stats(true, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
