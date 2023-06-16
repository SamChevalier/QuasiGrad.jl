using quasiGrad
using Revise

# ===============
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
    # great benchmark for Jpqe/etc !!
    # => path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D1/scenario_001.json"

# parameters
InFile1               = path
TimeLimitInSeconds    = 600.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

quasiGrad.compute_triage_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

# %% this is the master function which executes quasiGrad.
# 
#
# =====================================================\\
# TT: start time
start_time = time()

# I1. load the system data
jsn = quasiGrad.load_json(InFile1)

# I2. initialize the system
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, Div=Division);

@warn "homotopy ON"
qG.apply_grad_weight_homotopy = true

# I3. run an economic dispatch and update the states
quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# TT: time
time_spent_before_loop = time() - start_time

# TT: how much time is left?
time_left = NewTimeLimitInSeconds - time_spent_before_loop

# TT: time management:
quasiGrad.manage_time!(time_left, qG)

# TT: set an adam solve time
qG.adam_max_time = 60.0

# L1. run power flow
quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# %% L2. clean-up reserves by solving softly constrained LP
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

# L3. run adam
quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

quasiGrad.post_process_stats(true, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %%
# quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)


# %%
tii = :t1

quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii);
Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);

# %%
msc[:pinj_ideal][tii]
msc[:qinj_ideal][tii]

msc[:pinj0][tii]
msc[:qinj0][tii]

# %% =====================
qG.num_threads = 6

vmva_scale    = 1e-7
xfm_scale     = 1e-7
dc_scale      = 1e-5
power_scale   = 1e-4
reserve_scale = 1e-4
bin_scale     = 5e-3 # bullish!!!
alpha_0 = Dict(:vm     => vmva_scale,
                :va     => vmva_scale,
                # xfm
                :phi    => xfm_scale,
                :tau    => xfm_scale,
                # dc
                :dc_pfr => dc_scale,
                :dc_qto => dc_scale,
                :dc_qfr => dc_scale,
                # powers -- decay this!!
                :dev_q  => power_scale,
                :p_on   => power_scale,
                # reserves
                :p_rgu     => reserve_scale,
                :p_rgd     => reserve_scale,
                :p_scr     => reserve_scale,
                :p_nsc     => reserve_scale,
                :p_rrd_on  => reserve_scale,
                :p_rrd_off => reserve_scale,
                :p_rru_on  => reserve_scale,
                :p_rru_off => reserve_scale,
                :q_qrd     => reserve_scale,
                :q_qru     => reserve_scale,
                # bins
                :u_on_xfm     => bin_scale,
                :u_on_dev     => bin_scale,
                :u_step_shunt => bin_scale,
                :u_on_acline  => bin_scale)

qG.alpha_0 = alpha_0
# %% %%%%%%%%%%%%%
# stt0 = deepcopy(stt)

# %% %%%%%%%%%%%%%
stt = deepcopy(stt0)
# %% %%%%%%%%%%%%%
quasiGrad.flush_adam!(adm, mgd, prm, upd)

# %%
qG.pqbal_grad_weight_p    = (1e-3*prm.vio.p_bus)
qG.pqbal_grad_weight_q    = (1e-3*prm.vio.q_bus)
qG.constraint_grad_weight = (1e-3*prm.vio.p_bus)
qG.acflow_grad_weight     = (2.5e-2*prm.vio.s_flow)

adm_step    = 1
beta1       = qG.beta1
beta2       = qG.beta2
beta1_decay = 1.0
beta2_decay = 1.0
run_adam    = true

# step decay
# alpha = step_decay(adm_step, qG)

# decay beta
beta1_decay = beta1_decay*beta1
beta2_decay = beta2_decay*beta2

# update weight parameters?
#if qG.apply_grad_weight_homotopy == true
#    update_penalties!(prm, qG, time(), adam_start, adam_start+qG.adam_max_time)
#end

# compute all states and grads
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% take an adam step
quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

# %% ==
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
println(scr[:zms])