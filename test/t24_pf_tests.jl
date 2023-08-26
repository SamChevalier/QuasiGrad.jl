using quasiGrad
using Revise

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1.1_20230807/D1/C3S1N01576D1/scenario_001.json"

jsn  = quasiGrad.load_json(path)

# %% initialize
#t1 = time()
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);
#t = time() - t1
#println(t)

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
t1 = time()
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
t = time() - t1
println(t)

# %%
qG.num_threads = 10
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% =================
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)


# %% ============
qG.skip_ctg_eval = true

@btime quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
@btime quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

# %% ============
qG.skip_ctg_eval = false

@btime quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
@btime quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% ===========

# stt0 = deepcopy(stt)
# %%
stt = deepcopy(stt0);
qG.max_linear_pfs = 6
qG.max_linear_pfs_total = 6
qG.max_pf_dx = 1e-3

qG.Gurobi_pf_obj = "test_quad"
# %%
stt = deepcopy(stt0);
qG.max_linear_pfs = 3
qG.max_linear_pfs_total = 3
qG.max_pf_dx = 1e-3
qG.Gurobi_pf_obj = "test_quad"
@time quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys)

# %% = score!
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)
quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in 1:2)
zq = sum(lbf.zpf[:zq][tii] for tii in 1:2)
println(zq + zp)


# %% ==========================
tii = Int8(1)
@time Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii);
@time Jac = quasiGrad.build_acpf_Jac_and_pq0(qG, stt, sys, tii, Ybus_real, Ybus_imag);


# %%

A = randn(50,50)
B = randn(50,50)

@time C = A.+B;
@time D = A+B;
# %%
@time gg[1] .= 2*Ybus_real;
@time gg[1] = 2*Ybus_real;

# %% ==
# stt0 = deepcopy(stt);
stt = deepcopy(stt0);
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %%
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
stt = deepcopy(stt0);

# quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys)


qG.initial_pf_lbfgs_step = 1e-3
qG.lbfgs_adam_alpha_0    = 1e-3
qG.num_lbfgs_to_keep     = 10
qG.num_lbfgs_steps       = 1000
#qG.pqbal_grad_eps2       = 1e-2

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% apply reactive power
stt = deepcopy(stt0);
for tii in prm.ts.time_keys
    quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
end

# %% ============= 
stt = deepcopy(stt0);

@btime quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)

# %%
stt = deepcopy(stt0);

quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)
quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
println(zp)


# %% ==========***
stt = deepcopy(stt0);

qG.include_energy_costs_lbfgs      = false # true
qG.include_lbfgs_p0_regularization = false

# set the loss function to quadratic -- low gradient factor
qG.pqbal_grad_type = "quadratic_for_lbfgs"

# loop -- lbfgs
init_pf   = true
run_lbfgs = true
lbfgs_cnt = 0
zt0       = 0.0

# re-initialize the lbf(gs) struct
quasiGrad.flush_lbfgs!(lbf, prm, qG, stt)

# initialize: compute all states and grads
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)

# %% ==========================
emergency_stop = quasiGrad.solve_pf_lbfgs!(lbf, mgd, prm, qG, stt, upd)

# save zpf BEFORE updating with the new state -- don't track bias terms
for tii in prm.ts.time_keys
    lbf.step[:zpf_prev][tii] = (lbf.zpf[:zp][tii]+lbf.zpf[:zq][tii]) 
end

# %% compute all states and grads
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)

# increment
lbfgs_cnt += 1



# %%
quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)
quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
println(zp)

# %%

# loop -- lbfgs
while run_lbfgs == true
    # take an lbfgs step
    emergency_stop = quasiGrad.solve_pf_lbfgs!(lbf, mgd, prm, qG, stt, upd)

    # save zpf BEFORE updating with the new state -- don't track bias terms
    for tii in prm.ts.time_keys
        lbf.step[:zpf_prev][tii] = (lbf.zpf[:zp][tii]+lbf.zpf[:zq][tii]) 
    end

    # compute all states and grads
    quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)

    # store the first value
    zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
    zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
    zt = zp + zq
    if init_pf == true
        zt0 = copy(zt)
        init_pf = false
    end

    # print
    if qG.print_lbfgs_iterations == true
        ztr = round(zt; sigdigits = 3)
        zpr = round(zp; sigdigits = 3)
        zqr = round(zq; sigdigits = 3)
        stp = round(sum(lbf.step[:step][tii] for tii in prm.ts.time_keys)/sys.nT; sigdigits = 3)
        println("Total: $(ztr), P penalty: $(zpr), Q penalty: $(zqr), avg adam step: $(stp)!")
    end

    # increment
    lbfgs_cnt += 1

    # quit if the error gets too large relative to the first error
    if (lbfgs_cnt > qG.num_lbfgs_steps) || (zt > 5.0*zt0) || (emergency_stop == true)
        run_lbfgs = false
    end
end

# %% ==========================
model = Model(Gurobi.Optimizer)
@variable(model, x[1:10])
@constraint(model, x[dd] <= 0 for dd in 1:3)