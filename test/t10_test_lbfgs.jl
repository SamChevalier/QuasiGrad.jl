using quasiGrad
using Revise

# load the json
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
# call
jsn = quasiGrad.load_json(path)

# %% init
adm, cgd, ctg, flw, grd, idx, lbf, mgd, msc, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, false, 1.0);

# solve
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

# run an ED
ED = quasiGrad.solve_economic_dispatch(GRB, idx, prm, qG, scr, stt, sys, upd);
quasiGrad.apply_economic_dispatch_projection!(ED, idx, prm, qG, stt, sys);

# recompute the state
qG.eval_grad = false
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)
qG.eval_grad = true

# ===== new score?
quasiGrad.dcpf_initialization!(flw, idx, msc, ntk, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

# intialize lbfgs
lbfgs, lbfgs_diff, lbfgs_idx, lbfgs_map, lbfgs_step = quasiGrad.initialize_lbfgs(mgd, prm, sys, upd);
lbfgs_step[:alpha_0] = 0.005

# %% loop -- lbfgs
for ii in 1:1000
    # take an lbfgs step
    quasiGrad.lbfgs!(lbfgs, lbfgs_diff, lbfgs_idx, lbfgs_map, lbfgs_step, mgd, prm, qG, scr, stt, upd)

    # save the zms
    lbfgs_step[:nzms_prev] = scr[:nzms]

    # compute all states and grads
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

    # print
    println("The (non-penalized) market surplus is $(scr[:zms])!")
    println("step: $(lbfgs_step[:step])")
end

# %% loop -- adam
#
# initialize
adm_step    = 0
beta1       = qG.beta1
beta2       = qG.beta2
beta1_decay = 1.0
beta2_decay = 1.0
run_adam    = true
alpha       = 0.001

# flush adam at each restart
quasiGrad.flush_adam!(adm, prm, upd)

# -- loop
for ii in 1:1000
    adm_step += 1

    # decay beta
    beta1_decay = beta1_decay*beta1
    beta2_decay = beta2_decay*beta2

    # compute all states and grads
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

    # take an adam step
    quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

    # print
    println("The (non-penalized) market surplus is $(scr[:zms])!")
end

# %%

quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)

# one last clip + state computation -- no grad needed!
qG.eval_grad = false
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

# write the final solution
soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
quasiGrad.write_solution("solution.jl", qG, soln_dict, scr)
