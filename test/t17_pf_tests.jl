using quasiGrad
using Revise

path    = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
InFile1 = path
jsn     = quasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false, pert_size=1.0)

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)

# %%

qG.initial_pf_lbfgs_step = 0.05
quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)
# ^ grb off
stt0 = deepcopy(stt)

# %% reset ======================
stt = deepcopy(stt0)

# %% solve pf with GRB
quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)


# %% basically, lower the active power weight, so power moves more

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

scr[:encs]
scr[:enpr]
scr[:zp]  
scr[:zq]  
scr[:acl] 
scr[:xfm] 
