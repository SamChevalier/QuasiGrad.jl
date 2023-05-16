using quasiGrad
using Revise

# load the json
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"

# call
jsn = quasiGrad.load_json(path)

# %% init
adm, cgd, GRB, grd, idx, mgd, ntk, prm, qG, scr, stt, 
sys, upd, flw, dz_dpinj_base, theta_k_base, worst_ctgs = 
    quasiGrad.base_initialization(jsn, false, 1.0);

# solve
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

# run an ED
ED = quasiGrad.solve_economic_dispatch(GRB, idx, prm, qG, scr, stt, sys, upd);
quasiGrad.apply_economic_dispatch_projection!(ED, idx, prm, stt);

# recompute the state
qG.eval_grad = false
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)
qG.eval_grad = true

# ===== new score?
quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

# intialize lbfgs
dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf = quasiGrad.initialize_pf_lbfgs(mgd, prm, stt, sys, upd);

qG.scale_c_pbus_testing = 1e-4
qG.scale_c_qbus_testing = 1e-4
qG.cdist_psolve = 1e3

#prm.vio.p_bus   = 1e3
#prm.vio.q_bus   = 1e3

# loop -- lbfgs
for ii in 1:5000
    # take an lbfgs step
    quasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, scr, stt, upd, zpf)

    # save zpf BEFORE updating with the new state
    for tii in prm.ts.time_keys
        pf_lbfgs_step[:zpf_prev][tii] = (zpf[:zp][tii]+zpf[:zq][tii]) 
    end

    # compute all states and grads
    quasiGrad.update_states_and_grads_solve_pf!(dpf0, grd, idx, mgd, prm, qG, stt, sys, zpf)

    # print
    z   = sum(sum([zpf[:zp][tii] + zpf[:zq][tii] for tii in prm.ts.time_keys]))
    stp = sum(pf_lbfgs_step[:step][tii] for tii in prm.ts.time_keys)/sys.nT
    println("The power balance penalty is $(z) and average step is $(stp)!")

    #println(stt[:dev_q][:t1][1:2])
end

# %% write solution
quasiGrad.solve_Gurobi_projection!(GRB, idx, prm, qG, stt, sys, upd)
quasiGrad.quasiGrad.apply_Gurobi_projection!(GRB, idx, prm, stt)

# one last clip + state computation -- no grad needed!
qG.eval_grad = false
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

# write the final solution
soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
quasiGrad.write_solution("solution.jl", qG, soln_dict, scr)

git remote set-url origin git@github.com:SamChevalier/GO.git

git remote add origin https://github.com/SamChevalier/GO

# %% ================== 
# sum(sum([qG.cdist_psolve*(stt[:p_on][tii] - dpf0[:p_on][tii]).^2 for tii in prm.ts.time_keys]))
# z = sum(sum([zpf[:zq][tii] for tii in prm.ts.time_keys]))

