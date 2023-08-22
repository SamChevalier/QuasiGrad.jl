using quasiGrad
using Revise

# ===============
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"

    # great benchmark for Jpqe/etc !!
    # => path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D1/scenario_001.json"

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

# I2. initialize the system
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, Div=Division);

@warn "homotopy ON"
qG.apply_grad_weight_homotopy = true

# I3. run an economic dispatch and update the states
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# TT: time
time_spent_before_loop = time() - start_time

# TT: how much time is left?
time_left = NewTimeLimitInSeconds - time_spent_before_loop

# TT: time management:
quasiGrad.manage_time!(time_left, qG)

# % ======================================
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
# %% =====================================

# loop and solve: adam -> projection -> IBR
n_its = length(qG.pcts_to_round)
for (solver_itr, pct_round) in enumerate(qG.pcts_to_round)

    # TT: set an adam solve time
    qG.adam_max_time = qG.adam_solve_times[solver_itr]

    # L1. run power flow
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

    # L2. clean-up reserves by solving softly constrained LP
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # L3. run adam
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

    # L4. solve and apply projection
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

    # L5. on the second-to-last iteration, fix the shunts; otherwise, just snap them
    fix = solver_itr == (n_its-1)
    quasiGrad.snap_shunts!(fix, prm, qG, stt, upd)
end
##############################################################
##############################################################

# Now, we're in the End Game.
#
# with all binaries and shunts fixed and power flow solved, we..
#   E1. solve power flow one more time
#   E2. (softly) cleanup the reserves
#   E3. run adam one more time, with very tight constraint tolerances
#   E4. solve the (~MI)LP projection with very tight p/q
#   E5. cleanup power flow
#   E6. clearnup reserves
#   E7. prepare (and clip) and write solution
#   E7. post process (print stats)
#   
# ensure there are no more binaries/discrete variables:
quasiGrad.count_active_binaries!(prm, upd)

# E1. run power flow, one more time
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

# E2. clean-up reserves by solving softly constrained LP
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

# E3. run adam
qG.adam_max_time = qG.adam_solve_times[end]
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# E4. LP projection
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)

# E5. cleanup constrained powerflow
quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

# E6. cleanup reserves
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

# E7. write the final solution
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# E8. post process
quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)