using quasiGrad
using Revise

# %%
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
InFile1               = path
TimeLimitInSeconds    = 600.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 10.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# this is the master function which executes quasiGrad.
# 1. InFile1 -> if string, assume we need to load the jsn data
# 
#
# =====================================================\\
# start time
start_time = time()

# load the system data
jsn = quasiGrad.load_json(InFile1)

# initialize the system
adm, cgd, flw, GRB, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, 
sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs = quasiGrad.base_initialization(jsn, false, 1.0);

# run an economic dispatch and update the states
quasiGrad.economic_dispatch_initialization!(cgd, flw, GRB, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs)

# get a power flow solution
qG.num_lbfgs_steps = 50

quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# %% time
time_spent_before_loop = time() - start_time

# how much time is left?
time_left = NewTimeLimitInSeconds - time_spent_before_loop

# time management:
quasiGrad.manage_time!(time_left, qG)

# %% ===== adam

# plot tools
plt = Dict(:plot             => true,
            :first_plot      => true,
            :N_its           => 150,
            :global_adm_step => 0,
            :disp_freq       => 5)

qG.adam_max_time = 50.0
if plt[:first_plot] ax, fig, z_plt  = quasiGrad.initialize_plot(cgd, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs) end
quasiGrad.run_adam_with_plotting!(adm, ax, cgd, fig, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs, z_plt)

# %% loop and solve: adam -> projection -> IBR
n_its = length(qG.pcts_to_round)
for (solver_itr, pct_round) in enumerate(qG.pcts_to_round)

    # 0. set an adam solve time
    qG.adam_max_time = qG.adam_solve_times[solver_itr]

    # 1. run adam
    if plt[:plot]
        if plt[:first_plot] ax, fig, z_plt  = quasiGrad.initialize_plot(cgd, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs) end
        quasiGrad.run_adam_with_plotting!(adm, ax, cgd, fig, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs, z_plt)
    else
        quasiGrad.run_adam!(adm, cgd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs)
    end

    # 2. solve Gurobi projection
    quasiGrad.solve_Gurobi_projection!(GRB, idx, prm, qG, stt, sys, upd)

    # 3. fix binaries which are closest to their Gurobi solutions
    quasiGrad.batch_fix!(GRB, pct_round, prm, stt, sys, upd)

    # 4. using the previous solution, now update the state (i.e., apply the projection)
    quasiGrad.apply_Gurobi_projection!(GRB, idx, prm, stt)

    # run power flow
    quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

    # 5. on the second-to-last iteration, fix the shunts; otherwise, just snap them
    if solver_itr < (n_its-1)
        quasiGrad.snap_shunts!(false, prm, stt, upd)
    elseif solver_itr == (n_its-1)
        quasiGrad.snap_shunts!(true, prm, stt, upd)
    end
end
##############################################################
##############################################################
#
# now, with all binaries fixed, we run one last adam solve
# with tight tolerances, followed by one last Gurobi LP solve
    # => println([upd[:u_on_dev][tii] == Int64[] for tii in prm.ts.time_keys])
qG.adam_max_time = qG.adam_solve_times[end]
if plt[:plot]
    quasiGrad.run_adam_with_plotting!(adm, ax, cgd, flw, fig, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs, z_plt)
else
    quasiGrad.run_adam!(adm, cgd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs)
end
quasiGrad.solve_Gurobi_projection!(GRB, idx, prm, qG, stt, sys, upd)
quasiGrad.apply_Gurobi_projection!(GRB, idx, prm, stt)

# run power flow
quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# one last clip + state computation -- no grad needed!
qG.eval_grad = false
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

# write the final solution
soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
quasiGrad.write_solution("solution.jl", qG, soln_dict, scr)