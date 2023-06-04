using quasiGrad
using GLMakie
using Revise
using Plots
using Makie

# call the plotting tools
# include("../core/plotting.jl")

# %% ===============
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D2/C3S1N00600/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D1/C3S0N00073/scenario_002.json"

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
adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

# I3. run an economic dispatch and update the states
quasiGrad.economic_dispatch_initialization!(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# TT: time
time_spent_before_loop = time() - start_time

# TT: how much time is left?
time_left = NewTimeLimitInSeconds - time_spent_before_loop

# TT: time management:
quasiGrad.manage_time!(time_left, qG)

# TT: plot
plt = Dict(:plot            => false,
           :first_plot      => true,
           :N_its           => 150,
           :global_adm_step => 0,
           :disp_freq       => 5)

# loop and solve: adam -> projection -> IBR
n_its = length(qG.pcts_to_round)
for (solver_itr, pct_round) in enumerate(qG.pcts_to_round)

    # TT: set an adam solve time
    qG.adam_max_time = qG.adam_solve_times[solver_itr]

    # L1. run power flow
    quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

    # L2. clean-up reserves by solving softly constrained LP
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    # L3. run adam
    if plt[:plot]
        if plt[:first_plot] 
            ax, fig, z_plt  = quasiGrad.initialize_plot(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, wct) 
        end
        quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctb, ctd, fig, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, wct, z_plt)
    else
        quasiGrad.run_adam!(adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
    end

    # L4. solve Gurobi projection
    quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)

    # L5. fix binaries which are closest to their Gurobi solutions
    quasiGrad.batch_fix!(pct_round, prm, stt, sys, upd)

    # L6. update the state (i.e., apply the projection)
    quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)

    # L7. on the second-to-last iteration, fix the shunts; otherwise, just snap them
    fix = solver_itr == (n_its-1)
    quasiGrad.snap_shunts!(fix, prm, stt, upd)
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
#   E5. p/q fixed: cleanup the reserves (i.e., maximize)
#           - guarenteed feasible
#   E6. p/q fixed: cleanup power flow with LP solver (just v, theta)
#           - guatenteed feasible
#   E7. prepare (and clip) and write solution
#   E6. post process (print stats)
#   
# ensure there are no more binaries/discrete variables:
quasiGrad.count_active_binaries!(prm, upd)

# E1. run power flow, one more time
quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# E2. clean-up reserves by solving softly constrained LP
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

# E3. run adam
qG.adam_max_time = qG.adam_solve_times[end]
if plt[:plot]
    quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctb, ctd, flw, fig, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, wct, z_plt)
else
    quasiGrad.run_adam!(adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
end

# E4. LP projection
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd, final_projection = true)

# E5. cleanup reserves
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

# E6. cleanup powerflow
quasiGrad.cleanup_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

# E7. write the final solution
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# E8. post process
quasiGrad.post_process_stats(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)