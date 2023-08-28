using quasiGrad
using Revise

# common folder for calling
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# call the solver!
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_031.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N08316D1/scenario_001.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N23643D1/scenario_003.json"
InFile1 = tfp*"C3S3.1_20230606/D1/C3S3N08316D1/scenario_001.json"
InFile1 = tfp*"C3S2b_20230316/D1/C3S2N02000D1/scenario_001.json"
InFile1 = tfp*"C3S2b_20230316/D1/C3S2N06717D1/scenario_001.json"  # try this agian WITH zonal penalties care

#InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"

# InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
# this is the master function which executes quasiGrad.
# 
# 
# =====================================================\\
start_time = time()
jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);

qG.print_linear_pf_iterations = true
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
qG.adam_max_time  = 20.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true, last_solve=false)
quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 100.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(95.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)

qG.adam_max_time  = 20.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false, last_solve=false)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 100.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

quasiGrad.count_active_binaries!(prm, upd)

qG.adam_max_time  = 20.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false, last_solve=true)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 100.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
stt0 = deepcopy(stt);

    ## %% ===
    #stt = deepcopy(stt0);
    #@time quasiGrad.cleanup_constrained_pf_with_Gurobi!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    #quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    #quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
    #total_time = time() - start_time
    #quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    #println("grand time: $(total_time)")
    ## %% ===

stt = deepcopy(stt0);
@time quasiGrad.cleanup_constrained_pf_with_Gurobi_freeze_subset!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
total_time = time() - start_time
quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
println("grand time: $(total_time)")

# %% =============
stt = deepcopy(stt0);
#quasiGrad.cleanup_constrained_pf_with_Gurobi_constant_dev_p!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.cleanup_constrained_pf_with_Gurobi_freeze_subset!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)


# %% ============================= test functionally!
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

InFile1 = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_941.json" # first
quasiGrad.compute_quasiGrad_solution_practice(InFile1, 1.0, 1, "test", 1)

# %% === 
InFile1 = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_963.json" # last
quasiGrad.compute_quasiGrad_solution_practice(InFile1, 1.0, 1, "test", 1)

# %% ===
InFile1 = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_963.json" # last
quasiGrad.compute_quasiGrad_solution_23k_pf(InFile1, 1.0, 1, "test", 1)


# %%
@time quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# %% ===
@time soln_dict = quasiGrad.prepare_blank_solution(prm, stt, sys, qG);

# %%

@time quasiGrad.update_solution!(prm, soln_dict, stt, sys, qG);

# %% =========================
start_time = time()
jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);

qG.adam_max_time  = 70.0

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = true)

stt0 = deepcopy(stt);

# %% =========================
stt = deepcopy(stt0);
qG.print_linear_pf_iterations = true

quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys; first_solve = true)

# %% =========================
# TT: start time
start_time = time()
jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = true)
stt0 = deepcopy(stt);

# %%
stt = deepcopy(stt0);
qG.print_linear_pf_iterations = true
qG.max_linear_pfs = 3

quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys; first_solve = true)
# %%
#stt = deepcopy(stt0);
zen0 = sum(sum(stt.zen_dev[tii][idx.cs_devs]) for tii in prm.ts.time_keys) - sum(sum(@view stt.zen_dev[tii][idx.pr_devs]) for tii in prm.ts.time_keys)

println(zen0)
# %%
#stt = deepcopy(stt0);

tii = 1
#tii = 10
#tii = 16
quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
zen0 = sum(@view stt.zen_dev[tii][idx.cs_devs]) - sum(@view stt.zen_dev[tii][idx.pr_devs])

# %%

quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(95.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

# #
# #stt = deepcopy(stt0);
# qG.adam_max_time  = 60.0
# qG.print_zms = true
# #quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
# 
# quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = false)
# #stt0 = deepcopy(stt);
# 
# # %%
# vm_pf_t0      = 1e-5
# va_pf_t0      = 1e-5
# phi_pf_t0     = 1e-5
# tau_pf_t0     = 1e-5
# dc_pf_t0      = 1e-2
# power_pf_t0   = 1e-2
# bin_pf_t0     = 1e-2 # bullish!!!
# qG.alpha_pf_t0 = Dict(
#                :vm     => vm_pf_t0,
#                :va     => va_pf_t0,
#                # xfm
#                :phi    => phi_pf_t0,
#                :tau    => tau_pf_t0,
#                # dc
#                :dc_pfr => dc_pf_t0,
#                :dc_qto => dc_pf_t0,
#                :dc_qfr => dc_pf_t0,
#                # powers
#                :dev_q  => power_pf_t0,
#                :p_on   => power_pf_t0, # downscale active power!!!!
#                # bins
#                :u_step_shunt => bin_pf_t0)
# 
# vm_pf_tf    = 1e-7
# va_pf_tf    = 1e-7
# phi_pf_tf   = 1e-7
# tau_pf_tf   = 1e-7
# dc_pf_tf    = 1e-5
# power_pf_tf = 1e-5
# bin_pf_tf   = 1e-5 # bullish!!!
# qG.alpha_pf_tf = Dict(
#                 :vm     => vm_pf_tf,
#                 :va     => va_pf_tf,
#                 # xfm
#                 :phi    => phi_pf_tf,
#                 :tau    => tau_pf_tf,
#                 # dc
#                 :dc_pfr => dc_pf_tf,
#                 :dc_qto => dc_pf_tf,
#                 :dc_qfr => dc_pf_tf,
#                 # powers
#                 :dev_q  => power_pf_tf,
#                 :p_on   => power_pf_tf, # downscale active power!!!!
#                 # bins
#                 :u_step_shunt => bin_pf_tf)
# 
# stt = deepcopy(stt0)
# quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys; first_solve = false)
#

qG.adam_max_time  = 60.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

quasiGrad.count_active_binaries!(prm, upd)
qG.adam_max_time  = 60.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
quasiGrad.cleanup_constrained_pf_with_Gurobi!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

total_time = time() - start_time

# post process
quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# final print
println("grand time: $(total_time)")

# %% =====================
# finally, update states
qG.skip_ctg_eval = true
qG.eval_grad     = false
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
qG.skip_ctg_eval = false
qG.eval_grad     = true

# %%
# solution_file = "C3E3N04224D1_scenario_131_solution.json"
# quasiGrad.write_solution(solution_file, prm, qG, stt, sys)


println(sum(sum(stt.zon_dev[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zsu_dev[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zsd_dev[tii]) for tii in prm.ts.time_keys))

# %% ============================================
# pf test

# =====================================================\\
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_031.json"

jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);

# ed
quasiGrad.solve_economic_dispatch!(idx, prm, qG, scr, stt, sys, upd; include_sus_in_ed=true)
stt0 = deepcopy(stt);

# %% ===
qG.print_zms                     = true # print zms at every adam iteration?
qG.print_final_stats             = true # print stats at the end?
qG.print_linear_pf_iterations    = true

# %% ============== test 1
stt = deepcopy(stt0);

quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys; balanced=true)

# %% ===
stt = deepcopy(stt0);

qG.adam_max_time  = 100.0
quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = false)

#
quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys; first_solve = true)

# %%
vm_pf_t0      = 1e-4
va_pf_t0      = 1e-4
phi_pf_t0     = 1e-4
tau_pf_t0     = 1e-4
dc_pf_t0      = 1e-2
power_pf_t0   = 1e-2
bin_pf_t0     = 1e-3 # bullish!!!
qG.alpha_pf_t0 = Dict(
               :vm     => vm_pf_t0,
               :va     => va_pf_t0,
               # xfm
               :phi    => phi_pf_t0,
               :tau    => tau_pf_t0,
               # dc
               :dc_pfr => dc_pf_t0,
               :dc_qto => dc_pf_t0,
               :dc_qfr => dc_pf_t0,
               # powers
               :dev_q  => power_pf_t0,
               :p_on   => power_pf_t0/2.5, # downscale active power!!!!
               # bins
               :u_step_shunt => bin_pf_t0)


# %% for tii in prm.ts.time_keys
#    stt.va[tii] .= 0.0
#    stt.vm[tii] .= 1.0
#    #stt.dev_q[tii] .= 0.0
#end

# %% ===
stt = deepcopy(stt0);

tii = Int8(2)

# first, update the xfm phase shifters (whatever they may be..)
flw.ac_phi[tii][idx.ac_phi] .= copy.(stt.phi[tii])

# loop over each bus
for bus in 1:sys.nb
    # active power balance -- just devices
    # !! don't include shunt or dc constributions, 
    #    since power might not balance !!
    stt.pinj_dc[tii][bus] = 
        sum(stt.dev_p[tii][pr] for pr in idx.pr[bus]; init=0.0) - 
        sum(stt.dev_p[tii][cs] for cs in idx.cs[bus]; init=0.0)
end

# are we dealing with a balanced dcpf? (i.e., does power balance?)
if balanced == false
    # get the slack at this time
    @fastmath p_slack = 
        sum(@inbounds stt.dev_p[tii][pr] for pr in idx.pr_devs) -
        sum(@inbounds stt.dev_p[tii][cs] for cs in idx.cs_devs)

    # now, apply this slack power everywhere
    stt.pinj_dc[tii] .= stt.pinj_dc[tii] .- p_slack/sys.nb
end

# now, we need to solve Yb*theta = pinj, but we need to 
# take phase shifters into account first:
bt = -flw.ac_phi[tii].*ntk.b
c  = stt.pinj_dc[tii][2:end] - ntk.Er'*bt
# now, we need to solve Yb_r*theta_r = c via pcg


# solve with pcg -- va
_, ch = quasiGrad.cg!(flw.theta[tii], ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)

# test the krylov solution
if ~(ch.isconverged)
    # LU backup
    @info "Krylov failed -- using LU backup (dcpf)!"
    flw.theta[tii] .= ntk.Ybr\c
end

# update -- before updating, make sure that the largest 
# phase angle differences are smaller than pi/3! if they are not
# then scale the entire thing :)
max_delta = maximum(abs.((@view stt.va[tii][idx.ac_fr_bus]) .- (@view stt.va[tii][idx.ac_to_bus]) .- flw.ac_phi[tii]))

println(max_delta)

if max_delta > pi/3
    # downscale! otherwise, you could have a really bad linearization!
    stt.va[tii][2:end] .= 0.0
else
    stt.va[tii][2:end] .= copy.(flw.theta[tii])
    stt.va[tii][1]      = 0.0 # make sure
end

# %% ==============
@time quasiGrad.energy_costs!(grd, prm, qG, stt, sys)

# %%
encs_fixed = 0.0
enpr_fixed = 0.0
for tii in prm.ts.time_keys
    encs_fixed += sum(stt.zen_dev[tii][idx.cs_devs][stt.zen_dev[tii][idx.cs_devs] .> 0.0])
    enpr_fixed -= sum(stt.zen_dev[tii][idx.pr_devs][stt.zen_dev[tii][idx.pr_devs] .> 0.0])

    # now, for the ones with the opposite signs
    enpr_fixed += sum(stt.zen_dev[tii][idx.cs_devs][stt.zen_dev[tii][idx.cs_devs] .< 0.0])
    encs_fixed -= sum(stt.zen_dev[tii][idx.pr_devs][stt.zen_dev[tii][idx.pr_devs] .< 0.0])
end

