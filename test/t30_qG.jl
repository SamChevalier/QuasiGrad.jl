using quasiGrad
using BenchmarkTools
using Revise

# %% common folder for calling
tfp  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# call the solver!
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"

quasiGrad.compute_quasiGrad_solution_practice(InFile1, 1.0, 1, "test", 1)

# %% ============
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
quasiGrad.compute_quasiGrad_TIME_23643(InFile1, 1.0, 1, "test", 1)

# %% =========

# this is the master function which executes quasiGrad.
# 
# 
# =====================================================\\
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_031.json"

# TT: start time
start_time = time()

jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);

qG.adam_max_time  = 100.0

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)


for tii in prm.ts.time_keys
    stt.va[tii] .= 0.0
    stt.vm[tii] .= 1.0
    #stt.dev_q[tii] .= 0.0
end
#=
vm_pf_t0      = 1e-4
va_pf_t0      = 1e-4
phi_pf_t0     = 1e-4
tau_pf_t0     = 1e-4
dc_pf_t0      = 1e-2
power_pf_t0   = 1e-2
bin_pf_t0     = 1e-2 # bullish!!!
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
               :p_on   => power_pf_t0, # downscale active power!!!!
               # bins
               :u_step_shunt => bin_pf_t0)


qG.alpha_pf_t0[:dev_q] = 1e-2
qG.print_zms = true

qG.adam_max_time  = 100.0
quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys; first_solve = true)


vm_pf_t0      = 1e-4
va_pf_t0      = 1e-4
phi_pf_t0     = 1e-4
tau_pf_t0     = 1e-4
dc_pf_t0      = 1e-2
power_pf_t0   = 1e-2
bin_pf_t0     = 1e-2 # bullish!!!
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
               :p_on   => power_pf_t0, # downscale active power!!!!
               # bins
               :u_step_shunt => bin_pf_t0)

for tii in prm.ts.time_keys
    #stt.va[tii] .= 0.0
    stt.vm[tii] .= 1.0
    stt.dev_q[tii] .= 0.0
end
=#
#  =========
quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)

# %%
quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(99.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

# =...
quasiGrad.count_active_binaries!(prm, upd)
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=false)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

total_time = time() - start_time

# post process
quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# final print
println("grand time: $(total_time)")

#println("ed upper bounds: $(scr[:ed_obj])")
#println("zms: $(scr[:zms])")
#println("zms_p: $(scr[:zms_penalized])")
#
#println("zp: $(scr[:zp])")
#println("zq: $(scr[:zq])")
#
#println("zs_acline: $(scr[:acl])")
#println("zs_xfm: $(scr[:xfm])")
#
#println("z_enpr: $(scr[:enpr])")
#println("z_encs: $(scr[:encs])")

# %% =====================
# finally, update states
qG.skip_ctg_eval = true
qG.eval_grad     = false
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
qG.skip_ctg_eval = false
qG.eval_grad     = true

# solution_file = "C3E3N04224D1_scenario_131_solution.json"
# quasiGrad.write_solution(solution_file, prm, qG, stt, sys)


println(sum(sum(stt.zon_dev[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zsu_dev[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zsd_dev[tii]) for tii in prm.ts.time_keys))


# %%
for tii in prm.ts.time_keys
    for dev in 1:sys.ndev
        cst = minimum(prm.dev.cum_cost_blocks[dev][tii][1])
        if cst < 0.0
            println(cst)
        end
    end
end