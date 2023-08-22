using quasiGrad
using Revise

# common folder for calling
tfp  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# call the solver!
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json" 
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"

quasiGrad.compute_quasiGrad_solution_practice(InFile1, 1.0, 1, "test", 1)

# %% =========

# this is the master function which executes quasiGrad.
# 
# 
# =====================================================\\
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"

# TT: start time
start_time = time()

jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);

qG.adam_max_time  = 90.0

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
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

println("ed upper bounds: $(scr[:ed_obj])")
println("zms: $(scr[:zms])")
println("zms_p: $(scr[:zms_penalized])")

println("zp: $(scr[:zp])")
println("zq: $(scr[:zq])")

println("zs_acline: $(scr[:acl])")
println("zs_xfm: $(scr[:xfm])")

println("z_enpr: $(scr[:enpr])")
println("z_encs: $(scr[:encs])")