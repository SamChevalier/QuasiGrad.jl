using quasiGrad
using Revise

# => using Pkg
# => Pkg.activate(DEPOT_PATH[1])
# => using quasiGrad

# precompile
NewTimeLimitInSeconds = 600.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1

# %% ==============
pc("./src/precompile_617bus.json", NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

# %% common folder for calling
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# %% call the solver!
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %% =============
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %% ===
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# rerun with 

# %% === test this first
t0 = time()
InFile1 = tfp*"C3E2.1_20230515/D1/C3E2N02000D1/scenario_004.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %% === test this second
t0 = time()
InFile1 = tfp*"C3E2.1_20230515/D1/C3E2N06717D1/scenario_004.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %% === D2 test
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D2/C3E3N00617D2/scenario_001.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %% === D2 test
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D2/C3E3N01576D2/scenario_027.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %% ===
InFile1 = tfp*"C3E3.1_20230629/D2/C3E3N01576D2/scenario_027.json"
jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=Division, hpc_params=false, line_switching=0);
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% ===
quasiGrad.solve_LP_Gurobi_projection!(idx, prm, qG, stt, sys, upd)

# %% ===
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_043.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N06717D1/scenario_031.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %%
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N08316D1/scenario_001.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")



# %% ============== pc ===============
function pc(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    jsn = quasiGrad.load_json(InFile1)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
        quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);

    # assign a short run-time
    qG.adam_max_time = 3.0

    # in this case, run a minisolve with the 14 bus system
    quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
    quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    quasiGrad.count_active_binaries!(prm, upd)
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.cleanup_constrained_pf_with_Gurobi_freeze_subset!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

    qG.write_location = "local"
    quasiGrad.write_solution("junk.json", prm, qG, stt, sys)
    quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
end

# %% ===================== 
start_time = time()
NewTimeLimitInSeconds = 700.0

# =====================================================\\
tfp     = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"

jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    quasiGrad.base_initialization(jsn, Div=1, hpc_params=true);
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# baby systems
qG.max_linear_pfs = 3
qG.adam_max_time  = 30.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(75.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

qG.adam_max_time  = 5.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(90.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

qG.adam_max_time  = 5.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(99.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

qG.adam_max_time  = 5.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

# ====================================== #
quasiGrad.count_active_binaries!(prm, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
# ====================================== #

# time left? save 50 seconds for ramp_constrained solve
time_for_pf               = 5.0
time_for_final_activities = 50.0
time_spent = time() - start_time
time_left  = NewTimeLimitInSeconds - time_spent - time_for_final_activities - time_for_pf
if time_left > 15.0
    time_for_final_pf   = time_left*0.10
    time_for_final_adam = time_left*0.85

    qG.adam_max_time  = time_for_final_pf
    qG.max_linear_pfs = 1
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    qG.adam_max_time  = time_for_final_adam
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
else
    # just run a final projection -- this needs to be here!!
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
end
stt0 = deepcopy(stt);

# %% final activities
qG.print_linear_pf_iterations    = true

stt = deepcopy(stt0);

quasiGrad.cleanup_constrained_pf_with_Gurobi_freeze_subset_23k!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% final activities
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# %% =======================
tfp     = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
InFile1 = tfp*"C3E3.1_20230629/D2/C3E3N04224D2/scenario_131.json"

# this is the master function which executes quasiGrad (d2/3)
# 
# three network-size rounding schemes: 
#   - fewer than 10000  buses: 50, 75, 90, 99 100, 100
#   - more than 10000 buses:   95, 100, 100
start_time = time()

# =====================================================\\
jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
quasiGrad.base_initialization(jsn, Div=2, hpc_params=true);
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% ===================
qG.max_linear_pfs = 3
qG.adam_max_time  = 5.0*30.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 10.0*60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(50.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

qG.adam_max_time  = 5.0*5.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 10.0*60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(75.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

qG.adam_max_time  = 5.0*5.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 10.0*60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(90.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)  

qG.adam_max_time  = 5.0*5.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 10.0*60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(99.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(false, prm, qG, stt, upd)   

qG.adam_max_time  = 5.0*5.0
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
qG.adam_max_time  = 10.0*60.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)

# ====================================== #
quasiGrad.count_active_binaries!(prm, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
# ====================================== #

# time left? save 50 seconds for ramp_constrained solve
time_for_pf               = 3.0*10.0
time_for_final_activities = 3.0*50.0
time_spent = time() - start_time
time_left  = NewTimeLimitInSeconds - time_spent - time_for_final_activities - time_for_pf
if time_left > 30.0
    time_for_final_pf   = time_left*0.10
    time_for_final_adam = time_left*0.80

    qG.adam_max_time  = time_for_final_pf
    qG.max_linear_pfs = 1
    quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; last_solve=true)
    quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
    qG.adam_max_time  = time_for_final_adam
    quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
    quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
end

# final activities
quasiGrad.cleanup_constrained_pf_with_Gurobi_parallelized!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# %%
println(sum(sum(stt.zrgu_zonal[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zrgd_zonal[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zscr_zonal[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.znsc_zonal[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zrru_zonal[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zrrd_zonal[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zqru_zonal[tii]) for tii in prm.ts.time_keys))
println(sum(sum(stt.zqrd_zonal[tii]) for tii in prm.ts.time_keys))