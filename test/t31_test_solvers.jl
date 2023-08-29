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
pc("./src/precompile_37bus.json", NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

# common folder for calling
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# %% call the solver!
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %%
t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_143.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

# %%
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