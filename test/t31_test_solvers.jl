using Pkg
Pkg.activate(DEPOT_PATH[1])
using quasiGrad

# %%
#using Revise

# precompile
NewTimeLimitInSeconds = 600.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1
quasiGrad.pc("./src/precompile_37bus.json", NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

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

t0 = time()
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_143.json"
quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
tf = time() - t0
println("total time: $tf")

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

