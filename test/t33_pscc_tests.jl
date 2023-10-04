using quasiGrad
using Revise

# file path
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# define the testcases
InFile73   = tfp*"C3S0.1_20230804/D1/C3S0N00073D1/scenario_002.json"
InFile617  = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
InFile1576 = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
InFile2000 = tfp*"C3E2.1_20230515/D1/C3E2N02000D1/scenario_001.json"
InFile4224 = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"
InFile6049 = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_031.json"
InFile6717 = tfp*"C3E3.1_20230629/D1/C3E3N06717D1/scenario_031.json"
InFile8316 = tfp*"C3E3.1_20230629/D1/C3E3N08316D1/scenario_001.json"

# define the common solver information
NewTimeLimitInSeconds = 625.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0
run                   = true

quasiGrad.compute_quasiGrad_solution_d1(InFile6049, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
# %% ===

# quasiGrad.compute_quasiGrad_solution_d1(InFile73  , NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
# quasiGrad.compute_quasiGrad_solution_d1(InFile2000, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
# quasiGrad.compute_quasiGrad_solution_d1(InFile617 , NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
# 
# # %% call the d1 solver
# quasiGrad.compute_quasiGrad_solution_d1(InFile73  , NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
quasiGrad.compute_quasiGrad_solution_d1(InFile617 , NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
quasiGrad.compute_quasiGrad_solution_d1(InFile1576, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
quasiGrad.compute_quasiGrad_solution_d1(InFile2000, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
quasiGrad.compute_quasiGrad_solution_d1(InFile4224, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
# %% ===
quasiGrad.compute_quasiGrad_solution_d1(InFile6049, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)

# %%
quasiGrad.compute_quasiGrad_solution_d1(InFile6717, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)

# %%
quasiGrad.compute_quasiGrad_solution_d1(InFile8316, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)