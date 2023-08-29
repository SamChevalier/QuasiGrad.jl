using Pkg
Pkg.activate(DEPOT_PATH[1])

# using quasiGrad
# using Revise

# %% common folder for calling
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# call the solver!
InFile1               = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
NewTimeLimitInSeconds = 600.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1

quasiGrad.compute_quasiGrad_solution_d1(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)