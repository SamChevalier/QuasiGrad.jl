using quasiGrad

# test a dvision 1 test case
InFile                = "./data/C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
NewTimeLimitInSeconds = 600.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0
quasiGrad.compute_quasiGrad_solution_d1(InFile, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
