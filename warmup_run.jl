using quasiGrad
include("./MyJulia1.jl")

# test run
NewTimeLimitInSeconds = 10.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1
pc("./src/precompile_37bus.json", NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
