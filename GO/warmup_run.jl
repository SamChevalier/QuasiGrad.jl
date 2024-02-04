using QuasiGrad
include("./MyJulia1.jl")

# test run
NewTimeLimitInSeconds = 600.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1
pc("./src/precompile_617bus.json", NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
