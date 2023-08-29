using Pkg
Pkg.activate(DEPOT_PATH[1])

# call quasiGrad
using quasiGrad

# load MyJulia
include("./MyJulia1.jl")

# execute a minisolver
InFile1               = "./src/precompile_14bus.json"
TimeLimitInSeconds    = 1
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1
precompile_minisolver = true
MyJulia1(InFile1, TimeLimitInSeconds, Division, NetworkModel, AllowSwitching, precompile_minisolver=true)