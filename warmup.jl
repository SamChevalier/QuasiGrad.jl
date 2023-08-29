using Pkg
Pkg.activate(DEPOT_PATH[1])

@info "Running warmup.jl! Good luck." 

# load quasiGrad and MyJulia
include("./src/quasiGrad.jl")
include("./MyJulia1.jl")

# execute a minisolver
InFile1               = "./src/precompile_14bus.json"
TimeLimitInSeconds    = 1
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1
precompile_minisolver = true
minisolver(InFile1, TimeLimitInSeconds, Division, NetworkModel, AllowSwitching)