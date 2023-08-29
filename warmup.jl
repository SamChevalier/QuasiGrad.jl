using Pkg
Pkg.activate(DEPOT_PATH[1])

@info "Running warmup.jl! Good luck." 

# first, add quasiGrad
Pkg.add(url="https://github.com/SamChevalier/quasiGrad")

# %% now, precompile
for pkg in Pkg.installed()
   pkgname=pkg[1]; pkgsym=Symbol(pkgname)
   try
println("using...$pkgname")
        eval(:(using $pkgsym))
   catch
        println("could not precompile $pkgname")
   end
end

# %% load quasiGrad and MyJulia
#include("./src/quasiGrad.jl")
include("./MyJulia1.jl")

# execute a minisolver
InFile1               = "./src/precompile_37bus.json"
TimeLimitInSeconds    = 1
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 1
precompile_minisolver = true
MyJulia1(InFile1, TimeLimitInSeconds, Division, NetworkModel, AllowSwitching; precompile_minisolver=precompile_minisolver)