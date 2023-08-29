using Pkg
Pkg.activate(DEPOT_PATH[1])

# good luck!
@info "Running warmup.jl! Good luck." 

# first, add quasiGrad
Pkg.add(url="https://github.com/SamChevalier/quasiGrad")

# now, precompile
for pkg in Pkg.installed()
   pkgname=pkg[1]; pkgsym=Symbol(pkgname)
   try
         println("using...$pkgname")
         eval(:(using $pkgsym))
   catch
         println("could not precompile $pkgname")
   end
end

# load MyJulia and warmup_run
include("./MyJulia1.jl")
include("./warmup_run.jl")

# %% create system image
using PackageCompiler
create_sysimage(["quasiGrad"], sysimage_path=DEPOT_PATH[1]*"\\SamChevalier.so", precompile_execution_file="warmup_run.jl")