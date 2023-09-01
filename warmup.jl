using Pkg
Pkg.activate(DEPOT_PATH[1])
Pkg.status()

# good luck!
@info "Running warmup.jl! Good luck." 

# first, add quasiGrad
# => Pkg.add(url="https://github.com/SamChevalier/quasiGrad")

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
# => include("./MyJulia1.jl")
# => include("./warmup_run.jl")
# => 
# => # create system image
# => using PackageCompiler
# => create_sysimage(sysimage_path="/qfs/projects/goc/share/apps/Julia/julia-1.10.0-beta1-SamChevalier/share/julia/site/SamChevalierQG5.so", precompile_execution_file="warmup_run.jl")