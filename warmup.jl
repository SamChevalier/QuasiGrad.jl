using Pkg

for pkg in Pkg.installed()
   pkgname=pkg[1]; pkgsym=Symbol(pkgname)
   try
println("using...$pkgname")
        eval(:(using $pkgsym))
   catch
        println("could not precompile $pkgname")
   end
end

# call quasiGrad module
include("./src/quasiGrad.jl")

# now, run a sample workload
path = "./src/precompile_14bus.json"

# call the jsn and initialize
jsn = quasiGrad.load_json(path)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)