using BenchmarkTools
using quasiGrad
using Revise

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"

TimeLimitInSeconds    = 600.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0
jsn                   = quasiGrad.load_json(path)
    
# I2. initialize the system
adm, cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states = true);

# %% solve the ctgs
qG.num_threads = 6
@btime quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# test the solution
qG.eval_grad = false
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)
qG.eval_grad = true
println(scr[:zctg_avg])

# write a solution
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)