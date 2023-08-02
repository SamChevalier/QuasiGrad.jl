# test solver itself :)
include("../src/quasiGrad_dual.jl")
include("./test_functions.jl")

# files
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
file_name = "scenario_002.json"

# load
jsn = quasiGrad.load_json(data_dir*file_name)

# %% initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, msc, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, false, 1.0);

# %% set adam parameters
qG.alpha_0       = 0.05
qG.beta1         = 0.99
qG.beta2         = 0.995
qG.step_decay    = 0.999
qG.alpha_min     = 0.000001
qG.alpha_max     = 0.000005
qG.Ti            = 100
qG.decay_type    = "none"
qG.plot_scale_up = 2.5
qG.plot_scale_dn = 1e9
qG.adam_max_time = 60.0
qG.adam_max_its  = 250
qG.adam_stopper  = "time" # "iterations"
qG.constraint_grad_weight         = 1e5

# plot
plot_adam = true

# run :)
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd)
# %% project
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)


# %%

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)