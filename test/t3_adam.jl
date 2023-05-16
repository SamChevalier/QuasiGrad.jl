# test solver itself :)
include("../src/quasiGrad_dual.jl")
include("./test_functions.jl")

# files
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
file_name = "scenario_002.json"

# load
jsn = quasiGrad.load_json(data_dir*file_name)

# %% initialize
adm, cgd, GRB, grd, idx, mgd, ntk, prm, qG, scr, stt, 
sys, upd, flw, dz_dpinj_base, theta_k_base, worst_ctgs = 
    quasiGrad.base_initialization(jsn, true, 0.25);

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
qG.delta         = 1e5

# plot
plot_adam = true

# run :)
quasiGrad.run_adam!(adm, cgd, flw, grd, idx, mgd, ntk, plt, prm, qG, scr, stt, sys, upd,
                    dz_dpinj_base, theta_k_base, worst_ctgs)
# %% project
quasiGrad.solve_Gurobi_projection!(GRB, idx, prm, qG, stt, sys, upd)
quasiGrad.quasiGrad.apply_Gurobi_projection!(GRB, idx, prm, stt)


# %%

quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, 
                                        dz_dpinj_base, theta_k_base, worst_ctgs)