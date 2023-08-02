using quasiGrad
using Revise
#using Plots
using Makie

# %% define
s = 2     # set
d = 1     # division
c = 600   # case (num buses)
n = 1     # scenario
action = "just write"

# define the test set
if s == 1
    set = "C3S0_20221208/" 
elseif s == 2
    set = "C3S1_20221222/"
elseif s == 3
    set = "C3S2b_20230316/"
elseif s == 4
    set = "C3E1_20230214/"
end

# define the division
if d == 1
    dvn = "D1/"
elseif d == 2
    dvn = "D2/"
elseif d == 3
    dvn = "D3/"
end

# define the case
if c == 3
    case = "C3S0N00003/"
elseif c == 14
    case = "C3S0N00014/"
elseif c == 37
    case = "C3S0N00037/"
elseif c == 73
    case = "C3S0N00073/"
elseif c == 600
    case = "C3S1N00600/"
elseif c == 1576
    case = "C3S1N01576/"
elseif c == 4200
    case = "C3S1N04200/"
elseif c == 6049
    case = "C3S1N06049/"
end

if n == 1
    nro = "scenario_001.json"
elseif n == 2
    nro = "scenario_002.json"
elseif n == 3
    nro = "scenario_003.json"
end

# load the json
path = "../GO3_testcases/"*set*dvn*case*nro
#path = "../GO3_testcases/C3S1_20221222/D1/C3S1N01576/scenario_001.json"
#path = "../GO3_testcases/C3S1_20221222/D1/C3S1N06049/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N01576/scenario_001.json"

# call
jsn = quasiGrad.load_json(path)

# init
adm, cgd, ctg, flw, grd, idx, lbf, mgd, msc, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, false, 1.0);

# %% solve
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)
# run an ED
ED = quasiGrad.solve_economic_dispatch(GRB, idx, prm, qG, scr, stt, sys, upd);
quasiGrad.apply_economic_dispatch_projection!(ED, idx, prm, qG, stt, sys);

# recompute the state
qG.eval_grad = false
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)
qG.eval_grad = true

# ===== new score?
quasiGrad.dcpf_initialization!(flw, idx, msc, ntk, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

# %% now, run a dedicated adam-power flow :)
qG.adam_max_time = 300.0

# plot tools
plt = Dict(:plot            => true,
           :first_plot      => true,
           :N_its           => 150,
           :global_adm_step => 0,
           :disp_freq       => 1)
qG.alpha_0 = 0.0001
qG.constraint_grad_weight   = 1e6

if plt[:first_plot]
    ax, fig, z_plt = quasiGrad.initialize_plot(plt, scr)
end

quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctb, ctd, fig, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, upd, wct, z_plt)

# %%
#dev = 75
#for tii in prm.ts.time_keys
#    println(GRB[:p_rgu][tii][dev] + GRB[:p_scr][tii][dev] - prm.dev.p_syn_res_ub[dev]*GRB[:u_on_dev][tii][dev])
#end