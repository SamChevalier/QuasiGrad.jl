include("../src/quasiGrad.jl")

# ===============
using Pkg
Pkg.activate(".")

using JSON
using JuMP
using Plots
using Gurobi
using Statistics
using SparseArrays
using InvertedIndices

# call this first
include("../src/core/structs.jl")
include("../src/core/shunts.jl")
include("../src/core/devices.jl")
include("../src/io/read_data.jl")
include("../src/core/ac_flow.jl")
include("../src/core/scoring.jl")
include("../src/core/clipping.jl")
include("../src/io/write_data.jl")
include("../src/core/reserves.jl")
include("../src/core/opt_funcs.jl")
include("../src/scripts/solver.jl")
include("../src/core/master_grad.jl")
include("../src/core/contingencies.jl")
include("../src/core/power_balance.jl")
include("../src/core/initializations.jl")
include("../src/core/projection.jl")

# load things
    #data_dir  = "./test/data/c3/C3S0_20221208/D1/C3S0N00003/"
    #file_name = "scenario_003.json"
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
file_name = "scenario_002.json"

# %%
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D3/C3S0N00073/scenario_002.json"


# load
jsn = quasiGrad.load_json(path)

# %% initialize the system
adm, cgd, GRB, grd, idx, mgd, ntk, prm, qG, scr, stt, 
sys, upd, flw, dz_dpinj_base, theta_k_base, worst_ctgs = 
    quasiGrad.base_initialization(jsn, true, 0.25);

# reset -- to help with numerical conditioning of the market surplus function 
# (so that we can take its derivative numerically)
qG.scale_c_pbus_testing  = 0.00001
qG.scale_c_qbus_testing  = 0.00001
qG.scale_c_sflow_testing = 0.02

# %% test
#
include("./test_functions.jl")
# README: 1) make sure the ctg solver has a sufficiently high tolerance setting!
#         2) make sure standard gradients are being used
qG.pqbal_grad_mod_type = "standard"
qG.pcg_tol             = 1e-9
epsilon                = 5e-5     # maybe set larger when dealing with ctgs + krylov solver..

#
# %% 1. transformer phase shift (phi) =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:phi][tii][ind]

# perturb and test
stt[:phi][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)
#println((dzdx - dzdx_num)/dzdx_num)

# %% 2. transformer winding ratio (tau) =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:tau][tii][ind]

# perturb and test
stt[:tau][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 3. voltage magnitude =======================================================================
include("./test_functions.jl")
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:vm][tii][ind]

# perturb and test
stt[:vm][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 4. voltage phase =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:va][tii][ind]

# perturb and test
stt[:va][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 5. dc -- qfr =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nldc)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:dc_qfr][tii][ind]

# perturb and test
stt[:dc_qfr][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 6. dc -- qto =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nldc)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:dc_qto][tii][ind]

# perturb and test
stt[:dc_qto][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 7. dc -- pfr =======================================================================
#
# so, I think this works fine -- pdc is a zero sum game, so it increases
# power at one bus by the same amount it decreases power at another bus.
# thus, if one bus has too much, and one has too little, then the change
# in the cost function will always be symmetric.
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nldc)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:dc_pfr][tii][ind]

# perturb and test
stt[:dc_pfr][tii][ind] += epsilon 
stt[:dc_pto][tii][ind] -= epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 8. u_on_acline -- note: to test su/sd tests, you can add  =======================================================================
#       the following to the initialize_static_grads! function
#           prm.acline.connection_cost    .= 1000000.0
#           prm.acline.disconnection_cost .= 1000000.0
#
# be sure to turn them off after :)
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nl)); (ind == 0 ? ind = 1 : ind = ind)

# to ensure we're not tipping over into some new space
stt[:u_on_acline][tii][ind]               = 0.9  # for su :) -- reverse for sd :)
stt[:u_on_acline][prm.ts.tmin1[tii]][ind] = 0.5  # drives infeasibility!!!
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = copy(mgd[:u_on_acline][tii][ind])

# perturb and test
stt[:u_on_acline][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 9. u_on_xfm -- note: to test su/sd tests, you can add  =======================================================================
#       the following to the initialize_static_grads! function
#           prm.xfm.connection_cost    .= 1000000.0
#           prm.xfm.disconnection_cost .= 1000000.0
#
# be sure to turn them off after :)
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)

# to ensure we're not tipping over into some new space
stt[:u_on_xfm][tii][ind]                   = 0.9  # for su :) -- reverse for sd :)
stt[:u_on_xfm][prm.ts.tmin1[tii]][ind] = 0.5
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd[:u_on_xfm][tii][ind]

# perturb and test
stt[:u_on_xfm][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 10. :u_step_shunt =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = 8 #Int64(round(rand(1)[1]*sys.nsh)) -- most have a 0-bs/gs value, so choose wisely :)

# to ensure we're not tipping over into some new space
stt[:u_step_shunt][tii][ind] = 0.4
z0    = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx  = copy(mgd[:u_step_shunt][tii][ind])

# perturb and test
stt[:u_step_shunt][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 11. devices =======================================================================
#
# README: if you perturb over the max power, the device cost will error
#         out -- this happens a lot.
quasiGrad.clip_all!(prm, stt)
#
#              1         2      3       4       5       6      7        8          9          10           11        12      13
var     = [:u_on_dev, :dev_q, :p_on, :p_rgu, :p_rgd, :p_scr, :p_nsc, :p_rru_on, :p_rrd_on, :p_rru_off, :p_rrd_off, :q_qru, :q_qrd]    
vii     = 1
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.ndev)); (ind == 0 ? ind = 1 : ind = ind)
# %%

epsilon = 1e-6
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = copy(mgd[var[vii]][tii][ind])

# perturb and test
stt[var[vii]][tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% compute the states (without gradients)
if true == false
    qG.scale_c_pbus_testing  = 1.0
    qG.scale_c_qbus_testing  = 1.0
    qG.scale_c_sflow_testing = 1.0

    # run and write
    quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, 
                                        dz_dpinj_base, theta_k_base, worst_ctgs)
    quasiGrad.solve_Gurobi_projection!(GRB, idx, prm, qG, stt, sys, upd)
    quasiGrad.quasiGrad.apply_Gurobi_projection!(GRB, idx, prm, stt)
    quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, 
                                        dz_dpinj_base, theta_k_base, worst_ctgs)

    # write a solution :)
    soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
    quasiGrad.write_solution(data_dir*file_name, qG, soln_dict, scr)
end