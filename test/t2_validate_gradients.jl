# test solver itself :)
using quasiGrad
using Revise

# load things
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D3/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N01576D1/scenario_007.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"

# load
jsn = quasiGrad.load_json(path)

# initialize the system
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states = true);

# %% reset -- to help with numerical conditioning of the market surplus function 
# (so that we can take its derivative numerically)
qG.scale_c_pbus_testing  = 1.0 #1.0 #
qG.scale_c_qbus_testing  = 1.0 #1.0 #
qG.scale_c_sflow_testing = 1.0 #1.0 #

# test
# %%
include("./test_functions.jl")

# %%
# README: 1) make sure the ctg solver has a sufficiently high tolerance setting!
#         2) make sure standard gradients are being used
qG.pqbal_grad_type  = "standard"
qG.pcg_tol          = 1e-9
epsilon             = 1e-5     # maybe set larger when dealing with ctgs + krylov solver..

# other types
qG.constraint_grad_is_soft_abs = false
qG.acflow_grad_is_soft_abs     = false
qG.reserve_grad_is_soft_abs    = false

#
# %% 1. transformer phase shift (phi) =======================================================================
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)
epsilon = 1e-4
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.phi[tii][ind]

# perturb and test
stt.phi[tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)
#println((dzdx - dzdx_num)/dzdx_num)

# %% 2. transformer winding ratio (tau) =======================================================================
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.tau[tii][ind]

# perturb and test
stt.tau[tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 3. voltage magnitude =======================================================================
include("./test_functions.jl")
qG.scale_c_pbus_testing  = 1e-4
qG.scale_c_qbus_testing  = 1e-4
qG.scale_c_sflow_testing = 1.0

# %% for flow testing!!
epsilon = 1e-5
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.vm[tii][ind]

# perturb and test
stt.vm[tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 4. voltage phase =======================================================================
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.va[tii][ind]

# perturb and test
stt.va[tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 5. dc -- qfr =======================================================================
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nldc)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.dc_qfr[tii][ind]

# perturb and test
stt.dc_qfr[tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 6. dc -- qto =======================================================================
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nldc)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.dc_qto[tii][ind]

# perturb and test
stt.dc_qto[tii][ind] += epsilon 
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
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nldc)); (ind == 0 ? ind = 1 : ind = ind)
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.dc_pfr[tii][ind]

# perturb and test
stt.dc_pfr[tii][ind] += epsilon 
stt.dc_pto[tii][ind] -= epsilon 
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
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nl)); (ind == 0 ? ind = 1 : ind = ind)

# to ensure we're not tipping over into some new space
stt.u_on_acline[tii][ind]               = 0.9  # for su :) -- reverse for sd :)
stt.u_on_acline[prm.ts.tmin1[tii]][ind] = 0.5  # drives infeasibility!!!
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = copy(mgd.u_on_acline[tii][ind])

# perturb and test
stt.u_on_acline[tii][ind] += epsilon 
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
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)

# to ensure we're not tipping over into some new space
stt.u_on_xfm[tii][ind]                   = 0.9  # for su :) -- reverse for sd :)
stt.u_on_xfm[prm.ts.tmin1[tii]][ind] = 0.5
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx    = mgd.u_on_xfm[tii][ind]

# perturb and test
stt.u_on_xfm[tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 10. :u_step_shunt =======================================================================
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nsh)) # -- most have a 0-bs/gs value, so choose wisely :)

# to ensure we're not tipping over into some new space
stt.u_step_shunt[tii][ind] = 0.4
z0    = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx  = copy(mgd.u_step_shunt[tii][ind])

# perturb and test
stt.u_step_shunt[tii][ind] += epsilon 
zp = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 11. devices =======================================================================
#
# loop over time
for tii in prm.ts.time_keys
    stt.u_on_dev[tii] = rand(sys.ndev)
end

# %% README: if you perturb over the max power, the device cost will error
#         out -- this happens a lot.
#quasiGrad.clip_all!(prm, qG, stt, sys)
include("./test_functions.jl")

qG.pg_tol = 0.0
qG.constraint_grad_weight = 1e-6

qG.scale_c_pbus_testing  = 1.0
qG.scale_c_qbus_testing  = 1.0
qG.scale_c_sflow_testing = 1.0 #1e-4

qG.pqbal_grad_type  = "standard"
qG.pcg_tol          = 1e-9
epsilon             = 1e-5

# other types
qG.constraint_grad_is_soft_abs = false
qG.acflow_grad_is_soft_abs     = false
qG.reserve_grad_is_soft_abs    = false

#              1         2      3       4       5       6      7        8          9          10           11        12      13
var     = [:u_on_dev, :dev_q, :p_on, :p_rgu, :p_rgd, :p_scr, :p_nsc, :p_rru_on, :p_rrd_on, :p_rru_off, :p_rrd_off, :q_qru, :q_qrd]    
vii     = 1
tii     = Int8(round(rand(1)[1]*sys.nT)); (tii == 0 ? tii = Int8(1) : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.ndev)); (ind == 0 ? ind = 1 : ind = ind)

epsilon = 1e-5
z0      = calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
mgd_v   = getfield(mgd, var[vii])
stt_v   = getfield(stt, var[vii])
dzdx    = copy(mgd_v[tii][ind])

# perturb and test
stt_v[tii][ind] += epsilon 
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
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
    quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

    # write a solution :)
    soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
    quasiGrad.write_solution(data_dir*file_name, qG, soln_dict, scr)
end

# %% loop over time
for tii in prm.ts.time_keys
    stt.u_on_dev[tii] = rand(sys.ndev)
end