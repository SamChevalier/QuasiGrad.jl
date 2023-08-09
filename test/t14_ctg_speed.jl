using quasiGrad
using GLMakie
using Revise
using Plots
using Makie

# ===============
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"

InFile1               = path
TimeLimitInSeconds    = 1500.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# this is the master function which executes quasiGrad.
# 
#
# =====================================================\\
# TT: start time
start_time = time()

# I1. load the system data
jsn = quasiGrad.load_json(InFile1)

@warn "don't use btime for testing ctgs"

# %% I2. initialize the system
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, false, 1.0);

# %% ===============
@code_warntype quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% ===============
qG.score_all_ctgs = false
qG.eval_grad      = true

@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)


# %%
@btime flw.dz_dpinj .= quasiGrad.special_wmi_update(ctd[ctg_ii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.rhs);

# %%
a = ctd[ctg_ii]
b = ntk.u_k[ctg_ii]
c = ntk.g_k[ctg_ii]
d = flw.rhs

@btime flw.dz_dpinj .= quasiGrad.special_wmi_update(a, b, c, d);

# %%
@time quasiGrad.cg!(ctb[tii], ntk.Ybr, flw.c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its)

# %%
@time quasiGrad.get_largest_indices(bit, :xfm_sfr_plus, :xfm_sto_plus);

@time bit.xfm_sfr_plus .= (stt.xfm_sfr_plus .> 0.0) .&& (stt.xfm_sfr_plus .> stt.xfm_sto_plus);
@time bit.xfm_sto_plus .= (stt.xfm_sto_plus .> 0.0) .&& (stt.xfm_sto_plus .> stt.xfm_sfr_plus);
# %%
gamma_fr   = (flw.sfr_vio .> qG.grad_ctg_tol) .&& (flw.sfr_vio .> flw.sto_vio)
gamma_to   = (flw.sto_vio .> qG.grad_ctg_tol) .&& (flw.sto_vio .> flw.sfr_vio)

#
# slow alternative:
    # => max_sfst0 = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(stt.xfm_sfr_plus,stt.xfm_sto_plus)]
    # => ind_fr = max_sfst0 .== 1
    # => ind_to = max_sfst0 .== 2

# %%
quasiGrad.get_largest_ctg_indices(bit, flw, qG, :sfr_vio, :sto_vio)

#t1 = (flw.sfr_vio .> qG.grad_ctg_tol) .&& (flw.sfr_vio .> flw.sto_vio)
#t2 = (flw.sto_vio .> qG.grad_ctg_tol) .&& (flw.sto_vio .> flw.sfr_vio)

#bit.sfr_vio
#bit.sto_vio

# %%
@btime quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys);
@btime quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys);

# %%
quasiGrad.get_largest_ctg_indices(bit, flw, qG, :sfr_vio, :sto_vio)

t1 = (flw.sfr_vio .> qG.grad_ctg_tol) .&& (flw.sfr_vio .> flw.sto_vio)
t2 = (flw.sto_vio .> qG.grad_ctg_tol) .&& (flw.sto_vio .> flw.sfr_vio)

bit.sfr_vio 
bit.sto_vio 

# %%
quasiGrad.get_largest_indices(bit, :acline_sfr_plus, :acline_sto_plus)

t1 = (stt.acline_sfr_plus .> 0.0) .&& (stt.acline_sfr_plus .> stt.acline_sto_plus);
t2 = (stt.acline_sto_plus .> 0.0) .&& (stt.acline_sto_plus .> stt.acline_sfr_plus);

bit.acline_sfr_plus
bit.acline_sto_plus

# %%
quasiGrad.get_largest_indices(bit, :xfm_sfr_plus, :xfm_sto_plus)
t1 = (stt.xfm_sfr_plus .> 0.0) .&& (stt.xfm_sfr_plus .> stt.xfm_sto_plus);
t2 = (stt.xfm_sto_plus .> 0.0) .&& (stt.xfm_sto_plus .> stt.xfm_sfr_plus);

bit.xfm_sfr_plus
bit.xfm_sto_plus

# %%
@time flw.pflow_k .= ntk.Yfr*flw.theta_k;
@time quasiGrad.mul!(flw.pflow_k, ntk.Yfr, flw.theta_k);

# %%
@time flw.rhs .= ntk.Yfr'*(gc.*flw.dsmax_dp_flow);

# %%

@time quasiGrad.mul!(flw.rhs, ntk.Yfr', gc.*flw.dsmax_dp_flow);
@time quasiGrad.mul!(flw.rhs, AA, gc.*flw.dsmax_dp_flow);

# %%
@btime sum(bit.xfm_sfr_plus);
@btime 1 in bit.xfm_sfr_plus;

# %%
f1(bit::quasiGrad.Bit) = sum(bit.xfm_sfr_plus)
f2(bit::quasiGrad.Bit) = 1 in bit.xfm_sfr_plus

# %%
@btime f1($bit)
@btime f2($bit)