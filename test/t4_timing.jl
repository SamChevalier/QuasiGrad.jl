# test solver itself :)
using quasiGrad
using Revise

# include("../src/quasiGrad_dual.jl")
include("./test_functions.jl")

# files
path = "../GO3_testcases/C3S0_20221208/D1/C3S0N00073/scenario_002.json"
path = "../GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E1_20230214/D1/C3E1N01576D1/scenario_117.json"

# load
jsn = quasiGrad.load_json(path)

# %% initialize
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

# %%
@time quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% Timing tests
# 
# flush the gradient -- both master grad and some of the gradient terms
print("t1: ")
@time quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

# clip all basic states (i.e., the states which are iterated on)
print("t2: ")
@time quasiGrad.clip_all!(prm, qG, stt)

# compute network flows and injections
qG.eval_grad = true

print("t3: ")
@time quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)

print("t4: ")
@time quasiGrad.xfm_flows!(bit, grd, idx, msc, prm, qG, stt, sys)

print("t5: ")
@time quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)

# device powers
print("t6: ")
@time quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)

print("t7: ")
@time quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)

print("t8a: ")
@time quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

print("t8b: ")
@time quasiGrad.device_reactive_powers!(idx, prm, qG, stt, sys)

print("t9: ")
@time quasiGrad.energy_costs!(grd, prm, qG, stt, sys)

print("t10: ")
@time quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)

# ==========
print("t11: ")
@time quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

print("t12: ")
@time quasiGrad.device_reserve_costs!(prm, qG, stt)

# now, we can compute the power balances
print("t13: ")
@time quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)

# compute reserve margins and penalties
print("t14: ")
@time quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

# score the contingencies and take the gradients
print("t15: ")
@time quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

print("t16: ")
@time quasiGrad.score_zt!(idx, prm, qG, scr, stt)

print("t17: ")
@time quasiGrad.score_zbase!(qG, scr)

print("t18: ")
@time quasiGrad.score_zms!(scr)

# compute the master grad
print("t19: ")
@time quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
println("")

# %%
print("t20: ")
@time quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct);


# %%
@time quasiGrad.clip_all!(prm, qG, stt)
@btime quasiGrad.clip_all!(prm, qG, stt)

# %% take an adam step
adm_step    = 0
beta1       = qG.beta1
beta2       = qG.beta2
beta1_decay = 1.0
beta2_decay = 1.0
run_adam    = true
alpha       = copy(qG.alpha_0)

# %%
@btime quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

# %%
ProfileView.@profview quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)


# %%
qG.adam_max_time = 2.0

ProfileView.@profview quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# %% ------------------------------
# @code_warntype quasiGrad.energy_penalties!(grd, prm, qG, scr, stt, sys)
# @code_warntype quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)


# %% --- write
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% write a solution :)
soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
quasiGrad.write_solution(data_dir*file_name, qG, soln_dict, scr)

# %% ===================
# using ProfileView
# ProfileView.@profview quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
ProfileView.@profview quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

# %%
include("../src/core/contingencies.jl")
using InvertedIndices
#ProfileView.@profview @code_warntype
# %%
@benchmark quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
# %% ====
include("../src/core/contingencies.jl")

@btime special_wmi_update(ctd[ctg_ii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], rhs);

# %%

@btime y0 - u*g*quasiGrad.dot(u, x)
# %%
y0 = copy(ctd[ctg_ii])
u  = copy(ntk.u_k[ctg_ii])
g  = copy(ntk.g_k[ctg_ii])
x  = copy(rhs)

# %%
y = copy(y0)
s = 0.0
# loop once for the dot
for nzu_idx in quasiGrad.rowvals(u)
    s += u[nzu_idx] * x[nzu_idx]
end

# loop again for subtraction
gs = g*s
for nzu_idx in quasiGrad.rowvals(u)
    y[nzu_idx] = y0[nzu_idx] - gs*u[nzu_idx]
end
# %%
rhs = randn(616)
ctg_ii = 1
@btime ctd[1] -  ntk.u_k[1]

uk = Vector(ntk.u_k[ctg_ii])
uk = Vector(ntk.u_k[ctg_ii])
uks = ntk.u_k[ctg_ii]
# %%

@btime ctd[1] - Vector(ntk.u_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], rhs));
@btime ctd[1] - uk*quasiGrad.dot(uk, rhs);
@btime ctd[1] - uk*quasiGrad.dot(ntk.u_k[ctg_ii], rhs);
#@btime ctd[1] - Vector(ntk.u_k[ctg_ii]*quasiGrad.dot(uk, rhs));

# %%
uk = 0*Vector(ntk.u_k[ctg_ii])
wk = 0*Vector(ntk.w_k[ctg_ii])
rhs_s = copy(wks)
rhs_s .= rhs
@btime quasiGrad.dot(wk, rhs);
@btime quasiGrad.dot(wks, rhs);
@btime quasiGrad.dot(wks, rhs_s);

# %%

@btime rhs2 = rhs .- 5.0
# %%

@btime quasiGrad.dot(wks, rhs)
@btime quasiGrad.dot(wk, rhs)

# %%
tii = :t1
bus = 2
alpha = randn(600) 
@btime mgd[:vm][tii][bus] += sum(-alpha[bus-1]*grd[:sh_p][:vm][tii][idx.sh[bus]]; init=0.0)

# %%
@btime grd[:sh_p][:vm][tii]

@btime @view grd[:sh_p][:vm][tii]

# %%
ctg_ii = 1
t_ind = 1
c  = randn(616)
bt = randn(853)

tt = Matrix(ntk.Yfr)

@btime ctb[t_ind] - ntk.u_k[ctg_ii]*quasiGrad.dot(ntk.w_k[ctg_ii],c)

theta_k = ctb[t_ind] - ntk.u_k[ctg_ii]*quasiGrad.dot(ntk.w_k[ctg_ii],c)

@btime pflow_k = ntk.Yfr*theta_k  + bt
# %%

@btime pflow_k = tt*theta_k  + bt

# %%
@btime quasiGrad.mul!(pflow_k,tt,theta_k)

# %%

t = randn(1000)
ind = 1:100
@btime t[ind] .= 0

t = randn(1000)
ind = 1:100
@btime t[ind] .= 0.0

# %% ============== 
pflow_k = randn(853)
@btime sfr     = sqrt.(flw[:ac_qfr].^2 + pflow_k.^2);
@btime sfr2    = abs2.(flw[:ac_qfr],pflow_k);

# %%
vv = flw[:ac_qfr].^2;
@btime sqrt.(vv + pflow_k.^2);

# %%
t = randn(100000)

@btime fill!(t,0.0);
@btime z = zeros(100000);

# %% 
v1 = randn(100000)
v2 = randn(100000)

f1(v1, v2) = v1 + v2
f2(v1, v2) = v1 .+ v2

@btime v3 = f1(v1, v2)
@btime v4 = f2(v1, v2);

# %%

@time quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

# %%
line = 1
tii = :t1
@time  vmfrqfr = grd[:acline_qfr][:vmfr][tii][line]

# %%
@btime update_subset = upd[var_key][tii]