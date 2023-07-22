# test solver itself :)
using BenchmarkTools
using quasiGrad
using Revise

# include("../src/quasiGrad_dual.jl")
# include("./test_functions.jl")

# files
path = "../GO3_testcases/C3S0_20221208/D1/C3S0N00073/scenario_002.json"
path = "../GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"

path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N01576D1/scenario_007.json"

# load
jsn = quasiGrad.load_json(path)

# initialize
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, perturb_states=false);

quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# %% Timing tests
#qG.skip_ctg_eval = true
#adm_step    = 0
#beta1       = qG.beta1
#beta2       = qG.beta2
#beta1_decay = 1.0
#beta2_decay = 1.0
#run_adam    = true
#beta1_decay = beta1_decay*beta1
#beta2_decay = beta2_decay*beta2
#
## %% ====
#adm_step += 1
#quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
#quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)
## %% Timing tests
#
qG.skip_ctg_eval = true
qG.num_threads = 8
@btime quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% choose step sizes
vmva_scale    = 1e-5
xfm_scale     = 1e-5
dc_scale      = 1e-4
power_scale   = 1e-4
reserve_scale = 1e-4
bin_scale     = 1e-4
qG.alpha_0 = Dict(:vm     => vmva_scale,
                :va     => vmva_scale,
                # xfm
                :phi    => xfm_scale,
                :tau    => xfm_scale,
                # dc
                :dc_pfr => dc_scale,
                :dc_qto => dc_scale,
                :dc_qfr => dc_scale,
                # powers -- decay this!!
                :dev_q  => power_scale,
                :p_on   => power_scale,
                # reserves
                :p_rgu     => reserve_scale,
                :p_rgd     => reserve_scale,
                :p_scr     => reserve_scale,
                :p_nsc     => reserve_scale,
                :p_rrd_on  => reserve_scale,
                :p_rrd_off => reserve_scale,
                :p_rru_on  => reserve_scale,
                :p_rru_off => reserve_scale,
                :q_qrd     => reserve_scale,
                :q_qru     => reserve_scale,
                # bins
                :u_on_xfm     => bin_scale,
                :u_on_dev     => bin_scale,
                :u_step_shunt => bin_scale,
                :u_on_acline  => bin_scale)

# %% ==============
qG.print_freq                  = 1
qG.num_threads                 = 1
qG.apply_grad_weight_homotopy  = false
qG.take_adam_pf_steps          = false
qG.pqbal_grad_type             = "standard" #"soft_abs" #
qG.constraint_grad_is_soft_abs = false
qG.acflow_grad_is_soft_abs     = false
qG.reserve_grad_is_soft_abs    = false
qG.skip_ctg_eval               = true
qG.beta1                       = 0.9
qG.beta2                       = 0.99
qG.pqbal_grad_eps2             = 1e-8

qG.adam_max_time = 250.0
quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# %%

qG.eval_grad = true
qG.num_threads = 1

print("t1: ")
@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

print("t2: ")
@btime quasiGrad.clip_all!(prm, qG, stt)

print("t3: ")
@btime quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)

print("t4: ")
@btime quasiGrad.xfm_flows!(bit, grd, idx, msc, prm, qG, stt, sys)

print("t5: ")
@btime quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)

print("t6: ")
@btime quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)

print("t7: ")
@btime quasiGrad.device_startup_states!(grd, idx, mgd, msc, prm, qG, stt, sys)

print("t8a: ")
@btime quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

print("t8b: ")
@btime quasiGrad.device_reactive_powers!(idx, prm, qG, stt)

print("t9: ")
@btime quasiGrad.energy_costs!(grd, prm, qG, stt, sys)

print("t10: ")
@btime quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)

print("t11: ")
@btime quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

print("t12: ")
@btime quasiGrad.device_reserve_costs!(prm, qG, stt)

print("t13: ")
@btime quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)

print("t14: ")
@btime quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

print("t15: ")
    # @time quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

print("t16: ")
@btime quasiGrad.score_zt!(idx, prm, qG, scr, stt)

print("t17: ")
@btime quasiGrad.score_zbase!(qG, scr)

print("t18: ")
@btime quasiGrad.score_zms!(scr)

print("t19: ")
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)
println("")

print("t20: ")
@btime quasiGrad.adam!(adm, 0.9, 0.99, 0.9, 0.9, mgd, prm, qG, stt, upd)
println("")

# %% ===========================
print("t20: ")
@btime quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct);


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
# @code_warntype quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)


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
@btime mgd.vm[tii][bus] += sum(-alpha[bus-1]*grd.sh_p.vm[tii][idx.sh[bus]]; init=0.0)

# %%
@btime grd.sh_p.vm[tii]

@btime @view grd.sh_p.vm[tii]

# %%
ctg_ii = 1
tii = 1
c  = randn(616)
bt = randn(853)

tt = Matrix(ntk.Yfr)

@btime ctb[tii] - ntk.u_k[ctg_ii]*quasiGrad.dot(ntk.w_k[ctg_ii],c)

theta_k = ctb[tii] - ntk.u_k[ctg_ii]*quasiGrad.dot(ntk.w_k[ctg_ii],c)

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
@btime sfr     = sqrt.(flw.ac_qfr.^2 + pflow_k.^2);
@btime sfr2    = abs2.(flw.ac_qfr,pflow_k);

# %%
vv = flw.ac_qfr.^2;
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

@time quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)

# %%
line = 1
tii = :t1
@time  vmfrqfr = grd.acline_qfr.vmfr[tii][line]

# %%
@btime update_subset = upd[var_key][tii]

# %%
@time argmax([stt.u_on_dev[tii_inst][dev] for tii_inst in idx.Ts_sus_jft[dev][tii][ii]])

# %%

function f(stt::quasiGrad.State, idx::quasiGrad.Idx, dev::Union{Int32,Int64}, tii::Int8, ii::Int64)
    argmax(@view stt.u_on_dev[dev][idx.Ts_sus_jft[dev][tii][ii]])
end

# %%
b = 0

@time f(stt, idx, dev, tii, ii);

# %%
argmax([stt.u_on_dev[tii_inst][dev] for tii_inst in idx.Ts_sus_jft[dev][tii][ii]])

argmax(stt.u_on_dev[tii_inst][dev] for tii_inst in idx.Ts_sus_jft[dev][tii][ii])


# %%
tt = [zeros(1000) for ii in 1:1000]

@btime sum(tt[ii][10] for ii in 1:1000);
@btime sum(tt[10][1:1000]);

# %%
for jj in 1:2
    for kk in 1:5
        if kk == 3
            break
        end
        println(kk)
    end
end

# %%
dev = 177
tii = 2

cst = prm.dev.cum_cost_blocks[dev][tii][1]  # cost for each block (leading with 0)
pbk = prm.dev.cum_cost_blocks[dev][tii][2]  # power in each block (leading with 0)
pcm = prm.dev.cum_cost_blocks[dev][tii][3]  # accumulated power for each block!
nbk = length(pbk)

# get the cost!
stt.zen_dev[tii][dev] = dt*sum(cst[ii]*max(min(stt.dev_p[tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)

if stt.dev_p[tii][dev] == 0.0
    stt.zen_dev[tii][dev] = 0.0
else
    for ii in 2:nbk
        if stt.dev_p[tii][dev] >= pcm[ii]
            stt.zen_dev[tii][dev] += pbk[ii]*cst[ii]
        else
            stt.zen_dev[tii][dev] += (stt.dev_p[tii][dev] - pcm[ii-1])*cst[ii]
            break
        end
    end

    # add dt
    # stt.zen_dev[tii][dev] = stt.zen_dev[tii][dev]*dt
end

# %%

stt.p_on[tii] .= max.(stt.p_on[tii], stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii])

# %%

kk  = [zeros(sys.ndev) for ii in 1:sys.nT]
kk2 = [zeros(sys.nT) for ii in 1:sys.ndev]

# %%
function fr(kk::Vector{Vector{Float64}}, kk2::Vector{Vector{Float64}}, prm::quasiGrad.Param, sys::quasiGrad.System)
    for dev in 1:sys.ndev
        for tii in prm.ts.time_keys
            kk2[dev][tii] = kk[tii][dev]
        end
    end
end

# %%
@btime fr(kk, kk2, prm, sys)

# %%
function ff(t::Union{Int32,Int64})
    println(typeof(t))
    t^2
    println(t)
end


# %%

for var_key in adm.keys
    #println(var_key)
    #println(typeof(var_key))
    #println(getfield(adm,var_key.m))

    adam_states = getfield(adm,var_key)     
    state       = getfield(stt,var_key)

    state[1] = adam_states.m[1]

    adam_states.m[1] .= -55.0

    ## so, only progress if, either, this is standard adam (not in the pf stepper),
    ## or the variable is a power flow variable!
    #if (var_key in qG.adam_pf_variables) || standard_adam
    #    # loop over all time
    #    for tii in prm.ts.time_keys
    #        # states to update                                            
    #        if var_key in keys(upd)
    #            # => (var_key in keys(upd)) ? (update = upd[var_key][tii]) : (update = Colon())
    #            #    the above caused weird type instability, so we just copy and paste
    #            update_subset = upd[var_key][tii]
#
    #            # note -- it isn't clear how best to use @view -- it seems to be helpful when calling
    #            # an array subset when adding/subtracting, but now when taking products, etc.
#
    #            # update adam moments
    #                # => clipped_grad, if helpful! = clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max)
    #            adm.var_key.m[tii][update_subset] .= beta1.*(@view adm.var_key.m[tii][update_subset]) .+ (1.0-beta1).*mgd[var_key][tii][update_subset]
    #            adm.var_key.v[tii][update_subset] .= beta2.*(@view adm.var_key.v[tii][update_subset]) .+ (1.0-beta2).*(mgd[var_key][tii][update_subset].^2.0)
    #            stt.var_key[tii][update_subset]   .= (@view stt.var_key[tii][update_subset]) .- qG.alpha_0[var_key].*(adm.var_key.m[tii][update_subset]./(1.0-beta1_decay))./(sqrt.(adm.var_key.v[tii][update_subset]./(1.0-beta2_decay)) .+ qG.eps)
    #            
    #        else 
    #            # update adam moments
    #                # => clipped_grad, if helpful!  = clamp.(mgd[var_key][tii], -qG.grad_max, qG.grad_max)
    #            adm.var_key.m[tii] .= beta1.*adm.var_key.m[tii] .+ (1.0-beta1).*mgd[var_key][tii]
    #            adm.var_key.v[tii] .= beta2.*adm.var_key.v[tii] .+ (1.0-beta2).*(mgd[var_key][tii].^2.0)
    #            stt.var_key[tii]   .= stt.var_key[tii] .- qG.alpha_0[var_key].*(adm.var_key.m[tii]./(1.0-beta1_decay))./(sqrt.(adm.var_key.v[tii]./(1.0-beta2_decay)) .+ qG.eps)
    #        end
    #    end
    #end
end