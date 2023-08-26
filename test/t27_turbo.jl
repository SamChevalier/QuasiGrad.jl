using quasiGrad
using Revise

# files ===
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"

jsn  = quasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt)
qG.num_threads = 10

# %% ==========================
GC.gc()

print("t1: ")
@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

print("t2: ")
@btime quasiGrad.clip_all!(prm, qG, stt, sys)

print("t3: ")
@btime quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)

print("t4: ")
@btime quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)

print("t5: ")
@btime quasiGrad.shunts!(grd, idx, prm, qG, stt)

print("t6: ")
@btime quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)

print("t7: ")
@btime quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)

print("t8a: ")
@btime quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

print("t8b: ")
@btime quasiGrad.device_reactive_powers!(idx, prm, qG, stt)

print("t9: ")
@btime quasiGrad.energy_costs!(grd, prm, qG, stt, sys)

print("t10: ")
@btime quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)

# %%
print("t11: ")
@btime quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

# %%

print("t12: ")
@btime quasiGrad.device_reserve_costs!(prm, qG, stt)

print("t13: ")
@btime quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

print("t14: ")
@btime quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

# %%
print("t15: ")
ntk.s_max .= 100000.0
@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
# %%
print("t16: ")
@btime quasiGrad.score_zt!(idx, prm, qG, scr, stt)

print("t17: ")
@btime quasiGrad.score_zbase!(qG, scr)

print("t18: ")
@btime quasiGrad.score_zms!(scr)

# %%

print("t19: ")
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
println("")

# %%
v1 = randn(10)
v2 = randn(10)
v3 = randn(10)

# %%
quasiGrad.@turbo v1 .= v3.*v3 .+ v2

# %%
quasiGrad.@turbo stt.cos_ftp[tii] .= 2.2.*quasiGrad.LoopVectorization.pow_fast.(stt.va_fr[tii] .- stt.va_to[tii])

# %%

quasiGrad.@turbo v1 .= 1.1 .* quasiGrad.LoopVectorization.pow_fast.(v3, 2)

# %% ===
print("t20: ")
@btime quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
println("")

# %%
@code_warntype quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)

# %% === === 
ntk.s_max .= 1.0
qG.eval_grad = true
GC.gc()

@btime quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %%
qG.ctg_adam_counter         = 0   
qG.ctg_solve_frequency      = 3   
qG.always_solve_ctg         = false
@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %%
s = 0
b = prm.ts.time_keys
quasiGrad.@turbo for ii in eachindex(b)
    s += ii
end

# %%
quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)


# %%
f1() = quasiGrad.mul!(ctg.pflow_k[thrID], ntk.Yfr, ctg.theta_k[thrID])
f2() = ctg.pflow_k[thrID] .= ntk.Yfr*ctg.theta_k[thrID]
# %%
@btime f1()
@btime f2()

# %%
a = randn(100000)
b = randn(100000)
f1() = quasiGrad.dot(a, b)
f2() = quasiGrad.@turbo quasiGrad.dot(a, b)
f3() = @fastmath quasiGrad.@turbo quasiGrad.dot(a, b)

@btime f1()
@btime f2()
@btime f3()

# %% initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);

# %% run copper plate ED
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt)
qG.num_threads = 10

# %% ===============
@btime quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)



# %% line parameters
g_sr = prm.acline.g_sr
b_sr = prm.acline.b_sr
b_ch = prm.acline.b_ch
g_fr = prm.acline.g_fr
b_fr = prm.acline.b_fr
g_to = prm.acline.g_to
b_to = prm.acline.b_to

# call penalty costs
cs = prm.vio.s_flow * qG.scale_c_sflow_testing

# loop over time
tii = Int8(1)

#@batch per=core for tii in prm.ts.time_keys
# = > @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys

    # duration
    dt = prm.ts.duration[tii]

    # organize relevant line values
    vm_fr = @view stt.vm[tii][idx.acline_fr_bus]
    va_fr = @view stt.va[tii][idx.acline_fr_bus]
    vm_to = @view stt.vm[tii][idx.acline_to_bus]
    va_to = @view stt.va[tii][idx.acline_to_bus]
    
    # tools
    stt.cos_ftp[tii]  .= cos.(va_fr .- va_to)
    stt.sin_ftp[tii]  .= sin.(va_fr .- va_to)
    stt.vff[tii]      .= vm_fr.^2
    stt.vtt[tii]      .= vm_to.^2
    stt.vft[tii]      .= vm_fr.*vm_to
    
    # evaluate the function? we always need to in order to get the grd
    #
    # active power flow -- from -> to
    stt.pfr[tii] .= (g_sr.+g_fr).*stt.vff[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    stt.acline_pfr[tii] .= stt.u_on_acline[tii].*stt.pfr[tii]
    
    # reactive power flow -- from -> to
    stt.qfr[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*stt.vff[tii] .+ (b_sr.*stt.cos_ftp[tii] .- g_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    stt.acline_qfr[tii] .= stt.u_on_acline[tii].*stt.qfr[tii]
    
    # apparent power flow -- to -> from
    stt.acline_sfr[tii] .= sqrt.(stt.acline_pfr[tii].^2 .+ stt.acline_qfr[tii].^2)
    
    # active power flow -- to -> from
    stt.pto[tii] .= (g_sr.+g_to).*stt.vtt[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .+ b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    stt.acline_pto[tii] .= stt.u_on_acline[tii].*stt.pto[tii]
    
    # reactive power flow -- to -> from
    stt.qto[tii] .= (.-b_sr.-b_to.-b_ch./2.0).*stt.vtt[tii] .+ (b_sr.*stt.cos_ftp[tii] .+ g_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    stt.acline_qto[tii] .= stt.u_on_acline[tii].*stt.qto[tii]

    # apparent power flow -- to -> from
    stt.acline_sto[tii] .= sqrt.(stt.acline_pto[tii].^2 .+ stt.acline_qto[tii].^2)
    
    # penalty functions and scores
    stt.acline_sfr_plus[tii] .= stt.acline_sfr[tii] .- prm.acline.mva_ub_nom
    stt.acline_sto_plus[tii] .= stt.acline_sto[tii] .- prm.acline.mva_ub_nom
    stt.zs_acline[tii]       .= (dt*cs).*max.(stt.acline_sfr_plus[tii], stt.acline_sto_plus[tii], 0.0);
# end

# %%
#@btime stt.pfr[tii] .= (g_sr.+g_fr).*stt.vff[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
@btime stt.pfr[tii] .= (g_sr.+g_fr).*stt.vff[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vft[tii];
@btime quasiGrad.@turbo stt.pfr[tii] .= (g_sr.+g_fr).*stt.vff[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vft[tii];

# %%
@btime quasiGrad.@turbo stt.cos_ftp[tii]  .= .+ .-(va_fr .- va_to)

# %%
@btime quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys);

# %%

@time quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys);

# %%
quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)

# %%
quasiGrad.@batch per=core for tii in prm.ts.time_keys
#quasiGrad.@floop quasiGrad.ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
      println(Threads.threadid())
end

# %%
tii = Int8(1)
vm_fr = @view stt.vm[tii][idx.acline_fr_bus]

quasiGrad.@turbo vm_fr.^2

# %%
@btime quasiGrad.master_grad_zs_acline!(tii, idx, grd, mgd, qG, stt, sys)

# %%
@btime quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, qG, stt, sys)

# %%
@btime quasiGrad.master_grad_zp!(tii, prm, idx, grd, mgd, sys, run_devs = true)

# %%

@btime quasiGrad.master_grad_zq!(tii, prm, idx, grd, mgd, sys)
# %%
tii = Int8(1)
quasiGrad.@turbo for ln in 1:sys.nl
      # see binaries at the bottom
      #
      # update the master grad -- pfr
      mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrpfr[tii][ln]
      mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtopfr[tii][ln]
      mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrpfr[tii][ln]
      mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatopfr[tii][ln]

      # update the master grad -- qfr
      mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrqfr[tii][ln]
      mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtoqfr[tii][ln]
      mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrqfr[tii][ln]
      mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatoqfr[tii][ln]

      # update the master grad -- pto
      mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrpto[tii][ln]
      mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtopto[tii][ln]
      mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrpto[tii][ln]
      mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatopto[tii][ln]

      # update the master grad -- qto
      mgd.vm[tii][idx.acline_fr_bus[ln]] += stt.vmfrqto[tii][ln]
      mgd.vm[tii][idx.acline_to_bus[ln]] += stt.vmtoqto[tii][ln]
      mgd.va[tii][idx.acline_fr_bus[ln]] += stt.vafrqto[tii][ln]
      mgd.va[tii][idx.acline_to_bus[ln]] += stt.vatoqto[tii][ln]

      #if qG.update_acline_xfm_bins
      #    mgd.u_on_acline[tii][ln] += stt.uonpfr[tii][ln]
      #    mgd.u_on_acline[tii][ln] += stt.uonqfr[tii][ln]
      #    mgd.u_on_acline[tii][ln] += stt.uonpto[tii][ln]
      #    mgd.u_on_acline[tii][ln] += stt.uonqto[tii][ln]
      #end
  end

  # %%
  @turbo stt.acline_sfr[tii] .= sqrt.(stt.acline_pfr[tii].^2 .+ stt.acline_qfr[tii].^2)

# %%
a = abs.(randn(100000))
b = randn(100000)
c = randn(100000)
d = zeros(100000)

@btime d .= a.*b .+ c;
@btime @inbounds d .= a.*b .+ c;
@btime @fastmath d .= a.*b .+ c;
@btime @fastmath @inbounds d .= a.*b .+ c;

# %%
a = abs.(randn(100000))
d = zeros(100000)

@btime d .= sqrt.(a)
@btime @fastmath d .= sqrt.(a)
@btime d .= quasiGrad.LoopVectorization.sqrt_fast.(a)
@btime quasiGrad.@turbo d .= quasiGrad.LoopVectorization.sqrt_fast.(a)

# %%
f() =  @fastmath @inbounds p_slack = 
    sum(stt.dev_p[tii][pr] for pr in idx.pr_devs) -
    sum(stt.dev_p[tii][cs] for cs in idx.cs_devs) - 
    sum(stt.sh_p[tii])
fs() =  p_slack = 
    sum(stt.dev_p[tii][pr] for pr in idx.pr_devs) -
    sum(stt.dev_p[tii][cs] for cs in idx.cs_devs) - 
    sum(stt.sh_p[tii])

# %%
@btime f()
@btime fs()

# %%
ff() = @inbounds sum(stt.dev_p[tii][pr] for pr in idx.pr_devs)

# %%
@time ff()

# %%
E, Efr, Eto = quasiGrad.build_incidence(idx, prm, stt, sys)
Er = E[:,2:end]
ErT = copy(Er')

u_k = [zeros(sys.nb-1) for ctg_ii in 1:sys.nctg] # Dict(ctg_ii => zeros(sys.nb-1) for ctg_ii in 1:sys.nctg) # Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
g_k = zeros(sys.nctg) # => Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
z_k = [zeros(sys.nac) for ctg_ii in 1:sys.nctg]

# loop over components (see below for comments!!!)
# get the ordered names of all components
ac_ids = [prm.acline.id; prm.xfm.id ]
ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
ctg_out_ind = Dict(ctg_ii => Vector{Int64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
ctg_params  = Dict(ctg_ii => Vector{Float64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)


for ctg_ii in 1:sys.nctg
    cmpnts = prm.ctg.components[ctg_ii]
    for (cmp_ii,cmp) in enumerate(cmpnts)
        cmp_index = findfirst(x -> x == cmp, ac_ids) 
        ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
        ctg_params[ctg_ii][cmp_ii]  = -ac_b_params[cmp_index]
    end
end

function functt()
    quasiGrad.@floop quasiGrad.ThreadedEx(basesize = sys.nctg ÷ qG.num_threads) for ctg_ii in 1:sys.nctg
    # this code is optimized -- see above for comments!!!
        @fastmath u_k[ctg_ii] .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
    end
end

# %%
@btime functt()

# %%
y = u_k[1]
b = Er[ctg_out_ind[1][1],:]

LinearAlgebra.ldiv!(u_k[1], Ybr_Ch, Er[ctg_out_ind[1][1],:])


# ================
path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_997.json"
# path = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_963.json"
InFile1 = path
jsn = quasiGrad.load_json(InFile1)

# ========
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

for tii in prm.ts.time_keys
    stt.vm[tii] .= 1.0
    stt.va[tii] .= 0.0
    lbf.step[:step][tii] = 0.1
end

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %%
stt = deepcopy(stt0);
qG.pqbal_grad_eps2 = 1e-2
qG.eval_grad = true

# %%
qG.num_lbfgs_steps = 100
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
qG.max_linear_pfs       = 25
qG.max_linear_pfs_total = 25
xi, par = quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys);


# %%
@btime emergency_stop = quasiGrad.solve_pf_lbfgs!(lbf, mgd, prm, qG, stt, upd);

# %% quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt = deepcopy(stt0);

qG.max_pf_dx            = 1e-4
qG.max_linear_pfs       = 25
qG.max_linear_pfs_total = 25
quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)

# %%
qG.max_linear_pfs       = 25
qG.max_linear_pfs_total = 25
xi, par = quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys);

# %%
stt = deepcopy(stt0);
qG.num_lbfgs_steps = 250

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
# %%
stt0 = deepcopy(stt);

# %%
stt = deepcopy(stt0);

qG.max_pf_dx            = 1e-4
qG.max_linear_pfs       = 10
qG.max_linear_pfs_total = 10

xi, par = quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys);

# %% ==============


#include_sus = true
#quasiGrad.solve_economic_dispatch!(idx, prm, qG, scr, stt, sys, upd, include_sus_in_ed=include_sus)
#quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys) 

zp = sum(sum(stt.zp[tii]) for tii in prm.ts.time_keys)
zq = sum(sum(stt.zq[tii]) for tii in prm.ts.time_keys)
println(zq)
println(zp)
# %% ================

quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)

# %% 
quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys) 
zp = sum(sum(stt.zp[tii]) for tii in prm.ts.time_keys)
zq = sum(sum(stt.zq[tii]) for tii in prm.ts.time_keys)
println(zq)
#println(zp+zq)

# %%
tii = Int8(1)
bus = 1

# at this time, compute the pr and cs upper and lower bounds across all devices
stt.dev_qlb[tii] .= stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]
stt.dev_qub[tii] .= stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]

# for devices with reactive power equality constraints, just
# set the associated upper and lower bounds to the given production
for dev in idx.J_pqe
    stt.dev_qlb[tii][dev] = copy(stt.dev_q[tii][dev])
    stt.dev_qub[tii][dev] = copy(stt.dev_q[tii][dev])
end

# note: clipping is based on the upper/lower bounds, and not
# based on the beta linking equations -- so, we just treat
# that as a penalty, and not as a power balance factor
# 
# also, compute the dc line upper and lower bounds
dcfr_qlb = prm.dc.qdc_fr_lb
dcfr_qub = prm.dc.qdc_fr_ub
dcto_qlb = prm.dc.qdc_to_lb
dcto_qub = prm.dc.qdc_to_ub

# how does balance work? for reactive power,
# 0 = qb_slack + (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
#
# so, we take want to set:
# -qb_slack = (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
#for bus in 1:sys.nb

    # reactive power balance
    qb_slack = 
            # shunt        
            sum(stt.sh_q[tii][sh] for sh in idx.sh[bus]; init=0.0) +
            # acline
            sum(stt.acline_qfr[tii][ln] for ln in idx.bus_is_acline_frs[bus]; init=0.0) + 
            sum(stt.acline_qto[tii][ln] for ln in idx.bus_is_acline_tos[bus]; init=0.0) +
            # xfm
            sum(stt.xfm_qfr[tii][xfm] for xfm in idx.bus_is_xfm_frs[bus]; init=0.0) + 
            sum(stt.xfm_qto[tii][xfm] for xfm in idx.bus_is_xfm_tos[bus]; init=0.0)
            # dcline -- not included
            # consumers (positive) -- not included
            # producer (negative) -- not included

    # get limits -- Q
    pr_Qlb   = sum(stt.dev_qlb[tii][pr] for pr  in idx.pr[bus]; init=0.0)
    cs_Qlb   = sum(stt.dev_qlb[tii][cs] for cs  in idx.cs[bus]; init=0.0)
    pr_Qub   = sum(stt.dev_qub[tii][pr] for pr  in idx.pr[bus]; init=0.0) 
    cs_Qub   = sum(stt.dev_qub[tii][cs] for cs  in idx.cs[bus]; init=0.0)
    dcfr_Qlb = sum(dcfr_qlb[dcl]        for dcl in idx.bus_is_dc_frs[bus]; init=0.0)
    dcfr_Qub = sum(dcfr_qub[dcl]        for dcl in idx.bus_is_dc_frs[bus]; init=0.0)
    dcto_Qlb = sum(dcto_qlb[dcl]        for dcl in idx.bus_is_dc_tos[bus]; init=0.0)
    dcto_Qub = sum(dcto_qub[dcl]        for dcl in idx.bus_is_dc_tos[bus]; init=0.0) 
    
    # total: lb < -qb_slack < ub
    qub = cs_Qub + dcfr_Qub + dcto_Qub - pr_Qlb
    qlb = cs_Qlb + dcfr_Qlb + dcto_Qlb - pr_Qub

    # %%

    # now, apply Q
    if -qb_slack >= qub
        # => println("ub limit")
        # max everything out
        for cs in idx.cs[bus]
            stt.dev_q[tii][cs] = copy(stt.dev_qub[tii][cs])
        end
        for pr in idx.pr[bus]
            stt.dev_q[tii][pr] = copy(stt.dev_qlb[tii][pr])
        end
        for dcl in idx.bus_is_dc_frs[bus]
            stt.dc_qfr[tii][dcl] = copy(dcfr_qub[dcl])
        end
        for dcl in idx.bus_is_dc_tos[bus]
            stt.dc_qto[tii][dcl] = copy(dcfr_qub[dcl])
        end
    elseif -qb_slack < qlb
        # => println("lb limit")

        # min everything out
        for cs in idx.cs[bus]
            stt.dev_q[tii][cs] = copy(stt.dev_qlb[tii][cs])
        end
        for pr in idx.pr[bus]
            stt.dev_q[tii][pr] = copy(stt.dev_qub[tii][pr])
        end
        for dcl in idx.bus_is_dc_frs[bus]
            stt.dc_qfr[tii][dcl] = copy(dcfr_qlb[dcl])
        end
        for dcl in idx.bus_is_dc_tos[bus]
            stt.dc_qto[tii][dcl] = copy(dcfr_qlb[dcl])
        end
    else # in the middle -- all good -- no need to copy
        # => println("middle")
        lb_dist  = -qb_slack - qlb
        bnd_dist = qub - qlb
        scale    = lb_dist/bnd_dist

        # apply
        for cs in idx.cs[bus]
            stt.dev_q[tii][cs] = stt.dev_qlb[tii][cs] + scale*(stt.dev_qub[tii][cs] - stt.dev_qlb[tii][cs])
        end
        for pr in idx.pr[bus]
            stt.dev_q[tii][pr] = stt.dev_qub[tii][pr] - scale*(stt.dev_qub[tii][pr] - stt.dev_qlb[tii][pr])
        end
        for dcl in idx.bus_is_dc_frs[bus]
            stt.dc_qfr[tii][dcl] = dcfr_qlb[dcl] + scale*(dcfr_qub[dcl] - dcfr_qlb[dcl])
        end
        for dcl in idx.bus_is_dc_tos[bus]
            stt.dc_qto[tii][dcl] = dcfr_qlb[dcl] + scale*(dcfr_qub[dcl] - dcfr_qlb[dcl])
        end
    end
#end





# %% =============

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys) 

zp = sum(sum(stt.zp[tii]) for tii in prm.ts.time_keys)
zq = sum(sum(stt.zq[tii]) for tii in prm.ts.time_keys)
println(zp+zq)


quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)
#quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys) 
zp = sum(sum(stt.zp[tii]) for tii in prm.ts.time_keys)
zq = sum(sum(stt.zq[tii]) for tii in prm.ts.time_keys)
println(zp+zq)

# %% ==================================== C3S4X_20230809 ==================================== #
#
#    ====================================       D1       ==================================== #
path = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_941.json" # first
solution_file = "C3S4N00617D1_scenario_941"
load_solve_project_write(path, solution_file::String)

# %%
path = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_963.json" # last
solution_file = "C3S4N00617D1_scenario_963"
load_solve_project_write(path, solution_file::String)

#    ====================================       D2       ==================================== #
path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_991.json" # first
solution_file = "C3S4N00073D2_scenario_991"
load_solve_project_write(path, solution_file::String)

# %% ===

path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_997.json" # first
solution_file = "C3S4N00073D2_scenario_997"
load_solve_project_write(path, solution_file::String)

# %%
path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_997.json"

InFile1 = path
jsn = quasiGrad.load_json(InFile1)

adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys) 

zp = sum(sum(stt.zp[tii]) for tii in prm.ts.time_keys)
zq = sum(sum(stt.zq[tii]) for tii in prm.ts.time_keys)
println(zp+zq)


quasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)
quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys) 
zp = sum(sum(stt.zp[tii]) for tii in prm.ts.time_keys)
zq = sum(sum(stt.zq[tii]) for tii in prm.ts.time_keys)
println(zp+zq)

# %%
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys) 

quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
println(zq)

# %% ===

quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

# %%
quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
println(zp+zq)

# %%
stt = deepcopy(stt0);
quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)
quasiGrad.score_solve_pf!(lbf, prm, stt)
zp = sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
zt = zp + zq

# %% ======== ==============
qG.num_lbfgs_steps = 10000

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys)
