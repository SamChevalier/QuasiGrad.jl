using quasiGrad
using Revise

# files ===
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
jsn  = quasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt)
qG.num_threads = 10

# %% ===
@btime quasiGrad.clip_dc!(prm, qG, stt)
@btime quasiGrad.clip_xfm!(prm, qG, stt)
@btime quasiGrad.clip_shunts!(prm, qG, stt)
@btime quasiGrad.clip_voltage!(prm, qG, stt)
@btime quasiGrad.clip_onoff_binaries!(prm, qG, stt)
@btime quasiGrad.transpose_binaries!(prm, qG, stt)
@btime quasiGrad.clip_reserves!(prm, qG, stt)
# %%
@btime quasiGrad.clip_pq!(prm, qG, stt)

# %%
@btime quasiGrad.transpose_binaries!(prm, qG, stt)

# %% ==========================
print("t1: ")
@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

print("t2: ")
@btime quasiGrad.clip_all!(prm, qG, stt)

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

print("t11: ")
@btime quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

print("t12: ")
@btime quasiGrad.device_reserve_costs!(prm, qG, stt)

print("t13: ")
@btime quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

print("t14: ")
@btime quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

print("t15: ")
    # @time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

print("t16: ")
@btime quasiGrad.score_zt!(idx, prm, qG, scr, stt)

print("t17: ")
@btime quasiGrad.score_zbase!(qG, scr)

print("t18: ")
@btime quasiGrad.score_zms!(scr)

print("t19: ")
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
println("")

print("t20: ")
@btime quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
println("")

# %% === === 
ntk.s_max .= 1.0
qG.eval_grad = false
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

#@batch per=thread for tii in prm.ts.time_keys
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
quasiGrad.@batch per=thread for tii in prm.ts.time_keys
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