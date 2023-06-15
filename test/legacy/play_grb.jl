
@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys);

# %%
@btime quasiGrad.clip_all!(prm, qG, stt)

# %%
@btime sum(bit[:acline_sfr_plus])

# %%
@btime bit[:acline_sfr_plus][end] in bit[:acline_sfr_plus];

# %%
@btime quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)

# %%
@time quasiGrad.xfm_flows!(bit, grd, idx, msc, prm, qG, stt, sys);


# %%
@time quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)

# %%
@btime quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)

# %%
@time msc[:scale_fr_x][bit[:xfm_sfr_plus_x]] .= msc[:xfm_sfr_plus_x][bit[:xfm_sfr_plus_x]]./sqrt.(msc[:xfm_sfr_plus_x][bit[:xfm_sfr_plus_x]].^2 .+ qG.acflow_grad_eps2);


# %%
quasiGrad.power_balance_old!(grd, idx, msc, prm, qG, stt, sys)

zp0 = deepcopy(stt[:zp])
zq0 = deepcopy(stt[:zq])

quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)

zp1 = deepcopy(stt[:zp])
zq1 = deepcopy(stt[:zq])


# %%
f1(t::Float64, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, idx::quasiGrad.Idx, v::Vector{Float64}) = t = sum(stt[:dev_p][tii][cs] for cs in [@view idx.cs[bus]])
f2(t::Float64, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, idx::quasiGrad.Idx, v::Vector{Float64}) = t = sum(stt[:dev_p][tii][idx.cs[bus]] for cs in idx.cs[bus])

f3(t::Float64, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, idx::quasiGrad.Idx, v::Vector{Float64}) = t += v[ii] for ii in [1,2,3]

# %%
@time f1(t, stt, idx)
@time f2(t, stt, idx)

@time f3(t, stt, idx, v)

# %% ===========
function f4(t::Float64, v::Vector{Float64}, vv::Vector{Int64})
    for idx in vv
        t += v[idx]
    end
end

# %%
t = 0.0
v = randn(1000)
vv = [1,2,3,4,5,6]

@time f4(t,v,vv)

# %%
bus = 1299
tii = :t15
@time quasiGrad.pq_sums!(bus, idx, msc, stt, tii)

@time sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0)

# %%


@time sum(stt[:dev_p][tii][cs] for cs in idx.cs[bus])

t = 0.0
@time @view stt[:dev_p][tii][idx.cs[bus]]

@time sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0)

# %%
#ind_fr = [1,2,3]

# %%
quasiGrad.soft_abs_grad_ac_new(msc, qG, :acline_sfr_plus, 1)

# %%
@time quasiGrad.soft_abs_grad_vec!(v, qG.acflow_grad_eps2, x);

# %%
ind_fr = 1:143

@time quasiGrad.soft_abs_grad_vec!(msc[:xfm_sfr_plus_x][ind_fr], 0.01, msc[:scale_fr_x][ind_fr])

# %%
@time quasiGrad.soft_abs_grad_vec!(v1, 0.01, msc[:scale_fr_x][ind_fr])

# %%
f(vv::Vector{Float64}, v1::Vector{Float64}, v2::Vector{Float64}) = vv .= 2.0*(v1 .+ v2)
# %%
@btime f(vv, v1, v2);

# %%

f(v::Vector{Float64}, x::Vector{Float64}, qG::quasiGrad.QG) = v.= x./qG.acflow_grad_eps2

# %%
@btime f(x, v, qG);

# %%

# %%
quasiGrad.get_largest_indices(msc, bit, :xfm_sfr_plus_x, :xfm_sto_plus_x);


max_sfst0 = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(msc[:xfm_sfr_plus_x],msc[:xfm_sto_plus_x])]
ind_fr = max_sfst0 .== 1
ind_to = max_sfst0 .== 2

# %%

@time quasiGrad.soft_abs_grad_vec!(bit, msc[:xfm_sfr_plus_x], msc[:scale_fr_x], qG, :xfm_sfr_plus_x);
@time quasiGrad.soft_abs_grad_vec!(bit, msc[:xfm_sto_plus_x], msc[:scale_to_x], qG, :xfm_sto_plus_x);


# %%
@time quasiGrad.soft_abs_grad_vec!(bit, msc, qG, :xfm_sfr_plus_x, :xfm_sfr_plus_x, :scale_fr_x);

# %%

@time [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(msc[:xfm_sfr_plus_x],msc[:xfm_sto_plus_x])];

f(x::Vector{Float64}, v::Vector{Float64}, vv::Vector{Int64}) = x[vv] .= v[vv];

# %%

@time f(x,v,vv);

# %%
function f(x::Vector{Float64}, v::Vector{Float64}, vv::Vector{Int64})
    for ii in vv
        x[ii] = v[ii]
    end
end

# %%
@btime quasiGrad.device_reserve_costs!(prm, qG, stt)

# %%
@btime quasiGrad.device_reserve_costs_new!(prm, stt)

# %%

@btime quasiGrad.device_reserve_costs_new!(prm, stt, sys)

V2 = collect(eachrow(reduce(hcat, prm.dev.p_reg_res_up_cost)))

# %%

@time dt.*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*stt[:p_rgu][tii]

f(stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, tii::Symbol, dt::Float64, V2::Vector{Vector{Float64}}, t_ind::Int64) = stt[:p_rgd][tii] .= dt.*V2[t_ind].*stt[:p_rgu][tii];
f0(stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, tii::Symbol, dt::Float64, V2::Vector{Vector{Float64}}, t_ind::Int64, prm::quasiGrad.Param) = stt[:p_rgd][tii] .= dt.*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*stt[:p_rgu][tii];

# %%
dt = 0.5
@btime f(stt, tii, dt, V2, 1);
# %%
@btime f0(stt, tii, dt, V2, 1, prm);

# %%
@time stt[:p_rgu][tii] .= dt.*stt[:p_rgu][tii];

# %%
@btime quasiGrad.reserve_balance!(idx, msc, prm, stt, sys)

# %%
@btime quasiGrad.reserve_p_max!(idx, msc, stt, tii, zone)

# %%

@btime quasiGrad.reserve_balance_old!(idx, msc, prm, stt, sys)

# %%

@btime quasiGrad.reserve_balance!(idx, msc, prm, stt, sys)

# %%

reserve_type = :p_rgu 
tii = :t1
zone = 1
zone_type = :Pz

@btime quasiGrad.reserve_sum!(idx, msc, reserve_type, stt, tii, zone, zone_type)

# %%
@time quasiGrad.score_zt!(idx, prm, qG, scr, stt) 

# %%
@btime quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

# %%


@btime quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)
@btime quasiGrad.ideal_dispatch_faster!(idx, msc, stt, sys, tii)

# %%
@btime quasiGrad.device_reactive_powers!(idx, prm, qG, stt, sys)

# %%
@btime quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)

# %%

@btime quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)

# %%
@btime quasiGrad.batch_fix!(pct_round, prm, stt, sys, upd)

# %%
quasiGrad.batch_fix!(pct_round, prm, stt, sys, upd)

# %% 
@btime quasiGrad.energy_costs!(grd, prm, qG, stt, sys)

# %%
@btime quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)

# %%
dev = 1
t_ind = 1
dt = 0.2
cst = prm.dev.cum_cost_blocks[dev][t_ind][1]  # cost for each block (leading with 0)
pbk = prm.dev.cum_cost_blocks[dev][t_ind][2]  # power in each block (leading with 0)
pcm = prm.dev.cum_cost_blocks[dev][t_ind][3]  # accumulated power for each block!
nbk = length(pbk)

# get the cost!
@time dt*sum(cst[ii]*max(min(stt[:dev_p][tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)

@time dt*sum(cst_i*max(min(stt[:dev_p][tii][dev] - pcm_i, pbk_i), 0.0)  for (cst_i,pcm_i,pbk_i) in zip(cst[2:end],pcm[1:end-1],pbk[2:end]); init=0.0)

# %%
for (cst_i,pcm_i,pbk_i) in zip(cst[2:end],pcm[1:end-1],pbk[2:end])
    cst_i
    pcm_i
    pbk_i
end


# %%
# loop over each time period
tii = :t4
t_ind = 1
dt = prm.ts.duration[tii]

    # devices
    for dev in 1:sys.ndev
        cst = prm.dev.cum_cost_blocks[dev][t_ind][1]  # cost for each block (leading with 0)
        pbk = prm.dev.cum_cost_blocks[dev][t_ind][2]  # power in each block (leading with 0)
        pcm = prm.dev.cum_cost_blocks[dev][t_ind][3]  # accumulated power for each block!
        nbk = length(pbk)

        # get the cost!
        stt[:zen_dev][tii][dev] = dt*sum(cst[ii]*max(min(stt[:dev_p][tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)
            # fancy alternative => stt[:zen_dev][tii][dev] = dt*sum(cst_i*max(min(stt[:dev_p][tii][dev] - pcm_i, pbk_i), 0.0)  for (cst_i,pcm_i,pbk_i) in zip(cst[2:end],pcm[1:end-1],pbk[2:end]); init=0.0)
        # evaluate the grd? 
        #
        # WARNING -- this will break if stt[:dev_p] > pcm[end]! It will
        #            mean the device power is out of bounds, and this will
        #            call a price curve which does not exist.
        #                  ~ clipping will fix ~
        if qG.eval_grad
            # what is the index of the "active" block?
            # easier to understand:
            del = stt[:dev_p][tii][dev] .- pcm
            active_block_ind = argmin(del[del .>= 0.0])
            g1 = dt*cst[active_block_ind + 1] # where + 1 is due to the leading 0
            if stt[:dev_p][tii][dev] == 0.0
                g2 = dt*cst[2]
            else
                g2 = dt*cst[findfirst(stt[:dev_p][tii][dev] .< pcm)] # no +1 needed, because we find the upper block
            end
            println(dev)
            @assert(g1 == g2)
        end
    end

# %%
adm_step    = 0
beta1       = qG.beta1
beta2       = qG.beta2
beta1_decay = 1.0
beta2_decay = 1.0
run_adam    = true
beta1_decay = beta1_decay*beta1
beta2_decay = beta2_decay*beta2

@btime quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

# %%
@btime quasiGrad.quadratic_distance!(dpf0, mgd, prm, qG, stt)

# %%
dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf = quasiGrad.initialize_pf_lbfgs(mgd, prm, qG, stt, sys, upd);

# %%
@btime emergency_stop = quasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, stt, upd, zpf)

# %%
dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf = quasiGrad.initialize_pf_lbfgs(mgd, prm, qG, stt, sys, upd);
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(bit, cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)

@time emergency_stop = quasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, stt, upd, zpf)

# %%
@time pf_lbfgs[:x_prev][tii]     .= copy.(pf_lbfgs[:x_now][tii]);

# %%

@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys);

# %%
@time quasiGrad.master_grad_zs_acline!(tii, idx, stt, grd, mgd, sys)

# %%
@time quasiGrad.master_grad_zs_acline_fastesttt!(tii, idx, stt, grd, mgd, msc, sys)

# %%
@time quasiGrad.master_grad_zs_acline!(tii, idx, stt, grd, mgd, sys)

# %%
@time quasiGrad.master_grad_zs_xfm!(tii, idx, stt, grd, mgd, sys)

# %%

@time master_grad_zs_xfm_fastesttt!(tii, idx, stt, grd, mgd, msc, sys)

# %%
@time quasiGrad.master_grad_zs_acline!(tii, idx, stt, grd, mgd, sys)
@time quasiGrad.master_grad_zs_xfm!(tii, idx, stt, grd, mgd, sys)
@time quasiGrad.master_grad_zp!(tii, prm, idx, grd, mgd, sys)
@time quasiGrad.master_grad_zq!(tii, prm, idx, grd, mgd, sys)

# %%
@btime sum(mgd_com*grd[:sh_p][:vm][tii][sh] for sh in idx.sh[bus]; init=0.0)
@btime sum(mgd_com*grd[:sh_p][:vm][tii][idx.sh[bus]]; init=0.0)

# %%
print("t15: ")

@time quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

# %%

@time _, ch = quasiGrad.cg!(ctb[9], ntk.Ybr, flw[:c], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = false)


# %%

println(sum(sum(stt[:zhat_mndn][tii]) for tii in prm.ts.time_keys[2:end]))
println(sum(sum(stt[:zhat_mnup][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rup][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rd][tii])   for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rgu][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rgd][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_scr][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_nsc][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rruon][tii])    for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rruoff][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rrdon][tii])   for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_rrdoff][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_pmax][tii])      for tii in prm.ts.time_keys[2:end]))  
println(sum(sum(stt[:zhat_pmin][tii])       for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_pmaxoff][tii])    for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_qmax][tii])       for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_qmin][tii])       for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_qmax_beta][tii])  for tii in prm.ts.time_keys[2:end])) 
println(sum(sum(stt[:zhat_qmin_beta][tii]) for tii in prm.ts.time_keys[2:end])) 

# %%

dev = 373
tii = :t6
dev_p_previous = stt[:dev_p][prm.ts.tmin1[tii]][dev] 
dt = prm.ts.duration[tii]

cvio = stt[:dev_p][tii][dev] - dev_p_previous - dt*(prm.dev.p_ramp_up_ub[dev]     *(stt[:u_on_dev][tii][dev] - stt[:u_su_dev][tii][dev]) +     prm.dev.p_startup_ramp_ub[dev]*(stt[:u_su_dev][tii][dev] + 1.0 - stt[:u_on_dev][tii][dev]))

println(cvio)

# %%

prm.dev.q_lb_tmdv[t_ind]

# %%
@btime quasiGrad.clip_pq!(prm, qG, stt)

# %%
prm.dev.p_lb_tmdv[t_ind] => prm.dev.p_lb_tmdv[t_ind]
prm.dev.p_ub_tmdv[t_ind] => prm.dev.p_ub_tmdv[t_ind]

prm.dev.q_lb_tmdv[t_ind] => prm.dev.q_lb_tmdv[t_ind]
prm.dev.q_ub_tmdv[t_ind] => prm.dev.q_ub_tmdv[t_ind]

# %%    
t_ind = 18

println(prm.dev.p_reg_res_up_cost_tmdv[t_ind]              - prm.dev.p_reg_res_up_cost_tmdv[t_ind])
println(prm.dev.p_reg_res_down_cost_tmdv[t_ind]            - prm.dev.p_reg_res_down_cost_tmdv[t_ind])
println(prm.dev.p_syn_res_cost_tmdv[t_ind]                 - prm.dev.p_syn_res_cost_tmdv[t_ind])
println(prm.dev.p_nsyn_res_cost_tmdv[t_ind]                - prm.dev.p_nsyn_res_cost_tmdv[t_ind])
println(prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind]      - prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind])
println(prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind]     - prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind])
println(prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind]    - prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind])
println(prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind]   - prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind] )
println(prm.dev.q_res_up_cost_tmdv[t_ind]                  - prm.dev.q_res_up_cost_tmdv[t_ind])
println(prm.dev.q_res_down_cost_tmdv[t_ind]                - prm.dev.q_res_down_cost_tmdv[t_ind])