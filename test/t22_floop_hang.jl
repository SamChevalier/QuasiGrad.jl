using quasiGrad
using Revise
using Polyester

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00014/scenario_003.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
jsn  = quasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn);

# %% ================
qG.skip_ctg_eval = true
qG.num_threads   = 10
@btime quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% run copper plate ED
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)


# %% ===

model = quasiGrad.Model(quasiGrad.Gurobi.Optimizer)
quasiGrad.set_optimizer_attribute(model, "Threads", 3)
quasiGrad.@variable(model, 0 <= x <= 1)
quasiGrad.@objective(model, Max, x)
quasiGrad.optimize!(model)

# %%

qG.num_threads    = 10
qG.score_all_ctgs = false
qG.eval_grad      = true

qG.pcg_tol = 0.0003969886648255842
qG.pcg_tol = 0.00039

@btime quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys);

# %%
thrID = 1

@time quasiGrad.cg!(ctg.dz_dpinj[thrID], ntk.Ybr, ctg.rhs[thrID], statevars = ctg.grad_cg_statevars[thrID],  abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its);

@time ctg.dz_dpinj[thrID] .= ntk.Ybr\ctg.rhs[thrID];

# %% test sorting
zctg = [-10.0; -9.4; -22.8; -1; -0.0; -50.3]
worst_ctg_ids = [6; 3; 1; 2; 4; 5]

worst_ctg_ids

# %%
x = randn(25000)
f(x::Vector{Float64}) = sortperm(x)

# %%
a = collect(1:10)
@time deleteat!(a, [2;4;6;10])

#a = randn(10000)
#b = a[1:3:end]

#@time setdiff!(a, b);


# %%

@btime x .= f();
# %%


tt = worst_ctg_ids[sortperm(zctg[worst_ctg_ids])]
worst = 

# %%
flw.worst_ctg_ids[tii][1:num_ctg] .= union(flw.worst_ctg_ids[tii][1:num_wrst],quasiGrad.shuffle(setdiff(1:sys.nctg, flw.worst_ctg_ids[tii][1:num_wrst]))[1:num_rnd])


#zctg[worst_ctg_ids]
#flw.worst_ctg_ids[tii][1:num_ctg] .= sortperm(@view stt.zctg[tii][@view flw.worst_ctg_ids[tii][1:num_ctg]])

# %% -- no change
zctg          = randn(100)
nctg          = 100
num_ctg       = 10
num_wrst      = 5
num_rnd       = 5
zctg_scored   = randn(10)
worst_ctg_ids = [1; 4; 8; 19; 3; 6; 22; 43; 42; 5]

# %%
zctg_scored                         .= @view zctg[worst_ctg_ids]
worst_ctg_ids[1:num_wrst]           .= @view worst_ctg_ids[partialsortperm(zctg_scored, 1:num_wrst)]
worst_ctg_ids[(num_wrst+1:num_ctg)] .= @view quasiGrad.shuffle!(deleteat!(collect(1:nctg), sort(worst_ctg_ids[1:num_wrst])))[1:num_rnd]

# %%
b = randn(10000)
d = randn(10000)
c = @view b[1:440]
d[1:440] .= c
# %%

            # 1. partial sort the worst
            stt.zctg_scored[tii] .= @view stt.zctg[tii][1:num_ctg]
            partialsortperm!(stt.zctg_ix[tii], stt.zctg_scored[tii], num_wrst)

            # 2. stt.zctg_ix[tii] contains the sorted indices of "flw.worst_ctg_ids[tii][1:num_ctg]" -- apply them!
            flw.worst_ctg_ids[tii][1:num_wrst] .= @view flw.worst_ctg_ids[tii][stt.zctg_ix[tii]]

            # 3. select a random subset of the remainder :)
            flw.worst_ctg_ids[tii][(num_wrst+1:num_ctg)] .= quasiGrad.shuffle!(deleteat!(1:sys.nctg, flw.worst_ctg_ids[tii][1:num_wrst]))




            # sort: grab
            stt.zctg_scored[tii] .= @view stt.zctg[tii][1:num_ctg]

            partialsortperm!(stt.zctg_ix[tii], stt.zctg_scored[tii], num_wrst)


            flw.worst_ctg_ids[tii][1:num_ctg] .= flw.worst_ctg_ids[tii][sortperm(stt.zctg[tii][1:num_ctg])]



            #flw.worst_ctg_ids[tii][1:num_wrst] = union(flw.worst_ctg_ids[tii][1:num_wrst], quasiGrad.shuffle(
            flw.worst_ctg_ids[tii][(num_ctg+1):(num_rnd+num_ctg)]      
                
            

# %% 1. partial sort the worst
zctg_scored .= @view zctg[1:num_ctg]
partialsortperm(zctg_scored, 1:num_wrst)

# 2. stt.zctg_ix[tii] contains the sorted indices of "flw.worst_ctg_ids[tii][1:num_ctg]" -- apply them!
worst_ctg_ids[1:num_wrst] .= @view worst_ctg_ids[partialsortperm(zctg_scored, 1:num_wrst)]

# 3. select a random subset of the remainder :)
worst_ctg_ids[(num_wrst+1:num_ctg)] .= @view quasiGrad.shuffle!(deleteat!(collect(1:nctg), worst_ctg_ids[1:num_wrst]))[1:num_rnd]

# %%

tt = collect(1:10000)
@btime 352523 in tt




# %%
tii = Int8(1)
num_ctg = 219
@time stt.zctg[tii][@view flw.worst_ctg_ids[tii][1:num_ctg]]

# %%
lck = Threads.SpinLock()

@time Threads.lock(lck)
@time Threads.unlock(lck)
# %%
x0 = randn(10000)
y0 = randn(10000)
u  = randn(10000)
x  = randn(10000)
g  = randn()

f(x0::Vector{Float64}, y0::Vector{Float64}, u::Vector{Float64}, x::Vector{Float64}, g::Float64) = x0 .= y0 .- u.*(g*dot(u, x))
# %%

@time f(x0, y0, u, x, g)

# %%
x = randn(10)
tt = quasiGrad.IterativeSolvers.CGStateVariables(zero(x), similar(x), similar(x))

# %%
qG.num_threads = 1
qG.eval_grad   = true

qG.score_all_ctgs = true

@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %%
thrID = 1
tii = Int8(1)
ctg_ii = 1
cs = 1.0
flw.theta[tii] = randn(1575)


@time quasiGrad.cg!(flw.theta[tii], ntk.Ybr, flw.c[tii], statevars = flw.pf_cg_statevars[tii], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its);

# %%
thrID = 1
tii = Int8(1)
ctg_ii = 1
cs = 1.0

# see the "else" case for comments and details
@time ctg.theta_k[thrID] .= quasiGrad.wmi_update(flw.theta[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])
# => slow: ctg.pflow_k .= ntk.Yfr*ctg.theta_k .+ flw.bt
@time quasiGrad.mul!(ctg.pflow_k[thrID], ntk.Yfr, ctg.theta_k[thrID])
@time ctg.pflow_k[thrID] .+= flw.bt[tii]
@time ctg.sfr[thrID]     .= sqrt.(flw.qfr2[tii] .+ ctg.pflow_k[thrID].^2)
@time ctg.sto[thrID]     .= sqrt.(flw.qto2[tii] .+ ctg.pflow_k[thrID].^2)
@time ctg.sfr_vio[thrID] .= ctg.sfr[thrID] .- ntk.s_max
@time ctg.sto_vio[thrID] .= ctg.sto[thrID] .- ntk.s_max
@time ctg.sfr_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0
@time ctg.sto_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0
@time stt.zctg[tii][ctg_ii] = -cs*sum(max.(ctg.sfr_vio[thrID], ctg.sto_vio[thrID], 0.0))

# %%
thrID = 1
tii = Int8(1)
ctg_ii = 1
cs = 1.0

f() = ctg.theta_k[thrID] .= quasiGrad.wmi_update(flw.theta[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])

ff() = ctg.theta_k[thrID] .= flw.theta[tii] .- ntk.u_k[ctg_ii].*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], flw.c[tii]))

fff() = ntk.g_k[ctg_ii]#+(quasiGrad.dot(ntk.u_k[ctg_ii], flw.c[tii]))

# %%
@code_warntype quasiGrad.wmi_update(flw.theta[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])

# %%
t = 0.0
@time f();
@time ff();
@time t = fff();

# %%
@time ctg.theta_k[thrID] .= quasiGrad.wmi_update(flw.theta[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])

# %%
t = 0.0
v = [randn(1575) for ii in 1:100]
@time t = quasiGrad.dot(v[1], v[2])
@time t = quasiGrad.dot(ntk.u_k[ctg_ii], flw.c[tii])
@time t = quasiGrad.dot(ntk.u_k[ctg_ii], v[2])

# %%
function hh(ctg::quasiGrad.Contingency, ntk::quasiGrad.Network, flw::quasiGrad.Flow, ctg_ii::Int64, thrID::Int64)
    ctg.theta_k[thrID] .= flw.theta[thrID]
end

# %%

@time hh(ctg, ntk, flw, ctg_ii, thrID)

# %% run copper plate ED
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
qG.score_all_ctgs = true
quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% =============
# lck = Threads.ReentrantLock()
lck = Threads.SpinLock()

rtu = ones(Bool, 2*qG.num_threads)
rtu .= true

qG.num_threads = 10
println("===============")
quasiGrad.@floop quasiGrad.ThreadedEx(basesize = sys.nctg ÷ qG.num_threads) for ctg_ii in 1:sys.nctg
    # use a custom "thread ID" -- three indices: tii, ctg_ii, and thrID
    thrID = 0
    Threads.lock(lck) do
        thrID = findfirst(rtu)
        rtu[thrID] = false # now in use :)
    end

    #println(Threads.threadid())

    # all done!!
    Threads.lock(lck) do
        rtu[thrID] = true # not in use :)
    end
    println(rtu)

end

# %% ======
qG.skip_ctg_eval               = true
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% compute all states and grads
qG.skip_ctg_eval = true
qG.num_threads   = 10

function f() 
    #quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
    #quasiGrad.Polyester.reset_threads!()
    #quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
    #quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
    quasiGrad.clip_all!(prm, qG, stt, sys)
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
    #quasiGrad.clip_all!(prm, qG, stt, sys)
end

# %%
@btime f()

# %%
@btime quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()

# %%
function g(grd::quasiGrad.Grad, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System) 
    quasiGrad.clip_dc!(prm, qG, stt)
    quasiGrad.clip_xfm!(prm, qG, stt)
    quasiGrad.clip_shunts!(prm, qG, stt)
    quasiGrad.clip_voltage!(prm, qG, stt)
    quasiGrad.clip_onoff_binaries!(prm, qG, stt)
    quasiGrad.transpose_binaries!(prm, qG, stt)
    quasiGrad.clip_reserves!(prm, qG, stt)
    quasiGrad.clip_pq!(prm, qG, stt)
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
end

# %%
@btime f()
# %%
function h()
    t0 = time()
    for ii in 1:1000
        g(grd, mgd, prm, qG, stt, sys)
    end
    tavg = (time() - t0)/1000
    println(tavg)
end

# %%
h()

# %%
@benchmark g($grd, $mgd, $prm, $qG, $stt, $sys)

# %%
@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
@btime quasiGrad.clip_all!(prm, qG, stt, sys)

# %%
@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
@btime quasiGrad.clip_all!(prm, qG, stt, sys)
# %%

@benchmark quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
# @btime 

# %% ====================== 
qG.clip_pq_based_on_bins   = false
qG.acflow_grad_is_soft_abs = true
qG.num_threads = 10

@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
@btime quasiGrad.clip_all!(prm, qG, stt, sys)
@btime quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
@btime quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)
@btime quasiGrad.shunts!(grd, idx, prm, qG, stt)
@btime quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
@btime quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
@btime quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
@btime quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
@btime quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
@btime quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
@btime quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
@btime quasiGrad.device_reserve_costs!(prm, qG, stt)
@btime quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)
@btime quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)
# @time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
@btime quasiGrad.score_zt!(idx, prm, qG, scr, stt) 
@btime quasiGrad.score_zbase!(qG, scr)
@btime quasiGrad.score_zms!(scr)
# @btime quasiGrad.print_zms(qG, scr)
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

# %%
@time quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)

# %%
qG.num_threads = 1
@time quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

# %%
v = randn(50)
@batch per=core for ii in 1:100
    t = maximum(v[ij] for ij in 1:5)
end

# %%
qG.num_threads = 6

@btime quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)

# %%

@btime quasiGrad.acline_flows_poly!(grd, idx, prm, qG, stt, sys)

# %%
v = randn(1000)
f(v::Vector{Float64}) = argmax(@view v[1:10])
g(v::Vector{Float64}) = argmax(v[jj] for jj in 1:10)
@btime f(v)
@btime g(v)


# %% ===================
qG.apply_grad_weight_homotopy  = false
qG.take_adam_pf_steps          = false
qG.pqbal_grad_type             = "soft_abs" 
qG.constraint_grad_is_soft_abs = false
qG.acflow_grad_is_soft_abs     = false
qG.reserve_grad_is_soft_abs    = false


qG.skip_ctg_eval               = true
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

qG.adam_max_time = 60.0
qG.num_threads   = 10

# %%
qG.num_threads = 1
@btime quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

# %% ========================
qG.num_threads = 6
for ii in 1:1000
    # safepoint
    # GC.safepoint()

    # if we are here, we want to make sure we are running su/sd updates
    qG.run_susd_updates = true

    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    # clip all basic states (i.e., the states which are iterated on)
        # => println("clipping off!!!")
        # => println("bin_clip is true!")
    qG.clip_pq_based_on_bins = false
    quasiGrad.clip_all!(prm, qG, stt, sys)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, prm, qG, stt)

    # device powers
    quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
    quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
    quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
    quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
    quasiGrad.device_reserve_costs!(prm, qG, stt)

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

    # compute reserve margins and penalties (no grads here)
    quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

    # score the contingencies and take the gradients
    if qG.skip_ctg_eval
        #println("Skipping ctg evaluation!")
    else
        quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    end
    # score the market surplus function
    quasiGrad.score_zt!(idx, prm, qG, scr, stt) 
    quasiGrad.score_zbase!(qG, scr)
    quasiGrad.score_zms!(scr)

    # print the market surplus function value
    #quasiGrad.print_zms(qG, scr)

    # compute the master grad
    #GC.safepoint()
    #quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

    # print the market surplus function value
    #quasiGrad.print_zms(qG, scr)

    # compute the master grad
    # quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
    if mod(ii,1) == 0
        println(ii)
    end
end

# %%
qG.skip_ctg_eval = true
qG.num_threads   = 10
t1 = time()
for ii in 1:250
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
end
tend = time() - t1

# %%
tii = Int8(1)
@btime quasiGrad.master_grad_zs_acline!(tii, idx, grd, mgd, qG, stt, sys)

@btime quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, qG, stt, sys)


# %%
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

# %%
@time idx.pr_pzone[zone][argmax(@view stt.dev_p[tii][idx.pr_pzone[zone]])];

# %%
@time quasiGrad.master_grad_zp!(tii, prm, idx, grd, mgd, sys)
@time quasiGrad.master_grad_zq!(tii, prm, idx, grd, mgd, sys)

# %%
zone = 1
    # g17 (zrgu_zonal):
    # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgu_zonal] * cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone] #grd[:zrgu_zonal][:p_rgu_zonal_penalty][tii][zone]
    if qG.reserve_grad_is_soft_abs
        mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*quasiGrad.soft_abs_reserve_grad(stt.p_rgu_zonal_penalty[tii][zone], qG)
    else
        mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]*sign(stt.p_rgu_zonal_penalty[tii][zone])
    end
    mgd.p_rgu[tii][idx.dev_pzone[zone]] .-= mgd_com


# %%
v = ones(15)

v .+= 5.5

# %%

qG.num_threads = 7
for ii in 1:1000
    # start by looping over the terms which can be safely looped over in time
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        quasiGrad.master_grad_zs_acline_test!(tii, idx, grd, mgd, qG, sys)
    end
    println(ii)
end

# %% ==========
using Plots

num_outer_loops    = 1000
num_parallel_loops = 15
time_vec        = zeros(num_outer_loops)
rvec1           = randn(10000)
rvec2           = randn(10000)
for ii in 1:num_outer_loops
    t1 = time()
    Threads.@threads for jj in 1:num_parallel_loops
        rvec_copy1 = rvec1.*rvec2
        rvec_copy2 = rvec1.*rvec2
        rvec_copy3 = rvec1.*rvec2
        rvec_copy4 = rvec1.*rvec2
        rvec_copy5 = rvec1.*rvec2
    end
    time_vec[ii] = time() - t1
    println(ii)
end

# plot hanging
Plots.plot(1:1000, time_vec, yaxis = :log, xlabel = "outer loop iteration", ylabel = "time (secs, log scale)")


# %%
qG.num_threads  = 10
num_outer_loops = 5000
for ii in 1:num_outer_loops
    quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
    println(ii)
end

# %%
qG.num_threads = 6
for ii in 1:1000
    quasiGrad.@floop quasiGrad.ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        vmfrpfr = grd.zs_acline.acline_pfr[tii].*grd.acline_pfr.vmfr[tii]
        vmtopfr = grd.zs_acline.acline_pfr[tii].*grd.acline_pfr.vmto[tii]
        vafrpfr = grd.zs_acline.acline_pfr[tii].*grd.acline_pfr.vafr[tii]
        vatopfr = grd.zs_acline.acline_pfr[tii].*grd.acline_pfr.vato[tii]

        # final qfr gradients
        # OG => mgqfr   = mg_com.*qfr_com
        # mgqfr * everything below:
        vmfrqfr = grd.zs_acline.acline_qfr[tii].*grd.acline_qfr.vmfr[tii]
        vmtoqfr = grd.zs_acline.acline_qfr[tii].*grd.acline_qfr.vmto[tii]
        vafrqfr = grd.zs_acline.acline_qfr[tii].*grd.acline_qfr.vafr[tii]
        vatoqfr = grd.zs_acline.acline_qfr[tii].*grd.acline_qfr.vato[tii]
    end

    println(ii)
end
    #@floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
    #    quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, qG, stt, sys)
    #end

        # g15 (zp): nzms => zbase => zt => zp => (all p injection variables)
        #quasiGrad.master_grad_zp!(tii, prm, idx, grd, mgd, sys)
                
        # g16 (zq): nzms => zbase => zt => zq => (all q injection variables)
        #quasiGrad.master_grad_zq!(tii, prm, idx, grd, mgd, sys)
    #println(ii)
#end

# %%
@floop ThreadedEx(basesize = sys.ndev ÷ qG.num_threads) for dev in prm.dev.dev_keys
    println(dev)
end

# %%
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% ====================
using FLoops

# loop
nloops   = 1000
iiv      = collect(1:nloops)
z_mat    = Dict(ii => ones(100000) for ii in iiv)

@floop ThreadedEx(basesize = nloops ÷ 6) for ii in 1:nloops
    gamma = ii^2

    z_mat[ii] .= gamma
    println(ii)
end

# %% ====================
using FLoops

# loop
nloops   = 100
iiv      = collect(1:nloops)
z_mat    = Dict(ii => ones(100) for ii in iiv)

for jj in 1:10
    @floop ThreadedEx(basesize = nloops ÷ 6) for ii in 1:nloops
        z_mat[ii] += sin.(z_mat[ii]).^2
        println(ii)
    end
end

# %%

@floop ThreadedEx(basesize = nloops ÷ 6) for ii in 1:nloops
    z_mat[ii] .= sin.(z_mat[ii]).^2
end
println("end")

# %%
using Base.Threads

nloops = 10^8
nt=nthreads()

println("nthreads: ",nt)
x=0

@threads for i in 1:nloops
        global x+=1
end

# %%
function foo()
    Threads.@threads for i in 1:10
         rand()
     end
end

for i in 1:1000
    println(i)
    for j in 1:10000
        foo()
    end
end

# %%
using Base.Threads

nthreads()

function f()
        s = repeat(["123", "213", "231"], outer=1000)
        x = similar(s, Int)
        rx = r"1"
        @threads for i in 1:3000
            x[i] = findfirst(rx, s[i]).start
        end
        count(v -> v == 1, x)
    end

f() 

# %% =============
num_threads = 10
@floop ThreadedEx(basesize = 10) for ii in 1:500
    for jj in 1:3
        @reduce total_value += 1
        #println(Threads.threadid())
        #println("ii: $ii")
    end
end

println(total_value)

# %%
adm_step    = 0
beta1       = qG.beta1
beta2       = qG.beta2
beta1_decay = 1.0
beta2_decay = 1.0
run_adam    = true

@info "Running adam for $(qG.adam_max_time) seconds!"

# flush adam at each restart ?
# println("adam NOT flushed")
quasiGrad.flush_adam!(adm, flw, prm, upd)

# start the timer!
adam_start = time()

# increment
adm_step += 1

# step decay
# alpha = step_decay(adm_step, qG)

# decay beta
beta1_decay = beta1_decay*beta1
beta2_decay = beta2_decay*beta2

# %% compute all states and grads
qG.skip_ctg_eval = true
qG.num_threads   = 10
@btime quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %%
quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)

# %% take an adam step 
qG.num_threads = 1

#ProfileView.@profview 

@btime quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)

# %%

@code_warntype quasiGrad.call_adam_states(adm, mgd, stt, var_key)
# %%
@btime quasiGrad.call_adam_states(adm, mgd, stt, :vm);
# %%
for it in qG
    println(it)
end

# %%
for ty in [:vm, :va]
    adm.ty
end

# %%

function getDoSomething2(name::String)
    field = Symbol(name)
    code = quote
        (obj) -> obj.$field
    end
    return eval(code)
end

const doSomething2 = getDoSomething2("vm");

doSomething2(adm)

# %%
function iterator_f()
    code = quote
        Threads.@threads
    end
    return code
end

# %%
tt = 1
if tt == 1
    Threads.@threads for ii in 1:10
end

# %%
macro sayhello()
    return eval(Threads.@threads)
end

# %%
function ttb()
        # note, the reference bus is always bus #1
    #
    # first, get the ctg limits
    s_max_ctg = [prm.acline.mva_ub_em; prm.xfm.mva_ub_em]

    # get the ordered names of all components
    ac_ids = [prm.acline.id; prm.xfm.id ]

    # get the ordered (negative!!) susceptances
    ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
    
    # build the full incidence matrix: E = lines x buses
    E  = quasiGrad.build_incidence(idx, prm, stt, sys)
    Er = E[:,2:end]
    ErT = copy(Er')

    # get the diagonal admittance matrix   => Ybs == "b susceptance"
    Ybs = quasiGrad.spdiagm(ac_b_params)
    Yb  = E'*Ybs*E
    Ybr = Yb[2:end,2:end]  # use @view ? 

    # should we precondition the base case?
    #
    # Note: Ybr should be sparse!! otherwise,
    # the sparse preconditioner won't have any memory limits and
    # will be the full Chol-decomp -- not a big deal, I guess..
    if qG.base_solver == "pcg"
        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- use LU
            @warn "Not enough buses for Krylov! Using LU anyways."
        end

        # time is short -- let's jsut always use ldl preconditioner -- it's just as fast
        Ybr_ChPr = quasiGrad.Preconditioners.lldl(Ybr, memory = qG.cutoff_level);

        # OG#2 solution!
            # can we build cholesky?
            # if minimum(ac_b_params) < 0.0
            #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
            # else
            #     @info "Unsing an ldl preconditioner."
            #     Ybr_ChPr = quasiGrad.ldl(Ybr, qG.cutoff_level);
            # end

        # OG#1 solution!
            # # test for negative reactances -- @info "Preconditioning is disabled."
            # if minimum(ac_b_params) < 0.0
            #     # Amrit Pandey: "watch out for negatvive reactance! You will lose
            #     #                pos-sem-def of the Cholesky preconditioner."
            #     abs_b    = abs.(ac_b_params)
            #     abs_Ybr  = (E'*quasiGrad.spdiagm(abs_b)*E)[2:end,2:end] 
            #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(abs_Ybr, qG.cutoff_level)
            # else
            #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
            # end
    else
        # time is short -- let's jsut always use ldl preconditioner -- it's just as fast
        Ybr_ChPr = quasiGrad.Preconditioners.lldl(Ybr, memory = qG.cutoff_level);
            # # => Ybr_ChPr = quasiGrad.I
            # Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
    end
    
    # should we build the cholesky decomposition of the base case
    # admittance matrix? we build this to compute high-fidelity
    # solutions of the rank-1 update matrices
    if qG.build_basecase_cholesky
        if minimum(ac_b_params) < 0.0
            @info "Yb not PSd -- using ldlt (instead of cholesky) to construct WMI update vectors."
            Ybr_Ch = quasiGrad.ldlt(Ybr)
        else
            Ybr_Ch = quasiGrad.cholesky(Ybr)
        end
    else
        # this is nonsense
        Ybr_Ch = quasiGrad.I
    end

    # get the flow matrix
    Yfr  = Ybs*Er
    YfrT = copy(Yfr')

    # build the low-rank contingecy updates
    #
    # base: Y_b*theta_b = p
    # ctg:  Y_c*theta_c = p
    #       Y_c = Y_b + uk'*uk
    ctg_out_ind = Dict(ctg_ii => Vector{Int64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
    ctg_params  = Dict(ctg_ii => Vector{Float64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
    
    # should we build the full ctg matrices?
    if qG.build_ctg_full == true
        nac   = sys.nac
        Ybr_k = Dict(ctg_ii => quasiGrad.spzeros(nac,nac) for ctg_ii in 1:sys.nctg)
    else
        # build something small of the correct data type
        Ybr_k = Dict(1 => quasiGrad.spzeros(1,1))
    end

    # and/or, should we build the low rank ctg elements?
    if qG.build_ctg_lowrank == true
        # no need => v_k = Dict(ctg_ii => quasiGrad.spzeros(nac) for ctg_ii in 1:sys.nctg)
        # no need => b_k = Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
        u_k = Dict(ctg_ii => zeros(sys.nb-1) for ctg_ii in 1:sys.nctg) # Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
        g_k = Dict(ctg_ii => 0.0             for ctg_ii in 1:sys.nctg)
        # if the "w_k" formulation is wanted => w_k = Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
    else
        v_k = 0
        b_k = 0
    end

    # loop over components (see below for comments!!!)
    for ctg_ii in 1:sys.nctg
        cmpnts = prm.ctg.components[ctg_ii]
        for (cmp_ii,cmp) in enumerate(cmpnts)
            cmp_index = findfirst(x -> x == cmp, ac_ids) 
            ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
            ctg_params[ctg_ii][cmp_ii]  = -ac_b_params[cmp_index]
        end
    end
    quasiGrad.@batch per=core for ctg_ii in 1:sys.nctg
    # @floop ThreadedEx(basesize = sys.nctg ÷ qG.num_threads) for ctg_ii in 1:sys.nctg
        # this code is optimized -- see above for comments!!!
        u_k[ctg_ii] .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
        g_k[ctg_ii]  = -ac_b_params[ctg_out_ind[ctg_ii][1]]/(1.0+(quasiGrad.dot(Er[ctg_out_ind[ctg_ii][1],:],u_k[ctg_ii]))*-ac_b_params[ctg_out_ind[ctg_ii][1]])
    end
end

function ttf()
        # note, the reference bus is always bus #1
    #
    # first, get the ctg limits
    s_max_ctg = [prm.acline.mva_ub_em; prm.xfm.mva_ub_em]

    # get the ordered names of all components
    ac_ids = [prm.acline.id; prm.xfm.id ]

    # get the ordered (negative!!) susceptances
    ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
    
    # build the full incidence matrix: E = lines x buses
    E  = quasiGrad.build_incidence(idx, prm, stt, sys)
    Er = E[:,2:end]
    ErT = copy(Er')

    # get the diagonal admittance matrix   => Ybs == "b susceptance"
    Ybs = quasiGrad.spdiagm(ac_b_params)
    Yb  = E'*Ybs*E
    Ybr = Yb[2:end,2:end]  # use @view ? 

    # should we precondition the base case?
    #
    # Note: Ybr should be sparse!! otherwise,
    # the sparse preconditioner won't have any memory limits and
    # will be the full Chol-decomp -- not a big deal, I guess..
    if qG.base_solver == "pcg"
        if sys.nb <= qG.min_buses_for_krylov
            # too few buses -- use LU
            @warn "Not enough buses for Krylov! Using LU anyways."
        end

        # time is short -- let's jsut always use ldl preconditioner -- it's just as fast
        Ybr_ChPr = quasiGrad.Preconditioners.lldl(Ybr, memory = qG.cutoff_level);

        # OG#2 solution!
            # can we build cholesky?
            # if minimum(ac_b_params) < 0.0
            #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
            # else
            #     @info "Unsing an ldl preconditioner."
            #     Ybr_ChPr = quasiGrad.ldl(Ybr, qG.cutoff_level);
            # end

        # OG#1 solution!
            # # test for negative reactances -- @info "Preconditioning is disabled."
            # if minimum(ac_b_params) < 0.0
            #     # Amrit Pandey: "watch out for negatvive reactance! You will lose
            #     #                pos-sem-def of the Cholesky preconditioner."
            #     abs_b    = abs.(ac_b_params)
            #     abs_Ybr  = (E'*quasiGrad.spdiagm(abs_b)*E)[2:end,2:end] 
            #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(abs_Ybr, qG.cutoff_level)
            # else
            #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
            # end
    else
        # time is short -- let's jsut always use ldl preconditioner -- it's just as fast
        Ybr_ChPr = quasiGrad.Preconditioners.lldl(Ybr, memory = qG.cutoff_level);
            # # => Ybr_ChPr = quasiGrad.I
            # Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
    end
    
    # should we build the cholesky decomposition of the base case
    # admittance matrix? we build this to compute high-fidelity
    # solutions of the rank-1 update matrices
    if qG.build_basecase_cholesky
        if minimum(ac_b_params) < 0.0
            @info "Yb not PSd -- using ldlt (instead of cholesky) to construct WMI update vectors."
            Ybr_Ch = quasiGrad.ldlt(Ybr)
        else
            Ybr_Ch = quasiGrad.cholesky(Ybr)
        end
    else
        # this is nonsense
        Ybr_Ch = quasiGrad.I
    end

    # get the flow matrix
    Yfr  = Ybs*Er
    YfrT = copy(Yfr')

    # build the low-rank contingecy updates
    #
    # base: Y_b*theta_b = p
    # ctg:  Y_c*theta_c = p
    #       Y_c = Y_b + uk'*uk
    ctg_out_ind = Dict(ctg_ii => Vector{Int64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
    ctg_params  = Dict(ctg_ii => Vector{Float64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
    
    # should we build the full ctg matrices?
    if qG.build_ctg_full == true
        nac   = sys.nac
        Ybr_k = Dict(ctg_ii => quasiGrad.spzeros(nac,nac) for ctg_ii in 1:sys.nctg)
    else
        # build something small of the correct data type
        Ybr_k = Dict(1 => quasiGrad.spzeros(1,1))
    end

    # and/or, should we build the low rank ctg elements?
    if qG.build_ctg_lowrank == true
        # no need => v_k = Dict(ctg_ii => quasiGrad.spzeros(nac) for ctg_ii in 1:sys.nctg)
        # no need => b_k = Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
        u_k = Dict(ctg_ii => zeros(sys.nb-1) for ctg_ii in 1:sys.nctg) # Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
        g_k = Dict(ctg_ii => 0.0             for ctg_ii in 1:sys.nctg)
        # if the "w_k" formulation is wanted => w_k = Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
    else
        v_k = 0
        b_k = 0
    end

    # loop over components (see below for comments!!!)
    for ctg_ii in 1:sys.nctg
        cmpnts = prm.ctg.components[ctg_ii]
        for (cmp_ii,cmp) in enumerate(cmpnts)
            cmp_index = findfirst(x -> x == cmp, ac_ids) 
            ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
            ctg_params[ctg_ii][cmp_ii]  = -ac_b_params[cmp_index]
        end
    end
    # @batch per=core for ctg_ii in 1:sys.nctg
    quasiGrad.FLoops.assistant(false)
    quasiGrad.@floop quasiGrad.ThreadedEx(basesize = sys.nctg ÷ qG.num_threads) for ctg_ii in 1:sys.nctg
        # this code is optimized -- see above for comments!!!
        u_k[ctg_ii] .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
        g_k[ctg_ii]  = -ac_b_params[ctg_out_ind[ctg_ii][1]]/(1.0+(quasiGrad.dot(Er[ctg_out_ind[ctg_ii][1],:],u_k[ctg_ii]))*-ac_b_params[ctg_out_ind[ctg_ii][1]])
    end
end

# %%
qG.num_threads = 1
@btime ttb()
@btime ttf()

# %%
a      = 1:100
chunks = Iterators.partition(a, length(a) ÷ Threads.nthreads())

# %%

function sum_single(a)
    s = 0
    for i in a
        s += i
    end
    s
end

# %%
a = 1:100
chunks = Iterators.partition(a, length(a) ÷ Threads.nthreads())
tasks = map(chunks) do chunk
    Threads.@spawn sum_single(chunk)
end
chunk_sums = fetch.(tasks)

# %% ========
#return sum_single(chunk_sums)
gamma = 0
@floop ThreadedEx() for ii in 1:50
    @reduce(gamma += 1)
    println(gamma)
end

println(gamma)

# %%
l = Threads.SpinLock()
x = 0
Threads.@threads for i in 1:2
    Threads.lock(l)
    x += 1  # this block is executed only in one thread
    Threads.unlock(l)
end

println(x)

# %%

function f(c1::Vector{Vector{Float64}},c2::Vector{Vector{Float64}},c3::Vector{Vector{Float64}})
    l = Threads.SpinLock()
    #l = Threads.ReentrantLock()
    x = 0
    Threads.@threads for i in 1:100
        Threads.lock(l) do
            x += 1
        end

        #Threads.lock(l)
        #x += 1  # this block is executed only in one thread
        #Threads.unlock(l)

        c1[i] .= c1[i].^2
        c2[i] .= c2[i].^2
        c3[i] .= c3[i].^2
    end
    # println("===")
    return x
end

# %%
c1 = [zeros(500000) for ii in 1:100]
c2 = [zeros(500000) for ii in 1:100]
c3 = [zeros(500000) for ii in 1:100]

# %%

@btime f(c1, c2, c3)

# %%
#@time l = Threads.SpinLock()
@time l = Threads.ReentrantLock()

# %%===========
# => lck = Threads.SpinLock() -- SpinLock slower than ReentrantLock ..?

function ff()   
    lck = Threads.ReentrantLock()
    ready_to_use = ones(Bool, qG.num_threads)
    println("===")

    Threads.@threads for ii in 1:100
        Threads.lock(lck) do
            thrID = findfirst(ready_to_use)
            ready_to_use[thrID] = false
            println(thrID)
        end

        g = zeros(1000000)
        # all done!!
        Threads.lock(lck) do
            ready_to_use[thrID] = true
        end
    end

    println(ready_to_use)
end

ff()

# %%
for gg in zip(enumerate(3:8),17:22)
    println(gg)
end

# %%
function fg(v::Vector{Float64}, w::Vector{Float64})
    @floop ThreadedEx(basesize = 100 ÷ 10) for ii in 1:100
        @reduce(vvv .+= w)
    end

    return vvv
end

function fgd(v::Vector{Float64}, w::Vector{Float64})
    @floop ThreadedEx(basesize = 100 ÷ 10) for ii in 1:100
        @reduce() do (v; w)
            v .+= w
        end
    end

    return v
end


# %%
vv = ones(10000)
@btime fg(vv, vv);
@btime fgd(vv, vv);

# %%
vv = ones(150)

t1 = fg(vv, vv);

t2 = fgd(vv, vv);

# %% test ThreadsX
gamma = zeros()
v = [randn(10000) for ii in 1:100]
ThreadsX.sum(v[ii] for ii in 1:100)
sum(v[ii] for ii in 1:100)

# %% ==============
f1() = ThreadsX.sum(v[ii] for ii in 1:100);
f2() = sum(v[ii] for ii in 1:100);

# %%
t = randn(10000)
@time ThreadsX.sum(t)
@time sum(t)

# %%


function f3(vv, v)
    for ii in 1:100
        vv[ii] .+= v[ii]
    end
end

# %%
v = [randn(1000) for ii in 1:10]
vv = randn(1000)
f() = vv .= sum(v)

# %%
@time f();

# %%
v1 = randn(10000)
v2 = randn(10000)

# %%
@btime v1 .= f1();
@btime v2 .= f2();

# %%
v = [randn(10000) for ii in 1:100]

f1() = ThreadsX.sum(v[ii] for ii in 1:100);
f2() = sum(v[ii] for ii in 1:100);
f3() = ThreadsX.sum(v);
f4() = sum(v);

@btime f1();
@btime f2();
@btime f3();
@btime f4();

# %%
v = [randn(10000) for ii in 1:100]
w = randn(10000)

function f5(w::Vector{Float64},v::Vector{Vector{Float64}})
    for ii in 1:100
        for jj in 1:10000
            w[jj] += v[ii][jj]
        end
    end
end

# %%
@btime f5(w,v);