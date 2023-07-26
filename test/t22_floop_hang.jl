using quasiGrad
using Revise
using Polyester

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00014/scenario_003.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
jsn  = quasiGrad.load_json(path)

# initialize
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn, perturb_states=true);

# %% run copper plate ED
quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# %% compute all states and grads
qG.skip_ctg_eval = true
qG.num_threads   = 10

function f() 
    #quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
    #quasiGrad.Polyester.reset_threads!()
    #quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
    quasiGrad.clip_all!(prm, qG, stt)
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
    #quasiGrad.clip_all!(prm, qG, stt)
end

# %%
@btime quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()

# %%
function g(grd::quasiGrad.Grad, mgd::quasiGrad.Mgd, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System) 
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
@btime quasiGrad.clip_all!(prm, qG, stt)

# %%
@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
@btime quasiGrad.clip_all!(prm, qG, stt)
# %%

@benchmark quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
# @btime 

# %% ====================== 
qG.clip_pq_based_on_bins   = false
qG.acflow_grad_is_soft_abs = true
qG.num_threads = 10

@btime quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)
@btime quasiGrad.clip_all!(prm, qG, stt)
@btime quasiGrad.acline_flows!(grd, idx, msc, prm, qG, stt, sys)
@btime quasiGrad.xfm_flows!(grd, idx, msc, prm, qG, stt, sys)
@btime quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)
@btime quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
@btime quasiGrad.device_startup_states!(grd, idx, mgd, msc, prm, qG, stt, sys)
@btime quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
@btime quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
@btime quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
@btime quasiGrad.energy_penalties!(grd, idx, msc, prm, qG, scr, stt, sys)
@btime quasiGrad.penalized_device_constraints!(grd, idx, mgd, msc, prm, qG, scr, stt, sys)
@btime quasiGrad.device_reserve_costs!(prm, qG, stt)
@btime quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)
@btime quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)
# @time quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
@btime quasiGrad.score_zt!(idx, prm, qG, scr, stt) 
@btime quasiGrad.score_zbase!(qG, scr)
@btime quasiGrad.score_zms!(scr)
# @btime quasiGrad.print_zms(qG, scr)
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)

# %%
@time quasiGrad.device_startup_states!(grd, idx, mgd, msc, prm, qG, stt, sys)

# %%
qG.num_threads = 1
@time quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

# %%
v = randn(50)
@batch per=thread for ii in 1:100
    t = maximum(v[ij] for ij in 1:5)
end

# %%
qG.num_threads = 6

@btime quasiGrad.acline_flows!(grd, idx, msc, prm, qG, stt, sys)

# %%

@btime quasiGrad.acline_flows_poly!(grd, idx, msc, prm, qG, stt, sys)

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
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

qG.adam_max_time = 60.0
qG.num_threads   = 10

# %%
qG.num_threads = 1
@btime quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)

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
    quasiGrad.clip_all!(prm, qG, stt)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)

    # device powers
    quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
    quasiGrad.device_startup_states!(grd, idx, mgd, msc, prm, qG, stt, sys)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
    quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    quasiGrad.energy_penalties!(grd, idx, msc, prm, qG, scr, stt, sys)
    quasiGrad.penalized_device_constraints!(grd, idx, mgd, msc, prm, qG, scr, stt, sys)
    quasiGrad.device_reserve_costs!(prm, qG, stt)

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)

    # compute reserve margins and penalties (no grads here)
    quasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

    # score the contingencies and take the gradients
    if qG.skip_ctg_eval
        #println("Skipping ctg evaluation!")
    else
        quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
    end
    # score the market surplus function
    quasiGrad.score_zt!(idx, prm, qG, scr, stt) 
    quasiGrad.score_zbase!(qG, scr)
    quasiGrad.score_zms!(scr)

    # print the market surplus function value
    #quasiGrad.print_zms(qG, scr)

    # compute the master grad
    #GC.safepoint()
    #quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)

    # print the market surplus function value
    #quasiGrad.print_zms(qG, scr)

    # compute the master grad
    # quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)
    if mod(ii,1) == 0
        println(ii)
    end
end

# %%
qG.skip_ctg_eval = true
qG.num_threads   = 10
t1 = time()
for ii in 1:250
    quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
end
tend = time() - t1

# %%
tii = Int8(1)
@btime quasiGrad.master_grad_zs_acline!(tii, idx, grd, mgd, msc, qG, sys)

@btime quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, msc, qG, sys)


# %%
@btime quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)

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
    quasiGrad.master_grad!(cgd, grd, idx, mgd, msc, prm, qG, stt, sys)
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
    #    quasiGrad.master_grad_zs_xfm!(tii, idx, grd, mgd, qG, sys)
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
quasiGrad.run_adam!(adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

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
quasiGrad.flush_adam!(adm, prm, upd)

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
@btime quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %%
quasiGrad.acline_flows!(grd, idx, msc, prm, qG, stt, sys)

# %% take an adam step 
qG.num_threads = 1

#ProfileView.@profview 

@btime quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

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