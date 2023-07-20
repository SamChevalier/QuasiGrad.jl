using quasiGrad
using Revise

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00014/scenario_003.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
jsn  = quasiGrad.load_json(path)

# initialize
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn, perturb_states=false);

# %% run copper plate ED
quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# %% 
qG.apply_grad_weight_homotopy  = false
qG.take_adam_pf_steps          = false
qG.pqbal_grad_type             = "soft_abs" 
qG.constraint_grad_is_soft_abs = false
qG.acflow_grad_is_soft_abs     = false
qG.reserve_grad_is_soft_abs    = false
qG.skip_ctg_eval               = true

qG.adam_max_time = 60.0
qG.num_threads   = 10

# ========================
qG.num_threads = 4
for ii in 1:1000
    # safepoint
    #GC.safepoint()

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
    quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(bit, grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)

    # device powers
    quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
    quasiGrad.device_startup_states!(grd, idx, mgd, msc, prm, qG, stt, sys)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
    quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
    quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
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
    #quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

    # print the market surplus function value
    #quasiGrad.print_zms(qG, scr)

    # compute the master grad
    # quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
    if mod(ii,5) == 0
        println(ii)
    end
end

# %%

qG.num_threads = 6
for ii in 1:1000
    # start by looping over the terms which can be safely looped over in time
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        quasiGrad.master_grad_zs_acline_test!(tii, idx, grd, mgd, qG, sys)
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