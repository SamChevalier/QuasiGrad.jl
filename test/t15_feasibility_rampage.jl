using quasiGrad
using Revise

include("./test_functions.jl")

# %% ==================================== C3S1_20221222 ==================================== #
#
# =============== D1
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N06049D1/scenario_001.json"

# =============== D2
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D2/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D2/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D2/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N06049D2/scenario_001.json"

# =============== D3
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D3/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D3/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D3/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N06049D3/scenario_001.json"


# ==================================== C3S3.1_20230606 ==================================== #
#
# =============== D1
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00014D1/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D1/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N01576D1/scenario_007.json"
#path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N02000D1/scenario_001.json"
#path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N08316D1/scenario_001.json"
#path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"
#
# =============== D2
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00014D2/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D2/scenario_001.json"
#
# =============== D3
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00014D3/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D3/scenario_001.json"


# ==================================== C3E2D2_20230510 ==================================== #
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E2D2_20230510/C3E2N01576D2/scenario_002.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E2D2_20230510/C3E2N01576D2/scenario_130.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E2D2_20230510/C3E2N04224D2/scenario_033.json"

# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"

# path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"
# path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
# %% path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D2/scenario_001.json"
include("./test_functions.jl")
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N01576D1/scenario_007.json"
solution_file = "solution.jl"
load_and_project(path, solution_file)

# %% ===========
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"

InFile1 = path
jsn = quasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

# %% ======= Fix zsus
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N01576D1/scenario_007.json"
InFile1 = path
jsn = quasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

# %% solve
fix       = true
pct_round = 100.0
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% =============
adm_step    = 0
beta1       = qG.beta1
beta2       = qG.beta2
beta1_decay = 1.0
beta2_decay = 1.0
run_adam    = true
quasiGrad.flush_adam!(adm, flw, prm, upd)

# %% ====
adm_step += 1

# step decay
# alpha = step_decay(adm_step, qG)

# decay beta
beta1_decay = beta1_decay*beta1
beta2_decay = beta2_decay*beta2

# compute all states and grads
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# take an adam step

quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
# %% ==============

qG.adam_max_time = 10.0
@time quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% ==============

quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)

quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
# %%

quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)


# %% == 
for tii in prm.ts.time_keys
    for dev in 1:sys.ndev
        # first, we bound ("bnd") the startup state ("sus"):
        # the startup state can only be active if the device
        # has been on within some recent time period.
        #
        # flush the sus
        stt.zsus_dev[tii][dev] = 0.0

        # loop over sus (i.e., f in F)
        if prm.dev.num_sus[dev] > 0
            # before looping over the startup states, flush stt and sus_bnds
            stt.zsus_dev[tii] .= 0.0

            for ii in 1:prm.dev.num_sus[dev] # 1:min(prm.dev.num_sus[dev],1)
                if prm.dev.startup_states[dev][ii][1] < 0.0 # skip if 0! why are these even here?
                    # grab the sets of T_sus
                    # => T_sus_jft = idx.Ts_sus_jft[dev][tii][ii] # T_sus_jft, T_sus_jf = get_tsus_sets(tii, dev, prm, ii)
                    # => T_sus_jf  = idx.Ts_sus_jf[dev][tii][ii]  # T_sus_jft, T_sus_jf = get_tsus_sets(tii, dev, prm, ii)

                    if tii in idx.Ts_sus_jf[dev][tii][ii]
                        if tii == :t1
                            # this is an edge case, where there are no previous states which
                            # could be "on" (since we can't turn on the generator in the fixed
                            # past, and it wasn't on)
                            # ** stt[:u_sus_bnd][tii][dev][ii] = 0.0
                            u_sus_bnd = 0.0
                        else
                            u_on_max_ind = argmax([stt.u_on_dev[tii_inst][dev] for tii_inst in idx.Ts_sus_jft[dev][tii][ii]])
                            u_sus_bnd    = stt.u_on_dev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev]
                            # => alt: u_sus_bnd = maximum([stt.u_on_dev[tii_inst][dev] for tii_inst in idx.Ts_sus_jft[dev][tii][ii]])
                            
                            # => u_sus_bnd = maximum([stt.u_on_dev[tii_inst][dev] for tii_inst in T_sus_jft])
                            # ** stt[:u_sus_bnd][tii][dev][ii] = stt.u_on_dev[T_sus_jft[u_on_max_ind]][dev]
                        end
                        #
                        # note: u_on_max == stt.u_on_dev[T_sus_jft[u_on_max_ind]][dev]
                        #
                        # previous bound based on directly taking the max:
                            # stt[:u_sus_bnd][tii][dev][ii] = max.([stt.u_on_dev[tii_inst][dev] for tii_inst in T_sus_jft])
                        # previous bound based on the sum (rather than max)
                            # stt[:u_sus_bnd][tii][dev][ii] = max.(sum(stt.u_on_dev[tii_inst][dev] for tii_inst in T_sus_jft; init=0.0), 1.0)
                    else
                        # ok, in this case the device was on in a sufficiently recent time (based on
                        # startup conditions), so we don't need to compute a bound
                        u_sus_bnd = 1.0
                        # this!!! u_sus_bnd = 1.0
                        # ** stt[:u_sus_bnd][tii][dev][ii] = 1.0
                    end

                    # now, compute the discount/cost ==> this is "+=", since it is over all (f in F) states
                    if u_sus_bnd > 0.0
                        stt.zsus_dev[tii][ii] = prm.dev.startup_states[dev][ii][1]*min(stt.u_su_dev[tii][dev],u_sus_bnd)

                        #stt.zsus_dev[tii][dev] += prm.dev.startup_states[dev][ii][1]*min(stt.u_su_dev[tii][dev],u_sus_bnd)
                    end
                end
            end

            # now, we score, and then take a gradient
            ii = argmin(stt.zsus_dev[tii])
            if stt.zsus_dev[tii][ii] < 0.0
                # update the score and take the gradient
                stt.zsus_dev[tii][dev] += stt.zsus_dev[tii][ii]

                # this is all pretty expensive, so let's take the gradient right here
                #
                # evaluate gradient?
                if qG.eval_grad
                    # f it -- just recompute these
                    u_on_max_ind = argmax([stt.u_on_dev[tii_inst][dev] for tii_inst in idx.Ts_sus_jft[dev][tii][ii]])
                    u_sus_bnd    = stt.u_on_dev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev]

                    # OG => gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsus_dev] * prm.dev.startup_states[dev][ii][1]
                    # test which was smaller: u_su, or the su_bound?
                    #
                    # we want "<=" so that we never end up in a case where 
                    # we try to take the gradient of u_sus_bnd == 1 (see else case above)
                    if stt.u_su_dev[tii][dev] <= u_sus_bnd # ** stt[:u_sus_bnd][tii][dev][ii]
                        # in this case, there is an available discount, so we want u_su
                        # to feel a bit less downward pressure and rise up (potentially)
                        mgd.u_on_dev[tii][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev[tii][dev]
                        if tii != :t1
                            # previous time?
                            mgd.u_on_dev[prm.ts.tmin1[tii]][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev_prev[tii][dev]
                        end
                    else
                        # in this case, sus bound is lower than u_su, so we'll put some pressure on the
                        # previous largest u_on, trying to push it up, in order to extract a little value
                        # from this sus.. :)
                        #
                        # what time is associated with this derivative? it is the time associated with the max u_on
                        if tii != :t1
                            # skip the gradient if tii == :t1, since stt[:u_sus_bnd] == 0 and no gradient exists
                            # -- this is a weird edge case, but it does make sense if you think about it for
                            # long enough.....
                                # => tt_max = T_sus_jft[u_on_max_ind]
                            mgd.u_on_dev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev]
                            if idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind] != :t1
                                # previous time?
                                mgd.u_on_dev[prm.ts.tmin1[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]]][dev] += prm.dev.startup_states[dev][ii][1]*grd.u_su_dev.u_on_dev_prev[idx.Ts_sus_jft[dev][tii][ii][u_on_max_ind]][dev]
                            end
                        end
                    end
                end
            end
        end
    end
end

# %%
scr[:zsus] = -sum(sum(stt.zsus_dev[tii] for tii in prm.ts.time_keys))
println(-scr[:zsus])


# %% Test injections!!
tii = :t1

quasiGrad.ideal_dispatch!(idx, stt, sys, tii)

Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii);
Jac = quasiGrad.build_acpf_Jac_and_pq0(qG, stt, sys, tii, Ybus_real, Ybus_imag);

stt.pinj_ideal[tii]
stt.qinj_ideal[tii]

stt.pinj0[tii]
stt.qinj0[tii]


# %% ============== zsus :) ===========
using quasiGrad
using Revise
include("./test_functions.jl")

path    = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
InFile1 = path
jsn     = quasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=true, pert_size=1.0)

# %% solve
fix       = true
pct_round = 100.0
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)

# %%
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

#quasiGrad.reserve_cleanup_parallel!(idx, prm, qG, stt, sys, upd)

# %%
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% ======================= 
quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
quasiGrad.acline_flows_parallel!(bit, grd, idx, prm, qG, stt, sys)

# %% ===
@time quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

@time quasiGrad.solve_ctgs_parallel!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

# %% =====================:
quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
scr[:zsus] = -sum(sum(stt.zsus_dev[tii] for tii in prm.ts.time_keys))
println(-scr[:zsus])

# %% -- sum
zsus_devs = zeros(2066)
for tii in prm.ts.time_keys
    devs = findall(stt.zsus_dev[tii] .< 0.0)
    for dev in devs
        dev_id = prm.dev.id[dev]
        vals   = stt.zsus_dev[tii][dev]
        println("dev: $dev_id, t: $tii, z_sus: $vals")
    end
    zsus_devs .+= stt.zsus_dev[tii]
end

# %% all
devs = findall(zsus_devs.< 0.0)
println(zsus_devs[devs])
println(prm.dev.id[devs])
