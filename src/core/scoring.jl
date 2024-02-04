# compute the total market surplus value
function score_zms!(scr::Dict{Symbol, Float64})
    # compute the negative market surplus function
    #
    # remember -- we are minimizing the negative market surplus function!!
    #
    # this is contrary to the standard paradigm: maximize the positive ms
    scr[:zms]           =  scr[:zbase]           + scr[:zctg_min] + scr[:zctg_avg]
    scr[:zms_penalized] =  scr[:zbase_penalized] + scr[:zctg_min] + scr[:zctg_avg]
    scr[:nzms]          = -scr[:zms]
end

# print the total market surplus value
function print_zms(qG::QuasiGrad.QG, scr::Dict{Symbol, Float64})
    # print score ======
    scr[:cnt] += 1.0
    if (qG.print_zms == true) && mod(scr[:cnt],qG.print_freq) == 0
        zms   = round(scr[:zms];           sigdigits = 5)
        zms_p = round(scr[:zms_penalized]; sigdigits = 5)
        zctg  = round(scr[:zctg_avg] + scr[:zctg_min]; sigdigits = 8)
        println("Penalized zms: $(zms_p)! Standard zms: $(zms)! Ctg score: $(zctg)!")
    end
end

# print the total market surplus value
function print_zms_adam_pf(qG::QuasiGrad.QG, scr::Dict{Symbol, Float64})
    # print score ======
    scr[:cnt] += 1.0
    if (qG.print_zms == true) && mod(scr[:cnt],qG.print_freq) == 0
        zt  = scr[:zp] + scr[:zq] + scr[:acl] + scr[:xfm]
        ztr = round(zt; sigdigits = 5)
        println("Injection + flow penalties: $ztr")
    end
end

# compute zbase
function score_zbase!(qG::QuasiGrad.QG, scr::Dict{Symbol, Float64})
    # compute the market surplus function
    scr[:emnx]            = scr[:z_enmax]     + scr[:z_enmin]
    scr[:zbase]           = scr[:zt_original] + scr[:emnx]
    scr[:zbase_penalized] = scr[:zbase]       + scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst]
end

function score_zt!(idx::QuasiGrad.Index, prm::QuasiGrad.Param, qG::QuasiGrad.QG, scr::Dict{Symbol, Float64}, stt::QuasiGrad.State)
    # update the base case variable zt
    #
    # note: zt = zt_original + zt_original -- otherwise, we make no distinction,
    #       meaning all derivatives are taken wrt "zt"
    #
    # reset:
    scr[:zt_original] = 0.0
    scr[:zt_penalty]  = 0.0
    scr[:encs]        = 0.0
    scr[:enpr]        = 0.0
    scr[:zp]          = 0.0
    scr[:zq]          = 0.0
    scr[:acl]         = 0.0
    scr[:xfm]         = 0.0
    scr[:zone]        = 0.0
    scr[:rsv]         = 0.0
    scr[:zoud]        = 0.0
    scr[:zsus]        = 0.0

    # fist, compute some useful scores to report
    for tii in prm.ts.time_keys
        scr[:encs] += sum(@view stt.zen_dev[tii][idx.cs_devs])
        scr[:enpr] -= sum(@view stt.zen_dev[tii][idx.pr_devs])
        scr[:zp]   -= sum(stt.zp[tii])
        scr[:zq]   -= sum(stt.zq[tii])
        scr[:acl]  -= sum(stt.zs_acline[tii])
        scr[:xfm]  -= sum(stt.zs_xfm[tii])
    end

    # => scr[:encs] = +sum(sum(@view stt.zen_dev[tii][idx.cs_devs] for tii in prm.ts.time_keys))
    # => scr[:enpr] = -sum(sum(@view stt.zen_dev[tii][idx.pr_devs] for tii in prm.ts.time_keys))
    # => scr[:zp]   = -sum(sum(stt.zp[tii]                   for tii in prm.ts.time_keys))
    # => scr[:zq]   = -sum(sum(stt.zq[tii]                   for tii in prm.ts.time_keys))
    # => scr[:acl]  = -sum(sum(stt.zs_acline[tii]            for tii in prm.ts.time_keys)) 
    # => scr[:xfm]  = -sum(sum(stt.zs_xfm[tii]               for tii in prm.ts.time_keys)) 
    for tii in prm.ts.time_keys
        scr[:zone] -= (
            # zonal reserve penalties (P) 
            sum(stt.zrgu_zonal[tii]) +  
            sum(stt.zrgd_zonal[tii]) + 
            sum(stt.zscr_zonal[tii]) + 
            sum(stt.znsc_zonal[tii]) + 
            sum(stt.zrru_zonal[tii]) +  
            sum(stt.zrrd_zonal[tii]) + 
            # zonal reserve penalties (Q)
            sum(stt.zqru_zonal[tii]) + 
            sum(stt.zqrd_zonal[tii]))
        scr[:rsv] -= (
            # local reserve penalties
            sum(stt.zrgu[tii]) +
            sum(stt.zrgd[tii]) +
            sum(stt.zscr[tii]) +
            sum(stt.znsc[tii]) +
            sum(stt.zrru[tii]) +
            sum(stt.zrrd[tii]) +
            sum(stt.zqru[tii]) +
            sum(stt.zqrd[tii]))
        scr[:zoud] -= (
            sum(stt.zon_dev[tii]   ) + 
            sum(stt.zsu_dev[tii]   ) + 
            sum(stt.zsu_acline[tii]) +
            sum(stt.zsu_xfm[tii]   ) + 
            sum(stt.zsd_dev[tii]   ) + 
            sum(stt.zsd_acline[tii]) + 
            sum(stt.zsd_xfm[tii]   ))
        scr[:zsus] -= sum(stt.zsus_dev[tii])
    end

    # => scr[:zone] = -(
    # =>     # zonal reserve penalties (P) 
    # =>     sum(sum(stt.zrgu_zonal[tii] for tii in prm.ts.time_keys)) +  
    # =>     sum(sum(stt.zrgd_zonal[tii] for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zscr_zonal[tii] for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.znsc_zonal[tii] for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zrru_zonal[tii] for tii in prm.ts.time_keys)) +  
    # =>     sum(sum(stt.zrrd_zonal[tii] for tii in prm.ts.time_keys)) + 
    # =>     # zonal reserve penalties (Q)
    # =>     sum(sum(stt.zqru_zonal[tii] for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zqrd_zonal[tii] for tii in prm.ts.time_keys)))
    # => scr[:rsv] = -(
    # =>     # local reserve penalties
    # =>     sum(sum(stt.zrgu[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.zrgd[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.zscr[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.znsc[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.zrru[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.zrrd[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.zqru[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.zqrd[tii] for tii in prm.ts.time_keys)))
    # => scr[:zoud] = -(
    # =>     sum(sum(stt.zon_dev[tii]    for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zsu_dev[tii]    for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zsu_acline[tii] for tii in prm.ts.time_keys)) +
    # =>     sum(sum(stt.zsu_xfm[tii]    for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zsd_dev[tii]    for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zsd_acline[tii] for tii in prm.ts.time_keys)) + 
    # =>     sum(sum(stt.zsd_xfm[tii]    for tii in prm.ts.time_keys)))
    # => scr[:zsus] = -sum(sum(stt.zsus_dev[tii] for tii in prm.ts.time_keys))
    
    # compute the original "zt" score
    scr[:zt_original] =  scr[:encs] + scr[:enpr] + scr[:zoud] + scr[:zsus] + scr[:acl] +
                         scr[:xfm]  + scr[:rsv]  + scr[:zp]   + scr[:zq]   + scr[:zone]

    for tii in prm.ts.time_keys
        # original, explicit scoring function!
        "scr[:zt_original] += 
            # consumer revenues (POSITIVE)
            sum(stt.zen_dev[tii][dev] for dev in idx.cs_devs) - 
            # producer costs
            sum(stt.zen_dev[tii][dev] for dev in idx.pr_devs) - 
            # startup costs
            sum(stt.zsu_dev[tii]) - 
            sum(stt.zsu_acline[tii]) - 
            sum(stt.zsu_xfm[tii]) - 
            # shutdown costs
            sum(stt.zsd_dev[tii]) - 
            sum(stt.zsd_acline[tii]) - 
            sum(stt.zsd_xfm[tii]) - 
            # on-costs
            sum(stt.zon_dev[tii]) - 
            # time-dependent su costs
            sum(stt.zsus_dev[tii]) - 
            # ac branch overload costs
            sum(stt.zs_acline[tii]) - 
            sum(stt.zs_xfm[tii]) - 
            # local reserve penalties
            sum(stt.zrgu[tii]) -
            sum(stt.zrgd[tii]) -
            sum(stt.zscr[tii]) -
            sum(stt.znsc[tii]) -
            sum(stt.zrru[tii]) -
            sum(stt.zrrd[tii]) -
            sum(stt.zqru[tii]) -
            sum(stt.zqrd[tii]) -
            # power mismatch penalties
            sum(stt.zp[tii]) -
            sum(stt.zq[tii]) -
            # zonal reserve penalties (P)
            sum(stt.zrgu_zonal[tii]) -
            sum(stt.zrgd_zonal[tii]) -
            sum(stt.zscr_zonal[tii]) -
            sum(stt.znsc_zonal[tii]) -
            sum(stt.zrru_zonal[tii]) -
            sum(stt.zrrd_zonal[tii]) -
            # zonal reserve penalties (Q)
            sum(stt.zqru_zonal[tii]) -
            sum(stt.zqrd_zonal[tii])"
        # penalized constraints
        scr[:zt_penalty] += -qG.constraint_grad_weight*(
            sum(stt.zhat_mndn[tii]) + 
            sum(stt.zhat_mnup[tii]) + 
            sum(stt.zhat_rup[tii]) + 
            sum(stt.zhat_rd[tii])  + 
            sum(stt.zhat_rgu[tii]) + 
            sum(stt.zhat_rgd[tii]) + 
            sum(stt.zhat_scr[tii]) + 
            sum(stt.zhat_nsc[tii]) + 
            sum(stt.zhat_rruon[tii])  + 
            sum(stt.zhat_rruoff[tii]) +
            sum(stt.zhat_rrdon[tii])  +
            sum(stt.zhat_rrdoff[tii]) +
            # common set of pr and cs constraint variables (see below)
            sum(stt.zhat_pmax[tii])      + 
            sum(stt.zhat_pmin[tii])      + 
            sum(stt.zhat_pmaxoff[tii])   + 
            sum(stt.zhat_qmax[tii])      + 
            sum(stt.zhat_qmin[tii])      + 
            sum(stt.zhat_qmax_beta[tii]) + 
            sum(stt.zhat_qmin_beta[tii]))
    end
end

function score_solve_pf!(lbf::QuasiGrad.LBFGS, prm::QuasiGrad.Param, stt::QuasiGrad.State)
    # all we want to track here is the power flow score
    #
    # note: these scores are positive, since we try to minimize them!!
    for tii in prm.ts.time_keys
        lbf.zpf[:zp][tii] = sum(stt.zp[tii])
        lbf.zpf[:zq][tii] = sum(stt.zq[tii])
        lbf.zpf[:zs][tii] = sum(stt.zs_acline[tii]) + sum(stt.zs_xfm[tii])
    end
end

# soft abs derviative
function soft_abs(x::Float64, eps2::Float64)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    sqrt(x^2 + eps2)
end

# soft abs derviative -- constraints
function soft_abs_constraint_grad(x::Float64, qG::QuasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    if qG.constraint_grad_is_soft_abs
        return x/(QuasiGrad.LoopVectorization.sqrt_fast(QuasiGrad.LoopVectorization.pow_fast(x,2) + qG.constraint_grad_eps2))
    else
        return sign(x)
    end
end

# soft abs derviative -- reserves
function soft_abs_reserve_grad(x::Float64, qG::QuasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x/(QuasiGrad.LoopVectorization.sqrt_fast(QuasiGrad.LoopVectorization.pow_fast(x,2) + qG.reserve_grad_eps2))
end

# soft abs derviative -- acflow
function soft_abs_acflow_grad(x::Float64, qG::QuasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x/(QuasiGrad.LoopVectorization.sqrt_fast(QuasiGrad.LoopVectorization.pow_fast(x,2) + qG.acflow_grad_eps2))
end

# soft abs derviative -- ctg
function soft_abs_ctg_grad(x::Float64, qG::QuasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x/(QuasiGrad.LoopVectorization.sqrt_fast(QuasiGrad.LoopVectorization.pow_fast(x,2) + qG.ctg_grad_eps2))
end

function print_penalty_breakdown(idx::QuasiGrad.Index, prm::QuasiGrad.Param, qG::QuasiGrad.QG, scr::Dict{Symbol, Float64}, stt::QuasiGrad.State)
    println("")
    println("")
    println("=== === Scoring Report (**run after post-process routine**) === === ")
    println(" • zms: $(scr[:zms])")
    println(" • ed: $(scr[:ed_obj])")
    gap = round(100.0*scr[:zms]/scr[:ed_obj], sigdigits = 5)
    println(" • gap: $gap %")
    println("")

    # infeasibilities
    infeasibility_penalties = scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst]
    println("Infeasibility penalties: $infeasibility_penalties")
    println("")

    encs_fixed = 0.0
    enpr_fixed = 0.0
    for tii in prm.ts.time_keys
        encs_fixed += sum(stt.zen_dev[tii][idx.cs_devs][stt.zen_dev[tii][idx.cs_devs] .> 0.0])
        enpr_fixed -= sum(stt.zen_dev[tii][idx.pr_devs][stt.zen_dev[tii][idx.pr_devs] .> 0.0])
    
        # now, for the ones with the opposite signs -- this is very tricky :)
        enpr_fixed += sum(stt.zen_dev[tii][idx.cs_devs][stt.zen_dev[tii][idx.cs_devs] .< 0.0])
        encs_fixed -= sum(stt.zen_dev[tii][idx.pr_devs][stt.zen_dev[tii][idx.pr_devs] .< 0.0])
    end


    # rewards
    rewards        = scr[:zsus] + encs_fixed
    sus_percent    = round(100.0*scr[:zsus]/rewards, sigdigits = 5)
    energy_percent = round(100.0*encs_fixed/rewards, sigdigits = 5)

    println("Rewards: $rewards")
    println(" • energy: $energy_percent %")
    println(" • sus: $sus_percent %")
    println("")

    # penalties
    penalties = enpr_fixed + scr[:zoud] + scr[:acl] + scr[:xfm] + scr[:rsv] +
                    scr[:zp] + scr[:zq] + scr[:zone] + scr[:z_enmax] + scr[:z_enmin] + scr[:zctg_min] + scr[:zctg_avg]
    energy_percent     = round(100.0*enpr_fixed/penalties, sigdigits = 5)
    on_up_down_percent = round(100.0*scr[:zoud]/penalties, sigdigits = 5)  
    acline_percent     = round(100.0*scr[:acl]/penalties, sigdigits = 5)  
    xfm_percent        = round(100.0*scr[:xfm]/penalties, sigdigits = 5)  
    reserve_percent    = round(100.0*scr[:rsv]/penalties, sigdigits = 5)  
    pb_percent         = round(100.0*scr[:zp]/penalties, sigdigits = 5)  
    qb_percent         = round(100.0*scr[:zq]/penalties, sigdigits = 5)  
    zone_percent       = round(100.0*scr[:zone]/penalties, sigdigits = 5)
    enmax_percent      = round(100.0*scr[:z_enmax]/penalties, sigdigits = 5)  
    enmin_percent      = round(100.0*scr[:z_enmin]/penalties, sigdigits = 5)
    ctg_min_percent    = round(100.0*scr[:zctg_min]/penalties, sigdigits = 5)  
    ctg_avg_percent    = round(100.0*scr[:zctg_avg]/penalties, sigdigits = 5)

    println("Penalties: $penalties")
    println(" • energy: $energy_percent %")
    println(" • on/up/down: $on_up_down_percent %")
    println(" • acline: $acline_percent %")
    println(" • xfm: $xfm_percent %")
    println(" • reserve costs: $reserve_percent %")
    println(" • power balance (P): $pb_percent %")
    println(" • power balance (Q): $qb_percent %")
    println(" • zonal penalties: $zone_percent %")
    println(" • maximum energy: $enmax_percent %")
    println(" • minimum energy: $enmin_percent %")
    println(" • ctg min: $ctg_min_percent %")
    println(" • ctg avg: $ctg_avg_percent %")
    println("")
end