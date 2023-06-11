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
function print_zms(qG::quasiGrad.QG, scr::Dict{Symbol, Float64})
    # print score ======
    scr[:cnt] += 1.0
    if (qG.print_zms == true) && mod(scr[:cnt],qG.print_freq) == 0
        zms   = round(scr[:zms];           sigdigits = 5)
        zms_p = round(scr[:zms_penalized]; sigdigits = 5)
        println("Penalized zms: $(zms_p)! Standard zms: $(zms)!")
    end
end

# compute zbase
function score_zbase!(qG::quasiGrad.QG, scr::Dict{Symbol, Float64})
    # compute the market surplus function
    scr[:emnx]            = scr[:z_enmax]     + scr[:z_enmin]
    scr[:zbase]           = scr[:zt_original] + scr[:emnx]
    scr[:zbase_penalized] = scr[:zbase]       + scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst]
end

function score_zt!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    # update the base case variable zt
    #
    # note: zt = zt_original + zt_original -- otherwise, we make no distinction,
    #       meaning all derivatives are taken wrt "zt"
    #
    # reset:
    scr[:zt_original] = 0.0
    scr[:zt_penalty]  = 0.0

    # fist, compute some useful scores to report
    scr[:encs] = +sum(sum(stt[:zen_dev][tii][idx.cs_devs] for tii in prm.ts.time_keys))
    scr[:enpr] = -sum(sum(stt[:zen_dev][tii][idx.pr_devs] for tii in prm.ts.time_keys))
    scr[:zp]   = -sum(sum(stt[:zp][tii]                   for tii in prm.ts.time_keys))
    scr[:zq]   = -sum(sum(stt[:zq][tii]                   for tii in prm.ts.time_keys))
    scr[:acl]  = -sum(sum(stt[:zs_acline][tii]            for tii in prm.ts.time_keys)) 
    scr[:xfm]  = -sum(sum(stt[:zs_xfm][tii]               for tii in prm.ts.time_keys)) 

    scr[:zone] = -(
        # zonal reserve penalties (P) 
        sum(sum(stt[:zrgu_zonal][tii] for tii in prm.ts.time_keys)) +  
        sum(sum(stt[:zrgd_zonal][tii] for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zscr_zonal][tii] for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:znsc_zonal][tii] for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zrru_zonal][tii] for tii in prm.ts.time_keys)) +  
        sum(sum(stt[:zrrd_zonal][tii] for tii in prm.ts.time_keys)) + 
        # zonal reserve penalties (Q)
        sum(sum(stt[:zqru_zonal][tii] for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zqrd_zonal][tii] for tii in prm.ts.time_keys)))
    scr[:rsv] = -(
        # local reserve penalties
        sum(sum(stt[:zrgu][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:zrgd][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:zscr][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:znsc][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:zrru][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:zrrd][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:zqru][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:zqrd][tii] for tii in prm.ts.time_keys)))
    scr[:zoud] = -(
        sum(sum(stt[:zon_dev][tii]    for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zsu_dev][tii]    for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zsu_acline][tii] for tii in prm.ts.time_keys)) +
        sum(sum(stt[:zsu_xfm][tii]    for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zsd_dev][tii]    for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zsd_acline][tii] for tii in prm.ts.time_keys)) + 
        sum(sum(stt[:zsd_xfm][tii]    for tii in prm.ts.time_keys)))
    scr[:zsus] = -sum(sum(stt[:zsus_dev][tii] for tii in prm.ts.time_keys))
    
    # compute the original "zt" score
    scr[:zt_original] =  scr[:encs] + scr[:enpr] + scr[:zoud] + scr[:zsus] + scr[:acl] +
                         scr[:xfm]  + scr[:rsv]  + scr[:zp]   + scr[:zq]   + scr[:zone]

    for tii in prm.ts.time_keys
        # original, explicit scoring function!
        "scr[:zt_original] += 
            # consumer revenues (POSITIVE)
            sum(stt[:zen_dev][tii][dev] for dev in idx.cs_devs) - 
            # producer costs
            sum(stt[:zen_dev][tii][dev] for dev in idx.pr_devs) - 
            # startup costs
            sum(stt[:zsu_dev][tii]) - 
            sum(stt[:zsu_acline][tii]) - 
            sum(stt[:zsu_xfm][tii]) - 
            # shutdown costs
            sum(stt[:zsd_dev][tii]) - 
            sum(stt[:zsd_acline][tii]) - 
            sum(stt[:zsd_xfm][tii]) - 
            # on-costs
            sum(stt[:zon_dev][tii]) - 
            # time-dependent su costs
            sum(stt[:zsus_dev][tii]) - 
            # ac branch overload costs
            sum(stt[:zs_acline][tii]) - 
            sum(stt[:zs_xfm][tii]) - 
            # local reserve penalties
            sum(stt[:zrgu][tii]) -
            sum(stt[:zrgd][tii]) -
            sum(stt[:zscr][tii]) -
            sum(stt[:znsc][tii]) -
            sum(stt[:zrru][tii]) -
            sum(stt[:zrrd][tii]) -
            sum(stt[:zqru][tii]) -
            sum(stt[:zqrd][tii]) -
            # power mismatch penalties
            sum(stt[:zp][tii]) -
            sum(stt[:zq][tii]) -
            # zonal reserve penalties (P)
            sum(stt[:zrgu_zonal][tii]) -
            sum(stt[:zrgd_zonal][tii]) -
            sum(stt[:zscr_zonal][tii]) -
            sum(stt[:znsc_zonal][tii]) -
            sum(stt[:zrru_zonal][tii]) -
            sum(stt[:zrrd_zonal][tii]) -
            # zonal reserve penalties (Q)
            sum(stt[:zqru_zonal][tii]) -
            sum(stt[:zqrd_zonal][tii])"
        # penalized constraints
        scr[:zt_penalty] += -qG.constraint_grad_weight*(
            sum(stt[:zhat_mndn][tii]) + 
            sum(stt[:zhat_mnup][tii]) + 
            sum(stt[:zhat_rup][tii]) + 
            sum(stt[:zhat_rd][tii])  + 
            sum(stt[:zhat_rgu][tii]) + 
            sum(stt[:zhat_rgd][tii]) + 
            sum(stt[:zhat_scr][tii]) + 
            sum(stt[:zhat_nsc][tii]) + 
            sum(stt[:zhat_rruon][tii])  + 
            sum(stt[:zhat_rruoff][tii]) +
            sum(stt[:zhat_rrdon][tii])  +
            sum(stt[:zhat_rrdoff][tii]) +
            # common set of pr and cs constraint variables (see below)
            sum(stt[:zhat_pmax][tii])      + 
            sum(stt[:zhat_pmin][tii])      + 
            sum(stt[:zhat_pmaxoff][tii])   + 
            sum(stt[:zhat_qmax][tii])      + 
            sum(stt[:zhat_qmin][tii])      + 
            sum(stt[:zhat_qmax_beta][tii]) + 
            sum(stt[:zhat_qmin_beta][tii]))
    end
end

function score_solve_pf!(prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, zpf::Dict{Symbol, Dict{Symbol, Float64}})
    # all we want to track here is the power flow score
    #
    # note: these scores are positive, since we try to minimize them!!
    for tii in prm.ts.time_keys
        zpf[:zp][tii]   = sum(stt[:zp][tii])
        zpf[:zq][tii]   = sum(stt[:zq][tii])
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
function soft_abs_constraint_grad(x::Float64, qG::quasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x/(sqrt(x^2 + qG.constraint_grad_eps2))
end

# soft abs derviative -- reserves
function soft_abs_reserve_grad(x::Float64, qG::quasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x/(sqrt(x^2 + qG.reserve_grad_eps2))
end

# soft abs derviative
function soft_abs_grad_faster(x::Vector{Float64}, qG::quasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x./(sqrt.(x.^2 .+ qG.acflow_grad_eps2))
end

function soft_abs_grad_ac_old(x::Float64, qG::quasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x/(sqrt(x^2 + qG.acflow_grad_eps2))
end

function soft_abs_grad_ac(ind::Int64, msc::Dict{Symbol, Vector{Float64}}, qG::quasiGrad.QG, s::Symbol)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return msc[s][ind]/(sqrt(msc[s][ind]^2 + qG.acflow_grad_eps2))
end

# soft abs derviative
function soft_abs_grad_vec!(bit::Dict{Symbol, BitVector}, msc::Dict{Symbol, Vector{Float64}}, qG::quasiGrad.QG, s_bit::Symbol, s_in::Symbol, s_out::Symbol)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0) #
    msc[s_out][bit[s_bit]] .= @view msc[s_in][bit[s_bit]]#./(sqrt.((@view msc[s_in][bit[s_bit]]).^2 .+ qG.acflow_grad_eps2))
end
