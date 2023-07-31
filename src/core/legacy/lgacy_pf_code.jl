# get injection bounds
function get_injection_bounds!(idx::quasiGrad.Index, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # note 1: this function DOES take reactive power equality constrained injections
    #         into account: i.e., dev_q ∈ Jpqe is fixed constant.

    # note 2: this function DOES **NOT** take reactive power inequality constraints
    #         a la pqmax/pqmin via beta_max/min into account -- these are just penalized
    #         later on, and then projected onto.

    # note 3: dc line terminals are treated as independent injections.
    #         they are treated as linked on the active power side, though.

    # warning
    @info "note: the get_injection_bounds function does NOT take J pqe into account yet"

    # time index
    tii = prm.ts.time_key_ind[tii]

    # at this time, compute the pr and cs upper and lower bounds across all devices --
    # -> skip pqe equality links!
    dev_plb = stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii]
    dev_pub = stt.u_on_dev[tii].*prm.dev.p_ub_tmdv[tii]
    dev_qlb = stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]
    dev_qub = stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]

    # note: clipping is based on the upper/lower bounds, and not
    # based on the beta linking equations -- so, we just treat
    # that as a penalty, and not as a power balance factor
    # 
    # also, compute the dc line upper and lower bounds
    # => dcfr_plb = -prm.dc.pdc_ub
    # => dcfr_pub = prm.dc.pdc_ub
    # => dcto_plb = -prm.dc.pdc_ub
    # => dcto_pub = prm.dc.pdc_ub
    dcfr_qlb = prm.dc.qdc_fr_lb
    dcfr_qub = prm.dc.qdc_fr_ub
    dcto_qlb = prm.dc.qdc_to_lb
    dcto_qub = prm.dc.qdc_to_ub

    # loop
    for bus in 1:sys.nb

        # get limits -- P -- ignore dc line powers (included later!) +&&+ ignore devices in Jpqe (included later!)
        pr_Plb   = sum(dev_plb[idx.bus_to_pr_not_Jpqe[bus]]; init=0.0)
        cs_Plb   = sum(dev_plb[idx.bus_to_cs_not_Jpqe[bus]]; init=0.0)
        pr_Pub   = sum(dev_pub[idx.bus_to_pr_not_Jpqe[bus]]; init=0.0)
        cs_Pub   = sum(dev_pub[idx.bus_to_cs_not_Jpqe[bus]]; init=0.0)
        # => dcfr_Plb = sum(dcfr_plb[idx.bus_is_dc_frs[bus]]; init=0.0)
        # => dcfr_Pub = sum(dcfr_pub[idx.bus_is_dc_frs[bus]]; init=0.0)
        # => dcto_Plb = sum(dcto_plb[idx.bus_is_dc_tos[bus]]; init=0.0)
        # => dcto_Pub = sum(dcto_pub[idx.bus_is_dc_tos[bus]]; init=0.0) 

        # get limits -- Q -- ignore devices in Jpqe (included later!)
        pr_Qlb   = sum(dev_qlb[idx.bus_to_pr_not_Jpqe[bus]]; init=0.0)
        cs_Qlb   = sum(dev_qlb[idx.bus_to_cs_not_Jpqe[bus]]; init=0.0)
        pr_Qub   = sum(dev_qub[idx.bus_to_pr_not_Jpqe[bus]]; init=0.0) 
        cs_Qub   = sum(dev_qub[idx.bus_to_cs_not_Jpqe[bus]]; init=0.0)
        dcfr_Qlb = sum(dcfr_qlb[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcfr_Qub = sum(dcfr_qub[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcto_Qlb = sum(dcto_qlb[idx.bus_is_dc_tos[bus]]; init=0.0)
        dcto_Qub = sum(dcto_qub[idx.bus_is_dc_tos[bus]]; init=0.0) 

        # total: lb < -pb_slack < ub
        msc.pub[bus] = pr_Pub - (cs_Plb + dcfr_Plb + dcto_Plb)
        msc.plb[bus] = pr_Plb - (cs_Pub + dcfr_Pub + dcto_Pub)
        
        # total: lb < -qb_slack < ub
        msc.qub[bus] = pr_Qub - (cs_Qlb + dcfr_Qlb + dcto_Qlb)
        msc.qlb[bus] = pr_Qlb - (cs_Qub + dcfr_Qub + dcto_Qub)
    end
end

function power_flow_residual_KP!(idx::quasiGrad.Index, KP::Float64, pi_p::Vector{Float64}, residual::Vector{Float64}, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # loop over each bus and compute the residual
    for bus in 1:sys.nb
        # active power balance: stt[:pb][:slack][tii][bus] to record with time
        residual[bus] = 
            # slack!!!
            pi_p[bus]*KP + 
            # consumers (positive)
            sum(stt.dev_p[tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt.sh_p[tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt.acline_pfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt.acline_pto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt.xfm_pfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt.xfm_pto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt.dc_pfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt.dc_pto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt.dev_p[tii][idx.pr[bus]]; init=0.0)
    
        # reactive power balance
        residual[sys.nb + bus] = 
            # consumers (positive)
            sum(stt.dev_q[tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt.sh_q[tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt.acline_qfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt.acline_qto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt.xfm_qfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt.xfm_qto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt.dc_qfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt.dc_qto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt.dev_q[tii][idx.pr[bus]]; init=0.0)
    end
end


function power_flow_residual_kpkq!(idx::quasiGrad.Index, KP::Float64, KQ::Float64, pi_p::Vector{Float64}, pi_q::Vector{Float64}, residual::Vector{Float64}, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # loop over each bus and compute the residual
    for bus in 1:sys.nb
        # active power balance: stt[:pb][:slack][tii][bus] to record with time
        residual[bus] = 
            # slack!!!
            pi_p[bus]*KP + 
            # consumers (positive)
            sum(stt.dev_p[tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt.sh_p[tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt.acline_pfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt.acline_pto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt.xfm_pfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt.xfm_pto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt.dc_pfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt.dc_pto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt.dev_p[tii][idx.pr[bus]]; init=0.0)
    
        # reactive power balance
        residual[sys.nb + bus] = 
            # slack!!!
            pi_q[bus]*KQ + 
            # consumers (positive)
            sum(stt.dev_q[tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt.sh_q[tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt.acline_qfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt.acline_qto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt.xfm_qfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt.xfm_qto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt.dc_qfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt.dc_qto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt.dev_q[tii][idx.pr[bus]]; init=0.0)
    end
end

# correct the reactive power injections into the network
function apply_pq_injections!(idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # warning
    @info "note: the apply_pq_injections! function does NOT take J pqe into account yet"

    # time index
    tii = prm.ts.time_key_ind[tii]

    # at this time, compute the pr and cs upper and lower bounds across all devices
    dev_plb = stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii]
    dev_pub = stt.u_on_dev[tii].*prm.dev.p_ub_tmdv[tii]
    dev_qlb = stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]
    dev_qub = stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]

    # note: clipping is based on the upper/lower bounds, and not
    # based on the beta linking equations -- so, we just treat
    # that as a penalty, and not as a power balance factor
    # 
    # also, compute the dc line upper and lower bounds
    dcfr_plb = -prm.dc.pdc_ub
    dcfr_pub = prm.dc.pdc_ub
    dcto_plb = -prm.dc.pdc_ub
    dcto_pub = prm.dc.pdc_ub
    dcfr_qlb = prm.dc.qdc_fr_lb
    dcfr_qub = prm.dc.qdc_fr_ub
    dcto_qlb = prm.dc.qdc_to_lb
    dcto_qub = prm.dc.qdc_to_ub

    # how does balance work? for reactive power,
    # 0 = qb_slack + (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
    #
    # so, we take want to set:
    # -qb_slack = (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
    p_capacity = zeros(sys.nb)
    q_capacity = zeros(sys.nb)

    for bus in 1:sys.nb
        # active power balance
        pb_slack = 
                # shunt
                sum(stt.sh_p[tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt.acline_pfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt.acline_pto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt.xfm_pfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt.xfm_pto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0)
                # dcline -- not included
                # consumers (positive) -- not included
                # producer (negative) -- not included
        # reactive power balance
        qb_slack = 
                # shunt        
                sum(stt.sh_q[tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt.acline_qfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt.acline_qto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt.xfm_qfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt.xfm_qto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0)
                # dcline -- not included
                # consumers (positive) -- not included
                # producer (negative) -- not included

        # get limits -- P
        pr_Plb   = sum(dev_plb[idx.pr[bus]]; init=0.0)
        cs_Plb   = sum(dev_plb[idx.cs[bus]]; init=0.0)
        pr_Pub   = sum(dev_pub[idx.pr[bus]]; init=0.0) 
        cs_Pub   = sum(dev_pub[idx.cs[bus]]; init=0.0)
        dcfr_Plb = sum(dcfr_plb[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcfr_Pub = sum(dcfr_pub[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcto_Plb = sum(dcto_plb[idx.bus_is_dc_tos[bus]]; init=0.0)
        dcto_Pub = sum(dcto_pub[idx.bus_is_dc_tos[bus]]; init=0.0) 

        # get limits -- Q
        pr_Qlb   = sum(dev_qlb[idx.pr[bus]]; init=0.0)
        cs_Qlb   = sum(dev_qlb[idx.cs[bus]]; init=0.0)
        pr_Qub   = sum(dev_qub[idx.pr[bus]]; init=0.0) 
        cs_Qub   = sum(dev_qub[idx.cs[bus]]; init=0.0)
        dcfr_Qlb = sum(dcfr_qlb[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcfr_Qub = sum(dcfr_qub[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcto_Qlb = sum(dcto_qlb[idx.bus_is_dc_tos[bus]]; init=0.0)
        dcto_Qub = sum(dcto_qub[idx.bus_is_dc_tos[bus]]; init=0.0) 

        # total: lb < -pb_slack < ub
        pub = cs_Pub + dcfr_Pub + dcto_Pub - pr_Plb# + 100.0
        plb = cs_Plb + dcfr_Plb + dcto_Plb - pr_Pub# - 100.0
        
        # total: lb < -qb_slack < ub
        qub = cs_Qub + dcfr_Qub + dcto_Qub - pr_Qlb# + 100.0
        qlb = cs_Qlb + dcfr_Qlb + dcto_Qlb - pr_Qub# - 100.0

        # now, based on these bounds, we compute the amount of "room" left -- P
        if -pb_slack >= pub
            println("ub limit")
            p_capacity[bus] = 0.0
        elseif -pb_slack < plb
            println("lb limit")
            p_capacity[bus] = 0.0
        else # in the middle -- all good
            println("middle :)")
            p_capacity[bus] = min(abs(-pb_slack - plb), abs(pub - -pb_slack))
        end

        # now, apply Q
        if -qb_slack >= qub
            println("ub limit")
            # max everything out
            stt.dev_q[tii][idx.cs[bus]]             = dev_qub[idx.cs[bus]]
            stt.dev_q[tii][idx.pr[bus]]             = dev_qlb[idx.pr[bus]]
            stt.dc_qfr[tii][idx.bus_is_dc_frs[bus]] = dcfr_qub[idx.bus_is_dc_frs[bus]]
            stt.dc_qto[tii][idx.bus_is_dc_tos[bus]] = dcfr_qub[idx.bus_is_dc_tos[bus]]
        elseif -qb_slack < qlb
            println("lb limit")

            # min everything out
            stt.dev_q[tii][idx.cs[bus]]             = dev_qlb[idx.cs[bus]]
            stt.dev_q[tii][idx.pr[bus]]             = dev_qub[idx.pr[bus]]
            stt.dc_qfr[tii][idx.bus_is_dc_frs[bus]] = dcfr_qlb[idx.bus_is_dc_frs[bus]]
            stt.dc_qto[tii][idx.bus_is_dc_tos[bus]] = dcfr_qlb[idx.bus_is_dc_tos[bus]]
        else # in the middle -- all good
            println("middle")
            lb_dist  = -qb_slack - qlb
            bnd_dist = qub - qlb
            scale    = lb_dist/bnd_dist

            stt.dev_q[tii][idx.cs[bus]]             = dev_qlb[idx.cs[bus]]             + scale*(dev_qub[idx.cs[bus]]             - dev_qlb[idx.cs[bus]])
            stt.dev_q[tii][idx.pr[bus]]             = dev_qub[idx.pr[bus]]             - scale*(dev_qub[idx.pr[bus]]             - dev_qlb[idx.pr[bus]])
            stt.dc_qfr[tii][idx.bus_is_dc_frs[bus]] = dcfr_qlb[idx.bus_is_dc_frs[bus]] + scale*(dcfr_qub[idx.bus_is_dc_frs[bus]] - dcfr_qlb[idx.bus_is_dc_frs[bus]])
            stt.dc_qto[tii][idx.bus_is_dc_tos[bus]] = dcfr_qlb[idx.bus_is_dc_tos[bus]] + scale*(dcfr_qub[idx.bus_is_dc_tos[bus]] - dcfr_qlb[idx.bus_is_dc_tos[bus]])
        end

        # now, apply P using KP
        additional_power = KP*pi_p[bus]
        if -pb_slack >= pub
            println("ub limit")
            # max everything out
            stt.p_on[tii][idx.cs[bus]]              = dev_pub[idx.cs[bus]]
            stt.p_on[tii][idx.pr[bus]]              = dev_plb[idx.pr[bus]]
            stt.dc_pfr[tii][idx.bus_is_dc_frs[bus]] = dcfr_pub[idx.bus_is_dc_frs[bus]]
            stt.dc_pto[tii][idx.bus_is_dc_tos[bus]] = dcfr_pub[idx.bus_is_dc_tos[bus]]
        elseif -pb_slack < plb
            println("lb limit")

            # min everything out
            stt.p_on[tii][idx.cs[bus]]              = dev_plb[idx.cs[bus]]
            stt.p_on[tii][idx.pr[bus]]              = dev_pub[idx.pr[bus]]
            stt.dc_pfr[tii][idx.bus_is_dc_frs[bus]] = dcfr_plb[idx.bus_is_dc_frs[bus]]
            stt.dc_pto[tii][idx.bus_is_dc_tos[bus]] = dcfr_plb[idx.bus_is_dc_tos[bus]]
        else # in the middle -- all good
            println("middle")

            lb_dist  = -pb_slack - plb
            bnd_dist = pub - plb
            scale    = lb_dist/bnd_dist
            stt.p_on[tii][idx.cs[bus]]              = dev_plb[idx.cs[bus]]             + scale*(dev_pub[idx.cs[bus]]             - dev_plb[idx.cs[bus]])
            stt.p_on[tii][idx.pr[bus]]              = dev_pub[idx.pr[bus]]             - scale*(dev_pub[idx.pr[bus]]             - dev_plb[idx.pr[bus]])
            stt.dc_pfr[tii][idx.bus_is_dc_frs[bus]] = dcfr_plb[idx.bus_is_dc_frs[bus]] + scale*(dcfr_pub[idx.bus_is_dc_frs[bus]] - dcfr_plb[idx.bus_is_dc_frs[bus]])
            stt.dc_pto[tii][idx.bus_is_dc_tos[bus]] = dcfr_plb[idx.bus_is_dc_tos[bus]] + scale*(dcfr_pub[idx.bus_is_dc_tos[bus]] - dcfr_plb[idx.bus_is_dc_tos[bus]])
        end
    end
end

function transform_acpf_Jac(Jac::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, pi_p::Vector{Float64}, PQidx::Vector{Int64}, sys::quasiGrad.System)
    # variable structure: x = [v_reduced, KP, theta_reduced]
    # un-transformed Jac: [dp/dvm   dp/dva
    #                      dq/dvm   dq/dva]
    #
    # 1. keep q equations associated with PQ buses
    Jac = Jac[[1:sys.nb; sys.nb .+ PQidx], :]

    # 2. keep v variables associated with PQ buses
    Jac = Jac[:,[PQidx; (sys.nb+1):end]]

    # 2. update the row associated with KP
    nPQ = length(PQidx)
    Jac[1:sys.nb,       nPQ+1]  = pi_p
    Jac[(sys.nb+1):end, nPQ+1] .= 0.0

    # four steps:
    #   1. remove column associated with ref bus phase
    #   2. remove column associated with ref bus voltage
    #   3. add column associated with slack active power
    #   4. add column associated with slack reactive power
    #
    # update jac :)
    #Jac[sys.nb+1:end, 1         ]  = pi_q
    #Jac[1:sys.nb,     1         ] .= 0.0
    #Jac[sys.nb+1:end, sys.nb + 1] .= 0.0
    #Jac[1:sys.nb,     sys.nb + 1]  = pi_p
    # variable structure: x = [KQ, v_reduced, KP, theta_reduced]

    # 1. remove voltages from pv+ref buses
    #Jac = Jac[1:sys.nb, sys.nb+1:end]
    # 2. remove phase from from ref buses and add active power distributed slack
    #Jac[:,1] .= 1.0 ./ sys.nb

    # output
    return Jac
end

# compute slack factors
function slack_factors(idx::quasiGrad.Index, prm::quasiGrad.Param, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)

    # warning
    @info "note: the nodal_pq_bounds! function does NOT take J pqe into account yet"

    # time index
    tii = prm.ts.time_key_ind[tii]

    # at this time, compute the pr and cs upper and lower bounds across all devices
    dev_plb = stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii]
    dev_pub = stt.u_on_dev[tii].*prm.dev.p_ub_tmdv[tii]
    dev_qlb = stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]
    dev_qub = stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]

    # note: clipping is based on the upper/lower bounds, and not
    # based on the beta linking equations -- so, we just treat
    # that as a penalty, and not as a power balance factor
    # 
    # also, compute the dc line upper and lower bounds
    dcfr_plb = -prm.dc.pdc_ub
    dcfr_pub = prm.dc.pdc_ub
    dcto_plb = -prm.dc.pdc_ub
    dcto_pub = prm.dc.pdc_ub
    dcfr_qlb = prm.dc.qdc_fr_lb
    dcfr_qub = prm.dc.qdc_fr_ub
    dcto_qlb = prm.dc.qdc_to_lb
    dcto_qub = prm.dc.qdc_to_ub

    # how does balance work? for reactive power,
    # 0 = qb_slack + (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
    #
    # so, we take want to set:
    # -qb_slack = (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
    p_capacity = zeros(sys.nb)
    q_capacity = zeros(sys.nb)
    pub        = zeros(sys.nb)
    plb        = zeros(sys.nb)
    qub        = zeros(sys.nb)
    qlb        = zeros(sys.nb)


    for bus in 1:sys.nb
        # active power balance
        pb_slack = 
                # shunt
                sum(stt.sh_p[tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt.acline_pfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt.acline_pto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt.xfm_pfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt.xfm_pto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0)
                # dcline -- not included
                # consumers (positive) -- not included
                # producer (negative) -- not included
        # reactive power balance
        qb_slack = 
                # shunt        
                sum(stt.sh_q[tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt.acline_qfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt.acline_qto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt.xfm_qfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt.xfm_qto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0)
                # dcline -- not included
                # consumers (positive) -- not included
                # producer (negative) -- not included

        # get limits -- P
        pr_Plb   = sum(dev_plb[idx.pr[bus]]; init=0.0)
        cs_Plb   = sum(dev_plb[idx.cs[bus]]; init=0.0)
        pr_Pub   = sum(dev_pub[idx.pr[bus]]; init=0.0) 
        cs_Pub   = sum(dev_pub[idx.cs[bus]]; init=0.0)
        dcfr_Plb = sum(dcfr_plb[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcfr_Pub = sum(dcfr_pub[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcto_Plb = sum(dcto_plb[idx.bus_is_dc_tos[bus]]; init=0.0)
        dcto_Pub = sum(dcto_pub[idx.bus_is_dc_tos[bus]]; init=0.0) 

        # get limits -- Q
        pr_Qlb   = sum(dev_qlb[idx.pr[bus]]; init=0.0)
        cs_Qlb   = sum(dev_qlb[idx.cs[bus]]; init=0.0)
        pr_Qub   = sum(dev_qub[idx.pr[bus]]; init=0.0) 
        cs_Qub   = sum(dev_qub[idx.cs[bus]]; init=0.0)
        dcfr_Qlb = sum(dcfr_qlb[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcfr_Qub = sum(dcfr_qub[idx.bus_is_dc_frs[bus]]; init=0.0)
        dcto_Qlb = sum(dcto_qlb[idx.bus_is_dc_tos[bus]]; init=0.0)
        dcto_Qub = sum(dcto_qub[idx.bus_is_dc_tos[bus]]; init=0.0) 

        # total: lb < -pb_slack < ub
        pub[bus] = cs_Pub + dcfr_Pub + dcto_Pub - pr_Plb
        plb[bus] = cs_Plb + dcfr_Plb + dcto_Plb - pr_Pub
        
        # total: lb < -qb_slack < ub
        qub[bus] = cs_Qub + dcfr_Qub + dcto_Qub - pr_Qlb
        qlb[bus] = cs_Qlb + dcfr_Qlb + dcto_Qlb - pr_Qub

        # now, based on these bounds, we compute the amount of "room" left -- P
        if -pb_slack >= pub[bus]
            println("ub limit")
            p_capacity[bus] = 0.0
        elseif -pb_slack < plb[bus]
            println("lb limit")
            p_capacity[bus] = 0.0
        else # in the middle -- all good
            println("middle :)")
            p_capacity[bus] = min(abs(-pb_slack - plb[bus]), abs(pub[bus] - -pb_slack))
        end

        # now, based on these bounds, we compute the amount of "room" left -- Q
        if -qb_slack >= qub[bus]
            println("ub limit")
            q_capacity[bus] = 0.0
        elseif -qb_slack < qlb[bus]
            println("lb limit")
            q_capacity[bus] = 0.0
        else # in the middle -- all good
            println("middle :)")
            q_capacity[bus] = min(abs(-qb_slack - qlb[bus]), abs(qub[bus] - -qb_slack))
        end
    end

    # define participation factors!
    pi_p = p_capacity./sum(p_capacity)
    pi_q = q_capacity./sum(q_capacity)

    # if something bad happened, just set them all to 1/n
    if isnan(sum(pi_p))
        pi_p .= 1/sys.nb
    end

    if isnan(sum(pi_q))
        pi_q .= 1/sys.nb
    end

    # define the indices of buses which should be PQ (i.e., buses with almost no Q capacity)
    PQidx = collect(1:sys.nb)[isapprox.(q_capacity, 0.0, atol = 1e-3)]

    # output
    return pi_p, pi_q, PQidx, plb, pub, qlb, qub
end


# solve power flow
function solve_power_flow(grd::quasiGrad.Grad, idx::quasiGrad.Index, KP::Float64, pi_p::Vector{Float64}, prm::quasiGrad.Param, PQidx::Vector{Int64}, qG::quasiGrad.QG, residual::Vector{Float64}, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8, Ybus_real::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, Ybus_imag::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64})
    # initialize
    run_pf = true

    # start with all buses as PV if they have Q capacity
    buses = collect(1:sys.nb)
    # => alternative: PQidx  = buses[isapprox.(qub,qlb,atol=1e-6) || isapprox.(qub,Qinj,atol=1e-6) || isapprox.(qlb,Qinj,atol=1e-6)]
    PVidx  = setdiff(buses, PQidx)
    nPQ    = length(PQidx)
    residual_idx = [buses;           # => P
                   sys.nb .+ PQidx]  # => Q
    # note => ref = 1, but it is a PV bus :)

    # keep running?
    while run_pf == true
        #
        # build the state
        x = [stt.vm[tii][PQidx]; KP; stt.va[tii][2:end]]
        
        # loop over each bus and compute the residual
        quasiGrad.power_flow_residual!(idx, KP, pi_p, residual, stt, sys, tii)

        # test the residual for termination
        if quasiGrad.norm(residual[residual_idx]) < 1e-5
            run_pf = false
        else
            println("residual:")
            println(quasiGrad.norm(residual[residual_idx]))
            sleep(0.75)

            # update the Jacobian
            Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)
            Jac = quasiGrad.transform_acpf_Jac(Jac, pi_p, PQidx, sys)

            # take a Newton step -- do NOT put the step scaler inside the parantheses
            x = x - (Jac\residual[residual_idx])

            # update the state
            stt.vm[tii][PQidx] = x[1:nPQ]
            KP = x[nPQ + 1]
            stt.va[tii][2:end] = x[(nPQ+2):end]

            # update the flows and residual and such
            quasiGrad.update_states_for_distributed_slack_pf!(bit, grd, idx, prm, qG, stt)
        end
    end

    # output
    return KP
end

function power_flow_residual!(idx::quasiGrad.Index, residual::Vector{Float64}, stt::quasiGrad.State, sys::quasiGrad.System, tii::Int8)
    # loop over each bus and compute the residual
    for bus in 1:sys.nb
        # active power balance: stt[:pb][:slack][tii][bus] to record with time
        residual[bus] = 
            # consumers (positive)
            sum(stt.dev_p[tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt.sh_p[tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt.acline_pfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt.acline_pto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt.xfm_pfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt.xfm_pto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt.dc_pfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt.dc_pto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt.dev_p[tii][idx.pr[bus]]; init=0.0)
    
        # reactive power balance
        residual[sys.nb + bus] = 
            # consumers (positive)
            sum(stt.dev_q[tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt.sh_q[tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt.acline_qfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt.acline_qto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt.xfm_qfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt.xfm_qto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt.dc_qfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt.dc_qto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt.dev_q[tii][idx.pr[bus]]; init=0.0)
    end
end

function update_states_for_distributed_slack_pf!(bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    # in this function, we only update the flow, xfm, and shunt states
    #
    # clip voltage
    # println("clipping off")
    quasiGrad.clip_voltage!(prm, stt)

    # compute network flows and injections
    qG.eval_grad = false
    quasiGrad.acline_flows!(grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)
    qG.eval_grad = true
end