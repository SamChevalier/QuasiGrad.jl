function initialize_qG(prm::quasiGrad.Param; Div::Int64=1, hpc_params::Bool=false)
    # In this function, we hardcode a number of important parameters
    # and intructions related to how the qG solver will operate. 
    # It also contains all of the adam solver parameters!
    #
    # printing
    if hpc_params == true
        # turn OFF printing
        print_zms                     = false # print zms at every adam iteration?
        print_freq                    = 5    # (i.e., every how often)
        print_final_stats             = false # print stats at the end?
        print_lbfgs_iterations        = false
        print_projection_success      = false
        print_linear_pf_iterations    = false
        print_reserve_cleanup_success = false
    else
        print_zms                     = true # print zms at every adam iteration?
        print_freq                    = 5    # (i.e., every how often)
        print_final_stats             = true # print stats at the end?
        print_lbfgs_iterations        = true
        print_projection_success      = true
        print_linear_pf_iterations    = true
        print_reserve_cleanup_success = true
    end

    # compute sus on each adam iteration?
    compute_sus_on_each_iteration = false
    compute_sus_frequency         = 5

    # buses
    num_bus = length(prm.bus.id)

    # rounding percentages for the various divisions
    pcts_to_round_D1 = [75; 100.0]
    pcts_to_round_D2 = [25.0; 50.0; 80.0; 95.0; 100.0]
    pcts_to_round_D3 = [25.0; 50.0; 75.0; 90.0; 95.0; 99.0; 100.0]
    
    # which division are we in?
    if Div == 1
        pcts_to_round = pcts_to_round_D1
    elseif Div == 2
        pcts_to_round = pcts_to_round_D2
    elseif Div == 3
        pcts_to_round = pcts_to_round_D3
    else
        @warn "Division $(Div) not recognized. Using D1."
        pcts_to_round = pcts_to_round_D1
    end

    # so: adam 1   =>   round pcts_to_round[1]
    #     adam 2   =>   round pcts_to_round[2]
    #     adam 3   =>   round pcts_to_round[3]
    #             ....
    #     adam n   =>   round pcts_to_round[n]
    #     adam n+1 =>   no more rounding! just an LP solve.

    # adam clips anything larger than grad_max
    grad_max  = 1e11

    # these will be set programmatically
    adam_solve_times = zeros(length(pcts_to_round))

    # write location
    #   => "local" just writes in the same folder as the input
    #   => "GO" writes "solution.json" in the cwd
    write_location = "GO" # "local"

    # penalty gradients are expensive -- only compute the gradient
    # if the constraint is violated by more than this value
    pg_tol = 1e-9

    # amount to penalize constraint violations
        # => **replaced by constraint_grad_weight**
        # => delta                    = prm.vio.p_bus
        # => **replaced by constraint_grad_weight**

    # mainly for testing
    eval_grad = true

    # amount to de-prioritize binary selection over continuous variables
    # note: we don't actually care about binary values, other than the fact
    # that they imply su and sd power values
    binary_projection_weight = 0.1

    # amount to prioritize power selection over other variables
    p_on_projection_weight   = 10.0
    dev_q_projection_weight  = 5.0

    # gurobi feasibility tolerance -- needs to be 1e-8 for GO!
    FeasibilityTol = 9e-9
    IntFeasTol     = 9e-9

    # mip gap for Gurobi
    mip_gap = 1/100.0

    # Gurobi time limit (projection)
    time_lim = 3.0

    # for testing:
    scale_c_pbus_testing     = 1.0
    scale_c_qbus_testing     = 1.0
    scale_c_sflow_testing    = 1.0

    # ctg solver settings
    ctg_grad_cutoff          = -50.0  # don't take gradients of ctg violations that are smaller
                                      # (in magnitude) than this value -- not worth it!
    score_all_ctgs           = false  # this is used for testing/post-processing
    min_buses_for_krylov     = 25     # don't use Krylov if there are this many or fewer buses

    # adaptively the frac of ctgs to keep
    max_ctg_to_keep = min(50, length(prm.ctg.id))
    frac_ctg_keep   = max_ctg_to_keep/length(prm.ctg.id)
    # this is the fraction of ctgs that are scored and differentiated
    # i.e., 100*frac_ctg_keep% of them. half are random, and half a
    # the worst case performers from the previous adam iteration.

    # more finetuned:
        # => if length(prm.ctg.id) < 250
        # =>     # just keep all!
        # =>     frac_ctg_keep = 1.0
        # => elseif length(prm.ctg.id) < 1000
        # =>     frac_ctg_keep = 0.5
        # => elseif length(prm.ctg.id) < 2000
        # =>     frac_ctg_keep = 0.25
        # => elseif length(prm.ctg.id) < 5000
        # =>     frac_ctg_keep = 0.15
        # => else
        # =>     frac_ctg_keep = 0.1
        # => end

    # cg error: set the max allowable error (no error can be larger than this)
    emax = 5e-5*sqrt(num_bus)
    # more generally, emax > sqrt((Ax-b)'(Ax-b)), which grows at sqrt(n) for
    # a constant error value, so we scale given error by sqrt(n) -- seems reasonable
    #
    # ctg_max_to_score score: using sortperm of negative scores, so we score 1:X
    pcg_tol                  = emax
    max_pcg_its              = Int64(round(sqrt(num_bus)))
    grad_ctg_tol             = 1e-4  # only take the gradient of ctg violations larger than this
    base_solver              = "pcg" # "lu", "pcg" for approx
    ctg_solver               = "wmi" # "lu", "pcg", for approx, "wmi" for low rank updates
    build_ctg_full           = false # build the full contingency matrices?
    build_ctg_lowrank        = true  # build the low rank contingency elements?
                                     #  -- you don't need both ^, unless for testing
    # for setting cutoff_level = memory, from LimitedLDLFactorizations.lldl:
                                     # `memory::Int=0`: extra amount of memory to allocate for the incomplete factor `L`.
                                     # The total memory allocated is nnz(T) + n * `memory`, where
                                     # `T` is the strict lower triangle of A and `n` is the size of `A`;
    # therefore, cutoff_level / memory is self-scaling! meaning, it automatically gets multuplied by "n"
    cutoff_level                 = 10     # for preconditioner -- let's be bullish here
    build_basecase_cholesky      = true   # we use this to compute the cholesky decomp of the base case
    accuracy_sparsify_lr_updates = 1.0   # trim all (sorted) values in the lr update vectors which
                                          # contribute (less than) beyond this goal
    # NOTE on "accuracy_sparsify_lr_updates" -- it is only helpful when 
    #      the values of u[bit_vec] are applied to the input (y[bit_vec]),
    #      but I never got this to actually be faster, so it isn't implemented
    #      -- come back to this!

    # initialize adam parameters
    eps        = 1e-8         # for numerical stability -- keep at 1e-8 (?)
    beta1      = 0.75
    beta2      = 0.9

    # the following are NOT used
    alpha_min  = 0.001/10.0   # for cos decay
    alpha_max  = 0.001/0.5    # for cos decay
    Ti         = 100          # for cos decay -- at Tcurr == Ti, cos() => -1
    step_decay = 0.999        # for exp decay

    # choose step sizes
    vmva_scale    = 2.5e-6
    xfm_scale     = 1e-5
    dc_scale      = 1e-5
    power_scale   = 1e-4
    reserve_scale = 1e-4
    bin_scale     = 1e-2 # bullish!!!
    alpha_0 = Dict(:vm     => vmva_scale,
                   :va     => vmva_scale,
                   # xfm
                   :phi    => xfm_scale,
                   :tau    => xfm_scale,
                   # dc
                   :dc_pfr => dc_scale,
                   :dc_qto => dc_scale,
                   :dc_qfr => dc_scale,
                   # powers -- decay this!!
                   :dev_q  => power_scale,
                   :p_on   => power_scale,
                   # reserves
                   :p_rgu     => reserve_scale,
                   :p_rgd     => reserve_scale,
                   :p_scr     => reserve_scale,
                   :p_nsc     => reserve_scale,
                   :p_rrd_on  => reserve_scale,
                   :p_rrd_off => reserve_scale,
                   :p_rru_on  => reserve_scale,
                   :p_rru_off => reserve_scale,
                   :q_qrd     => reserve_scale,
                   :q_qru     => reserve_scale,
                   # bins
                   :u_on_xfm     => bin_scale,
                   :u_on_dev     => bin_scale,
                   :u_step_shunt => bin_scale,
                   :u_on_acline  => bin_scale)

    # specify step size decay approach: "cos", "none", or "exponential"
    decay_type = "none" #"cos"

    # adam plotting
    plot_scale_up = 2.0
    plot_scale_dn = 1e8

    # adam runtime
    adam_max_time = 60.0 # only one is true -- overwritten in GO iterations
    adam_max_its  = 300  # only one is true
    adam_stopper  = "time" # "iterations"

    # gradient modifications ====================================================
    apply_grad_weight_homotopy = true
    # see the homotopy file for more parameters!!!

    # gradient modifications -- power balance
    pqbal_grad_type     = "quadratic_for_lbfgs" #"soft_abs", "standard"
    pqbal_grad_weight_p = prm.vio.p_bus # standard: prm.vio.p_bus
    pqbal_grad_weight_q = prm.vio.q_bus # standard: prm.vio.q_bus
    pqbal_grad_eps2     = 1e-5

    # gradient modification for constraints
    constraint_grad_is_soft_abs = true # "standard"
    constraint_grad_weight = prm.vio.p_bus
    constraint_grad_eps2 = 1e-4

    # gradient modification for ac flow penalties
    acflow_grad_is_soft_abs = true
    acflow_grad_weight = prm.vio.s_flow
    acflow_grad_eps2 = 1e-4

    # gradient modification for ctg flow penalties
    # => @warn "ctg gradient modification not yet implemented!"
    # NOTE -- we ran out of time in E3, and did NOT implement the softabs
    #         on ctg gradients -- this is probably fine -- you can easily
    #         compute it when the max() operator is applied, here:
    #           => bit[:sfr_vio] .= (flw[:sfr_vio] .> qG.grad_ctg_tol) .&& (flw[:sfr_vio] .> flw[:sto_vio])
    #           => bit[:sto_vio] .= (flw[:sto_vio] .> qG.grad_ctg_tol) .&& (flw[:sto_vio] .> flw[:sfr_vio])
    ctg_grad_is_soft_abs = true # "standard"
    ctg_grad_weight = prm.vio.s_flow
    ctg_grad_eps2 = 1e-4

    # gradient modification for reserves (we do NOT perturb reserve gradient weights -- they are small enough!)
    reserve_grad_is_soft_abs = true # "standard"
    reserve_grad_eps2 = 1e-4

    # ===========================================================================

    # shall we compute injections when we build the Jacobian?
    compute_pf_injs_with_Jac   = true
    max_pf_dx                  = 1e-3    # stop when max delta < 5e-4
    max_pf_dx_final_solve      = 5e-5    # final pf solve
    max_linear_pfs             = 2       # stop when number of pfs > max_linear_pfs
    max_linear_pfs_final_solve = 4       # stop when number of pfs > max_linear_pfs_final_solve
    max_linear_pfs_total       = 6       # this includes failures
    Gurobi_pf_obj              = "min_dispatch_distance" # or, "min_dispatch_perturbation"

    # don't use given initializations
    initialize_shunt_to_given_value = false
    initialize_vm_to_given_value    = true

    # power flow solve parameters ==============================
    #
    # strength of quadratic distance regularization
    cdist_psolve = 1e5

    # turn of su/sd updates, which is expensive, during power flow solves
    run_susd_updates = true

    # bias terms of solving bfgs
    include_energy_costs_lbfgs      = false
    include_lbfgs_p0_regularization = false
    initial_pf_lbfgs_step           = 0.1
    lbfgs_map_over_all_time         = false # assume the same set of variables
                                            # are optimized at each time step
    # set the number of lbfgs steps
    if num_bus < 100
        num_lbfgs_steps = 350
    elseif num_bus < 500    
        num_lbfgs_steps = 300   
    elseif num_bus < 1000    
        num_lbfgs_steps = 250
    else
        num_lbfgs_steps = 175
    end

    # when we clip p/q in bounds, should we clip based on binary values?
    # generally, only do this on the LAST adam iteration,
    clip_pq_based_on_bins = true

    # for the_quasiGrad! solver itself
    first_qG_step      = true
    first_qG_step_size = 1e-15

    # skip ctg eval?
    skip_ctg_eval = false

    # adam solving pf
    take_adam_pf_steps = true
    num_adam_pf_step   = 3
    adam_pf_variables  = [:dc_pfr, :dc_qto, :dc_qfr, :vm, :va, :phi, :tau, :u_step_shunt]

    # build the mutable struct
    qG = QG(
        print_projection_success,
        print_reserve_cleanup_success,
        compute_sus_on_each_iteration,
        compute_sus_frequency,
        pcts_to_round,
        cdist_psolve,
        run_susd_updates,
        grad_max,
        adam_solve_times,
        write_location,
        pg_tol,
        eval_grad,
        binary_projection_weight,
        p_on_projection_weight,
        dev_q_projection_weight,
        print_final_stats,
        FeasibilityTol,
        IntFeasTol,
        mip_gap,
        time_lim,
        print_zms,
        print_freq,
        scale_c_pbus_testing,
        scale_c_qbus_testing,
        scale_c_sflow_testing,
        ctg_grad_cutoff,
        score_all_ctgs,
        min_buses_for_krylov,
        frac_ctg_keep,
        pcg_tol,
        max_pcg_its,
        grad_ctg_tol,
        cutoff_level,
        build_basecase_cholesky,      
        accuracy_sparsify_lr_updates,
        base_solver,
        ctg_solver,
        build_ctg_full,
        build_ctg_lowrank,
        eps,
        beta1,
        beta2,
        alpha_0,
        alpha_min,
        alpha_max,
        Ti,
        step_decay,
        decay_type,
        plot_scale_up, 
        plot_scale_dn, 
        adam_max_time, 
        adam_max_its,
        adam_stopper,
        apply_grad_weight_homotopy, 
        pqbal_grad_type,
        pqbal_grad_weight_p,
        pqbal_grad_weight_q,
        pqbal_grad_eps2,
        constraint_grad_is_soft_abs, 
        constraint_grad_weight,
        constraint_grad_eps2,
        acflow_grad_is_soft_abs,
        acflow_grad_weight,
        acflow_grad_eps2,
        ctg_grad_is_soft_abs,
        ctg_grad_weight,
        ctg_grad_eps2,
        reserve_grad_is_soft_abs,
        reserve_grad_eps2,
        compute_pf_injs_with_Jac,
        max_pf_dx,
        max_pf_dx_final_solve,
        max_linear_pfs,
        max_linear_pfs_final_solve,
        max_linear_pfs_total,
        print_linear_pf_iterations,
        Gurobi_pf_obj,
        initialize_shunt_to_given_value,
        initialize_vm_to_given_value,
        include_energy_costs_lbfgs,
        include_lbfgs_p0_regularization,
        print_lbfgs_iterations,
        initial_pf_lbfgs_step,
        lbfgs_map_over_all_time,
        num_lbfgs_steps,
        clip_pq_based_on_bins,
        first_qG_step,
        first_qG_step_size,
        skip_ctg_eval,
        take_adam_pf_steps,
        num_adam_pf_step,  
        adam_pf_variables)
    
    # output
    return qG
end

function base_initialization(jsn::Dict{String, Any}; Div::Int64=1, hpc_params::Bool = false, perturb_states::Bool=false, pert_size::Float64=1.0)
    # perform all initializations from the jsn data
    # 
    # parse the input jsn data
    prm, idx, sys = parse_json(jsn)

    # build the qg structure
    qG = initialize_qG(prm, Div=Div, hpc_params=hpc_params)

    # intialize (empty) states
    bit, cgd, grd, mgd, scr, stt, msc = initialize_states(idx, prm, sys, qG)

    # initialize the states which adam will update -- the rest are fixed
    adm = initialize_adam_states(qG, sys)

    # define the states which adam can/will update, and fix the rest!
    upd = identify_update_states(prm, idx, stt, sys)

    # initialize the contingency network structure and reusable vectors in dicts
    ntk, flw = initialize_ctg(sys, prm, qG, idx)

    # shall we randomly perutb the states?
    if perturb_states == true
        @info "applying perturbation of size $pert_size with random device binaries"
        perturb_states!(stt, prm, sys, pert_size)

        # re-call the update function -- this must always
        # be called after a random perturbation!
        upd = identify_update_states(prm, idx, stt, sys)
    end

    # contingency structure: ctb and ctd
    ctb = [zeros(sys.nb-1) for _ in 1:sys.nT]     # ctb
    ctd = [zeros(sys.nb-1) for _ in 1:sys.nctg]   # ctd

    # build the full set of vectors, but only use the top fraction (for testing ease, mostly)
    wct = [collect(1:sys.nctg) for _ in 1:sys.nT] # wct

    # I don't want to put these into dicts, because I want them
    # to be as fasttt as possible, and we read/write into them a lot!
        # alt: => ctb  = [zeros(sys.nb-1) for _ in 1:sys.nT]
        # alt: => ctd  = [zeros(sys.nb-1) for _ in 1:sys.nctg]
        # alt: => wct  = [collect(1:sys.nctg) for _ in 1:sys.nT]

    # output
    return adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct
end

function initialize_indices(prm::quasiGrad.Param, sys::quasiGrad.System)
    # define the flow indices (used to update a flow vector)
    ac_line_flows    = collect(1:sys.nl)
    ac_xfm_flows     = collect((sys.nl + 1):(sys.nac))
    ac_phi           = collect((sys.nl + 1):(sys.nac))  # indices of ac devices with potential phase shift

    # Next, we define a mapping: from bus ids to indices
    # Note: "Int64" is safe, unless there is an id we are looking for which
    # is not inside of "prm.bus.id)", but this should never be the case
    acline_fr_bus = Int64.(indexin(prm.acline.fr_bus,prm.bus.id))
    acline_to_bus = Int64.(indexin(prm.acline.to_bus,prm.bus.id))
    xfm_fr_bus    = Int64.(indexin(prm.xfm.fr_bus,prm.bus.id))
    xfm_to_bus    = Int64.(indexin(prm.xfm.to_bus,prm.bus.id))
    dc_fr_bus     = Int64.(indexin(prm.dc.fr_bus,prm.bus.id))
    dc_to_bus     = Int64.(indexin(prm.dc.to_bus,prm.bus.id))
    shunt_bus     = Int64.(indexin(prm.shunt.bus,prm.bus.id))

    # create a dictionary which maps buses to the lines/xfms/dc lines,
    # where this bus is a "fr" bus or a "to" bus
    bus_is_acline_frs = Dict(name => Vector{Int64}() for name in 1:sys.nb)
    bus_is_acline_tos = Dict(name => Vector{Int64}() for name in 1:sys.nb)
    bus_is_xfm_frs    = Dict(name => Vector{Int64}() for name in 1:sys.nb)
    bus_is_xfm_tos    = Dict(name => Vector{Int64}() for name in 1:sys.nb)
    bus_is_dc_frs     = Dict(name => Vector{Int64}() for name in 1:sys.nb)
    bus_is_dc_tos     = Dict(name => Vector{Int64}() for name in 1:sys.nb)

    # loop and populate! acline
    for bus = 1:sys.nb
        # acline
        bus_is_acline_frs[bus] = findall(x -> x .== bus, acline_fr_bus)
        bus_is_acline_tos[bus] = findall(x -> x .== bus, acline_to_bus)

        # xfm
        bus_is_xfm_frs[bus] = findall(x -> x .== bus, xfm_fr_bus)
        bus_is_xfm_tos[bus] = findall(x -> x .== bus, xfm_to_bus)

        # dc line
        bus_is_dc_frs[bus] = findall(x -> x .== bus, dc_fr_bus)
        bus_is_dc_tos[bus] = findall(x -> x .== bus, dc_to_bus)
    end

    # split into producers and consumers
    pr_inds = findall(x -> x == "producer", prm.dev.device_type)
    cs_inds = findall(x -> x == "consumer", prm.dev.device_type)

    # pr and cs and shunt device mappings
    bus_to_pr          = Dict(ii => Int64[] for ii in 1:(sys.nb))
    bus_to_cs          = Dict(ii => Int64[] for ii in 1:(sys.nb))
    bus_to_sh          = Dict(ii => Int64[] for ii in 1:(sys.nb))
    bus_to_pr_not_Jpqe = Dict(ii => Int64[] for ii in 1:(sys.nb))
    bus_to_cs_not_Jpqe = Dict(ii => Int64[] for ii in 1:(sys.nb))
    bus_to_pr_and_Jpqe = Dict(ii => Int64[] for ii in 1:(sys.nb))
    bus_to_cs_and_Jpqe = Dict(ii => Int64[] for ii in 1:(sys.nb))

    # we are also going to append the devices associated with a given bus
    # into their corresponding zones -- slow, but necessary
    pr_pzone  = Dict(ii => Int64[] for ii in 1:(sys.nzP))
    cs_pzone  = Dict(ii => Int64[] for ii in 1:(sys.nzP))
    dev_pzone = Dict(ii => Int64[] for ii in 1:(sys.nzP))

    pr_qzone  = Dict(ii => Int64[] for ii in 1:(sys.nzQ))
    cs_qzone  = Dict(ii => Int64[] for ii in 1:(sys.nzQ))
    dev_qzone = Dict(ii => Int64[] for ii in 1:(sys.nzQ))

    # 1:1 mapping, from device number, to its bus
    device_to_bus = zeros(Int64, sys.ndev)

    for bus = 1:sys.nb
        # get the devices tied to this bus
        bus_id             = prm.bus.id[bus]
        dev_on_bus_inds    = findall(x -> x == bus_id, prm.dev.bus)
        sh_dev_on_bus_inds = findall(x -> x == bus_id, prm.shunt.bus)

        # broadcast
        device_to_bus[dev_on_bus_inds] .= bus

        # are the devices consumers or producers?
        pr_devs_on_bus = dev_on_bus_inds[in.(dev_on_bus_inds,Ref(pr_inds))]
        cs_devs_on_bus = dev_on_bus_inds[in.(dev_on_bus_inds,Ref(cs_inds))]

        # update dictionaries
        bus_to_pr[bus] = pr_devs_on_bus
        bus_to_cs[bus] = cs_devs_on_bus
        bus_to_sh[bus] = sh_dev_on_bus_inds

        # filter out the devs in Jpqe
        bus_to_pr_not_Jpqe[bus] = setdiff(pr_devs_on_bus, prm.dev.J_pqe)
        bus_to_cs_not_Jpqe[bus] = setdiff(cs_devs_on_bus, prm.dev.J_pqe)

        # keep just the devs in Jpqe
        bus_to_pr_and_Jpqe[bus] = intersect(pr_devs_on_bus, prm.dev.J_pqe)
        bus_to_cs_and_Jpqe[bus] = intersect(cs_devs_on_bus, prm.dev.J_pqe)

        # first, the active power zones
        for pzone_id in prm.bus.active_rsvid[bus]
            # grab the zone index
            pz_ind = findfirst(x -> x == pzone_id, prm.reserve.id_pzone)

            # push the pr and cs devices into a list
            append!(pr_pzone[pz_ind], pr_devs_on_bus)
            append!(cs_pzone[pz_ind], cs_devs_on_bus)
            append!(dev_pzone[pz_ind], pr_devs_on_bus, cs_devs_on_bus)
        end

        # second, the REactive power zones
        for qzone_id in prm.bus.reactive_rsvid[bus]
            # grab the zone index
            qz_ind = findfirst(x -> x == qzone_id, prm.reserve.id_qzone)

            # push the pr and cs devices into a list
            append!(pr_qzone[qz_ind], pr_devs_on_bus)
            append!(cs_qzone[qz_ind], cs_devs_on_bus)
            append!(dev_qzone[qz_ind], pr_devs_on_bus, cs_devs_on_bus)
        end
    end

    # let's also get the set of pr/cs not in pqe (without equality linking)
    pr_and_Jpqe = intersect(pr_inds, prm.dev.J_pqe)
    cs_and_Jpqe = intersect(cs_inds, prm.dev.J_pqe)
    pr_not_Jpqe = setdiff(pr_inds,   prm.dev.J_pqe)
    cs_not_Jpqe = setdiff(cs_inds,   prm.dev.J_pqe)

    # build the various timing sets (Ts) needed by devices -- this is quite
    # a bit of data to store, but regenerating it each time is way too slow
    Ts_mndn, Ts_mnup, Ts_sdpc, ps_sdpc_set, Ts_supc,
    ps_supc_set, Ts_sus_jft, Ts_sus_jf, Ts_en_max, 
    Ts_en_min, Ts_su_max = build_time_sets(prm, sys)

    # combine
    idx = Idx(
        acline_fr_bus,
        acline_to_bus,
        xfm_fr_bus,
        xfm_to_bus,
        dc_fr_bus,
        dc_to_bus,
        ac_line_flows,        # index of acline flows in a vector of all lines
        ac_xfm_flows,         # index of xfm flows in a vector of all line flows
        ac_phi,               # index of xfm shifts in a vector of all lines
        bus_is_acline_frs,
        bus_is_acline_tos,
        bus_is_xfm_frs,
        bus_is_xfm_tos,
        bus_is_dc_frs,
        bus_is_dc_tos,
        prm.dev.J_pqe,
        prm.dev.J_pqmax,       # NOTE: J_pqmax == J_pqmin :)
        prm.dev.J_pqmax,       # NOTE: J_pqmax == J_pqmin :)
        prm.xfm.J_fpd,
        prm.xfm.J_fwr,
        bus_to_pr,             # maps a bus number to the pr's on that bus
        bus_to_cs,             # maps a bus number to the cs's on that bus
        bus_to_sh,             # maps a bus number to the sh's on that bus
        bus_to_pr_not_Jpqe,    # maps a bus number to the pr's on that bus (not in Jpqe)
        bus_to_cs_not_Jpqe,    # maps a bus number to the cs's on that bus (not in Jpqe)
        bus_to_pr_and_Jpqe,    # maps a bus number to the pr's on that bus (also in Jpqe)
        bus_to_cs_and_Jpqe,    # maps a bus number to the cs's on that bus (also in Jpqe)
        shunt_bus,             # simple list of buses for shunt devices
        pr_inds,               # simple list of producer inds
        cs_inds,               # simple list of consumer inds
        pr_and_Jpqe,           # prs that have equality links on Q
        cs_and_Jpqe,           # css that have equality links on Q
        pr_not_Jpqe,           # prs that DO NOT have equality links on Q
        cs_not_Jpqe,           # css that DO NOT have equality links on Q
        device_to_bus,         # device index => bus index
        pr_pzone,              # maps a pzone number to the list of producers in that zone
        cs_pzone,              # maps a pzone number to the list of consumers in that zone
        dev_pzone,             # maps a pzone number to the list of devices in that zone
        pr_qzone,              # maps a qzone number to the list of producers in that zone
        cs_qzone,              # maps a qzone number to the list of consumers in that zone
        dev_qzone,             # maps a qzone number to the list of devices in that zone
        Ts_mndn,               # time sets!!
        Ts_mnup, 
        Ts_sdpc, 
        ps_sdpc_set, 
        Ts_supc, 
        ps_supc_set, 
        Ts_sus_jft, 
        Ts_sus_jf, 
        Ts_en_max, 
        Ts_en_min, 
        Ts_su_max)
    
        #= idx = Dict(
        :acline_fr_bus     => acline_fr_bus,
        :acline_to_bus     => acline_to_bus,
        :xfm_fr_bus        => xfm_fr_bus,
        :xfm_to_bus        => xfm_to_bus,
        :ac_line_flows     => ac_line_flows,        # index of acline flows in a vector of all lines
        :ac_xfm_flows      => ac_xfm_flows,         # index of xfm flows in a vector of all line flows
        :ac_phi            => ac_phi,               # index of xfm shifts in a vector of all lines
        :bus_is_acline_frs => bus_is_acline_frs,
        :bus_is_acline_tos => bus_is_acline_tos,
        :bus_is_xfm_frs    => bus_is_xfm_frs,
        :bus_is_xfm_tos    => bus_is_xfm_tos,
        :bus_is_dc_frs     => bus_is_dc_frs,
        :bus_is_dc_tos     => bus_is_dc_tos,
        :J_pqe             => prm.dev.J_pqe,
        :J_pqmax           => prm.dev.J_pqmax,  # NOTE: J_pqmax == J_pqmin
        :J_pqmin           => prm.dev.J_pqmax,
        :J_fpd             => prm.xfm.J_fpd,
        :J_fwr             => prm.xfm.J_fwr,
        :pr                => bus_to_pr,             # maps a bus number to the pr's on that bus
        :cs                => bus_to_cs,             # maps a bus number to the cs's on that bus
        :sh                => bus_to_sh,             # maps a bus number to the sh's on that bus
        :shunt_bus         => shunt_bus,             # simple list of buses for shunt devices
        :pr_devs           => pr_inds,               # simple list of producer inds
        :cs_devs           => cs_inds,               # simple list of consumer inds
        :pr_pzone          => pr_pzone,              # maps a pzone number to the list of producers in that zone
        :cs_pzone          => cs_pzone,              # maps a pzone number to the list of consumers in that zone
        :dev_pzone         => dev_pzone,             # maps a pzone number to the list of devices in that zone
        :pr_qzone          => pr_qzone,              # maps a qzone number to the list of producers in that zone
        :cs_qzone          => cs_qzone,              # maps a qzone number to the list of consumers in that zone
        :dev_qzone         => dev_qzone)             # maps a qzone number to the list of devices in that zone
        =#

    # output
    return  idx
end

function initialize_states(idx::quasiGrad.Idx, prm::quasiGrad.Param, sys::quasiGrad.System, qG::quasiGrad.QG)
    # define the time elements
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

    # build stt
    stt = build_state(prm, sys, qG)

    # build grd
    grd = build_grad(prm, sys)

    # build the (fully initialized) cgd: constant gradient structure
    cgd = build_constant_gradient(idx, prm, sys)

    # build scr
    scr = Dict(
        :nzms            => 0.0,   # this is what we explicitly minimize
        :zms             => 0.0,   # this is what we implicitly maximize
        :zbase           => 0.0,
        :zctg_min        => 0.0,
        :zctg_avg        => 0.0,
        :zhat_mxst       => 0.0,
        :zt_original     => 0.0, # zt = zt_original + zt_penalty
        :zt_penalty      => 0.0, # zt = zt_original + zt_penalty
        :z_enmax         => 0.0,
        :z_enmin         => 0.0,
        # many extra things for score plotting -- not super well named
        :zms_penalized   => 0.0,
        :zbase_penalized => 0.0,
        :emnx            => 0.0,
        :zp              => 0.0,
        :zq              => 0.0,
        :acl             => 0.0,
        :xfm             => 0.0,
        :zoud            => 0.0,
        :zone            => 0.0,
        :rsv             => 0.0,
        :enpr            => 0.0,
        :encs            => 0.0,
        :zsus            => 0.0,
        :cnt             => 0.0,
        :ed_obj          => 0.0) # hold the ed solution

    # GRB = Gurobi solutions -- Vector{Float64}(undef,(sys.ndev))
    #
    # not needed anymore -- to save memory, we pass the solution
    # directly to the stt dict!
    #
    # GRB = Dict(
    #     :u_on_dev  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_on      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :dev_q     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_rgu     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_rgd     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_scr     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_nsc     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_rru_on  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_rru_off => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_rrd_on  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :p_rrd_off => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :q_qru     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
    #     :q_qrd     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)))

    # vector of miscellaneous vectors (so I don't have to initialize a new one each time)
    msc = Dict(
        :pinj_ideal => zeros(sys.nb),
        :qinj_ideal => zeros(sys.nb),
        :pb_slack   => zeros(sys.nb),
        :qb_slack   => zeros(sys.nb),
        :pub        => zeros(sys.nb),
        :plb        => zeros(sys.nb),
        :qub        => zeros(sys.nb),
        :qlb        => zeros(sys.nb),
        :pinj0      => zeros(sys.nb), # bias point
        :qinj0      => zeros(sys.nb), # bias point
        :pinj_dc    => zeros(sys.nb),
        :cos_ftp    => zeros(sys.nl),
        :sin_ftp    => zeros(sys.nl),
        :vff        => zeros(sys.nl),
        :vtt        => zeros(sys.nl),
        :vft        => zeros(sys.nl),
        :pfr        => zeros(sys.nl),
        :pto        => zeros(sys.nl),
        :qfr        => zeros(sys.nl),
        :qto        => zeros(sys.nl),
        :acline_pfr => zeros(sys.nl),
        :acline_pto => zeros(sys.nl),
        :acline_qfr => zeros(sys.nl),
        :acline_qto => zeros(sys.nl),
        :acline_sfr => zeros(sys.nl),
        :acline_sto => zeros(sys.nl),
        :acline_sfr_plus => zeros(sys.nl),
        :acline_sto_plus => zeros(sys.nl),
        :cos_ftp_x    => zeros(sys.nx),
        :sin_ftp_x    => zeros(sys.nx),
        :vff_x        => zeros(sys.nx),
        :vtt_x        => zeros(sys.nx),
        :vft_x        => zeros(sys.nx),
        :vt_tau_x     => zeros(sys.nx),
        :vf_tau_x     => zeros(sys.nx),
        :vf_tau2_x    => zeros(sys.nx),
        :vff_tau2_x   => zeros(sys.nx),
        :vft_tau_x    => zeros(sys.nx),
        :vft_tau2_x   => zeros(sys.nx),
        :vff_tau3_x   => zeros(sys.nx),
        :pfr_x        => zeros(sys.nx),
        :pto_x        => zeros(sys.nx),
        :qfr_x        => zeros(sys.nx),
        :qto_x        => zeros(sys.nx),
        :xfm_pfr_x    => zeros(sys.nx),
        :xfm_pto_x    => zeros(sys.nx),
        :xfm_qfr_x    => zeros(sys.nx),
        :xfm_qto_x    => zeros(sys.nx),
        :xfm_sfr_x    => zeros(sys.nx),
        :xfm_sto_x    => zeros(sys.nx),
        :xfm_sfr_plus_x  => zeros(sys.nx),
        :xfm_sto_plus_x  => zeros(sys.nx),
        :acline_scale_fr => zeros(sys.nl),
        :acline_scale_to => zeros(sys.nl),
        :scale_fr_x      => zeros(sys.nx),
        :scale_to_x      => zeros(sys.nx),
        :vm2_sh          => zeros(sys.nsh),
        :g_tv_shunt      => zeros(sys.nsh),
        :b_tv_shunt      => zeros(sys.nsh))

    bit = Dict(
        :acline_sfr_plus => BitArray(zeros(Int, sys.nl)), # indices assocaited with acline_sfr_plus_x > acline_sto_plus_x && 0
        :acline_sto_plus => BitArray(zeros(Int, sys.nl)), # indices assocaited with acline_sto_plus_x > acline_sfr_plus_x && 0
        :xfm_sfr_plus_x  => BitArray(zeros(Int, sys.nx)), # indices assocaited with xfm_sfr_plus_x > xfm_sto_plus_x && 0
        :xfm_sto_plus_x  => BitArray(zeros(Int, sys.nx)), # indices assocaited with xfm_sto_plus_x > xfm_sfr_plus_x && 0
        :sfr_vio        => BitArray(zeros(Int, sys.nac)), # indices assocaited with ctg flows: sfr_vio
        :sto_vio        => BitArray(zeros(Int, sys.nac)), # indices assocaited with ctg flows: sto_vio
        )
        
    # mgd = master grad -- this is the gradient which relates the negative market surplus function 
    # with all "basis" variables -- i.e., the variables for which all others are computed.
    # These are *exactly* (I think..) the variables which are reported in the solution file
    mgd = build_master_grad(prm, sys)

    # output
    return bit, cgd, grd, mgd, scr, stt, msc
end

function identify_update_states(prm::quasiGrad.Param, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # in this function, we will handle five types of fixed variables:
    #   1. must run binaries
    #   2. planned outage binaries and their powers
    #   3. variables which take a pre-defined fixed value of 0
    #   4. phase angle reference
    #   5. line/xfm binaries
        #    -> binaries which have been rounded and fixed (i.e., -- ibr)
        #       via Gurobi will handled later
    #
    # all other, non-fixed states are "variable" states
    # ncs == non-constant states
    tkeys = prm.ts.time_keys
    upd   = Dict(:u_on_dev     => Dict(tkeys[ii] => collect(1:sys.ndev) for ii in 1:(sys.nT)),
                 :p_rrd_off    => Dict(tkeys[ii] => collect(1:sys.ndev) for ii in 1:(sys.nT)),
                 :p_nsc        => Dict(tkeys[ii] => collect(1:sys.ndev) for ii in 1:(sys.nT)),
                 :p_rru_off    => Dict(tkeys[ii] => collect(1:sys.ndev) for ii in 1:(sys.nT)),
                 :q_qru        => Dict(tkeys[ii] => collect(1:sys.ndev) for ii in 1:(sys.nT)),
                 :q_qrd        => Dict(tkeys[ii] => collect(1:sys.ndev) for ii in 1:(sys.nT)),
                 :phi          => Dict(tkeys[ii] => collect(1:sys.nx)   for ii in 1:(sys.nT)),
                 :tau          => Dict(tkeys[ii] => collect(1:sys.nx)   for ii in 1:(sys.nT)),
                 :dc_pto       => Dict(tkeys[ii] => collect(1:sys.nldc) for ii in 1:(sys.nT)),
                 :va           => Dict(tkeys[ii] => collect(1:sys.nb)   for ii in 1:(sys.nT)),
                 :u_on_acline  => Dict(tkeys[ii] => collect(1:sys.nl)   for ii in 1:(sys.nT)),
                 :u_on_xfm     => Dict(tkeys[ii] => collect(1:sys.nx)   for ii in 1:(sys.nT)),
                 :u_step_shunt => Dict(tkeys[ii] => collect(1:sys.nsh)  for ii in 1:(sys.nT)))

    # 1. must run sets =======================================
    for dev in reverse(1:sys.ndev)
        tmr  = get_tmr(dev, prm)
        for tii in tmr
            # set the binaries
            stt[:u_on_dev][tii][dev] = 1.0

            # remove the device from the update list -- 
            # this is safe because we are looping over the
            # devices in reverse order
            deleteat!(upd[:u_on_dev][tii],dev);
        end

    # 2. planned outages =======================================
        tout = get_tout(dev, prm)
        for tii in tout
            # set the binaries
            stt[:u_on_dev][tii][dev] .= 0.0
            # remove the device from the update list -- 
            # this is safe because we are looping over the
            # devices in reverse order -- furthermore, tmr and 
            # tout are mutually exclusive, so it is
            # safe to test the same device for removal from the
            # update list for must run and then outage
            deleteat!(upd[:u_on_dev][tii],dev);
        end
    end

    # 3. pre-defined fixed values =======================================
    #
    # first, we fix the values correctly
    for tii in prm.ts.time_keys
        stt[:p_rrd_off][tii][idx.pr_devs] .= 0.0   # see (106)
        stt[:p_nsc][tii][idx.cs_devs]     .= 0.0   # see (107)
        stt[:p_rru_off][tii][idx.cs_devs] .= 0.0   # see (108)
        stt[:q_qru][tii][idx.pr_and_Jpqe] .= 0.0   # see (117)
        stt[:q_qrd][tii][idx.pr_and_Jpqe] .= 0.0   # see (118)
        stt[:q_qru][tii][idx.cs_and_Jpqe] .= 0.0   # see (127)
        stt[:q_qrd][tii][idx.cs_and_Jpqe] .= 0.0   # see (128)
        stt[:phi][tii][idx.J_fpd]          = prm.xfm.init_phi[idx.J_fpd] # see (144)
        stt[:tau][tii][idx.J_fwr]          = prm.xfm.init_tau[idx.J_fwr] # see (144)
        stt[:dc_pto][tii]                  = -stt[:dc_pfr][tii]

        # remove states from the update list -- this is safe
        deleteat!(upd[:p_rrd_off][tii],idx.pr_devs)
        deleteat!(upd[:p_nsc][tii],idx.cs_devs)
        deleteat!(upd[:p_rru_off][tii],idx.cs_devs)
        deleteat!(upd[:q_qru][tii],idx.J_pqe)
        deleteat!(upd[:q_qrd][tii],idx.J_pqe)
        deleteat!(upd[:phi][tii],idx.J_fpd)
        deleteat!(upd[:tau][tii],idx.J_fwr)
        deleteat!(upd[:dc_pto][tii], collect(1:sys.nldc))
    end

    # 4. phase angle reference =======================================
    #
    # always bus #1 !!!
    for tii in prm.ts.time_keys
        stt[:va][tii][1] = 0
        deleteat!(upd[:va][tii], 1)
    end

    # 5. sadly, don't switch lines this round
    for tii in prm.ts.time_keys
        upd[:u_on_acline][tii] = Int64[]
        upd[:u_on_xfm][tii]    = Int64[]
    end

    # output
    return upd
end

function build_sys(json_data::Dict{String, Any})
    nb    = length(json_data["network"]["bus"])
    nx    = length(json_data["network"]["two_winding_transformer"])
    nsh   = length(json_data["network"]["shunt"])
    nl    = length(json_data["network"]["ac_line"])
    nac   = nl+nx
    nldc  = length(json_data["network"]["dc_line"])
    ndev  = length(json_data["network"]["simple_dispatchable_device"])
    ncs   = sum((ii["device_type"]=="consumer") for ii in json_data["network"]["simple_dispatchable_device"])
    npr   = sum((ii["device_type"]=="producer") for ii in json_data["network"]["simple_dispatchable_device"])
    nT    = json_data["time_series_input"]["general"]["time_periods"]
    nvar  = 2*nb + 2*nx + nx + nl
    nzP   = length(json_data["network"]["active_zonal_reserve"])
    nzQ   = length(json_data["network"]["reactive_zonal_reserve"])
    nctg  = length(json_data["reliability"]["contingency"])
    sys   = System(
                    nb,
                    nx,
                    nsh,
                    nl,
                    nac,
                    nldc,
                    ndev,
                    ncs,
                    npr,
                    nT,
                    nvar,
                    nzP,
                    nzQ,
                    nctg)

    # output
    return sys
end

function join_params(ts_prm::Dict, dc_prm::Dict, ctg_prm::Dict, bus_prm::Dict, xfm_prm::Dict, vio_prm::Dict, shunt_prm::Dict, acline_prm::Dict, device_prm::Dict, reserve_prm::Dict)
    prm = Dict(
        :ts          => ts_prm,
        :dc          => dc_prm,
        :ctg         => ctg_prm,
        :bus         => bus_prm,
        :xfm         => xfm_prm,
        :vio         => vio_prm,
        :dev         => device_prm,
        :shunt       => shunt_prm,
        :acline      => acline_prm,
        :reserve     => reserve_prm)

    # output
    return prm
end

# build everything that will be needed to solve ctgs
function initialize_ctg(sys::quasiGrad.System, prm::quasiGrad.Param, qG::quasiGrad.QG, idx::quasiGrad.Idx)
    # note, the reference bus is always bus #1
    #
    # first, get the ctg limits
    s_max_ctg = [prm.acline.mva_ub_em; prm.xfm.mva_ub_em]

    # get the ordered names of all components
    ac_ids = [prm.acline.id; prm.xfm.id ]

    # get the ordered (negative!!) susceptances
    ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
    
    # build the full incidence matrix: E = lines x buses
    E  = build_incidence(idx, prm, sys)
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

    for ctg_ii in 1:sys.nctg
        # components
        cmpnts = prm.ctg.components[ctg_ii]
        for (cmp_ii,cmp) in enumerate(cmpnts)
            # get the cmp index and b
            cmp_index = findfirst(x -> x == cmp, ac_ids) 
            cmp_b     = -ac_b_params[cmp_index] # negative, because we subtract it out

            # output
            ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
            ctg_params[ctg_ii][cmp_ii]  = cmp_b

            # -> y_diag[cmp_index] = sqrt(cmp_b)
                # we record these in ctg
                # ctg_out_ind[ctg_ii]
        end

        # next, should we build the actual, full ctg matrix?
        if qG.build_ctg_full == true
            # direct construction..
            #
            # NOTE: this is written assuming multiple elements can be
            # simultaneously outaged
            Ybs_k = copy(Ybs)
            Ybs_k[CartesianIndex.(tuple.(ctg_out_ind[ctg_ii],ctg_out_ind[ctg_ii]))] .= 0.0
            Ybr_k[ctg_ii] = Er'*Ybs_k*Er
        end

        # and/or, should we build the low rank ctg elements?
        if qG.build_ctg_lowrank == true
            # .. vs low rank
            #
            # NOTE: this is written assuming only ONE element
            # can be outaged

            # no need to save:
                # v_k[ctg_ii] =  Er[ctg_out_ind[ctg_ii][1],:]
                # b_k[ctg_ii] = -ac_b_params[ctg_out_ind[ctg_ii][1]]
            v_k = Er[ctg_out_ind[ctg_ii][1],:]
            b_k = -ac_b_params[ctg_out_ind[ctg_ii][1]]
            #
            # construction: 
            # 
            # Ybr_k[ctg_ii] = ctg[:Ybr] + v*beta*v'
            #               = ctg[:Ybr] + vLR_k[ctg_ii]*beta*vLR_k[ctg_ii]
            #
            # if v, b saved:
                # u_k[ctg_ii] = Ybr\Array(v_k[ctg_ii])
                # w_k[ctg_ii] = b_k[ctg_ii]*u_k[ctg_ii]/(1+(v_k[ctg_ii]'*u_k[ctg_ii])*b_k[ctg_ii])
            # LU fac => u_k[ctg_ii] = Ybr\Vector(v_k)
                # this is very slow -- we need to us cg and then enforce sparsity!
                # Float64.(Vector(v_k)) is not needed! cg can handle sparse :)
                # quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl=Ybr_ChPr)
            # enforce sparsity -- should be sparse anyways
                # u_k[ctg_ii][abs.(u_k[ctg_ii]) .< 1e-8] .= 0.0

            # we want to sparsify a high-fidelity solution:
            # uk_d = Ybr_Ch\v_k[ctg_ii]
            # quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl=Ybr_ChPr)
            # u_k[ctg_ii] = Ybr\Vector(v_k)
            # u_k[ctg_ii] = C\Vector(v_k)
            if qG.build_basecase_cholesky
                u_k_local = (Ybr_Ch\v_k)[:]
            else
                u_k_local = Ybr\Vector(v_k)
            end
            # sparsify
            abs_u_k           = abs.(u_k_local)
            u_k_ii_SmallToBig = sortperm(abs_u_k)
            bit_vec           = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> (1.0 - qG.accuracy_sparsify_lr_updates)
            # edge case is caught! bit_vec will never be empty. Say, abs_u_k[u_k_ii_SmallToBig] = [0,0,1], then we have
            # bit_vec = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> 0.01%, say => bit_vec = [0,0,1] 
            # 
            # also, we use ".>" because we only want to include all elements that contribute to meeting the stated accuracy goal
            u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]] = u_k_local[u_k_ii_SmallToBig[bit_vec]]
            # this is ok, since u_k and w_k have the same sparsity pattern
            # => for the "w_k" formulation: w_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]] = b_k*u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]]/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]]))*b_k)
            g_k[ctg_ii] = b_k/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii]))*b_k)
        end
    end

    # initialize ctg state
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

    # build the phase angle solution dict -- this will have "sys.nb-1" angles for each solution,
    # since theta1 = 0, and it will have n_ctg+1 solutions, because the base case solution will be
    # optionally saved at the end.. similar for pflow_k
    # theta_k       = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1)) for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
    # pflow_k       = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nac))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
    # theta_k       = Dict(tkeys[ii] => [zeros(sys.nb-1) for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
    # pflow_k       = Dict(tkeys[ii] => [zeros(sys.nac)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
    # this is the gradient solution assuming a base case admittance (it is then rank 1 corrected to dz_dpinj)
    # ctd = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 
    # ctd = Dict(tkeys[ii] => [zeros(sys.nb-1)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))   
    # this is the gradient solution, corrected from ctd
    # dz_dpinj      = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 
    # dz_dpinj      = Dict(tkeys[ii] => [zeros(sys.nb-1)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 

    # "local" storage for apparent power flows (not needed across time)
    # sfr     = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
    # sto     = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
    # sfr_vio = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
    # sto_vio = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)   

    # phase shift derivatives
    #   => consider power injections:  pinj = (p_pr-p_cs-p_sh-p_fr_dc-p_to_dc-alpha*slack) + Er^T*phi*b
    #      => Er^T*phi*b
    # ~ skip the reference bus! -- fr_buses = positive in the incidence matrix; to_buses = negative..
    xfm_at_bus      = Dict(bus => vcat(idx.bus_is_xfm_frs[bus],idx.bus_is_xfm_tos[bus]) for bus in 2:sys.nb)
    xfm_at_bus_sign = Dict(bus => vcat(idx.bus_is_xfm_frs[bus],-idx.bus_is_xfm_tos[bus]) for bus in 2:sys.nb)
    xfm_phi_scalars = Dict(bus => ac_b_params[xfm_at_bus[bus] .+ sys.nl].*sign.(xfm_at_bus_sign[bus]) for bus in 2:sys.nb)

    # compute the constant acline Ybus matrix
    Ybus_acline_real, Ybus_acline_imag = quasiGrad.initialize_acline_Ybus(idx, prm, sys)

    # network parameters
    ntk = Ntk(
            s_max_ctg,     # max contingency flows
            E,             # full incidence matrix
            Er,            # reduced incidence matrix
            ErT,           # Er transposed :)
            Yb,            # full Ybus (DC)    
            Ybr,           # reduced Ybus (DC)
            Yfr,           # reduced flow matrix (DC)
            YfrT,          # Yfr transposed
            ctg_out_ind,   # for each ctg, the list of line indices
            ctg_params,    # for each ctg, the list of (negative) params
            Ybr_k,         # if build_ctg == true, reduced admittance matrix for each ctg 
            ac_b_params,   # base case susceptance parameters
            xfm_at_bus,
            xfm_phi_scalars,
            prm.ctg.alpha,
            prm.ctg.components,
            prm.ctg.id,
            prm.ctg.ctg_inds,
            Ybr_Ch,           # base case Cholesky
            Ybr_ChPr,         # base case preconditioner (everyone uses it!)
            u_k,              # low rank update vector: u_k = Y\v, w_k = b*u/(1+v'*u*b)
            g_k,              # low rank update scalar: g_k = b/(1+v'*u*b)
            Ybus_acline_real, # constant acline Ybus 
            Ybus_acline_imag) # constant acline Ybus 
    
    # v_k,        # low rank update vectors: v*b*v'
    # b_k,        # low rank update scalar: v*b*v'
    # ctg -- not used -- seemed slow
    # ctg = Dict(
    #    :pflow_k         => pflow_k,
    #    :theta_k         => theta_k,
    #    :dz_dpinj        => dz_dpinj,
    #    :ctd   => ctd)
    # ctg
    #flw = Dict(
    #    :sfr             => sfr,     
    #    :sto             => sto,     
    #    :sfr_vio         => sfr_vio, 
    #    :sto_vio         => sto_vio)
    # return ctg, flw, ntk
    flw = Dict(
        :ac_phi          => zeros(sys.nac),
        :ac_qfr          => zeros(sys.nac),
        :ac_qto          => zeros(sys.nac),
        :qfr2            => zeros(sys.nac),
        :qto2            => zeros(sys.nac),
        :bt              => zeros(sys.nac),
        :dsmax_dp_flow   => zeros(sys.nac),
        :dsmax_dqfr_flow => zeros(sys.nac),
        :dsmax_dqto_flow => zeros(sys.nac),
        :pflow_k         => zeros(sys.nac),
        :sfr             => zeros(sys.nac),  
        :sto             => zeros(sys.nac),  
        :sfr_vio         => zeros(sys.nac), 
        :sto_vio         => zeros(sys.nac), 
        :p_inj           => zeros(sys.nb),
        :theta_k         => zeros(sys.nb-1),
        :rhs             => zeros(sys.nb-1),
        :dz_dpinj        => zeros(sys.nb-1),
        :dz_dpinj_all    => zeros(sys.nb-1),
        :c               => zeros(sys.nb-1))

    return ntk, flw
end

function build_incidence(idx::quasiGrad.Idx, prm::quasiGrad.Param, sys::quasiGrad.System)
    # loop over all ac devices and construct incidence matrix
    m = sys.nac
    n = sys.nb

    # acline
    row_acline    = prm.acline.line_inds
    col_acline_fr = idx.acline_fr_bus
    col_acline_to = idx.acline_to_bus
    E_acline_fr   = quasiGrad.sparse(row_acline,col_acline_fr, 1, m, n)
    E_acline_to   = quasiGrad.sparse(row_acline,col_acline_to, -1, m, n)

    # xfm
    row_xfm    = sys.nl .+ prm.xfm.xfm_inds
    col_xfm_fr = idx.xfm_fr_bus
    col_xfm_to = idx.xfm_to_bus
    E_xfm_fr   = quasiGrad.sparse(row_xfm,col_xfm_fr, 1, m, n)
    E_xfm_to   = quasiGrad.sparse(row_xfm,col_xfm_to, -1, m, n)

    # combine the output
    E = E_acline_fr + E_acline_to + E_xfm_fr + E_xfm_to

    # output
    return E
end

function initialize_acline_Ybus(idx::quasiGrad.Idx, prm::quasiGrad.Param, sys::quasiGrad.System)
    # loop over all ac devices and construct incidence matrix
    #
    # note: this assumes all lines are on!
    m = sys.nac
    n = sys.nb

    # full: Ybus = Ybus_acline + Ybus_xfm + Ybus_shunts
    # => Ybus_acline   = spzeros(n,n) # constant!!
    # => Ybus_xfm      = spzeros(n,n) # time varying
    # => Ybus_shunt    = spzeros(n,n) # time varying

    # acline ===========================
    row_acline    = prm.acline.line_inds
    col_acline_fr = idx.acline_fr_bus
    col_acline_to = idx.acline_to_bus
    E_acline_fr   = quasiGrad.sparse(row_acline,col_acline_fr, 1, m, n)
    E_acline_to   = quasiGrad.sparse(row_acline,col_acline_to, -1, m, n)
    E_acline      = E_acline_fr + E_acline_to

    # build diagonal admittance matrices
    Yd_acline_series = quasiGrad.spdiagm(m, m, prm.acline.g_sr + im*prm.acline.b_sr) # zero pads!
    Yd_acline_shunt  = quasiGrad.spzeros(Complex{Float64}, n, n)

    # loop and populate
    for line in 1:sys.nl
        fr = idx.acline_fr_bus[line]
        to = idx.acline_to_bus[line]

        # fr first
        Yd_acline_shunt[fr,fr] += prm.acline.g_fr[line]
        Yd_acline_shunt[fr,fr] += im*prm.acline.b_fr[line]
        Yd_acline_shunt[fr,fr] += im*prm.acline.b_ch[line]/2.0

        # to second
        Yd_acline_shunt[to,to] += prm.acline.g_to[line]
        Yd_acline_shunt[to,to] += im*prm.acline.b_to[line]
        Yd_acline_shunt[to,to] += im*prm.acline.b_ch[line]/2.0
    end

    # output
    Ybus_acline      = E_acline'*Yd_acline_series*E_acline + Yd_acline_shunt
    Ybus_acline_real = real(Ybus_acline)
    Ybus_acline_imag = imag(Ybus_acline)

    # output
    return Ybus_acline_real, Ybus_acline_imag
end

function update_Ybus(idx::quasiGrad.Idx, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol)
    # this function updates the Ybus matrix using the time-varying shunt and xfm values
    #
    # NOTE: this assumes all xfms are on
    n = sys.nb
    Ybus_xfm_imag   = spzeros(n,n)
    Ybus_shunt_imag = spzeros(n,n)
    Ybus_xfm_real   = spzeros(n,n)
    Ybus_shunt_real = spzeros(n,n)
    #
    # xfm ===========================
    for xfm in 1:sys.nx
        # prepare
        cos_phi  = cos(stt[:phi][tii][xfm])
        sin_phi  = sin(stt[:phi][tii][xfm])
        series_y = prm.xfm.g_sr[xfm] + im*prm.xfm.b_sr[xfm]

        # populate!
        yff = (series_y + im*prm.xfm.b_ch[xfm]/2.0 + prm.xfm.g_fr[xfm] + im*prm.xfm.b_fr[xfm])/(stt[:tau][tii][xfm]^2)
        ytt = (series_y + im*prm.xfm.b_ch[xfm]/2.0 + prm.xfm.g_to[xfm] + im*prm.xfm.b_to[xfm])
        yft = (-series_y)/(stt[:tau][tii][xfm]*(cos_phi - im*sin_phi))
        ytf = (-series_y)/(stt[:tau][tii][xfm]*(cos_phi + im*sin_phi))
    
        # populate real!
        Ybus_xfm_real[idx.xfm_fr_bus[xfm], idx.xfm_fr_bus[xfm]] += real(yff)
        Ybus_xfm_real[idx.xfm_to_bus[xfm], idx.xfm_to_bus[xfm]] += real(ytt)
        Ybus_xfm_real[idx.xfm_fr_bus[xfm], idx.xfm_to_bus[xfm]] += real(yft)
        Ybus_xfm_real[idx.xfm_to_bus[xfm], idx.xfm_fr_bus[xfm]] += real(ytf)

        # populate imag
        Ybus_xfm_imag[idx.xfm_fr_bus[xfm], idx.xfm_fr_bus[xfm]] += imag(yff)
        Ybus_xfm_imag[idx.xfm_to_bus[xfm], idx.xfm_to_bus[xfm]] += imag(ytt)
        Ybus_xfm_imag[idx.xfm_fr_bus[xfm], idx.xfm_to_bus[xfm]] += imag(yft)
        Ybus_xfm_imag[idx.xfm_to_bus[xfm], idx.xfm_fr_bus[xfm]] += imag(ytf)
    end

    # shunt ===========================
    for shunt in 1:sys.nsh
        bus = idx.shunt_bus[shunt]
        Ybus_shunt_real[bus,bus] += prm.shunt.gs[shunt]*(stt[:u_step_shunt][tii][shunt])
        Ybus_shunt_imag[bus,bus] += prm.shunt.bs[shunt]*(stt[:u_step_shunt][tii][shunt])
    end

    # construct the output
    Ybus_real = ntk.Ybus_acline_real + Ybus_xfm_real + Ybus_shunt_real
    Ybus_imag = ntk.Ybus_acline_imag + Ybus_xfm_imag + Ybus_shunt_imag

    # output
    return Ybus_real, Ybus_imag
end


function build_DCY(prm::quasiGrad.Param, sys::quasiGrad.System)
    # call the vectors of susceptances and build Yx
    Yx = -quasiGrad.spdiagm([prm.acline.b_sr; prm.xfm.b_sr])

    # build the reduced admittance matrix
    Yb  = prm[:E]'*prm[:Yx]*prm[:E]
    Ybr = prm[:Yb][2:end,2:end]       # use @view ?

    # update based on line status at each time period
    ichol = CholeskyPreconditioner(prm[:Yb_red], 3)

    # notes: propertynames(ichol)      = (:ldlt, :memory)
    #        propertynames(ichol.ldlt) = (:L, :D, :P, :α)

    # get the incomplete cholesky factorization
end

# get the "must run" times
function get_tmr(dev::Int64, prm::quasiGrad.Param)
    # two cases (mutually exclusive) -- test which is applicable
    if prm.dev.init_accu_down_time[dev] > 0
        t_set = prm.ts.time_keys[isapprox.(prm.dev.on_status_lb[dev],1.0)]

    else  # necessarily true -> prm.dev.init_accu_up_time[dev] > 0
        mr_up       = prm.dev.init_accu_up_time[dev] .+ prm.ts.start_time .+ quasiGrad.eps_time .< prm.dev.in_service_time_lb[dev]
        valid_times = isapprox.(prm.dev.on_status_lb[dev],1.0) .|| mr_up
        t_set       = prm.ts.time_keys[valid_times]
    end

    # output
    return t_set 
end

# get the "must run" times
function get_tout(dev::Int64, prm::quasiGrad.Param)
    # two cases (mutually exclusive) -- test which is applicable
    if prm.dev.init_accu_up_time[dev] > 0
        t_set = prm.ts.time_keys[isapprox.(prm.dev.on_status_ub[dev],0.0)]

    else  # necessarily true -> prm.dev.init_accu_down_time[dev] > 0
        out_dwn     = prm.dev.init_accu_down_time[dev] .+ prm.ts.start_time .+ quasiGrad.eps_time .< prm.dev.down_time_lb[dev]
        valid_times = isapprox.(prm.dev.on_status_ub[dev],0.0) .|| out_dwn
        t_set       = prm.ts.time_keys[valid_times]
    end
    
    # output
    return t_set
end

# depricated!! left here for historical purposes :)
function initialize_static_grads!(idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, sys::quasiGrad.System, qG::quasiGrad.QG)
    # there is a subset of gradients whose values are static:
    # set those static gradients here!
    #
    # negative market surplus function: see score_zms!()
    grd[:nzms][:zbase]    = -1.0
    grd[:nzms][:zctg_min] = -1.0
    grd[:nzms][:zctg_avg] = -1.0

    # zbase: see score_zbase!()
    grd[:zbase][:zt]        = 1.0
    grd[:zbase][:z_enmax]   = 1.0
    grd[:zbase][:z_enmin]   = 1.0
    grd[:zbase][:zhat_mxst] = -qG.constraint_grad_weight

    # zt: see score_zt!()
    #
    # consumer revenues and costs
    grd[:zt][:zen_dev][idx.cs_devs] = +ones(sys.ncs)
    grd[:zt][:zen_dev][idx.pr_devs] = -ones(sys.npr)
    # startup costs
    grd[:zt][:zsu_dev]    = -1.0
    grd[:zt][:zsu_acline] = -1.0
    grd[:zt][:zsu_xfm]    = -1.0
    # shutdown costs
    grd[:zt][:zsd_dev]    = -1.0
    grd[:zt][:zsd_acline] = -1.0
    grd[:zt][:zsd_xfm]    = -1.0
    # on-costs
    grd[:zt][:zon_dev]    = -1.0
    # time-dependent su costs
    grd[:zt][:zsus_dev]   = -1.0
    # ac line overload costs
    grd[:zt][:zs_acline]  = -1.0
    grd[:zt][:zs_xfm]     = -1.0
    # local reserve penalties (producers and consumers)
    grd[:zt][:zrgu] = -1.0
    grd[:zt][:zrgd] = -1.0
    grd[:zt][:zscr] = -1.0
    grd[:zt][:znsc] = -1.0
    grd[:zt][:zrru] = -1.0
    grd[:zt][:zrrd] = -1.0
    grd[:zt][:zqru] = -1.0
    grd[:zt][:zqrd] = -1.0
    # power mismatch penalties
    grd[:zt][:zp] = -1.0
    grd[:zt][:zq] = -1.0
    # zonal reserve penalties (P)
    grd[:zt][:zrgu_zonal] = -1.0
    grd[:zt][:zrgd_zonal] = -1.0
    grd[:zt][:zscr_zonal] = -1.0
    grd[:zt][:znsc_zonal] = -1.0
    grd[:zt][:zrru_zonal] = -1.0
    grd[:zt][:zrrd_zonal] = -1.0
    # zonal reserve penalties (Q)
    grd[:zt][:zqru_zonal] = -1.0
    grd[:zt][:zqrd_zonal] = -1.0

    grd[:zt][:zhat_mndn]   = -qG.constraint_grad_weight
    grd[:zt][:zhat_mnup]   = -qG.constraint_grad_weight
    grd[:zt][:zhat_rup]    = -qG.constraint_grad_weight
    grd[:zt][:zhat_rd]     = -qG.constraint_grad_weight
    grd[:zt][:zhat_rgu]    = -qG.constraint_grad_weight
    grd[:zt][:zhat_rgd]    = -qG.constraint_grad_weight
    grd[:zt][:zhat_scr]    = -qG.constraint_grad_weight
    grd[:zt][:zhat_nsc]    = -qG.constraint_grad_weight
    grd[:zt][:zhat_rruon]  = -qG.constraint_grad_weight
    grd[:zt][:zhat_rruoff] = -qG.constraint_grad_weight
    grd[:zt][:zhat_rrdon]  = -qG.constraint_grad_weight
    grd[:zt][:zhat_rrdoff] = -qG.constraint_grad_weight
    # common set of pr and cs constraint variables (see below)
    grd[:zt][:zhat_pmax]      = -qG.constraint_grad_weight
    grd[:zt][:zhat_pmin]      = -qG.constraint_grad_weight
    grd[:zt][:zhat_pmaxoff]   = -qG.constraint_grad_weight
    grd[:zt][:zhat_qmax]      = -qG.constraint_grad_weight
    grd[:zt][:zhat_qmin]      = -qG.constraint_grad_weight
    grd[:zt][:zhat_qmax_beta] = -qG.constraint_grad_weight
    grd[:zt][:zhat_qmin_beta] = -qG.constraint_grad_weight

    # for testing the connection costs
    # prm.acline.connection_cost    .= 1000000.0
    # prm.acline.disconnection_cost .= 1000000.0
    # prm.xfm.connection_cost       .= 1000000.0
    # prm.xfm.disconnection_cost    .= 1000000.0
end

function initialize_adam_states(qG::quasiGrad.QG, sys::quasiGrad.System)
    # build the adm dictionary, which has the same set of
    # entries (keys) as the mgd dictionary
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

    # build the dict
    adm = Dict( :vm            => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT))),
                :va            => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT))),
                :tau           =>  Dict(:m  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT))),
                :phi           => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT))),
                :dc_pfr        => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT))),
                :dc_qfr        => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT))),
                :dc_qto        => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT))),
                :u_on_acline   => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT))),
                :u_on_xfm      => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT))),
                :u_step_shunt  => Dict( :m  => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT))),
                # device variables
                :u_on_dev      => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_on          => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :dev_q         => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rgu         => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rgd         => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_scr         => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_nsc         => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rru_on      => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rrd_on      => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rru_off     => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rrd_off     => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :q_qru         => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :q_qrd         => Dict( :m  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))))
    return adm
end

function initialize_exhastive_adam_states(qG::quasiGrad.QG, sys::quasiGrad.System)
    # build the adm dictionary, which has the same set of
    # entries (keys) as the mgd dictionary -- extra states for adagrad, the_quasiGrad, etc..
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

    # build the dict
    adm = Dict( :vm            => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nb)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT))),
                :va            => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nb)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT))),
                :tau            => Dict(:m         => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nx)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT))),
                :phi           => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nx)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT))),
                :dc_pfr        => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nldc)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT))),
                :dc_qfr        => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nldc)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT))),
                :dc_qto        => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nldc)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nldc)   for ii in 1:(sys.nT))),
                :u_on_acline   => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nl)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT))),
                :u_on_xfm      => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nx)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT))),
                :u_step_shunt  => Dict( :m         => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.nsh)  for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT))),
                # device variables
                :u_on_dev      => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_on          => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :dev_q         => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rgu         => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rgd         => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_scr         => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_nsc         => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rru_on      => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rrd_on      => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rru_off     => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :p_rrd_off     => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :q_qru         => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))),
                :q_qrd         => Dict( :m         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :v         => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :mhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :vhat      => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :qG_step   => Dict(tkeys[ii] => qG.first_qG_step_size*ones(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_stt  => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT)),
                                        :prev_grad => Dict(tkeys[ii] => zeros(sys.ndev)   for ii in 1:(sys.nT))))
    return adm
end

function build_state(prm::quasiGrad.Param, sys::quasiGrad.System, qG::quasiGrad.QG)
    # define the time elements
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

    # stt -- use initial values
    stt = Dict(
        # network -- set all network voltages to there given initial values
        :vm              => Dict(tkeys[ii] => ones(sys.nb)          for ii in 1:(sys.nT)), # this is a flat start
        :va              => Dict(tkeys[ii] => copy(prm.bus.init_va) for ii in 1:(sys.nT)), # this will be overwritten
        # aclines
        :acline_pfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :acline_qfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :acline_pto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :acline_qto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :u_on_acline     => Dict(tkeys[ii] => ones(sys.nl)                    for ii in 1:(sys.nT)),
        :u_su_acline     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :u_sd_acline     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        # xfms
        :phi          => Dict(tkeys[ii] => copy(prm.xfm.init_phi)          for ii in 1:(sys.nT)),
        :tau          => Dict(tkeys[ii] => copy(prm.xfm.init_tau)          for ii in 1:(sys.nT)),
        :xfm_pfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :xfm_qfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :xfm_pto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :xfm_qto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :u_on_xfm     => Dict(tkeys[ii] => ones(sys.nx)                    for ii in 1:(sys.nT)),
        :u_su_xfm     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :u_sd_xfm     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        # all reactive ac flows -- used for ctgs
        :ac_qfr       => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nac)) for ii in 1:(sys.nT)),
        :ac_qto       => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nac)) for ii in 1:(sys.nT)),
        # all line phase shifts -- used for ctgs
        :ac_phi       => Dict(tkeys[ii]  => zeros(sys.nac) for ii in 1:(sys.nT)),
        # dc lines
        :dc_pfr       => Dict(tkeys[ii] => copy(prm.dc.init_pdc_fr)  for ii in 1:(sys.nT)),
        :dc_pto       => Dict(tkeys[ii] => copy(-prm.dc.init_pdc_fr) for ii in 1:(sys.nT)),
        :dc_qfr       => Dict(tkeys[ii] => copy(prm.dc.init_qdc_fr)  for ii in 1:(sys.nT)),
        :dc_qto       => Dict(tkeys[ii] => copy(prm.dc.init_qdc_to)  for ii in 1:(sys.nT)),
        # shunts
        :sh_p         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
        :sh_q         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
        :u_step_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
        # producing and consuming devices
        :u_on_dev   => Dict(tkeys[ii] => ones(sys.ndev)  for ii in 1:(sys.nT)),
        :dev_p      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :dev_q      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # *** placeholder for the Gurobi projection solutions ***
        :u_on_dev_GRB => Dict(tkeys[ii] => ones(sys.ndev)  for ii in 1:(sys.nT)),
        # devices variables
        :u_su_dev  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :u_sd_dev  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :u_sum     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_on      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :p_su      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :p_sd      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        # device powers
        :p_rgu     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rgd     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_scr     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_nsc     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rru_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rru_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rrd_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rrd_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :q_qru     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :q_qrd     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # scoring and penalties
        :zctg       => Dict(tkeys[ii] => zeros(sys.nctg) for ii in 1:(sys.nT)),    
        # revenues -- with devices, lump consumers and producers together
        :zen_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # startup costs
        :zsu_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zsu_acline => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)), 
        :zsu_xfm    => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
        # shutdown costs
        :zsd_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zsd_acline => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),  
        :zsd_xfm    => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
        # on-costs
        :zon_dev    => Dict(tkeys[ii]  => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        # time-dependent su costs
        :zsus_dev   => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        # ac branch overload costs
        :zs_acline  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
        :zs_xfm     => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
        # local reserve penalties (producers)
        :zrgu       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zrgd       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zscr       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :znsc       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zrru       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zrrd       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zqru       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zqrd       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # power mismatch penalties
        :zp          => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)),
        :zq          => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)),
        # zonal reserve penalties (P)
        :zrgu_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zrgd_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zscr_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :znsc_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zrru_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zrrd_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        # zonal reserve penalties (Q)
        :zqru_zonal   => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        :zqrd_zonal   => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        # endogenous zonal state
        :p_rgu_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rgd_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_scr_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_nsc_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        # actual zonal state
        :p_rgu_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rgd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_scr_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_nsc_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :q_qru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        :q_qrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        # penalized constraints
        :zhat_mndn         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_mnup         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rup          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rd           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rgu          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rgd          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_scr          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_nsc          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rruon        => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rruoff       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rrdon        => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rrdoff       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        # more penalized constraints
        :zhat_pmax         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_pmin         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_pmaxoff      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmax         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmin         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmax_beta    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmin_beta    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # for ctg
        :p_inj             => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),
        )

        # shunts -- should we use the supplied initialization? default: no!
        if qG.initialize_shunt_to_given_value
            for tii in tkeys
                stt[:u_step_shunt][tii] = copy(prm.shunt.init_step)
            end
        end

        # vm -- should we use the supplied initialization? default: no!
        if qG.initialize_vm_to_given_value
            for tii in tkeys
                stt[:vm][tii] = copy(prm.bus.init_vm)
            end
        end

    # output
    return stt
end

function build_master_grad(prm::quasiGrad.Param, sys::quasiGrad.System)
    # define the time elements
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

    # mgd = master grad -- this is the gradient which relates the negative market surplus function 
    # with all "basis" variables -- i.e., the variables for which all others are computed.
    # These are *exactly* (I think..) the variables which are reported in the solution file
    mgd = Dict(:vm            => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),       
               :va            => Dict(tkeys[ii] => zeros(sys.nb)   for ii in 1:(sys.nT)),           
               :tau           => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),            
               :phi           => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)), 
               :dc_pfr        => Dict(tkeys[ii] => zeros(sys.nldc) for ii in 1:(sys.nT)), 
               :dc_qfr        => Dict(tkeys[ii] => zeros(sys.nldc) for ii in 1:(sys.nT)), 
               :dc_qto        => Dict(tkeys[ii] => zeros(sys.nldc) for ii in 1:(sys.nT)), 
               :u_on_acline   => Dict(tkeys[ii] => zeros(sys.nl)   for ii in 1:(sys.nT)),  
               :u_on_xfm      => Dict(tkeys[ii] => zeros(sys.nx)   for ii in 1:(sys.nT)),  
               :u_step_shunt  => Dict(tkeys[ii] => zeros(sys.nsh)  for ii in 1:(sys.nT)),
               # device variables
               :u_on_dev      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_on          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :dev_q         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_rgu         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_rgd         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_scr         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_nsc         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_rru_on      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_rrd_on      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_rru_off     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :p_rrd_off     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :q_qru         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
               :q_qrd         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)))
    
    # output
    return mgd
end

function build_grad(prm::quasiGrad.Param, sys::quasiGrad.System)
    # define the time elements
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

    # grd = grad -- these are primarily non-constant gradients (although,
    #               a few of them are constant, but oh well).
    grd = Dict(
        # aclines
        :acline_pfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_qfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_pto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_qto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :zs_acline => Dict(:acline_pfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                           :acline_qfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                           :acline_pto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                           :acline_qto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        # xfms
        :xfm_pfr =>  Dict(  :vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_qfr =>  Dict(  :vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_pto =>  Dict(  :vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_qto =>  Dict(  :vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :zs_xfm => Dict(:xfm_pfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                        :xfm_qfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                        :xfm_pto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                        :xfm_qto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        # shunts
        :sh_p => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                      :g_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
        :sh_q => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                      :b_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
        :zp         => Dict(:pb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
        :zq         => Dict(:qb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
        # energy costs
        :zen_dev    => Dict(:dev_p               => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        # zon_dev, zsu_dev, zsd_dev (also for lines and xfms)
        # ...
        # these next two dictionaries take the derivatives of the su/sd states with respect to
        # both the current on status, and the previous on status
        :u_su_dev => Dict(:u_on_dev            => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                          :u_on_dev_prev       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :u_sd_dev => Dict(:u_on_dev            => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                          :u_on_dev_prev       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :u_su_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :u_sd_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :u_su_xfm => Dict(:u_on_xfm            => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                          :u_on_xfm_prev       => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :u_sd_xfm => Dict(:u_on_xfm            => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                          :u_on_xfm_prev       => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        # these two following elements are unique -- they serve to collect all of the 
        # coefficients applied to the same partial derivatives (e.g.,
        # a1*dxdp, a2*dxdp, a3*dxdp => dxdp[tii][dev] = a1+a2+a3)
        :dx => Dict(:dp => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                    :dq => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))))
    
    # output
    return grd
end

# state reset -- broken :)
function state_reset(stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    # set everything to 0
    for (k0,v0) in stt
        if typeof(v0) == Float64
            stt[k0] = 0.0
        elseif typeof(v0) == Vector{Float64}
            stt[k0] = zeros(length(v0))
        else
            for (k1,v1) in v0
                if typeof(v1) == Float64
                    stt[k0][k1] = 0.0
                elseif typeof(v1) == Vector{Float64}
                    stt[k0][k1] = zeros(length(v1))
                else
                    for (k2,v2) in v1
                        if typeof(v2) == Float64
                            stt[k0][k1][k2] = 0.0
                        elseif typeof(v2) == Vector{Float64}
                            stt[k0][k1][k2] = zeros(length(v2))
                        end
                    end
                end
            end
        end
    end
end

# perturb, clip, and fix states
function perturb_states!(stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, sys::quasiGrad.System, pert_size::Float64)
    # perturb all states
    for tii in prm.ts.time_keys
        stt[:u_on_acline][tii]  = ones(sys.nl)
        stt[:u_on_xfm][tii]     = ones(sys.nx)
        stt[:u_on_dev][tii]     = Int64.(rand(Bool, sys.ndev))
        stt[:vm][tii]           = pert_size*0.1*randn(sys.nb) .+ 1.0
        stt[:va][tii]           = pert_size*0.1*randn(sys.nb)
        stt[:tau][tii]          = pert_size*0.1*randn(sys.nx) .+ 1.0
        stt[:phi][tii]          = pert_size*0.1*randn(sys.nx)        
        stt[:dc_pfr][tii]       = pert_size*0.1*rand(sys.nldc)    
        stt[:dc_qfr][tii]       = pert_size*0.1*rand(sys.nldc)       
        stt[:dc_qto][tii]       = pert_size*0.1*rand(sys.nldc)  
        stt[:u_step_shunt][tii] = pert_size*4.0*rand(sys.nsh) # bigger spread
        stt[:p_on][tii]         = pert_size*rand(sys.ndev)    
        stt[:dev_q][tii]        = pert_size*rand(sys.ndev)  
        stt[:p_rgu][tii]        = pert_size*rand(sys.ndev) 
        stt[:p_rgd][tii]        = pert_size*rand(sys.ndev)   
        stt[:p_scr][tii]        = pert_size*rand(sys.ndev)      
        stt[:p_nsc][tii]        = pert_size*rand(sys.ndev)    
        stt[:p_rru_on][tii]     = pert_size*rand(sys.ndev)      
        stt[:p_rrd_on][tii]     = pert_size*rand(sys.ndev)      
        stt[:p_rru_off][tii]    = pert_size*rand(sys.ndev)  
        stt[:p_rrd_off][tii]    = pert_size*rand(sys.ndev)     
        stt[:q_qru][tii]        = pert_size*rand(sys.ndev)    
        stt[:q_qrd][tii]        = pert_size*rand(sys.ndev) 
    end
end

function build_constant_gradient(idx::quasiGrad.Idx, prm::quasiGrad.Param, sys::quasiGrad.System)
    # build a structure which hold constant gradient information
    #
    tkeys = prm.ts.time_keys

    # contingency costs
    ctg_avg = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.s_flow/sys.nctg for ii in 1:(sys.nT))
    ctg_min = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.s_flow          for ii in 1:(sys.nT))

    # device on costs
    dzon_dev_du_on_dev = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.dev.on_cost for ii in 1:(sys.nT))

    # device energy costs
    dzt_dzen              = zeros(sys.ndev)
    dzt_dzen[idx.cs_devs] = +ones(sys.ncs)
    dzt_dzen[idx.pr_devs] = -ones(sys.npr)

    # device reserve gradients
    dzrgu_dp_rgu     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_reg_res_up_cost,ii)            for ii in 1:(sys.nT))
    dzrgd_dp_rgd     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_reg_res_down_cost,ii)          for ii in 1:(sys.nT))
    dzscr_dp_scr     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_syn_res_cost,ii)               for ii in 1:(sys.nT))
    dznsc_dp_nsc     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_nsyn_res_cost,ii)              for ii in 1:(sys.nT))
    dzrru_dp_rru_on  = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_up_online_cost,ii)    for ii in 1:(sys.nT))
    dzrru_dp_rru_off = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_up_offline_cost,ii)   for ii in 1:(sys.nT))
    dzrrd_dp_rrd_on  = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_down_online_cost,ii)  for ii in 1:(sys.nT))
    dzrrd_dp_rrd_off = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_down_offline_cost,ii) for ii in 1:(sys.nT))
    dzqru_dq_qru     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.q_res_up_cost,ii)                for ii in 1:(sys.nT))
    dzqrd_dq_qrd     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.q_res_down_cost,ii)              for ii in 1:(sys.nT))
    
    # zonal gradients
    dzrgu_zonal_dp_rgu_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rgu_zonal for ii in 1:(sys.nT))
    dzrgd_zonal_dp_rgd_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rgd_zonal for ii in 1:(sys.nT))
    dzscr_zonal_dp_scr_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.scr_zonal for ii in 1:(sys.nT))
    dznsc_zonal_dp_nsc_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.nsc_zonal for ii in 1:(sys.nT))
    dzrru_zonal_dp_rru_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rru_zonal for ii in 1:(sys.nT))
    dzrrd_zonal_dp_rrd_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rrd_zonal for ii in 1:(sys.nT))
    dzqru_zonal_dq_qru_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.qru_zonal for ii in 1:(sys.nT))
    dzqrd_zonal_dq_qrd_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.qrd_zonal for ii in 1:(sys.nT))
    
    # build the cgd
    cgd = Cgd(
        ctg_avg,
        ctg_min,
        dzon_dev_du_on_dev,
        dzt_dzen,
        dzrgu_dp_rgu,    
        dzrgd_dp_rgd,    
        dzscr_dp_scr,    
        dznsc_dp_nsc,    
        dzrru_dp_rru_on, 
        dzrru_dp_rru_off,
        dzrrd_dp_rrd_on, 
        dzrrd_dp_rrd_off,
        dzqru_dq_qru,    
        dzqrd_dq_qrd,    
        dzrgu_zonal_dp_rgu_zonal_penalty,
        dzrgd_zonal_dp_rgd_zonal_penalty,
        dzscr_zonal_dp_scr_zonal_penalty,
        dznsc_zonal_dp_nsc_zonal_penalty,
        dzrru_zonal_dp_rru_zonal_penalty,
        dzrrd_zonal_dp_rrd_zonal_penalty,
        dzqru_zonal_dq_qru_zonal_penalty,
        dzqrd_zonal_dq_qrd_zonal_penalty)

    # output
    return cgd
end

# manage time
function manage_time!(time_left::Float64, qG::quasiGrad.QG)
    # how long do the adam iterations have?
    # adam will run length(qG.pcts_to_round) + 1 times
    num_adam_solve   = length(qG.pcts_to_round) + 1
    adam_solve_times = 10 .^(range(1,stop=0.75,length=num_adam_solve))

    # scale to account for 50% of the time -- the rest is for Gurobi
    # and for printing the solution..
    # => we want: alpha*sum(adam_solve_times) = 0.5*time_left
    alpha            = 0.45*time_left/sum(adam_solve_times)
    adam_solve_times = alpha*adam_solve_times

    # update qG
    qG.adam_solve_times = adam_solve_times
end

function build_time_sets(prm::quasiGrad.Param, sys::quasiGrad.System)
    # initialize: dev => time => set of time indices
    # note: we use Int(sys.nT/2) because we don't want to initialize
    #       the structure with more or less memory than needed.. super heuristic.
    Ts_mndn     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    Ts_mnup     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    Ts_sdpc     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    ps_sdpc_set = [[Vector{Float64}(undef, Int64(round(sys.nT/2)))                                                for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    Ts_supc     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    ps_supc_set = [[Vector{Float64}(undef, Int64(round(sys.nT/2)))                                                for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    Ts_sus_jft  = [[[Vector{Symbol}(undef, Int64(round(sys.nT/2))) for i in 1:prm.dev.num_sus[dev]]               for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    Ts_sus_jf   = [[[Vector{Symbol}(undef, Int64(round(sys.nT/2))) for i in 1:prm.dev.num_sus[dev]]               for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
    Ts_en_max   = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))  for i in 1:length(prm.dev.energy_req_ub[dev])]                               for dev in 1:sys.ndev]
    Ts_en_min   = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))  for i in 1:length(prm.dev.energy_req_lb[dev])]                               for dev in 1:sys.ndev]
    Ts_su_max   = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))  for i in 1:length(prm.dev.startups_ub[dev])]                                 for dev in 1:sys.ndev]

    # loop over devices
    for dev in 1:sys.ndev

        # loop over time
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # up and down times
            Ts_mndn[dev][t_ind] = get_tmindn(tii, dev, prm)
            Ts_mnup[dev][t_ind] = get_tminup(tii, dev, prm)

            # startup/down power curves
            Ts_sdpc[dev][t_ind], ps_sdpc_set[dev][t_ind] = get_sdpc(tii, dev, prm)
            Ts_supc[dev][t_ind], ps_supc_set[dev][t_ind] = get_supc(tii, dev, prm)

            # loop over sus (i.e., f in F)
            for ii in 1:prm.dev.num_sus[dev]
                Ts_sus_jft[dev][t_ind][ii], Ts_sus_jf[dev][t_ind][ii] = get_tsus_sets(tii, dev, prm, ii)
            end
        end

        # => Wub = prm.dev.energy_req_ub[dev]
        # => Wlb = prm.dev.energy_req_lb[dev]
        # max energy
        for (w_ind, w_params) in enumerate(prm.dev.energy_req_ub[dev])
            Ts_en_max[dev][w_ind] = get_tenmax(w_params, prm)
        end

        # min energy
        for (w_ind, w_params) in enumerate(prm.dev.energy_req_lb[dev])
            Ts_en_min[dev][w_ind] = get_tenmin(w_params, prm)
        end

        # max start ups
        for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
            Ts_su_max[dev][w_ind] = get_tsumax(w_params, prm)
        end
    end

    # output
    return Ts_mndn, Ts_mnup, Ts_sdpc, ps_sdpc_set, Ts_supc, ps_supc_set, 
           Ts_sus_jft, Ts_sus_jf, Ts_en_max, Ts_en_min, Ts_su_max
end

function initialize_lbfgs(mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # first, how many historical gradients do we keep? 
    # 2 < n < 21 according to Wright

    @info "this is general lbfgs solver -- it does not work well"

    num_lbfgs_to_keep = 8

    # define the mapping indices which put gradient and state
    # information into aggregated forms -- to be populated!
    tkeys = prm.ts.time_keys

    lbfgs_map = Dict(
        :vm            => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),       
        :va            => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),           
        :tau           => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),            
        :phi           => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :dc_pfr        => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :dc_qfr        => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :dc_qto        => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :u_on_acline   => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),  
        :u_on_xfm      => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),  
        :u_step_shunt  => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),
        :u_on_dev      => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_on          => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :dev_q         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_rgu         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_rgd         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_scr         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_nsc         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_rru_on      => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_rrd_on      => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_rru_off     => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :p_rrd_off     => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :q_qru         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
        :q_qrd         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)))

    # next, how many lbfgs states are there? let's base this on "upd"
    n_lbfgs = 0
    for var_key in keys(mgd)
        for tii in prm.ts.time_keys
            if var_key in keys(upd)
                n = length(upd[var_key][tii]) # number
                if n == 0
                    lbfgs_map[var_key][tii] = Int64[]
                else
                    lbfgs_map[var_key][tii] = collect(1:n) .+ n_lbfgs
                end

                # update the total number
                n_lbfgs += n
            else
                n = length(mgd[var_key][tii])
                if n == 0
                    lbfgs_map[var_key][tii] = Int64[]
                else
                    lbfgs_map[var_key][tii] = collect(1:n) .+ n_lbfgs
                end

                # update the total number
                n_lbfgs += n
            end
        end
    end

    # build the lbfgs dict
    lbfgs = Dict(:x_now      => zeros(n_lbfgs),
                 :x_new      => zeros(n_lbfgs),
                 :x_prev     => zeros(n_lbfgs),
                 :gradf_now  => zeros(n_lbfgs),
                 :gradf_prev => zeros(n_lbfgs),
                 :alpha      => zeros(num_lbfgs_to_keep),
                 :rho        => zeros(num_lbfgs_to_keep))

    # build the lbfgs dict
    lbfgs_diff = Dict(:s => [zeros(n_lbfgs) for _ in 1:num_lbfgs_to_keep],
                      :y => [zeros(n_lbfgs) for _ in 1:num_lbfgs_to_keep])

    # step size
    lbfgs_step = Dict(:nzms_prev   => 0.0,
                      :beta1_decay => 1.0,
                      :beta2_decay => 1.0,
                      :m           => 0.0,   
                      :v           => 0.0,   
                      :mhat        => 0.0,
                      :vhat        => 0.0,
                      :step        => 1.0,
                      :alpha_0     => 1.0)
    # => lbfgs_step = 1.0

    # indices to track where previous differential vectors are stored --
    # lbfgs_idx[1] is always the most recent data, and lbfgs_idx[end] is the oldest
    lbfgs_idx = Int64.(zeros(num_lbfgs_to_keep))

    return lbfgs, lbfgs_diff, lbfgs_idx, lbfgs_map, lbfgs_step
end

function initialize_pf_lbfgs(mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # first, how many historical gradients do we keep? 
    # 2 < n < 21 according to Wright
    num_lbfgs_to_keep = 10

    # define the mapping indices which put gradient and state
    # information into aggregated forms -- to be populated!
    tkeys = prm.ts.time_keys

    # shall we define the mapping based on time?
    if qG.lbfgs_map_over_all_time == true
        # in this case, include all time indices -- not generally needed
        @warn "you need to now update the type of pf_lbfgs_map"
        pf_lbfgs_map = Dict(
            :vm            => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),       
            :va            => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),           
            :tau           => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),            
            :phi           => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
            :dc_pfr        => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
            :dc_qfr        => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
            :dc_qto        => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),  
            :u_step_shunt  => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)),
            :p_on          => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)), 
            :dev_q         => Dict(tkeys[ii] => Int64[] for ii in 1:(sys.nT)))

        # next, how many pf_lbfgs states are there at each time? let's base this on "upd"
        n_lbfgs = zeros(Int64, sys.nT)
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                if var_key in keys(upd)
                    n = length(upd[var_key][tii]) # number
                    if n == 0
                        pf_lbfgs_map[var_key][tii] = Int64[]
                    else
                        # this is fine, because we will grab "1:n" of the update variables,
                        # i.e., all of them, later on
                        pf_lbfgs_map[var_key][tii] = collect(1:n) .+ n_lbfgs[t_ind]
                    end

                    # update the total number
                    n_lbfgs[t_ind] += n
                else
                    n = length(mgd[var_key][tii])
                    if n == 0
                        pf_lbfgs_map[var_key][tii] = Int64[]
                    else
                        pf_lbfgs_map[var_key][tii] = collect(1:n) .+ n_lbfgs[t_ind]
                    end

                    # update the total number
                    n_lbfgs[t_ind] += n
                end
            end
        end
    else
        # in this case, include all time indices -- not generally needed
        pf_lbfgs_map = Dict(
            :vm            => Int64[],       
            :va            => Int64[],           
            :tau           => Int64[],            
            :phi           => Int64[], 
            :dc_pfr        => Int64[], 
            :dc_qfr        => Int64[], 
            :dc_qto        => Int64[],  
            :u_step_shunt  => Int64[],
            :p_on          => Int64[], 
            :dev_q         => Int64[])

        # next, how many pf_lbfgs states are there at each time? let's base this on "upd"
        n_lbfgs = 0
        tii     = :t1
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            if var_key in keys(upd)
                n = length(upd[var_key][tii]) # number
                if n == 0
                    pf_lbfgs_map[var_key] = Int64[]
                else
                    pf_lbfgs_map[var_key] = collect(1:n) .+ n_lbfgs
                end

                # update the total number
                n_lbfgs += n
            else
                n = length(mgd[var_key][tii])
                if n == 0
                    pf_lbfgs_map[var_key] = Int64[]
                else
                    pf_lbfgs_map[var_key] = collect(1:n) .+ n_lbfgs
                end

                # update the total number
                n_lbfgs += n
            end
        end
    end

    # build the pf_lbfgs dict
    pf_lbfgs = Dict(:x_now      => Dict(tkeys[ii] => zeros(n_lbfgs)           for ii in 1:(sys.nT)),
                    :x_new      => Dict(tkeys[ii] => zeros(n_lbfgs)           for ii in 1:(sys.nT)),
                    :x_prev     => Dict(tkeys[ii] => zeros(n_lbfgs)           for ii in 1:(sys.nT)),
                    :gradf_now  => Dict(tkeys[ii] => zeros(n_lbfgs)           for ii in 1:(sys.nT)),
                    :gradf_prev => Dict(tkeys[ii] => zeros(n_lbfgs)           for ii in 1:(sys.nT)),
                    :q          => Dict(tkeys[ii] => zeros(n_lbfgs)           for ii in 1:(sys.nT)),
                    :r          => Dict(tkeys[ii] => zeros(n_lbfgs)           for ii in 1:(sys.nT)),
                    :alpha      => Dict(tkeys[ii] => zeros(num_lbfgs_to_keep) for ii in 1:(sys.nT)),
                    :rho        => Dict(tkeys[ii] => zeros(num_lbfgs_to_keep) for ii in 1:(sys.nT)))

    # build the pf_lbfgs difference dict
    pf_lbfgs_diff = Dict(:s => Dict(tkeys[ii] => [zeros(n_lbfgs) for _ in 1:num_lbfgs_to_keep] for ii in 1:(sys.nT)),
                         :y => Dict(tkeys[ii] => [zeros(n_lbfgs) for _ in 1:num_lbfgs_to_keep] for ii in 1:(sys.nT)))

    # step size control -- for adam!
    pf_lbfgs_step = Dict(:zpf_prev    => Dict(tkeys[ii] => 0.0    for ii in 1:(sys.nT)),
                         :beta1_decay => Dict(tkeys[ii] => 1.0    for ii in 1:(sys.nT)),
                         :beta2_decay => Dict(tkeys[ii] => 1.0    for ii in 1:(sys.nT)),
                         :m           => Dict(tkeys[ii] => 0.0    for ii in 1:(sys.nT)),   
                         :v           => Dict(tkeys[ii] => 0.0    for ii in 1:(sys.nT)),   
                         :mhat        => Dict(tkeys[ii] => 0.0    for ii in 1:(sys.nT)),
                         :vhat        => Dict(tkeys[ii] => 0.0    for ii in 1:(sys.nT)),
                         :step        => Dict(tkeys[ii] => 0.0    for ii in 1:(sys.nT)),
                         :alpha_0     => Dict(tkeys[ii] => 0.001  for ii in 1:(sys.nT)))

    # indices to track where previous differential vectors are stored --
    # lbfgs_idx[1] is always the most recent data, and lbfgs_idx[end] is the oldest
    pf_lbfgs_idx = Int64.(zeros(num_lbfgs_to_keep))

    # create a scoring dict
    zpf = Dict(
        :zp  => Dict(tkeys[ii] => 0.0 for ii in 1:(sys.nT)),
        :zq  => Dict(tkeys[ii] => 0.0 for ii in 1:(sys.nT)))
    
    # create the dict for regularizing the solution
    dpf0 = Dict(
        :p_on => Dict(tkeys[ii] => copy(stt[:p_on][tkeys[ii]]) for ii in 1:(sys.nT)))

    return dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf
end

# print struct fields
function struct_fields(input_struct)
    println(fieldnames(typeof(input_struct)))
end