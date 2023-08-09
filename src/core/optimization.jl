# adam solver -- take steps for every element in master_grad (only two states are tracked: m and v)
function adam!(adm::quasiGrad.Adam, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}}; standard_adam = true)
    # note: for "adam_pf", just set standard_adam = false
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for var_key in adm.keys
        # so, only progress if, either, this is standard adam (not in the pf stepper),
        # or the variable is a power flow variable!
        if (var_key in qG.adam_pf_variables) || standard_adam
            # call states and gradients ("getproperty" and "call_adam_states" slower)
            adam_states = getfield(adm, var_key)     
            state       = getfield(stt, var_key)
            grad        = getfield(mgd, var_key)

            # loop over all time
            for tii in prm.ts.time_keys
                if var_key in keys(upd)
                    if !isempty(upd[var_key][tii])
                        @inbounds @fastmath @simd for updates in upd[var_key][tii]
                            adam_states.m[tii][updates] = qG.beta1*adam_states.m[tii][updates] + qG.one_min_beta1*grad[tii][updates]
                            adam_states.v[tii][updates] = qG.beta2*adam_states.v[tii][updates] + qG.one_min_beta2*(grad[tii][updates]^2.0)
                            state[tii][updates]         = state[tii][updates]  - qG.alpha_tnow[var_key]*(adam_states.m[tii][updates]/qG.one_min_beta1_decay)/(sqrt(adam_states.v[tii][updates]/qG.one_min_beta2_decay) + qG.eps)
                        end
                    end

                    # vectorized version => 
                        # note -- it isn't clear how best to use @view -- it seems to be helpful when calling
                        # an array subset when adding/subtracting, but now when taking products, etc.
                        # 
                        # update adam moments
                            # => clipped_grad, if helpful! = clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max)
                            # => update_subset = upd[var_key][tii]
                            # => adam_states.m[tii][update_subset] .= qG.beta1.*(@view adam_states.m[tii][update_subset]) .+ qG.one_min_beta1.*(@view grad[tii][update_subset])
                            # => adam_states.v[tii][update_subset] .= qG.beta2.*(@view adam_states.v[tii][update_subset]) .+ qG.one_min_beta2.*((@view grad[tii][update_subset]).^2.0)
                            # => state[tii][update_subset]         .= (@view state[tii][update_subset]) .- qG.alpha_tnow[var_key].*(adam_states.m[tii][update_subset]./qG.one_min_beta1_decay)./(sqrt.(adam_states.v[tii][update_subset]./qG.one_min_beta2_decay) .+ qG.eps)
                else 
                    if !isempty(adam_states.m[tii])
                        # update adam moments
                            # => clipped_grad, if helpful!  = clamp.(mgd[var_key][tii], -qG.grad_max, qG.grad_max)
                        @turbo adam_states.m[tii] .= qG.beta1.*adam_states.m[tii] .+ qG.one_min_beta1.*grad[tii]
                        @turbo adam_states.v[tii] .= qG.beta2.*adam_states.v[tii] .+ qG.one_min_beta2.*quasiGrad.LoopVectorization.pow_fast.(grad[tii],2.0)
                        @turbo state[tii]         .= state[tii] .- qG.alpha_tnow[var_key].*(adam_states.m[tii]./qG.one_min_beta1_decay)./(quasiGrad.LoopVectorization.sqrt_fast.(adam_states.v[tii]./qG.one_min_beta2_decay) .+ qG.eps)
                    end
                end
            end
        end
    end
end

# adam solver -- take steps for every element in master_grad (only two states are tracked: m and v)
function simple_adam!(adm::quasiGrad.Adam, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}}; standard_adam = true)
    # note: for "adam_pf", just set standard_adam = false
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for var_key in adm.keys
        # so, only progress if, either, this is standard adam (not in the pf stepper),
        # or the variable is a power flow variable!
        if (var_key in qG.adam_pf_variables) || standard_adam
            # call states and gradients ("getproperty" and "call_adam_states" slower)
            adam_states = getfield(adm, var_key)     
            state       = getfield(stt, var_key)
            grad        = getfield(mgd, var_key)

            # loop over all time
            for tii in prm.ts.time_keys
                if var_key in keys(upd)
                    if !isempty(upd[var_key][tii])
                        @inbounds @fastmath @simd for updates in upd[var_key][tii]
                            adam_states.m[tii][updates] = qG.beta1*adam_states.m[tii][updates] + qG.one_min_beta1*grad[tii][updates]
                            adam_states.v[tii][updates] = qG.beta2*adam_states.v[tii][updates] + qG.one_min_beta2*(grad[tii][updates]^2.0)
                            state[tii][updates]         = state[tii][updates]  - qG.alpha_tnow[var_key]*(adam_states.m[tii][updates]/qG.one_min_beta1_decay)/(sqrt(adam_states.v[tii][updates]/qG.one_min_beta2_decay) + qG.eps)
                        end
                    end
                else 
                    if !isempty(adam_states.m[tii])
                        @turbo adam_states.m[tii] .= qG.beta1.*adam_states.m[tii] .+ qG.one_min_beta1.*grad[tii]
                        adam_states.v[tii] .= qG.beta2.*adam_states.v[tii] .+ qG.one_min_beta2.*(grad[tii].^2.0)
                        state[tii]         .= state[tii] .- qG.alpha_tnow[var_key].*(adam_states.m[tii]./qG.one_min_beta1_decay)./(sqrt.(adam_states.v[tii]./qG.one_min_beta2_decay) .+ qG.eps)
                    end
                end
            end
        end
    end
end

#=
# adam -- direction preserving!
function adam_dp!(adm::quasiGrad.Adam, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}}; standard_adam = true)
    #
    # note: for "adam_pf", just set standard_adam = false
    #
    # loop over the keys in mgd
    for var_key in adm.keys
        # so, only progress if, either, this is standard adam (not in the pf stepper),
        # or the variable is a power flow variable!
        if (var_key in qG.adam_pf_variables) || standard_adam
            # call states and gradients
            adam_states = getfield(adm, var_key)     
            state       = getfield(stt, var_key)
            grad        = getfield(mgd, var_key)
            for tii in prm.ts.time_keys
                if var_key in keys(upd)
                    if !isempty(upd[var_key][tii])

                        for updates in upd[var_key][tii]
                            @inbounds adam_states.m[tii][updates] = beta1*adam_states.m[tii][updates] + (1.0-beta1)*grad[tii][updates]
                        end
                        
                        # get the largest
                        lrgst = maximum(abs.(adam_states.m[tii][upd[var_key][tii]]))
                        if lrgst < 1e-6
                            lrgst = 1.0
                        end

                        for updates in upd[var_key][tii]
                            @inbounds adam_states.v[tii][updates] = beta2*adam_states.v[tii][updates] + (1.0-beta2)*(grad[tii][updates]^2.0)
                            @inbounds state[tii][updates]         = state[tii][updates]  - (adam_states.m[tii][updates]/lrgst)*qG.alpha_tnow[var_key]*(adam_states.m[tii][updates]/(1.0-beta1_decay))/(sqrt(adam_states.v[tii][updates]/(1.0-beta2_decay)) + qG.eps)
                        end
                    end
                else 
                    if !isempty(adam_states.m[tii])
                        @inbounds adam_states.m[tii] .= beta1.*adam_states.m[tii] .+ (1.0-beta1).*grad[tii]

                        # get the largest
                        lrgst = maximum(abs.(adam_states.m[tii]))
                        if lrgst < 1e-6
                            lrgst = 1.0
                        end

                        @inbounds adam_states.v[tii] .= beta2.*adam_states.v[tii] .+ (1.0-beta2).*(grad[tii].^2.0)
                        @inbounds state[tii]         .= state[tii] .- (abs.(adam_states.m[tii])./lrgst).*qG.alpha_tnow[var_key].*(adam_states.m[tii]./(1.0-beta1_decay))./(sqrt.(adam_states.v[tii]./(1.0-beta2_decay)) .+ qG.eps)
                    end
                end
            end
        end
    end
end

# adam -- direction preserving!
function adam_dp_sqrt!(adm::quasiGrad.Adam, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}}; standard_adam = true)
    #
    # note: for "adam_pf", just set standard_adam = false
    #
    # loop over the keys in mgd
    for var_key in adm.keys
        # so, only progress if, either, this is standard adam (not in the pf stepper),
        # or the variable is a power flow variable!
        if (var_key in qG.adam_pf_variables) || standard_adam
            # call states and gradients
            nn = 10.0

            adam_states = getfield(adm, var_key)     
            state       = getfield(stt, var_key)
            grad        = getfield(mgd, var_key)
            for tii in prm.ts.time_keys
                if var_key in keys(upd)
                    if !isempty(upd[var_key][tii])

                        for updates in upd[var_key][tii]
                            @inbounds adam_states.m[tii][updates] = beta1*adam_states.m[tii][updates] + (1.0-beta1)*grad[tii][updates]
                        end

                        max_val = 1.0
                        if maximum((abs.(adam_states.m[tii][upd[var_key][tii]])).^(1/nn)) > 1e-6
                            max_val = maximum((abs.(adam_states.m[tii][upd[var_key][tii]])).^(1/nn))
                        end

                        for updates in upd[var_key][tii]
                            @inbounds adam_states.v[tii][updates] = beta2*adam_states.v[tii][updates] + (1.0-beta2)*(grad[tii][updates]^2.0)
                            @inbounds state[tii][updates]         = state[tii][updates]  - (((abs.(adam_states.m[tii][updates])).^(1/nn))/max_val)*qG.alpha_tnow[var_key]*(adam_states.m[tii][updates]/(1.0-beta1_decay))/(sqrt(adam_states.v[tii][updates]/(1.0-beta2_decay)) + qG.eps)
                            #@inbounds state[tii][updates]         = state[tii][updates]  - qG.alpha_tnow[var_key]*(adam_states.m[tii][updates]/(1.0-beta1_decay))/(sqrt(adam_states.v[tii][updates]/(1.0-beta2_decay)) + qG.eps)
                        end
                    end
                else 
                    if !isempty(adam_states.m[tii])
                        @inbounds adam_states.m[tii] .= beta1.*adam_states.m[tii] .+ (1.0-beta1).*grad[tii]

                        # get the largest
                        scale = (abs.(adam_states.m[tii])).^(1/nn)
                        if maximum((abs.(adam_states.m[tii])).^(1/nn)) < 1e-6
                            scale = scale
                        else
                            scale = scale./maximum((abs.(adam_states.m[tii])).^(1/nn))
                        end

                        @inbounds adam_states.v[tii] .= beta2.*adam_states.v[tii] .+ (1.0-beta2).*(grad[tii].^2.0)
                        #@inbounds state[tii]         .= state[tii] .- qG.alpha_tnow[var_key].*(adam_states.m[tii]./(1.0-beta1_decay))./(sqrt.(adam_states.v[tii]./(1.0-beta2_decay)) .+ qG.eps)
                        @inbounds state[tii]         .= state[tii] .- scale.*qG.alpha_tnow[var_key].*(adam_states.m[tii]./(1.0-beta1_decay))./(sqrt.(adam_states.v[tii]./(1.0-beta2_decay)) .+ qG.eps)
                    end
                end
            end
        end
    end
end
=#
# adam solver -- take steps for every element in the master_grad list
function flush_adam!(adm::quasiGrad.Adam, prm::quasiGrad.Param, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # loop over the keys in mgd
    for var_key in adm.keys
        adam_states = getfield(adm,var_key)

        # loop over all time
        for tii in prm.ts.time_keys
            # states to update                 
            if var_key in keys(upd)
                # flush the adam moments
                adam_states.m[tii][upd[var_key][tii]] .= 0.0
                adam_states.v[tii][upd[var_key][tii]] .= 0.0
            else
                # flush the adam moments
                adam_states.m[tii] .= 0.0
                adam_states.v[tii] .= 0.0
            end
        end
    end
end

function run_adam!(
        adm::quasiGrad.Adam,
        cgd::quasiGrad.ConstantGrad,
        ctg::quasiGrad.Contingency,
        flw::quasiGrad.Flow,
        grd::quasiGrad.Grad, 
        idx::quasiGrad.Index,
        mgd::quasiGrad.MasterGrad,
        ntk::quasiGrad.Network,
        prm::quasiGrad.Param,
        qG::quasiGrad.QG, 
        scr::Dict{Symbol, Float64},
        stt::quasiGrad.State, 
        sys::quasiGrad.System,
        upd::Dict{Symbol, Vector{Vector{Int64}}})

    # re-initialize
    qG.adm_step      = 0
    qG.beta1_decay   = 1.0
    qG.beta2_decay   = 1.0
    qG.one_min_beta1 = 1.0 - qG.beta1 # here for testing, in case beta1 is changed
    qG.one_min_beta2 = 1.0 - qG.beta2 # here for testing, in case beta1 is changed
    run_adam         = true

    @info "Running adam for $(qG.adam_max_time) seconds!"
    
    # flush adam at each restart ?
    quasiGrad.flush_adam!(adm, prm, upd)

    # start the timer!
    adam_start = time()

    # loop over adam steps
    while run_adam
        # increment
        adm_step += 1

        # step decay
        if qG.decay_adam_step == true
            quasiGrad.adam_step_decay!(qG, time(), adam_start, adam_start+qG.adam_max_time)
        end

        # decay beta and pre-compute
        qG.beta1_decay         = qG.beta1_decay*qG.beta1
        qG.beta2_decay         = qG.beta2_decay*qG.beta2
        qG.one_min_beta1_decay = (1.0-qG.beta1_decay)
        qG.one_min_beta2_decay = (1.0-qG.beta2_decay)

        # update weight parameters?
        if qG.apply_grad_weight_homotopy == true
            quasiGrad.update_penalties!(prm, qG, time(), adam_start, adam_start+qG.adam_max_time)
        end

        # compute all states and grads
        quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

        # take an adam step
        quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
        GC.safepoint()
        # experiments!
            # => quasiGrad.adaGrad!(adm, alpha, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)
            # => quasiGrad.the_quasiGrad!(adm, mgd, prm, qG, stt, upd)
            # => quasiGrad.adam_with_ls!(adm, alpha, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd, cgd, ctb, ctd, flw, grd, idx, ntk, scr, sys, wct)

        # take intermediate pf steps?
        if qG.take_adam_pf_steps == true
            for _ in 1:qG.num_adam_pf_step
                # update the power injection-associated gradients
                quasiGrad.update_states_and_grads_for_adam_pf!(grd, idx, mgd, prm, qG, stt, sys)

                # take an adam pf step (standard_adam=false)
                quasiGrad.adam!(adm, mgd, prm, qG, stt, upd, standard_adam = false)
            end
        end

        # stop?
        run_adam = quasiGrad.adam_termination(adam_start, qG, run_adam)
    end

    # one last clip + state computation -- no grad needed!
    qG.eval_grad = false
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    qG.eval_grad = true
end

function update_states_and_grads!(
    cgd::quasiGrad.ConstantGrad, 
    ctg::quasiGrad.Contingency,
    flw::quasiGrad.Flow, 
    grd::quasiGrad.Grad, 
    idx::quasiGrad.Index, 
    mgd::quasiGrad.MasterGrad,
    ntk::quasiGrad.Network, 
    prm::quasiGrad.Param, 
    qG::quasiGrad.QG, 
    scr::Dict{Symbol, Float64}, 
    stt::quasiGrad.State,
    sys::quasiGrad.System)

    # safepoint
    GC.safepoint()
    
    # if we are here, we want to make sure we are running su/sd updates
    qG.clip_pq_based_on_bins = false
    qG.run_susd_updates = true

    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    # clip all basic states (i.e., the states which are iterated on)
    quasiGrad.clip_all!(prm, qG, stt)

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
        # => println("Skipping ctg evaluation!")
    else
        if qG.ctg_adam_counter == qG.ctg_solve_frequency
            quasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
            qG.ctg_adam_counter = 0
        end
    end
    # score the market surplus function
    quasiGrad.score_zt!(idx, prm, qG, scr, stt) 
    quasiGrad.score_zbase!(qG, scr)
    quasiGrad.score_zms!(scr)

    # print the market surplus function value
        # quasiGrad.print_zms(qG, scr)

    # compute the master grad
    quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
end

function update_states_and_grads_for_adam_pf!(grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # update the non-device states which affect power balance
    #
    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    # clip all basic states (i.e., the states which are iterated on)
    quasiGrad.clip_for_adam_pf!(prm, qG, stt)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, prm, qG, stt)

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

    # compute the master grad
    quasiGrad.master_grad_adam_pf!(grd, idx, mgd, prm, sys)
end

function batch_fix!(pct_round::Float64, prm::quasiGrad.Param, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # loop and concatenate
    bin_vec_del = Vector{Float64}(undef,(sys.nT*sys.ndev))

    for tii in prm.ts.time_keys
        bin_inds               = (1:sys.ndev) .+ (tii-1)*sys.ndev
        bin_vec_del[bin_inds] .= abs.(stt.u_on_dev[tii] .- stt.u_on_dev_GRB[tii])
    end

    # sort and find the binaries that are closest to Gurobi's solution

    # which ones do we fix?
    num_bin_fix = Int64(round(sys.nT*sys.ndev*pct_round/100.0))
    bins_to_fix = sortperm(bin_vec_del)[1:num_bin_fix]
    # FYI!!! => most_sim_to_least_sim = sortperm(abs.(bin_vec_del))

    # now, we loop over time and check for each binary in "bins_to_fix"
    for tii in prm.ts.time_keys
        bin_inds          = (1:sys.ndev) .+ (tii-1)*sys.ndev
        local_bins_to_fix = findall(bin_inds .∈ [bins_to_fix])

        # now, we have that:
            # i)  upd[:u_on_dev][tii] are binaries that should move, and
            # ii) local_bins_to_fix are binaries that should not move
        # there will be overlap, but there local_bins_to_fix might not contain
        # everything that upd contains -- thus, we want their union!
        upd[:u_on_dev][tii] = setdiff(upd[:u_on_dev][tii], local_bins_to_fix)

        # now, for "bin_inds" which are to be fixed, delete them
        # no!! => deleteat!(upd[:u_on_dev][tii],local_bins_to_fix)
    end
end

# test solution soln_status
function solution_status(model::quasiGrad.Model)
    # to get all potential statuses, call: typeof(termination_status(model))
    # 
    #   OPTIMIZE_NOT_CALLED = 0
    #   OPTIMAL = 1
    #   INFEASIBLE = 2
    #   DUAL_INFEASIBLE = 3
    #   LOCALLY_SOLVED = 4
    #   LOCALLY_INFEASIBLE = 5
    #   INFEASIBLE_OR_UNBOUNDED = 6
    #   ALMOST_OPTIMAL = 7
    #   ALMOST_INFEASIBLE = 8
    #   ALMOST_DUAL_INFEASIBLE = 9
    #   ALMOST_LOCALLY_SOLVED = 10
    #   ITERATION_LIMIT = 11
    #   TIME_LIMIT = 12
    #   NODE_LIMIT = 13
    #   SOLUTION_LIMIT = 14
    #   MEMORY_LIMIT = 15
    #   OBJECTIVE_LIMIT = 16
    #   NORM_LIMIT = 17
    #   OTHER_LIMIT = 18
    #   SLOW_PROGRESS = 19
    #   NUMERICAL_ERROR = 20
    #   INVALID_MODEL = 21
    #   INVALID_OPTION = 22
    #   INTERRUPTED = 23
    #   OTHER_ERROR = 24
    soln_status = Int(termination_status(model))
    if soln_status in [1, 4, 7] # optimal, locally solved, or almost optimal
        soln_valid = true
    else
        if Int(primal_status(model)) == 1
            @warn "Projection solution not optimal, but it is feasible."
            soln_valid = true
        else
            soln_valid = false
        end
    end

    # output
    return soln_valid
end

function adam_termination(adam_start::Float64, qG::quasiGrad.QG, run_adam::Bool)
    # stopping criteria
    if qG.adam_stopper == "time"
        if time() - adam_start >= qG.adam_max_time
            run_adam = false
        end
    elseif qG.adam_stopper == "iterations"
        if qG.adm_step >= qG.adam_max_its
            run_adam = false
        end
    else
        # uh-oh -- no stopper!
    end

    # output
    return run_adam
end

# call states
function call_adam_states(adm::quasiGrad.Adam, mgd::quasiGrad.MasterGrad, stt::quasiGrad.State, var_key::Symbol)
    # we need this function because calling struct field programatically
    # isn't straightforward, and getfield() tends to allocate
    if var_key == :vm
        adam_states = adm.vm   
        state       = stt.vm
        grad        = mgd.vm
    elseif var_key == :va
        adam_states = adm.va
        state       = stt.va
        grad        = mgd.va
    elseif var_key == :tau
        adam_states = adm.tau
        state       = stt.tau
        grad        = mgd.tau
    elseif var_key == :phi
        adam_states = adm.phi
        state       = stt.phi
        grad        = mgd.phi
    elseif var_key == :dc_pfr
        adam_states = adm.dc_pfr
        state       = stt.dc_pfr
        grad        = mgd.dc_pfr
    elseif var_key == :dc_qfr
        adam_states = adm.dc_qfr
        state       = stt.dc_qfr
        grad        = mgd.dc_qfr
    elseif var_key == :dc_qto
        adam_states = adm.dc_qto
        state       = stt.dc_qto
        grad        = mgd.dc_qto
    elseif var_key == :u_on_acline
        adam_states = adm.u_on_acline
        state       = stt.u_on_acline
        grad        = mgd.u_on_acline
    elseif var_key == :u_on_xfm
        adam_states = adm.u_on_xfm
        state       = stt.u_on_xfm
        grad        = mgd.u_on_xfm
    elseif var_key == :u_step_shunt
        adam_states = adm.u_step_shunt
        state       = stt.u_step_shunt
        grad        = mgd.u_step_shunt
    elseif var_key == :u_on_dev
        adam_states = adm.u_on_dev
        state       = stt.u_on_dev
        grad        = mgd.u_on_dev
    elseif var_key == :p_on
        adam_states = adm.p_on
        state       = stt.p_on
        grad        = mgd.p_on
    elseif var_key == :dev_q
        adam_states = adm.dev_q
        state       = stt.dev_q
        grad        = mgd.dev_q
    elseif var_key == :p_rgu
        adam_states = adm.p_rgu
        state       = stt.p_rgu
        grad        = mgd.p_rgu
    elseif var_key == :p_rgd
        adam_states = adm.p_rgd
        state       = stt.p_rgd
        grad        = mgd.p_rgd
    elseif var_key == :p_scr
        adam_states = adm.p_scr
        state       = stt.p_scr
        grad        = mgd.p_scr
    elseif var_key == :p_nsc
        adam_states = adm.p_nsc
        state       = stt.p_nsc
        grad        = mgd.p_nsc
    elseif var_key == :p_rru_on
        adam_states = adm.p_rru_on
        state       = stt.p_rru_on
        grad        = mgd.p_rru_on
    elseif var_key == :p_rrd_on
        adam_states = adm.p_rrd_on
        state       = stt.p_rrd_on
        grad        = mgd.p_rrd_on
    elseif var_key == :p_rru_off
        adam_states = adm.p_rru_off
        state       = stt.p_rru_off
        grad        = mgd.p_rru_off
    elseif var_key == :p_rrd_off
        adam_states = adm.p_rrd_off
        state       = stt.p_rrd_off
        grad        = mgd.p_rrd_off
    elseif var_key == :q_qru
        adam_states = adm.q_qru
        state       = stt.q_qru
        grad        = mgd.q_qru
    elseif var_key == :q_qrd
        adam_states = adm.q_qrd
        state       = stt.q_qrd
        grad        = mgd.q_qrd
    else
        println("Field not recognized.")
    end

    # output
    return adam_states, grad, state
end