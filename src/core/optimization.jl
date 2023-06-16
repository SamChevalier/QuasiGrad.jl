# adam solver -- take steps for every element in the master_grad list
#
# only two states are tracked here (m and v)
function adam!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}}; standard_adam = true)
    #
    # note: for "adam_pf", just set standard_adam = false
    #
    # loop over the keys in mgd
    for var_key in keys(mgd)
        # so, only progress if, either, this is standard adam (not in the pf stepper),
        # or the variable is a power flow variable!
        if (var_key in qG.adam_pf_variables) || standard_adam
            # loop over all time
            for tii in prm.ts.time_keys
                # states to update                                            
                if var_key in keys(upd)
                    # => (var_key in keys(upd)) ? (update = upd[var_key][tii]) : (update = Colon())
                    #    the above caused weird type instability, so we just copy and paste
                    update_subset = upd[var_key][tii]

                    # note -- it isn't clear how best to use @view -- it seems to be helpful when calling
                    # an array subset when adding/subtracting, but now when taking products, etc.

                    # update adam moments
                        # => clipped_grad                 = clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max)
                    adm[var_key][:m][tii][update_subset] .= beta1.*(@view adm[var_key][:m][tii][update_subset]) .+ (1.0-beta1).*clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max)
                    adm[var_key][:v][tii][update_subset] .= beta2.*(@view adm[var_key][:v][tii][update_subset]) .+ (1.0-beta2).*clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max).^2.0
                    stt[var_key][tii][update_subset]     .= (@view stt[var_key][tii][update_subset]) .- qG.alpha_0[var_key].*(adm[var_key][:m][tii][update_subset]./(1.0-beta1_decay))./(sqrt.(adm[var_key][:v][tii][update_subset]./(1.0-beta2_decay)) .+ qG.eps)
                    
                else 
                    # update adam moments
                        # => clipped_grad  = clamp.(mgd[var_key][tii], -qG.grad_max, qG.grad_max)
                    adm[var_key][:m][tii] .= beta1.*adm[var_key][:m][tii] .+ (1.0-beta1).*clamp.(mgd[var_key][tii], -qG.grad_max, qG.grad_max)
                    adm[var_key][:v][tii] .= beta2.*adm[var_key][:v][tii] .+ (1.0-beta2).*clamp.(mgd[var_key][tii], -qG.grad_max, qG.grad_max).^2.0
                    stt[var_key][tii]     .= stt[var_key][tii] .- qG.alpha_0[var_key].*(adm[var_key][:m][tii]./(1.0-beta1_decay))./(sqrt.(adm[var_key][:v][tii]./(1.0-beta2_decay)) .+ qG.eps)
                end
            end
        end
    end
end

# adam solver -- take steps for every element in the master_grad list
function flush_adam!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # loop over the keys in mgd
    for var_key in keys(mgd)

        # loop over all time
        for tii in prm.ts.time_keys
            # states to update                 
            if var_key in keys(upd)
                # flush the adam moments
                adm[var_key][:m][tii][upd[var_key][tii]]    .= 0.0
                adm[var_key][:v][tii][upd[var_key][tii]]    .= 0.0
                # => adm[var_key][:mhat][tii][update_subset] .= 0.0
                # => adm[var_key][:vhat][tii][update_subset] .= 0.0
            else
                # flush the adam moments
                adm[var_key][:m][tii]    .= 0.0
                adm[var_key][:v][tii]    .= 0.0
                # => adm[var_key][:mhat][tii] .= 0.0
                # => adm[var_key][:vhat][tii] .= 0.0
            end
        end
    end
end

function run_adam!(
        adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}},
        bit::Dict{Symbol, Dict{Symbol, BitVector}},
        cgd::quasiGrad.Cgd,
        ctb::Vector{Vector{Float64}},
        ctd::Vector{Vector{Float64}},
        flw::Dict{Symbol, Dict{Symbol, Vector{Float64}}},
        grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, 
        idx::quasiGrad.Idx,
        mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}},
        msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}},
        ntk::quasiGrad.Ntk,
        prm::quasiGrad.Param,
        qG::quasiGrad.QG, 
        scr::Dict{Symbol, Float64},
        stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
        sys::quasiGrad.System,
        upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, 
        wct::Vector{Vector{Int64}})

    # initialize
    adm_step    = 0
    beta1       = qG.beta1
    beta2       = qG.beta2
    beta1_decay = 1.0
    beta2_decay = 1.0
    run_adam    = true

    @info "Running adam for $(qG.adam_max_time) seconds!"
    
    # flush adam at each restart ?
    # println("adam NOT flushed")
    quasiGrad.flush_adam!(adm, mgd, prm, upd)

    # start the timer!
    adam_start = time()

    # loop over adam steps
    while run_adam
        # increment
        adm_step += 1

        # step decay
        # alpha = step_decay(adm_step, qG)

        # decay beta
        beta1_decay = beta1_decay*beta1
        beta2_decay = beta2_decay*beta2

        # update weight parameters?
        if qG.apply_grad_weight_homotopy == true
            update_penalties!(prm, qG, time(), adam_start, adam_start+qG.adam_max_time)
        end

        # compute all states and grads
        quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

        # take an adam step
        quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)
        # experiments!
            # => quasiGrad.adaGrad!(adm, alpha, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)
            # => quasiGrad.the_quasiGrad!(adm, mgd, prm, qG, stt, upd)
            # => quasiGrad.adam_with_ls!(adm, alpha, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd, cgd, ctb, ctd, flw, grd, idx, msc, ntk, scr, sys, wct)

        # take intermediate pf steps?
        if qG.take_adam_pf_steps == true
            for _ in 1:qG.num_adam_pf_step
                # update the power injection-associated gradients
                quasiGrad.update_states_and_grads_for_adam_pf!(bit, grd, idx, mgd, msc, prm, qG, stt, sys)

                # take an adam pf step (standard_adam=false)
                quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd, standard_adam = false)
            end
        end

        # stop?
        run_adam = adam_termination(adam_start, adm_step, qG, run_adam)
    end

    # one last clip + state computation -- no grad needed!
    qG.eval_grad = false
    quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
    qG.eval_grad = true
end

function update_states_and_grads!(
    bit::Dict{Symbol, Dict{Symbol, BitVector}},
    cgd::quasiGrad.Cgd, 
    ctb::Vector{Vector{Float64}},
    ctd::Vector{Vector{Float64}}, 
    flw::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, 
    idx::quasiGrad.Idx, 
    mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    ntk::quasiGrad.Ntk, 
    prm::quasiGrad.Param, 
    qG::quasiGrad.QG, 
    scr::Dict{Symbol, Float64}, 
    stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    sys::quasiGrad.System, 
    wct::Vector{Vector{Int64}})
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
    quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, qG, stt, sys)
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
        println("Skipping ctg evaluation!")
    else
        quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
    end
    # score the market surplus function
    quasiGrad.score_zt!(idx, prm, qG, scr, stt) 
    quasiGrad.score_zbase!(qG, scr)
    quasiGrad.score_zms!(scr)

    # print the market surplus function value
    quasiGrad.print_zms(qG, scr)

    # compute the master grad
    quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
end

function update_states_and_grads_for_adam_pf!(bit::Dict{Symbol, Dict{Symbol, BitVector}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # update the non-device states which affect power balance
    #
    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    # clip all basic states (i.e., the states which are iterated on)
    quasiGrad.clip_for_adam_pf!(prm, qG, stt)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(bit, grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)

    # compute the master grad
    quasiGrad.master_grad_adam_pf!(grd, idx, mgd, prm, sys)
end

function update_states_and_grads_for_solve_pf_lbfgs!(bit::Dict{Symbol, Dict{Symbol, BitVector}}, cgd::quasiGrad.Cgd, dpf0::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, zpf::Dict{Symbol, Dict{Symbol, Float64}})
    # in this function, we only update the states and gradients needed
    # to solve a single-time-period ACOPF with lbfgs:
    # 1) flush
    # 2) clip (including both p_on, based on binary values and dev_q)
    # 3) line, xfm, and shunt
    # 4a) device power (p/q) 
    # 4b) optional: power costs
    # 5) power balance
    # 6) score quadratic distance metric 
    # 7) run the master grad

    # if we are here, we want to make sure we are NOT running su/sd updates
    qG.run_susd_updates = false

    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    # clip all basic states (i.e., the states which are iterated on)
    qG.clip_pq_based_on_bins = true
    quasiGrad.clip_all!(prm, qG, stt)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.xfm_flows!(bit, grd, idx, msc, prm, qG, stt, sys)
    quasiGrad.shunts!(grd, idx, msc, prm, qG, stt)

    # device powers
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, qG, stt, sys)

    # include OPF costs? this regularizes/biases the solution
    if qG.include_energy_costs_lbfgs
        quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    end

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, msc, prm, qG, stt, sys)

    # take quadratic distance and directly take/apply the master gradient
    if qG.include_lbfgs_p0_regularization
        quasiGrad.quadratic_distance!(dpf0, mgd, prm, qG, stt)
    end

    # score
    quasiGrad.score_solve_pf!(prm, stt, zpf)

    # compute the master grad
    quasiGrad.master_grad_solve_pf!(cgd, grd, idx, mgd, prm, qG, stt, sys)
end

function batch_fix!(pct_round::Float64, prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # loop and concatenate
    bin_vec_del = Vector{Float64}(undef,(sys.nT*sys.ndev))

    for tii in prm.ts.time_keys
        bin_inds              = (1:sys.ndev) .+ (prm.ts.time_key_ind[tii]-1)*sys.ndev
        bin_vec_del[bin_inds] = stt[:u_on_dev][tii] .- stt[:u_on_dev_GRB][tii]
    end

    # sort and find the binaries that are closest to Gurobi's solution

    # which ones do we fix?
    num_bin_fix = Int64(round(sys.nT*sys.ndev*pct_round/100.0))
    bins_to_fix = sortperm(abs.(bin_vec_del))[1:num_bin_fix]
    # FYI!!! => most_sim_to_least_sim = sortperm(abs.(bin_vec_del))

    # now, we loop over time and check for each binary in "bins_to_fix"
    for tii in prm.ts.time_keys
        bin_inds          = (1:sys.ndev) .+ (prm.ts.time_key_ind[tii]-1)*sys.ndev
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

# lbfgs
function lbfgs!(lbfgs::Dict{Symbol, Vector{Float64}}, lbfgs_diff::Dict{Symbol, Vector{Vector{Float64}}}, lbfgs_idx::Vector{Int64}, lbfgs_map::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, lbfgs_step::Dict{Symbol, Float64}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # NOTE: based on testing on May 10 or so, lbfgs does NOT outperform adam,
    #       More fundamentally, it has a problem: the states "lbfgs[:x_now]",
    #       etc. need to modified after binaries are fixed. Right now, they
    #       are not. (i.e., some of the lbfgs states need to be removed).
    #
    # note: lbfgs_idx is a set of ordered indices, where the first is the most
    #       recent step information, and the last is the oldest step information
    #       in the following order: (k-1), (k-2)
    #
    # prepare the lbfgs structures -- x and gradf
    for var_key in keys(mgd)
        for tii in prm.ts.time_keys
            # states to update
            if var_key in keys(upd)
                update_subset = upd[var_key][tii]
                lbfgs[:x_now][lbfgs_map[var_key][tii]]     = copy(stt[var_key][tii][update_subset]) # no update_subset needed on lbfgs side
                lbfgs[:gradf_now][lbfgs_map[var_key][tii]] = copy(mgd[var_key][tii][update_subset]) # no update_subset needed on lbfgs side
            else
                lbfgs[:x_now][lbfgs_map[var_key][tii]]     = copy(stt[var_key][tii])
                lbfgs[:gradf_now][lbfgs_map[var_key][tii]] = copy(mgd[var_key][tii])
            end
        end
    end

    # if this is the very first iteration, just take a gradient step
    if sum(lbfgs_idx) == 0
        # if this is the very first iteration, just take a gradient step
        lbfgs[:x_new] = lbfgs[:x_now] - 1e-10*lbfgs[:gradf_now]

        # pass x back into the state vector
        for var_key in keys(mgd)
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    update_subset = upd[var_key][tii]
                    stt[var_key][tii][update_subset] = lbfgs[:x_new][lbfgs_map[var_key][tii]] # no update_subset needed
                else
                    stt[var_key][tii]                = lbfgs[:x_new][lbfgs_map[var_key][tii]]
                end
            end
        end

        # update the lbfgs states and grads
        lbfgs[:x_prev]     = copy(lbfgs[:x_now])
        lbfgs[:gradf_prev] = copy(lbfgs[:gradf_now])

        # now, let's initialize lbfgs_idx
        lbfgs_idx[1] = 1
    else
        # udpdate the state difference
        idx_km1 = lbfgs_idx[1]
        lbfgs_diff[:s][idx_km1] = lbfgs[:x_now]     - lbfgs[:x_prev]
        lbfgs_diff[:y][idx_km1] = lbfgs[:gradf_now] - lbfgs[:gradf_prev]
        lbfgs[:rho][idx_km1]    = 1/(quasiGrad.dot(lbfgs_diff[:s][idx_km1],lbfgs_diff[:y][idx_km1]))

        # now, double-loop and compute lbfgs values
        q = copy.(lbfgs[:gradf_now])
        for ii in lbfgs_idx[lbfgs_idx .!= 0] # k-1, k-2, ..., k-m
            lbfgs[:alpha][ii] = lbfgs[:rho][ii]*quasiGrad.dot(lbfgs_diff[:s][ii], q)
            q                -= lbfgs[:alpha][ii]*lbfgs_diff[:y][ii]
        end
        
        # set "r", which will be H*grad
        r = q*quasiGrad.dot(lbfgs_diff[:s][idx_km1],lbfgs_diff[:y][idx_km1])/quasiGrad.dot(lbfgs_diff[:y][idx_km1], lbfgs_diff[:y][idx_km1])
        
        # compute H*grad
        for ii in reverse(lbfgs_idx[lbfgs_idx .!= 0]) # k-m, k-m+1, ..., k-1
            # skip beta -- defined implicitly below
            r += lbfgs_diff[:s][ii]*(lbfgs[:alpha][ii] - lbfgs[:rho][ii]*quasiGrad.dot(lbfgs_diff[:y][ii], r))
        end

        if sum(lbfgs_idx) == 1
            # this is the first step, so just use 0.1
            lbfgs_step[:step] = 0.1
        else
            # decay beta
            lbfgs_step[:beta1_decay] = lbfgs_step[:beta1_decay]*qG.beta1
            lbfgs_step[:beta2_decay] = lbfgs_step[:beta2_decay]*qG.beta2

            # have the STEP take a step with adam!
            grad              = (scr[:nzms] - lbfgs_step[:nzms_prev])/lbfgs_step[:step]
            lbfgs_step[:m]    = qG.beta1.*lbfgs_step[:m] + (1.0-qG.beta1).*grad
            lbfgs_step[:v]    = qG.beta2.*lbfgs_step[:v] + (1.0-qG.beta2).*grad.^2.0
            lbfgs_step[:mhat] = lbfgs_step[:m]/(1.0-lbfgs_step[:beta1_decay])
            lbfgs_step[:vhat] = lbfgs_step[:v]/(1.0-lbfgs_step[:beta2_decay])
            lbfgs_step[:step] = lbfgs_step[:step] - lbfgs_step[:alpha_0]*lbfgs_step[:mhat]/(sqrt.(lbfgs_step[:vhat]) .+ qG.eps)
        end

        # lbfgs step
        lbfgs[:x_new] = lbfgs[:x_now] - lbfgs_step[:step]*r

        # pass x back into the state vector
        for var_key in keys(mgd)
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    update_subset = upd[var_key][tii]
                    stt[var_key][tii][update_subset] = lbfgs[:x_new][lbfgs_map[var_key][tii]] # no update_subset needed
                else
                    stt[var_key][tii]                = lbfgs[:x_new][lbfgs_map[var_key][tii]]
                end
            end
        end

        # update the lbfgs states and grads
        lbfgs[:x_prev]     = copy(lbfgs[:x_now])
        lbfgs[:gradf_prev] = copy(lbfgs[:gradf_now])

        # finally, update the lbfgs indices -- rule: lbfgs_idx[1] is where 
        # we write the newest data, and every next index is successively
        # older data -- oldest data gets bumped when the dataset if full.
        #
        # v = [data(0), -, -]  => lbfgs_idx = [1,0,0]
        #
        # v = [data(0), data(1), -]  => lbfgs_idx = [2,1,0]
        #
        # v = [data(0), data(1), data(2)]  => lbfgs_idx = [3,2,1]
        # 
        # v = [data(3), data(1), data(2)]  => lbfgs_idx = [1,3,2]
        #
        # v = [data(3), data(4), data(2)]  => lbfgs_idx = [2,1,3]
        #
        # ....
        #
        # so, 1 becomes 2, 2 becomes 3, etc. :
        if 0 ∈ lbfgs_idx
            circshift!(lbfgs_idx, -1)
            lbfgs_idx[1] = lbfgs_idx[2] + 1
        else
            circshift!(lbfgs_idx, -1)
        end
    end
end

# lbfgs
function solve_pf_lbfgs!(pf_lbfgs::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, pf_lbfgs_diff::Dict{Symbol, Dict{Symbol, Vector{Vector{Float64}}}}, pf_lbfgs_idx::Vector{Int64}, pf_lbfgs_map::Dict{Symbol, Vector{Int64}}, pf_lbfgs_step::Dict{Symbol, Dict{Symbol, Float64}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, zpf::Dict{Symbol})
    # note: pf_lbfgs_idx is a set of ordered indices, where the first is the most
    #       recent step information, and the last is the oldest step information
    #       in the following order: (k-1), (k-2)
    #
    # prepare the lbfgs structures -- x and gradf
    emergency_stop = false
    for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
        for tii in prm.ts.time_keys
            # states to update
            if var_key in keys(upd)
                pf_lbfgs[:x_now][tii][pf_lbfgs_map[var_key]]     .= copy.(stt[var_key][tii][upd[var_key][tii]]) # no update_subset needed on lbfgs side
                pf_lbfgs[:gradf_now][tii][pf_lbfgs_map[var_key]] .= copy.(mgd[var_key][tii][upd[var_key][tii]]) # no update_subset needed on lbfgs side
            else
                pf_lbfgs[:x_now][tii][pf_lbfgs_map[var_key]]     .= copy.(stt[var_key][tii])
                pf_lbfgs[:gradf_now][tii][pf_lbfgs_map[var_key]] .= copy.(mgd[var_key][tii])
            end
        end
    end

    # if this is the very first iteration, just take a gradient step
    if sum(pf_lbfgs_idx) == 0
        # we solve pf at each instant, so loop over all time!
        for tii in prm.ts.time_keys
            # if this is the very first iteration, just take a gradient step
            pf_lbfgs[:x_new][tii] .= pf_lbfgs[:x_now][tii] .- 1e-11*pf_lbfgs[:gradf_now][tii]
        end

        # pass x back into the state vector
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    stt[var_key][tii][upd[var_key][tii]] .= pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key]] # no update_subset needed
                else
                    stt[var_key][tii]                    .= pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key]]
                end
            end
        end

        # update the lbfgs states and grads
        for tii in prm.ts.time_keys
            pf_lbfgs[:x_prev][tii]     .= copy.(pf_lbfgs[:x_now][tii])
            pf_lbfgs[:gradf_prev][tii] .= copy.(pf_lbfgs[:gradf_now][tii])
        end

        # now, let's initialize lbfgs_idx
        pf_lbfgs_idx[1] = 1
    else
        # we solve pf at each instant, so loop over all time!
            # => for tii in prm.ts.time_keys
        @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys

            # udpdate the state difference
            idx_km1 = pf_lbfgs_idx[1]
            pf_lbfgs_diff[:s][tii][idx_km1] .= pf_lbfgs[:x_now][tii]     .- pf_lbfgs[:x_prev][tii]
            pf_lbfgs_diff[:y][tii][idx_km1] .= pf_lbfgs[:gradf_now][tii] .- pf_lbfgs[:gradf_prev][tii]
            rho                             = quasiGrad.dot(pf_lbfgs_diff[:s][tii][idx_km1], pf_lbfgs_diff[:y][tii][idx_km1])
            if abs(rho) < 1e-6
                # in this case, lbfgs is stalling out and might return a NaN if we're not careful
                #emergency_stop = true
                #@info "Breaking out of lbfgs loop! s'*y too small. NaN possible."
                break # this breaks from everything 
            end
            
            # if not y^T * s is not too small, go and ahead and take its inverse: rho === 1/(y^T * s)
            pf_lbfgs[:rho][tii][idx_km1] = 1.0/rho

            # now, double-loop and compute lbfgs values
            pf_lbfgs[:q][tii] .= copy.(pf_lbfgs[:gradf_now][tii])
            for ii in pf_lbfgs_idx[pf_lbfgs_idx .!= 0] # k-1, k-2, ..., k-m
                pf_lbfgs[:alpha][tii][ii] = pf_lbfgs[:rho][tii][ii]*quasiGrad.dot(pf_lbfgs_diff[:s][tii][ii], pf_lbfgs[:q][tii])
                pf_lbfgs[:q][tii]       .-= pf_lbfgs[:alpha][tii][ii].*pf_lbfgs_diff[:y][tii][ii]
            end
            
            # set "r", which will be H*grad
            pf_lbfgs[:r][tii] .= pf_lbfgs[:q][tii].*(quasiGrad.dot(pf_lbfgs_diff[:s][tii][idx_km1], pf_lbfgs_diff[:y][tii][idx_km1])/quasiGrad.dot(pf_lbfgs_diff[:y][tii][idx_km1], pf_lbfgs_diff[:y][tii][idx_km1]))
            
            # compute H*grad
            for ii in reverse(pf_lbfgs_idx[pf_lbfgs_idx .!= 0]) # k-m, k-m+1, ..., k-1
                # skip beta -- defined implicitly below
                pf_lbfgs[:r][tii] .+= pf_lbfgs_diff[:s][tii][ii].*(pf_lbfgs[:alpha][tii][ii] - pf_lbfgs[:rho][tii][ii]*quasiGrad.dot(pf_lbfgs_diff[:y][tii][ii], pf_lbfgs[:r][tii]))
            end

            # step size: let adam control?
            if sum(pf_lbfgs_idx) == 1
                # this is the first step, so just use qG.initial_pf_lbfgs_step (about ~0.1)
                pf_lbfgs_step[:step][tii] = qG.initial_pf_lbfgs_step
            else
                # decay beta
                pf_lbfgs_step[:beta1_decay][tii] = pf_lbfgs_step[:beta1_decay][tii]*qG.beta1
                pf_lbfgs_step[:beta2_decay][tii] = pf_lbfgs_step[:beta2_decay][tii]*qG.beta2

                # have the STEP take a step with adam!
                grad                      = ((zpf[:zp][tii]+zpf[:zq][tii]) - pf_lbfgs_step[:zpf_prev][tii])/pf_lbfgs_step[:step][tii]
                pf_lbfgs_step[:m][tii]    = qG.beta1.*pf_lbfgs_step[:m][tii] + (1.0-qG.beta1).*grad
                pf_lbfgs_step[:v][tii]    = qG.beta2.*pf_lbfgs_step[:v][tii] + (1.0-qG.beta2).*grad.^2.0
                pf_lbfgs_step[:mhat][tii] = pf_lbfgs_step[:m][tii]/(1.0-pf_lbfgs_step[:beta1_decay][tii])
                pf_lbfgs_step[:vhat][tii] = pf_lbfgs_step[:v][tii]/(1.0-pf_lbfgs_step[:beta2_decay][tii])
                pf_lbfgs_step[:step][tii] = pf_lbfgs_step[:step][tii] - pf_lbfgs_step[:alpha_0][tii]*pf_lbfgs_step[:mhat][tii]/(sqrt.(pf_lbfgs_step[:vhat][tii]) .+ qG.eps)
            end

            # lbfgs step
            pf_lbfgs[:x_new][tii] .= pf_lbfgs[:x_now][tii] .- pf_lbfgs_step[:step][tii].*pf_lbfgs[:r][tii]
        end

        # pass x back into the state vector
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    stt[var_key][tii][upd[var_key][tii]] .= copy.(pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key]]) # no update_subset needed
                else
                    stt[var_key][tii]                    .= copy.(pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key]])
                end
            end
        end

        # update the lbfgs states and grads
        for tii in prm.ts.time_keys
            pf_lbfgs[:x_prev][tii]     .= copy.(pf_lbfgs[:x_now][tii])
            pf_lbfgs[:gradf_prev][tii] .= copy.(pf_lbfgs[:gradf_now][tii])
        end

        # finally, update the lbfgs indices -- rule: lbfgs_idx[1] is where 
        # we write the newest data, and every next index is successively
        # older data -- oldest data gets bumped when the dataset if full.
        #
        # v = [data(0), -, -]  => lbfgs_idx = [1,0,0]
        #
        # v = [data(0), data(1), -]  => lbfgs_idx = [2,1,0]
        #
        # v = [data(0), data(1), data(2)]  => lbfgs_idx = [3,2,1]
        # 
        # v = [data(3), data(1), data(2)]  => lbfgs_idx = [1,3,2]
        #
        # v = [data(3), data(4), data(2)]  => lbfgs_idx = [2,1,3]
        #
        # ....
        #
        # so, 1 becomes 2, 2 becomes 3, etc. :
        if 0 ∈ pf_lbfgs_idx
            circshift!(pf_lbfgs_idx, -1)
            pf_lbfgs_idx[1] = pf_lbfgs_idx[2] + 1
        else
            circshift!(pf_lbfgs_idx, -1)
        end
    end

    # output
    return emergency_stop
end

function quadratic_distance!(dpf0::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # grab the distance between p_on and its initial value -- this is something we 
        # minimize, so the regularization function is positive valued
        # => value not needed: zdist      =   qG.cdist_psolve*(stt[:p_on][tii] - dpf0[:p_on][tii]).^2
            # => zdist_grad = 2.0.*qG.cdist_psolve*(stt[:p_on][tii] .- dpf0[:p_on][tii])

        # now, apply the gradients directly (no need to use dp_alpha!())
        mgd[:p_on][tii] .+= (2.0*qG.cdist_psolve).*(stt[:p_on][tii] .- dpf0[:p_on][tii])
    end
end

function build_acpf_Jac_and_pq0(msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol, Ybus_real::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, Ybus_imag::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64})

    # build the admittance structure
    NY  = [Ybus_real -Ybus_imag;
          -Ybus_imag -Ybus_real]

    # complex voltage
    cva = cos.(stt[:va][tii])
    sva = sin.(stt[:va][tii])
    vr  = stt[:vm][tii].*cva
    vi  = stt[:vm][tii].*sva

    # complex current:
    # ic = Yc*vc = (Ycr + j*Yci) * (vr + j*vi)
    # Ir = Ycr*vr - Yci*vi
    # Ii = Yci*vr + Ycr*vi
    Ir = Ybus_real*vr - Ybus_imag*vi
    Ii = Ybus_imag*vr + Ybus_real*vi

    # Populate MI
    MIr = quasiGrad.spdiagm(sys.nb, sys.nb, Ir)
    MIi = quasiGrad.spdiagm(sys.nb, sys.nb, Ii)
    MI  = [MIr  MIi;
          -MIi  MIr]

    # Populate MV
    MVr = quasiGrad.spdiagm(sys.nb, sys.nb, vr)
    MVi = quasiGrad.spdiagm(sys.nb, sys.nb, vi)
    MV  = [MVr -MVi;
           MVi  MVr]

    # Populate RV
    RV = [quasiGrad.spdiagm(sys.nb, sys.nb, cva)  quasiGrad.spdiagm(sys.nb, sys.nb,-vi); 
          quasiGrad.spdiagm(sys.nb, sys.nb, sva)  quasiGrad.spdiagm(sys.nb, sys.nb,vr)];

    # build full Jacobian
    Jac = (MI + MV*NY)*RV

    # also compute injections?
    if qG.compute_pf_injs_with_Jac
        # complex coordinates -- don't actually do this :)
        # => Yb = Ybus_real + im*Ybus_imag
        # => vc = stt[:vm][tii].*(exp.(im*stt[:va][tii]))
        # => ic = Yb*vc
        # => sc = vc.*conj.(ic)
        # => pinj = real(sc)
        # => qinj = imag(sc)
        msc[:pinj0][tii] .= (vr.*Ir .+ vi.*Ii)
        msc[:qinj0][tii] .= (vi.*Ir .- vr.*Ii)
    end

    # Output
    return Jac
end

# solve power flow
function newton_power_flow(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, KP::Float64, pi_p::Vector{Float64}, prm::quasiGrad.Param, PQidx::Vector{Int64}, qG::quasiGrad.QG, residual::Vector{Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol, Ybus_real::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, Ybus_imag::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64})
    # initialize
    run_pf = true

    # ============
    @warn "This function doesn't work -- but it can be easily updated if we want it to."

    # start with all buses as PV if they have Q capacity
    buses = 1:sys.nb
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
        x = [stt[:vm][tii]; stt[:va][tii][2:end]]
        
        # loop over each bus and compute the residual
        quasiGrad.power_flow_residual!(idx, residual, stt, sys, tii)

        # test the residual for termination
        if quasiGrad.norm(residual[residual_idx]) < 1e-5
            run_pf = false
        else
            println("residual:")
            println(quasiGrad.norm(residual[residual_idx]))
            sleep(0.75)

            # update the Jacobian
            Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)

            # take a Newton step -- do NOT put the step scaler inside the parantheses
            x = x - (Jac\residual[residual_idx])

            # update the state
            stt[:vm][tii][PQidx] = x[1:nPQ]
            stt[:va][tii][2:end] = x[(nPQ+2):end]

            # update the flows and residual and such
            quasiGrad.update_states_for_pf!(grd, idx, prm, qG, stt)
        end
    end
end

function power_flow_residual!(idx::quasiGrad.Idx, residual::Vector{Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol)
    # loop over each bus and compute the residual
    for bus in 1:sys.nb
        # active power balance: stt[:pb][:slack][tii][bus] to record with time
        residual[bus] = 
            # consumers (positive)
            sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt[:acline_pfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt[:acline_pto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt[:xfm_pfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt[:xfm_pto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0)
    
        # reactive power balance
        residual[sys.nb + bus] = 
            # consumers (positive)
            sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) +
            # shunt
            sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) +
            # acline
            sum(stt[:acline_qfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
            sum(stt[:acline_qto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
            # xfm
            sum(stt[:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
            sum(stt[:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
            # dcline
            sum(stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
            sum(stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
            # producer (negative)
            -sum(stt[:dev_q][tii][idx.pr[bus]]; init=0.0)
    end
end

function solve_power_flow!(bit::Dict{Symbol, Dict{Symbol, BitVector}}, cgd::quasiGrad.Cgd, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # 1. fire up lbfgs, as controlled by adam, WITH regularization + OPF
    # 2. after a short period, use Gurobi to solve successive power flows
    # 3. pass solution back to lbfgs to clean up, WITHOUT regularization + OPF -- not needed yet
    #
    # step 1: intialize lbfgs =======================================
    dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf = quasiGrad.initialize_pf_lbfgs(mgd, prm, qG, stt, sys, upd);

    # turn on extra influence
    qG.include_energy_costs_lbfgs      = false # true
    qG.include_lbfgs_p0_regularization = true

    # set the loss function to quadratic -- low gradient factor
    qG.pqbal_grad_type = "quadratic_for_lbfgs"

    # loop -- lbfgs
    init_pf   = true
    run_lbfgs = true
    lbfgs_cnt = 0
    zt0       = 0.0

    # initialize: compute all states and grads
    quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(bit, cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)

    # loop -- lbfgs
    while run_lbfgs == true
        # take an lbfgs step
        emergency_stop = quasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, stt, upd, zpf)

        # save zpf BEFORE updating with the new state -- don't track bias terms
        for tii in prm.ts.time_keys
            pf_lbfgs_step[:zpf_prev][tii] = (zpf[:zp][tii]+zpf[:zq][tii]) 
        end

        # compute all states and grads
        quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(bit, cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)

        # store the first value
        zp = sum(sum([zpf[:zp][tii] for tii in prm.ts.time_keys]))
        zq = sum(sum([zpf[:zq][tii] for tii in prm.ts.time_keys]))
        zt = zp + zq
        if init_pf == true
            zt0 = copy(zt)
            init_pf = false
        end

        # print
        if qG.print_lbfgs_iterations == true
            ztr = round(zt; sigdigits = 3)
            zpr = round(zp; sigdigits = 3)
            zqr = round(zq; sigdigits = 3)
            stp = round(sum(pf_lbfgs_step[:step][tii] for tii in prm.ts.time_keys)/sys.nT; sigdigits = 3)
            println("Total: $(ztr), P penalty: $(zpr), Q penalty: $(zqr), avg adam step: $(stp)!")
        end

        # increment
        lbfgs_cnt += 1

        # quit if the error gets too large relative to the first error
        if (lbfgs_cnt > qG.num_lbfgs_steps) || (zt > 5.0*zt0) || (emergency_stop == true)
            run_lbfgs = false
        end
    end

    # step 2: Gurobi linear solve (projection)
    quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

    # change the gradient type back
    qG.pqbal_grad_type = "soft_abs"
end

function ideal_dispatch!(idx::quasiGrad.Idx, msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol)
    # here, we compute the "ideal" dipatch point (after a few steps of LBFGS)
    #
    # pinj = p_pr - p_cs - p_dc
    #
    # loop over each bus
    for bus in 1:sys.nb
        # active power
        msc[:pinj_ideal][tii][bus] = 
            # consumers (positive)
            -sum(stt[:dev_p][tii][cs] for cs in idx.cs[bus]; init=0.0) +
            # dcline
            -sum(stt[:dc_pfr][tii][dc_fr] for dc_fr in idx.bus_is_dc_frs[bus]; init=0.0) + 
            -sum(stt[:dc_pto][tii][dc_to] for dc_to in idx.bus_is_dc_tos[bus]; init=0.0) +
            # producer (negative)
            sum(stt[:dev_p][tii][pr] for pr in idx.pr[bus]; init=0.0)
    
        # also, find the reactive power point
        msc[:qinj_ideal][tii][bus] = 
            # consumers (positive)
            -sum(stt[:dev_q][tii][cs] for cs in idx.cs[bus]; init=0.0) +
            # dcline
            -sum(stt[:dc_qfr][tii][dc_fr] for dc_fr in idx.bus_is_dc_frs[bus]; init=0.0) + 
            -sum(stt[:dc_qto][tii][dc_to] for dc_to in idx.bus_is_dc_tos[bus] ; init=0.0) +
            # producer (negative)
            sum(stt[:dev_q][tii][pr] for pr in idx.pr[bus]; init=0.0)
    end
end

function solve_linear_pf_with_Gurobi_NOT_parallel!(idx::quasiGrad.Idx, msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, qG::quasiGrad.QG,  stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # Solve linearized power flow with Gurobi -- use margin tinkering to guarentee convergence. Only consinder upper 
    # and lower bounds on the p/q production (no other limits).
    #
    # ask Gurobi to solve a linearize power flow. two options here:
    #   1) define device variables which are bounded, and then insert them into the power balance expression
    #   2) just define power balance bounds based on device characteristics, and then, at the end, optimally
    #      dispatch the devices via updating.
    # 
    # I am pretty sure 1) is slower, but it is (1) safer, (2) simpler, and (3) more flexible -- and it will
    # require less trouble-shooting. Plus, it will be easy to update/improve in the future! Go with it.
    #
    #
    # here is power balance:
    #
    # p_pr - p_cs - pdc = p_lines/xfm/shunt => this is typical.
    #
    # vm0 = stt[:vm][tii]
    # va0 = stt[:va][tii][2:end-1]
    #
    # bias point: msc[:pinj0, qinj0] === Y * stt[:vm, va]
    qG.Gurobi_pf_obj            = "min_dispatch_distance"
    qG.compute_pf_injs_with_Jac = true

    # build and empty the model!
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "Threads", qG.num_threads)
    set_string_names_on_creation(model, false)
    set_silent(model)

    @info "Running lineaized power flow across $(sys.nT) time periods."

    # loop over time
    for (t_ind, tii) in enumerate(prm.ts.time_keys)

        # initialize
        run_pf    = true
        pf_cnt    = 0  # successes
        total_pfs = 0  # successes AND fails

        # set a q_margin and a v_margin to make convergence easier (0.0 is nominal)
        first_fail = true
        q_margin   = 0.0
        v_margin   = 0.0

        # 1. update the ideal dispatch point (active power) -- we do this just once
            # quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)
            # this is no longer needed, because we penalize device injections directly

        # 2. update y_bus and Jacobian and bias point -- this
        #    only needs to be done once per time, since xfm/shunt
        #    values are not changing between iterations
        Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

        # loop over pf solves
        while run_pf == true

            # safety margins should NEVER be 0
            q_margin = max(q_margin, 0.0)
            v_margin = max(v_margin, 0.0)

            # increment
            total_pfs += 1
            pf_cnt    += 1

            # first, rebuild the jacobian, and update the
            # base points: msc[:pinj0], msc[:qinj0]
            Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
            
            # empty model
            empty!(model)

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])
            set_start_value.(x_in, [stt[:vm][tii]; stt[:va][tii][2:end]])

            # assign
            dvm = x_in[1:sys.nb]
            dva = x_in[(sys.nb+1):end]

            # note:
            # vm   = vm0   + dvm
            # va   = va0   + dva
            # pinj = pinj0 + dpinj
            # qinj = qinj0 + dqinj
            #
            # key equation:
            #                       dPQ .== Jac*dVT
            #                       dPQ + basePQ(v) = devicePQ
            #
            #                       Jac*dVT + basePQ(v) == devicePQ
            #
            # so, we don't actually need to model dPQ explicitly (cool)
            #
            # so, the optimizer asks, how shall we tune dVT in order to produce a power perurbation
            # which, when added to the base point, lives inside the feasible device region?
            #
            # based on the result, we only have to actually update the device set points on the very
            # last power flow iteration, where we have converged.

            # now, model all nodal injections from the device/dc line side, all put in nodal_p/q
            nodal_p = Vector{AffExpr}(undef, sys.nb)
            nodal_q = Vector{AffExpr}(undef, sys.nb)
            for bus in 1:sys.nb
                # now, we need to loop and set the affine expressions to 0, and then add powers
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                nodal_p[bus] = AffExpr(0.0)
                nodal_q[bus] = AffExpr(0.0)
            end

            # create a flow variable for each dc line and sum these into the nodal vectors
            if sys.nldc == 0
                # nothing to see here
            else

                # define dc variables
                @variable(model, pdc_vars[1:sys.nldc])    # oriented so that fr = + !!
                @variable(model, qdc_fr_vars[1:sys.nldc])
                @variable(model, qdc_to_vars[1:sys.nldc])

                set_start_value.(pdc_vars, stt[:dc_pfr][tii])
                set_start_value.(qdc_fr_vars, stt[:dc_qfr][tii])
                set_start_value.(qdc_to_vars, stt[:dc_qto][tii])

                # bound dc power
                @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
                @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
                @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

                # loop and add to the nodal injection vectors
                for dcl in 1:sys.nldc
                    add_to_expression!(nodal_p[idx.dc_fr_bus[dcl]], -pdc_vars[dcl])
                    add_to_expression!(nodal_p[idx.dc_to_bus[dcl]], +pdc_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_fr_bus[dcl]], -qdc_fr_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_to_bus[dcl]], -qdc_to_vars[dcl])
                end
            end
            
            # next, deal with devices
            @variable(model, dev_p_vars[1:sys.ndev])
            @variable(model, dev_q_vars[1:sys.ndev])
  
            set_start_value.(dev_p_vars, stt[:dev_p][tii])
            set_start_value.(dev_q_vars, stt[:dev_q][tii])

            # call the bounds -- note: this is fairly approximate,
            # since these bounds do not include, e.g., ramp rate constraints
            # between the various time windows -- this will be addressed in the
            # final, constrained power flow solve
            dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
            dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
            dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
            dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]

            # ignore binaries?
            # => dev_plb = prm.dev.p_lb_tmdv[t_ind]
            # => dev_pub = prm.dev.p_ub_tmdv[t_ind]
            # => dev_qlb = prm.dev.q_lb_tmdv[t_ind]
            # => dev_qub = prm.dev.q_ub_tmdv[t_ind]

            # first, define p_on at this time
            # => p_on = dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii]

            # bound
            @constraint(model, dev_plb + stt[:p_su][tii] + stt[:p_sd][tii] .<= dev_p_vars .<= dev_pub + stt[:p_su][tii] + stt[:p_sd][tii])
            # alternative: => @constraint(model, dev_plb .<= dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii] .<= dev_pub)
            @constraint(model, (dev_qlb .- q_margin) .<= dev_q_vars .<= (dev_qub .+ q_margin))

            # apply additional bounds: J_pqe (equality constraints)
            if ~isempty(idx.J_pqe)
                @constraint(model, dev_q_vars[idx.J_pqe] - prm.dev.beta[idx.J_pqe].*dev_p_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe].*stt[:u_sum][tii][idx.J_pqe])
                # alternative: @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt[:u_sum][tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe])
            end

            # apply additional bounds: J_pqmin/max (inequality constraints)
            #
            # note: when the reserve products are negelected, pr and cs constraints are the same
            #   remember: idx.J_pqmax == idx.J_pqmin
            if ~isempty(idx.J_pqmax)
                @constraint(model, dev_q_vars[idx.J_pqmax] .<= prm.dev.q_0_ub[idx.J_pqmax].*stt[:u_sum][tii][idx.J_pqmax] .+ prm.dev.beta_ub[idx.J_pqmax].*dev_p_vars[idx.J_pqmax])
                @constraint(model, prm.dev.q_0_lb[idx.J_pqmax].*stt[:u_sum][tii][idx.J_pqmax] .+ prm.dev.beta_lb[idx.J_pqmax].*dev_p_vars[idx.J_pqmax] .<= dev_q_vars[idx.J_pqmax])
            end

            # great, now just update the nodal injection vectors
            for dev in 1:sys.ndev
                if dev in idx.pr_devs
                    # producers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], dev_p_vars[dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], dev_q_vars[dev])
                else
                    # consumers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], -dev_p_vars[dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], -dev_q_vars[dev])
                end
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, (prm.bus.vm_lb .- v_margin) - stt[:vm][tii] .<= dvm .<= (prm.bus.vm_ub .+ v_margin) - stt[:vm][tii])
            # alternative: => @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm .<= prm.bus.vm_ub)

            # mapping
            JacP_noref = @view Jac[1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
            JacQ_noref = @view Jac[(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]

            @constraint(model, JacP_noref*x_in + msc[:pinj0][tii] .== nodal_p)
            @constraint(model, JacQ_noref*x_in + msc[:qinj0][tii] .== nodal_q)

            # objective: hold p (and v?) close to its initial value
            # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
            if qG.Gurobi_pf_obj == "min_dispatch_distance"
                # this finds a solution close to the dispatch point -- does not converge without v,a regularization
                obj = AffExpr(0.0)

                # every 10% of devices, introduce a new slack variable -- limit complexity
                num_dev_per_slack  = Int(round(0.10*sys.ndev))
                slack_power_weight = 15.0

                tmp_devp = @variable(model)
                add_to_expression!(obj, tmp_devp, slack_power_weight/(sys.nb/num_dev_per_slack))
                for dev in 1:sys.ndev
                    if mod(dev,num_dev_per_slack) == 0
                        tmp_devp = @variable(model)
                        add_to_expression!(obj, tmp_devp, slack_power_weight/(sys.nb/num_dev_per_slack))
                    end
                    @constraint(model, stt[:dev_p][tii][dev] - dev_p_vars[dev] <= tmp_devp)
                    @constraint(model, dev_p_vars[dev] - stt[:dev_p][tii][dev] <= tmp_devp)
                end

                # regularize against voltage movement -- this adds light 
                # regularization and causes convergence
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                add_to_expression!(obj, tmp_vm)
                add_to_expression!(obj, tmp_va)
                for bus in 1:sys.nb
                    # voltage regularization
                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    # phase regularization
                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end

                    # for injections:
                        # => @constraint(model, msc[:pinj_ideal][bus] - nodal_p[bus] <= tmp_p)
                        # => @constraint(model, nodal_p[bus] - msc[:pinj_ideal][bus] <= tmp_p)
                end

            elseif qG.Gurobi_pf_obj == "min_dispatch_perturbation"
                # this finds a solution with minimum movement -- not really needed
                # now that "min_dispatch_distance" converges
                tmp_p  = @variable(model)
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                for bus in 1:sys.nb
                    #tmp = @variable(model)
                    @constraint(model, -JacP_noref[bus,:]*x_in <= tmp_p)
                    @constraint(model,  JacP_noref[bus,:]*x_in <= tmp_p)

                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end
                    # for l1 norm: add_to_expression!(obj, tmp)
                end
                add_to_expression!(obj, tmp_vm)
                add_to_expression!(obj, tmp_va)
                add_to_expression!(obj, tmp_p, 100.0)
            else
                @warn "pf solver objective not recognized!"
            end

            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # test solution!
            soln_valid = solution_status(model)

            # if we have reached our max number of tries, jjsut quit after this
            if total_pfs == qG.max_linear_pfs_total
                run_pf = false
            end

            # test validity
            if soln_valid == true
                # no matter what, we update the voltage soluion
                stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
                stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

                if q_margin > 0.0
                    # Let's update the margins, but this isn't a valid (marginless) solution yet..
                    if (v_margin > 0.0)
                        # in this case, let's try again, but without voltage support
                        v_margin = 0.0
                        q_margin = 0.01
                    else
                        q_margin = 0.01
                    end
                else
                    # valid solution, and no margins needed!
                    #
                    # take the norm of dv and da
                    max_dx = maximum(abs.(value.(x_in)))
                    
                    # println("========================================================")
                    if qG.print_linear_pf_iterations == true
                        println(termination_status(model), ". time: $(tii). ","objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
                    end
                    # println("========================================================")
                    #
                    # shall we terminate?
                    if (max_dx < qG.max_pf_dx) || (pf_cnt == qG.max_linear_pfs)
                        run_pf = false

                        # now, apply the updated injections to the devices
                        stt[:dev_p][tii]  = value.(dev_p_vars)
                        stt[:p_on][tii]   = stt[:dev_p][tii] - stt[:p_su][tii] - stt[:p_sd][tii]
                        stt[:dev_q][tii]  = value.(dev_q_vars)
                        if sys.nldc > 0
                            stt[:dc_pfr][tii] =  value.(pdc_vars)
                            stt[:dc_pto][tii] = -value.(pdc_vars)  # also, performed in clipping
                            stt[:dc_qfr][tii] = value.(qdc_fr_vars)
                            stt[:dc_qto][tii] = value.(qdc_to_vars)
                        end
                    end
                end
            else
                # the solution is NOT valid, so we should increase bounds and try again
                @warn "Linearized power flow failed at time ($tii) -- increasing margins."
                if first_fail == true
                    first_fail = false
                    q_margin = 0.01
                else
                    q_margin += 0.025
                    v_margin += 0.001
                end
            end
        end
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

function step_decay(adm_step::Int64, qG::quasiGrad.QG)
    # what type of step decay should we employ?
    if qG.decay_type == "cos"
        alpha = qG.alpha_min + 0.5*(qG.alpha_max - qG.alpha_min)*(1+cos((adm_step/qG.Ti)*pi))
    elseif qG.decay_type == "exponential"
        alpha = qG.alpha_0*(qG.step_decay^adm_step)
    else
        @assert qG.decay_type == "none"
        alpha = copy(qG.alpha_0)
    end

    # output
    return alpha
end

function adam_termination(adam_start::Float64, adm_step::Int64, qG::quasiGrad.QG, run_adam::Bool)
    # stopping criteria
    if qG.adam_stopper == "time"
        if time() - adam_start >= qG.adam_max_time
            run_adam = false
        end
    elseif qG.adam_stopper == "iterations"
        if adm_step >= qG.adam_max_its
            run_adam = false
        end
    else
        # uh-oh -- no stopper!
    end

    # output
    return run_adam
end

function solve_parallel_linear_pf_with_Gurobi!(idx::quasiGrad.Idx, msc::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, qG::quasiGrad.QG,  stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # Solve linearized power flow with Gurobi -- use margin tinkering to guarentee convergence. Only consinder upper 
    # and lower bounds on the p/q production (no other limits).
    #
    # ask Gurobi to solve a linearize power flow. two options here:
    #   1) define device variables which are bounded, and then insert them into the power balance expression
    #   2) just define power balance bounds based on device characteristics, and then, at the end, optimally
    #      dispatch the devices via updating.
    # 
    # I am pretty sure 1) is slower, but it is (1) safer, (2) simpler, and (3) more flexible -- and it will
    # require less trouble-shooting. Plus, it will be easy to update/improve in the future! Go with it.
    #
    #
    # here is power balance:
    #
    # p_pr - p_cs - pdc = p_lines/xfm/shunt => this is typical.
    #
    # vm0 = stt[:vm][tii]
    # va0 = stt[:va][tii][2:end-1]
    #
    # bias point: msc[:pinj0, qinj0] === Y * stt[:vm, va]
    qG.Gurobi_pf_obj            = "min_dispatch_distance"
    qG.compute_pf_injs_with_Jac = true

    @info "Running parallel lineaized power flows across $(sys.nT) time periods."

    # loop over time
    Threads.@threads for t_ind in 1:sys.nT

        # call the time
        tii = prm.ts.time_keys[t_ind]

        # initialize
        run_pf    = true
        pf_cnt    = 0  # successes
        total_pfs = 0  # successes AND fails

        # set a q_margin and a v_margin to make convergence easier (0.0 is nominal)
        first_fail = true
        q_margin   = 0.0
        v_margin   = 0.0

        # 1. update the ideal dispatch point (active power) -- we do this just once
            # quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)
            # this is no longer needed, because we penalize device injections directly

        # 2. update y_bus and Jacobian and bias point -- this
        #    only needs to be done once per time, since xfm/shunt
        #    values are not changing between iterations
        Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

        # loop over pf solves
        while run_pf == true

            # build an empty model!
            model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => 1))
            set_string_names_on_creation(model, false)

            # lower the tolerance a bit..
            quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", 1e-5)

            # safety margins should NEVER be 0
            q_margin = max(q_margin, 0.0)
            v_margin = max(v_margin, 0.0)

            # increment
            total_pfs += 1
            pf_cnt    += 1

            # first, rebuild the jacobian, and update the
            # base points: msc[:pinj0], msc[:qinj0]
            Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])
            set_start_value.(x_in, [stt[:vm][tii]; stt[:va][tii][2:end]])

            # assign
            dvm = x_in[1:sys.nb]
            dva = x_in[(sys.nb+1):end]

            # note:
            # vm   = vm0   + dvm
            # va   = va0   + dva
            # pinj = pinj0 + dpinj
            # qinj = qinj0 + dqinj
            #
            # key equation:
            #                       dPQ .== Jac*dVT
            #                       dPQ + basePQ(v) = devicePQ
            #
            #                       Jac*dVT + basePQ(v) == devicePQ
            #
            # so, we don't actually need to model dPQ explicitly (cool)
            #
            # so, the optimizer asks, how shall we tune dVT in order to produce a power perurbation
            # which, when added to the base point, lives inside the feasible device region?
            #
            # based on the result, we only have to actually update the device set points on the very
            # last power flow iteration, where we have converged.

            # now, model all nodal injections from the device/dc line side, all put in nodal_p/q
            nodal_p = Vector{AffExpr}(undef, sys.nb)
            nodal_q = Vector{AffExpr}(undef, sys.nb)
            for bus in 1:sys.nb
                # now, we need to loop and set the affine expressions to 0, and then add powers
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                nodal_p[bus] = AffExpr(0.0)
                nodal_q[bus] = AffExpr(0.0)
            end

            # create a flow variable for each dc line and sum these into the nodal vectors
            if sys.nldc == 0
                # nothing to see here
            else

                # define dc variables
                @variable(model, pdc_vars[1:sys.nldc])    # oriented so that fr = + !!
                @variable(model, qdc_fr_vars[1:sys.nldc])
                @variable(model, qdc_to_vars[1:sys.nldc])

                set_start_value.(pdc_vars, stt[:dc_pfr][tii])
                set_start_value.(qdc_fr_vars, stt[:dc_qfr][tii])
                set_start_value.(qdc_to_vars, stt[:dc_qto][tii])

                # bound dc power
                @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
                @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
                @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

                # loop and add to the nodal injection vectors
                for dcl in 1:sys.nldc
                    add_to_expression!(nodal_p[idx.dc_fr_bus[dcl]], -pdc_vars[dcl])
                    add_to_expression!(nodal_p[idx.dc_to_bus[dcl]], +pdc_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_fr_bus[dcl]], -qdc_fr_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_to_bus[dcl]], -qdc_to_vars[dcl])
                end
            end
            
            # next, deal with devices
            @variable(model, dev_p_vars[1:sys.ndev])
            @variable(model, dev_q_vars[1:sys.ndev])
  
            set_start_value.(dev_p_vars, stt[:dev_p][tii])
            set_start_value.(dev_q_vars, stt[:dev_q][tii])

            # call the bounds -- note: this is fairly approximate,
            # since these bounds do not include, e.g., ramp rate constraints
            # between the various time windows -- this will be addressed in the
            # final, constrained power flow solve
            dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
            dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
            dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
            dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]

            # ignore binaries?
            # => dev_plb = prm.dev.p_lb_tmdv[t_ind]
            # => dev_pub = prm.dev.p_ub_tmdv[t_ind]
            # => dev_qlb = prm.dev.q_lb_tmdv[t_ind]
            # => dev_qub = prm.dev.q_ub_tmdv[t_ind]

            # first, define p_on at this time
            # => p_on = dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii]

            # bound
            @constraint(model, dev_plb + stt[:p_su][tii] + stt[:p_sd][tii] .<= dev_p_vars .<= dev_pub + stt[:p_su][tii] + stt[:p_sd][tii])
            # alternative: => @constraint(model, dev_plb .<= dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii] .<= dev_pub)
            @constraint(model, (dev_qlb .- q_margin) .<= dev_q_vars .<= (dev_qub .+ q_margin))

            # apply additional bounds: J_pqe (equality constraints)
            if ~isempty(idx.J_pqe)
                @constraint(model, dev_q_vars[idx.J_pqe] - prm.dev.beta[idx.J_pqe].*dev_p_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe].*stt[:u_sum][tii][idx.J_pqe])
                # alternative: @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt[:u_sum][tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe])
            end

            # apply additional bounds: J_pqmin/max (inequality constraints)
            #
            # note: when the reserve products are negelected, pr and cs constraints are the same
            #   remember: idx.J_pqmax == idx.J_pqmin
            if ~isempty(idx.J_pqmax)
                @constraint(model, dev_q_vars[idx.J_pqmax] .<= prm.dev.q_0_ub[idx.J_pqmax].*stt[:u_sum][tii][idx.J_pqmax] .+ prm.dev.beta_ub[idx.J_pqmax].*dev_p_vars[idx.J_pqmax])
                @constraint(model, prm.dev.q_0_lb[idx.J_pqmax].*stt[:u_sum][tii][idx.J_pqmax] .+ prm.dev.beta_lb[idx.J_pqmax].*dev_p_vars[idx.J_pqmax] .<= dev_q_vars[idx.J_pqmax])
            end

            # great, now just update the nodal injection vectors
            for dev in 1:sys.ndev
                if dev in idx.pr_devs
                    # producers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], dev_p_vars[dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], dev_q_vars[dev])
                else
                    # consumers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], -dev_p_vars[dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], -dev_q_vars[dev])
                end
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, (prm.bus.vm_lb .- v_margin) - stt[:vm][tii] .<= dvm .<= (prm.bus.vm_ub .+ v_margin) - stt[:vm][tii])
            # alternative: => @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm .<= prm.bus.vm_ub)

            # mapping
            JacP_noref = @view Jac[1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
            JacQ_noref = @view Jac[(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]

            @constraint(model, JacP_noref*x_in + msc[:pinj0][tii] .== nodal_p)
            @constraint(model, JacQ_noref*x_in + msc[:qinj0][tii] .== nodal_q)

            # find the subset of devices with the highest bounds
                # => if (sys.npr > 15) && (sys.ncs > 15)
                # =>     dev_bound_inds = reverse(sortperm(dev_pub - dev_plb))
                # =>     pr_to_vary     = intersect(dev_bound_inds, idx.pr_not_Jpqe)[1:Int(round(sys.npr/10))]
                # =>     cs_to_vary     = intersect(dev_bound_inds, idx.cs_not_Jpqe)[1:Int(round(sys.ncs/10))]
                # => else
                # =>     # just vary them all..
                # =>     pr_to_vary     = idx.pr_devs
                # =>     cs_to_vary     = idx.cs_devs
                # => end

            # objective: hold p (and v?) close to its initial value
            # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
            if qG.Gurobi_pf_obj == "min_dispatch_distance"
                # this finds a solution close to the dispatch point -- does not converge without v,a regularization
                obj = AffExpr(0.0)

                # constraint some devices!
                    # => for dev in 1:sys.ndev
                    # =>     if (dev in pr_to_vary) || (dev in cs_to_vary)
                    # =>         tmp_devp = @variable(model)
                    # =>         @constraint(model, stt[:dev_p][tii][dev] - dev_p_vars[dev] <= tmp_devp)
                    # =>         @constraint(model, dev_p_vars[dev] - stt[:dev_p][tii][dev] <= tmp_devp)
                    # =>         add_to_expression!(obj, tmp_devp, 5.0/(0.1*sys.nb))
                    # =>     else
                    # =>         @constraint(model, stt[:dev_p][tii][dev] == dev_p_vars[dev])
                    # =>     end
                    # => end

                # every 10% of devices, introduce a new slack variable -- limit complexity
                num_dev_per_slack  = Int(round(0.1*sys.ndev))
                slack_power_weight = 2.5
                
                tmp_devp = @variable(model)
                add_to_expression!(obj, tmp_devp, slack_power_weight/(sys.nb/num_dev_per_slack))
                for dev in 1:sys.ndev
                    if mod(dev,num_dev_per_slack) == 0
                        tmp_devp = @variable(model)
                        add_to_expression!(obj, tmp_devp, slack_power_weight/(sys.nb/num_dev_per_slack))
                    end
                    @constraint(model, stt[:dev_p][tii][dev] - dev_p_vars[dev] <= tmp_devp)
                    @constraint(model, dev_p_vars[dev] - stt[:dev_p][tii][dev] <= tmp_devp)
                end

                # regularize against voltage movement -- this adds light 
                # regularization and causes convergence
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                add_to_expression!(obj, tmp_vm)
                add_to_expression!(obj, tmp_va)
                for bus in 1:sys.nb
                    # voltage regularization
                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    # phase regularization
                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end

                    # for injections:
                        # => @constraint(model, msc[:pinj_ideal][bus] - nodal_p[bus] <= tmp_p)
                        # => @constraint(model, nodal_p[bus] - msc[:pinj_ideal][bus] <= tmp_p)
                end

            elseif qG.Gurobi_pf_obj == "min_dispatch_perturbation"
                # this finds a solution with minimum movement -- not really needed
                # now that "min_dispatch_distance" converges
                tmp_p  = @variable(model)
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                for bus in 1:sys.nb
                    #tmp = @variable(model)
                    @constraint(model, -JacP_noref[bus,:]*x_in <= tmp_p)
                    @constraint(model,  JacP_noref[bus,:]*x_in <= tmp_p)

                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end
                    # for l1 norm: add_to_expression!(obj, tmp)
                end
                add_to_expression!(obj, tmp_vm)
                add_to_expression!(obj, tmp_va)
                add_to_expression!(obj, tmp_p, 100.0)
            else
                @warn "pf solver objective not recognized!"
            end

            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # test solution!
            soln_valid = solution_status(model)

            # if we have reached our max number of tries, jjsut quit after this
            if total_pfs == qG.max_linear_pfs_total
                run_pf = false
            end

            # test validity
            if soln_valid == true
                # no matter what, we update the voltage soluion
                stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
                stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

                # now, apply the updated injections to the devices
                #stt[:dev_p][tii]  = value.(dev_p_vars)
                #stt[:p_on][tii]   = stt[:dev_p][tii] - stt[:p_su][tii] - stt[:p_sd][tii]
                #stt[:dev_q][tii]  = value.(dev_q_vars)
                #if sys.nldc > 0
                #    stt[:dc_pfr][tii] =  value.(pdc_vars)
                #    stt[:dc_pto][tii] = -value.(pdc_vars)  # also, performed in clipping
                #    stt[:dc_qfr][tii] = value.(qdc_fr_vars)
                #    stt[:dc_qto][tii] = value.(qdc_to_vars)
                #end

                if q_margin > 0.0
                    # Let's update the margins, but this isn't a valid (marginless) solution yet..
                    if (v_margin > 0.0)
                        # in this case, let's try again, but without voltage support
                        v_margin = 0.0
                        q_margin = 0.01
                    else
                        q_margin = 0.01
                    end
                else
                    # valid solution, and no margins needed!
                    #
                    # take the norm of dv and da
                    max_dx = maximum(abs.(value.(x_in)))
                    
                    # println("========================================================")
                    if qG.print_linear_pf_iterations == true
                        println(termination_status(model), ". time: $(tii). ","objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
                    end
                    # println("========================================================")
                    #
                    # shall we terminate?
                    if (max_dx < qG.max_pf_dx) || (pf_cnt == qG.max_linear_pfs)
                        run_pf = false

                        # now, apply the updated injections to the devices
                        stt[:dev_p][tii]  = value.(dev_p_vars)
                        stt[:p_on][tii]   = stt[:dev_p][tii] - stt[:p_su][tii] - stt[:p_sd][tii]
                        stt[:dev_q][tii]  = value.(dev_q_vars)
                        if sys.nldc > 0
                            stt[:dc_pfr][tii] =  value.(pdc_vars)
                            stt[:dc_pto][tii] = -value.(pdc_vars)  # also, performed in clipping
                            stt[:dc_qfr][tii] = value.(qdc_fr_vars)
                            stt[:dc_qto][tii] = value.(qdc_to_vars)
                        end
                    end
                end
            else
                # the solution is NOT valid, so we should increase bounds and try again
                @warn "Linearized power flow failed at time ($tii) -- increasing margins."
                if first_fail == true
                    first_fail = false
                    q_margin = 0.01
                else
                    q_margin += 0.025
                    v_margin += 0.001
                end
            end
        end
    end
end