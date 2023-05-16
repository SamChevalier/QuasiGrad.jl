# adam solver -- take steps for every element in the master_grad list
function adam!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, alpha::Float64, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    
    # loop over the keys in mgd
    for var_key in keys(mgd)

        # loop over all time
        for tii in prm.ts.time_keys
            # states to update                                            
            if var_key in keys(upd)
                # => (var_key in keys(upd)) ? (update = upd[var_key][tii]) : (update = Colon())
                #    the above caused weird type instability, so we just copy and paste
                update_subset = upd[var_key][tii]

                # downscale the p_on gradient? Do this when we are
                # biasing the power flow solution towards a point
                # which is as close as possible to the economic dispatach solution.
                if (var_key == :p_on) && (qG.bias_pf == true)
                    clipped_grad = qG.bias_pf_scale_pgrad*clamp.(mgd[var_key][tii][update_subset],-qG.grad_max,qG.grad_max)
                else
                    clipped_grad = clamp.(mgd[var_key][tii][update_subset],-qG.grad_max,qG.grad_max)
                end

                # update adam moments
                clipped_grad                            = clamp.(mgd[var_key][tii][update_subset],-qG.grad_max,qG.grad_max)
                adm[var_key][:m][tii][update_subset]    = beta1.*adm[var_key][:m][tii][update_subset] + (1.0-beta1).*clipped_grad
                adm[var_key][:v][tii][update_subset]    = beta2.*adm[var_key][:v][tii][update_subset] + (1.0-beta2).*clipped_grad.^2.0
                adm[var_key][:mhat][tii][update_subset] = adm[var_key][:m][tii][update_subset]/(1.0-beta1_decay)
                adm[var_key][:vhat][tii][update_subset] = adm[var_key][:v][tii][update_subset]/(1.0-beta2_decay)

                # update the states
                stt[var_key][tii][update_subset] = stt[var_key][tii][update_subset] - alpha*adm[var_key][:mhat][tii][update_subset]./(sqrt.(adm[var_key][:vhat][tii][update_subset]) .+ qG.eps)
                
            else 
                # downscale the p_on gradient? Do this when we are
                # biasing the power flow solution towards a point
                # which is as close as possible to the economic dispatach solution.
                if (var_key == :p_on) && (qG.bias_pf == true)
                    clipped_grad = qG.bias_pf_scale_pgrad*clamp.(mgd[var_key][tii],-qG.grad_max,qG.grad_max)
                else
                    clipped_grad = clamp.(mgd[var_key][tii],-qG.grad_max,qG.grad_max)
                end

                # update adam moments
                clipped_grad                     = clamp.(mgd[var_key][tii],-qG.grad_max,qG.grad_max)
                adm[var_key][:m][tii]    = beta1.*adm[var_key][:m][tii] + (1.0-beta1).*clipped_grad
                adm[var_key][:v][tii]    = beta2.*adm[var_key][:v][tii] + (1.0-beta2).*clipped_grad.^2.0
                adm[var_key][:mhat][tii] = adm[var_key][:m][tii]/(1.0-beta1_decay)
                adm[var_key][:vhat][tii] = adm[var_key][:v][tii]/(1.0-beta2_decay)

                # update the states
                stt[var_key][tii] = stt[var_key][tii] - alpha*adm[var_key][:mhat][tii]./(sqrt.(adm[var_key][:vhat][tii]) .+ qG.eps)
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
                update_subset = upd[var_key][tii]

                # flush the adam moments
                adm[var_key][:m][tii][update_subset]    .= 0.0
                adm[var_key][:v][tii][update_subset]    .= 0.0
                adm[var_key][:mhat][tii][update_subset] .= 0.0
                adm[var_key][:vhat][tii][update_subset] .= 0.0
            else
                # flush the adam moments
                adm[var_key][:m][tii]    .= 0.0
                adm[var_key][:v][tii]    .= 0.0
                adm[var_key][:mhat][tii] .= 0.0
                adm[var_key][:vhat][tii] .= 0.0
            end
        end
    end
end

function run_adam_with_plotting!(
        adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}},
        ax::quasiGrad.Makie.Axis,
        cgd::quasiGrad.Cgd,
        fig::quasiGrad.Makie.Figure,
        flw::Dict{Symbol, Vector{Float64}},
        grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, 
        idx::quasiGrad.Idx,
        mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
        ntk::quasiGrad.Ntk,
        plt::Dict{Symbol, Integer},
        prm::quasiGrad.Param,
        qG::quasiGrad.QG, 
        scr::Dict{Symbol, Float64},
        stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
        sys::quasiGrad.System,
        upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}},
        dz_dpinj_base::Vector{Vector{Float64}},
        theta_k_base::Vector{Vector{Float64}},   
        worst_ctgs::Vector{Vector{Int64}},
        z_plt::Dict{Symbol, Dict{Symbol, Float64}})

    # initialize
    adm_step    = 0
    beta1       = qG.beta1
    beta2       = qG.beta2
    beta1_decay = 1.0
    beta2_decay = 1.0
    run_adam    = true
    
    # flush adam at each restart
    #if qG.flush_adam == true
    #    quasiGrad.flush_adam!(adm, mgd, prm, upd)
    #end

    # add Gurobi Projection line?
    if !plt[:first_plot]
        # add a dark vertical line
        quasiGrad.Makie.lines!(ax, [plt[:global_adm_step], plt[:global_adm_step]], [-20, 20], color = :black, linestyle = :dot, linewidth = 3.0)
    end

    # start the timer!
    adam_start = time()

    # loop over adam steps
    while run_adam
        # increment
        adm_step += 1
        plt[:global_adm_step] += 1 # for plotting

        # what type of step decay should we employ?
        if qG.decay_type == "cos"
            alpha = qG.alpha_min + 0.5*(qG.alpha_max - qG.alpha_min)*(1+cos((adm_step/qG.Ti)*pi))
        elseif qG.decay_type == "exponential"
            alpha = qG.alpha_0*(qG.step_decay^adm_step)
        else
            @assert qG.decay_type == "none"
            alpha = copy(qG.alpha_0)
        end

        # decay beta
        beta1_decay = beta1_decay*beta1
        beta2_decay = beta2_decay*beta2

        # compute all states and grads
        quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

        # take an adam step
        quasiGrad.adam!(adm, alpha, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

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

        # plot the progress
        quasiGrad.update_plot!(adm_step, ax, fig, plt, qG, scr, z_plt)
        display(fig)
    end

    # one last clip + state computation -- no grad needed!
    qG.eval_grad = false
    quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

    # turn it back on
    qG.eval_grad = true
end

function run_adam!(
        adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}},
        cgd::quasiGrad.Cgd,
        flw::Dict{Symbol, Vector{Float64}},
        grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, 
        idx::quasiGrad.Idx,
        mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
        ntk::quasiGrad.Ntk,
        plt::Dict{Symbol, Integer},
        prm::quasiGrad.Param,
        qG::quasiGrad.QG, 
        scr::Dict{Symbol, Float64},
        stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
        sys::quasiGrad.System,
        upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}},
        dz_dpinj_base::Vector{Vector{Float64}},
        theta_k_base::Vector{Vector{Float64}},   
        worst_ctgs::Vector{Vector{Int64}})

    # initialize
    adm_step    = 0
    beta1       = qG.beta1
    beta2       = qG.beta2
    beta1_decay = 1.0
    beta2_decay = 1.0
    run_adam    = true
    
    # flush adam at each restart
    quasiGrad.flush_adam!(adm, mgd, prm, upd)

    # start the timer!
    adam_start = time()

    # loop over adam steps
    while run_adam
        # increment
        plt[:global_adm_step] += 1 # for plotting
        adm_step += 1

        # what type of step decay should we employ?
        if qG.decay_type == "cos"
            alpha = qG.alpha_min + 0.5*(qG.alpha_max - qG.alpha_min)*(1+cos((adm_step/qG.Ti)*pi))
        elseif qG.decay_type == "exponential"
            alpha = qG.alpha_0*(qG.step_decay^adm_step)
        else
            @assert qG.decay_type == "none"
            alpha = copy(qG.alpha_0)
        end

        # decay beta
        beta1_decay = beta1_decay*beta1
        beta2_decay = beta2_decay*beta2

        # compute all states and grads
        quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

        # take an adam step
        quasiGrad.adam!(adm, alpha, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

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
    end

    # one last clip + state computation -- no grad needed!
    qG.eval_grad = false
    quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

    # turn it back on
    qG.eval_grad = true
end

function update_states_and_grads!(cgd::quasiGrad.Cgd, flw::Dict{Symbol, Vector{Float64}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, dz_dpinj_base::Vector{Vector{Float64}}, theta_k_base::Vector{Vector{Float64}}, worst_ctgs::Vector{Vector{Int64}})
    # if we are here, we want to make sure we are running su/sd updates
    qG.run_susd_updates = true

    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, sys)

    # clip all basic states (i.e., the states which are iterated on)
    quasiGrad.clip_all!(prm, stt)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(grd, idx, prm, qG, stt)
    quasiGrad.xfm_flows!(grd, idx, prm, qG, stt)
    quasiGrad.shunts!(grd, idx, prm, qG, stt)

    # device powers
    quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
    quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, stt, sys)
    quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
    quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
    quasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)
    quasiGrad.device_reserve_costs!(prm, stt)

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

    # compute reserve margins and penalties (no grads here)
    quasiGrad.reserve_balance!(idx, prm, stt, sys)

    # score the contingencies and take the gradients
    if qG.bias_pf == false # in this case, don't evaluate the expensive ctg grads
        quasiGrad.solve_ctgs!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)
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

function update_states_and_grads_solve_pf!(dpf0::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, zpf::Dict{Symbol, Dict{Symbol, Float64}})
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
    quasiGrad.flush_gradients!(grd, mgd, prm, sys)

    # clip all basic states (i.e., the states which are iterated on)
    quasiGrad.clip_all!(prm, stt)
    
    # compute network flows and injections
    quasiGrad.acline_flows!(grd, idx, prm, qG, stt)
    quasiGrad.xfm_flows!(grd, idx, prm, qG, stt)
    quasiGrad.shunts!(grd, idx, prm, qG, stt)

    # device powers
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
    quasiGrad.device_reactive_powers!(idx, prm, stt, sys)
    # => add this back in? quasiGrad.energy_costs!(grd, prm, qG, stt, sys)

    # now, we can compute the power balances
    quasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

    # take quadratic distance and directly take/apply the master gradient
    quasiGrad.quadratic_distance!(dpf0, mgd, prm, qG, stt)

    # score
    quasiGrad.score_solve_pf!(prm, stt, zpf)

    # compute the master grad
    quasiGrad.master_grad_solve_pf!(grd, idx, mgd, prm, qG, stt, sys)
end

function batch_fix!(GRB::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, pct_round::Float64, prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # loop and concatenate
    bin_vec_del = Vector{Float64}(undef,(sys.nT*sys.ndev))

    for tii in prm.ts.time_keys
        bin_inds              = (1:sys.ndev) .+ (prm.ts.time_key_ind[tii]-1)*sys.ndev
        bin_vec_del[bin_inds] = stt[:u_on_dev][tii] - GRB[:u_on_dev][tii]
    end

    # sort and find the binaries that are closest to Gurobi's solution
    most_sim_to_least_sim = sortperm(abs.(bin_vec_del))

    # which ones do we fix?
    num_bin_fix = Int64(round(sys.nT*sys.ndev*pct_round/100.0))
    bins_to_fix = most_sim_to_least_sim[1:num_bin_fix]

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
        # no => deleteat!(upd[:u_on_dev][tii],local_bins_to_fix)
    end
end

# initialize the plot
function initialize_plot(plt::Dict{Symbol, Integer}, scr::Dict{Symbol, Float64})
    fig = quasiGrad.Makie.Figure(resolution=(1200, 600), fontsize=18) 
    ax  = quasiGrad.Makie.Axis(fig[1, 1], xlabel = "adam iteration", ylabel = "score values (z)", title = "quasiGrad")
    quasiGrad.Makie.xlims!(ax, [1, plt[:N_its]])

    # set ylims -- this is tricky, since we use "1000" as zero, so the scale goes,
    # -10^4 -10^3 0 10^3 10^4...
    min_y     = (-log10(abs(scr[:zms])) - 1.0) + 3.0
    min_y_int = ceil(min_y)

    max_y     = (+log10(scr[:ed_obj]) + 0.25) - 3.0
    max_y_int = floor(max_y)

    # since "1000" is our reference -- see scaling function notes
    y_vec = collect((min_y_int):(max_y_int))
    quasiGrad.Makie.ylims!(ax, [min_y, max_y])
    tick_name = String[]
    for yv in y_vec
        if yv == 0
            push!(tick_name,"0")
        elseif yv < 0
            push!(tick_name,"-10^"*string(Int(abs(yv - 3.0))))
        else
            push!(tick_name,"+10^"*string(Int(abs(yv + 3.0))))
        end
    end
    ax.yticks = (y_vec, tick_name)
    display(fig)

    # define current and previous dicts
    z_plt = Dict(:prev => Dict(
                            :zms  => 0.0,
                            :pzms => 0.0,       
                            :zhat => 0.0,
                            :ctg  => 0.0,
                            :zp   => 0.0,
                            :zq   => 0.0,
                            :acl  => 0.0,
                            :xfm  => 0.0,
                            :zoud => 0.0,
                            :zone => 0.0,
                            :rsv  => 0.0,
                            :enpr => 0.0,
                            :encs => 0.0,
                            :emnx => 0.0,
                            :zsus => 0.0),
                :now => Dict(
                            :zms  => 0.0,
                            :pzms => 0.0,     
                            :zhat => 0.0,
                            :ctg  => 0.0,
                            :zp   => 0.0,
                            :zq   => 0.0,
                            :acl  => 0.0,
                            :xfm  => 0.0,
                            :zoud => 0.0,
                            :zone => 0.0,
                            :rsv  => 0.0,
                            :enpr => 0.0,
                            :encs => 0.0,
                            :emnx => 0.0,
                            :zsus => 0.0))

    # output
    return ax, fig, z_plt
end

# update the plot
function update_plot!(adm_step::Int64, ax::quasiGrad.Makie.Axis, fig::quasiGrad.Makie.Figure, plt::Dict{Symbol, Integer}, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, z_plt::Dict{Symbol, Dict{Symbol, Float64}})
    #
    # is this the first plot? if adm_step == 1, then we don't plot (just update)
    if adm_step > 1 # !(plt[:first_plot] || adm_step == 1)
        # first, set the current values
        z_plt[:now][:zms]  = scale_z(scr[:zms])
        z_plt[:now][:pzms] = scale_z(scr[:zms_penalized])      
        z_plt[:now][:zhat] = scale_z(scr[:zt_penalty] - qG.delta*scr[:zhat_mxst])
        z_plt[:now][:ctg]  = scale_z(scr[:zctg_min] + scr[:zctg_avg])
        z_plt[:now][:emnx] = scale_z(scr[:emnx])
        z_plt[:now][:zp]   = scale_z(scr[:zp])
        z_plt[:now][:zq]   = scale_z(scr[:zq])
        z_plt[:now][:acl]  = scale_z(scr[:acl])
        z_plt[:now][:xfm]  = scale_z(scr[:xfm])
        z_plt[:now][:zoud] = scale_z(scr[:zoud])
        z_plt[:now][:zone] = scale_z(scr[:zone])
        z_plt[:now][:rsv]  = scale_z(scr[:rsv])
        z_plt[:now][:enpr] = scale_z(scr[:enpr])
        z_plt[:now][:encs] = scale_z(scr[:encs])
        z_plt[:now][:zsus] = scale_z(scr[:zsus])

        # x-axis
        if plt[:global_adm_step] > plt[:N_its]
            plt[:N_its] = plt[:N_its] + 150
            quasiGrad.Makie.xlims!(ax, [1, plt[:N_its]])
        end

        # now, plot!
        #
        # add an economic dipatch upper bound
        l0 = quasiGrad.Makie.lines!(ax, [0, 1e4], [log10(scr[:ed_obj]) - 3.0, log10(scr[:ed_obj]) - 3.0], color = :coral1, linestyle = :dash, linewidth = 5.0)

        l1  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zms],  z_plt[:now][:zms] ], color = :cornflowerblue, linewidth = 4.5)
        l2  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:pzms], z_plt[:now][:pzms]], color = :mediumblue,     linewidth = 3.0)

        l3  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zhat], z_plt[:now][:zhat]], color = :goldenrod1, linewidth = 2.0)

        l4  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:ctg] , z_plt[:now][:ctg] ], color = :lightslateblue)

        l5  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zp]  , z_plt[:now][:zp]  ], color = :firebrick, linewidth = 3.5)
        l6  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zq]  , z_plt[:now][:zq]  ], color = :salmon1,   linewidth = 2.0)

        l7  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:acl] , z_plt[:now][:acl] ], color = :darkorange1, linewidth = 3.5)
        l8  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:xfm] , z_plt[:now][:xfm] ], color = :orangered1,  linewidth = 2.0)
        
        l9  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zoud], z_plt[:now][:zoud]], color = :grey95, linewidth = 3.5, linestyle = :solid)
        l10 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zone], z_plt[:now][:zone]], color = :gray89, linewidth = 3.0, linestyle = :dot)
        l11 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:rsv] , z_plt[:now][:rsv] ], color = :gray75, linewidth = 2.5, linestyle = :dash)
        l12 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:emnx], z_plt[:now][:emnx]], color = :grey38, linewidth = 2.0, linestyle = :dashdot)
        l13 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zsus], z_plt[:now][:zsus]], color = :grey0,  linewidth = 1.5, linestyle = :dashdotdot)

        l14 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:enpr], z_plt[:now][:enpr]], color = :forestgreen, linewidth = 3.5)
        l15 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:encs], z_plt[:now][:encs]], color = :darkgreen,   linewidth = 2.0)
        
        if plt[:first_plot] == true  # this will occur only once
            plt[:first_plot] = false # toggle

            # define trace lables
            label = Dict(
                :zms  => "market surplus",
                :pzms => "penalized market surplus",       
                :zhat => "constraint penalties", 
                :ctg  => "contingency penalties",
                :zp   => "active power balance",
                :zq   => "reactive power balance",
                :acl  => "acline flow",  
                :xfm  => "xfm flow", 
                :zoud => "on/up/down costs",
                :zone => "zonal reserve penalties",
                :rsv  => "local reserve penalties",
                :enpr => "energy costs (pr)",   
                :encs => "energy revenues (cs)",   
                :emnx => "min/max energy violations",
                :zsus => "start-up state discount",
                :ed   => "economic dispatch (bound)")

            # build legend ==================
            quasiGrad.Makie.Legend(fig[1, 2], [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15],
                            [label[:ed],  label[:zms],  label[:pzms], label[:zhat], label[:ctg], label[:zp],   label[:zq],   label[:acl],
                             label[:xfm], label[:zoud], label[:zone], label[:rsv], label[:emnx], label[:zsus], label[:enpr], 
                             label[:encs]],
                             halign = :right, valign = :top, framevisible = false)
        end

        # display the figure
        if mod(adm_step,plt[:disp_freq]) == 0
            # display(fig) => this does nothing!!
            sleep(1e-10) # I don't know why this needs to be here..
        end
    end

    # update the previous values!
    z_plt[:prev][:zms]  = scale_z(scr[:zms])
    z_plt[:prev][:pzms] = scale_z(scr[:zms_penalized])      
    z_plt[:prev][:zhat] = scale_z(scr[:zt_penalty] - qG.delta*scr[:zhat_mxst])
    z_plt[:prev][:ctg]  = scale_z(scr[:zctg_min] + scr[:zctg_avg])
    z_plt[:prev][:emnx] = scale_z(scr[:emnx])
    z_plt[:prev][:zp]   = scale_z(scr[:zp])
    z_plt[:prev][:zq]   = scale_z(scr[:zq])
    z_plt[:prev][:acl]  = scale_z(scr[:acl])
    z_plt[:prev][:xfm]  = scale_z(scr[:xfm])
    z_plt[:prev][:zoud] = scale_z(scr[:zoud])
    z_plt[:prev][:zone] = scale_z(scr[:zone])
    z_plt[:prev][:rsv]  = scale_z(scr[:rsv])
    z_plt[:prev][:enpr] = scale_z(scr[:enpr])
    z_plt[:prev][:encs] = scale_z(scr[:encs])
    z_plt[:prev][:zsus] = scale_z(scr[:zsus])
end

# function to rescale scores for plotting :)
function scale_z(z::Float64)
    sgn  = sign(z .+ 1e-6)
    absz = abs(z)
    if absz < 1000.0 # clip
        absz = 1000.0
    end
    if sgn < 0
        # shift up two
        zs = sgn*log10(absz) + 3.0
    else
        # shift down two
        zs = sgn*log10(absz) - 3.0
        # +10^5 => 2
        # +10^4 => 1
        # -10^1/2/3 = +10^1/2/3 => 0
        # -10^4 => -1
        # -10^5 => -2
    end

    # output
    return zs
end

# lbfgs
function lbfgs!(lbfgs::Dict{Symbol, Vector{Float64}}, lbfgs_diff::Dict{Symbol, Vector{Vector{Float64}}}, lbfgs_idx::Vector{Int64}, lbfgs_map::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, lbfgs_step::Dict{Symbol, Float64}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # NOTE: based on testing on May 10 or so, lbfgs does not outperform adam,
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
        q = copy(lbfgs[:gradf_now])
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
function solve_pf_lbfgs!(pf_lbfgs::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, pf_lbfgs_diff::Dict{Symbol, Dict{Symbol, Vector{Vector{Float64}}}}, pf_lbfgs_idx::Vector{Int64}, pf_lbfgs_map::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, pf_lbfgs_step::Dict{Symbol, Dict{Symbol, Float64}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, zpf::Dict{Symbol})
    # note: pf_lbfgs_idx is a set of ordered indices, where the first is the most
    #       recent step information, and the last is the oldest step information
    #       in the following order: (k-1), (k-2)
    #
    # prepare the lbfgs structures -- x and gradf
    for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
        for tii in prm.ts.time_keys
            # states to update
            if var_key in keys(upd)
                update_subset = upd[var_key][tii]
                pf_lbfgs[:x_now][tii][pf_lbfgs_map[var_key][tii]]     = copy(stt[var_key][tii][update_subset]) # no update_subset needed on lbfgs side
                pf_lbfgs[:gradf_now][tii][pf_lbfgs_map[var_key][tii]] = copy(mgd[var_key][tii][update_subset]) # no update_subset needed on lbfgs side
            else
                pf_lbfgs[:x_now][tii][pf_lbfgs_map[var_key][tii]]     = copy(stt[var_key][tii])
                pf_lbfgs[:gradf_now][tii][pf_lbfgs_map[var_key][tii]] = copy(mgd[var_key][tii])
            end
        end
    end

    # if this is the very first iteration, just take a gradient step
    if sum(pf_lbfgs_idx) == 0
        # we solve pf at each instant, so loop over all time!
        for tii in prm.ts.time_keys
            # if this is the very first iteration, just take a gradient step
            pf_lbfgs[:x_new][tii] = pf_lbfgs[:x_now][tii] - 1e-11*pf_lbfgs[:gradf_now][tii]
        end

        # pass x back into the state vector
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    update_subset = upd[var_key][tii]
                    stt[var_key][tii][update_subset] = pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key][tii]] # no update_subset needed
                else
                    stt[var_key][tii]                = pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key][tii]]
                end
            end
        end

        # update the lbfgs states and grads
        pf_lbfgs[:x_prev]     = deepcopy(pf_lbfgs[:x_now])
        pf_lbfgs[:gradf_prev] = deepcopy(pf_lbfgs[:gradf_now])

        # now, let's initialize lbfgs_idx
        pf_lbfgs_idx[1] = 1
    else
        # we solve pf at each instant, so loop over all time!
        for tii in prm.ts.time_keys
            # udpdate the state difference
            idx_km1 = pf_lbfgs_idx[1]
            pf_lbfgs_diff[:s][tii][idx_km1] = pf_lbfgs[:x_now][tii]     - pf_lbfgs[:x_prev][tii]
            pf_lbfgs_diff[:y][tii][idx_km1] = pf_lbfgs[:gradf_now][tii] - pf_lbfgs[:gradf_prev][tii]
            pf_lbfgs[:rho][tii][idx_km1]    = 1.0/(quasiGrad.dot(pf_lbfgs_diff[:s][tii][idx_km1],pf_lbfgs_diff[:y][tii][idx_km1]))

            # now, double-loop and compute lbfgs values
            q = copy(pf_lbfgs[:gradf_now][tii])
            for ii in pf_lbfgs_idx[pf_lbfgs_idx .!= 0] # k-1, k-2, ..., k-m
                pf_lbfgs[:alpha][tii][ii] = pf_lbfgs[:rho][tii][ii]*quasiGrad.dot(pf_lbfgs_diff[:s][tii][ii], q)
                q                        -= pf_lbfgs[:alpha][tii][ii]*pf_lbfgs_diff[:y][tii][ii]
            end
            
            # set "r", which will be H*grad
            r = q*quasiGrad.dot(pf_lbfgs_diff[:s][tii][idx_km1], pf_lbfgs_diff[:y][tii][idx_km1])/quasiGrad.dot(pf_lbfgs_diff[:y][tii][idx_km1], pf_lbfgs_diff[:y][tii][idx_km1])
            
            # compute H*grad
            for ii in reverse(pf_lbfgs_idx[pf_lbfgs_idx .!= 0]) # k-m, k-m+1, ..., k-1
                # skip beta -- defined implicitly below
                r += pf_lbfgs_diff[:s][tii][ii]*(pf_lbfgs[:alpha][tii][ii] - pf_lbfgs[:rho][tii][ii]*quasiGrad.dot(pf_lbfgs_diff[:y][tii][ii], r))
            end

            if sum(pf_lbfgs_idx) == 1
                # this is the first step, so just use 0.15
                pf_lbfgs_step[:step][tii] = 0.10
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
            pf_lbfgs[:x_new][tii] = pf_lbfgs[:x_now][tii] - pf_lbfgs_step[:step][tii]*r
        end

        # pass x back into the state vector
        for var_key in [:vm, :va, :tau, :phi, :dc_pfr, :dc_qfr, :dc_qto, :u_step_shunt, :p_on, :dev_q]
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    update_subset = upd[var_key][tii]
                    stt[var_key][tii][update_subset] = pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key][tii]] # no update_subset needed
                else
                    stt[var_key][tii]                = pf_lbfgs[:x_new][tii][pf_lbfgs_map[var_key][tii]]
                end
            end
        end

        # update the lbfgs states and grads
        pf_lbfgs[:x_prev]     = deepcopy(pf_lbfgs[:x_now])
        pf_lbfgs[:gradf_prev] = deepcopy(pf_lbfgs[:gradf_now])

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
end

function quadratic_distance!(dpf0::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    for tii in prm.ts.time_keys
        # grab the distance between p_on and its initial value -- this is something we 
        # minimize, so the regularization function is positive valued
        zdist      =   qG.cdist_psolve*(stt[:p_on][tii] - dpf0[:p_on][tii]).^2
        zdist_grad = 2*qG.cdist_psolve*(stt[:p_on][tii] - dpf0[:p_on][tii])

        # now, apply the gradients directly (no need to use dp_alpha!())
        mgd[:p_on][tii] .+= zdist_grad
    end
end 