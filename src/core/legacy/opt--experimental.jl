# adam solver -- take steps for every element in the master_grad list
function full_adam!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, alpha::Float64, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    
    # loop over the keys in mgd
    for var_key in keys(mgd)

        # loop over all time
        for tii in prm.ts.time_keys
            # states to update                                            
            if var_key in keys(upd)
                # => (var_key in keys(upd)) ? (update = upd[var_key][tii]) : (update = Colon())
                #    the above caused weird type instability, so we just copy and paste
                update_subset = upd[var_key][tii]

                # update adam moments
                clipped_grad                            = clamp.(mgd[var_key][tii][update_subset],-qG.grad_max,qG.grad_max)
                adm[var_key][:m][tii][update_subset]    = beta1.*adm[var_key][:m][tii][update_subset] + (1.0-beta1).*clipped_grad
                adm[var_key][:v][tii][update_subset]    = beta2.*adm[var_key][:v][tii][update_subset] + (1.0-beta2).*clipped_grad.^2.0
                adm[var_key][:mhat][tii][update_subset] = adm[var_key][:m][tii][update_subset]/(1.0-beta1_decay)
                adm[var_key][:vhat][tii][update_subset] = adm[var_key][:v][tii][update_subset]/(1.0-beta2_decay)

                # update the states
                stt[var_key][tii][update_subset] = stt[var_key][tii][update_subset] - alpha*adm[var_key][:mhat][tii][update_subset]./(sqrt.(adm[var_key][:vhat][tii][update_subset]) .+ qG.eps)
                
            else 
                # update adam moments
                clipped_grad             = clamp.(mgd[var_key][tii],-qG.grad_max,qG.grad_max)
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

function adam_with_ls!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, alpha::Float64, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, cgd::quasiGrad.Cgd, ctb::Vector{Vector{Float64}}, ctd::Vector{Vector{Float64}}, flw::Dict{Symbol, Vector{Float64}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, msc::Dict{Symbol, Vector{Float64}}, ntk::quasiGrad.Ntk, scr::Dict{Symbol, Float64}, sys::quasiGrad.System, wct::Vector{Vector{Int64}})
    
    z0 = copy(scr[:zms_penalized])

    # loop over the keys in mgd
    for var_key in keys(mgd)

        # loop over all time
        for tii in prm.ts.time_keys
            # states to update                                            
            if var_key in keys(upd)
                # => (var_key in keys(upd)) ? (update = upd[var_key][tii]) : (update = Colon())
                #    the above caused weird type instability, so we just copy and paste
                update_subset = upd[var_key][tii]

                # update adam moments
                clipped_grad                            = clamp.(mgd[var_key][tii][update_subset],-qG.grad_max,qG.grad_max)
                adm[var_key][:m][tii][update_subset]    = beta1.*adm[var_key][:m][tii][update_subset] + (1.0-beta1).*clipped_grad
                adm[var_key][:v][tii][update_subset]    = beta2.*adm[var_key][:v][tii][update_subset] + (1.0-beta2).*clipped_grad.^2.0
                adm[var_key][:mhat][tii][update_subset] = adm[var_key][:m][tii][update_subset]/(1.0-beta1_decay)
                adm[var_key][:vhat][tii][update_subset] = adm[var_key][:v][tii][update_subset]/(1.0-beta2_decay)

                # update the states
                adm[var_key][:prev_stt][tii][update_subset]  = copy(stt[var_key][tii][update_subset])
                adm[var_key][:prev_grad][tii][update_subset] = copy(clipped_grad) #copy(adm[var_key][:mhat][tii][update_subset]./(sqrt.(adm[var_key][:vhat][tii][update_subset]) .+ qG.eps))
                
                stt[var_key][tii][update_subset] = stt[var_key][tii][update_subset] - alpha*adm[var_key][:mhat][tii][update_subset]./(sqrt.(adm[var_key][:vhat][tii][update_subset]) .+ qG.eps)
                
            else
                # update adam moments
                clipped_grad                     = clamp.(mgd[var_key][tii],-qG.grad_max,qG.grad_max)
                adm[var_key][:m][tii]    = beta1.*adm[var_key][:m][tii] + (1.0-beta1).*clipped_grad
                adm[var_key][:v][tii]    = beta2.*adm[var_key][:v][tii] + (1.0-beta2).*clipped_grad.^2.0
                adm[var_key][:mhat][tii] = adm[var_key][:m][tii]/(1.0-beta1_decay)
                adm[var_key][:vhat][tii] = adm[var_key][:v][tii]/(1.0-beta2_decay)

                # update the states
                adm[var_key][:prev_stt][tii]  = copy(stt[var_key][tii])
                adm[var_key][:prev_grad][tii] = copy(clipped_grad) #copy(adm[var_key][:mhat][tii]./(sqrt.(adm[var_key][:vhat][tii]) .+ qG.eps))
                
                stt[var_key][tii] = stt[var_key][tii] - alpha*adm[var_key][:mhat][tii]./(sqrt.(adm[var_key][:vhat][tii]) .+ qG.eps)
            end
        end
    end

    # line search, now that we have the search directions
    run_ls = true
    alpha  = qG.alpha_0
    while run_ls
        z_0 = copy(scr[:zms_penalized])
        qG.eval_grad = false
        quasiGrad.update_states_and_grads!(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
        qG.eval_grad = true
        z_new = copy(scr[:zms_penalized])
        if z_new > z0
            # good, let's try a slightly larger step
            alpha = alpha*1.1
            # println("update!")
        else
            # bad, let's try a slightly smaller step
            alpha = alpha/1.1
            run_ls = false
        end
        for var_key in keys(mgd)
            for tii in prm.ts.time_keys                                          
                if var_key in keys(upd)
                    update_subset = upd[var_key][tii]
                    stt[var_key][tii][update_subset] = adm[var_key][:prev_stt][tii][update_subset] - alpha*adm[var_key][:prev_grad][tii][update_subset]
                else
                    stt[var_key][tii] = adm[var_key][:prev_stt][tii] - alpha*adm[var_key][:prev_grad][tii]
                end
            end
        end
    end
end

# adam solver -- take steps for every element in the master_grad list
function adaGrad!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, alpha::Float64, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    
    # loop over the keys in mgd
    for var_key in keys(mgd)

        # loop over all time
        for tii in prm.ts.time_keys
            # states to update                                            
            if var_key in keys(upd)
                # => (var_key in keys(upd)) ? (update = upd[var_key][tii]) : (update = Colon())
                #    the above caused weird type instability, so we just copy and paste
                update_subset = upd[var_key][tii]


                # update adam moments
                clipped_grad                         = clamp.(mgd[var_key][tii][update_subset],-qG.grad_max,qG.grad_max)
                adm[var_key][:v][tii][update_subset] = adm[var_key][:v][tii][update_subset] + clipped_grad.^2.0
                #stt[var_key][tii][update_subset]     = stt[var_key][tii][update_subset] - alpha*clipped_grad./(sqrt.(adm[var_key][:v][tii][update_subset]) .+ qG.eps)
                
                stt[var_key][tii][update_subset] = stt[var_key][tii][update_subset] - alpha*clipped_grad

            else 

                # update adam moments
                clipped_grad          = clamp.(mgd[var_key][tii],-qG.grad_max,qG.grad_max)
                adm[var_key][:v][tii] = adm[var_key][:v][tii] + clipped_grad.^2.0
                #stt[var_key][tii]     = stt[var_key][tii] - alpha*clipped_grad./(sqrt.(adm[var_key][:v][tii]) .+ qG.eps)
                
                stt[var_key][tii] = stt[var_key][tii] - alpha*clipped_grad

            end
        end
    end
end

# adam solver -- take steps for every element in the master_grad list
function the_quasiGrad!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # is this our first step?
    if qG.first_qG_step == true
        init_step = true
        # update
        qG.first_qG_step = false
    else
        init_step = false
    end

    # loop over the keys in mgd
    for var_key in keys(mgd)
        # loop over all time
        for tii in prm.ts.time_keys
            # states to update                                            
            if var_key in keys(upd)
                update_subset = upd[var_key][tii]

                if init_step == true
                    # new gradient
                    g_new  = clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max)

                    # update the previous gradient
                    adm[var_key][:prev_stt][tii][update_subset]  = copy(stt[var_key][tii][update_subset])
                    adm[var_key][:prev_grad][tii][update_subset] = copy(g_new)
                    
                    # take a step!
                    stt[var_key][tii][update_subset] = stt[var_key][tii][update_subset] - adm[var_key][:qG_step][tii][update_subset].*g_new
                else
                    # new gradient
                    g_new         = clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max)
                    abs_g_new     = abs.(g_new)
                    g_old         = adm[var_key][:prev_grad][tii][update_subset]
                    abs_g_old     = abs.(g_new)

                    # define scenarios
                    no_stt_change  = isapprox.(stt[var_key][tii][update_subset], adm[var_key][:prev_stt][tii][update_subset])
                    tiny_gradients = abs.(g_new) .< 1e-9
                    sign_change    = ((sign.(g_old) .!= sign.(g_new))) .&& (!).(tiny_gradients) .&& (!).(no_stt_change)
                    normal_cases   = (!).(tiny_gradients .|| sign_change .|| no_stt_change)
                    big_increase   = normal_cases .&& (10.0*abs_g_old .< abs_g_new)
                    increase       = normal_cases .&& (1.5*abs_g_old  .< abs_g_new) .&& (abs_g_new .< 10.0*abs_g_old)
                    small_change   = normal_cases .&& (0.9*abs_g_old  .< abs_g_new) .&& (abs_g_new .< 1.5*abs_g_old)
                    decrease       = normal_cases .&& (0.1*abs_g_old  .< abs_g_new) .&& (abs_g_new .< 0.9*abs_g_old)
                    big_decrease   = normal_cases .&& (abs_g_new  .< 0.1*abs_g_old)

                    # rule 0: if the graidient is tiny, do nothing
                    # rule 1: if the sign changed, drop step by an order of magnitude
                    # rule 3: if the sign didn't change, and the gradient is similar, increase step size
                    adm[var_key][:qG_step][tii][update_subset][sign_change]  = 0.1*adm[var_key][:qG_step][tii][update_subset][sign_change]
                    adm[var_key][:qG_step][tii][update_subset][big_increase] = 2.5*adm[var_key][:qG_step][tii][update_subset][big_increase]
                    adm[var_key][:qG_step][tii][update_subset][increase]     = 2.0*adm[var_key][:qG_step][tii][update_subset][increase]
                    adm[var_key][:qG_step][tii][update_subset][small_change] = 1.25*adm[var_key][:qG_step][tii][update_subset][small_change]
                    adm[var_key][:qG_step][tii][update_subset][decrease]     = 0.9*adm[var_key][:qG_step][tii][update_subset][decrease]
                    adm[var_key][:qG_step][tii][update_subset][big_decrease] = 0.25*adm[var_key][:qG_step][tii][update_subset][big_decrease]

                    # update the previous gradient
                    adm[var_key][:prev_stt][tii][update_subset]  = copy(stt[var_key][tii][update_subset])
                    adm[var_key][:prev_grad][tii][update_subset] = copy(g_new)

                    # take a step!
                    stt[var_key][tii][update_subset] = stt[var_key][tii][update_subset] - adm[var_key][:qG_step][tii][update_subset].*g_new
                end
            else
                # update all states
                if init_step == true
                    # new gradient
                    g_new = clamp.(mgd[var_key][tii],-qG.grad_max,qG.grad_max)

                    # update the previous gradient
                    adm[var_key][:prev_stt][tii]  = copy(stt[var_key][tii])
                    adm[var_key][:prev_grad][tii] = copy(g_new)
                    
                    # take a step!
                    stt[var_key][tii] = stt[var_key][tii] - adm[var_key][:qG_step][tii].*g_new
                else
                    # new gradient
                    g_new         = clamp.(mgd[var_key][tii], -qG.grad_max, qG.grad_max)
                    abs_g_new     = abs.(g_new)
                    g_old         = adm[var_key][:prev_grad][tii]
                    abs_g_old     = abs.(g_new)

                    # define scenarios
                    no_stt_change  = isapprox.(stt[var_key][tii], adm[var_key][:prev_stt][tii])
                    tiny_gradients = abs.(g_new) .< 1e-9
                    sign_change    = ((sign.(g_old) .!= sign.(g_new))) .&& (!).(tiny_gradients) .&& (!).(no_stt_change)
                    normal_cases   = (!).(tiny_gradients .|| sign_change .|| no_stt_change)
                    big_increase   = normal_cases .&& (10.0*abs_g_old .< abs_g_new)
                    increase       = normal_cases .&& (1.5*abs_g_old  .< abs_g_new) .&& (abs_g_new .< 10.0*abs_g_old)
                    small_change   = normal_cases .&& (0.9*abs_g_old  .< abs_g_new) .&& (abs_g_new .< 1.5*abs_g_old)
                    decrease       = normal_cases .&& (0.1*abs_g_old  .< abs_g_new) .&& (abs_g_new .< 0.9*abs_g_old)
                    big_decrease   = normal_cases .&& (abs_g_new  .< 0.1*abs_g_old)

                    # rule 0: if the graidient is tiny, do nothing
                    # rule 1: if the sign changed, drop step by an order of magnitude
                    # rule 3: if the sign didn't change, and the gradient is similar, increase step size
                    adm[var_key][:qG_step][tii][sign_change]  = 0.1*adm[var_key][:qG_step][tii][sign_change]
                    adm[var_key][:qG_step][tii][big_increase] = 2.5*adm[var_key][:qG_step][tii][big_increase]
                    adm[var_key][:qG_step][tii][increase]     = 2.0*adm[var_key][:qG_step][tii][increase]
                    adm[var_key][:qG_step][tii][small_change] = 1.25*adm[var_key][:qG_step][tii][small_change]
                    adm[var_key][:qG_step][tii][decrease]     = 0.9*adm[var_key][:qG_step][tii][decrease]
                    adm[var_key][:qG_step][tii][big_decrease] = 0.25*adm[var_key][:qG_step][tii][big_decrease]

                    # update the previous gradient
                    adm[var_key][:prev_stt][tii]  = copy(stt[var_key][tii])
                    adm[var_key][:prev_grad][tii] = copy(g_new)

                    # take a step!
                    stt[var_key][tii] = stt[var_key][tii] - adm[var_key][:qG_step][tii].*g_new
                end
            end
        end
    end
end