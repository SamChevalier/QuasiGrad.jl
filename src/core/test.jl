function jack_solves_adam_pf!(beta1::Float64, beta2::Float64, step_t0_dict::Dict{Symbol, Float64}, step_tf_dict::Dict{Symbol, Float64}, adam_max_time::Float64, p1::Float64, p2::Float64,  adm::QuasiGrad.Adam, cgd::QuasiGrad.ConstantGrad, ctg::QuasiGrad.Contingency, flw::QuasiGrad.Flow, grd::QuasiGrad.Grad, idx::QuasiGrad.Index, mgd::QuasiGrad.MasterGrad, ntk::QuasiGrad.Network, prm::QuasiGrad.Param, qG::QuasiGrad.QG, scr::Dict{Symbol, Float64}, stt::QuasiGrad.State, sys::QuasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}; clip_pq_based_on_bins::Bool=false)
    # here we go! basically, we only compute a small subset of pf-relevant gradients
    @info "Running adam-powerflow for $(qG.adam_max_time) seconds!"

    # initialize
    grad_vec, m, mhat, state_vec, step_size, v, vhat, step_tf, step_t0 = QuasiGrad.initialize_adam_states(step_t0_dict, step_tf_dict, prm, qG, stt, upd)

    # define some important parameter values
    t0                = time()
    tf                = t0 + adam_max_time
    adam_step_number  = 0
    keep_running_adam = true

    # save penalty values
    penality_values = zeros(Int(adam_max_time*2000))

    # loop over adam steps
    while keep_running_adam

        ######## ######## preliminaries ######## ########
        tnow = time()
        adam_step_number += 1
        QuasiGrad.update_penalties!(prm, qG, tnow, t0, tf)
        QuasiGrad.update_states_and_grads_for_adam_pf!(cgd, grd, idx, mgd, prm, qG, scr, stt, sys; clip_pq_based_on_bins=false)
        penality_values[adam_step_number] = scr[:zp] + scr[:zq] + scr[:acl] + scr[:xfm]
        ######## ######## preliminaries ######## ########

        # 1. adam step decay
        step_size = QuasiGrad.adam_step_COS_decay_test(p1, p2, step_size, step_tf, step_t0, tnow, t0, tf)
        #step_size = QuasiGrad.adam_step_decay_test(p1, p2, step_size, step_tf, step_t0, tnow, t0, tf)

        # 2. unpack the state and gradient vectors
        grad_vec, state_vec = QuasiGrad.unpack_grad_and_state_test(grad_vec, mgd, prm, qG, state_vec, stt, upd)

        # 3. take an adam step
        QuasiGrad.test_adam!(adam_step_number, beta1, beta2, grad_vec, m, mhat, state_vec, step_size, v, vhat)

        # 4. repack the state
        QuasiGrad.repack_state_test!(prm, qG, state_vec, stt, upd)

        # 5. should adam stop?
        if tnow > t0+adam_max_time
            keep_running_adam = false
        end

        # 6. add a safe point
        GC.safepoint()
    end

    # output
    return penality_values[1:adam_step_number]
end

function test_adam!(adam_step::Int64, beta1::Float64, beta2::Float64, grad::Vector{Float64}, m::Vector{Float64}, mhat::Vector{Float64}, state::Vector{Float64}, step_size::Vector{Float64}, v::Vector{Float64}, vhat::Vector{Float64})
    
    # now, apply adam
    @turbo @. m     = beta1*m + (1.0-beta1)*grad
    @turbo @. v     = beta2*v + (1.0-beta2)*(grad^2)
    @turbo @. mhat  = m/(1.0-beta1^adam_step)
    @turbo @. vhat  = v/(1.0-beta2^adam_step)
    @turbo @. state = state - step_size*mhat/(sqrt(vhat) + 1e-8)

    # output
    return state
end

function adam_step_decay_test(p1::Float64, p2::Float64, step_size::Vector{Float64}, step_tf::Vector{Float64}, step_t0::Vector{Float64}, tnow::Float64, t0::Float64, tf::Float64)
    # depending on where we are between t0 and tf, compute a normalized
    # scalar value beta which acts as a homotopy parameter

    # normalize time between -1 and +1
    tnorm = 2.0*(tnow-t0)/(tf - t0) - 1.0

    # define beta
    beta = exp(p1*tnorm)/(p2 + exp(p1*tnorm))

    # decay
    @. step_size = 10.0 ^ (-beta*log10(step_t0/step_tf) + log10(step_t0))

    # output
    return step_size
end

function adam_step_COS_decay_test(p1::Float64, p2::Float64, step_size::Vector{Float64}, step_tf::Vector{Float64}, step_t0::Vector{Float64}, tnow::Float64, t0::Float64, tf::Float64)
    # depending on where we are between t0 and tf, compute a normalized
    # scalar value beta which acts as a homotopy parameter

    # normalize time between -1 and +1
    tnorm = 2.0*(tnow-t0)/(tf - t0) - 1.0
    output = @. -cos(exp(-tnorm)) .+ 1

    # cos decay step
    @. step_size =  step_t0*output/1.91173   + step_tf*(1.91173-output)/1.91173

    # output
    return step_size
end


function unpack_grad_and_state_test(grad_vec::Vector{Float64}, mgd::QuasiGrad.MasterGrad, prm::QuasiGrad.Param, qG::QuasiGrad.QG, state_vec::Vector{Float64}, stt::QuasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}})
    index = [0]
    for var_key in qG.adam_pf_variables
        # call states and gradients
        state = getfield(stt, var_key)
        grad  = getfield(mgd, var_key)   
        for tii in prm.ts.time_keys
            if var_key in keys(upd)
                n_el = length(state[tii][upd[var_key][tii]])
                if n_el != 0
                    index = (index[end] + 1):(index[end] + n_el)
                    state_vec[index] = state[tii][upd[var_key][tii]]
                    grad_vec[index]  = grad[tii][upd[var_key][tii]]
                end
            else
                n_el = length(state[tii])
                if n_el != 0
                    index = (index[end] + 1):(index[end] + n_el)
                    state_vec[index] = state[tii]
                    grad_vec[index]  = grad[tii]
                end
            end
        end
    end

    return grad_vec, state_vec
end

function repack_state_test!(prm::QuasiGrad.Param, qG::QuasiGrad.QG, state_vec::Vector{Float64}, stt::QuasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}})
    index = [0]
    for var_key in qG.adam_pf_variables
        # call states and gradients
        state = getfield(stt, var_key)
        for tii in prm.ts.time_keys
            if var_key in keys(upd)
                n_el = length(state[tii][upd[var_key][tii]])
                if n_el != 0
                    index = (index[end] + 1):(index[end] + n_el)
                    state[tii][upd[var_key][tii]] = state_vec[index]
                end
            else
                n_el = length(state[tii])
                if n_el != 0
                    index = (index[end] + 1):(index[end] + n_el)
                    state[tii] = state_vec[index]
                end
            end
        end
    end
end

function initialize_adam_states(step_t0_dict::Dict{Symbol, Float64}, step_tf_dict::Dict{Symbol, Float64}, prm::QuasiGrad.Param, qG::QuasiGrad.QG, stt::QuasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}})
    total_el = 0
    for var_key in qG.adam_pf_variables
        # call states and gradients
        state = getfield(stt, var_key)
        for tii in prm.ts.time_keys
            if var_key in keys(upd)
                n_el      = length(state[tii][upd[var_key][tii]])
                total_el += n_el
            else
                n_el      = length(state[tii])
                total_el += n_el
            end
        end
    end

    m         = zeros(total_el)
    mhat      = zeros(total_el)
    v         = zeros(total_el)
    vhat      = zeros(total_el)
    grad_vec  = zeros(total_el)
    state_vec = zeros(total_el)
    step_size = zeros(total_el)
    step_t0   = zeros(total_el)
    step_tf   = zeros(total_el)

    # now, we need to populate step_tf and step_t0
    index = [0]
    for var_key in qG.adam_pf_variables
        # call states and gradients
        state  = getfield(stt, var_key)
        for tii in prm.ts.time_keys
            if var_key in keys(upd)
                n_el = length(state[tii][upd[var_key][tii]])
                if n_el != 0
                    index = (index[end] + 1):(index[end] + n_el)
                    step_t0[index] .= step_t0_dict[var_key]
                    step_tf[index] .= step_tf_dict[var_key]
                end
            else
                n_el = length(state[tii])
                if n_el != 0
                    index = (index[end] + 1):(index[end] + n_el)
                    step_t0[index] .= step_t0_dict[var_key]
                    step_tf[index] .= step_tf_dict[var_key]
                end
            end
        end
    end

    # output
    return grad_vec, m, mhat, state_vec, step_size, v, vhat, step_tf, step_t0
end
