# Test #1 -- test if the power flow equations and gradients were coded up correctly :)
function test1_acline_flows_and_grads(prm::quasiGrad.Param, idx::quasiGrad.Idx, state::Dict{Symbol,Vector{Float64}}, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol,quasiGrad.SparseMatrixCSC{Float64, Int64}})
    vm          = [1.01; 0.95; 0.99];
    va          = [0.24; -0.2; 0.1];
    tau         = [1.01; 0.97]
    phi         = [0.13; -0.1]
    u_on_acline = [1.0; 1.0]
    u_on_xfm    = [1.0; 1.0]

    state = Dict(
        :vm          => vm,
        :va          => va,
        :tau         => tau,
        :phi         => phi,
        :u_on_acline => u_on_acline,
        :u_on_xfm    => u_on_xfm)

    # give some shunts values, for testing
    prm.acline.g_fr[1] = 0.1;
    prm.acline.g_to[1] = 0.02;
    prm.acline.b_fr[1] = 0.05;
    prm.acline.b_to[1] = 0.2;

    # call power flow
    acline_flows_test!(true, grad, jac, prm, state, idx)

    # ===================== test: do the flows make sense?
    r_series, x_series, b_ch, g_fr, b_fr, g_to, b_to, from_bus, to_bus = parse_json_acline_raw_params(json_data)

    # update
    g_fr[1] = 0.1;
    g_to[1] = 0.02;
    b_fr[1] = 0.05;
    b_to[1] = 0.2;

    p_fr = Float64[]
    q_fr = Float64[]
    s_fr = Float64[]
    p_to = Float64[]
    q_to = Float64[]
    s_to = Float64[]

    # loop
    for ii in 1:length(from_bus)
        z = r_series[ii] + im*x_series[ii]
        y = 1/z
        Yb = [(y+im*b_ch[ii]/2+g_fr[ii]+im*b_fr[ii])  -y;
            -y         y+im*b_ch[ii]/2+g_to[ii]+im*b_to[ii]]

        vfr   = vm[from_bus[ii]]*exp(im*va[from_bus[ii]])
        vto   = vm[to_bus[ii]]*exp(im*va[to_bus[ii]])
        iline = Yb*[vfr; vto]
        ifr   = iline[1]
        ito   = iline[2]
        push!(p_fr,real(vfr*conj(ifr)))
        push!(q_fr,imag(vfr*conj(ifr)))
        push!(s_fr,abs.(vfr*conj(ifr)))
        
        push!(p_to,real(vto*conj(ito)))
        push!(q_to,imag(vto*conj(ito)))
        push!(s_to,abs.(vto*conj(ito)))
    end

    # error
    max_flow_error = maximum(abs.([p_fr -  state[:acline_pfr]; q_fr -  state[:acline_qfr]; s_fr -  state[:acline_sfr]; p_to -  state[:acline_pto]; q_to -  state[:acline_qto]; s_to -  state[:acline_sto];]))

    # output
    @info "Maximum acline flow error: $max_flow_error"

    # ===================== test: now that the flows are correct, do the gradients make sense?
    #
    #
    #
    eps_val  = 1e-6
    states_0 = deepcopy(state)
    func_0   = deepcopy(state)
    grad_0   = deepcopy(grad)

    # dvmfr
    state = deepcopy(states_0)
    state[:vm][1] = state[:vm][1] + eps_val
    quasiGrad.update_acline_flows_and_grads!(true, grad, jac, prm, state, idx)
    df_dvmfr_num = ([state[:acline_pfr][1] state[:acline_qfr][1] state[:acline_pto][1] state[:acline_qto][1]] - 
                    [func_0[:acline_pfr][1] func_0[:acline_qfr][1] func_0[:acline_pto][1] func_0[:acline_qto][1]])/eps_val
    df_dvmfr     = [grad_0[:acline_dpfr_dvmfr][1] grad_0[:acline_dqfr_dvmfr][1] grad_0[:acline_dpto_dvmfr][1] grad_0[:acline_dqto_dvmfr][1]]

    # dvmto
    state = deepcopy(states_0)
    state[:vm][2] = state[:vm][2] + eps_val
    quasiGrad.update_acline_flows_and_grads!(true, grad, jac, prm, state, idx)
    df_dvmto_num = ([state[:acline_pfr][1] state[:acline_qfr][1] state[:acline_pto][1] state[:acline_qto][1]] - 
                    [func_0[:acline_pfr][1] func_0[:acline_qfr][1] func_0[:acline_pto][1] func_0[:acline_qto][1]])/eps_val
    df_dvmto     = [grad_0[:acline_dpfr_dvmto][1] grad_0[:acline_dqfr_dvmto][1] grad_0[:acline_dpto_dvmto][1] grad_0[:acline_dqto_dvmto][1]]

    # dvafr
    state = deepcopy(states_0)
    state[:va][1] = state[:va][1] + eps_val
    quasiGrad.update_acline_flows_and_grads!(true, grad, jac, prm, state, idx)
    df_dvafr_num = ([state[:acline_pfr][1] state[:acline_qfr][1] state[:acline_pto][1] state[:acline_qto][1]] - 
                    [func_0[:acline_pfr][1] func_0[:acline_qfr][1] func_0[:acline_pto][1] func_0[:acline_qto][1]])/eps_val
    df_dvafr     = [grad_0[:acline_dpfr_dvafr][1] grad_0[:acline_dqfr_dvafr][1] grad_0[:acline_dpto_dvafr][1] grad_0[:acline_dqto_dvafr][1]]

    # dvafr
    state = deepcopy(states_0)
    state[:va][2] = state[:va][2] + eps_val
    quasiGrad.update_acline_flows_and_grads!(true, grad, jac, prm, state, idx)
    df_dvato_num = ([state[:acline_pfr][1] state[:acline_qfr][1] state[:acline_pto][1] state[:acline_qto][1]] - 
                    [func_0[:acline_pfr][1] func_0[:acline_qfr][1] func_0[:acline_pto][1] func_0[:acline_qto][1]])/eps_val
    df_dvato     = [grad_0[:acline_dpfr_dvato][1] grad_0[:acline_dqfr_dvato][1] grad_0[:acline_dpto_dvato][1] grad_0[:acline_dqto_dvato][1]]

    # duon
    state = deepcopy(states_0)
    state[:u_on_acline][1] = state[:u_on_acline][1] + eps_val
    quasiGrad.update_acline_flows_and_grads!(true, grad, jac, prm, state, idx)
    df_duon_num = ([state[:acline_pfr][1] state[:acline_qfr][1] state[:acline_pto][1] state[:acline_qto][1]] - 
                   [func_0[:acline_pfr][1] func_0[:acline_qfr][1] func_0[:acline_pto][1] func_0[:acline_qto][1]])/eps_val
    df_duon     = [grad_0[:acline_dpfr_duon][1] grad_0[:acline_dqfr_duon][1] grad_0[:acline_dpto_duon][1] grad_0[:acline_dqto_duon][1]]
 
    # s_fr derivative -- this is implicit, so let's do it manually
    state[:acline_pfr] = randn(2)
    state[:acline_qfr] = randn(2)
    state[:acline_pto] = randn(2)
    state[:acline_qto] = randn(2)
    state[:acline_sfr] = sqrt.(state[:acline_pfr].^2 + state[:acline_qfr].^2)
    state[:acline_sto] = sqrt.(state[:acline_pto].^2 + state[:acline_qto].^2)
    grad[:acline_dsfr_dpfr]  = state[:acline_pfr]./state[:acline_sfr]
    grad[:acline_dsfr_dqfr]  = state[:acline_qfr]./state[:acline_sfr]
    grad[:acline_dsto_dpto]  = state[:acline_pto]./state[:acline_sto]
    grad[:acline_dsto_dqto]  = state[:acline_qto]./state[:acline_sto]

    func_0  = deepcopy(state)
    state[:acline_pfr][1] = state[:acline_pfr][1] + eps_val
    state[:acline_qto][1] = state[:acline_qto][1] + eps_val
    state[:acline_sfr] = sqrt.(state[:acline_pfr].^2 + state[:acline_qfr].^2)
    state[:acline_sto] = sqrt.(state[:acline_pto].^2 + state[:acline_qto].^2)
    df_dpq_num = ([state[:acline_sfr][1] state[:acline_sto][1]] - [func_0[:acline_sfr][1] func_0[:acline_sto][1]])/eps_val
    df_dpq     =  [grad[:acline_dsfr_dpfr][1] grad[:acline_dsto_dqto][1]]     

    # error
    max_grad_error = maximum(abs.(hcat(df_dvmfr_num  - df_dvmfr,
                                        df_dvmto_num - df_dvmto,
                                        df_dvafr_num - df_dvafr,
                                        df_dvato_num - df_dvato,
                                        df_duon_num  - df_duon,
                                        df_dpq_num   - df_dpq)))

    # maximum gradient error (single line)
    @info "Maximum acline grad error (single line): $max_grad_error"
end

# Test #2 -- xfm
function test2_xfm_flows_and_grads(prm::quasiGrad.Param, idx::quasiGrad.Idx, state::Dict{Symbol,Vector{Float64}}, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol,quasiGrad.SparseMatrixCSC{Float64, Int64}})
    vm          = [1.01; 0.95; 0.99];
    va          = [0.24; -0.2; 0.1];
    tau         = [1.01; 0.97]
    phi         = [0.13; -0.1]
    u_on_acline = [1.0; 1.0]
    u_on_xfm    = [1.0; 1.0]

    state = Dict(
        :vm          => vm,
        :va          => va,
        :tau         => tau,
        :phi         => phi,
        :u_on_acline => u_on_acline,
        :u_on_xfm    => u_on_xfm)

    # give some shunts values, for testing
    prm.xfm.g_fr[1] = 0.1;
    prm.xfm.g_to[1] = 0.02;
    prm.xfm.b_fr[1] = 0.05;
    prm.xfm.b_to[1] = 0.2;

    # call power flow
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)

    # ===================== test: do the flows make sense?
    r_series, x_series, b_ch, g_fr, b_fr, g_to, b_to, from_bus, to_bus = parse_json_xfm_raw_params(json_data)

    # update
    g_fr[1] = 0.1;
    g_to[1] = 0.02;
    b_fr[1] = 0.05;
    b_to[1] = 0.2;

    p_fr = Float64[]
    q_fr = Float64[]
    s_fr = Float64[]
    p_to = Float64[]
    q_to = Float64[]
    s_to = Float64[]

    # loop
    for ii in 1:length(from_bus)
        z = r_series[ii] + im*x_series[ii]
        y = 1/z
        c = tau[ii]*exp(im*phi[ii])
        Yb = [(y+im*b_ch[ii]/2+g_fr[ii]+im*b_fr[ii])/(abs(c)^2)  -y/conj(c);
            -y/c         y+im*b_ch[ii]/2+g_to[ii]+im*b_to[ii]]

        vfr   = vm[from_bus[ii]]*exp(im*va[from_bus[ii]])
        vto   = vm[to_bus[ii]]*exp(im*va[to_bus[ii]])
        iline = Yb*[vfr; vto]
        ifr   = iline[1]
        ito   = iline[2]
        push!(p_fr,real(vfr*conj(ifr)))
        push!(q_fr,imag(vfr*conj(ifr)))
        push!(s_fr,abs.(vfr*conj(ifr)))
        push!(p_to,real(vto*conj(ito)))
        push!(q_to,imag(vto*conj(ito)))
        push!(s_to,abs.(vto*conj(ito)))
    end

    # error
    max_flow_error = maximum(abs.([p_fr -  state[:xfm_pfr]; q_fr -  state[:xfm_qfr]; s_fr -  state[:xfm_sfr]; p_to -  state[:xfm_pto]; q_to -  state[:xfm_qto]; s_to -  state[:xfm_sto];]))

    # output
    @info "Maximum xfm flow error: $max_flow_error"

    # ===================== test: now that the flows are correct, do the gradients make sense?
    #
    #
    #
    eps_val  = 1e-6
    states_0 = deepcopy(state)
    func_0  = deepcopy(state)
    grad_0  = deepcopy(grad)

    # dvmfr
    state = deepcopy(states_0)
    state[:vm][2] = state[:vm][2] + eps_val
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    df_dvmfr_num = ([state[:xfm_pfr][1] state[:xfm_qfr][1] state[:xfm_pto][1] state[:xfm_qto][1]] - 
                    [func_0[:xfm_pfr][1] func_0[:xfm_qfr][1] func_0[:xfm_pto][1] func_0[:xfm_qto][1]])/eps_val
    df_dvmfr     = [grad_0[:xfm_dpfr_dvmfr][1] grad_0[:xfm_dqfr_dvmfr][1] grad_0[:xfm_dpto_dvmfr][1] grad_0[:xfm_dqto_dvmfr][1]]

    # dvmto
    state = deepcopy(states_0)
    state[:vm][3] = state[:vm][3] + eps_val
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    df_dvmto_num = ([state[:xfm_pfr][1] state[:xfm_qfr][1] state[:xfm_pto][1] state[:xfm_qto][1]] - 
                    [func_0[:xfm_pfr][1] func_0[:xfm_qfr][1] func_0[:xfm_pto][1] func_0[:xfm_qto][1]])/eps_val
    df_dvmto     = [grad_0[:xfm_dpfr_dvmto][1] grad_0[:xfm_dqfr_dvmto][1] grad_0[:xfm_dpto_dvmto][1] grad_0[:xfm_dqto_dvmto][1]]

    # dvafr
    state = deepcopy(states_0)
    state[:va][2] = state[:va][2] + eps_val
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    df_dvafr_num = ([state[:xfm_pfr][1] state[:xfm_qfr][1] state[:xfm_pto][1] state[:xfm_qto][1]] - 
                    [func_0[:xfm_pfr][1] func_0[:xfm_qfr][1] func_0[:xfm_pto][1] func_0[:xfm_qto][1]])/eps_val
    df_dvafr     = [grad_0[:xfm_dpfr_dvafr][1] grad_0[:xfm_dqfr_dvafr][1] grad_0[:xfm_dpto_dvafr][1] grad_0[:xfm_dqto_dvafr][1]]

    # dvato
    state = deepcopy(states_0)
    state[:va][3] = state[:va][3] + eps_val
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    df_dvato_num = ([state[:xfm_pfr][1] state[:xfm_qfr][1] state[:xfm_pto][1] state[:xfm_qto][1]] - 
                    [func_0[:xfm_pfr][1] func_0[:xfm_qfr][1] func_0[:xfm_pto][1] func_0[:xfm_qto][1]])/eps_val
    df_dvato     = [grad_0[:xfm_dpfr_dvato][1] grad_0[:xfm_dqfr_dvato][1] grad_0[:xfm_dpto_dvato][1] grad_0[:xfm_dqto_dvato][1]]

    # dtau
    state = deepcopy(states_0)
    state[:tau][1] = state[:tau][1] + eps_val
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    df_dtau_num = ([state[:xfm_pfr][1] state[:xfm_qfr][1] state[:xfm_pto][1] state[:xfm_qto][1]] - 
                   [func_0[:xfm_pfr][1] func_0[:xfm_qfr][1] func_0[:xfm_pto][1] func_0[:xfm_qto][1]])/eps_val
    df_dtau     = [grad_0[:xfm_dpfr_dtau][1] grad_0[:xfm_dqfr_dtau][1] grad_0[:xfm_dpto_dtau][1] grad_0[:xfm_dqto_dtau][1]]

    # dphi
    state = deepcopy(states_0)
    state[:phi][1] = state[:phi][1] + eps_val
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    df_dphi_num = ([state[:xfm_pfr][1] state[:xfm_qfr][1] state[:xfm_pto][1] state[:xfm_qto][1]] - 
                   [func_0[:xfm_pfr][1] func_0[:xfm_qfr][1] func_0[:xfm_pto][1] func_0[:xfm_qto][1]])/eps_val
    df_dphi     = [grad_0[:xfm_dpfr_dphi][1] grad_0[:xfm_dqfr_dphi][1] grad_0[:xfm_dpto_dphi][1] grad_0[:xfm_dqto_dphi][1]]

    # duon
    state = deepcopy(states_0)
    state[:u_on_xfm][1] = state[:u_on_xfm][1] + eps_val
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    df_duon_num = ([state[:xfm_pfr][1] state[:xfm_qfr][1] state[:xfm_pto][1] state[:xfm_qto][1]] - 
                   [func_0[:xfm_pfr][1] func_0[:xfm_qfr][1] func_0[:xfm_pto][1] func_0[:xfm_qto][1]])/eps_val
    df_duon     = [grad_0[:xfm_dpfr_duon][1] grad_0[:xfm_dqfr_duon][1] grad_0[:xfm_dpto_duon][1] grad_0[:xfm_dqto_duon][1]]
    
    # s_fr derivative -- this is implicit, so let's do it manually
    state[:xfm_pfr] = randn(2)
    state[:xfm_qfr] = randn(2)
    state[:xfm_pto] = randn(2)
    state[:xfm_qto] = randn(2)
    state[:xfm_sfr] = sqrt.(state[:xfm_pfr].^2 + state[:xfm_qfr].^2)
    state[:xfm_sto] = sqrt.(state[:xfm_pto].^2 + state[:xfm_qto].^2)
    grad[:xfm_dsfr_dpfr]  = state[:xfm_pfr]./state[:xfm_sfr]
    grad[:xfm_dsfr_dqfr]  = state[:xfm_qfr]./state[:xfm_sfr]
    grad[:xfm_dsto_dpto]  = state[:xfm_pto]./state[:xfm_sto]
    grad[:xfm_dsto_dqto]  = state[:xfm_qto]./state[:xfm_sto]

    func_0  = deepcopy(state)
    state[:xfm_pfr][1] = state[:xfm_pfr][1] + eps_val
    state[:xfm_qto][1] = state[:xfm_qto][1] + eps_val
    state[:xfm_sfr] = sqrt.(state[:xfm_pfr].^2 + state[:xfm_qfr].^2)
    state[:xfm_sto] = sqrt.(state[:xfm_pto].^2 + state[:xfm_qto].^2)
    df_dpq_num = ([state[:xfm_sfr][1] state[:xfm_sto][1]] - [func_0[:xfm_sfr][1] func_0[:xfm_sto][1]])/eps_val
    df_dpq     =  [grad[:xfm_dsfr_dpfr][1] grad[:xfm_dsto_dqto][1]]     

    # error
    max_grad_error = maximum(abs.(hcat(df_dvmfr_num - df_dvmfr,
                                       df_dvmto_num - df_dvmto,
                                       df_dvafr_num - df_dvafr,
                                       df_dvato_num - df_dvato,
                                       df_dtau_num  - df_dtau,
                                       df_dphi_num  - df_dphi,
                                       df_duon_num  - df_duon,
                                       df_dpq_num   - df_dpq)))

    # maximum gradient error (single xfm )
    @info "Maximum grad error: $max_grad_error"
end

# Test #3 -- adam test: solve for power flows (lines + xfms)
function test3_adam_solve_pf(prm::quasiGrad.Param, idx::quasiGrad.Idx, state::Dict{Symbol,Vector{Float64}}, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol,quasiGrad.SparseMatrixCSC{Float64, Int64}}, N_steps::Int64, dt::Float64, adam_prm::Dict{Symbol, Float64})
    vm          = [1.0; 1.01; 0.97];
    va          = [0.0; -0.025; 0.02];
    tau         = [1.025; 1.0]
    phi         = [0.0; 0.015]
    u_on_acline = [1.0; 1.0]
    u_on_xfm    = [1.0; 1.0]

    state = Dict(
        :vm          => vm,
        :va          => va,
        :tau         => tau,
        :phi         => phi,
        :u_on_acline => u_on_acline,
        :u_on_xfm    => u_on_xfm)

    # initial flows
    quasiGrad.update_acline_flows_and_grads!(true, state, grad, jac, prm, state, idx)
    quasiGrad.update_xfm_flows_and_grads!(true, state, grad, jac, prm, state, idx)

    # save the solution target
    func_0   = deepcopy(state)
    states_0 = deepcopy(state)

    # test loss gradient
    state[:vm]  = [1.0; 1.0; 1.0]
    state[:va]  = [0.0; 0.0; 0.0]
    state[:tau] = [1.0; 1.0]
    state[:phi] = [0.0; 0.0]
    test3_subtest_loss_grad(state, grad, jac, prm, state, idx, func_0)

    adam_states = Dict(
        :adam_step   => [0.0],
        :alpha_decay => [adam_prm[:alpha]],
        :GO_states   => vcat([1.0; 1.0; 1.0], [0.0; 0.0; 0.0], [1.0; 1.0], [0.0; 0.0]),
        :m           => zeros(10),
        :v           => zeros(10),
        :mhat        => zeros(10),
        :vhat        => zeros(10))

    # what parameters do we update?
    idx[:adam][:update] = [2; 3; 5; 6; 7; 10] 
    
    # log results -- initialize
    grad_log  = zeros(N_steps,10)
    state_log = zeros(N_steps,10)
    loss_log  = zeros(N_steps)

    # loop
    for ii in 1:N_steps

        # call the state
        ########
        state[:vm]  = adam_states[:GO_states][idx[:varstack][:t1][:vm]]
        state[:va]  = adam_states[:GO_states][idx[:varstack][:t1][:va]]
        state[:tau] = adam_states[:GO_states][idx[:varstack][:t1][:tau]]
        state[:phi] = adam_states[:GO_states][idx[:varstack][:t1][:phi]]

        #state[:phi]  = adam_states[:GO_states][9:10]

        # flows
        quasiGrad.acline_flows!(true, grad, jac, prm, state, idx)
        quasiGrad.xfm_flows!(true, grad, jac, prm, state, idx)

        # define loss and its grad
        loss, loss_grad = test3_loss_and_grad(func_0, state, grad)
        if ii%25 == 0
            println(loss)
        end
        sleep(dt)

        # log the results
        grad_log[ii,:]  = loss_grad
        state_log[ii,:] = adam_states[:GO_states]
        loss_log[ii]    = loss

        # adam step
        adam_states = quasiGrad.adam(adam_prm, adam_states, loss_grad, idx)
    
    end

    # output
    return grad_log, state_log, loss_log, adam_states
end

# define the loss and grad for test 3
function test3_loss_and_grad(func_0, state, grad)
    loss_val = sum((func_0[:acline_pfr]  - state[:acline_pfr]).^2 + 
                (func_0[:acline_qfr] - state[:acline_qfr]).^2 + 
                (func_0[:acline_pto] - state[:acline_pto]).^2 +
                (func_0[:acline_qto] - state[:acline_qto]).^2 +
                (func_0[:xfm_pfr]    - state[:xfm_pfr]).^2 +
                (func_0[:xfm_qfr]    - state[:xfm_qfr]).^2 +
                (func_0[:xfm_pto]    - state[:xfm_pto]).^2 +
                (func_0[:xfm_qto]    - state[:xfm_qto]).^2)

    # build the partials
    loss_grad_partial_acline = 2*[(func_0[:acline_pfr] - state[:acline_pfr])*(-1);
                                  (func_0[:acline_qfr] - state[:acline_qfr])*(-1); 
                                  (func_0[:acline_pto] - state[:acline_pto])*(-1);
                                  (func_0[:acline_qto] - state[:acline_qto])*(-1)]
    loss_grad_partial_xfm = 2*[(func_0[:xfm_pfr]  - state[:xfm_pfr])*(-1);
                                (func_0[:xfm_qfr] - state[:xfm_qfr])*(-1);
                                (func_0[:xfm_pto] - state[:xfm_pto])*(-1);
                                (func_0[:xfm_qto] - state[:xfm_qto])*(-1)]

    # full gradient
    loss_grad = jac[:acline][:,1:10]'*loss_grad_partial_acline + jac[:xfm][:,1:10]'*loss_grad_partial_xfm

    return loss_val, loss_grad
end

# does the jacobian work? grad = -2*J'*f..
function test3_subtest_loss_grad(state, grad, jac, prm, idx, func_0)

    # test the loss gradient
    quasiGrad.acline_flows!(true, grad, jac, prm, state, idx)
    quasiGrad.xfm_flows!(true, grad, jac, prm, state, idx)

    # get the loss value and gradient
    loss_val, loss_grad = test3_loss_and_grad(func_0, state, grad)

    # copy
    states_n0   = deepcopy(state)
    loss_val_n0 = deepcopy(loss_val)

    # numerical gradient -- single entry
    eps_val           = 1e-8
    states_n0[:vm][2] = states_n0[:vm][2] + eps_val

    # flows
    quasiGrad.update_acline_flows_and_grads!(true, grad, jac, prm, states_n0, idx)
    quasiGrad.update_xfm_flows_and_grads!(true, grad, jac, prm, states_n0, idx)
    
    # new loss value
    loss_val_n, ~ = test3_loss_and_grad(func_0, state, grad)

    # print -- single entry
    print("Gradient error: ")
    println((loss_val_n - loss_val_n0)/eps_val - loss_grad[2])
    println("================")

end

# ac line flows
function acline_flows_test!(eval_grad::Bool, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol, quasiGrad.SparseMatrixCSC{Float64, Int64}}, prm::quasiGrad.Param, state::Dict{Symbol,Vector{Float64}}, idx::quasiGrad.Idx)
    # line parameters
    g_sr = prm.acline.g_sr
    b_sr = prm.acline.b_sr
    b_ch = prm.acline.b_ch
    g_fr = prm.acline.g_fr
    b_fr = prm.acline.b_fr
    g_to = prm.acline.g_to
    b_to = prm.acline.b_to

        # call statuses
        u_on_lines = state[:u_on_acline]

        # organize relevant line values
        vm_fr      = state[:vm][idx.acline_fr_bus]
        va_fr      = state[:va][idx.acline_fr_bus]
        vm_to      = state[:vm][idx.acline_to_bus]
        va_to      = state[:va][idx.acline_to_bus]
        
        # tools
        cos_ftp  = cos.(va_fr - va_to)
        sin_ftp  = sin.(va_fr - va_to)
        vff      = vm_fr.^2
        vtt      = vm_to.^2
        vft      = vm_fr.*vm_to 
        
        # evaluate the function? we always need to in order to get the grad
        #
        # active power flow -- from -> to
        state[:acline_pfr] = u_on_lines.*((g_sr+g_fr).*vff + 
                (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft)
        
        # reactive power flow -- from -> to
        state[:acline_qfr] = u_on_lines.*((-b_sr-b_fr-b_ch/2).*vff +
                (b_sr.*cos_ftp - g_sr.*sin_ftp).*vft)

        # apparent power flow -- to -> from
        state[:acline_sfr] = sqrt.(state[:acline_pfr].^2 + state[:acline_qfr].^2)
        
        # active power flow -- to -> from
        state[:acline_pto] = u_on_lines.*((g_sr+g_to).*vtt + 
                (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft)
        
        # reactive power flow -- to -> from
        state[:acline_qto] = u_on_lines.*((-b_sr-b_to-b_ch/2).*vtt +
                (b_sr.*cos_ftp + g_sr.*sin_ftp).*vft)

        # apparent power flow -- to -> from
        state[:acline_sto] = sqrt.(state[:acline_pto].^2 + state[:acline_qto].^2)
        
        # ====================================================== #
        # ====================================================== #
        #
        # evaluate the grad?
        if eval_grad == true
                # Gradients: active power flow -- from -> to
                grad[:acline_dpfr_dvmfr] = u_on_lines.*(2*(g_sr+g_fr).*vm_fr + 
                        (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_to)
                grad[:acline_dpfr_dvmto] = u_on_lines.*(
                        (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_fr)
                grad[:acline_dpfr_dvafr] = u_on_lines.*(
                        (g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
                grad[:acline_dpfr][:vato] = u_on_lines.*(
                        (-g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
                grad[:acline_dpfr][:uon] = state[:acline_pfr]

                # ====================================================== #
                # Gradients: reactive power flow -- from -> to
                grad[:acline_qfr][:vmfr] = u_on_lines.*(2*(-b_sr-b_fr-b_ch/2).*vm_fr +
                        (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_to)
                grad[:acline_qfr][:vmto] = u_on_lines.*(
                        (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_fr)
                grad[:acline_qfr][:vafr] = u_on_lines.*(
                        (-b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
                grad[:acline_qfr][:vato] = u_on_lines.*(
                        (b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
                grad[:acline_qfr][:uon]  = state[:acline_qfr]

                # ====================================================== #
                # Gradients: apparent power flow -- from -> to
                grad[:acline_sfr][:pfr]  = state[:acline_pfr]./state[:acline_sfr]
                grad[:acline_sfr][:qfr]  = state[:acline_qfr]./state[:acline_sfr]

                # ====================================================== #
                # Gradients: active power flow -- to -> from
                grad[:acline_pto][:vmfr] = u_on_lines.*( 
                        (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_to)
                grad[:acline_pto][:vmto] = u_on_lines.*(2*(g_sr+g_to).*vm_to +
                        (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_fr)
                grad[:acline_pto][:vafr] = u_on_lines.*(
                        (g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
                grad[:acline_pto][:vato] = u_on_lines.*(
                        (-g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
                grad[:acline_pto][:uon] = state[:acline_pto]

                # ====================================================== #
                # Gradients: reactive power flow -- to -> from
                grad[:acline_qto][:vmfr] = u_on_lines.*(
                        (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_to)
                grad[:acline_qto][:vmto] = u_on_lines.*(2*(-b_sr-b_to-b_ch/2).*vm_to +
                        (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_fr)
                grad[:acline_qto][:vafr] = u_on_lines.*(
                        (-b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
                grad[:acline_qto][:vato] = u_on_lines.*(
                        (b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
                grad[:acline_qto][:uon] = state[:acline_qto]

                # ====================================================== #
                # Gradients: apparent power flow -- to -> from
                grad[:acline_sto][:pto]  = state[:acline_pto]./state[:acline_sto]
                grad[:acline_sto][:qto]  = state[:acline_qto]./state[:acline_sto]

                # now, what we really want is the gradients accociated with a single
                # given variable -- place them into a sparse Jacobian matrix
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dpfr_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dpfr_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dpfr_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dpfr_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dqfr_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dqfr_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dqfr_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dqfr_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dpto_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dpto_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dpto_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dpto_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dqto_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dqto_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dqto_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dqto_dvato]
        end
end

# perturb, clip, and fix states!
function perturb!(stt, prm, idx, grd, sys, qG, gamma)
    # we pertub, clip and fix states in the mastergrad
    for tii in prm.ts.time_keys
        stt[:vm][tii]  = 1.0 .+ gamma*0.1*randn(sys.nb)
        stt[:va][tii]  = gamma*0.1*randn(sys.nb)
        stt[:tau][tii] = 1.0 .+ gamma*0.1*randn(sys.nx)
        stt[:phi][tii]          = gamma*0.1*randn(sys.nx)        
        stt[:dc_pfr][tii]       = gamma*0.1*rand(sys.nldc)    
        stt[:dc_qfr][tii]       = gamma*0.1*rand(sys.nldc)       
        stt[:dc_qto][tii]       = gamma*0.1*rand(sys.nldc)  
        stt[:u_on_acline][tii]  = 0*gamma*rand(sys.nl)     +  ones(sys.nl)
        stt[:u_on_xfm][tii]     = 0*gamma*rand(sys.nx)     +  ones(sys.nx)
        stt[:u_step_shunt][tii] = gamma*rand(sys.nsh) 
        stt[:u_on_dev][tii]     = 0*gamma*rand(sys.ndev)   +  ones(sys.ndev)
        stt[:p_on][tii]         = gamma*rand(sys.ndev)    
        stt[:dev_q][tii]        = gamma*rand(sys.ndev)  
        stt[:p_rgu][tii]        = gamma*rand(sys.ndev) 
        stt[:p_rgd][tii]        = gamma*rand(sys.ndev)   
        stt[:p_scr][tii]        = gamma*rand(sys.ndev)      
        stt[:p_nsc][tii]        = gamma*rand(sys.ndev)    
        stt[:p_rru_on][tii]     = gamma*rand(sys.ndev)      
        stt[:p_rrd_on][tii]     = gamma*rand(sys.ndev)      
        stt[:p_rru_off][tii]    = gamma*rand(sys.ndev)  
        stt[:p_rrd_off][tii]    = gamma*rand(sys.ndev)     
        stt[:q_qru][tii]        = gamma*rand(sys.ndev)    
        stt[:q_qrd][tii]        = gamma*rand(sys.ndev) 
    end
end

# define a function which perturbs -- don't clip!
function calc_nzms_qG(grd, idx, mgd, prm, qG, stt, sys)

        # compute states and grads
        quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
    
        # output
        return scr[:nzms]
end

# define a function which perturbs -- don't clip!
function calc_nzms(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
    # if we are here, we want to make sure we are running su/sd updates
    qG.run_susd_updates = true

    # flush the gradient -- both master grad and some of the gradient terms
    quasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

    # don't clip!!
    
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
    @info "ctg solve on"
    quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
    # @info "ctg solve off"
    
    # score the market surplus function
    quasiGrad.score_zt!(idx, prm, qG, scr, stt) 
    quasiGrad.score_zbase!(qG, scr)
    quasiGrad.score_zms!(scr)

    # compute the master grad
    quasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)
    # output

    return -scr[:zms_penalized] # previously => scr[:nzms]
end

# solve, and then return the solution
function grb_speed(GRB)
    tkeys = [Symbol("t"*string(ii)) for ii in 1:42]
    for tii in tkeys
        GRB[:u_on_dev][tii]  = GRB[:u_on_dev][tii].^2 .+ 1.6
        GRB[:p_on][tii]      = GRB[:p_on][tii].^2 .+ 1.6      
        GRB[:dev_q][tii]     = GRB[:dev_q][tii].^2 .+ 1.6
        GRB[:p_rgu][tii]     = GRB[:p_rgu][tii].^2 .+ 1.6     
        GRB[:p_rgd][tii]     = GRB[:p_rgd][tii].^2 .+ 1.6     
        GRB[:p_scr][tii]     = GRB[:p_scr][tii].^2 .+ 1.6     
        GRB[:p_nsc][tii]     = GRB[:p_nsc][tii].^2 .+ 1.6     
        GRB[:p_rru_on][tii]  = GRB[:p_rru_on][tii].^2 .+ 1.6  
        GRB[:p_rru_off][tii] = GRB[:p_rru_off][tii].^2 .+ 1.6 
        GRB[:p_rrd_on][tii]  = GRB[:p_rrd_on][tii].^2 .+ 1.6  
        GRB[:p_rrd_off][tii] = GRB[:p_rrd_off][tii].^2 .+ 1.6 
        GRB[:q_qru][tii]     = GRB[:q_qru][tii].^2 .+ 1.6     
        GRB[:q_qrd][tii]     = GRB[:q_qrd][tii].^2 .+ 1.6     
    end
end

# solve, and then return the solution
function grb_speed_dict(GRB::Dict)
    tkeys = [Symbol("t"*string(ii)) for ii in 1:42]
    for tii in tkeys
        GRB[:u_on_dev][tii]  = GRB[:u_on_dev][tii].^2 .+ 1.6
        GRB[:p_on][tii]      = GRB[:p_on][tii].^2 .+ 1.6      
        GRB[:dev_q][tii]     = GRB[:dev_q][tii].^2 .+ 1.6
        GRB[:p_rgu][tii]     = GRB[:p_rgu][tii].^2 .+ 1.6     
        GRB[:p_rgd][tii]     = GRB[:p_rgd][tii].^2 .+ 1.6     
        GRB[:p_scr][tii]     = GRB[:p_scr][tii].^2 .+ 1.6     
        GRB[:p_nsc][tii]     = GRB[:p_nsc][tii].^2 .+ 1.6     
        GRB[:p_rru_on][tii]  = GRB[:p_rru_on][tii].^2 .+ 1.6  
        GRB[:p_rru_off][tii] = GRB[:p_rru_off][tii].^2 .+ 1.6 
        GRB[:p_rrd_on][tii]  = GRB[:p_rrd_on][tii].^2 .+ 1.6  
        GRB[:p_rrd_off][tii] = GRB[:p_rrd_off][tii].^2 .+ 1.6 
        GRB[:q_qru][tii]     = GRB[:q_qru][tii].^2 .+ 1.6     
        GRB[:q_qrd][tii]     = GRB[:q_qrd][tii].^2 .+ 1.6     
    end
end

# solve, and then return the solution
function grb_speed_typed(GRB::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    tkeys = [Symbol("t"*string(ii)) for ii in 1:42]
    for tii in tkeys
        GRB[:u_on_dev][tii]  = GRB[:u_on_dev][tii].^2 .+ 1.6
        GRB[:p_on][tii]      = GRB[:p_on][tii].^2 .+ 1.6      
        GRB[:dev_q][tii]     = GRB[:dev_q][tii].^2 .+ 1.6
        GRB[:p_rgu][tii]     = GRB[:p_rgu][tii].^2 .+ 1.6     
        GRB[:p_rgd][tii]     = GRB[:p_rgd][tii].^2 .+ 1.6     
        GRB[:p_scr][tii]     = GRB[:p_scr][tii].^2 .+ 1.6     
        GRB[:p_nsc][tii]     = GRB[:p_nsc][tii].^2 .+ 1.6     
        GRB[:p_rru_on][tii]  = GRB[:p_rru_on][tii].^2 .+ 1.6  
        GRB[:p_rru_off][tii] = GRB[:p_rru_off][tii].^2 .+ 1.6 
        GRB[:p_rrd_on][tii]  = GRB[:p_rrd_on][tii].^2 .+ 1.6  
        GRB[:p_rrd_off][tii] = GRB[:p_rrd_off][tii].^2 .+ 1.6 
        GRB[:q_qru][tii]     = GRB[:q_qru][tii].^2 .+ 1.6     
        GRB[:q_qrd][tii]     = GRB[:q_qrd][tii].^2 .+ 1.6     
    end
end

function load_and_project(path::String, solution_file::String)
    InFile1 = path
    jsn = quasiGrad.load_json(InFile1)

    # initialize
    adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn)

    # solve
    fix       = true
    pct_round = 100.0
    quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
    quasiGrad.write_solution(solution_file, prm, qG, stt, sys)
    quasiGrad.post_process_stats(true, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
end