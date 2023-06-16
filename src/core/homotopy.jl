function update_penalties!(prm::quasiGrad.Param, qG::quasiGrad.QG, tnow::Float64, t0::Float64, tf::Float64)
    # this function scales the penalties associated with power balance, line flow
    # violation, and constraint penalties as a function of TIME.
    #
    #
    # depending on where we are between t0 and tf, compute a normalized
    # scalar value beta which acts as a homotopy parameter
    tnorm = 2.0*(tnow-t0)/(tf - t0) - 1.0 # scale between -1 and 1
    beta  = exp(7.0*tnorm)/(3.0 + exp(7.0*tnorm))
    
    # plot it:
    #   x = -1:0.01:1
    #   Plots.plot(x,exp.(8.0*x)./(2.5 .+ exp.(8.0*x)))

    # modify the epsilon parameters
    qG.pqbal_grad_eps2      = 1.0 * (1.0-beta) + beta * 1e-5
    qG.constraint_grad_eps2 = 1.0 * (1.0-beta) + beta * 1e-4
    qG.acflow_grad_eps2     = 1.0 * (1.0-beta) + beta * 1e-4

    # also, update the constraint penalties!
    qG.pqbal_grad_weight_p    = (1e-2*prm.vio.p_bus)*(1.0-beta) + beta*prm.vio.p_bus
    qG.pqbal_grad_weight_q    = (1e-2*prm.vio.q_bus)*(1.0-beta) + beta*prm.vio.q_bus
    qG.constraint_grad_weight = (1e-2*prm.vio.p_bus)*(1.0-beta) + beta*prm.vio.p_bus
    if prm.vio.s_flow < 1.5e3
        # just leave these
        qG.acflow_grad_weight = copy(prm.vio.s_flow)
    else
        # don't lessen these quite so much
        qG.acflow_grad_weight = (5e-2*prm.vio.s_flow)*(1.0-beta)  + beta*prm.vio.s_flow
    end
end