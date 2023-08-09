function update_penalties!(prm::quasiGrad.Param, qG::quasiGrad.QG, tnow::Float64, t0::Float64, tf::Float64)
    # this function scales the penalties associated with power balance, line flow
    # violation, and constraint penalties as a function of TIME.
    #
    #
    # depending on where we are between t0 and tf, compute a normalized
    # scalar value beta which acts as a homotopy parameter
    tnorm = 2.0*(tnow-t0)/(tf - t0) - 1.0 # scale between -1 and 1
    beta  = exp(6.0*tnorm)/(1.0 + exp(6.0*tnorm))
    
    # plot it:
    #   x = -1:0.01:1
    #   Plots.plot(x,exp.(7.0*x)./(3.0 .+ exp.(7.0*x)))

    # modify the epsilon parameters
    qG.pqbal_grad_eps2      = 1e-1 * (1.0-beta) + beta * 1e-5
    qG.constraint_grad_eps2 = 1e-1 * (1.0-beta) + beta * 1e-5
    qG.acflow_grad_eps2     = 1e-1 * (1.0-beta) + beta * 1e-5

    # also, update the constraint penalties!
    qG.pqbal_grad_weight_p    = (1.0*prm.vio.p_bus)*(1.0-beta) + beta*prm.vio.p_bus
    qG.pqbal_grad_weight_q    = (1.0*prm.vio.q_bus)*(1.0-beta) + beta*prm.vio.q_bus
    qG.constraint_grad_weight = (1.0*prm.vio.p_bus)*(1.0-beta) + beta*prm.vio.p_bus
    if prm.vio.s_flow < 1.5e3
        # just leave these
        qG.acflow_grad_weight = copy(prm.vio.s_flow)
    else
        # don't lessen these quite so much
        qG.acflow_grad_weight = (1.0*prm.vio.s_flow)*(1.0-beta)  + beta*prm.vio.s_flow
    end
end


function adam_step_decay!(qG::quasiGrad.QG, tnow::Float64, t0::Float64, tf::Float64)
    # depending on where we are between t0 and tf, compute a normalized
    # scalar value beta which acts as a homotopy parameter
    tnorm = 2.0*(tnow-t0)/(tf - t0) - 1.0 # scale between -1 and 1
    beta  = exp(7.0*tnorm)/(0.5 + exp(7.0*tnorm))

    # type of decay
    if qG.homotopy_with_cos_decay == true
        beta0 = 1.0 - exp(50.0*(tnorm+0.7))/(0.0001 + exp(50.0*(tnorm+0.7)))
        beta1 = exp(12.0*tnorm)/(0.0025 + exp(12.0*tnorm))
        beta2 = exp(25.0*tnorm)/(0.5    + exp(25.0*tnorm))
        beta3 = exp(12.0*tnorm)/(50.0   + exp(12.0*tnorm))
        beta  = beta0 + beta1 - beta2 + beta3
    else
        beta  = exp(7.0*tnorm)/(0.5 + exp(7.0*tnorm))
    end

    # loop and apply the updated steps
    for stp_key in keys(qG.alpha_tnow)
        qG.alpha_tnow[stp_key] = qG.alpha_t0[stp_key]*(1.0-beta) + qG.alpha_tf[stp_key]*beta
    end
end

# %%
using Plots

t0 = 1.0
tf = 100.04
tnow = t0:0.01:tf
tnorm = 2.0*(tnow .- t0)./(tf - t0) .- 1.0
beta0  = 1.0 .- exp.(50.0*(tnorm.+0.7))./(0.0001 .+ exp.(50.0*(tnorm.+0.7)))
beta1  = exp.(12.0*tnorm)./(0.0025 .+ exp.(12.0*tnorm))
beta2  = exp.(25.0*tnorm)./(0.5 .+ exp.(25.0*tnorm))
beta3  = exp.(12.0*tnorm)./(50.0 .+ exp.(12.0*tnorm))

Plots.plot(tnow, beta0)
Plots.plot!(tnow, beta1)
Plots.plot!(tnow, beta2)
Plots.plot!(tnow, beta3)

Plots.plot(tnow, beta0 + beta1 - beta2 + beta3)

#Plots.plot!(tnow, cos.(4.0*tnorm))