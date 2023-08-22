"key: make sure step size decays faster/before the penalty tightening!
      otherwise, things might be a bit unstable! just a theory, though."

function update_penalties!(prm::quasiGrad.Param, qG::quasiGrad.QG, tnow::Float64, t0::Float64, tf::Float64)
    # this function scales the penalties associated with power balance, line flow
    # violation, and constraint penalties as a function of TIME.
    #
    # => x = -1:0.01:1
    # => Plots.plot(x,exp.(7.0*x)./(3.0 .+ exp.(7.0*x)))
    #
    # depending on where we are between t0 and tf, compute a normalized
    # scalar value beta which acts as a homotopy parameter
    tnorm = 2.0*(tnow-t0)/(tf - t0) - 1.0 # scale between -1 and 1
    beta  = exp(5.0*tnorm)/(1.0 + exp(5.0*tnorm))

    # modify the epsilon parameters
    eps2_t0                 = 1e-2
    eps2_tf                 = 1e-9
    log_eps2_ratio          = log10(eps2_t0/eps2_tf)
    esp2_update             = 10.0 ^ (-beta*log_eps2_ratio + log10(eps2_t0))
    qG.pqbal_grad_eps2      = esp2_update
    qG.constraint_grad_eps2 = esp2_update
    qG.acflow_grad_eps2     = esp2_update
    qG.ctg_grad_eps2        = esp2_update
    qG.reserve_grad_eps2    = esp2_update

    # alternative, linear update rule:
        # => qG.pqbal_grad_eps2      = 1e-2 * (1.0-beta) + beta * 1e-7
        # => qG.constraint_grad_eps2 = 1e-2 * (1.0-beta) + beta * 1e-7
        # => qG.acflow_grad_eps2     = 1e-2 * (1.0-beta) + beta * 1e-7
        # => qG.ctg_grad_eps2        = 1e-2 * (1.0-beta) + beta * 1e-7
        # => qG.reserve_grad_eps2    = 1e-2 * (1.0-beta) + beta * 1e-7

    # also, update the constraint penalties -- use linear growth:
    qG.pqbal_grad_weight_p    = (0.1*prm.vio.p_bus)*(1.0-beta)  + beta*prm.vio.p_bus
    qG.pqbal_grad_weight_q    = (0.1*prm.vio.q_bus)*(1.0-beta)  + beta*prm.vio.q_bus
    qG.constraint_grad_weight = (0.1*prm.vio.p_bus)*(1.0-beta)  + beta*prm.vio.p_bus
    qG.ctg_grad_weight        = (0.1*prm.vio.s_flow)*(1.0-beta) + beta*prm.vio.s_flow

    if prm.vio.s_flow < 1.5e3
        qG.acflow_grad_weight = copy(prm.vio.s_flow)
    else
        qG.acflow_grad_weight = (0.1*prm.vio.s_flow)*(1.0-beta)  + beta*prm.vio.s_flow
    end
end


function adam_step_decay!(qG::quasiGrad.QG, tnow::Float64, t0::Float64, tf::Float64; adam_pf::Bool=false)
    # depending on where we are between t0 and tf, compute a normalized
    # scalar value beta which acts as a homotopy parameter
    tnorm = 2.0*(tnow-t0)/(tf - t0) - 1.0 # scale between -1 and 1
    beta  = exp(4.0*tnorm)/(0.6 + exp(4.0*tnorm))

    if adam_pf == true
        # ***************** special case *****************
        for stp_key in qG.adam_pf_variables
            log_stp_ratio          = log10(qG.alpha_pf_t0[stp_key]/qG.alpha_pf_tf[stp_key])
            qG.alpha_tnow[stp_key] = 10.0 ^ (-beta*log_stp_ratio + log10(qG.alpha_pf_t0[stp_key]))
        end
    else
        # loop and compute the homotopy based on a log-transformation fall-off
        for stp_key in keys(qG.alpha_tnow)
            log_stp_ratio          = log10(qG.alpha_t0[stp_key]/qG.alpha_tf[stp_key])
            qG.alpha_tnow[stp_key] = 10.0 ^ (-beta*log_stp_ratio + log10(qG.alpha_t0[stp_key]))
        end
    end

    #= this is just a cleanup
    if adam_first_solve == true
        # loop and compute the homotopy based on a log-transformation fall-off
        #
        # order of magnitude+ smaller! (25)
        for stp_key in keys(qG.alpha_tnow)
            log_stp_ratio          = log10(qG.alpha_t0[stp_key]/qG.alpha_tf[stp_key])
            qG.alpha_tnow[stp_key] = 10.0 ^ (-beta*log_stp_ratio + log10(qG.alpha_t0[stp_key]/10.0))
        end
    else
        # loop and compute the homotopy based on a log-transformation fall-off
        for stp_key in keys(qG.alpha_tnow)
            log_stp_ratio          = log10(qG.alpha_t0[stp_key]/qG.alpha_tf[stp_key])
            qG.alpha_tnow[stp_key] = 10.0 ^ (-beta*log_stp_ratio + log10(qG.alpha_t0[stp_key]))
        end
    end
    =#

    # plotting:
        # => x = -1:0.01:1
        # => beta  = exp.(5.0.*x)./(0.5 .+ exp.(5.0.*x))
        # => log_stp_ratio          = log10(1e-1/1e-4)
        # => stp = -beta.*log_stp_ratio .+ log10.(1e-1)
        # => Plots.plot(x, stp)
        # => Plots.plot!(x, 10 .^ stp)

    # plotting:
        # => stp_0 = 1e-2
        # => stp_f = 1e-7
        # => ds    = stp_0/stp_f
        # => lds   = log10(ds)
        # => x = -1:0.01:1
        # => Plots.plot(x, -beta*lds .+ log10(stp_0))

    # type of decay
        # => if qG.homotopy_with_cos_decay == true
        # =>     beta0 = 1.0 - exp(50.0*(tnorm+0.7))/(0.0001 + exp(50.0*(tnorm+0.7)))
        # =>     beta1 = exp(12.0*tnorm)/(0.0025 + exp(12.0*tnorm))
        # =>     beta2 = exp(25.0*tnorm)/(0.5    + exp(25.0*tnorm))
        # =>     beta3 = exp(12.0*tnorm)/(50.0   + exp(12.0*tnorm))
        # =>     beta  = beta0 + beta1 - beta2 + beta3
        # => else
        # =>     beta  = exp(7.0*tnorm)/(0.5 + exp(7.0*tnorm))
        # => end
        # => # loop and apply the updated steps
        # => for stp_key in keys(qG.alpha_tnow)
        # =>     qG.alpha_tnow[stp_key] = qG.alpha_t0[stp_key]*(1.0-beta) + qG.alpha_tf[stp_key]*beta
        # => end

    # for plot testing
        # => using Plots
        # => t0    = 1.0
        # => tf    = 100.04
        # => tnow  = t0:0.01:tf
        # => tnorm = 2.0*(tnow .- t0)./(tf - t0) .- 1.0
        # => beta0 = 1.0 .- exp.(50.0*(tnorm.+0.7))./(0.0001 .+ exp.(50.0*(tnorm.+0.7)))
        # => beta1 = exp.(12.0*tnorm)./(0.0025 .+ exp.(12.0*tnorm))
        # => beta2 = exp.(25.0*tnorm)./(0.5 .+ exp.(25.0*tnorm))
        # => beta3 = exp.(12.0*tnorm)./(50.0 .+ exp.(12.0*tnorm))
        # => Plots.plot(tnow, beta0)
        # => Plots.plot!(tnow, beta1)
        # => Plots.plot!(tnow, beta2)
        # => Plots.plot!(tnow, beta3)
        # => Plots.plot(tnow, beta0 + beta1 - beta2 + beta3)
end