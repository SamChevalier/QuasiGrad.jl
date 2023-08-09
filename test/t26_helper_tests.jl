using quasiGrad
using Revise

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00014/scenario_003.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
jsn  = quasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);

# run copper plate ED
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt)
qG.num_threads = 10

# %% step 1: cleanup tests
qG.skip_ctg_eval = true

stt = deepcopy(stt0)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
@time quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% step 2: projection tests
stt = deepcopy(stt0)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

#qG.print_projection_success = false

@time quasiGrad.project!(50.0, idx, prm, qG, stt, sys, upd, final_projection = false)

# %%
@time quasiGrad.apply_Gurobi_projection_and_states!(idx, prm, qG, stt, sys)

# %%
@time quasiGrad.batch_fix!(51.0, prm, stt, sys, upd);

# %% step 3: projection tests
stt = deepcopy(stt0)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

# %% step 4: snap shunts test
fix = solver_itr == (n_its-1)
quasiGrad.snap_shunts!(fix, prm, qG, stt, upd)

qG.num_threads = 10

# %% ===========
stt = deepcopy(stt0)
@time GC.gc()
@time quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% =======

@time quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
#@time quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)


# %% stt0 = deepcopy(stt)
qG.max_linear_pfs       = 6
qG.max_linear_pfs_total = 6

stt = deepcopy(stt0)
quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)


# %%
#quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
#quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# initialize plot
plt = Dict(:plot            => false,
           :first_plot      => true,
           :N_its           => 150,
           :global_adm_step => 0,
           :disp_freq       => 5)

# initialize
ax, fig, z_plt  = quasiGrad.initialize_plot(cgd, ctg, flw, grd, idx, mgd, ntk, plt, prm, qG, scr, stt, sys)

qG.print_freq                  = 10
qG.num_threads                 = 10
qG.apply_grad_weight_homotopy  = false
qG.take_adam_pf_steps          = false
qG.pqbal_grad_type             = "soft_abs" 
qG.pqbal_grad_eps2             = 1e-1

qG.constraint_grad_is_soft_abs = true
qG.acflow_grad_is_soft_abs     = true
qG.reserve_grad_is_soft_abs    = true
qG.skip_ctg_eval               = true

qG.beta1                       = 0.9
qG.beta2                       = 0.99

#qG.beta1                       = 0.98
#qG.beta2                       = 0.995

qG.pqbal_grad_eps2             = 1e-3
qG.constraint_grad_eps2        = 1e-3
qG.acflow_grad_eps2            = 1e-3
qG.reserve_grad_eps2           = 1e-3
qG.ctg_grad_eps2               = 1e-3
qG.adam_max_time               = 250.0
qG.apply_grad_weight_homotopy  = false
qG.decay_adam_step             = false

quasiGrad.run_adam_with_plotting!(adm, ax, cgd, ctg, fig, flw, grd, idx, mgd, 
                                  ntk, plt, prm, qG, scr, stt, sys, upd, z_plt)

# %% ============
pct_round = 0.0
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)

# %% ===================
stt = deepcopy(stt0)
#quasiGrad.project!(50.0, idx, prm, qG, stt, sys, upd, final_projection = false)

quasiGrad.apply_q_injections!(idx, prm, qG, stt, sys)
quasiGrad.solve_parallel_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

#qG.num_lbfgs_steps = 2000

#adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);
#quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)
# %%

quasiGrad.cleanup_constrained_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

# %%
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
# %% 
quasiGrad.solve_economic_dispatch!(idx, prm, qG, scr, stt, sys, upd, include_sus_in_ed=true)

tii = 5
dt  = prm.ts.duration[tii]
dev_p_previous = stt.dev_p[4]

gam = stt.dev_p[5] .- stt.dev_p[4] .- dt.*(prm.dev.p_ramp_up_ub.*(stt.u_on_dev[tii] .- stt.u_su_dev[tii]) .+ prm.dev.p_startup_ramp_ub.*(stt.u_su_dev[tii] .+ 1.0 .- stt.u_on_dev[tii]))

sort(gam)

# %%
usd = quasiGrad.solve_economic_dispatch!(idx, prm, qG, scr, stt, sys, upd, include_sus_in_ed=true)

# %%
tii = 5
dt  = prm.ts.duration[tii]

for dev in 1:sys.ndev
    dev_p_previous = stt.dev_p[4][dev]
    gg = stt.dev_p[tii][dev] - dev_p_previous - dt*(prm.dev.p_ramp_up_ub[dev]*(stt.u_on_dev[tii][dev] - stt.u_su_dev[tii][dev]) + prm.dev.p_startup_ramp_ub[dev]*(stt.u_su_dev[tii][dev] + 1.0 - stt.u_on_dev[tii][dev]))
    #if gg > 0.0
        println(gg)
    #end
end

# %% ==========
tii = Int8(1)
quasiGrad.update_Yflow!(idx, ntk, prm, stt, sys, tii)

# %%
tii = Int8(1)
stt.phi[tii] = 0.1*randn(144)
stt.tau[tii] = 1.0 .+ 0.1*randn(144)

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.update_Yflow!(idx, ntk, prm, stt, sys, tii)

#ntk.Yflow_fr_real[tii] .= ntk.Yflow_acline_series_real .+ ntk.Yflow_xfm_series_fr_real[tii] .+ ntk.Yflow_acline_shunt_fr_real .+ ntk.Yflow_xfm_shunt_fr_real[tii]
#ntk.Yflow_fr_imag[tii] .= ntk.Yflow_acline_series_imag .+ ntk.Yflow_xfm_series_fr_imag[tii] .+ ntk.Yflow_acline_shunt_fr_imag .+ ntk.Yflow_xfm_shunt_fr_imag[tii]
#ntk.Yflow_to_real[tii] .= -ntk.Yflow_acline_series_real .+ ntk.Yflow_xfm_series_to_real[tii] .+ ntk.Yflow_acline_shunt_to_real .+ ntk.Yflow_xfm_shunt_to_real[tii]
#ntk.Yflow_to_imag[tii] .= -ntk.Yflow_acline_series_imag .+ ntk.Yflow_xfm_series_to_imag[tii] .+ ntk.Yflow_acline_shunt_to_imag .+ ntk.Yflow_xfm_shunt_to_imag[tii]

Vph   = stt.vm[tii].*exp.(im*stt.va[tii])
pq_fr = (ntk.Efr*Vph).*conj((ntk.Yflow_fr_real[tii] + im*ntk.Yflow_fr_imag[tii])*Vph)
pq_to = (ntk.Eto*Vph).*conj((ntk.Yflow_to_real[tii] + im*ntk.Yflow_to_imag[tii])*Vph)

println(maximum(abs.(real(pq_fr[1:sys.nl]) - stt.acline_pfr[tii])))
println(maximum(abs.(imag(pq_fr[1:sys.nl]) - stt.acline_qfr[tii])))

println(maximum(abs.(real(pq_to[1:sys.nl]) - stt.acline_pto[tii])))
println(maximum(abs.(imag(pq_to[1:sys.nl]) - stt.acline_qto[tii])))

println(maximum(abs.(real(pq_fr[sys.nl+1:end]) - stt.xfm_pfr[tii])))
println(maximum(abs.(imag(pq_fr[sys.nl+1:end]) - stt.xfm_qfr[tii])))

println(maximum(abs.(real(pq_to[sys.nl+1:end]) - stt.xfm_pto[tii])))
println(maximum(abs.(imag(pq_to[sys.nl+1:end]) - stt.xfm_qto[tii])))

# %% build the Jacobian
ntk.Yflow_fr_real[tii] 
ntk.Yflow_fr_imag[tii] 
ntk.Yflow_to_real[tii] 
ntk.Yflow_to_imag[tii] 

# build the admittance structure
NYf = [ntk.Yflow_fr_real[tii] -ntk.Yflow_fr_imag[tii];
      -ntk.Yflow_fr_imag[tii] -ntk.Yflow_fr_real[tii]]


stt.Ir_flow_fr[tii] .= ntk.Yflow_fr_real[tii]*stt.vr[tii] - ntk.Yflow_fr_imag[tii]*stt.vi[tii]
stt.Ii_flow_fr[tii] .= ntk.Yflow_fr_imag[tii]*stt.vr[tii] + ntk.Yflow_fr_real[tii]*stt.vi[tii]

# Populate MI
MIr = quasiGrad.spdiagm(sys.nb, sys.nb, stt.Ir_flow_fr[tii])*ntk.Efr
MIi = quasiGrad.spdiagm(sys.nb, sys.nb, stt.Ii_flow_fr[tii])*ntk.Efr
MI  = [MIr MIi;
        -MIi MIr]

# Populate MV
MVr = quasiGrad.spdiagm(sys.nb, sys.nb, ntk.Efr*stt.vr[tii])
MVi = quasiGrad.spdiagm(sys.nb, sys.nb, ntk.Efr*stt.vi[tii])
MV  = [MVr -MVi;
       MVi  MVr]

# Populate RV
RV = [quasiGrad.spdiagm(sys.nb, sys.nb, stt.cva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, .-stt.vi[tii]); 
      quasiGrad.spdiagm(sys.nb, sys.nb, stt.sva[tii])  quasiGrad.spdiagm(sys.nb, sys.nb, stt.vr[tii])];
       
# Build Jacobian
Jpqf = (MI + MV*NYf)*RV   
      

stt.


    # Compute Currents and Voltages
    Icf = (YLE + YSEx)*Vc
    Ir  = real(Icf)
    Ii  = imag(Icf)
    Scf = (Ex*Vc).*conj(Icf)
    Sf  = abs.(Scf)
    Pf  = real(Scf)
    Qf  = imag(Scf)
    cTh = cos.(Theta)
    sTh = sin.(Theta)
    Vr  = real(Vc)
    Vi  = imag(Vc)

    # Populate RV
    RV = [Diagonal(cTh) Diagonal(-Vi); Diagonal(sTh) Diagonal(Vr)];

    # Populate MV
    MVr = Diagonal(Ex*Vr)
    MVi = Diagonal(Ex*Vi)
    MV  = [MVr -MVi;
           MVi  MVr]

    # Populate MI
    MIr = Diagonal(Ir)*Ex
    MIi = Diagonal(Ii)*Ex
    MI  = [MIr MIi;
          -MIi MIr]

    # Build Jacobian
    stt.Jac_pq_flow_fr[tii] = (MI + MV*NYf)*RV

    # first, populate V -> S
    stt.Jac_flow_fr[tii][:,1:sys.nb] = 

    # second, populate Th -> S

    # Build apparent power Jacobian
    Sf[Sf .== 0] .= 1 # This is arbitrary, since Pf/Qf = 0
    J = 0.5*Diagonal(1 ./Sf)*[2*Diagonal(Pf)*Jpqf[1:nF,1:nB]+2*Diagonal(Qf)*Jpqf[nF+1:end,1:nB] 2*Diagonal(Pf)*Jpqf[1:nF,nB+1:end]+2*Diagonal(Qf)*Jpqf[nF+1:end,nB+1:end]]

    if reduce == 1
          # Update if reduction is helpful
    end

    return J



# %%
 # ntk.Yd_acline_series_real .+ ntk.Yflow_xfm_series_fr_real[tii] .+ ntk.Yflow_acline_shunt_fr_real .+ ntk.Yflow_xfm_shunt_fr_real[tii]
 # ntk.Yd_acline_series_imag .+ ntk.Yflow_xfm_series_fr_imag[tii] .+ ntk.Yflow_acline_shunt_fr_imag .+ ntk.Yflow_xfm_shunt_fr_imag[tii]
 # ntk.Yd_acline_series_real .+ ntk.Yflow_xfm_series_to_real[tii] .+ ntk.Yflow_acline_shunt_to_real .+ ntk.Yflow_xfm_shunt_to_real[tii]
 # ntk.Yd_acline_series_imag .+ ntk.Yflow_xfm_series_to_imag[tii] .+ ntk.Yflow_acline_shunt_to_imag .+ ntk.Yflow_xfm_shunt_to_imag[tii]

# %% =============

ntk.Yflow_fr_real[tii] + im*ntk.Yflow_fr_imag[tii]
ntk.Yflow_fr_imag[tii]
    
ntk.Yflow_to_real[tii]
ntk.Yflow_to_imag[tii]

# %% step 1: compute p/q flows
ntk.E*(stt.vm[tii].*exp.(im*stt.va[tii]))

# %% Apparent Power Flow Jacobian
function PF_JacobianSf(Vm,Theta,Vc,YLE,YSEx,Ex,NYf,reduce)

    # Compute Currents and Voltages
    Icf = (YLE + YSEx)*Vc
    Ir  = real(Icf)
    Ii  = imag(Icf)
    Scf = (Ex*Vc).*conj(Icf)
    Sf  = abs.(Scf)
    Pf  = real(Scf)
    Qf  = imag(Scf)
    cTh = cos.(Theta)
    sTh = sin.(Theta)
    Vr  = real(Vc)
    Vi  = imag(Vc)

    # Populate RV
    RV = [Diagonal(cTh) Diagonal(-Vi); Diagonal(sTh) Diagonal(Vr)];

    # Populate MV
    MVr = Diagonal(Ex*Vr)
    MVi = Diagonal(Ex*Vi)
    MV  = [MVr -MVi;
           MVi  MVr]

    # Populate MI
    MIr = Diagonal(Ir)*Ex
    MIi = Diagonal(Ii)*Ex
    MI  = [MIr MIi;
          -MIi MIr]

    # Build Jacobian
    Jpqf = (MI + MV*NYf)*RV
    nB = length(Theta)
    nF = length(Sf)

    # Build apparent power Jacobian
    Sf[Sf .== 0] .= 1 # This is arbitrary, since Pf/Qf = 0
    J = 0.5*Diagonal(1 ./Sf)*[2*Diagonal(Pf)*Jpqf[1:nF,1:nB]+2*Diagonal(Qf)*Jpqf[nF+1:end,1:nB] 2*Diagonal(Pf)*Jpqf[1:nF,nB+1:end]+2*Diagonal(Qf)*Jpqf[nF+1:end,nB+1:end]]

    if reduce == 1
          # Update if reduction is helpful
    end

    return J
end

# %%
quasiGrad.project!(10.0, idx, prm, qG, stt, sys, upd, final_projection = false)

# E7. write the final solution
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)

# %% E8. post process
quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% test the flow jacobians :)
tii = Int8(1)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.update_Ybus!(idx, ntk, prm, stt, sys, tii)
quasiGrad.update_Yflow!(idx, ntk, prm, stt, sys, tii)
quasiGrad.build_Jac_and_pq0!(ntk, qG, stt, sys, tii)
quasiGrad.build_Jac_sfr_and_sfr0!(idx, ntk, prm, stt, sys, tii)

# %%
s0 = copy(stt.xfm_sfr[tii])

# perturb
epsilon_val = 1e-5
stt.vm[tii][1413] = stt.vm[tii][1413] + epsilon_val

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.update_Ybus!(idx, ntk, prm, stt, sys, tii)
quasiGrad.update_Yflow!(idx, ntk, prm, stt, sys, tii)
quasiGrad.build_Jac_and_pq0!(ntk, qG, stt, sys, tii)

# %%
quasiGrad.build_Jac_sfr_and_sfr0!(idx, ntk, prm, stt, sys, tii)
# %%
snew = copy(stt.xfm_sfr[tii])

grad_num  = (snew[end] - s0[end])/epsilon_val
grad_true = ntk.Jac_sflow_fr[tii][2371,1413]

println(grad_num)
println(grad_true)

# %%
g .= stt.pflow_over_sflow_fr[tii].*ntk.Jac_pq_flow_fr[tii][1:sys.nac      , 1:sys.nb] .+ stt.qflow_over_sflow_fr[tii].*ntk.Jac_pq_flow_fr[tii][(sys.nac+1):end, 1:sys.nb]

hh = stt.pflow_over_sflow_fr[tii].*ntk.Jac_pq_flow_fr[tii][1:sys.nac      , (sys.nb+1):end] .+ stt.qflow_over_sflow_fr[tii].*ntk.Jac_pq_flow_fr[tii][(sys.nac+1):end, (sys.nb+1):end]