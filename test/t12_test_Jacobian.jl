using quasiGrad
using Revise

# load the json
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"

# call
jsn = quasiGrad.load_json(path)

# %% init
adm, cgd, flw, GRB, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, 
sys, upd, dz_dpinj_base, theta_k_base, worst_ctgs = quasiGrad.base_initialization(jsn, false, 1.0);

# solve
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

# run an ED
ED = quasiGrad.solve_economic_dispatch(GRB, idx, prm, qG, scr, stt, sys, upd);
quasiGrad.apply_economic_dispatch_projection!(ED, idx, prm, stt);

# ===== new score?
quasiGrad.dcpf_initialization!(flw, idx, msc, ntk, prm, qG, stt, sys)

# %%
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

# %% let's go!
quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# %% intialize lbfgs
#dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf = quasiGrad.initialize_pf_lbfgs(mgd, prm, stt, sys, upd);

#t#ii = :t1
#qG.cdist_psolve = 1.0
#residual = zeros(2*sys.nb)
#quasiGrad.power_flow_residual!(idx, residual, stt, sys, tii)

#for tii in prm.ts.time_keys
#    stt[:va][tii] .= 0.0
#end

# %% loop -- lbfgs
qG.initial_pf_lbfgs_step = 0.1
dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf = quasiGrad.initialize_pf_lbfgs(mgd, prm, stt, sys, upd);

for ii in 1:10000
    # take an lbfgs step
    quasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, stt, upd, zpf)

    # save zpf BEFORE updating with the new state
    for tii in prm.ts.time_keys
        pf_lbfgs_step[:zpf_prev][tii] = (zpf[:zp][tii]+zpf[:zq][tii]) 
    end

    # compute all states and grads
    quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)

    # print
    zp = round(sum(sum([zpf[:zp][tii] for tii in prm.ts.time_keys])); sigdigits = 3)
    zq = round(sum(sum([zpf[:zq][tii] for tii in prm.ts.time_keys])); sigdigits = 3)
    stp = round(sum(pf_lbfgs_step[:step][tii] for tii in prm.ts.time_keys)/sys.nT; sigdigits = 5)
    println("Total: $(zp+zq) P penalty: $(zp), Q penalty: $(zq), average adam step: $(stp)!")

    #println(stt[:vm][:t1][5])
    #println(sum(stt[:p_on][:t1]))
    #println(sum(stt[:dev_q][:t1]))
    #println(sum(stt[:vm][:t1]))
    #println(sum(stt[:va][:t1]))
    #println(sum(stt[:phi][:t1]))
    #println("===========================================")
    #sleep(0.25)
end
bv
# %% ===========
qG.max_linear_pfs = 1

#ProfileView.@profview
quasiGrad.solve_linear_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

# %% ============= 
tii = :t1
@code_warntype quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

# %%
@code_warntype quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

# %%
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

@code_warntype quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);

# %% ===========
#
tii = :t1

# 1. update the ideal dispatch point (active power)
quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

# 2. update the injection bounds
quasiGrad.get_injection_bounds!(idx, msc, prm, stt, sys, tii)

# 3. update y_bus and Jacobian and bias point
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

# Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag)

# %% =====
qG.Gurobi_pf_obj = "min_dispatch_distance"
#qG.Gurobi_pf_obj = "min_dispatch_perturbation"
Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
quasiGrad.solve_linear_pf_with_Gurobi!(Jac, msc, prm, qG, stt, sys, tii);

# %% now, solve Newton-based power flow
pi_p, pi_q, PQidx = quasiGrad.slack_factors(idx, prm, stt, sys, tii)
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)
residual = zeros(2*sys.nb)
tii      = :t1
KP       = 0.0
KP = quasiGrad.solve_power_flow(grd, idx, KP, pi_p, prm, PQidx, qG, residual, stt, sys, tii, Ybus_real, Ybus_imag)
quasiGrad.apply_pq_injections!(idx, prm, qG, stt, sys, tii)

# %%

quasiGrad.power_flow_residual!(idx, 0.0, pi_p, residual, stt, sys, tii)
println(quasiGrad.norm(residual[residual_idx]))


# %% ==================
#
# turn of clipping to test this!!!
residual = zeros(2*sys.nb)

test_pf = true
if test_pf == true
    tii = :t1
    stt[:phi][tii] =        0.1*randn(sys.nx)
    stt[:tau][tii] = 1.0 .+ 0.1*randn(sys.nx)
    quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, dz_dpinj_base, theta_k_base, worst_ctgs)

    # compute the admittance
    Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

    # Does the Ybus work?
    Yb = Ybus_real + im*Ybus_imag

    # complex voltage
    vc = stt[:vm][tii].*(exp.(im*stt[:va][tii]))
    ic = Yb*vc
    sc = vc.*conj.(ic)

    # compare
    pb = zeros(sys.nb)
    qb = zeros(sys.nb)
    # loop over each bus
    for bus in 1:sys.nb
        # active power balance: stt[:pb][:slack][tii][bus] to record with time
        pb[bus] = 
                # shunt
                sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt[:acline_pfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt[:acline_pto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt[:xfm_pfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt[:xfm_pto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0)
        
        # reactive power balance
        qb[bus] = 
                # shunt
                sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt[:acline_qfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt[:acline_qto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt[:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt[:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0) 
    end

    # result
    println("error:")
    println(maximum(pb - (real.(sc))))
    println(maximum(qb - (imag.(sc))))

    # ------- numerically test jacobian -- keep clipping off!
    for ind in 1:sys.nb
        Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
        quasiGrad.power_flow_residual!(idx, residual, stt, sys, tii)

        Jac0      = copy(Jac)
        residial0 = copy(residual)
        eps       = 1e-6

        stt[:vm][tii][ind] = stt[:vm][tii][ind] + eps
        quasiGrad.update_states_for_distributed_slack_pf!(grd, idx, prm, qG, stt)
        quasiGrad.power_flow_residual!(idx, residual, stt, sys, tii)

        # test
        t1 = (residual - residial0)/eps
        t2 = Jac0[:,ind]
        @info quasiGrad.norm(t1-t2)
    end
end

# %% solve power flow ==============================
#
# initialize residual
tii = :t1
pi_p, pi_q, PQidx = quasiGrad.slack_factors(idx, prm, stt, sys, tii)
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)
residual = zeros(2*sys.nb)
KP = 0.0
KQ = 0.0

#quasiGrad.solve_power_flow(grd, idx, KP, KQ, pi_p, pi_q, prm, qG, residual, stt, sys, tii, Ybus_real, Ybus_imag)

#pi_p .= 1.0/sys.nb
#pi_q .= 1.0/sys.nb

# build the state
# %% 
#x = copy([KQ; stt[:vm][tii][2:end]; KP; stt[:va][tii][2:end]])
x = copy([KP; stt[:va][tii][2:end]])

# loop over each bus and compute the residual
quasiGrad.power_flow_residual!(idx, KP, KQ, pi_p, pi_q, residual, stt, sys, tii)

#  test the residual for termination
if quasiGrad.norm(residual[1:sys.nb]) < 1e-5
    run_pf = false
else
    println(quasiGrad.norm(residual[1:sys.nb]))
    #sleep(0.75)

    # update the Jacobian
    Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)
    #quasiGrad.transform_acpf_Jac!(Jac, pi_p, pi_q, sys)
    
    Jac = Jac[1:sys.nb, sys.nb+1:end]
    # 2. remove phase from from ref buses and add active power distributed slack
    Jac[:,1] .= 1.0 ./ sys.nb

    # take a Newton step
    x = x - (Jac\residual[1:sys.nb])
    #x = x - (Jac\residual)

    # update the state
    #KQ = x[1]
    #stt[:vm][tii][2:end] = x[2:sys.nb]
    #KP = x[sys.nb+1]
    #stt[:va][tii][2:end] = x[sys.nb+2:end]
    KP = x[1]
    stt[:va][tii][2:end] = x[2:end]

    # update the flows and residual and such
    quasiGrad.update_states_for_distributed_slack_pf!(grd, idx, prm, qG, stt)
end







# %% test jacobian
# %%
pi_p .= 1.0/sys.nb
pi_q .= 1.0/sys.nb

# build the state
x = copy([KQ; stt[:vm][tii][2:end]; KP; stt[:va][tii][2:end]])

residual = zeros(2*sys.nb)
KP = 0.0
KQ = 0.0
pi_p .= 1.0/sys.nb
pi_q .= 1.0/sys.nb

# loop over each bus and compute the residual
quasiGrad.power_flow_residual!(idx, KP, KQ, pi_p, pi_q, residual, stt, sys, tii)
Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)

# %%
#  test the residual for termination
if quasiGrad.norm(residual) < 1e-1
    run_pf = false
else
    println(quasiGrad.norm(residual))
    #sleep(0.75)

    # update the Jacobian
    Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)
    quasiGrad.transform_acpf_Jac!(Jac, pi_p, pi_q, sys)

    # update the state
    KQ = x[1]
    stt[:vm][tii][2:end] = x[2:sys.nb]
    KP = x[sys.nb+1]
    stt[:va][tii][2:end] = x[sys.nb+2:end]

    # update the flows and residual and such
    quasiGrad.update_states_for_distributed_slack_pf!(grd, idx, prm, qG, stt)
end


# %% %%%%%%%%%%%%%%%%%%%%%%%%%
# initialize residual
tii = :t1
pi_p, pi_q, PQidx = quasiGrad.slack_factors(idx, prm, stt, sys, tii)
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)
residual = zeros(2*sys.nb)
KP = 0.0
KQ = 0.0

# %%
quasiGrad.power_flow_residual!(idx, 0.0, pi_p, residual, stt, sys, tii)
println(quasiGrad.norm(residual[residual_idx]))

# ==========

# start with all buses as PV if they have Q capacity
buses = collect(1:sys.nb)
# => alternative: PQidx  = buses[isapprox.(Qub,Qlb,atol=1e-6) || isapprox.(Qub,Qinj,atol=1e-6) || isapprox.(Qlb,Qinj,atol=1e-6)]
PVidx  = setdiff(buses, PQidx)
nPQ    = length(PQidx)
residual_idx = [buses;           # => P
               sys.nb .+ PQidx]  # => Q
# %% note => ref = 1, but it is a PV bus :)

# build the state
x = [stt[:vm][tii][PQidx]; KP; stt[:va][tii][2:end]]
println(x[1:3])

# loop over each bus and compute the residual
quasiGrad.power_flow_residual!(idx, KP, pi_p, residual, stt, sys, tii)

# test the residual for termination
if quasiGrad.norm(residual[residual_idx]) < 1e-1
    run_pf = false
else
    println(quasiGrad.norm(residual[residual_idx]))

    # update the Jacobian
    Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)
    Jac = quasiGrad.transform_acpf_Jac(Jac, pi_p, PQidx, sys)

    # take a Newton step -- do NOT put the scalar inside the parantheses
    x = x #- 0.01*(Jac\residual[residual_idx])

    # update the state
    stt[:vm][tii][PQidx] = x[PQidx]
    KP = x[nPQ + 1]
    stt[:va][tii][2:end] = x[(nPQ+2):end]

    # update the flows and residual and such
    quasiGrad.update_states_for_distributed_slack_pf!(grd, idx, prm, qG, stt)
end

#println(x)
# after power flow, we must (1) apply active power updates, and (2)

# %% ========
pi_p, pi_q, PQidx = quasiGrad.slack_factors(idx, prm, stt, sys, tii)
#PQidx = [2]

Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)
residual = zeros(2*sys.nb)
residual_idx = [buses;           # => P
                sys.nb .+ PQidx]  # => Q
tii = :t1
KP  = 0.0

KP = quasiGrad.solve_power_flow(grd, idx, KP, pi_p, prm, PQidx, qG, residual, stt, sys, tii, Ybus_real, Ybus_imag)
quasiGrad.apply_pq_injections!(idx, prm, qG, stt, sys, tii)
# now, update all device power injections

# now, recomputed the Jacobian and have Gurobi solve a quick linearized power flow

# %%
quasiGrad.power_flow_residual!(idx, KP, pi_p, residual, stt, sys, tii)
println(quasiGrad.norm(residual[residual_idx]))


# %% =========== *******************
KP = 0.0
KQ = 0.0
residual = zeros(2*sys.nb)
pi_p .= 1.0/sys.nb
pi_q .= 1.0/sys.nb

# %%
x = copy([KQ; stt[:vm][tii][2:end]; KP; stt[:va][tii][2:end]])
quasiGrad.power_flow_residual_kpkq!(idx, KP, KQ, pi_p, pi_q, residual, stt, sys, tii)
println(quasiGrad.norm(residual[residual_idx]))

Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)
Jac[sys.nb+1:end, 1         ]  = pi_q
Jac[1:sys.nb,     1         ] .= 0.0
Jac[sys.nb+1:end, sys.nb + 1] .= 0.0
Jac[1:sys.nb,     sys.nb + 1]  = pi_p

x = x - (Jac\residual)

# update the state
KQ = x[1]
stt[:vm][tii][2:end] = x[2:sys.nb]
KP = x[sys.nb+1]
stt[:va][tii][2:end] = x[sys.nb+2:end]

quasiGrad.update_states_for_distributed_slack_pf!(grd, idx, prm, qG, stt)

# %%

quasiGrad.apply_pq_injections!(idx, prm, qG, stt, sys, tii)

# %%

quasiGrad.power_flow_residual!(idx, 0.0, pi_p, residual, stt, sys, tii)
println(quasiGrad.norm(residual[residual_idx]))

# %%
quasiGrad.power_flow_residual!(idx, 0.0, pi_p, residual, stt, sys, tii)
Jac = quasiGrad.build_acpf_Jac(stt, sys, tii, Ybus_real, Ybus_imag)

# remove ref bus phase and ref bus active power
Jac = Jac[:, [1:sys.nb; sys.nb+2:end]]
Jac = Jac[2:end, :]

# take a Newton step -- do NOT put the step scaler inside the parantheses
x = copy([stt[:vm][tii]; stt[:va][tii][2:end]])
x = x - (Jac\residual[2:end])
stt[:vm][tii]        = x[1:sys.nb]
stt[:va][tii][2:end] = x[(sys.nb+1):end]
quasiGrad.update_states_for_distributed_slack_pf!(grd, idx, prm, qG, stt)

quasiGrad.power_flow_residual!(idx, 0.0, pi_p, residual, stt, sys, tii)
println(quasiGrad.norm(residual[2:end]))

# steps
#1. solve newton
#2. linearize and pass to gurobi
#3. continue with lbfgs

# %% test linear power flow
tii = :t1
plb, pub, qlb, qub = quasiGrad.get_injection_bounds(idx, prm, stt, sys, tii)

# %% ============
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)
Jac, pinj0, qinj0 = quasiGrad.build_acpf_Jac(true, stt, sys, tii, Ybus_real, Ybus_imag)

# %%
dvm, dva = quasiGrad.solve_linear_pf_with_Gurobi!(Jac, pinj0, plb, prm, pub, qG, qinj0, qlb, qub, stt, sys, tii)

# %% try again
stt[:vm][tii]        = stt[:vm][tii] + dvm
stt[:va][tii][2:end] = stt[:va][tii][2:end] + dva

# %%
Jac, pinj0, qinj0 = quasiGrad.build_acpf_Jac(true, stt, sys, tii, Ybus_real, Ybus_imag)
@time dvm, dva = quasiGrad.solve_linear_pf_with_Gurobi!(Jac, pinj0, plb, prm, pub, qG, qinj0, qlb, qub, stt, sys, tii);

# %%stt[:vm][tii] = stt[:vm][tii] + dvm
# ask Gurobi to solve a linearize power flow
#
# here is power balance:
#
# p_pr - p_cs - () = p_lines => this is typical.
#
vm0 = stt[:vm][tii]
va0 = stt[:va][tii][2:end-1]

# build and empty the model!
model = Model(quasiGrad.Gurobi.Optimizer)
empty!(model)

# quiet down!!!
quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

# define the variables (single time index)
@variable(model, x_in[1:(2*sys.nb - 1)])
@variable(model, x_out[1:2*sys.nb])

# assign
dvm   = x_in[1:sys.nb]
dva   = x_in[(sys.nb+1):end]
dpinj = x_out[1:sys.nb]
dqinj = x_out[(sys.nb+1):end]

#
# note:
# vm   = vm0   + dvm
# va   = va0   + dva
# pinj = pinj0 + dpinj
# qinj = qinj0 + dqinj

# bound variables
@constraint(model, prm.bus.vm_lb .<= vm0   + dvm   .<= prm.bus.vm_ub)
@constraint(model, plb           .<= -pinj0 - dpinj .<= pub          )
@constraint(model, qlb           .<= -qinj0 - dqinj .<= qub          )

# mapping
@constraint(model, x_out .== Jac[:,[1:sys.nb; (sys.nb+2):end]]*x_in)

# objective: hold p (and v?) close to their initial values
obj = AffExpr(0.0)
for ii in 1:sys.nb
    tmp = @variable(model)
    @constraint(model, dpinj[ii]  <= tmp)
    @constraint(model, -dpinj[ii] <= tmp)
    quasiGrad.add_to_expression!(obj, tmp)
end

# set the objective
@objective(model, Min, obj)

# solve
quasiGrad.optimize!(model)
# println("========================================================")
println(quasiGrad.termination_status(model),". ",quasiGrad.primal_status(model),". objective value: ", quasiGrad.objective_value(model))
# println("========================================================")

# %% ==============================
tii = :t1
t_ind = 1

qG.Gurobi_pf_obj            = "min_dispatch_distance"
qG.compute_pf_injs_with_Jac = true

# build and empty the model!
model = Model(quasiGrad.Gurobi.Optimizer)
@info "Running lineaized power flow across $(sys.nT) time periods."

# initialize
init_pf = true
run_pf  = true
pf_cnt  = 0

# 1. update the ideal dispatch point (active power) -- we do this just once
quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

# 2. update the injection bounds (upper and lower P/Q bounds) -- no longer needed
# quasiGrad.get_injection_bounds!(idx, msc, prm, stt, sys, tii)

# 3. update y_bus and Jacobian and bias point -- this
#    only needs to be done once per time, since xfm/shunt
#    values are not changing between iterations
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

# %% loop over pf solves

while run_pf == true
    run_pf = false
    # increment
    pf_cnt += 1

    # first, rebuild the jacobian, and update the
    # base points: msc[:pinj0], msc[:qinj0]
    Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
    
    # quiet down!!!
    empty!(model)
    set_silent(model)
    quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

    # define the variables (single time index)
    @variable(model, x_in[1:(2*sys.nb - 1)])
    @variable(model, x_out[1:2*sys.nb])

    # assign
    dvm   = x_in[1:sys.nb]
    dva   = x_in[(sys.nb+1):end]
    dpinj = x_out[1:sys.nb]
    dqinj = x_out[(sys.nb+1):end]
    #
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

        # bound dc power
        @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
        @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
        @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

        # now, we need to loop and set the affine expressions to 0, and then add powers
        #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
        for bus in 1:sys.nb
            # sum over line powers
            nodal_p[bus] -= sum(pdc_vars[idx.bus_is_dc_frs[bus]];    init=0.0)
            nodal_p[bus] += sum(pdc_vars[idx.bus_is_dc_tos[bus]];    init=0.0)
            nodal_q[bus] -= sum(qdc_fr_vars[idx.bus_is_dc_frs[bus]]; init=0.0)
            nodal_q[bus] -= sum(qdc_to_vars[idx.bus_is_dc_tos[bus]]; init=0.0)
        end
    end
    
    # next, deal with devices
    @variable(model, dev_p_vars[1:sys.ndev])
    @variable(model, dev_q_vars[1:sys.ndev])

    # call the bounds
    dev_plb = stt[:u_on_dev][tii].*getindex.(prm.dev.p_lb,t_ind)
    dev_pub = stt[:u_on_dev][tii].*getindex.(prm.dev.p_ub,t_ind)
    dev_qlb = stt[:u_sum][tii].*getindex.(prm.dev.q_lb,t_ind)
    dev_qub = stt[:u_sum][tii].*getindex.(prm.dev.q_ub,t_ind)

    # first, define p_on at this time
    p_on = dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii]

    # bound
    @constraint(model, dev_plb .<= p_on       .<= dev_pub)
    @constraint(model, dev_qlb .<= dev_q_vars .<= dev_qub)

    # apply additional bounds: J_pqe (equality constraints)
    if ~isempty(idx.J_pqe)
        @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[dev]*stt[:u_sum][tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe])
    end

    # apply additional bounds: J_pqmin/max (inequality constraints)
    #
    # note: when the reserve products are negelected, pr and cs constraints are the same
    #   remember: idx.J_pqmax == idx.J_pqmin
    if ~isempty(idx.J_pqmax)
        @constraint(model, dev_q_vars[idx.J_pqmax] .<= prm.dev.q_0_ub[idx.J_pqmax]*stt[:u_sum][tii][idx.J_pqmax] + prm.dev.beta_ub[idx.J_pqmax]*dev_p_vars[idx.J_pqmax])
        @constraint(model, prm.dev.q_0_lb[idx.J_pqmax]*stt[:u_sum][tii][idx.J_pqmax] + prm.dev.beta_lb[idx.J_pqmax]*dev_p_vars[idx.J_pqmax] .<= dev_q_vars[idx.J_pqmax])
    end

    # great, now just update the nodal injection vectors
    for bus in 1:sys.nb
        # sum over line powers
        nodal_p[bus] += sum(dev_p_vars[idx.pr[bus]]; init=0.0)
        nodal_p[bus] -= sum(dev_p_vars[idx.cs[bus]]; init=0.0)
        nodal_q[bus] += sum(dev_q_vars[idx.pr[bus]]; init=0.0)
        nodal_q[bus] -= sum(dev_q_vars[idx.cs[bus]]; init=0.0)
    end

    # bound system variables ==============================================
    #
    # bound variables -- voltage
    @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm .<= prm.bus.vm_ub)

    # mapping
    JacP_noref = @view Jac[1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
    JacQ_noref = @view Jac[(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]

    # balance p and q
    @constraint(model, JacP_noref*x_in + msc[:pinj0] .== nodal_p)
    @constraint(model, JacQ_noref*x_in + msc[:qinj0] .== nodal_q)

    # objective: hold p (and v?) close to its initial value
    # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
    if qG.Gurobi_pf_obj == "min_dispatch_distance"
        # this finds a solution close to the dispatch point -- does not converge without v,a regularization
        obj    = AffExpr(0.0)
        tmp_vm = @variable(model)
        tmp_va = @variable(model)
        for bus in 1:sys.nb
            tmp = @variable(model)
            @constraint(model, msc[:pinj_ideal][bus] - nodal_p[bus] <= tmp)
            @constraint(model, nodal_p[bus] - msc[:pinj_ideal][bus] <= tmp)
            quasiGrad.add_to_expression!(obj, tmp)

            # voltage regularization
            @constraint(model, -dvm[bus] <= tmp_vm)
            @constraint(model,  dvm[bus] <= tmp_vm)

            # phase regularization
            if bus > 1
                @constraint(model, -dva[bus-1] <= tmp_va)
                @constraint(model,  dva[bus-1] <= tmp_va)
            end
        end

        # this adds light regularization and causes convergence
        quasiGrad.add_to_expression!(obj, tmp_vm)
        quasiGrad.add_to_expression!(obj, tmp_va)

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
        obj = tmp_p + tmp_vm + tmp_va
    else
        @warn "pf solver objective not recognized!"
    end

    # set the objective
    @objective(model, Min, obj)

    # solve
    quasiGrad.optimize!(model)

    # take the norm of dv
    norm_dv = quasiGrad.norm(quasiGrad.value.(dvm))
    
    # println("========================================================")
    println(quasiGrad.termination_status(model),". ", quasiGrad.primal_status(model),". objective value: ", round(quasiGrad.objective_value(model), sigdigits = 5), "dv norm: ", round(norm_dv, sigdigits = 5))
    # println("========================================================")

    # now, update the state vector with the soluion
    stt[:vm][tii]        = stt[:vm][tii]        + quasiGrad.value.(dvm)
    stt[:va][tii][2:end] = stt[:va][tii][2:end] + quasiGrad.value.(dva)

    # shall we terminate?
    #if (norm_dv < 1e-3) || (pf_cnt == qG.max_linear_pfs)
    #    run_pf = false
    #end
end

# %%
rQ = sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) +
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

# %%
rQ2 = sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) +
-sum(stt[:dev_q][tii][idx.pr[bus]]; init=0.0)

# %% test lbfgs gradient ===========
qG.pqbal_grad_mod_type = "standard"
#qG.pqbal_grad_mod_type = "standard"
epsilon                = 1e-6     # maybe set larger when dealing with ctgs + krylov solver..
qG.cdist_psolve        = 0.0

# %% gradient modifications -- power balance
qG.pqbal_grad_mod_type     = "soft_abs"
qG.pqbal_grad_mod_eps2     = 1e-8


# %%
#  1. transformer phase shift (phi) =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
z0      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx    = copy(mgd[:phi][tii][ind])

# perturb and test
stt[:phi][tii][ind] += epsilon
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
zp = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 2. transformer turns ratio (tau) =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nx)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
z0      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx    = copy(mgd[:tau][tii][ind])

# perturb and test
stt[:tau][tii][ind] += epsilon
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
zp = sum(stt[:zp][tii]) + sum(stt[:zq][tii])

println("========")
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 3. voltage magnitude =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
z0      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx    = mgd[:vm][tii][ind]

# perturb and test
stt[:vm][tii][ind] += epsilon 
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
zp = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx_num = (zp - z0)/epsilon

println(dzdx)
println(dzdx_num)

# %% 4. voltage phase =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
z0      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx    = mgd[:va][tii][ind]

# perturb and test
stt[:va][tii][ind] += epsilon 
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
zp      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)


# %% 10. :u_step_shunt =======================================================================
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.nsh)) # -- most have a 0-bs/gs value, so choose wisely :)

# to ensure we're not tipping over into some new space
stt[:u_step_shunt][tii][ind] = 0.4
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
z0      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx  = copy(mgd[:u_step_shunt][tii][ind])

# perturb and test
stt[:u_step_shunt][tii][ind] += epsilon 
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
zp      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 11. p_on
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.ndev)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
z0      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx    = copy(mgd[:p_on][tii][ind])

# perturb and test
stt[:p_on][tii][ind] += epsilon 
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
zp      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 11. dev_q
tii     = Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
ind     = Int64(round(rand(1)[1]*sys.ndev)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
z0      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx    = copy(mgd[:dev_q][tii][ind])

# perturb and test
stt[:dev_q][tii][ind] += epsilon 
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, dpf0, grd, idx, mgd, msc, prm, qG, stt, sys, zpf)
zp      = sum(stt[:zp][tii]) + sum(stt[:zq][tii])
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)