# test solver itself :)
include("../src/quasiGrad_dual.jl")
include("./test_functions.jl")

# %% here for convenience :)
if true == false
    include("../src/quasiGrad_dual.jl")
    sys = quasiGrad.build_sys(jsn)
end

# %%
sys = quasiGrad.build_sys(jsn)

# %% file
#data_dir  = "./test/data/c3/C3S0_20221208/D1/C3S0N00003/"
#file_name = "scenario_003.json"

# 73 bus
#data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
#file_name = "scenario_002.json"

# 73 bus -- modified!!
# xfm phase shifters are given nonzero upper and lower bounds -- for: test ctg
# 
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073_phi_mod/"
file_name = "scenario_002_phi_mod.json"

# read and parse the input data
jsn, prm, idx, sys = quasiGrad.load_and_parse_json(data_dir*file_name)
qG                 = quasiGrad.initialize_qG(prm)
qG.eval_grad     = true
qG.constraint_grad_weight         = 1000.0

# initialize
cgd, GRB, grd, mgd, scr, stt = quasiGrad.initialize_states(idx, prm, sys);

# perturb stt
perturb!(stt, prm, idx, grd, sys, qG, 1.0)

# initialize static gradients
quasiGrad.initialize_static_grads!(prm, idx, grd, sys, qG)
quasiGrad.clip_all!(prm, qG, stt)

# %% initialize the states which adam will update -- the rest are fixed + ctg
adm = quasiGrad.initialize_adam_states(sys)
upd = quasiGrad.identify_update_states(prm, idx, stt, sys)
ntk = quasiGrad.initialize_ctg(sys, prm, qG, idx)

quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# call the ctg solver
#include("../src/core/initializations.jl")
#include("../src/core/contingencies.jl")
#include("../src/core/scoring.jl")
#include("./test_functions.jl")

# %% ====== local tests
include("../src/core/initializations.jl")
include("../src/core/contingencies.jl")

# build ctg
# ctg = initialize_ctg(sys, prm, qG, idx)

# solve ctg (with gradients)
qG.eval_grad = true
quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

# %%
# duration
dt = prm.ts.duration[tii]

# compute the shared, common gradient terms (time dependent)
#
# gc_avg = grd[:nzms][:zctg_avg] * grd[:zctg_avg][:zctg_avg_t] * grd[:zctg_avg_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
#        = (-1)                  *                (1)          *       (1/sys.nctg)        *          (-1)       *   dt*prm.vio.s_flow
#        = dt*prm.vio.s_flow/sys.nctg
gc_avg   = cgd.ctg_avg[tii] * qG.scale_c_sflow_testing

# gc_min = (-1)                  *                (1)          *            (1)          *          (-1)       *   dt*prm.vio.s_flow
#        = grd[:nzms][:zctg_min] * grd[:zctg_min][:zctg_min_t] * grd[:zctg_min_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
gc_min   = cgd.ctg_min[tii] * qG.scale_c_sflow_testing

# get the slack at this time
p_slack = 
    sum(stt[:dev_p][tii][pr] for pr in idx.pr_devs) -
    sum(stt[:dev_p][tii][cs] for cs in idx.cs_devs) - 
    sum(stt[:sh_p][tii])

# loop over each bus
for bus in 1:sys.nb
    # active power balance
    flw[:p_inj][bus] = 
        sum(stt[:dev_p][tii][pr] for pr in idx.pr[bus]; init=0.0) - 
        sum(stt[:dev_p][tii][cs] for cs in idx.cs[bus]; init=0.0) - 
        sum(stt[:sh_p][tii][sh] for sh in idx.sh[bus]; init=0.0) - 
        sum(stt[:dc_pfr][tii][dc_fr] for dc_fr in idx.bus_is_dc_frs[bus]; init=0.0) - 
        sum(stt[:dc_pto][tii][dc_to] for dc_to in idx.bus_is_dc_tos[bus]; init=0.0) - 
        ntk.alpha*p_slack
end

# also, we need to update the flows on all lines! and the phase shift
flw[:ac_qfr][idx.ac_line_flows] .= stt[:acline_qfr][tii]
flw[:ac_qfr][idx.ac_xfm_flows]  .= stt[:xfm_qfr][tii]
flw[:ac_qto][idx.ac_line_flows] .= stt[:acline_qto][tii]
flw[:ac_qto][idx.ac_xfm_flows]  .= stt[:xfm_qto][tii]
flw[:ac_phi][idx.ac_phi]        .= stt[:phi][tii]

# compute square flows
flw[:qfr2] .= flw[:ac_qfr].^2
flw[:qto2] .= flw[:ac_qto].^2

# solve for the flows across each ctg
#   p  =  @view flw[:p_inj][2:end]
flw[:bt] .= .-flw[:ac_phi].*ntk.b
# now, we have flw[:p_inj] = Yb*theta + E'*bt
#   c = p - ntk.Er'*bt
#
# simplified:
flw[:c] .= (@view flw[:p_inj][2:end]) .- ntk.Er'*flw[:bt]

# solve the base case with pcg
if qG.base_solver == "lu"
    ctb[t_ind]  .= ntk.Ybr\flw[:c]

# error with this type !!!
# elseif qG.base_solver == "cholesky"
#    ctb[t_ind]  = ntk.Ybr_Ch\c

elseif qG.base_solver == "pcg"
    if sys.nb <= qG.min_buses_for_krylov
        # too few buses -- just use LU
        ctb[t_ind] .= ntk.Ybr\flw[:c]
    else
        # solve with a hot start!
        #
        # note: ctg[:ctb][tii][end] is modified in place,
        # and it represents the base case solution
        _, ch = quasiGrad.cg!(ctb[t_ind], ntk.Ybr, flw[:c], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
        
        # test the krylov solution
        if ~(ch.isconverged)
            @info "Krylov failed -- using LU backup (ctg flows)!"
            ctb[t_ind] = ntk.Ybr\flw[:c]
        end
    end
else
    println("base case solve type not recognized :)")
end

# set all ctg scores to 0:
stt[:zctg][tii] .= 0.0

# zero out the gradients, which will be collected and applied all at once!
flw[:dz_dpinj_all] .= 0.0

# %% do we want to score all ctgs? for testing/post processing
if qG.score_all_ctgs == true
    ###########################################################
    # => up above: @info "Warning -- scoring all contingencies! No gradients."
    ###########################################################
    for ctg_ii in 1:sys.nctg
        # see the "else" case for comments and details
        theta_k = quasiGrad.special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
        pflow_k = ntk.Yfr*theta_k  + flw[:bt]
        sfr     = sqrt.(flw[:qfr2] + pflow_k.^2)
        sto     = sqrt.(flw[:qto2] + pflow_k.^2)
        sfr_vio = sfr - ntk.s_max
        sto_vio = sto - ntk.s_max
        sfr_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
        sto_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
        smax_vio = max.(sfr_vio, sto_vio, 0.0)
        zctg_s = dt*prm.vio.s_flow*smax_vio * qG.scale_c_sflow_testing
        stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)
    end

    # score
    scr[:zctg_min] += minimum(stt[:zctg][tii])
    scr[:zctg_avg] += sum(stt[:zctg][tii])/sys.nctg
else
end

# %%
theta_k = quasiGrad.special_wmi_update(ctb[t_ind], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw[:c])
pflow_k = ntk.Yfr*theta_k  + flw[:bt]
sfr     = sqrt.(flw[:qfr2] + pflow_k.^2)
sto     = sqrt.(flw[:qto2] + pflow_k.^2)
sfr_vio = sfr - ntk.s_max
sto_vio = sto - ntk.s_max
sfr_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
sto_vio[ntk.ctg_out_ind[ctg_ii]] .= 0.0
smax_vio = max.(sfr_vio, sto_vio, 0.0)
zctg_s = dt*prm.vio.s_flow*smax_vio * qG.scale_c_sflow_testing
stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)

line = argmax(smax_vio)
fr_bus = idx.acline_fr_bus[line]
idx.pr[fr_bus]
idx.cs[fr_bus]

# %% 1. test if the gradient solution is correct -- p_on
quasiGrad.flush_gradients!(grd, mgd, prm, sys)
qG.pcg_tol = 0.00000001

qG.scale_c_sflow_testing = 1000.0

epsilon = 1e-5
tii     = :t1 #Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
t_ind   = 1
ind     = 101   #Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    for dev in 1:sys.ndev
        quasiGrad.apply_dev_q_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dq][tii][dev])
        quasiGrad.apply_dev_p_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dp][tii][dev])
    end
end
z0      = copy(-(scr[:zbase] + scr[:zctg_min] + scr[:zctg_avg]))
dzdx    = copy(mgd[:p_on][tii][ind])

# update device power
stt[:p_on][tii][ind] += epsilon
stt[:dev_p][tii] = stt[:p_on][tii] + stt[:p_su][tii] + stt[:p_sd][tii]
quasiGrad.flush_gradients!(grd, mgd, prm, sys)
quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
zp      = copy(-(scr[:zbase] + scr[:zctg_min] + scr[:zctg_avg]))
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% 1. test if the gradient solution is correct -- vm
quasiGrad.flush_gradients!(grd, mgd, prm, sys)
qG.pcg_tol = 0.000000001

epsilon = 1e-4
tii     = :t1 #Symbol("t"*string(Int64(round(rand(1)[1]*sys.nT)))); (tii == :t0 ? tii = :t1 : tii = tii)
t_ind   = 1
ind     = 2   #Int64(round(rand(1)[1]*sys.nb)); (ind == 0 ? ind = 1 : ind = ind)
quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
for tii in prm.ts.time_keys
    for dev in 1:sys.ndev
        quasiGrad.apply_dev_q_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dq][tii][dev])
        
        quasiGrad.apply_dev_p_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dp][tii][dev])
    end
end
z0      = copy(-(scr[:zbase] + scr[:zctg_min] + scr[:zctg_avg]))
dzdx    = copy(mgd[:vm][tii][ind])

# update device power
stt[:vm][tii][ind] += epsilon
quasiGrad.flush_gradients!(grd, mgd, prm, sys)
quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
zp      = copy(-(scr[:zbase] + scr[:zctg_min] + scr[:zctg_avg]))
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)



# %% Crazy idea test :) =======================
# solve a ctg
qG.pcg_tol = 1e-9

quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
# grab a flow
tii    = :t1
ctg_ii = 1

# perturb a power injection
line = 19
dev = 208    
id  = prm.dev.bus[dev]
bus = findall(x -> x.== id, prm.bus.id)[1]
pf  = copy(ctg[:pflow_k][tii][end][line])

stt[:p_on][tii][dev] += epsilon
stt[:dev_p][tii] = stt[:p_on][tii] + stt[:p_su][tii] + stt[:p_sd][tii]
quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
pf_new = copy(ctg[:pflow_k][tii][end][line])

# how much did the power flow change? can the ptdf matrix explain it?
ptdf = ctg[:Yfr]*inv(Matrix(ctg[:Ybr]))

# instead, test with the LR update idea!!
Av          = ptdf*ones(sys.nb-1)
ptdf_update = ptdf - Av*(ones(sys.nb-1)')/(sys.nb)
ptdf_alt    = ptdf*(I-ones(72)*ones(72)'/(sys.nb))

println((pf_new - pf)/epsilon)
println(ptdf_update[line,bus-1])
println(ptdf_alt)
println(ptdf[line,bus-1])



# %% compute the states (without gradients)
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# solve and apply a Gurobi projection!
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)

# %% re-compute the states (without gradients)
#ctg[:theta_k][tii][end] = zeros(sys.nb-1)
#qG.pcg_tol = 0.01
#qG.pcg_tol = 0.001
#qG.pcg_tol = 0.000001
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% write a solution :)
soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
quasiGrad.write_solution(data_dir*file_name, qG, soln_dict, scr)

# %% --------------
include("../src/quasiGrad_dual.jl")
sys = quasiGrad.build_sys(jsn)

# %% ================================
include("../src/core/initializations.jl")
ntk = initialize_ctg(sys, prm, qG, idx)

# %% linear system solve tests
using SparseArrays
using LinearAlgebra
using Preconditioners
using IterativeSolvers

# %% --
n  = 10000
A  = sprand(n, n, 0.00025)
A  = A + A' + 25*I
b  = randn(n)
F  = ldlt(A)
C  = cholesky(A)

# %% -- direct solve
@time x1 = A\b;
@time x2 = F\b;
@time x3 = C\b;

# %% standard cg
n  = 1000
A  = sprand(n, n, 0.0025)
A  = A + A' + 2*I
A  = ctg[:Ybr]
b  = randn(n)
b  = randn(72)
p  = CholeskyPreconditioner(A, 2);
pd = DiagonalPreconditioner(A)
x0 = A\b

@time x4  = cg(A, b, abstol = 1e-4);
@time x40 = cg!(x0, A, b, abstol = 1e-4);

@time x5 = cg(A, b, abstol = 1e-4, Pl=p);
@time x6 = cg(A, b, abstol = 1e-4, Pl=pd);

# %%
p  = CholeskyPreconditioner(A, 0);
A_approx =  ((p.ldlt.L + I)*sparse(diagm(p.ldlt.D))*(p.ldlt.L' + I))
norm(A_approx - A[p.ldlt.P, p.ldlt.P])

# %% preconditioned krylov
#p  = CholeskyPreconditioner(A, 100);
@time x5 = cg(A, b, abstol = 1e-4, Pl=p);

#x1 - x5
# %%

#t = 4

# Solve the system of equations
#b = A*ones(1000)
#x = cg(A, b, Pl=p)

# %%
p  = CholeskyPreconditioner(A, 25);
A_approx =  ((p.ldlt.L + I)*sparse(diagm(p.ldlt.D))*(p.ldlt.L' + I))
norm(A_approx - A[p.ldlt.P, p.ldlt.P])






# %% LR update tests ================================================
# note, the reference bus is always bus #1
#
# first, get the ctg limits
s_max_ctg = [prm.acline.mva_ub_em; prm.xfm.mva_ub_em]

# get the ordered names of all components
ac_ids = [prm.acline.id; prm.xfm.id ]

# get the ordered (negative!!) susceptances
ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]

# build the full incidence matrix: E = lines x buses
E  = quasiGrad.build_incidence(idx, prm, sys)
Er = E[:,2:end]

# get the diagonal admittance matrix   => Ybs == "b susceptance"
Ybs = quasiGrad.spdiagm(ac_b_params)
Yb  = E'*Ybs*E
Ybr = Yb[2:end,2:end]  # use @view ? 

# should we precondition the base case?
if qG.base_solver == "pcg"
    Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
else
    Ybr_ChPr = quasiGrad.I
end

# get the flow matrix
Yfr = Ybs*Er

# build the low-rank contingecy updates
#
# base: Y_b*theta_b = p
# ctg:  Y_c*theta_c = p
#       Y_c = Y_b + uk'*uk
ctg_out_ind = Dict(ctg_ii => Vector{Int64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
ctg_params  = Dict(ctg_ii => Vector{Float64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)

# should we build the full ctg matrices?
build_ctg = true
if build_ctg == true
    nac   = sys.nac
    Ybr_k = Dict(ctg_ii => quasiGrad.spzeros(nac,nac) for ctg_ii in 1:sys.nctg)
else
    Ybr_k = 0
end

# %%
ctg_ii = 1
cmpnts = prm.ctg.components[ctg_ii]
cmp_ii = 1
cmp    = cmpnts[cmp_ii]
cmp_index = findfirst(x -> x == cmp, ac_ids)
cmp_b     = -ac_b_params[cmp_index]
ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
ctg_params[ctg_ii][cmp_ii]  = cmp_b

# %% build option 1:
ctg_ii = 1
Ybs_k = copy(Ybs)
Ybs_k[CartesianIndex.(tuple.(ctg_out_ind[ctg_ii],ctg_out_ind[ctg_ii]))] .= 0
Ybr_k[ctg_ii] = Er'*Ybs_k*Er

# %% build option 2:
v        =  Er[ctg_out_ind[ctg_ii],:]
beta     = -ac_b_params[ctg_out_ind[ctg_ii]]
ctg_ii   = 1
Ybs_k_lr = ctg[:Ybr] + v'*beta*v

println(norm(Ybs_k_lr - Ybr_k[1]))

# %%
for ctg_ii in 1:sys.nctg
    # components
    cmpnts = prm.ctg.components[ctg_ii]
    for (cmp_ii,cmp) in enumerate(cmpnts)
        # get the cmp index and b
        cmp_index = findfirst(x -> x == cmp, ac_ids) 
        cmp_b     = -ac_b_params[cmp_index] # negative, because we subtract it out

        # output
        ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
        ctg_params[ctg_ii][cmp_ii]  = cmp_b

        #y_diag[cmp_index] = sqrt(cmp_b)
        # we record these in ctg
        # ctg_out_ind[ctg_ii]
    end

    # next, should we build the actual, full ctg matrix?
    if build_ctg == true
        Ybs_k = copy(Ybs)
        Ybs_k[CartesianIndex.(tuple.(ctg_out_ind[ctg_ii],ctg_out_ind[ctg_ii]))] .= 0
        Ybr_k[ctg_ii] = Er'*Ybs_k*Er
    end
end

# %% test ctg
ac_phi              = zeros(sys.nac)
ac_qfr              = zeros(sys.nac)
ac_qto              = zeros(sys.nac)
dsmax_dp_flow       = zeros(sys.nac)
dsmax_dqfr_flow     = zeros(sys.nac)
dsmax_dqto_flow     = zeros(sys.nac)
ctd       = [zeros(sys.nb-1) for _ in 1:sys.nctg]
p_inj               = zeros(sys.nb)
ctb        = [zeros(sys.nb-1) for _ in 1:sys.nT]
wct = [collect(1:sys.nctg) for _ in 1:sys.nT]

# %%
quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)
# %%
include("../src/core/contingencies.jl")

# 
qG.frac_ctg_keep = 0.01
qG.eval_grad     = false
qG.base_solver   = "pcg"

@time solve_ctgs!(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys,
                          ac_phi, ac_qfr, ac_qto, dsmax_dp_flow, dsmax_dqfr_flow,     
                          dsmax_dqto_flow, ctd, p_inj, ctb, wct)


# %%
@btime wct[t_ind] = sortperm(stt[:zctg][tii])
@btime sortperm(stt[:zctg][tii])
# %%

@btime (@view p_inj[2:end]) - ntk.Er'*bt
# %%
@time         p_slack = 
sum(stt[:dev_p][tii][idx.pr_devs]) -
sum(stt[:dev_p][tii][idx.cs_devs]) - 
sum(stt[:sh_p][tii])
# %%
solve_ctgs!(cgd, 
grd, 
idx, 
mgd, 
ntk, 
prm, 
qG, 
scr,
stt, 
sys,
ac_phi,
ac_qfr,              
ac_qto,                           
dsmax_dp_flow,  
dsmax_dqfr_flow,     
dsmax_dqto_flow,  
ctd,
p_inj,          
ctb,   
wct)