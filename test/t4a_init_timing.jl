# test solver itself :)
include("../src/quasiGrad_dual.jl")
include("./test_functions.jl")

using BenchmarkTools
using ProfileView

# %% files
path = "../GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
#path = "../GO3_testcases/C3S0_20221208/D1/C3S0N00073/scenario_002.json"
# path = "../GO3_testcases/C3S1_20221222/D3/C3S1N06049/scenario_001.json"

# load
print("jsn: ")
json_data = quasiGrad.load_json(path);

adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

# %% build the system struct
print("sys: ")
@time sys = quasiGrad.build_sys(json_data);

# parse the network elements
print("dc: ")
@time dc_prm      = quasiGrad.parse_json_dc(json_data);


print("ctg: ")
@time ctg_prm     = quasiGrad.parse_json_ctg(json_data);

 
print("bus: ")
@time bus_prm     = quasiGrad.parse_json_bus(json_data);


print("xfm: ")
@time xfm_prm     = quasiGrad.parse_json_xfm(json_data);


print("shunt: ")
@time shunt_prm   = quasiGrad.parse_json_shunt(json_data);


print("acline: ")
@time acline_prm  = quasiGrad.parse_json_acline(json_data);


print("dev: ")
@time device_prm  = quasiGrad.parse_json_device(json_data);

# parse violation, and then parse the reserve, which updates vio_prm
print("vio: ")
@time reserve_prm, vio_prm = quasiGrad.parse_json_reserve_and_vio(json_data);

# read the time series data
print("ts: ")
@time ts_prm = quasiGrad.parse_json_timeseries(json_data);

@time prm = quasiGrad.Param(
    ts_prm,
    dc_prm, 
    ctg_prm, 
    bus_prm, 
    xfm_prm, 
    vio_prm,
    shunt_prm, 
    acline_prm, 
    device_prm, 
    reserve_prm);
@time qG = quasiGrad.initialize_qG(prm);

print("idx: ")
@time idx = quasiGrad.initialize_indices(prm, sys);

# %%
ProfileView.@profview quasiGrad.build_time_sets(prm, sys);
@time Ts_mndn, Ts_mnup, Ts_sdpc, ps_sdpc_set, Ts_supc,
ps_supc_set, Ts_sus_jft, Ts_sus_jf, Ts_en_max, 
Ts_en_min, Ts_su_max = quasiGrad.build_time_sets(prm, sys);

# %%
@time cgd, GRB, grd, mgd, scr, stt = quasiGrad.initialize_states(idx, prm, sys);

# %% initialize the states which adam will update -- the rest are fixed
@time adm = quasiGrad.initialize_adam_states(sys);

# %% define the states which adam can/will update, and fix the rest!
@time upd = quasiGrad.identify_update_states(prm, idx, stt, sys)

# %% initialize the contingency network structure
@btime ntk, flw = quasiGrad.initialize_ctg(sys, prm, qG, idx);

# %% initialize
adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

# %% note, the reference bus is always bus #1
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
    if sys.nb <= qG.min_buses_for_krylov
        # too few buses -- use LU
        @warn "Not enough buses for Krylov! using LU."
        Ybr_ChPr = quasiGrad.I
    else
        @info "Preconditioning is disabled."
        # test for negative reactances
        if minimum(ac_b_params) < 0.0
            # Amrit Pandey: "watch out for negatvive reactance! You will lose
            #                pos-sem-def of the Cholesky preconditioner."
            abs_b   = abs.(ac_b_params)
            abs_Ybr = (E'*quasiGrad.spdiagm(ac_b_params)*E)[2:end,2:end] 
            Ybr_ChPr = quasiGrad.CholeskyPreconditioner(abs_Ybr, qG.cutoff_level);
            #Ybr_ChPr = quasiGrad.I
        else
            Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
            Ybr_ChPr = quasiGrad.I
        end
    end
else
    Ybr_ChPr = quasiGrad.I
end

# get the flow matrix
Yfr = Ybs*Er

# %% build the low-rank contingecy updates
#
# base: Y_b*theta_b = p
# ctg:  Y_c*theta_c = p
#       Y_c = Y_b + uk'*uk
ctg_out_ind = Dict(ctg_ii => Vector{Int64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
ctg_params  = Dict(ctg_ii => Vector{Float64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)

# should we build the full ctg matrices?
if qG.build_ctg_full == true
    nac   = sys.nac
    Ybr_k = Dict(ctg_ii => quasiGrad.spzeros(nac,nac) for ctg_ii in 1:sys.nctg)
else
    # build something small of the correct data type
    Ybr_k = Dict(1 => quasiGrad.spzeros(1,1))
end

# %% and/or, should we build the low rank ctg elements?
if qG.build_ctg_lowrank == true
    # no need => v_k = Dict(ctg_ii => quasiGrad.spzeros(nac) for ctg_ii in 1:sys.nctg)
    # no need => b_k = Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
    u_k = Dict(ctg_ii => zeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
    w_k = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
else
    v_k = 0
    b_k = 0
end

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
end

# %%
C = quasiGrad.cholesky(Ybr);
for ctg_ii in 1:sys.nctg
    println(ctg_ii)
    # next, should we build the actual, full ctg matrix?
    if qG.build_ctg_full == true
        # direct construction..
        #
        # NOTE: this is written assuming multiple elements can be
        # simultaneously outaged
        Ybs_k = copy(Ybs)
        Ybs_k[CartesianIndex.(tuple.(ctg_out_ind[ctg_ii],ctg_out_ind[ctg_ii]))] .= 0
        Ybr_k[ctg_ii] = Er'*Ybs_k*Er
    end

    # and/or, should we build the low rank ctg elements?
    if qG.build_ctg_lowrank == true
        # .. vs low rank
        #
        # NOTE: this is written assuming only ONE element
        # can be outaged

        # no need to save:
            # v_k[ctg_ii] =  Er[ctg_out_ind[ctg_ii][1],:]
            # b_k[ctg_ii] = -ac_b_params[ctg_out_ind[ctg_ii][1]]
        v_k = Er[ctg_out_ind[ctg_ii][1],:]
        b_k = -ac_b_params[ctg_out_ind[ctg_ii][1]]
        #
        # construction: 
        # 
        # Ybr_k[ctg_ii] = ctg[:Ybr] + v*beta*v'
        #               = ctg[:Ybr] + vLR_k[ctg_ii]*beta*vLR_k[ctg_ii]
        #
        # if v, b saved:
            # u_k[ctg_ii] = Ybr\Array(v_k[ctg_ii])
            # w_k[ctg_ii] = b_k[ctg_ii]*u_k[ctg_ii]/(1+(v_k[ctg_ii]'*u_k[ctg_ii])*b_k[ctg_ii])
         #quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl = Ybr_ChPr)
        # u_k[ctg_ii] = Ybr\Vector(v_k)
        # u_k[ctg_ii] = C\Vector(v_k)
        u_k_local = C\Vector(v_k)
        # sparsify
        abs_u_k           = abs.(u_k_local)
        u_k_ii_SmallToBig = sortperm(abs_u_k, rev=true)
        bit_vec           = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> (1.0 - qG.accuracy_sparsify_lr_updates)
        # edge case is caught! bit_vec will never be empty. Say, abs_u_k[u_k_ii_SmallToBig] = [0,0,1], then we have
        # bit_vec = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> 0.01%, say => bit_vec = [0,0,1] 
        # 
        # also, we use ".>" because we only want to include all elements that contribute to meeting the stated accuracy goal
        u_k[ctg_ii][bit_vec] = u_k_local[bit_vec]
        # this is ok, since u_k and w_k have the same sparsity pattern
        w_k[ctg_ii][bit_vec] = b_k*u_k[ctg_ii][bit_vec]/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii]))*b_k)
    end
end

# %% test -- 
am = zeros(sys.nctg)
for ctg_ii in 1:sys.nctg
    am[ctg_ii] = argmax(cumsum(sort(abs.(u_k[ctg_ii]), rev=true))/sum(abs.(u_k[ctg_ii])) .> 0.98)/6048.0
end

v_k = Er[ctg_out_ind[ctg_ii][1],:]


# %%
qG.pcg_tol = 0.000001
for ctg_ii in 1:sys.nctg
    println(ctg_ii)
    v_k = Er[ctg_out_ind[ctg_ii][1],:]
    b_k = -ac_b_params[ctg_out_ind[ctg_ii][1]]
    u_k[ctg_ii] = C\Vector(Float64.(v_k))
    #quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl=Ybr_ChPr)
end

# %%
vvv = zeros(sys.nctg)
for ctg_ii in 1:sys.nctg
    vvv[ctg_ii] = (sum(sort(abs.(u_k[ctg_ii].^2))[5000:end])/sum(sort(abs.(u_k[ctg_ii].^2))))
end

# %%

@time pp = cumsum(u_k[ctg_ii])
quasiGrad.plot(sort(abs.(u_k[253].^2)),yaxis = :log)


# %%
qG.cutoff_level = 25
Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);

# %%
vv0 = quasiGrad.spzeros(100000)
vv1 = zeros(100000)
vv2 = randn(100000)
vv3 = randn(100000)

@time quasiGrad.dot(vv1,vv0);
@time quasiGrad.dot(vv1,vv2);
@time quasiGrad.dot(vv3,vv2);
@time vv1'*vv2;

# %%

qG.pcg_tol = 0.001
v_k = Er[ctg_out_ind[ctg_ii][1],:]
b_k = -ac_b_params[ctg_out_ind[ctg_ii][1]]
u_k[ctg_ii] .= 0.0
vk = Vector(Float64.(v_k))
cc = copy(u_k[ctg_ii])

@time quasiGrad.cg!(cc, Ybr, vk, abstol = qG.pcg_tol, Pl=Ybr_ChPr);
@time tt = Ybr\Vector(v_k);

# %% initialize ctg state
tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

# build the phase angle solution dict -- this will have "sys.nb-1" angles for each solution,
# since theta1 = 0, and it will have n_ctg+1 solutions, because the base case solution will be
# optionally saved at the end.. similar for pflow_k
# theta_k       = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1)) for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# pflow_k       = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nac))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# theta_k       = Dict(tkeys[ii] => [zeros(sys.nb-1) for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# pflow_k       = Dict(tkeys[ii] => [zeros(sys.nac)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# this is the gradient solution assuming a base case admittance (it is then rank 1 corrected to dz_dpinj)
# ctd = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 
# ctd = Dict(tkeys[ii] => [zeros(sys.nb-1)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))   
# this is the gradient solution, corrected from ctd
# dz_dpinj      = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 
# dz_dpinj      = Dict(tkeys[ii] => [zeros(sys.nb-1)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 

# "local" storage for apparent power flows (not needed across time)
# sfr     = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
# sto     = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
# sfr_vio = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
# sto_vio = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)   

# phase shift derivatives
#   => consider power injections:  pinj = (p_pr-p_cs-p_sh-p_fr_dc-p_to_dc-alpha*slack) + Er^T*phi*b
#      => Er^T*phi*b
# ~ skip the reference bus! -- fr_buses = positive in the incidence matrix; to_buses = negative..
xfm_at_bus      = Dict(bus => vcat(idx.bus_is_xfm_frs[bus],idx.bus_is_xfm_tos[bus]) for bus in 2:sys.nb)
xfm_at_bus_sign = Dict(bus => vcat(idx.bus_is_xfm_frs[bus],-idx.bus_is_xfm_tos[bus]) for bus in 2:sys.nb)
xfm_phi_scalars = Dict(bus => ac_b_params[xfm_at_bus[bus] .+ sys.nl].*sign.(xfm_at_bus_sign[bus]) for bus in 2:sys.nb)

# %%
abstol = 0.001
bb = Float64.(Vector(v_k))
t1 = quasiGrad.cg(Ybr, bb, abstol = qG.pcg_tol);

vv = b_k*t1/(1.0+(quasiGrad.dot(v_k,t1))*b_k)

# %%
t2 = Ybr\Vector(v_k);

# %%
c = randn(6048)
@btime quasiGrad.dot(v_k,c);

v_kf = Float64.(Vector(v_k))
@btime quasiGrad.dot(v_k,c);

# %%
@btime Float64.(v_k);
#@btime Float64.(Vector(v_k));

# %% ==================
qG.cutoff_level = 50
ctg_ii   = 15
v_k      = Er[ctg_out_ind[ctg_ii][1],:]
Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
C        = quasiGrad.cholesky(Ybr);

# vf = Float64.(v_k)
#@time v_kk = Vector(Float64.(v_k))
v_kk = randn(6048)
qG.pcg_tol = 0.0001
# @time quasiGrad.cg(Ybr, v_kk, abstol = qG.pcg_tol);
@time t1 = quasiGrad.cg(Ybr, v_kk, abstol = qG.pcg_tol, Pl = Ybr_ChPr);
@time t2 = Ybr\v_kk;
@time t3 = C\v_kk;

# %% ======
qG.pcg_tol = 0.005
#vv  = Vector(Float64.(v_k))
vv  = randn(6048)
qG.pcg_tol = 10.0
xx = zeros(6048)
@time quasiGrad.cg!(xx, Ybr, vv, abstol = qG.pcg_tol);
@time Ybr\vv;

# %%
qG.pcg_tol = 2.0
t1 = quasiGrad.cg(Ybr, vv, abstol = qG.pcg_tol);
t2 = Ybr\vv;

plot(t1)
plot!(t2)


# %% ==================
qG.cutoff_level = 15
Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
#v_kk = randn(6048)
qG.pcg_tol = 0.001
# @time quasiGrad.cg(Ybr, v_kk, abstol = qG.pcg_tol);
@time t1  = quasiGrad.cg(Ybr, v_kk, abstol = qG.pcg_tol, Pl = Ybr_ChPr);
@time t1p = Ybr_ChPr\v_kk;

r = quasiGrad.norm(Ybr*t1 - v_kk)

# %%
max_error = 0.01

quasiGrad.norm([1,1,1])

# %%
@time t2 = Ybr\v_kk;
@time t3 = C\v_kk;

# %%
#A = randn(500,500)
#A = A + A' + 100*quasiGrad.I
#A  = quasiGrad.sprand(100, 100, 0.25)
#A  = A + A' + 10*quasiGrad.I
p  = quasiGrad.CholeskyPreconditioner(A, 53);
A_approx =  ((p.ldlt.L + quasiGrad.I)*quasiGrad.sparse(quasiGrad.diagm(p.ldlt.D))*(p.ldlt.L' + quasiGrad.I))

quasiGrad.norm(A_approx - A[p.ldlt.P, p.ldlt.P])


# %%
@btime t = zeros(100000);

# %%
A = randn(10,10)
A = A + A' + 100.0*quasiGrad.I
C = quasiGrad.cholesky(A);

# %%
zs = quasiGrad.sprand(100000,0.01)
z  = randn(100000)

@btime z-zs
@btime z-Vector(zs)

# %% 
v = randn(100000)
@btime abs2.(v);
@btime v.^2;
# %%
@btime sin.(v);

# %% ===============
mutable struct Dm2
    pinj::Vector{Float64}
end
dd = Dict(
    :pinj => zeros(100000))

dm3 = Dm2(zeros(100000))

# %% test
vv1 = randn(100000)
vv2 = randn(100000)
@btime dm3.pinj  = vv1;
@btime dd[:pinj] = vv2;
@btime dm3.pinj.^2
@btime dd[:pinj].^2