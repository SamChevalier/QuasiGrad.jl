using quasiGrad
using Revise

include("./test_functions.jl")

# ==================================== C3S1_20221222 ==================================== #
#
# =============== D1
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N06049D1/scenario_001.json"

# =============== D2
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D2/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D2/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D2/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N06049D2/scenario_001.json"

# =============== D3
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D3/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D3/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N04200D3/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N06049D3/scenario_001.json"


# ==================================== C3S3.1_20230606 ==================================== #
#
# =============== D1
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00014D1/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D1/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N01576D1/scenario_007.json"
#path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N02000D1/scenario_001.json"
#path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N08316D1/scenario_001.json"
#path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"
#
# =============== D2
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00014D2/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D2/scenario_001.json"
#
# =============== D3
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00014D3/scenario_001.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D3/scenario_001.json"


# ==================================== C3E2D2_20230510 ==================================== #
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E2D2_20230510/C3E2N01576D2/scenario_002.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E2D2_20230510/C3E2N01576D2/scenario_130.json"
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3E2D2_20230510/C3E2N04224D2/scenario_033.json"

# path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"
# path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
# path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D2/scenario_001.json"

solution_file = "solution.jl"
load_and_project(path, solution_file)

# %% ===========
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"

InFile1 = path
jsn = quasiGrad.load_json(InFile1)
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn)



# %% ============== zsus :) ===========
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N01576D1/scenario_007.json"

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
quasiGrad.snap_shunts!(true, prm, stt, upd)
quasiGrad.write_solution(solution_file, prm, qG, stt, sys)
quasiGrad.post_process_stats(true, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% =====================:
quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
scr[:zsus] = -sum(sum(stt[:zsus_dev][tii] for tii in prm.ts.time_keys))
println(-scr[:zsus])




# %% fix choleky issue
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"

InFile1 = path
jsn     = quasiGrad.load_json(InFile1)

prm, idx, sys = quasiGrad.parse_json(jsn)

# build the qg structure
Div = 1

qG = quasiGrad.initialize_qG(prm, Div=Div)

# intialize (empty) states
bit, cgd, grd, mgd, scr, stt, msc = quasiGrad.initialize_states(idx, prm, sys, qG)

# initialize the states which adam will update -- the rest are fixed
adm = quasiGrad.initialize_adam_states(qG, sys)

# define the states which adam can/will update, and fix the rest!
upd = quasiGrad.identify_update_states(prm, idx, stt, sys)

# initialize the contingency network structure and reusable vectors in dicts
# ntk, flw = quasiGrad.initialize_ctg(sys, prm, qG, idx)

# %% note, the reference bus is always bus #1 ==============================
#
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
ErT = copy(Er')

# get the diagonal admittance matrix   => Ybs == "b susceptance"
Ybs = quasiGrad.spdiagm(ac_b_params)
Yb  = E'*Ybs*E
Ybr = Yb[2:end,2:end]  # use @view ? 

# %% ===
# should we precondition the base case?
#
# Note: Ybr should be sparse!! otherwise,
# the sparse preconditioner won't have any memory limits and
# will be the full Chol-decomp -- not a big deal, I guess..
if qG.base_solver == "pcg"
    if sys.nb <= qG.min_buses_for_krylov
        # too few buses -- use LU
        @warn "Not enough buses for Krylov! Using LU anyways."
    end

    # time is short -- let's jsut always use ldl preconditioner -- it's just as fast
    Ybr_ChPr = quasiGrad.Preconditioners.lldl(Ybr, memory = qG.cutoff_level);

    # OG#2 solution!
        # can we build cholesky?
        # if minimum(ac_b_params) < 0.0
        #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
        # else
        #     @info "Unsing an ldl preconditioner."
        #     Ybr_ChPr = quasiGrad.ldl(Ybr, qG.cutoff_level);
        # end

    # OG#1 solution!
        # # test for negative reactances -- @info "Preconditioning is disabled."
        # if minimum(ac_b_params) < 0.0
        #     # Amrit Pandey: "watch out for negatvive reactance! You will lose
        #     #                pos-sem-def of the Cholesky preconditioner."
        #     abs_b    = abs.(ac_b_params)
        #     abs_Ybr  = (E'*quasiGrad.spdiagm(abs_b)*E)[2:end,2:end] 
        #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(abs_Ybr, qG.cutoff_level)
        # else
        #     Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
        # end
else
    # time is short -- let's jsut always use ldl preconditioner -- it's just as fast
    Ybr_ChPr = quasiGrad.Preconditioners.lldl(Ybr, memory = qG.cutoff_level);
        # # => Ybr_ChPr = quasiGrad.I
        # Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
end

# should we build the cholesky decomposition of the base case
# admittance matrix? we build this to compute high-fidelity
# solutions of the rank-1 update matrices
if qG.build_basecase_cholesky
    if minimum(ac_b_params) < 0.0
        @info "Yb not PSd -- using ldlt (instead of cholesky) to construct WMI update vectors."
        Ybr_Ch = quasiGrad.ldlt(Ybr)
    else
        Ybr_Ch = quasiGrad.cholesky(Ybr)
    end
else
    # this is nonsense
    Ybr_Ch = quasiGrad.I
end

# get the flow matrix
Yfr  = Ybs*Er
YfrT = copy(Yfr')

# build the low-rank contingecy updates
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

# and/or, should we build the low rank ctg elements?
if qG.build_ctg_lowrank == true
    # no need => v_k = Dict(ctg_ii => quasiGrad.spzeros(nac) for ctg_ii in 1:sys.nctg)
    # no need => b_k = Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
    u_k = Dict(ctg_ii => zeros(sys.nb-1) for ctg_ii in 1:sys.nctg) # Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
    g_k = Dict(ctg_ii => 0.0             for ctg_ii in 1:sys.nctg)
    # if the "w_k" formulation is wanted => w_k = Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
else
    v_k = 0
    b_k = 0
end

# %% ===
for ctg_ii in 1:sys.nctg
    println(ctg_ii)
    # components
    cmpnts = prm.ctg.components[ctg_ii]
    for (cmp_ii,cmp) in enumerate(cmpnts)
        # get the cmp index and b
        cmp_index = findfirst(x -> x == cmp, ac_ids) 
        cmp_b     = -ac_b_params[cmp_index] # negative, because we subtract it out

        # output
        ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
        ctg_params[ctg_ii][cmp_ii]  = cmp_b

        # -> y_diag[cmp_index] = sqrt(cmp_b)
            # we record these in ctg
            # ctg_out_ind[ctg_ii]
    end

    # next, should we build the actual, full ctg matrix?
    if qG.build_ctg_full == true
        # direct construction..
        #
        # NOTE: this is written assuming multiple elements can be
        # simultaneously outaged
        Ybs_k = copy(Ybs)
        Ybs_k[CartesianIndex.(tuple.(ctg_out_ind[ctg_ii],ctg_out_ind[ctg_ii]))] .= 0.0
        Ybr_k[ctg_ii] = Er'*Ybs_k*Er
    end

    # and/or, should we build the low rank ctg elements?
    if qG.build_ctg_lowrank == true
        # this code is optimized -- see above for comments
        u_k[ctg_ii] .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
        g_k[ctg_ii]  = -ac_b_params[ctg_out_ind[ctg_ii][1]]/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii]))*-ac_b_params[ctg_out_ind[ctg_ii][1]])
    end
end

# %%
dd = quasiGrad.Preconditioners.lldl(Ybr, memory = 45);
vv = Er[ctg_out_ind[ctg_ii][1],:]
vvv = Vector(Float64.(vv))
@time Ybr_Ch\vv;
@time dd\vvv;

# %%
A = Ybr

LLDL = quasiGrad.Preconditioners.lldl(A, memory = 10)
sol = ones(A.n)
b = A * sol
x = LLDL \ b


# %% ===============
t1 = time()
for ctg_ii in 1:500 #sys.nctg
    println(ctg_ii)
    
    # and/or, should we build the low rank ctg elements?
    if qG.build_ctg_lowrank == true
        # .. vs low rank
        #
        # NOTE: this is written assuming only ONE element
        # can be outaged

        # no need to save:
            # v_k[ctg_ii] =  Er[ctg_out_ind[ctg_ii][1],:]
            # b_k[ctg_ii] = -ac_b_params[ctg_out_ind[ctg_ii][1]]
        #v_k = Er[ctg_out_ind[ctg_ii][1],:]
        #b_k = -ac_b_params[ctg_out_ind[ctg_ii][1]]
        #
        # construction: 
        # 
        # Ybr_k[ctg_ii] = ctg[:Ybr] + v*beta*v'
        #               = ctg[:Ybr] + vLR_k[ctg_ii]*beta*vLR_k[ctg_ii]
        #
        # if v, b saved:
            # u_k[ctg_ii] = Ybr\Array(v_k[ctg_ii])
            # w_k[ctg_ii] = b_k[ctg_ii]*u_k[ctg_ii]/(1+(v_k[ctg_ii]'*u_k[ctg_ii])*b_k[ctg_ii])
        # LU fac => u_k[ctg_ii] = Ybr\Vector(v_k)
            # this is very slow -- we need to us cg and then enforce sparsity!
            # Float64.(Vector(v_k)) is not needed! cg can handle sparse :)
            # quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl=Ybr_ChPr)
        # enforce sparsity -- should be sparse anyways
            # u_k[ctg_ii][abs.(u_k[ctg_ii]) .< 1e-8] .= 0.0

        # we want to sparsify a high-fidelity solution:
        # uk_d = Ybr_Ch\v_k[ctg_ii]
        # quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl=Ybr_ChPr)
        # u_k[ctg_ii] = Ybr\Vector(v_k)
        # u_k[ctg_ii] = C\Vector(v_k)
        u_k[ctg_ii] .= (Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:])
        g_k[ctg_ii] = -ac_b_params[ctg_out_ind[ctg_ii][1]]/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii]))*-ac_b_params[ctg_out_ind[ctg_ii][1]])

        #u_k_local = (Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:])[:]

        # sparsify
        #abs_u_k           = abs.(u_k_local)
        #u_k_ii_SmallToBig = sortperm(abs_u_k)
        #bit_vec           = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> (1.0 - qG.accuracy_sparsify_lr_updates)
        # edge case is caught! bit_vec will never be empty. Say, abs_u_k[u_k_ii_SmallToBig] = [0,0,1], then we have
        # bit_vec = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> 0.01%, say => bit_vec = [0,0,1] 
        # 
        # also, we use ".>" because we only want to include all elements that contribute to meeting the stated accuracy goal
        #u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]] = u_k_local[u_k_ii_SmallToBig[bit_vec]]
        # this is ok, since u_k and w_k have the same sparsity pattern
        # => for the "w_k" formulation: w_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]] = b_k*u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]]/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]]))*b_k)
        #g_k[ctg_ii] = b_k/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii]))*b_k)
    end
end
println(time() - t1)
# %%
u_k[ctg_ii] .= (Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:])

# %%
ctg_ii    = 17
v_k       = Er[ctg_out_ind[ctg_ii][1],:]
u_k_local = (Ybr_Ch\v_k)[:]

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

# compute the constant acline Ybus matrix
Ybus_acline_real, Ybus_acline_imag = quasiGrad.initialize_acline_Ybus(idx, prm, sys)

# %%
v = zeros(10000)
vv = randn(10000)

@btime t= quasiGrad.dot($v,$v)

@btime t = quasiGrad.dot($vv,$vv)
