using quasiGrad
using Revise

include("./test_functions.jl")

# %% ==================================== C3S1_20221222 ==================================== #
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
# path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
# %% path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N00037D2/scenario_001.json"
include("./test_functions.jl")

path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"

solution_file = "solution.jl"
load_and_project(path, solution_file)

# %% ===========
path  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S3.1_20230606/C3S3N23643D1/scenario_001.json"

InFile1 = path
jsn = quasiGrad.load_json(InFile1)
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn)

# %% =======
InFile1 = path
jsn = quasiGrad.load_json(InFile1)

# initialize
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn)

# solve
fix       = true
pct_round = 100.0
quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.solve_power_flow!(bit, cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)

quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
quasiGrad.write_solution(solution_file, prm, qG, stt, sys)
quasiGrad.post_process_stats(true, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)


# %% Test injections!!
tii = :t1

quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii);
Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);

msc[:pinj_ideal][tii]
msc[:qinj_ideal][tii]

msc[:pinj0][tii]
msc[:qinj0][tii]


# %% ============== zsus :) ===========
using quasiGrad
using Revise
include("./test_functions.jl")

path    = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
InFile1 = path
jsn     = quasiGrad.load_json(InFile1)

# initialize
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn, perturb_states=true, pert_size=1.0)

# %% solve
fix       = true
pct_round = 100.0
quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)

# %%
quasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd)

#quasiGrad.reserve_cleanup_parallel!(idx, prm, qG, stt, sys, upd)

# %%
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
quasiGrad.post_process_stats(true, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %% ======================= 
quasiGrad.acline_flows!(bit, grd, idx, msc, prm, qG, stt, sys)
quasiGrad.acline_flows_parallel!(bit, grd, idx, msc, prm, qG, stt, sys)

# %% ===
@time quasiGrad.solve_ctgs!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

@time quasiGrad.solve_ctgs_parallel!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, wct)

# %% =====================:
quasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)
scr[:zsus] = -sum(sum(stt[:zsus_dev][tii] for tii in prm.ts.time_keys))
println(-scr[:zsus])

# %% -- sum
zsus_devs = zeros(2066)
for tii in prm.ts.time_keys
    devs = findall(stt[:zsus_dev][tii] .< 0.0)
    for dev in devs
        dev_id = prm.dev.id[dev]
        vals   = stt[:zsus_dev][tii][dev]
        println("dev: $dev_id, t: $tii, z_sus: $vals")
    end
    zsus_devs .+= stt[:zsus_dev][tii]
end

# %% all
devs = findall(zsus_devs.< 0.0)
println(zsus_devs[devs])
println(prm.dev.id[devs])




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
E  = quasiGrad.build_incidence(idx, prm, stt, sys)
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

# %%

# call the time
t_ind = 1
tii = prm.ts.time_keys[t_ind]

# initialize
run_pf    = true
pf_cnt    = 0  # successes
total_pfs = 0  # successes AND fails

# set a q_margin and a v_margin to make convergence easier (0.0 is nominal)
first_fail = true
q_margin   = 0.0
v_margin   = 0.0

# 1. update the ideal dispatch point (active power) -- we do this just once
    # quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)
    # this is no longer needed, because we penalize device injections directly

# 2. update y_bus and Jacobian and bias point -- this
#    only needs to be done once per time, since xfm/shunt
#    values are not changing between iterations
Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

# loop over pf solves
#while run_pf == true

    # build an empty model!
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(quasiGrad.GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => 1))
    set_string_names_on_creation(model, false)

    # lower the tolerance a bit..
    quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", 1e-5)

    # safety margins should NEVER be 0
    q_margin = max(q_margin, 0.0)
    v_margin = max(v_margin, 0.0)

    # increment
    total_pfs += 1
    pf_cnt    += 1

    # first, rebuild the jacobian, and update the
    # base points: msc[:pinj0], msc[:qinj0]
    Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);

    # define the variables (single time index)
    @variable(model, x_in[1:(2*sys.nb - 1)])
    set_start_value.(x_in, [stt[:vm][tii]; stt[:va][tii][2:end]])

    # assign
    dvm = x_in[1:sys.nb]
    dva = x_in[(sys.nb+1):end]

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
        # now, we need to loop and set the affine expressions to 0, and then add powers
        #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
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

        set_start_value.(pdc_vars, stt[:dc_pfr][tii])
        set_start_value.(qdc_fr_vars, stt[:dc_qfr][tii])
        set_start_value.(qdc_to_vars, stt[:dc_qto][tii])

        # bound dc power
        @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
        @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
        @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

        # loop and add to the nodal injection vectors
        for dcl in 1:sys.nldc
            add_to_expression!(nodal_p[idx.dc_fr_bus[dcl]], -pdc_vars[dcl])
            add_to_expression!(nodal_p[idx.dc_to_bus[dcl]], +pdc_vars[dcl])
            add_to_expression!(nodal_q[idx.dc_fr_bus[dcl]], -qdc_fr_vars[dcl])
            add_to_expression!(nodal_q[idx.dc_to_bus[dcl]], -qdc_to_vars[dcl])
        end
    end
    
    # next, deal with devices
    @variable(model, dev_p_vars[1:sys.ndev])
    @variable(model, dev_q_vars[1:sys.ndev])

    set_start_value.(dev_p_vars, stt[:dev_p][tii])
    set_start_value.(dev_q_vars, stt[:dev_q][tii])

    # call the bounds -- note: this is fairly approximate,
    # since these bounds do not include, e.g., ramp rate constraints
    # between the various time windows -- this will be addressed in the
    # final, constrained power flow solve
    dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
    dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
    dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
    dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]

    # ignore binaries?
    # => dev_plb = prm.dev.p_lb_tmdv[t_ind]
    # => dev_pub = prm.dev.p_ub_tmdv[t_ind]
    # => dev_qlb = prm.dev.q_lb_tmdv[t_ind]
    # => dev_qub = prm.dev.q_ub_tmdv[t_ind]

    # first, define p_on at this time
    # => p_on = dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii]

    # bound
    @constraint(model, dev_plb + stt[:p_su][tii] + stt[:p_sd][tii] .<= dev_p_vars .<= dev_pub + stt[:p_su][tii] + stt[:p_sd][tii])
    # alternative: => @constraint(model, dev_plb .<= dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii] .<= dev_pub)
    @constraint(model, (dev_qlb .- q_margin) .<= dev_q_vars .<= (dev_qub .+ q_margin))

    # apply additional bounds: J_pqe (equality constraints)
    if ~isempty(idx.J_pqe)
        @constraint(model, dev_q_vars[idx.J_pqe] - prm.dev.beta[idx.J_pqe].*dev_p_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe].*stt[:u_sum][tii][idx.J_pqe])
        # alternative: @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt[:u_sum][tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe])
    end

    # apply additional bounds: J_pqmin/max (inequality constraints)
    #
    # note: when the reserve products are negelected, pr and cs constraints are the same
    #   remember: idx.J_pqmax == idx.J_pqmin
    if ~isempty(idx.J_pqmax)
        @constraint(model, dev_q_vars[idx.J_pqmax] .<= prm.dev.q_0_ub[idx.J_pqmax].*stt[:u_sum][tii][idx.J_pqmax] .+ prm.dev.beta_ub[idx.J_pqmax].*dev_p_vars[idx.J_pqmax])
        @constraint(model, prm.dev.q_0_lb[idx.J_pqmax].*stt[:u_sum][tii][idx.J_pqmax] .+ prm.dev.beta_lb[idx.J_pqmax].*dev_p_vars[idx.J_pqmax] .<= dev_q_vars[idx.J_pqmax])
    end

    # great, now just update the nodal injection vectors
    for dev in 1:sys.ndev
        if dev in idx.pr_devs
            # producers
            add_to_expression!(nodal_p[idx.device_to_bus[dev]], dev_p_vars[dev])
            add_to_expression!(nodal_q[idx.device_to_bus[dev]], dev_q_vars[dev])
        else
            # consumers
            add_to_expression!(nodal_p[idx.device_to_bus[dev]], -dev_p_vars[dev])
            add_to_expression!(nodal_q[idx.device_to_bus[dev]], -dev_q_vars[dev])
        end
    end

    # bound system variables ==============================================
    #
    # bound variables -- voltage
    @constraint(model, (prm.bus.vm_lb .- v_margin) - stt[:vm][tii] .<= dvm .<= (prm.bus.vm_ub .+ v_margin) - stt[:vm][tii])
    # alternative: => @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm .<= prm.bus.vm_ub)

    # mapping
    JacP_noref = @view Jac[1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
    JacQ_noref = @view Jac[(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]

    @constraint(model, JacP_noref*x_in + msc[:pinj0][tii] .== nodal_p)
    @constraint(model, JacQ_noref*x_in + msc[:qinj0][tii] .== nodal_q)

    # find the subset of devices with the highest bounds
        # => if (sys.npr > 15) && (sys.ncs > 15)
        # =>     dev_bound_inds = reverse(sortperm(dev_pub - dev_plb))
        # =>     pr_to_vary     = intersect(dev_bound_inds, idx.pr_not_Jpqe)[1:Int(round(sys.npr/10))]
        # =>     cs_to_vary     = intersect(dev_bound_inds, idx.cs_not_Jpqe)[1:Int(round(sys.ncs/10))]
        # => else
        # =>     # just vary them all..
        # =>     pr_to_vary     = idx.pr_devs
        # =>     cs_to_vary     = idx.cs_devs
        # => end

    # objective: hold p (and v?) close to its initial value
    # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
    if qG.Gurobi_pf_obj == "min_dispatch_distance"
        # this finds a solution close to the dispatch point -- does not converge without v,a regularization
        obj = AffExpr(0.0)

        # constraint some devices!
            # => for dev in 1:sys.ndev
            # =>     if (dev in pr_to_vary) || (dev in cs_to_vary)
            # =>         tmp_devp = @variable(model)
            # =>         @constraint(model, stt[:dev_p][tii][dev] - dev_p_vars[dev] <= tmp_devp)
            # =>         @constraint(model, dev_p_vars[dev] - stt[:dev_p][tii][dev] <= tmp_devp)
            # =>         add_to_expression!(obj, tmp_devp, 5.0/(0.1*sys.nb))
            # =>     else
            # =>         @constraint(model, stt[:dev_p][tii][dev] == dev_p_vars[dev])
            # =>     end
            # => end

        # every 10% of devices, introduce a new slack variable -- limit complexity
        num_dev_per_slack  = Int(round(0.1*sys.ndev))
        slack_power_weight = 2.5
        
        tmp_devp = @variable(model)
        add_to_expression!(obj, tmp_devp, slack_power_weight/(sys.nb/num_dev_per_slack))
        for dev in 1:sys.ndev
            if mod(dev,num_dev_per_slack) == 0
                tmp_devp = @variable(model)
                add_to_expression!(obj, tmp_devp, slack_power_weight/(sys.nb/num_dev_per_slack))
            end
            @constraint(model, stt[:dev_p][tii][dev] - dev_p_vars[dev] <= tmp_devp)
            @constraint(model, dev_p_vars[dev] - stt[:dev_p][tii][dev] <= tmp_devp)
        end

        # regularize against voltage movement -- this adds light 
        # regularization and causes convergence
        tmp_p  = @variable(model)
        tmp_q  = @variable(model)
        tmp_vm = @variable(model)
        tmp_va = @variable(model)
        add_to_expression!(obj, tmp_p)
        add_to_expression!(obj, tmp_q)
        add_to_expression!(obj, tmp_vm)
        add_to_expression!(obj, tmp_va)
        for bus in 1:sys.nb
            # voltage regularization
            @constraint(model, -dvm[bus] <= tmp_vm)
            @constraint(model,  dvm[bus] <= tmp_vm)

            # phase regularization
            if bus > 1
                @constraint(model, -dva[bus-1] <= tmp_va)
                @constraint(model,  dva[bus-1] <= tmp_va)
            end

            # for injections:
            @constraint(model, msc[:pinj0][tii][bus] - nodal_p[bus] <= tmp_p)
            @constraint(model, nodal_p[bus] - msc[:pinj0][tii][bus] <= tmp_p)
            @constraint(model, msc[:qinj0][tii][bus] - nodal_q[bus] <= tmp_q)
            @constraint(model, nodal_q[bus] - msc[:qinj0][tii][bus] <= tmp_q)
        end

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
        add_to_expression!(obj, tmp_vm)
        add_to_expression!(obj, tmp_va)
        add_to_expression!(obj, tmp_p, 100.0)
    else
        @warn "pf solver objective not recognized!"
    end

    # set the objective
    @objective(model, Min, obj)

    # solve
    optimize!(model)

    # test solution!
    # soln_valid = solution_status(model)
    
    stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
    stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

    #
    # take the norm of dv and da
    max_dx = maximum(abs.(value.(x_in)))
    
    # println("========================================================")
    if qG.print_linear_pf_iterations == true
        println(termination_status(model), ". time: $(tii). ","objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
    end

    # now, apply the updated injections to the devices
    stt[:dev_p][tii]  = value.(dev_p_vars)
    stt[:p_on][tii]   = stt[:dev_p][tii] - stt[:p_su][tii] - stt[:p_sd][tii]
    stt[:dev_q][tii]  = value.(dev_q_vars)
    if sys.nldc > 0
        stt[:dc_pfr][tii] =  value.(pdc_vars)
        stt[:dc_pto][tii] = -value.(pdc_vars)  # also, performed in clipping
        stt[:dc_qfr][tii] = value.(qdc_fr_vars)
        stt[:dc_qto][tii] = value.(qdc_to_vars)
    end
