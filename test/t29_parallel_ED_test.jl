using quasiGrad
using Revise

# %% ======================= 617D1 === -- standard ED
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json" 
path = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"

jsn  = quasiGrad.load_json(path)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);
qG.print_projection_success = false

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% ======================= 617D1 === -- parallel ED
tfp  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json" 
path = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"

jsn  = quasiGrad.load_json(path)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);
qG.print_projection_success = false

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);


# %% test -- wmi factor acceleration!!!
Base.GC.gc()

@time quasiGrad.initialize_ctg_test(stt, sys, prm, qG, idx)

# %% =============
E, Efr, Eto = quasiGrad.build_incidence(idx, prm, stt, sys)
Er = E[:,2:end]
ErT = copy(Er')




# %% === ========

# note, the reference bus is always bus #1
#
# first, get the ctg limits
s_max_ctg = [prm.acline.mva_ub_em; prm.xfm.mva_ub_em]

# get the ordered names of all components
ac_ids = [prm.acline.id; prm.xfm.id ]

# get the ordered (negative!!) susceptances
ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]

# build the full incidence matrix: E = lines x buses
E, Efr, Eto = quasiGrad.build_incidence(idx, prm, stt, sys)
Er = E[:,2:end]
ErT = copy(Er')

# get the diagonal admittance matrix   => Ybs == "b susceptance"
Ybs = quasiGrad.spdiagm(ac_b_params)
Yb  = E'*Ybs*E
Ybr = Yb[2:end,2:end]  # use @view ? 

# should we precondition the base case?
#
# Note: Ybr should be sparse!! otherwise,
# the sparse preconditioner won't have any memory limits and
# will be the full Chol-decomp -- not a big deal, I guess..
if qG.base_solver == "pcg"
    if sys.nb <= qG.min_buses_for_krylov
        # too few buses -- use LU
        println("Not enough buses for Krylov! Using LU for all Ax=b system solves.")
    end

    # time is short -- let's jsut always use ldl preconditioner -- it's just as fast
    Ybr_ChPr = quasiGrad.Preconditioners.lldl(Ybr, memory = qG.cutoff_level);
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
#
# build something small of the correct data type
Ybr_k = Dict(1 => quasiGrad.spzeros(1,1))

# and/or, should we build the low rank ctg elements?
#
# no need => v_k = Dict(ctg_ii => quasiGrad.spzeros(nac) for ctg_ii in 1:sys.nctg)
# no need => b_k = Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
u_k = [zeros(sys.nb-1) for ctg_ii in 1:sys.nctg] # Dict(ctg_ii => zeros(sys.nb-1) for ctg_ii in 1:sys.nctg) # Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
g_k = zeros(sys.nctg) # => Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
z_k = [zeros(sys.nac) for ctg_ii in 1:sys.nctg]
# if the "w_k" formulation is wanted => w_k = Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)

# loop over components (see below for comments!!!)
for ctg_ii in 1:sys.nctg
    cmpnts = prm.ctg.components[ctg_ii]
    for (cmp_ii,cmp) in enumerate(cmpnts)
        cmp_index = findfirst(x -> x == cmp, ac_ids) 
        ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
        ctg_params[ctg_ii][cmp_ii]  = -ac_b_params[cmp_index]
    end
end



# %% ==============
ctg_ii = 1
zrs = zeros(sys.nb-1)
zrs .= @view Er[ctg_out_ind[ctg_ii][1],:]
# %%
quasiGrad.cg!(u_k[ctg_ii], Ybr, zrs, abstol = qG.pcg_tol, Pl=Ybr_ChPr, maxiter = qG.max_pcg_its)

# %% ==============
lck = Threads.SpinLock() # => Threads.ReentrantLock(); SpinLock slower, but safer, than ReentrantLock ..?

# define a vector of bools acciated with multithread storage:
ready_to_use = ones(Bool, qG.num_threads+2)
ready_to_use .= true
lck = Threads.SpinLock()

                            ## solve for the wmi factors!
                            #Threads.@threads for ctg_ii in 1:sys.nctg
                            #    # u_k[ctg_ii] .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
                            #
                            #    quasiGrad.cg!(u_k[ctg_ii], Ybr, Er[ctg_out_ind[ctg_ii][1],:], abstol = qG.pcg_tol, Pl=Ybr_ChPr, maxiter = qG.max_pcg_its)
                            #    
                            #end

zrs = zeros(sys.nb-1, qG.num_threads+2)

t1 = time()

Threads.@threads for ctg_ii in 1:sys.nctg
    # use a custom "thread ID" -- three indices: tii, ctg_ii, and thrID
    thrID = Int16(1)
    Threads.lock(lck)
        thrIdx = findfirst(ready_to_use)
        if thrIdx != Nothing
            thrID = Int16(thrIdx)
        end
        ready_to_use[thrID] = false # now in use :)
    Threads.unlock(lck)

    # zrs[thrID] .= @view Er[ctg_out_ind[ctg_ii][1],:]
    
    # this code is optimized -- see above for comments!!!
    # u_k[ctg_ii]        .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
    #quasiGrad.cg!(u_k[ctg_ii], Ybr, zrs[thrID], abstol = qG.pcg_tol, Pl=Ybr_ChPr, maxiter = qG.max_pcg_its)
    #quasiGrad.@turbo g_k[ctg_ii]  = -ac_b_params[ctg_out_ind[ctg_ii][1]]/(1.0+(quasiGrad.dot(Er[ctg_out_ind[ctg_ii][1],:],u_k[ctg_ii]))*-ac_b_params[ctg_out_ind[ctg_ii][1]])
    #quasiGrad.@turbo mul!(z_k[ctg_ii], Yfr, u_k[ctg_ii])

    # all done!!
    Threads.lock(lck)
        ctg.ready_to_use[thrID] = true
    Threads.unlock(lck)
end

t_ctg = time() - t1
println("WMI factor time: $t_ctg")