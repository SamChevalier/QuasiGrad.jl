using quasiGrad
using Revise

# ======================= 617D1 === -- parallel ED
tfp  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json" 
path = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"
path = tfp*"C3E3.1_20230629/D1/C3E3N08316D1/scenario_001.json"

# %% ===
jsn  = quasiGrad.load_json(path)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, perturb_states=false);
qG.print_projection_success = false

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)
quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

stt0 = deepcopy(stt);

# %% =========
quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% ==============
stt = deepcopy(stt0);

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve=true)

# %% =================
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys);

println(scr[:zp] + scr[:zq])

# %%
vm_scale_t0      = 2.5e-5
va_scale_t0      = 2.5e-5
phi_scale_t0     = 2.5e-5
tau_scale_t0     = 2.5e-5 
dc_scale_t0      = 1e-2
power_scale_t0   = 1e-2
reserve_scale_t0 = 1e-2
bin_scale_t0     = 1e-2 # bullish!!!
qG.alpha_pf_t0 = Dict(:vm    => vm_scale_t0,
               :va     => va_scale_t0,
               # xfm
               :phi    => phi_scale_t0,
               :tau    => tau_scale_t0,
               # dc
               :dc_pfr => dc_scale_t0,
               :dc_qto => dc_scale_t0,
               :dc_qfr => dc_scale_t0,
               # powers -- decay this!!
               :dev_q  => power_scale_t0,
               :p_on   => power_scale_t0,
               # reserves
               :p_rgu     => reserve_scale_t0,
               :p_rgd     => reserve_scale_t0,
               :p_scr     => reserve_scale_t0,
               :p_nsc     => reserve_scale_t0,
               :p_rrd_on  => reserve_scale_t0,
               :p_rrd_off => reserve_scale_t0,
               :p_rru_on  => reserve_scale_t0,
               :p_rru_off => reserve_scale_t0,
               :q_qrd     => reserve_scale_t0,
               :q_qru     => reserve_scale_t0,
               # bins
               :u_on_xfm     => bin_scale_t0,
               :u_on_dev     => bin_scale_t0,
               :u_step_shunt => bin_scale_t0,
               :u_on_acline  => bin_scale_t0)
# %%
#stt = deepcopy(stt0);

# balance: pq = pq0 + J*dva  =>  dva = J\(pq-pq0)
for tii in prm.ts.time_keys[1:5]
    # update the Jacobian
    quasiGrad.update_Ybus!(idx, ntk, prm, stt, sys, tii)
    quasiGrad.build_Jac_and_pq0!(ntk, qG, stt, sys, tii)
    J = ntk.Jac[tii][:, [1:sys.nb; (sys.nb+2):end]]

    # compute injections
    nodal_p = zeros(sys.nb)
    nodal_q = zeros(sys.nb)

    # add dc flow injections
    for dcl in 1:sys.nldc
        nodal_p[idx.dc_fr_bus[dcl]] -= stt.dc_pfr[tii][dcl]
        nodal_p[idx.dc_to_bus[dcl]] += stt.dc_pfr[tii][dcl]
        nodal_q[idx.dc_fr_bus[dcl]] -= stt.dc_qfr[tii][dcl]
        nodal_q[idx.dc_to_bus[dcl]] -= stt.dc_qto[tii][dcl]
    end

    # add device injections
    for dev in idx.pr_devs # producers
        nodal_p[idx.device_to_bus[dev]] += stt.dev_p[tii][dev]
        nodal_q[idx.device_to_bus[dev]] += stt.dev_q[tii][dev]
    end
    for dev in idx.cs_devs # consumers
        nodal_p[idx.device_to_bus[dev]] -= stt.dev_p[tii][dev]
        nodal_q[idx.device_to_bus[dev]] -= stt.dev_q[tii][dev]
    end

    # solve least squares: pq = pq0 + J*dva  =>  dva = J\(pq-pq0)
    b = [nodal_p - stt.pinj0[tii];
         nodal_q - stt.qinj0[tii]]
    dva = J\b

    println(tii)

    # update
    stt.vm[tii]        = stt.vm[tii]        .+ dva[1:sys.nb]
    stt.va[tii][2:end] = stt.va[tii][2:end] .+ dva[(sys.nb+1):end]
end

# %% ==============
quasiGrad.initialize_ctg_lists!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)


# %% === solve pf

stt = deepcopy(stt0);
qG.adam_max_time  = 25.0
qG.max_linear_pfs = 1
qG.run_lbfgs      = false

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = true)


# %% =================
stt = deepcopy(stt0);

qG.skip_ctg_eval = true
qG.adam_max_time  = 100.0
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%





quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)


# %% =================
quasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)
zp = -sum(lbf.zpf[:zp][tii] for tii in prm.ts.time_keys)
zq = -sum(lbf.zpf[:zq][tii] for tii in prm.ts.time_keys)
zs = -sum(lbf.zpf[:zs][tii] for tii in prm.ts.time_keys)
zt = zp + zq + zs

# %%
qG.decay_adam_step             = true
qG.apply_grad_weight_homotopy  = true
qG.homotopy_with_cos_decay     = false
qG.num_threads                 = 10
qG.print_zms                   = true
qG.adam_max_time               = 50.0

qG.skip_ctg_eval = true
quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)


# %%
qG.adam_max_time = 50.0
quasiGrad.run_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)


# %%
vm_scale_t0      = 2e-5
va_scale_t0      = 2e-5
phi_scale_t0     = 2e-5
tau_scale_t0     = 2e-5 
dc_scale_t0      = 100.0*1e-2
power_scale_t0   = 100.0*1e-2
reserve_scale_t0 = 100.0*1e-2
bin_scale_t0     = 100.0*1e-2 # bullish!!!
qG.alpha_tnow = Dict(:vm    => vm_scale_t0,
               :va     => va_scale_t0,
               # xfm
               :phi    => phi_scale_t0,
               :tau    => tau_scale_t0,
               # dc
               :dc_pfr => dc_scale_t0,
               :dc_qto => dc_scale_t0,
               :dc_qfr => dc_scale_t0,
               # powers -- decay this!!
               :dev_q  => power_scale_t0,
               :p_on   => power_scale_t0,
               # reserves
               :p_rgu     => reserve_scale_t0,
               :p_rgd     => reserve_scale_t0,
               :p_scr     => reserve_scale_t0,
               :p_nsc     => reserve_scale_t0,
               :p_rrd_on  => reserve_scale_t0,
               :p_rrd_off => reserve_scale_t0,
               :p_rru_on  => reserve_scale_t0,
               :p_rru_off => reserve_scale_t0,
               :q_qrd     => reserve_scale_t0,
               :q_qru     => reserve_scale_t0,
               # bins
               :u_on_xfm     => bin_scale_t0,
               :u_on_dev     => bin_scale_t0,
               :u_step_shunt => bin_scale_t0,
               :u_on_acline  => bin_scale_t0)

# %%

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = true)




# %% ===
quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = false)

# %% solve adam with homotopy!
stt = deepcopy(stt0);

qG.decay_adam_step             = true
qG.apply_grad_weight_homotopy  = true
qG.homotopy_with_cos_decay     = false
qG.num_threads                 = 10
qG.print_zms                   = true
qG.adam_max_time               = 150.0
qG.take_adam_pf_steps          = false
qG.beta1                       = 0.9
qG.beta2                       = 0.99
qG.ctg_solve_frequency         = 3
qG.always_solve_ctg            = false
qG.skip_ctg_eval               = true
qG.ctg_memory         = 0.15            
qG.one_min_ctg_memory = 1.0 - qG.ctg_memory

quasiGrad.run_adam!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %%
qG.always_solve_ctg            = true
qG.skip_ctg_eval               = false

quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% ==================
# choose adam step sizes (initial)
vm_scale_t0      = 1e-5
va_scale_t0      = 1e-5
phi_scale_t0     = 1e-5
tau_scale_t0     = 1e-5 

dc_scale_t0      = 100.0*1e-2
power_scale_t0   = 100.0*1e-2
reserve_scale_t0 = 100.0*1e-2
bin_scale_t0     = 100.0*1e-2 # bullish!!!
qG.alpha_t0 = Dict(:vm    => vm_scale_t0,
                :va     => va_scale_t0,
                # xfm
                :phi    => phi_scale_t0,
                :tau    => tau_scale_t0,
                # dc
                :dc_pfr => dc_scale_t0,
                :dc_qto => dc_scale_t0,
                :dc_qfr => dc_scale_t0,
                # powers -- decay this!!
                :dev_q  => power_scale_t0,
                :p_on   => power_scale_t0,
                # reserves
                :p_rgu     => reserve_scale_t0,
                :p_rgd     => reserve_scale_t0,
                :p_scr     => reserve_scale_t0,
                :p_nsc     => reserve_scale_t0,
                :p_rrd_on  => reserve_scale_t0,
                :p_rrd_off => reserve_scale_t0,
                :p_rru_on  => reserve_scale_t0,
                :p_rru_off => reserve_scale_t0,
                :q_qrd     => reserve_scale_t0,
                :q_qru     => reserve_scale_t0,
                # bins
                :u_on_xfm     => bin_scale_t0,
                :u_on_dev     => bin_scale_t0,
                :u_step_shunt => bin_scale_t0,
                :u_on_acline  => bin_scale_t0)

# choose adam step sizes (final)
vm_scale_tf      = 1e-7
va_scale_tf      = 1e-7
phi_scale_tf     = 1e-7
tau_scale_tf     = 1e-7

dc_scale_tf      = 1e-5
power_scale_tf   = 1e-5
reserve_scale_tf = 1e-5
bin_scale_tf     = 1e-5 # bullish!!!
qG.alpha_tf = Dict(:vm    => vm_scale_tf,
                :va     => va_scale_tf,
                # xfm
                :phi    => phi_scale_tf,
                :tau    => tau_scale_tf,
                # dc
                :dc_pfr => dc_scale_tf,
                :dc_qto => dc_scale_tf,
                :dc_qfr => dc_scale_tf,
                # powers -- decay this!!
                :dev_q  => power_scale_tf,
                :p_on   => power_scale_tf,
                # reserves
                :p_rgu     => reserve_scale_tf,
                :p_rgd     => reserve_scale_tf,
                :p_scr     => reserve_scale_tf,
                :p_nsc     => reserve_scale_tf,
                :p_rrd_on  => reserve_scale_tf,
                :p_rrd_off => reserve_scale_tf,
                :p_rru_on  => reserve_scale_tf,
                :p_rru_off => reserve_scale_tf,
                :q_qrd     => reserve_scale_tf,
                :q_qru     => reserve_scale_tf,
                # bins
                :u_on_xfm     => bin_scale_tf,
                :u_on_dev     => bin_scale_tf,
                :u_step_shunt => bin_scale_tf,
                :u_on_acline  => bin_scale_tf)





# %% solve pf
stt = deepcopy(stt0);

# %%
tii = 10

qG.num_lbfgs_steps       = 150
qG.initial_pf_lbfgs_step = 0.001
qG.max_linear_pfs        = 2

quasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = true)


# %% test -- wmi factor acceleration!!!
Base.GC.gc()
# %%

@time quasiGrad.initialize_ctg(stt, sys, prm, qG, idx);

# %% =============
E, Efr, Eto = quasiGrad.build_incidence(idx, prm, stt, sys)
Er = E[:,2:end]
ErT = copy(Er')

# %%
if sys.nT == 18
    parallel_runs = 1:1
    t_keys = [prm.ts.time_keys]
elseif sys.nT == 42
    parallel_runs = 1:3
    t_keys = [prm.ts.time_keys[1:14];
              prm.ts.time_keys[15:28];
              prm.ts.time_keys[29:42]]
elseif sys.nT == 48
    parallel_runs = 1:3
    t_keys = [prm.ts.time_keys[1:16];
              prm.ts.time_keys[17:32];
              prm.ts.time_keys[33:48]]
end
# %% ==========================



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

Threads.@threads for ctg_ii in 1:sys.nctg
    # this code is optimized -- see above for comments!!!
    u_k[ctg_ii]        .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
    quasiGrad.@turbo g_k[ctg_ii]  = -ac_b_params[ctg_out_ind[ctg_ii][1]]/(1.0+(quasiGrad.dot(Er[ctg_out_ind[ctg_ii][1],:],u_k[ctg_ii]))*-ac_b_params[ctg_out_ind[ctg_ii][1]])
    quasiGrad.@turbo quasiGrad.mul!(z_k[ctg_ii], Yfr, u_k[ctg_ii])
end


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
u_k = [zeros(sys.nb-1) for ctg_ii in 1:sys.nctg]

zrs = [zeros(sys.nb-1) for ij in 1:(qG.num_threads+2)]
t1 = time()

qG.pcg_tol = 0.0006499230723708769
qG.pcg_tol = 0.006499230723708769

Threads.@threads for ctg_ii in 1:sys.nctg
    # use a custom "thread ID" -- three indices: tii, ctg_ii, and thrID
    thrID = 1
    Threads.lock(lck)
        thrIdx = findfirst(ready_to_use)
        if thrIdx != Nothing
            thrID = thrIdx
        end
        ready_to_use[thrID] = false # now in use :)
    Threads.unlock(lck)

    zrs[thrID] .= @view Er[ctg_out_ind[ctg_ii][1],:]
    
    # this code is optimized -- see above for comments!!!
    #u_k[ctg_ii]        .= Ybr_Ch\Er[ctg_out_ind[ctg_ii][1],:]
    quasiGrad.cg!(u_k[ctg_ii], Ybr, zrs[thrID], abstol = qG.pcg_tol, Pl=Ybr_ChPr, maxiter = qG.max_pcg_its)
    quasiGrad.@turbo g_k[ctg_ii]  = -ac_b_params[ctg_out_ind[ctg_ii][1]]/(1.0+(quasiGrad.dot(Er[ctg_out_ind[ctg_ii][1],:],u_k[ctg_ii]))*-ac_b_params[ctg_out_ind[ctg_ii][1]])
    quasiGrad.@turbo quasiGrad.mul!(z_k[ctg_ii], Yfr, u_k[ctg_ii])

    # all done!!
    Threads.lock(lck)
        ready_to_use[thrID] = true
    Threads.unlock(lck)
end

t_ctg = time() - t1
println("WMI factor time: $t_ctg")

# %%
if sys.nT == 18
    parallel_runs = 1:1
    t_keys = [prm.ts.time_keys]
elseif sys.nT == 42
    parallel_runs = 1:3
    t_keys = [prm.ts.time_keys[1:14];
              prm.ts.time_keys[15:28];
              prm.ts.time_keys[29:42]]
elseif sys.nT == 48
    parallel_runs = 1:3
    t_keys = [prm.ts.time_keys[1:16];
              prm.ts.time_keys[17:32];
              prm.ts.time_keys[33:48]]
end

# %% 

function mydotavx(a::Vector{Float64}, b::Vector{Float64})
    s = 0.0
    quasiGrad.@turbo for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end

# %% ===
a = randn(23000)
b = randn(23000)
@btime quasiGrad.dot($a, $b);
@btime mydotavx($a, $b);

# %% ===========
tii = Int8(1)
ctg_ii = 1360
# apply WMI update! don't use function -- it allocates: ctg.theta_k[thrID] .= wmi_update(flw.theta[tii], ntk.u_k[ctg_ii], ntk.g_k[ctg_ii], flw.c[tii])
quasiGrad.@turbo ctg.theta_k[thrID] .= flw.theta[tii] .- ntk.u_k[ctg_ii].*(ntk.g_k[ctg_ii]*quasiGrad.dot(ntk.u_k[ctg_ii], flw.c[tii]))
quasiGrad.@turbo quasiGrad.mul!(ctg.pflow_k[thrID], ntk.Yfr, ctg.theta_k[thrID])
quasiGrad.@turbo ctg.pflow_k[thrID] .+= flw.bt[tii]
quasiGrad.@turbo ctg.sfr[thrID]     .= quasiGrad.LoopVectorization.sqrt_fast.(flw.qfr2[tii] .+ quasiGrad.LoopVectorization.pow_fast.(ctg.pflow_k[thrID],2))
quasiGrad.@turbo ctg.sto[thrID]     .= quasiGrad.LoopVectorization.sqrt_fast.(flw.qto2[tii] .+ quasiGrad.LoopVectorization.pow_fast.(ctg.pflow_k[thrID],2))
quasiGrad.@turbo ctg.sfr_vio[thrID] .= ctg.sfr[thrID] .- ntk.s_max
quasiGrad.@turbo ctg.sto_vio[thrID] .= ctg.sto[thrID] .- ntk.s_max
ctg.sfr_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0
ctg.sto_vio[thrID][ntk.ctg_out_ind[ctg_ii]] .= 0.0

# %%

m1 = max.(ctg.sfr_vio[thrID], ctg.sto_vio[thrID], 0.0)
# %%

# %%
first_solve = true


    tii = Int8(1)

        # initialize
        first_iteration = true # only used when "first solve" == true
        run_pf          = true # used to kill the pf iterations
        pf_itr_cnt      = 0    # total number of successes

        # update y_bus -- this only needs to be done once per time, 
        # since xfm/shunt values are not changing between iterations
        quasiGrad.update_Ybus!(idx, ntk, prm, stt, sys, tii)


            quasiGrad.update_Yflow!(idx, ntk, prm, stt, sys, tii)

            t1 = time()

            # build an empty model! lowering the tolerance doesn't seem to help!
            model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(quasiGrad.GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => 1))
            set_string_names_on_creation(model, false)

            # increment
            pf_itr_cnt += 1

            # first, rebuild jacobian, and update base points: stt.pinj0, stt.qinj0
            quasiGrad.build_Jac_and_pq0!(ntk, qG, stt, sys, tii)
            if first_solve == true
                # flows are only constrained in the first solve
                quasiGrad.build_Jac_sfr_and_sfr0!(idx, ntk, prm, stt, sys, tii)
            end

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])

            # assign
            dvm = @view x_in[1:sys.nb]
            dva = @view x_in[(sys.nb+1):end]
            set_start_value.(dvm, stt.vm[tii])
            set_start_value.(dva, @view stt.va[tii][2:end])

            # voltage penalty -- penalizes voltages out-of-bounds
            if first_solve == true
                @variable(model, vm_penalty[1:sys.nb])
            end

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

                set_start_value.(pdc_vars, stt.dc_pfr[tii])
                set_start_value.(qdc_fr_vars, stt.dc_qfr[tii])
                set_start_value.(qdc_to_vars, stt.dc_qto[tii])

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
            set_start_value.(dev_p_vars, stt.dev_p[tii])
            set_start_value.(dev_q_vars, stt.dev_q[tii])

            # call the bounds -- note: this is fairly approximate,
            # since these bounds do not include, e.g., ramp rate constraints
            # between the various time windows -- this will be addressed in the
            # final, constrained power flow solve                            # => ignore binaries?
                # => stt.dev_plb[tii] .= stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii]    # => stt.dev_plb[tii] .= prm.dev.p_lb_tmdv[tii]
                # => stt.dev_pub[tii] .= stt.u_on_dev[tii].*prm.dev.p_ub_tmdv[tii]    # => stt.dev_pub[tii] .= prm.dev.p_ub_tmdv[tii]
                # => stt.dev_qlb[tii] .= stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]       # => stt.dev_qlb[tii] .= prm.dev.q_lb_tmdv[tii]
                # => stt.dev_qub[tii] .= stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]       # => stt.dev_qub[tii] .= prm.dev.q_ub_tmdv[tii]

            if first_solve == true
                # ignore binaries
                stt.dev_plb[tii] .= 0.0
                stt.dev_pub[tii] .= prm.dev.p_ub_tmdv[tii]
                stt.dev_qlb[tii] .= min.(0.0, prm.dev.q_lb_tmdv[tii])
                stt.dev_qub[tii] .= max.(0.0, prm.dev.q_ub_tmdv[tii])
            else
                # later on, bound power based on binary values!
                stt.dev_plb[tii] .= stt.u_on_dev[tii].*prm.dev.p_lb_tmdv[tii]
                stt.dev_pub[tii] .= stt.u_on_dev[tii].*prm.dev.p_ub_tmdv[tii]
                stt.dev_qlb[tii] .= stt.u_sum[tii].*prm.dev.q_lb_tmdv[tii]   
                stt.dev_qub[tii] .= stt.u_sum[tii].*prm.dev.q_ub_tmdv[tii]   
            end

            # first, define p_on at this time
                # => p_on = dev_p_vars - stt.p_su[tii] - stt.p_sd[tii]
            @constraint(model, stt.dev_plb[tii] .+ stt.p_su[tii] .+ stt.p_sd[tii] .<= dev_p_vars .<= stt.dev_pub[tii] .+ stt.p_su[tii] .+ stt.p_sd[tii])
            @constraint(model, stt.dev_qlb[tii] .<= dev_q_vars .<= stt.dev_qub[tii])

            # apply additional bounds: J_pqe (equality constraints)
            for dev in idx.J_pqe
                @constraint(model, dev_q_vars[dev] - prm.dev.beta[dev]*dev_p_vars[dev] == prm.dev.q_0[dev]*stt.u_sum[tii][dev])
            end

            # apply additional bounds: J_pqmin/max (inequality constraints)
            #
            # note: when the reserve products are negelected, pr and cs constraints are the same
            #   remember: idx.J_pqmax == idx.J_pqmin
            for dev in idx.J_pqmax
                @constraint(model, dev_q_vars[dev] <= prm.dev.q_0_ub[dev]*stt.u_sum[tii][dev] + prm.dev.beta_ub[dev]*dev_p_vars[dev])
                @constraint(model, prm.dev.q_0_lb[dev]*stt.u_sum[tii][dev] + prm.dev.beta_lb[dev]*dev_p_vars[dev] <= dev_q_vars[dev])
            end

            # great, now just update the nodal injection vectors
            for dev in idx.pr_devs # producers
                add_to_expression!(nodal_p[idx.device_to_bus[dev]], dev_p_vars[dev])
                add_to_expression!(nodal_q[idx.device_to_bus[dev]], dev_q_vars[dev])
            end
            for dev in idx.cs_devs # consumers
                add_to_expression!(nodal_p[idx.device_to_bus[dev]], -dev_p_vars[dev])
                add_to_expression!(nodal_q[idx.device_to_bus[dev]], -dev_q_vars[dev])
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            if first_solve == true
                @constraint(model, prm.bus.vm_lb .- vm_penalty .<= stt.vm[tii] .+ dvm)
                @constraint(model,                                 stt.vm[tii] .+ dvm .<= prm.bus.vm_ub .+ vm_penalty)
            else
                @constraint(model, prm.bus.vm_lb               .<= stt.vm[tii] .+ dvm)
                @constraint(model,                                 stt.vm[tii] .+ dvm .<= prm.bus.vm_ub)
            end

            # mapping
            JacP_noref      = ntk.Jac[tii][1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
            JacQ_noref      = ntk.Jac[tii][(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]
            @constraint(model, JacP_noref*x_in .+ stt.pinj0[tii] .== nodal_p)
            @constraint(model, JacQ_noref*x_in .+ stt.qinj0[tii] .== nodal_q)

            # finally, bound the apparent power flow in the lines and transformers -- only do
            # this on the first solve, though!!
            if first_solve == true
                JacSfr_acl_noref = ntk.Jac_sflow_fr[tii][1:sys.nl,       [1:sys.nb; (sys.nb+2):end]]
                JacSfr_xfm_noref = ntk.Jac_sflow_fr[tii][(sys.nl+1):end, [1:sys.nb; (sys.nb+2):end]]
                @constraint(model, JacSfr_acl_noref*x_in + stt.acline_sfr[tii] .<= 1.15 .* prm.acline.mva_ub_nom)
                @constraint(model, JacSfr_xfm_noref*x_in + stt.xfm_sfr[tii]    .<= 1.15 .* prm.xfm.mva_ub_nom)
            end
            
            if first_solve == true
                if first_iteration == true
                    first_iteration = false # toggle
                    # on the first run, just find a pf solution for active power, and
                    # penalize any voltage violation
                    obj = @expression(model, 
                            1e3*(vm_penalty'*vm_penalty) + x_in'*x_in +
                            # => (stt.pinj0[tii] .- nodal_p)'*(stt.pinj0[tii] .- nodal_p) + 
                            # => (stt.qinj0[tii] .- nodal_q)'*(stt.qinj0[tii] .- nodal_q) + 
                            (stt.dev_q[tii] .- dev_q_vars)'*(stt.dev_q[tii] .- dev_q_vars) +
                            10.0*(stt.dev_p[tii] .- dev_p_vars)'*(stt.dev_p[tii] .- dev_p_vars))
                else 
                    obj = @expression(model, 
                            50.0*(vm_penalty'*vm_penalty) + x_in'*x_in +
                            # => (stt.pinj0[tii] .- nodal_p)'*(stt.pinj0[tii] .- nodal_p) + 
                            # => (stt.qinj0[tii] .- nodal_q)'*(stt.qinj0[tii] .- nodal_q) + 
                            (stt.dev_q[tii] .- dev_q_vars)'*(stt.dev_q[tii] .- dev_q_vars) +
                            10.0*(stt.dev_p[tii] .- dev_p_vars)'*(stt.dev_p[tii] .- dev_p_vars))
                end
            else
                obj = @expression(model, 
                    x_in'*x_in +
                    # => (stt.pinj0[tii] .- nodal_p)'*(stt.pinj0[tii] .- nodal_p) + 
                    # => (stt.qinj0[tii] .- nodal_q)'*(stt.qinj0[tii] .- nodal_q) + 
                    (stt.dev_q[tii] .- dev_q_vars)'*(stt.dev_q[tii] .- dev_q_vars) +
                    10.0*(stt.dev_p[tii] .- dev_p_vars)'*(stt.dev_p[tii] .- dev_p_vars))
            end

            # set the objective
            @objective(model, Min, obj)

            # solve --- 
            build_time = time() - t1
            t2 = time()
            optimize!(model)
            solve_time = time() - t2

            # print time stats
            if qG.print_linear_pf_iterations == true
                println("build time: ", round(build_time, sigdigits = 4), ". solve time: ", round(solve_time, sigdigits = 4))
            end

# %%
quasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys; first_solve = false)

# %%
# using Plots
# x = -1:0.01:1
# beta  = exp.(3.0.*x)./(0.6 .+ exp.(3.0.*x))
# 
# Plots.plot!(x, 1 .- beta)

# %%
Mpf = quasiGrad.spdiagm(sys.nac, sys.nac, stt.pflow_over_sflow_fr[tii])
Mqf = quasiGrad.spdiagm(sys.nac, sys.nac, stt.qflow_over_sflow_fr[tii])

tt = Mpf*(ntk.Jac_pq_flow_fr[tii][1:sys.nac      , 1:sys.nb]) + Mqf*(ntk.Jac_pq_flow_fr[tii][(sys.nac+1):end, 1:sys.nb])

rr = Mpf*(ntk.Jac_pq_flow_fr[tii][1:sys.nac      , (sys.nb+1):end]) + Mqf*(ntk.Jac_pq_flow_fr[tii][(sys.nac+1):end, (sys.nb+1):end])

# %%
for tii in prm.ts.time_keys
    ntk.Jac_sflow_fr[tii] = quasiGrad.spzeros(sys.nac, 2*sys.nb)
end

# %% ==============
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys);

println(sum(sum(stt.zhat_mndn[tii]) for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_mnup[tii]) for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_rup[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_rd[tii])  for tii in prm.ts.time_keys))  
println(sum(sum(stt.zhat_rgu[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_rgd[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_scr[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_nsc[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_rruon[tii])   for tii in prm.ts.time_keys))  
println(sum(sum(stt.zhat_rruoff[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_rrdon[tii])   for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_rrdoff[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_pmax[tii])    for tii in prm.ts.time_keys))    
println(sum(sum(stt.zhat_pmin[tii])    for tii in prm.ts.time_keys))    
println(sum(sum(stt.zhat_pmaxoff[tii])    for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_qmax[tii])    for tii in prm.ts.time_keys))    
println(sum(sum(stt.zhat_qmin[tii])  for tii in prm.ts.time_keys))      
println(sum(sum(stt.zhat_qmax_beta[tii])  for tii in prm.ts.time_keys)) 
println(sum(sum(stt.zhat_qmin_beta[tii]) for tii in prm.ts.time_keys)) 