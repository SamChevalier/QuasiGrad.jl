include("../src/quasiGrad_dual.jl")
include("../src/scripts/solver.jl")

# define
s = 2     # set
d = 1     # division
c = 600   # case (num buses)
n = 1     # scenario
action = "just write"

# define the test set
if s == 1
    set = "C3S0_20221208/" 
elseif s == 2
    set = "C3S1_20221222/"
elseif s == 3
    set = "C3S2b_20230316/"
elseif s == 4
    set = "C3E1_20230214/"
end

# define the division
if d == 1
    dvn = "D1/"
elseif d == 2
    dvn = "D2/"
elseif d == 3
    dvn = "D3/"
end

# define the case
if c == 3
    case = "C3S0N00003/"
elseif c == 14
    case = "C3S0N00014/"
elseif c == 37
    case = "C3S0N00037/"
elseif c == 73
    case = "C3S0N00073/"
elseif c == 600
    case = "C3S1N00600/"
elseif c == 1576
    case = "C3S1N01576/"
elseif c == 4200
    case = "C3S1N04200/"
elseif c == 6049
    case = "C3S1N06049/"
end

if n == 1
    nro = "scenario_001.json"
elseif n == 2
    nro = "scenario_002.json"
elseif n == 3
    nro = "scenario_003.json"
end

# load the json
path = "../GO3_testcases/"*set*dvn*case*nro
#path = "../GO3_testcases/C3S1_20221222/D1/C3S1N01576/scenario_001.json"
#path = "../GO3_testcases/C3S1_20221222/D1/C3S1N06049/scenario_001.json"

# call
jsn = quasiGrad.load_json(path)

# %% init
adm, cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, false, 1.0);

# %% solve
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

# grb
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

# soln
soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
quasiGrad.write_solution(path, qG, soln_dict, scr)

# %% write solution, or test solver?
if action == "just write"
    # call
    jsn = quasiGrad.load_json(path)
        
    # initialize the system
    adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
    stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

    # set some qG params
    qG.pqbal_grad_type   = "standard"
    qG.pcg_tol               = 1e-5
    qG.print_final_stats     = true
    qG.constraint_grad_weight                 = 1e-9
    qG.scale_c_pbus_testing  = 1.0
    qG.scale_c_qbus_testing  = 1.0
    qG.scale_c_sflow_testing = 1.0
    # qG.pcg_tol = 1e-5

    # solve
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)
    quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
    quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)
    quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys)

    # write
    soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
    quasiGrad.write_solution(data_dir*file_name, qG, soln_dict, scr)

elseif action == "actually solve"
    # solve
    InFile1               = path
    NewTimeLimitInSeconds = 570.0
    Division              = d
    NetworkModel          = "test"
    AllowSwitching        = 0
    compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
end

# %% === reset
for tii in prm.ts.time_keys
    #stt.u_step_shunt[tii] = zeros(2)
    ctg[:theta_k][tii][end] = zeros(2)
end

# %% ============= scoring junk :)
for tii in prm.ts.time_keys
    scr[:zt_original] += 
    # consumer revenues (POSITIVE)
    sum(stt.zen_dev[tii][dev] for dev in idx.cs_devs) - 
    # producer costs
    sum(stt.zen_dev[tii][dev] for dev in idx.pr_devs) - 
    # startup costs
    sum(stt.zsu_dev[tii]) - 
    sum(stt.zsu_acline[tii]) - 
    sum(stt.zsu_xfm[tii]) - 
    # shutdown costs
    sum(stt.zsd_dev[tii]) - 
    sum(stt.zsd_acline[tii]) - 
    sum(stt.zsd_xfm[tii]) - 
    # on-costs
    sum(stt.zon_dev[tii]) - 
    # time-dependent su costs
    sum(stt.zsus_dev[tii]) - 
    # ac branch overload costs
    sum(stt.zs_acline[tii]) - 
    sum(stt.zs_xfm[tii]) - 
    # local reserve penalties
    sum(stt.zrgu[tii]) -
    sum(stt.zrgd[tii]) -
    sum(stt.zscr[tii]) -
    sum(stt.znsc[tii]) -
    sum(stt.zrru[tii]) -
    sum(stt.zrrd[tii]) -
    sum(stt.zqru[tii]) -
    sum(stt.zqrd[tii]) -
    # power mismatch penalties
    sum(stt.zp[tii]) -
    sum(stt.zq[tii]) -
    # zonal reserve penalties (P)
    sum(stt.zrgu_zonal[tii]) -
    sum(stt.zrgd_zonal[tii]) -
    sum(stt.zscr_zonal[tii]) -
    sum(stt.znsc_zonal[tii]) -
    sum(stt.zrru_zonal[tii]) -
    sum(stt.zrrd_zonal[tii]) -
    # zonal reserve penalties (Q)
    sum(stt.zqru_zonal[tii]) -
    sum(stt.zqrd_zonal[tii])
    # penalized constraints
    scr[:zt_penalty] += -qG.constraint_grad_weight*(
    sum(stt.zhat_mndn[tii]) + 
    sum(stt.zhat_mnup[tii]) + 
    sum(stt.zhat_rup[tii]) + 
    sum(stt.zhat_rd[tii])  + 
    sum(stt.zhat_rgu[tii]) + 
    sum(stt.zhat_rgd[tii]) + 
    sum(stt.zhat_scr[tii]) + 
    sum(stt.zhat_nsc[tii]) + 
    sum(stt.zhat_rruon[tii])  + 
    sum(stt.zhat_rruoff[tii]) +
    sum(stt.zhat_rrdon[tii])  +
    sum(stt.zhat_rrdoff[tii]) +
    # common set of pr and cs constraint variables (see below)
    sum(stt.zhat_pmax[tii])      + 
    sum(stt.zhat_pmin[tii])      + 
    sum(stt.zhat_pmaxoff[tii])   + 
    sum(stt.zhat_qmax[tii])      + 
    sum(stt.zhat_qmin[tii])      + 
    sum(stt.zhat_qmax_beta[tii]) + 
    sum(stt.zhat_qmin_beta[tii]))
end

# %% =========

zp = sum(sum(stt.zp[tii] for tii in prm.ts.time_keys))
zq = sum(sum(stt.zq[tii] for tii in prm.ts.time_keys))

# %%
zen_cs = sum(sum(stt.zen_dev[tii][dev] for dev in idx.cs_devs) for tii in prm.ts.time_keys)
zen_pr = sum(sum(stt.zen_dev[tii][dev] for dev in idx.pr_devs) for tii in prm.ts.time_keys)

# %%

zrgu_zonal = sum(sum(stt.zrgu_zonal[tii]) for tii in prm.ts.time_keys)
zrgd_zonal = sum(sum(stt.zrgd_zonal[tii]) for tii in prm.ts.time_keys)
zscr_zonal = sum(sum(stt.zscr_zonal[tii]) for tii in prm.ts.time_keys)
znsc_zonal = sum(sum(stt.znsc_zonal[tii]) for tii in prm.ts.time_keys)
zrru_zonal = sum(sum(stt.zrru_zonal[tii]) for tii in prm.ts.time_keys)
zrrd_zonal = sum(sum(stt.zrrd_zonal[tii]) for tii in prm.ts.time_keys)
zqru_zonal = sum(sum(stt.zqru_zonal[tii]) for tii in prm.ts.time_keys)
zqrd_zonal = sum(sum(stt.zqrd_zonal[tii]) for tii in prm.ts.time_keys)

sum(sum(stt.zrgd[tii]) for tii in prm.ts.time_keys)
sum(sum(stt.zqrd[tii]) for tii in prm.ts.time_keys)

sum(sum(stt.zon_dev[tii]) for tii in prm.ts.time_keys)


sum(sum(stt.zs_acline[tii]) for tii in prm.ts.time_keys)
sum(sum(stt.zs_xfm[tii]) for tii in prm.ts.time_keys)

# %% ===============
vio_p = zeros(617,18)
vio_q = zeros(617,18)
zqt   = 0.0
for tii in prm.ts.time_keys
    cp         = prm.vio.p_bus * qG.scale_c_pbus_testing
    cq         = prm.vio.q_bus * qG.scale_c_qbus_testing
    dt         = prm.ts.duration[tii]
    vio_p[:,tii] = stt.zp[tii]/(cp*dt)
    vio_q[:,tii] = stt.zq[tii]/(cq*dt)

    zqt = sum(vio_q[:,tii])*cq*0.5
end

# %% call penalty costt
cp       = prm.vio.p_bus * qG.scale_c_pbus_testing
cq       = prm.vio.q_bus * qG.scale_c_qbus_testing
vio_qb   = zeros(3,18)
zq_sc    = 0.0
pb_slack = Vector{Float64}(undef,(sys.nb))
qb_slack = Vector{Float64}(undef,(sys.nb))

# %% loop over each time period and compute the power balance
for tii in prm.ts.time_keys
    # duration
    dt = prm.ts.duration[tii]

    # loop over each bus
    for bus in 1:sys.nb
        # active power balance: stt[:pb_slack][tii][bus] to record with time
        pb_slack[bus] = 
                # consumers (positive)
                sum(stt.dev_p[tii][idx.cs[bus]]; init=0.0) +
                # shunt
                sum(stt.sh_p[tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt.acline_pfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt.acline_pto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt.xfm_pfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt.xfm_pto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
                # dcline
                sum(stt.dc_pfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
                sum(stt.dc_pto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
                # producer (negative)
               -sum(stt.dev_p[tii][idx.pr[bus]]; init=0.0)
        
        # reactive power balance
        qb_slack[bus] = 
                # consumers (positive)
                sum(stt.dev_q[tii][idx.cs[bus]]; init=0.0) +
                # shunt        
                0*sum(stt.sh_q[tii][idx.sh[bus]]; init=0.0) +
                # acline
                sum(stt.acline_qfr[tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                sum(stt.acline_qto[tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                # xfm
                sum(stt.xfm_qfr[tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                sum(stt.xfm_qto[tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
                # dcline
                sum(stt.dc_qfr[tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
                sum(stt.dc_qto[tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
                # producer (negative)
               -sum(stt.dev_q[tii][idx.pr[bus]]; init=0.0)
    end

    # actual mismatch penalty
    vio_qb[:,tii] = qb_slack
    zq_sc += sum(abs.(qb_slack))*cq*dt
end


# %% min -- uptime
for tii in prm.ts.time_keys
    dev       = 2
    T_mnup    = idx.Ts_mnup[dev][tii] # => get_tminup(tii, dev, prm)
    cvio      = max(stt.u_sd_dev[tii][dev] + sum(stt.u_su_dev[tii_inst][dev] for tii_inst in T_mnup; init=0.0) - 1.0 , 0.0)
    println(cvio)
end

# %%
println("start")
for tii in prm.ts.time_keys
    if tii == :t1
        println(stt.u_on_dev[tii][dev])
        println("========")
    else
        dev = 2
        tm1 = prm.ts.tmin1[tii]
        println(stt.u_on_dev[tii][dev])
        println(stt.u_on_dev[tii][dev] - stt.u_on_dev[tm1][dev])
        println(stt.u_su_dev[tii][dev])
        println(stt.u_sd_dev[tii][dev])
        println("========")
    end
end

# %% ===========
sd = min.(stt.u_on_dev[tii] - stt.u_on_dev[prm.ts.tmin1[tii]], 0.0)
#stt.u_sd_dev[tii] = -min.(stt.u_on_dev[tii] - stt.u_on_dev[prm.ts.tmin1[tii]], 0.0)

# %%

idx.Ts_mnup[dev][tii]

# %%
current_start_time = prm.ts.start_time_dict[tii]

# all other times minus d_min -- note: d_up_min = prm.dev.in_service_time_lb[dev]
valid_times = (current_start_time - prm.dev.in_service_time_lb[dev] + quasiGrad.eps_time .< prm.ts.start_time) .&& (prm.ts.start_time .< current_start_time)
t_set       = prm.ts.time_keys[valid_times]

# %%
for tii in prm.ts.time_keys
    dev = 2
    #tm1 = prm.ts.tmin1[tii]
    println([stt.u_su_dev[tii][dev] stt.u_sd_dev[tii][dev]])
    println("========")
end

# %% ===
println("start")
for tii in prm.ts.time_keys
    dev = 2
    println(stt.u_on_dev[tii][dev])
    println("========")
end