using quasiGrad
#using GLMakie
using Revise
using Plots
#using Makie

# call the plotting tools
# include("../core/plotting.jl")

#  ===============
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
#path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D1/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D2/C3S1N00600/scenario_001.json"

InFile1               = path
TimeLimitInSeconds    = 600.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# this is the master function which executes quasiGrad.
# 1. InFile1 -> if string, assume we need to load the jsn data
#
#
# =====================================================\\
# start time
start_time = time()

# I1. load the system data
jsn = quasiGrad.load_json(InFile1)

# I2. initialize the system
adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr,
stt, sys, upd, wct = quasiGrad.base_initialization(jsn, false, 1.0);

# I3. run an economic dispatch and update the states
quasiGrad.economic_dispatch_initialization!(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# L1. run power flow
quasiGrad.solve_power_flow!(cgd, grd, idx, mgd, msc, ntk, prm, qG, stt, sys, upd)

# L2. clean-up reserves by solving softly constrained LP
quasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd)

# =============
qG.adam_max_time = 100.0
qG.alpha_0       = 2.5e-6
quasiGrad.run_adam!(adm, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# %%
quasiGrad.cleanup_pf_with_Gurobi!(idx, msc, ntk, prm, qG, stt, sys)

# compare the current gradient (cg) and the prvious gradient (pg).
# compare the current state and the prvious state.
#
# if cg \approx pg
#   step = 1.5*step
#
# elseif cg > pg
#   step = 1.5*step
#
#
# %%
stt0 = deepcopy(stt)

# %%
#
#
quasiGrad.update_states_and_grads!(cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
println(scr[:zms_penalized])

# %% test solver
function MyJulia1(InFile1::String, TimeLimitInSeconds::Any, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    println("running MyJulia1")
    println("  $(InFile1)")
    println("  $(TimeLimitInSeconds)")
    println("  $(Division)")
    println("  $(NetworkModel)")
    println("  $(AllowSwitching)")

    # how long did package loading take? Give it 20 sec for now..
    NewTimeLimitInSeconds = Float64(TimeLimitInSeconds) - 20.0

    # compute the solution
    quasiGrad.compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)
end

# %%
path                  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
InFile1               = path
TimeLimitInSeconds    = 600.0
NewTimeLimitInSeconds = TimeLimitInSeconds - 35.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

MyJulia1(InFile1, TimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

# %%
# L4. solve Gurobi projection
quasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)

# %% L5. fix binaries which are closest to their Gurobi solutions
pct_round = 100.0
quasiGrad.batch_fix!(pct_round, prm, stt, sys, upd)

# L6. update the state (i.e., apply the projection)
quasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)

# %%
# qG.clip_pq_based_on_bins: should we clip p and q based on the current values of the binaries?
#                           there are pros and cons to both decisions, so it is probably best
#                           to alternate..
t_ind = 5
tii = :t5

#=
if qG.clip_pq_based_on_bins == true
    stt[:p_on][tii] = max.(stt[:p_on][tii], stt[:u_on_dev][tii].*getindex.(prm.dev.p_lb,t_ind))
    stt[:p_on][tii] = min.(stt[:p_on][tii], stt[:u_on_dev][tii].*getindex.(prm.dev.p_ub,t_ind))
else
    # watch out!! 
    # => the absolute bounds are given by 0 < p < p_ub, so this is how we clip!
    stt[:p_on][tii] = max.(stt[:p_on][tii], 0.0)
    stt[:p_on][tii] = min.(stt[:p_on][tii], getindex.(prm.dev.p_ub,t_ind))
end
=#
# clip q -- we clip very simply based on (112), (113), (122), (123), where q_qru is negelcted!
#
if qG.clip_pq_based_on_bins == true
    stt[:dev_q][tii] = max.(stt[:dev_q][tii], stt[:u_sum][tii].*getindex.(prm.dev.q_lb,t_ind))
    stt[:dev_q][tii] = min.(stt[:dev_q][tii], stt[:u_sum][tii].*getindex.(prm.dev.q_ub,t_ind))
else
    # watch out!! 
    # => the absolute bounds are given by q_lb < q < q_ub, but
    # we don't know if q_lb is positive or negative, so to include 0,
    # we take min.(q_lb, 0.0). the upper bound is fine.
    stt[:dev_q][tii] = max.(stt[:dev_q][tii], min.(getindex.(prm.dev.q_lb,t_ind), 0.0))
    stt[:dev_q][tii] = min.(stt[:dev_q][tii], getindex.(prm.dev.q_ub,t_ind))
end

# %%
t_ind = 4
tii = :t4
vals = Float64[]
vals2 = Float64[]
for dev in 1:sys.ndev
    if dev in idx.cs_devs
        push!(vals, stt[:q_qru][tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev])
        push!(vals2, stt[:dev_q][tii][dev] + stt[:q_qrd][tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev])
    else
        push!(vals, 0.0)
        push!(vals2, 0.0)
    end
end

# %%
# a(n unfinished) version where all devices are solved at once!
model = Model(Gurobi.Optimizer; add_bridges = false)
set_string_names_on_creation(model, false)
set_silent(model)

# status update
@info "Running MILP projection across $(sys.ndev) devices."

# %%
quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), 1e-10)
quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), 1e-10)

# %%
get_attribute(model, MOI.RelativeGapTolerance())
get_attribute(model, MOI.AbsoluteGapTolerance())
