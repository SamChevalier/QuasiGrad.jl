using quasiGrad
using Revise

# add the "solver"
include("./test_functions.jl")

# test folder path
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# %% ==================================== C3S4X_20230809 ==================================== #
#
# ====================================       D1       ==================================== #
path = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_941.json" # first
solution_file = "C3S4N00617D1_scenario_941_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_963.json" # last
solution_file = "C3S4N00617D1_scenario_963_solution.json"
load_solve_project_write(path, solution_file)

#    ====================================       D2       ==================================== #
path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_991.json" # first
solution_file = "C3S4N00073D2_scenario_991_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_997.json" # last
solution_file = "C3S4N00073D2_scenario_997_solution.json"
load_solve_project_write(path, solution_file)

# ==================================== C3E3.1_20230629 =================================== #
#
# ====================================       D1       ==================================== #
path = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json" 
solution_file = "C3E3N00617D1_scenario_001_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
solution_file = "C3E3N01576D1_scenario_027_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_131.json"
solution_file = "C3E3N04224D1_scenario_131_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N04224D1/scenario_143.json"
solution_file = "C3E3N04224D1_scenario_143_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_031.json"
solution_file = "C3E3N06049D1_scenario_031_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_043.json"
solution_file = "C3E3N06049D1_scenario_043_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N06717D1/scenario_031.json"
solution_file = "C3E3N06717D1_scenario_031_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N06717D1/scenario_048.json"
solution_file = "C3E3N06717D1_scenario_048_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N08316D1/scenario_001.json"
solution_file = "C3E3N08316D1_scenario_001_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D1/C3E3N23643D1/scenario_003.json"
solution_file = "C3E3N23643D1_scenario_003_solution.json"
load_solve_project_write(path, solution_file)

# %% ====================================       D2       ==================================== #
path = tfp*"C3E3.1_20230629/D2/C3E3N00073D2/scenario_231.json"
solution_file = "C3E3N00073D2_scenario_231_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N00073D2/scenario_248.json"
solution_file = "C3E3N00073D2_scenario_248_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N00617D2/scenario_001.json"
solution_file = "C3E3N00617D2_scenario_001_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N01576D2/scenario_027.json"
solution_file = "C3E3N01576D2_scenario_027_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N01576D2/scenario_043.json"
solution_file = "C3E3N01576D2_scenario_043_solution.json"
load_solve_project_write(path, solution_file)

# %% ==================

path = tfp*"C3E3.1_20230629/D2/C3E3N04224D2/scenario_131.json"
solution_file = "C3E3N04224D2_scenario_131_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N04224D2/scenario_148.json"
solution_file = "C3E3N04224D2_scenario_148_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N06049D2/scenario_031.json"
solution_file = "C3E3N06049D2_scenario_031_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N06049D2/scenario_043.json"
solution_file = "C3E3N06049D2_scenario_043_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N08316D2/scenario_001.json"
solution_file = "C3E3N08316D2_scenario_001_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D2/C3E3N23643D2/scenario_003.json"
solution_file = "C3E3N23643D2_scenario_003_solution.json"
load_solve_project_write(path, solution_file)

# ====================================       D3       ==================================== #
path = tfp*"C3E3.1_20230629/D3/C3E3N00073D3/scenario_231.json"
solution_file = "C3E3N00073D3_scenario_231_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N00073D3/scenario_248.json"
solution_file = "C3E3N00073D3_scenario_248_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N00617D3/scenario_001.json"
solution_file = "C3E3N00617D3_scenario_001_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N01576D3/scenario_027.json"
solution_file = "C3E3N01576D3_scenario_027_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N04224D3/scenario_131.json"
solution_file = "C3E3N04224D3_scenario_131_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N04224D3/scenario_143.json"
solution_file = "C3E3N04224D3_scenario_143_solution.json"
load_solve_project_write(path, solution_file)
#=
path = tfp*"C3E3.1_20230629/D3/C3E3N06049D3/scenario_131.json"
solution_file = "C3E3N06049D3_scenario_131_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N06049D3/scenario_143.json"
solution_file = "C3E3N06049D3_scenario_143_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N08316D3/scenario_001.json"
solution_file = "C3E3N08316D3_scenario_001_solution.json"
load_solve_project_write(path, solution_file)

path = tfp*"C3E3.1_20230629/D3/C3E3N23643D3/scenario_003.json"
solution_file = "C3E3N23643D3_scenario_003_solution.json"
load_solve_project_write(path, solution_file)
=#
# %% test -- ED solver! standard method
#=
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N06049D1/scenario_031.json"

InFile1 = path
jsn = quasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)
qG.write_location                = "local"
qG.eval_grad                     = true
qG.always_solve_ctg              = true
qG.skip_ctg_eval                 = false
qG.print_zms                     = false # print zms at every adam iteration?
qG.print_final_stats             = false # print stats at the end?
qG.print_lbfgs_iterations        = false
qG.print_projection_success      = false
qG.print_linear_pf_iterations    = false
qG.print_reserve_cleanup_success = false

quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% ===================
path = tfp*"C3E3.1_20230629/D1/C3E3N06717D1/scenario_048.json"
solution_file = "C3E3N06717D1_scenario_048_solution.json"
load_solve_project_write(path, solution_file)
=#

# %% ==========
path = tfp*"C3E3.1_20230629/D2/C3E3N01576D2/scenario_043.json"
solution_file = "C3E3N01576D2_scenario_043_solution.json"

# load!
InFile1 = path
jsn = quasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

# write locally
qG.write_location   = "local"
qG.eval_grad        = true
qG.always_solve_ctg = true
qG.skip_ctg_eval    = false

# turn off all printing
qG.print_zms                     = false # print zms at every adam iteration?
qG.print_final_stats             = false # print stats at the end?
qG.print_lbfgs_iterations        = false
qG.print_projection_success      = false
qG.print_linear_pf_iterations    = false
qG.print_reserve_cleanup_success = false

# solve
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
quasiGrad.solve_power_flow!(cgd, grd, idx, lbf, mgd, ntk, prm, qG, stt, sys, upd)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)
quasiGrad.snap_shunts!(true, prm, qG, stt, upd)
quasiGrad.post_process_stats(true, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
quasiGrad.write_solution(solution_file, prm, qG, stt, sys)

# %% ====================
stt0 = deepcopy(stt)

# %% =============
for tii in prm.ts.time_keys
    for dev in prm.dev.dev_keys
        stt.zsus_dev_local[tii][dev]  .= 0.0
        stt.u_sus_bnd[tii][dev] .= 0.0
    end
end

# parallel loop over devices
for dev in prm.dev.dev_keys
    # loop over each time period
    for tii in prm.ts.time_keys
        # first, we bound ("bnd") the startup state ("sus"):
        # the startup state can only be active if the device
        # has been on within some recent time period.
        #
        # flush the sus (state)
        stt.zsus_dev[tii][dev] = 0.0

        # loop over sus (i.e., f in F)
        if prm.dev.num_sus[dev] > 0
            for ii in 1:prm.dev.num_sus[dev]
                if tii in idx.Ts_sus_jf[dev][tii][ii]
                    if tii == 1 && prm.dev.init_on_status == 1.0
                        # this is an edge case, where there are no previous states which
                        # could be "on" (since we can't turn on the generator in the fixed
                        # past, and it wasn't on)
                        stt.u_sus_bnd[tii][dev][ii] = 0.0
                    elseif tii == 1 && prm.dev.init_on_status[dev] == 0.0
                        stt.u_sus_bnd[tii][dev][ii] = 0.0
                    else
                        # grab the largest
                        #stt.u_sus_bnd[tii][dev][ii] = quasiGrad.max_binary(dev, idx, ii, stt, tii) 
                        stt.u_sus_bnd[tii][dev][ii] = maximum(stt.u_on_dev_Trx[dev][tij] for tij in idx.Ts_sus_jft[dev][tii][ii])
                    end
                else
                    # ok, in this case the device was on in a sufficiently recent time (based on
                    # startup conditions), so we don't need to compute a bound
                    stt.u_sus_bnd[tii][dev][ii] = 1.0
                end

                # now, compute the discount/cost
                #if stt.u_sus_bnd[tii][dev][ii] > 0.0
                    stt.zsus_dev_local[tii][dev][ii] = prm.dev.startup_states[dev][ii][1]*min(stt.u_su_dev_Trx[dev][tii], stt.u_sus_bnd[tii][dev][ii])
                #end
            end

            # now, we score, and then take a gradient

                ii = argmin(stt.zsus_dev_local[tii][dev])
                # update the score and take the gradient  ==> this is "+=", since it is over all (f in F) states
                stt.zsus_dev[tii][dev] += stt.zsus_dev_local[tii][dev][ii]

        end
    end
end

scr[:zsus] = sum(sum(stt.zsus_dev[tii]) for tii in prm.ts.time_keys)

# %% =========================
# %% -- sum
zsus_devs = zeros(2066)
for tii in prm.ts.time_keys
    devs = findall(stt.zsus_dev[tii] .< 0.0)
    for dev in devs
        dev_id = prm.dev.id[dev]
        vals   = stt.zsus_dev[tii][dev]
        println("dev: $dev_id, t: $tii, z_sus: $vals")
    end
    zsus_devs .+= stt.zsus_dev[tii]
end


# %%
sum(sum(sum.(stt.zsus_dev_local[tii])) for tii in prm.ts.time_keys) 

# %%
gg = zeros(5690)
jj = 1
for dev in prm.dev.dev_keys
    # loop over each time period
    for ii in 1:prm.dev.num_sus[dev]
        gg[jj] = prm.dev.startup_states[dev][ii][1]
        jj = jj + 1
    end
end

# %% ========
model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(quasiGrad.GRB_ENV[]), "OutputFlag" => 0, MOI.Silent() => true, "Threads" => qG.num_threads); add_bridges = false)
z_sus = Vector{AffExpr}(undef, sys.ndev)

# we define these as vectors so we can parallelize safely
for dev in prm.dev.dev_keys
    z_sus[dev]   = AffExpr(0.0)
end

u_sus = Dict{Int32, Vector{Vector{quasiGrad.VariableRef}}}(tii => [@variable(model, [sus = 1:prm.dev.num_sus[dev]], Bin) for dev in 1:sys.ndev] for tii in prm.ts.time_keys)  

for dev in prm.dev.dev_keys
    # loop over each time period and define the hard constraints
    for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]

        # 0. startup states
            if prm.dev.num_sus[dev] > 0
                # 1. here is the cost:
                add_to_expression!(z_sus[dev], sum(u_sus[tii][dev][ii]*prm.dev.startup_states[dev][ii][1] for ii in 1:prm.dev.num_sus[dev]))

                # 2. the device cannot be in a startup state unless it is starting up!
                @constraint(model, sum(u_sus[tii][dev]; init=0.0) <= stt.u_su_dev[tii][dev])

                # 3. make sure the device was "on" in a sufficiently recent time period
                for ii in 1:prm.dev.num_sus[dev] # these are the sus indices
                    if tii in idx.Ts_sus_jf[dev][tii][ii] # do we need the constraint?
                        @constraint(model, u_sus[tii][dev][ii] <= sum(stt.u_on_dev[tij][dev] for tij in idx.Ts_sus_jft[dev][tii][ii]))
                    end
                end
            end
    end
end

obj = -sum(z_sus)

# %% set the final objective
@objective(model, Min, obj)

# solve
optimize!(model)