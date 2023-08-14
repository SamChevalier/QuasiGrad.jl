using quasiGrad
using Revise

# add the "solver"
include("./test_functions.jl")

# test folder path
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# ==================================== C3S4X_20230809 ==================================== #
#
# ====================================       D1       ==================================== #
path = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_941.json" # first
solution_file = "C3S4N00617D1_scenario_941"
load_solve_project_write(path, solution_file)

path = tfp*"C3S4X_20230809/D1/C3S4N00617D1/scenario_963.json" # last
solution_file = "C3S4N00617D1_scenario_963"
load_solve_project_write(path, solution_file)

#    ====================================       D2       ==================================== #
path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_991.json" # first
solution_file = "C3S4N00073D2_scenario_991"
load_solve_project_write(path, solution_file)

path = tfp*"C3S4X_20230809/D2/C3S4N00073D2/scenario_997.json" # last
solution_file = "C3S4N00073D2_scenario_997"
load_solve_project_write(path, solution_file)


