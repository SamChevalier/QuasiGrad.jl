using quasiGrad
using Revise

include("./test_functions.jl")

# ===============
path          = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
#path          = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
solution_file = "solution.jl"

load_and_project(path, solution_file)