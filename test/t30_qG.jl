using quasiGrad
using Revise

# common folder for calling
tfp  = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"

# call the solver!
InFile1 = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json" 
quasiGrad.compute_quasiGrad_solution_practice(InFile1, 1.0, 1, "test", 1)