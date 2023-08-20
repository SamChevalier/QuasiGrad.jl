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
