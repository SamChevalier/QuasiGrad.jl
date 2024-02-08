@testset "test " begin

    # define test
    InFile                = "./data/C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
    NewTimeLimitInSeconds = 600.0
    Division              = 1
    NetworkModel          = "test"
    AllowSwitching        = 0

    jsn = QuasiGrad.load_json(InFile1)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
        QuasiGrad.base_initialization(jsn, Div=Division, hpc_params=true, line_switching=AllowSwitching);
    QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
end