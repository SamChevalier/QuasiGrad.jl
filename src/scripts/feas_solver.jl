function compute_quasiGrad_solution_feas(InFile1::String, NewTimeLimitInSeconds::Float64, Division::Int64, NetworkModel::String, AllowSwitching::Int64)
    jsn = quasiGrad.load_json(InFile1)

    # initialize
    adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn)

    # solve
    fix       = true
    pct_round = 100.0
    quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = false)
    quasiGrad.project!(pct_round, idx, prm, qG, stt, sys, upd, final_projection = true)
    quasiGrad.snap_shunts!(true, prm, stt, upd)
    
    quasiGrad.write_solution("solution.jl", prm, qG, stt, sys)
end