function shunts!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    for tii in prm.ts.time_keys
        vm  = stt[:vm][tii][idx.shunt_bus]
        vm2 = vm.^2

        # define the shunts values
        g_tv_shunt = prm.shunt.gs.*(stt[:u_step_shunt][tii])
        b_tv_shunt = prm.shunt.bs.*(stt[:u_step_shunt][tii])

        # define the injections
        stt[:sh_p][tii] =  g_tv_shunt .* vm2
        stt[:sh_q][tii] = -b_tv_shunt .* vm2

        # evaluate the grd?
        if qG.eval_grad
            # injection gradients
            grd[:sh_p][:vm][tii]         =  2*g_tv_shunt .* vm
            grd[:sh_p][:g_tv_shunt][tii] =  vm2

            grd[:sh_q][:vm][tii]         = -2*b_tv_shunt .* vm
            grd[:sh_q][:b_tv_shunt][tii] = -vm2
        end
    end
end