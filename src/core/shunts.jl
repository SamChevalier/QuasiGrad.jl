function shunts!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, msc::Dict{Symbol, Vector{Float64}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    for tii in prm.ts.time_keys
        msc[:vm2_sh] .= (@view stt[:vm][tii][idx.shunt_bus]).^2

        # define the shunts values
        msc[:g_tv_shunt] .= prm.shunt.gs .* stt[:u_step_shunt][tii]
        msc[:b_tv_shunt] .= prm.shunt.bs .* stt[:u_step_shunt][tii]
 
        # define the injections
        stt[:sh_p][tii] .=   msc[:g_tv_shunt] .* msc[:vm2_sh]
        stt[:sh_q][tii] .= .-msc[:b_tv_shunt] .* msc[:vm2_sh]

        # evaluate the grd?
        if qG.eval_grad
            # injection gradients
            grd[:sh_p][:vm][tii]         .=  2.0.*msc[:g_tv_shunt] .* (@view stt[:vm][tii][idx.shunt_bus])
            grd[:sh_p][:g_tv_shunt][tii] .=  msc[:vm2_sh]

            grd[:sh_q][:vm][tii]         .= .-2.0.*msc[:b_tv_shunt] .* (@view stt[:vm][tii][idx.shunt_bus])
            grd[:sh_q][:b_tv_shunt][tii] .= .-msc[:vm2_sh]
        end
    end
end