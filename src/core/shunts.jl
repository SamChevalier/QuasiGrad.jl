function shunts!(grd::quasiGrad.Grad, idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    Threads.@threads for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        stt.vm2_sh[tii] .= (@view stt.vm[tii][idx.shunt_bus]).^2

        # define the shunts values
        stt.g_tv_shunt[tii] .= prm.shunt.gs .* stt.u_step_shunt[tii]
        stt.b_tv_shunt[tii] .= prm.shunt.bs .* stt.u_step_shunt[tii]
 
        # define the injections
        stt.sh_p[tii] .=   stt.g_tv_shunt[tii] .* stt.vm2_sh[tii]
        stt.sh_q[tii] .= .-stt.b_tv_shunt[tii] .* stt.vm2_sh[tii]

        # evaluate the grd?
        if qG.eval_grad
            # injection gradients
            grd.sh_p.vm[tii]         .=  2.0.*stt.g_tv_shunt[tii] .* (@view stt.vm[tii][idx.shunt_bus])
            grd.sh_p.g_tv_shunt[tii] .=  stt.vm2_sh[tii]

            grd.sh_q.vm[tii]         .= .-2.0.*stt.b_tv_shunt[tii] .* (@view stt.vm[tii][idx.shunt_bus])
            grd.sh_q.b_tv_shunt[tii] .= .-stt.vm2_sh[tii]
        end
    end
end