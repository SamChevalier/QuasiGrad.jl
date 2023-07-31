function shunts!(grd::quasiGrad.Grad, idx::quasiGrad.Index, msc::quasiGrad.Msc, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State)
    @batch per=thread for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        msc.vm2_sh[tii] .= (@view stt.vm[tii][idx.shunt_bus]).^2

        # define the shunts values
        msc.g_tv_shunt[tii] .= prm.shunt.gs .* stt.u_step_shunt[tii]
        msc.b_tv_shunt[tii] .= prm.shunt.bs .* stt.u_step_shunt[tii]
 
        # define the injections
        stt.sh_p[tii] .=   msc.g_tv_shunt[tii] .* msc.vm2_sh[tii]
        stt.sh_q[tii] .= .-msc.b_tv_shunt[tii] .* msc.vm2_sh[tii]

        # evaluate the grd?
        if qG.eval_grad
            # injection gradients
            grd.sh_p.vm[tii]         .=  2.0.*msc.g_tv_shunt[tii] .* (@view stt.vm[tii][idx.shunt_bus])
            grd.sh_p.g_tv_shunt[tii] .=  msc.vm2_sh[tii]

            grd.sh_q.vm[tii]         .= .-2.0.*msc.b_tv_shunt[tii] .* (@view stt.vm[tii][idx.shunt_bus])
            grd.sh_q.b_tv_shunt[tii] .= .-msc.vm2_sh[tii]
        end
    end

    # sleep tasks
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
end