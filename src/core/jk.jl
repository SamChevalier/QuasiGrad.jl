# %%
# ac line flows
function acline_flows_st!(bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Idx, msc::quasiGrad.Msc, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    # line parameters
    g_sr = prm.acline.g_sr
    b_sr = prm.acline.b_sr
    b_ch = prm.acline.b_ch
    g_fr = prm.acline.g_fr
    b_fr = prm.acline.b_fr
    g_to = prm.acline.g_to
    b_to = prm.acline.b_to

    # call penalty costs
    cs = prm.vio.s_flow * qG.scale_c_sflow_testing

    # loop over time
    #@floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
    for tii in prm.ts.time_keys

        # duration
        dt = prm.ts.duration[tii]

        # organize relevant line values
        vm_fr = @view stt_im.vm[tii][idx.acline_fr_bus]
        va_fr = @view stt_im.va[tii][idx.acline_fr_bus]
        vm_to = @view stt_im.vm[tii][idx.acline_to_bus]
        va_to = @view stt_im.va[tii][idx.acline_to_bus]
        
        # tools
        msc_im.cos_ftp[tii]  .= cos.(va_fr .- va_to)
        msc_im.sin_ftp[tii]  .= sin.(va_fr .- va_to)
        msc_im.vff[tii]      .= vm_fr.^2
        msc_im.vtt[tii]      .= vm_to.^2
        msc_im.vft[tii]      .= vm_fr.*vm_to
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        msc_im.pfr[tii] .= (g_sr.+g_fr).*msc_im.vff[tii] .+ (.-g_sr.*msc_im.cos_ftp[tii] .- b_sr.*msc_im.sin_ftp[tii]).*msc_im.vft[tii]
        stt_im.acline_pfr[tii] .= stt_im.u_on_acline[tii].*msc_im.pfr[tii]
        
        # reactive power flow -- from -> to
        msc_im.qfr[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*msc_im.vff[tii] .+ (b_sr.*msc_im.cos_ftp[tii] .- g_sr.*msc_im.sin_ftp[tii]).*msc_im.vft[tii]
        stt_im.acline_qfr[tii] .= stt_im.u_on_acline[tii].*msc_im.qfr[tii]
        
        # apparent power flow -- to -> from
        msc_im.acline_sfr[tii] .= sqrt.(stt_im.acline_pfr[tii].^2 .+ stt_im.acline_qfr[tii].^2)
        
        # active power flow -- to -> from
        msc_im.pto[tii] .= (g_sr.+g_to).*msc_im.vtt[tii] .+ (.-g_sr.*msc_im.cos_ftp[tii] .+ b_sr.*msc_im.sin_ftp[tii]).*msc_im.vft[tii]
        stt_im.acline_pto[tii] .= stt_im.u_on_acline[tii].*msc_im.pto[tii]
        
        # reactive power flow -- to -> from
        msc_im.qto[tii] .= (.-b_sr.-b_to.-b_ch./2.0).*msc_im.vtt[tii] .+ (b_sr.*msc_im.cos_ftp[tii] .+ g_sr.*msc_im.sin_ftp[tii]).*msc_im.vft[tii]
        stt_im.acline_qto[tii] .= stt_im.u_on_acline[tii].*msc_im.qto[tii]

        # apparent power flow -- to -> from
        msc_im.acline_sto[tii] .= sqrt.(stt_im.acline_pto[tii].^2 .+ stt_im.acline_qto[tii].^2)
        
        # penalty functions and scores
        msc_im.acline_sfr_plus[tii] .= msc_im.acline_sfr[tii] .- prm.acline.mva_ub_nom
        msc_im.acline_sto_plus[tii] .= msc_im.acline_sto[tii] .- prm.acline.mva_ub_nom
        stt_im.zs_acline[tii]  .= (dt*cs).*max.(msc_im.acline_sfr_plus[tii], msc_im.acline_sto_plus[tii], 0.0)
        
        # ====================================================== #
        # ====================================================== #
        #
        # evaluate the grd?
        if qG.eval_grad

            # Gradients: active power flow -- from -> to
            grd.acline_pfr.vmfr[tii] .= stt_im.u_on_acline[tii].*(2.0.*(g_sr.+g_fr).*vm_fr .+ 
                    (.-g_sr.*msc_im.cos_ftp[tii] .- b_sr.*msc_im.sin_ftp[tii]).*vm_to)
            grd.acline_pfr.vmto[tii] .= stt_im.u_on_acline[tii].*(
                    (.-g_sr.*msc_im.cos_ftp[tii] .- b_sr.*msc_im.sin_ftp[tii]).*vm_fr)
            grd.acline_pfr.vafr[tii] .= stt_im.u_on_acline[tii].*(
                    (g_sr.*msc_im.sin_ftp[tii] .- b_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            grd.acline_pfr.vato[tii] .= stt_im.u_on_acline[tii].*(
                    (.-g_sr.*msc_im.sin_ftp[tii] .+ b_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            if qG.change_ac_device_bins
                grd.acline_pfr.uon[tii] .= msc_im.pfr[tii]   
            end
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            
            grd.acline_qfr.vmfr[tii] .= stt_im.u_on_acline[tii].*(2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*vm_fr .+
                    (b_sr.*msc_im.cos_ftp[tii] .- g_sr.*msc_im.sin_ftp[tii]).*vm_to)
            grd.acline_qfr.vmto[tii] .= stt_im.u_on_acline[tii].*(
                    (b_sr.*msc_im.cos_ftp[tii] .- g_sr.*msc_im.sin_ftp[tii]).*vm_fr)
            grd.acline_qfr.vafr[tii] .= stt_im.u_on_acline[tii].*(
                    (.-b_sr.*msc_im.sin_ftp[tii] .- g_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            grd.acline_qfr.vato[tii] .= stt_im.u_on_acline[tii].*(
                    (b_sr.*msc_im.sin_ftp[tii] .+ g_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            if qG.change_ac_device_bins
                grd.acline_qfr.uon[tii] .= msc_im.qfr[tii] 
            end
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
                            
            grd.acline_pto.vmfr[tii] .= stt_im.u_on_acline[tii].*( 
                    (.-g_sr.*msc_im.cos_ftp[tii] .+ b_sr.*msc_im.sin_ftp[tii]).*vm_to)
            grd.acline_pto.vmto[tii] .= stt_im.u_on_acline[tii].*(2.0.*(g_sr.+g_to).*vm_to .+
                    (.-g_sr.*msc_im.cos_ftp[tii] .+ b_sr.*msc_im.sin_ftp[tii]).*vm_fr)
            grd.acline_pto.vafr[tii] .= stt_im.u_on_acline[tii].*(
                    (g_sr.*msc_im.sin_ftp[tii] .+ b_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            grd.acline_pto.vato[tii] .= stt_im.u_on_acline[tii].*(
                    (.-g_sr.*msc_im.sin_ftp[tii] .- b_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            if qG.change_ac_device_bins
                grd.acline_pto.uon[tii] .= msc_im.pto[tii]
            end
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            
            grd.acline_qto.vmfr[tii] .= stt_im.u_on_acline[tii].*(
                    (b_sr.*msc_im.cos_ftp[tii] .+ g_sr.*msc_im.sin_ftp[tii]).*vm_to)
            grd.acline_qto.vmto[tii] .= stt_im.u_on_acline[tii].*(2.0.*(.-b_sr.-b_to.-b_ch./2.0).*vm_to .+
                    (b_sr.*msc_im.cos_ftp[tii] .+ g_sr.*msc_im.sin_ftp[tii]).*vm_fr)
            grd.acline_qto.vafr[tii] .= stt_im.u_on_acline[tii].*(
                    (.-b_sr.*msc_im.sin_ftp[tii] .+ g_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            grd.acline_qto.vato[tii] .= stt_im.u_on_acline[tii].*(
                    (b_sr.*msc_im.sin_ftp[tii] .- g_sr.*msc_im.cos_ftp[tii]).*msc_im.vft[tii])
            if qG.change_ac_device_bins
                grd.acline_qto.uon[tii] .= msc_im.qto[tii] 
            end

            # apply gradients
            grd.zs_acline.acline_pfr[tii] .= 0.0
            grd.zs_acline.acline_qfr[tii] .= 0.0
            grd.zs_acline.acline_pto[tii] .= 0.0
            grd.zs_acline.acline_qto[tii] .= 0.0  

            # indicators
            # => slower :( quasiGrad.get_largest_indices(msc, bit, :acline_sfr_plus, :acline_sto_plus)
            bit.acline_sfr_plus[tii] .= (msc_im.acline_sfr_plus[tii] .> 0.0) .&& (msc_im.acline_sfr_plus[tii] .> msc_im.acline_sto_plus[tii]);
            bit.acline_sto_plus[tii] .= (msc_im.acline_sto_plus[tii] .> 0.0) .&& (msc_im.acline_sto_plus[tii] .> msc_im.acline_sfr_plus[tii]); 
            #
            # slower alternative
                # => max_sfst0 = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(msc_im.acline_sfr_plus[tii], msc_im.acline_sto_plus[tii])]
                # => ind_fr = max_sfst0 .== 1
                # => ind_to = max_sfst0 .== 2

            # flush -- set to 0 => [Not(ind_fr)] and [Not(ind_to)] was slow/memory inefficient
            grd.zs_acline.acline_pfr[tii] .= 0.0
            grd.zs_acline.acline_qfr[tii] .= 0.0
            grd.zs_acline.acline_pto[tii] .= 0.0
            grd.zs_acline.acline_qto[tii] .= 0.0

            if qG.acflow_grad_is_soft_abs
                # compute the scaled gradients
                if sum(bit.acline_sfr_plus[tii]) > 0
                    msc_im.acline_scale_fr[tii][bit.acline_sfr_plus[tii]]             .= msc_im.acline_sfr_plus[tii][bit.acline_sfr_plus[tii]]./sqrt.(msc_im.acline_sfr_plus[tii][bit.acline_sfr_plus[tii]].^2 .+ qG.acflow_grad_eps2);
                    grd.zs_acline.acline_pfr[tii][bit.acline_sfr_plus[tii]] .= msc_im.acline_scale_fr[tii][bit.acline_sfr_plus[tii]].*((dt*qG.acflow_grad_weight).*stt_im.acline_pfr[tii][bit.acline_sfr_plus[tii]]./msc_im.acline_sfr[tii][bit.acline_sfr_plus[tii]])
                    grd.zs_acline.acline_qfr[tii][bit.acline_sfr_plus[tii]] .= msc_im.acline_scale_fr[tii][bit.acline_sfr_plus[tii]].*((dt*qG.acflow_grad_weight).*stt_im.acline_qfr[tii][bit.acline_sfr_plus[tii]]./msc_im.acline_sfr[tii][bit.acline_sfr_plus[tii]])
                end
                # compute the scaled gradients
                if sum(bit.acline_sto_plus[tii]) > 0
                    msc_im.acline_scale_to[tii][bit.acline_sto_plus[tii]]             .= msc_im.acline_sto_plus[tii][bit.acline_sto_plus[tii]]./sqrt.(msc_im.acline_sto_plus[tii][bit.acline_sto_plus[tii]].^2 .+ qG.acflow_grad_eps2);
                    grd.zs_acline.acline_pto[tii][bit.acline_sto_plus[tii]] .= msc_im.acline_scale_to[tii][bit.acline_sto_plus[tii]].*((dt*qG.acflow_grad_weight).*stt_im.acline_pto[tii][bit.acline_sto_plus[tii]]./msc_im.acline_sto[tii][bit.acline_sto_plus[tii]])
                    grd.zs_acline.acline_qto[tii][bit.acline_sto_plus[tii]] .= msc_im.acline_scale_to[tii][bit.acline_sto_plus[tii]].*((dt*qG.acflow_grad_weight).*stt_im.acline_qto[tii][bit.acline_sto_plus[tii]]./msc_im.acline_sto[tii][bit.acline_sto_plus[tii]])
                end
            else
                # gradients
                grd.zs_acline.acline_pfr[tii][bit.acline_sfr_plus[tii]] .= (dt*cs).*stt_im.acline_pfr[tii][bit.acline_sfr_plus[tii]]./msc_im.acline_sfr[tii][bit.acline_sfr_plus[tii]]
                grd.zs_acline.acline_qfr[tii][bit.acline_sfr_plus[tii]] .= (dt*cs).*stt_im.acline_qfr[tii][bit.acline_sfr_plus[tii]]./msc_im.acline_sfr[tii][bit.acline_sfr_plus[tii]]
                grd.zs_acline.acline_pto[tii][bit.acline_sto_plus[tii]] .= (dt*cs).*stt_im.acline_pto[tii][bit.acline_sto_plus[tii]]./msc_im.acline_sto[tii][bit.acline_sto_plus[tii]]
                grd.zs_acline.acline_qto[tii][bit.acline_sto_plus[tii]] .= (dt*cs).*stt_im.acline_qto[tii][bit.acline_sto_plus[tii]]./msc_im.acline_sto[tii][bit.acline_sto_plus[tii]]
            end

            #= Previous gradient junk
            # loop
            if qG.acflow_grad_is_soft_abs
                dtgw = dt*qG.acflow_grad_weight
                for xx in 1:sys.nl
                    if (msc_im.acline_sfr_plus[tii][xx] >= msc_im.acline_sto_plus[tii][xx]) && (msc_im.acline_sfr_plus[tii][xx] > 0.0)
                        #msc.pub[1] = quasiGrad.soft_abs_grad_ac(xx, msc, qG, :acline_sfr_plus)
                        grd.zs_acline.acline_pfr[tii][xx] = msc.pub[1]#*dtgw#*stt_im.acline_pfr[tii][xx]/msc_im.acline_sfr[tii][xx]
                        grd.zs_acline.acline_qfr[tii][xx] = msc.pub[1]#*dtgw#*stt_im.acline_qfr[tii][xx]/msc_im.acline_sfr[tii][xx]
                    elseif (msc_im.acline_sto_plus[tii][xx] > 0.0)
                        #msc.pub[1] = quasiGrad.soft_abs_grad_ac(xx, msc, qG, :acline_sto_plus)
                        grd.zs_acline.acline_pto[tii][xx] = msc.pub[1]#*dtgw#*stt_im.acline_pto[tii][xx]/msc_im.acline_sto[tii][xx]
                        grd.zs_acline.acline_qto[tii][xx] = msc.pub[1]#*dtgw#*stt_im.acline_qto[tii][xx]/msc_im.acline_sto[tii][xx]
                    end
                end
            # no softabs -- use standard
            else
                dtcs = dt*cs
                for xx in 1:sys.nl
                    if (msc_im.acline_sfr_plus[tii][xx] >= msc_im.acline_sto_plus[tii][xx]) && (msc_im.acline_sfr_plus[tii][xx] > 0.0)
                        grd.zs_acline.acline_pfr[tii][xx] = dtcs*stt_im.acline_pfr[tii][xx]/msc_im.acline_sfr[tii][xx]
                        grd.zs_acline.acline_qfr[tii][xx] = dtcs*stt_im.acline_qfr[tii][xx]/msc_im.acline_sfr[tii][xx]
                    elseif (msc_im.acline_sto_plus[tii][xx] > 0.0)
                        grd.zs_acline.acline_pto[tii][xx] = dtcs*stt_im.acline_pto[tii][xx]/msc_im.acline_sto[tii][xx]
                        grd.zs_acline.acline_qto[tii][xx] = dtcs*stt_im.acline_qto[tii][xx]/msc_im.acline_sto[tii][xx]
                    end
                end
            end
            =#

            # ====================================================== #
            # Gradients: apparent power flow -- to -> from
            #
            # penalty function derivatives
            #=
            grd[:acline_sfr_plus][:acline_pfr][tii] = stt_im.acline_pfr[tii]./acline_sfr
            grd[:acline_sfr_plus][:acline_qfr][tii] = stt_im.acline_qfr[tii]./acline_sfr
            grd[:acline_sto_plus][:acline_pto][tii] = stt_im.acline_pto[tii]./acline_sto
            grd[:acline_sto_plus][:acline_qto][tii] = stt_im.acline_qto[tii]./acline_sto 
            max_sfst0  = [argmax([spfr,spto,0]) for (spfr,spto) in zip(acline_sfr_plus,acline_sto_plus)]
            grd[:zs_acline][:acline_sfr_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sfr_plus][tii][max_sfst0 .== 1] .= dt*cs
            grd[:zs_acline][:acline_sto_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sto_plus][tii][max_sfst0 .== 2] .= dt*cs
            =#
        end
    end
end