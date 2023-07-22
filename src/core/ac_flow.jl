# ac line flows
function acline_flows!(bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Idx, msc::quasiGrad.Msc, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
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
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys

        # duration
        dt = prm.ts.duration[tii]

        # organize relevant line values
        vm_fr = @view stt.vm[tii][idx.acline_fr_bus]
        va_fr = @view stt.va[tii][idx.acline_fr_bus]
        vm_to = @view stt.vm[tii][idx.acline_to_bus]
        va_to = @view stt.va[tii][idx.acline_to_bus]
        
        # tools
        msc.cos_ftp[tii]  .= cos.(va_fr .- va_to)
        msc.sin_ftp[tii]  .= sin.(va_fr .- va_to)
        msc.vff[tii]      .= vm_fr.^2
        msc.vtt[tii]      .= vm_to.^2
        msc.vft[tii]      .= vm_fr.*vm_to
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        msc.pfr[tii] .= (g_sr.+g_fr).*msc.vff[tii] .+ (.-g_sr.*msc.cos_ftp[tii] .- b_sr.*msc.sin_ftp[tii]).*msc.vft[tii]
        stt.acline_pfr[tii] .= stt.u_on_acline[tii].*msc.pfr[tii]
        
        # reactive power flow -- from -> to
        msc.qfr[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*msc.vff[tii] .+ (b_sr.*msc.cos_ftp[tii] .- g_sr.*msc.sin_ftp[tii]).*msc.vft[tii]
        stt.acline_qfr[tii] .= stt.u_on_acline[tii].*msc.qfr[tii]
        
        # apparent power flow -- to -> from
        msc.acline_sfr[tii] .= sqrt.(stt.acline_pfr[tii].^2 .+ stt.acline_qfr[tii].^2)
        
        # active power flow -- to -> from
        msc.pto[tii] .= (g_sr.+g_to).*msc.vtt[tii] .+ (.-g_sr.*msc.cos_ftp[tii] .+ b_sr.*msc.sin_ftp[tii]).*msc.vft[tii]
        stt.acline_pto[tii] .= stt.u_on_acline[tii].*msc.pto[tii]
        
        # reactive power flow -- to -> from
        msc.qto[tii] .= (.-b_sr.-b_to.-b_ch./2.0).*msc.vtt[tii] .+ (b_sr.*msc.cos_ftp[tii] .+ g_sr.*msc.sin_ftp[tii]).*msc.vft[tii]
        stt.acline_qto[tii] .= stt.u_on_acline[tii].*msc.qto[tii]

        # apparent power flow -- to -> from
        msc.acline_sto[tii] .= sqrt.(stt.acline_pto[tii].^2 .+ stt.acline_qto[tii].^2)
        
        # penalty functions and scores
        msc.acline_sfr_plus[tii] .= msc.acline_sfr[tii] .- prm.acline.mva_ub_nom
        msc.acline_sto_plus[tii] .= msc.acline_sto[tii] .- prm.acline.mva_ub_nom
        stt.zs_acline[tii]  .= (dt*cs).*max.(msc.acline_sfr_plus[tii], msc.acline_sto_plus[tii], 0.0)
        
        # ====================================================== #
        # ====================================================== #
        #
        # evaluate the grd?
        if qG.eval_grad

            # Gradients: active power flow -- from -> to
            grd.acline_pfr.vmfr[tii] .= stt.u_on_acline[tii].*(2.0.*(g_sr.+g_fr).*vm_fr .+ 
                    (.-g_sr.*msc.cos_ftp[tii] .- b_sr.*msc.sin_ftp[tii]).*vm_to)
            grd.acline_pfr.vmto[tii] .= stt.u_on_acline[tii].*(
                    (.-g_sr.*msc.cos_ftp[tii] .- b_sr.*msc.sin_ftp[tii]).*vm_fr)
            grd.acline_pfr.vafr[tii] .= stt.u_on_acline[tii].*(
                    (g_sr.*msc.sin_ftp[tii] .- b_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            grd.acline_pfr.vato[tii] .= stt.u_on_acline[tii].*(
                    (.-g_sr.*msc.sin_ftp[tii] .+ b_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_pfr.uon[tii] .= msc.pfr[tii]   
            end
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            
            grd.acline_qfr.vmfr[tii] .= stt.u_on_acline[tii].*(2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*vm_fr .+
                    (b_sr.*msc.cos_ftp[tii] .- g_sr.*msc.sin_ftp[tii]).*vm_to)
            grd.acline_qfr.vmto[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*msc.cos_ftp[tii] .- g_sr.*msc.sin_ftp[tii]).*vm_fr)
            grd.acline_qfr.vafr[tii] .= stt.u_on_acline[tii].*(
                    (.-b_sr.*msc.sin_ftp[tii] .- g_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            grd.acline_qfr.vato[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*msc.sin_ftp[tii] .+ g_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_qfr.uon[tii] .= msc.qfr[tii] 
            end
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
                           
            grd.acline_pto.vmfr[tii] .= stt.u_on_acline[tii].*( 
                    (.-g_sr.*msc.cos_ftp[tii] .+ b_sr.*msc.sin_ftp[tii]).*vm_to)
            grd.acline_pto.vmto[tii] .= stt.u_on_acline[tii].*(2.0.*(g_sr.+g_to).*vm_to .+
                    (.-g_sr.*msc.cos_ftp[tii] .+ b_sr.*msc.sin_ftp[tii]).*vm_fr)
            grd.acline_pto.vafr[tii] .= stt.u_on_acline[tii].*(
                    (g_sr.*msc.sin_ftp[tii] .+ b_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            grd.acline_pto.vato[tii] .= stt.u_on_acline[tii].*(
                    (.-g_sr.*msc.sin_ftp[tii] .- b_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_pto.uon[tii] .= msc.pto[tii]
            end
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            
            grd.acline_qto.vmfr[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*msc.cos_ftp[tii] .+ g_sr.*msc.sin_ftp[tii]).*vm_to)
            grd.acline_qto.vmto[tii] .= stt.u_on_acline[tii].*(2.0.*(.-b_sr.-b_to.-b_ch./2.0).*vm_to .+
                    (b_sr.*msc.cos_ftp[tii] .+ g_sr.*msc.sin_ftp[tii]).*vm_fr)
            grd.acline_qto.vafr[tii] .= stt.u_on_acline[tii].*(
                    (.-b_sr.*msc.sin_ftp[tii] .+ g_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            grd.acline_qto.vato[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*msc.sin_ftp[tii] .- g_sr.*msc.cos_ftp[tii]).*msc.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_qto.uon[tii] .= msc.qto[tii] 
            end

            # apply gradients
            grd.zs_acline.acline_pfr[tii] .= 0.0
            grd.zs_acline.acline_qfr[tii] .= 0.0
            grd.zs_acline.acline_pto[tii] .= 0.0
            grd.zs_acline.acline_qto[tii] .= 0.0  

            # indicators
            # => slower :( quasiGrad.get_largest_indices(msc, bit, :acline_sfr_plus, :acline_sto_plus)
            bit.acline_sfr_plus[tii] .= (msc.acline_sfr_plus[tii] .> 0.0) .&& (msc.acline_sfr_plus[tii] .> msc.acline_sto_plus[tii]);
            bit.acline_sto_plus[tii] .= (msc.acline_sto_plus[tii] .> 0.0) .&& (msc.acline_sto_plus[tii] .> msc.acline_sfr_plus[tii]); 
            #
            # slower alternative
                # => max_sfst0 = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(msc.acline_sfr_plus[tii], msc.acline_sto_plus[tii])]
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
                    msc.acline_scale_fr[tii][bit.acline_sfr_plus[tii]]      .= msc.acline_sfr_plus[tii][bit.acline_sfr_plus[tii]]./sqrt.(msc.acline_sfr_plus[tii][bit.acline_sfr_plus[tii]].^2 .+ qG.acflow_grad_eps2);
                    grd.zs_acline.acline_pfr[tii][bit.acline_sfr_plus[tii]] .= msc.acline_scale_fr[tii][bit.acline_sfr_plus[tii]].*((dt*qG.acflow_grad_weight).*stt.acline_pfr[tii][bit.acline_sfr_plus[tii]]./msc.acline_sfr[tii][bit.acline_sfr_plus[tii]])
                    grd.zs_acline.acline_qfr[tii][bit.acline_sfr_plus[tii]] .= msc.acline_scale_fr[tii][bit.acline_sfr_plus[tii]].*((dt*qG.acflow_grad_weight).*stt.acline_qfr[tii][bit.acline_sfr_plus[tii]]./msc.acline_sfr[tii][bit.acline_sfr_plus[tii]])
                end
                # compute the scaled gradients
                if sum(bit.acline_sto_plus[tii]) > 0
                    msc.acline_scale_to[tii][bit.acline_sto_plus[tii]]      .= msc.acline_sto_plus[tii][bit.acline_sto_plus[tii]]./sqrt.(msc.acline_sto_plus[tii][bit.acline_sto_plus[tii]].^2 .+ qG.acflow_grad_eps2);
                    grd.zs_acline.acline_pto[tii][bit.acline_sto_plus[tii]] .= msc.acline_scale_to[tii][bit.acline_sto_plus[tii]].*((dt*qG.acflow_grad_weight).*stt.acline_pto[tii][bit.acline_sto_plus[tii]]./msc.acline_sto[tii][bit.acline_sto_plus[tii]])
                    grd.zs_acline.acline_qto[tii][bit.acline_sto_plus[tii]] .= msc.acline_scale_to[tii][bit.acline_sto_plus[tii]].*((dt*qG.acflow_grad_weight).*stt.acline_qto[tii][bit.acline_sto_plus[tii]]./msc.acline_sto[tii][bit.acline_sto_plus[tii]])
                end
            else
                # gradients
                grd.zs_acline.acline_pfr[tii][bit.acline_sfr_plus[tii]] .= (dt*cs).*stt.acline_pfr[tii][bit.acline_sfr_plus[tii]]./msc.acline_sfr[tii][bit.acline_sfr_plus[tii]]
                grd.zs_acline.acline_qfr[tii][bit.acline_sfr_plus[tii]] .= (dt*cs).*stt.acline_qfr[tii][bit.acline_sfr_plus[tii]]./msc.acline_sfr[tii][bit.acline_sfr_plus[tii]]
                grd.zs_acline.acline_pto[tii][bit.acline_sto_plus[tii]] .= (dt*cs).*stt.acline_pto[tii][bit.acline_sto_plus[tii]]./msc.acline_sto[tii][bit.acline_sto_plus[tii]]
                grd.zs_acline.acline_qto[tii][bit.acline_sto_plus[tii]] .= (dt*cs).*stt.acline_qto[tii][bit.acline_sto_plus[tii]]./msc.acline_sto[tii][bit.acline_sto_plus[tii]]
            end

            #= Previous gradient junk
            # loop
            if qG.acflow_grad_is_soft_abs
                dtgw = dt*qG.acflow_grad_weight
                for xx in 1:sys.nl
                    if (msc.acline_sfr_plus[tii][xx] >= msc.acline_sto_plus[tii][xx]) && (msc.acline_sfr_plus[tii][xx] > 0.0)
                        #msc.pub[1] = quasiGrad.soft_abs_grad_ac(xx, msc, qG, :acline_sfr_plus)
                        grd.zs_acline.acline_pfr[tii][xx] = msc.pub[1]#*dtgw#*stt.acline_pfr[tii][xx]/msc.acline_sfr[tii][xx]
                        grd.zs_acline.acline_qfr[tii][xx] = msc.pub[1]#*dtgw#*stt.acline_qfr[tii][xx]/msc.acline_sfr[tii][xx]
                    elseif (msc.acline_sto_plus[tii][xx] > 0.0)
                        #msc.pub[1] = quasiGrad.soft_abs_grad_ac(xx, msc, qG, :acline_sto_plus)
                        grd.zs_acline.acline_pto[tii][xx] = msc.pub[1]#*dtgw#*stt.acline_pto[tii][xx]/msc.acline_sto[tii][xx]
                        grd.zs_acline.acline_qto[tii][xx] = msc.pub[1]#*dtgw#*stt.acline_qto[tii][xx]/msc.acline_sto[tii][xx]
                    end
                end
            # no softabs -- use standard
            else
                dtcs = dt*cs
                for xx in 1:sys.nl
                    if (msc.acline_sfr_plus[tii][xx] >= msc.acline_sto_plus[tii][xx]) && (msc.acline_sfr_plus[tii][xx] > 0.0)
                        grd.zs_acline.acline_pfr[tii][xx] = dtcs*stt.acline_pfr[tii][xx]/msc.acline_sfr[tii][xx]
                        grd.zs_acline.acline_qfr[tii][xx] = dtcs*stt.acline_qfr[tii][xx]/msc.acline_sfr[tii][xx]
                    elseif (msc.acline_sto_plus[tii][xx] > 0.0)
                        grd.zs_acline.acline_pto[tii][xx] = dtcs*stt.acline_pto[tii][xx]/msc.acline_sto[tii][xx]
                        grd.zs_acline.acline_qto[tii][xx] = dtcs*stt.acline_qto[tii][xx]/msc.acline_sto[tii][xx]
                    end
                end
            end
            =#

            # ====================================================== #
            # Gradients: apparent power flow -- to -> from
            #
            # penalty function derivatives
            #=
            grd[:acline_sfr_plus][:acline_pfr][tii] = stt.acline_pfr[tii]./acline_sfr
            grd[:acline_sfr_plus][:acline_qfr][tii] = stt.acline_qfr[tii]./acline_sfr
            grd[:acline_sto_plus][:acline_pto][tii] = stt.acline_pto[tii]./acline_sto
            grd[:acline_sto_plus][:acline_qto][tii] = stt.acline_qto[tii]./acline_sto 
            max_sfst0  = [argmax([spfr,spto,0]) for (spfr,spto) in zip(acline_sfr_plus,acline_sto_plus)]
            grd[:zs_acline][:acline_sfr_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sfr_plus][tii][max_sfst0 .== 1] .= dt*cs
            grd[:zs_acline][:acline_sto_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sto_plus][tii][max_sfst0 .== 2] .= dt*cs
            =#
        end
    end
end

# xfm line flows
function xfm_flows!(bit::quasiGrad.Bit, grd::quasiGrad.Grad, idx::quasiGrad.Idx, msc::quasiGrad.Msc, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    g_sr = prm.xfm.g_sr
    b_sr = prm.xfm.b_sr
    b_ch = prm.xfm.b_ch
    g_fr = prm.xfm.g_fr
    b_fr = prm.xfm.b_fr
    g_to = prm.xfm.g_to
    b_to = prm.xfm.b_to
    
    # call penalty costs
    cs = prm.vio.s_flow * qG.scale_c_sflow_testing
    
    # loop over time
    @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]

        # call stt
        phi      = stt.phi[tii]
        tau      = stt.tau[tii]
        u_on_xfm = stt.u_on_xfm[tii]
        
        # organize relevant line values
        vm_fr = @view stt.vm[tii][idx.xfm_fr_bus]
        va_fr = @view stt.va[tii][idx.xfm_fr_bus]
        vm_to = @view stt.vm[tii][idx.xfm_to_bus]
        va_to = @view stt.va[tii][idx.xfm_to_bus]
        
        # tools
        msc.cos_ftp_x[tii]  .= cos.(va_fr .- va_to .- phi)
        msc.sin_ftp_x[tii]  .= sin.(va_fr .- va_to .- phi)
        msc.vff_x[tii]      .= vm_fr.^2
        msc.vtt_x[tii]      .= vm_to.^2
        msc.vft_x[tii]      .= vm_fr.*vm_to
        msc.vt_tau_x[tii]   .= vm_to./tau
        msc.vf_tau_x[tii]   .= vm_fr./tau
        msc.vf_tau2_x[tii]  .= msc.vf_tau_x[tii]./tau
        msc.vff_tau2_x[tii] .= msc.vff_x[tii]./(tau.^2)
        msc.vft_tau_x[tii]  .= msc.vft_x[tii]./tau
        msc.vft_tau2_x[tii] .= msc.vft_tau_x[tii]./tau
        msc.vff_tau3_x[tii] .= msc.vff_tau2_x[tii]./tau
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        msc.pfr_x[tii] .= (g_sr.+g_fr).*msc.vff_tau2_x[tii] .+ (.-g_sr.*msc.cos_ftp_x[tii] .- b_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau_x[tii]
        stt.xfm_pfr[tii] .= u_on_xfm.*msc.pfr_x[tii]
        
        # reactive power flow -- from -> to
        msc.qfr_x[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*msc.vff_tau2_x[tii] .+ (b_sr.*msc.cos_ftp_x[tii] .- g_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau_x[tii]
        stt.xfm_qfr[tii] .= u_on_xfm.*msc.qfr_x[tii]
        
        # apparent power flow -- from -> to
        msc.xfm_sfr_x[tii] .= sqrt.(stt.xfm_pfr[tii].^2 .+ stt.xfm_qfr[tii].^2)
        
        # active power flow -- to -> from
        msc.pto_x[tii] .= (g_sr.+g_to).*msc.vtt_x[tii] .+ (.-g_sr.*msc.cos_ftp_x[tii] .+ b_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau_x[tii]
        stt.xfm_pto[tii] .= u_on_xfm.*msc.pto_x[tii]
        
        # reactive power flow -- to -> from
        msc.qto_x[tii] .= (.-b_sr.-b_to.-b_ch./2.0).*msc.vtt_x[tii] .+ (b_sr.*msc.cos_ftp_x[tii] .+ g_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau_x[tii]
        stt.xfm_qto[tii] .= u_on_xfm.*msc.qto_x[tii]
        
        # apparent power flow -- to -> from
        msc.xfm_sto_x[tii] .= sqrt.(stt.xfm_pto[tii].^2 .+ stt.xfm_qto[tii].^2)
        
        # penalty functions and scores
        msc.xfm_sfr_plus_x[tii]  .= msc.xfm_sfr_x[tii] .- prm.xfm.mva_ub_nom
        msc.xfm_sto_plus_x[tii]  .= msc.xfm_sto_x[tii] .- prm.xfm.mva_ub_nom
        stt.zs_xfm[tii]     .= dt*cs.*max.(msc.xfm_sfr_plus_x[tii], msc.xfm_sto_plus_x[tii], 0.0)
        # ====================================================== #
        # ====================================================== #
        
        # evaluate the grd?
        if qG.eval_grad    
            # Gradients: active power flow -- from -> to
            grd.xfm_pfr.vmfr[tii] .= u_on_xfm.*(2.0.*(g_sr.+g_fr).*msc.vf_tau2_x[tii] .+ 
                    (.-g_sr.*msc.cos_ftp_x[tii] .- b_sr.*msc.sin_ftp_x[tii]).*msc.vt_tau_x[tii])
            grd.xfm_pfr.vmto[tii] .= u_on_xfm.*(
                    (.-g_sr.*msc.cos_ftp_x[tii] .- b_sr.*msc.sin_ftp_x[tii]).*msc.vf_tau_x[tii])
            grd.xfm_pfr.vafr[tii] .= u_on_xfm.*(
                    (g_sr.*msc.sin_ftp_x[tii] .- b_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_pfr.vato[tii] .= u_on_xfm.*(
                    (.-g_sr.*msc.sin_ftp_x[tii] .+ b_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_pfr.tau[tii] .= u_on_xfm.*((-2.0).*(g_sr.+g_fr).*msc.vff_tau3_x[tii] .+ 
                    .-(.-g_sr.*msc.cos_ftp_x[tii] .- b_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau2_x[tii])
            grd.xfm_pfr.phi[tii] .= grd.xfm_pfr.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_pfr.uon[tii] .= msc.pfr_x[tii]
            end

            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            grd.xfm_qfr.vmfr[tii] .= u_on_xfm.*(2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*msc.vf_tau2_x[tii] .+
                    (b_sr.*msc.cos_ftp_x[tii] .- g_sr.*msc.sin_ftp_x[tii]).*msc.vt_tau_x[tii])
            grd.xfm_qfr.vmto[tii] .= u_on_xfm.*(
                    (b_sr.*msc.cos_ftp_x[tii] .- g_sr.*msc.sin_ftp_x[tii]).*msc.vf_tau_x[tii])
            grd.xfm_qfr.vafr[tii] .= u_on_xfm.*(
                    (.-b_sr.*msc.sin_ftp_x[tii] .- g_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_qfr.vato[tii] .= u_on_xfm.*(
                    (b_sr.*msc.sin_ftp_x[tii] .+ g_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_qfr.tau[tii]  .= u_on_xfm.*(.-2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*msc.vff_tau3_x[tii] .+
                    .-(b_sr.*msc.cos_ftp_x[tii] .- g_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau2_x[tii])
            grd.xfm_qfr.phi[tii]  .= grd.xfm_qfr.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_qfr.uon[tii]  .= msc.qfr_x[tii]
            end
            
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
            grd.xfm_pto.vmfr[tii] .= u_on_xfm.*( 
                    (.-g_sr.*msc.cos_ftp_x[tii] .+ b_sr.*msc.sin_ftp_x[tii]).*msc.vt_tau_x[tii])
            grd.xfm_pto.vmto[tii] .= u_on_xfm.*(2.0.*(g_sr.+g_to).*vm_to .+
                    (.-g_sr.*msc.cos_ftp_x[tii] .+ b_sr.*msc.sin_ftp_x[tii]).*msc.vf_tau_x[tii])
            grd.xfm_pto.vafr[tii] .= u_on_xfm.*(
                    (g_sr.*msc.sin_ftp_x[tii] .+ b_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_pto.vato[tii] .= u_on_xfm.*(
                    (.-g_sr.*msc.sin_ftp_x[tii] .- b_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_pto.tau[tii] .= u_on_xfm.*(
                    .-(.-g_sr.*msc.cos_ftp_x[tii] .+ b_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau2_x[tii])
            grd.xfm_pto.phi[tii] .= grd.xfm_pto.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_pto.uon[tii] .= msc.pto_x[tii]
            end
    
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            grd.xfm_qto.vmfr[tii] .= u_on_xfm.*(
                    (b_sr.*msc.cos_ftp_x[tii] .+ g_sr.*msc.sin_ftp_x[tii]).*msc.vt_tau_x[tii])
            grd.xfm_qto.vmto[tii] .= u_on_xfm.*(2.0.*(.-b_sr.-b_to.-b_ch./2.0).*vm_to .+
                    (b_sr.*msc.cos_ftp_x[tii] .+ g_sr.*msc.sin_ftp_x[tii]).*msc.vf_tau_x[tii])
            grd.xfm_qto.vafr[tii] .= u_on_xfm.*(
                    (.-b_sr.*msc.sin_ftp_x[tii] .+ g_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_qto.vato[tii] .= u_on_xfm.*(
                    (b_sr.*msc.sin_ftp_x[tii] .- g_sr.*msc.cos_ftp_x[tii]).*msc.vft_tau_x[tii])
            grd.xfm_qto.tau[tii]  .= u_on_xfm.*(
                    .-(b_sr.*msc.cos_ftp_x[tii] .+ g_sr.*msc.sin_ftp_x[tii]).*msc.vft_tau2_x[tii])
            grd.xfm_qto.phi[tii] .= grd.xfm_qto.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_qto.uon[tii] .= msc.qto_x[tii]
            end

            # indicators
            # slower :( => quasiGrad.get_largest_indices(msc, bit, :xfm_sfr_plus_x, :xfm_sto_plus_x)
            bit.xfm_sfr_plus_x[tii] .= (msc.xfm_sfr_plus_x[tii] .> 0.0) .&& (msc.xfm_sfr_plus_x[tii] .> msc.xfm_sto_plus_x[tii]);
            bit.xfm_sto_plus_x[tii] .= (msc.xfm_sto_plus_x[tii] .> 0.0) .&& (msc.xfm_sto_plus_x[tii] .> msc.xfm_sfr_plus_x[tii]);
            #
            # slow alternative:
                # => max_sfst0 = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(msc.xfm_sfr_plus_x[tii],msc.xfm_sto_plus_x[tii])]
                # => ind_fr = max_sfst0 .== 1
                # => ind_to = max_sfst0 .== 2
    
            # flush -- set to 0 => [Not(ind_fr)] and [Not(ind_to)] was slow/memory inefficient
            grd.zs_xfm.xfm_pfr[tii] .= 0.0
            grd.zs_xfm.xfm_qfr[tii] .= 0.0
            grd.zs_xfm.xfm_pto[tii] .= 0.0
            grd.zs_xfm.xfm_qto[tii] .= 0.0
    
            if qG.acflow_grad_is_soft_abs
                # compute the scaled gradients
                if sum(bit.xfm_sfr_plus_x[tii]) > 0
                    msc.scale_fr_x[tii][bit.xfm_sfr_plus_x[tii]]     .= msc.xfm_sfr_plus_x[tii][bit.xfm_sfr_plus_x[tii]]./sqrt.(msc.xfm_sfr_plus_x[tii][bit.xfm_sfr_plus_x[tii]].^2 .+ qG.acflow_grad_eps2);
                    grd.zs_xfm.xfm_pfr[tii][bit.xfm_sfr_plus_x[tii]] .= msc.scale_fr_x[tii][bit.xfm_sfr_plus_x[tii]].*((dt*qG.acflow_grad_weight).*stt.xfm_pfr[tii][bit.xfm_sfr_plus_x[tii]]./msc.xfm_sfr_x[tii][bit.xfm_sfr_plus_x[tii]])
                    grd.zs_xfm.xfm_qfr[tii][bit.xfm_sfr_plus_x[tii]] .= msc.scale_fr_x[tii][bit.xfm_sfr_plus_x[tii]].*((dt*qG.acflow_grad_weight).*stt.xfm_qfr[tii][bit.xfm_sfr_plus_x[tii]]./msc.xfm_sfr_x[tii][bit.xfm_sfr_plus_x[tii]])
                end

                # compute the scaled gradients
                if sum(bit.xfm_sto_plus_x[tii]) > 0
                    msc.scale_to_x[tii][bit.xfm_sto_plus_x[tii]]     .= msc.xfm_sto_plus_x[tii][bit.xfm_sto_plus_x[tii]]./sqrt.(msc.xfm_sto_plus_x[tii][bit.xfm_sto_plus_x[tii]].^2 .+ qG.acflow_grad_eps2);
                    grd.zs_xfm.xfm_pto[tii][bit.xfm_sto_plus_x[tii]] .= msc.scale_to_x[tii][bit.xfm_sto_plus_x[tii]].*((dt*qG.acflow_grad_weight).*stt.xfm_pto[tii][bit.xfm_sto_plus_x[tii]]./msc.xfm_sto_x[tii][bit.xfm_sto_plus_x[tii]])
                    grd.zs_xfm.xfm_qto[tii][bit.xfm_sto_plus_x[tii]] .= msc.scale_to_x[tii][bit.xfm_sto_plus_x[tii]].*((dt*qG.acflow_grad_weight).*stt.xfm_qto[tii][bit.xfm_sto_plus_x[tii]]./msc.xfm_sto_x[tii][bit.xfm_sto_plus_x[tii]])
                end
            else
                # gradients
                grd.zs_xfm.xfm_pfr[tii][bit.xfm_sfr_plus_x[tii]] .= (dt*cs).*stt.xfm_pfr[tii][bit.xfm_sfr_plus_x[tii]]./msc.xfm_sfr_x[tii][bit.xfm_sfr_plus_x[tii]]
                grd.zs_xfm.xfm_qfr[tii][bit.xfm_sfr_plus_x[tii]] .= (dt*cs).*stt.xfm_qfr[tii][bit.xfm_sfr_plus_x[tii]]./msc.xfm_sfr_x[tii][bit.xfm_sfr_plus_x[tii]]
                grd.zs_xfm.xfm_pto[tii][bit.xfm_sto_plus_x[tii]] .= (dt*cs).*stt.xfm_pto[tii][bit.xfm_sto_plus_x[tii]]./msc.xfm_sto_x[tii][bit.xfm_sto_plus_x[tii]]
                grd.zs_xfm.xfm_qto[tii][bit.xfm_sto_plus_x[tii]] .= (dt*cs).*stt.xfm_qto[tii][bit.xfm_sto_plus_x[tii]]./msc.xfm_sto_x[tii][bit.xfm_sto_plus_x[tii]]
            end

            #= Previous gradient junk!
            # apply gradients
            if qG.acflow_grad_is_soft_abs
                dtgw = dt*qG.acflow_grad_weight
                for xx in 1:sys.nx
                    if (msc.xfm_sfr_plus_x[tii][xx] >= msc.xfm_sto_plus_x[tii][xx]) && (msc.xfm_sfr_plus_x[tii][xx] > 0.0)
                        sc = soft_abs_grad_ac(msc.xfm_sfr_plus_x[tii][xx], qG)
                        grd.zs_xfm.xfm_pfr[tii][xx] = sc*(dtgw*stt.xfm_pfr[tii][xx]/msc.xfm_sfr_x[tii][xx])
                        grd.zs_xfm.xfm_qfr[tii][xx] = sc*(dtgw*stt.xfm_qfr[tii][xx]/msc.xfm_sfr_x[tii][xx])
                    elseif (msc.xfm_sto_plus_x[tii][xx] > 0.0)
                        sc = soft_abs_grad_ac(msc.xfm_sto_plus_x[tii][xx], qG)
                        grd.zs_xfm.xfm_pto[tii][xx] = sc*(dtgw*stt.xfm_pto[tii][xx]/msc.xfm_sto_x[tii][xx])
                        grd.zs_xfm.xfm_qto[tii][xx] = sc*(dtgw*stt.xfm_qto[tii][xx]/msc.xfm_sto_x[tii][xx])
                    else
                        grd.zs_xfm.xfm_pfr[tii][xx] = 0.0
                        grd.zs_xfm.xfm_qfr[tii][xx] = 0.0
                        grd.zs_xfm.xfm_pto[tii][xx] = 0.0
                        grd.zs_xfm.xfm_qto[tii][xx] = 0.0  
                    end
                end
            # no softabs -- use standard
            else
                dtcs = dt*cs
                for xx in 1:sys.nx
                    if (msc.xfm_sfr_plus_x[tii][xx] >= msc.xfm_sto_plus_x[tii][xx]) && (msc.xfm_sfr_plus_x[tii][xx] > 0.0)
                        grd.zs_xfm.xfm_pfr[tii][xx] = dtcs*stt.xfm_pfr[tii][xx]/msc.xfm_sfr_x[tii][xx]
                        grd.zs_xfm.xfm_qfr[tii][xx] = dtcs*stt.xfm_qfr[tii][xx]/msc.xfm_sfr_x[tii][xx]
                    elseif (msc.xfm_sto_plus_x[tii][xx] > 0.0)
                        grd.zs_xfm.xfm_pto[tii][xx] = dtcs*stt.xfm_pto[tii][xx]/msc.xfm_sto_x[tii][ind_to]
                        grd.zs_xfm.xfm_qto[tii][xx] = dtcs*stt.xfm_qto[tii][xx]/msc.xfm_sto_x[tii][ind_to]
                    else
                        grd.zs_xfm.xfm_pfr[tii][xx] = 0.0
                        grd.zs_xfm.xfm_qfr[tii][xx] = 0.0
                        grd.zs_xfm.xfm_pto[tii][xx] = 0.0
                        grd.zs_xfm.xfm_qto[tii][xx] = 0.0  
                    end
                end
            end
            =#

            # ====================================================== #
            # Gradients: apparent power flow -- to -> from
            #
            # penalty function derivatives
            #=
            grd[:msc.xfm_sfr_plus_x[tii]][:xfm_pfr][tii] = stt.xfm_pfr[tii]./msc.xfm_sfr_x[tii]
            grd[:msc.xfm_sfr_plus_x[tii]][:xfm_qfr][tii] = stt.xfm_qfr[tii]./msc.xfm_sfr_x[tii]
            grd[:msc.xfm_sto_plus_x[tii]][:xfm_pto][tii] = stt.xfm_pto[tii]./msc.xfm_sto_x[tii]
            grd[:msc.xfm_sto_plus_x[tii]][:xfm_qto][tii] = stt.xfm_qto[tii]./msc.xfm_sto_x[tii]
    
            max_sfst0  = [argmax([spfr,spto,0]) for (spfr,spto) in zip(msc.xfm_sfr_plus_x[tii],msc.xfm_sto_plus_x[tii])]
            grd[:zs_xfm][:msc.xfm_sfr_plus_x[tii]][tii] = zeros(length(max_sfst0))
            grd[:zs_xfm][:msc.xfm_sfr_plus_x[tii]][tii][max_sfst0 .== 1] .= dt*cs
            grd[:zs_xfm][:msc.xfm_sto_plus_x[tii]][tii] = zeros(length(max_sfst0))
            grd[:zs_xfm][:msc.xfm_sto_plus_x[tii]][tii][max_sfst0 .== 2] .= dt*cs
            =#
        end
    end
end   