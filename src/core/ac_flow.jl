# ac line flows
function acline_flows!(grd::quasiGrad.Grad, idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
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
    
    # loop over time -- use per=core
    @batch per=core for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
    
        # duration
        dt = prm.ts.duration[tii]
    
        # organize relevant line values
        stt.vm_fr[tii] .= @view stt.vm[tii][idx.acline_fr_bus]
        stt.va_fr[tii] .= @view stt.va[tii][idx.acline_fr_bus]
        stt.vm_to[tii] .= @view stt.vm[tii][idx.acline_to_bus]
        stt.va_to[tii] .= @view stt.va[tii][idx.acline_to_bus]
        
        # tools
        @turbo stt.cos_ftp[tii] .= quasiGrad.LoopVectorization.cos_fast.(stt.va_fr[tii] .- stt.va_to[tii])
        @turbo stt.sin_ftp[tii] .= quasiGrad.LoopVectorization.sin_fast.(stt.va_fr[tii] .- stt.va_to[tii])
        @turbo stt.vff[tii]     .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_fr[tii],2)
        @turbo stt.vtt[tii]     .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_to[tii],2)
        @turbo stt.vft[tii]     .= stt.vm_fr[tii].*stt.vm_to[tii]
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        @turbo stt.pfr[tii] .= (g_sr.+g_fr).*stt.vff[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
        @turbo stt.acline_pfr[tii] .= stt.u_on_acline[tii].*stt.pfr[tii]
        
        # reactive power flow -- from -> to
        @turbo stt.qfr[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*stt.vff[tii] .+ (b_sr.*stt.cos_ftp[tii] .- g_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
        @turbo stt.acline_qfr[tii] .= stt.u_on_acline[tii].*stt.qfr[tii]
        
        # apparent power flow -- to -> from
        @turbo stt.acline_sfr[tii] .= quasiGrad.LoopVectorization.sqrt_fast.(quasiGrad.LoopVectorization.pow_fast.(stt.acline_pfr[tii],2) .+ quasiGrad.LoopVectorization.pow_fast.(stt.acline_qfr[tii],2))
        
        # active power flow -- to -> from
        @turbo stt.pto[tii] .= (g_sr.+g_to).*stt.vtt[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .+ b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
        @turbo stt.acline_pto[tii] .= stt.u_on_acline[tii].*stt.pto[tii]
        
        # reactive power flow -- to -> from
        @turbo stt.qto[tii] .= (.-b_sr.-b_to.-b_ch./2.0).*stt.vtt[tii] .+ (b_sr.*stt.cos_ftp[tii] .+ g_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
        @turbo stt.acline_qto[tii] .= stt.u_on_acline[tii].*stt.qto[tii]
    
        # apparent power flow -- to -> from
        @turbo stt.acline_sto[tii] .= quasiGrad.LoopVectorization.sqrt_fast.(quasiGrad.LoopVectorization.pow_fast.(stt.acline_pto[tii],2) .+ quasiGrad.LoopVectorization.pow_fast.(stt.acline_qto[tii],2))
        
        # penalty functions and scores
        @turbo stt.acline_sfr_plus[tii] .= stt.acline_sfr[tii] .- prm.acline.mva_ub_nom
        @turbo stt.acline_sto_plus[tii] .= stt.acline_sto[tii] .- prm.acline.mva_ub_nom
        @turbo stt.zs_acline[tii]       .= (dt*cs).*max.(stt.acline_sfr_plus[tii], stt.acline_sto_plus[tii], 0.0)

        # ====================================================== #
        # ====================================================== #
        #
        # evaluate the grd?
        if qG.eval_grad
    
            # Gradients: active power flow -- from -> to
            @turbo grd.acline_pfr.vmfr[tii] .= stt.u_on_acline[tii].*(2.0.*(g_sr.+g_fr).*stt.vm_fr[tii] .+ 
                    (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vm_to[tii])
            @turbo grd.acline_pfr.vmto[tii] .= stt.u_on_acline[tii].*(
                    (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vm_fr[tii])
            @turbo grd.acline_pfr.vafr[tii] .= stt.u_on_acline[tii].*(
                    (g_sr.*stt.sin_ftp[tii] .- b_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            @turbo grd.acline_pfr.vato[tii] .= stt.u_on_acline[tii].*(
                    (.-g_sr.*stt.sin_ftp[tii] .+ b_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_pfr.uon[tii] .= stt.pfr[tii]   
            end
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            
            @turbo grd.acline_qfr.vmfr[tii] .= stt.u_on_acline[tii].*(2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*stt.vm_fr[tii] .+
                    (b_sr.*stt.cos_ftp[tii] .- g_sr.*stt.sin_ftp[tii]).*stt.vm_to[tii])
            @turbo grd.acline_qfr.vmto[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*stt.cos_ftp[tii] .- g_sr.*stt.sin_ftp[tii]).*stt.vm_fr[tii])
            @turbo grd.acline_qfr.vafr[tii] .= stt.u_on_acline[tii].*(
                    (.-b_sr.*stt.sin_ftp[tii] .- g_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            @turbo grd.acline_qfr.vato[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*stt.sin_ftp[tii] .+ g_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_qfr.uon[tii] .= stt.qfr[tii] 
            end
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
                           
            @turbo grd.acline_pto.vmfr[tii] .= stt.u_on_acline[tii].*( 
                    (.-g_sr.*stt.cos_ftp[tii] .+ b_sr.*stt.sin_ftp[tii]).*stt.vm_to[tii])
            @turbo grd.acline_pto.vmto[tii] .= stt.u_on_acline[tii].*(2.0.*(g_sr.+g_to).*stt.vm_to[tii] .+
                    (.-g_sr.*stt.cos_ftp[tii] .+ b_sr.*stt.sin_ftp[tii]).*stt.vm_fr[tii])
            @turbo grd.acline_pto.vafr[tii] .= stt.u_on_acline[tii].*(
                    (g_sr.*stt.sin_ftp[tii] .+ b_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            @turbo grd.acline_pto.vato[tii] .= stt.u_on_acline[tii].*(
                    (.-g_sr.*stt.sin_ftp[tii] .- b_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_pto.uon[tii] .= stt.pto[tii]
            end
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            
            @turbo grd.acline_qto.vmfr[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*stt.cos_ftp[tii] .+ g_sr.*stt.sin_ftp[tii]).*stt.vm_to[tii])
            @turbo grd.acline_qto.vmto[tii] .= stt.u_on_acline[tii].*(2.0.*(.-b_sr.-b_to.-b_ch./2.0).*stt.vm_to[tii] .+
                    (b_sr.*stt.cos_ftp[tii] .+ g_sr.*stt.sin_ftp[tii]).*stt.vm_fr[tii])
            @turbo grd.acline_qto.vafr[tii] .= stt.u_on_acline[tii].*(
                    (.-b_sr.*stt.sin_ftp[tii] .+ g_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            @turbo grd.acline_qto.vato[tii] .= stt.u_on_acline[tii].*(
                    (b_sr.*stt.sin_ftp[tii] .- g_sr.*stt.cos_ftp[tii]).*stt.vft[tii])
            if qG.update_acline_xfm_bins
                grd.acline_qto.uon[tii] .= stt.qto[tii] 
            end
    
            # loop and take the gradients of the penalties
            if qG.acflow_grad_is_soft_abs
                # softabs
                @fastmath @inbounds @simd for ln in 1:sys.nl
                    if (stt.acline_sfr_plus[tii][ln] > 0.0) && (stt.acline_sfr_plus[tii][ln] > stt.acline_sto_plus[tii][ln])
                        scale_fr                          = dt*qG.acflow_grad_weight*quasiGrad.soft_abs_acflow_grad(stt.acline_sfr_plus[tii][ln], qG)
                        grd.zs_acline.acline_pfr[tii][ln] = scale_fr*stt.acline_pfr[tii][ln]/stt.acline_sfr[tii][ln]
                        grd.zs_acline.acline_qfr[tii][ln] = scale_fr*stt.acline_qfr[tii][ln]/stt.acline_sfr[tii][ln]
                        grd.zs_acline.acline_pto[tii][ln] = 0.0
                        grd.zs_acline.acline_qto[tii][ln] = 0.0
                    elseif (stt.acline_sto_plus[tii][ln] > 0.0) && (stt.acline_sto_plus[tii][ln] > stt.acline_sfr_plus[tii][ln])
                        grd.zs_acline.acline_pfr[tii][ln] = 0.0
                        grd.zs_acline.acline_qfr[tii][ln] = 0.0
                        scale_to                          = dt*qG.acflow_grad_weight*quasiGrad.soft_abs_acflow_grad(stt.acline_sto_plus[tii][ln], qG)
                        grd.zs_acline.acline_pto[tii][ln] = scale_to*stt.acline_pto[tii][ln]/stt.acline_sto[tii][ln]
                        grd.zs_acline.acline_qto[tii][ln] = scale_to*stt.acline_qto[tii][ln]/stt.acline_sto[tii][ln]
                    else 
                        grd.zs_acline.acline_pfr[tii][ln] = 0.0
                        grd.zs_acline.acline_qfr[tii][ln] = 0.0
                        grd.zs_acline.acline_pto[tii][ln] = 0.0
                        grd.zs_acline.acline_qto[tii][ln] = 0.0
                    end
                end
            else
                # standard
                @fastmath @inbounds @simd for ln in 1:sys.nl
                    if (stt.acline_sfr_plus[tii][ln] > 0.0) && (stt.acline_sfr_plus[tii][ln] > stt.acline_sto_plus[tii][ln])
                        grd.zs_acline.acline_pfr[tii][ln] = dt*cs*stt.acline_pfr[tii][ln]/stt.acline_sfr[tii][ln]
                        grd.zs_acline.acline_qfr[tii][ln] = dt*cs*stt.acline_qfr[tii][ln]/stt.acline_sfr[tii][ln]
                        grd.zs_acline.acline_pto[tii][ln] = 0.0
                        grd.zs_acline.acline_qto[tii][ln] = 0.0
                    elseif (stt.acline_sto_plus[tii][ln] > 0.0) && (stt.acline_sto_plus[tii][ln] > stt.acline_sfr_plus[tii][ln])
                        grd.zs_acline.acline_pfr[tii][ln] = 0.0
                        grd.zs_acline.acline_qfr[tii][ln] = 0.0
                        grd.zs_acline.acline_pto[tii][ln] = dt*cs*stt.acline_pto[tii][ln]/stt.acline_sto[tii][ln]
                        grd.zs_acline.acline_qto[tii][ln] = dt*cs*stt.acline_qto[tii][ln]/stt.acline_sto[tii][ln]
                    else 
                        grd.zs_acline.acline_pfr[tii][ln] = 0.0
                        grd.zs_acline.acline_qfr[tii][ln] = 0.0
                        grd.zs_acline.acline_pto[tii][ln] = 0.0
                        grd.zs_acline.acline_qto[tii][ln] = 0.0
                    end
                end
            end
        end
    end
    
    # sleep tasks
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
end

# xfm line flows
function xfm_flows!(grd::quasiGrad.Grad, idx::quasiGrad.Index, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::quasiGrad.State, sys::quasiGrad.System)
    g_sr = prm.xfm.g_sr
    b_sr = prm.xfm.b_sr
    b_ch = prm.xfm.b_ch
    g_fr = prm.xfm.g_fr
    b_fr = prm.xfm.b_fr
    g_to = prm.xfm.g_to
    b_to = prm.xfm.b_to
    
    # call penalty costs
    cs = prm.vio.s_flow * qG.scale_c_sflow_testing
    
    # loop over time -- use "per=core"
    @batch per=core for tii in prm.ts.time_keys
    # => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys
    
        # duration
        dt = prm.ts.duration[tii]
    
        # call stt
        phi      = stt.phi[tii]
        tau      = stt.tau[tii]
        u_on_xfm = stt.u_on_xfm[tii]
        
        # organize relevant line values
        stt.vm_fr_x[tii] .= @view stt.vm[tii][idx.xfm_fr_bus]
        stt.va_fr_x[tii] .= @view stt.va[tii][idx.xfm_fr_bus]
        stt.vm_to_x[tii] .= @view stt.vm[tii][idx.xfm_to_bus]
        stt.va_to_x[tii] .= @view stt.va[tii][idx.xfm_to_bus]
        
        # tools
        @turbo stt.cos_ftp_x[tii]  .= quasiGrad.LoopVectorization.cos_fast.(stt.va_fr_x[tii] .- stt.va_to_x[tii] .- phi)
        @turbo stt.sin_ftp_x[tii]  .= quasiGrad.LoopVectorization.sin_fast.(stt.va_fr_x[tii] .- stt.va_to_x[tii] .- phi)
        @turbo stt.vff_x[tii]      .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_fr_x[tii],2)
        @turbo stt.vtt_x[tii]      .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_to_x[tii],2)
        @turbo stt.vft_x[tii]      .= stt.vm_fr_x[tii].*stt.vm_to_x[tii]
        @turbo stt.vt_tau_x[tii]   .= stt.vm_to_x[tii]./tau
        @turbo stt.vf_tau_x[tii]   .= stt.vm_fr_x[tii]./tau
        @turbo stt.vf_tau2_x[tii]  .= stt.vf_tau_x[tii]./tau
        @turbo stt.vff_tau2_x[tii] .= stt.vff_x[tii]./quasiGrad.LoopVectorization.pow_fast.(tau,2)
        @turbo stt.vft_tau_x[tii]  .= stt.vft_x[tii]./tau
        @turbo stt.vft_tau2_x[tii] .= stt.vft_tau_x[tii]./tau
        @turbo stt.vff_tau3_x[tii] .= stt.vff_tau2_x[tii]./tau
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        @turbo stt.pfr_x[tii] .= (g_sr.+g_fr).*stt.vff_tau2_x[tii] .+ (.-g_sr.*stt.cos_ftp_x[tii] .- b_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau_x[tii]
        @turbo stt.xfm_pfr[tii] .= u_on_xfm.*stt.pfr_x[tii]
        
        # reactive power flow -- from -> to
        @turbo stt.qfr_x[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*stt.vff_tau2_x[tii] .+ (b_sr.*stt.cos_ftp_x[tii] .- g_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau_x[tii]
        @turbo stt.xfm_qfr[tii] .= u_on_xfm.*stt.qfr_x[tii]
        
        # apparent power flow -- from -> to
        @turbo stt.xfm_sfr[tii] .= quasiGrad.LoopVectorization.sqrt_fast.(quasiGrad.LoopVectorization.pow_fast.(stt.xfm_pfr[tii],2) .+ quasiGrad.LoopVectorization.pow_fast.(stt.xfm_qfr[tii],2))
        
        # active power flow -- to -> from
        @turbo stt.pto_x[tii] .= (g_sr.+g_to).*stt.vtt_x[tii] .+ (.-g_sr.*stt.cos_ftp_x[tii] .+ b_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau_x[tii]
        @turbo stt.xfm_pto[tii] .= u_on_xfm.*stt.pto_x[tii]
        
        # reactive power flow -- to -> from
        @turbo stt.qto_x[tii] .= (.-b_sr.-b_to.-b_ch./2.0).*stt.vtt_x[tii] .+ (b_sr.*stt.cos_ftp_x[tii] .+ g_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau_x[tii]
        @turbo stt.xfm_qto[tii] .= u_on_xfm.*stt.qto_x[tii]
        
        # apparent power flow -- to -> from
        @turbo stt.xfm_sto[tii] .= quasiGrad.LoopVectorization.sqrt_fast.(quasiGrad.LoopVectorization.pow_fast.(stt.xfm_pto[tii],2) .+ quasiGrad.LoopVectorization.pow_fast.(stt.xfm_qto[tii],2))
        
        # penalty functions and scores
        @turbo stt.xfm_sfr_plus[tii]  .= stt.xfm_sfr[tii] .- prm.xfm.mva_ub_nom
        @turbo stt.xfm_sto_plus[tii]  .= stt.xfm_sto[tii] .- prm.xfm.mva_ub_nom
        @turbo stt.zs_xfm[tii]        .= dt*cs.*max.(stt.xfm_sfr_plus[tii], stt.xfm_sto_plus[tii], 0.0)
        # ====================================================== #
        # ====================================================== #
        
        # evaluate the grd?
        if qG.eval_grad    
            # Gradients: active power flow -- from -> to
            @turbo grd.xfm_pfr.vmfr[tii] .= u_on_xfm.*(2.0.*(g_sr.+g_fr).*stt.vf_tau2_x[tii] .+ 
                    (.-g_sr.*stt.cos_ftp_x[tii] .- b_sr.*stt.sin_ftp_x[tii]).*stt.vt_tau_x[tii])
            @turbo grd.xfm_pfr.vmto[tii] .= u_on_xfm.*(
                    (.-g_sr.*stt.cos_ftp_x[tii] .- b_sr.*stt.sin_ftp_x[tii]).*stt.vf_tau_x[tii])
            @turbo grd.xfm_pfr.vafr[tii] .= u_on_xfm.*(
                    (g_sr.*stt.sin_ftp_x[tii] .- b_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_pfr.vato[tii] .= u_on_xfm.*(
                    (.-g_sr.*stt.sin_ftp_x[tii] .+ b_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_pfr.tau[tii] .= u_on_xfm.*((-2.0).*(g_sr.+g_fr).*stt.vff_tau3_x[tii] .+ 
                    .-(.-g_sr.*stt.cos_ftp_x[tii] .- b_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau2_x[tii])
            grd.xfm_pfr.phi[tii] .= grd.xfm_pfr.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_pfr.uon[tii] .= stt.pfr_x[tii]
            end
    
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            @turbo grd.xfm_qfr.vmfr[tii] .= u_on_xfm.*(2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*stt.vf_tau2_x[tii] .+
                    (b_sr.*stt.cos_ftp_x[tii] .- g_sr.*stt.sin_ftp_x[tii]).*stt.vt_tau_x[tii])
            @turbo grd.xfm_qfr.vmto[tii] .= u_on_xfm.*(
                    (b_sr.*stt.cos_ftp_x[tii] .- g_sr.*stt.sin_ftp_x[tii]).*stt.vf_tau_x[tii])
            @turbo grd.xfm_qfr.vafr[tii] .= u_on_xfm.*(
                    (.-b_sr.*stt.sin_ftp_x[tii] .- g_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_qfr.vato[tii] .= u_on_xfm.*(
                    (b_sr.*stt.sin_ftp_x[tii] .+ g_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_qfr.tau[tii]  .= u_on_xfm.*(.-2.0.*(.-b_sr.-b_fr.-b_ch./2.0).*stt.vff_tau3_x[tii] .+
                    .-(b_sr.*stt.cos_ftp_x[tii] .- g_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau2_x[tii])
            grd.xfm_qfr.phi[tii]  .= grd.xfm_qfr.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_qfr.uon[tii]  .= stt.qfr_x[tii]
            end
            
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
            @turbo grd.xfm_pto.vmfr[tii] .= u_on_xfm.*( 
                    (.-g_sr.*stt.cos_ftp_x[tii] .+ b_sr.*stt.sin_ftp_x[tii]).*stt.vt_tau_x[tii])
            @turbo grd.xfm_pto.vmto[tii] .= u_on_xfm.*(2.0.*(g_sr.+g_to).*stt.vm_to_x[tii] .+
                    (.-g_sr.*stt.cos_ftp_x[tii] .+ b_sr.*stt.sin_ftp_x[tii]).*stt.vf_tau_x[tii])
            @turbo grd.xfm_pto.vafr[tii] .= u_on_xfm.*(
                    (g_sr.*stt.sin_ftp_x[tii] .+ b_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_pto.vato[tii] .= u_on_xfm.*(
                    (.-g_sr.*stt.sin_ftp_x[tii] .- b_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_pto.tau[tii] .= u_on_xfm.*(
                    .-(.-g_sr.*stt.cos_ftp_x[tii] .+ b_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau2_x[tii])
            grd.xfm_pto.phi[tii] .= grd.xfm_pto.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_pto.uon[tii] .= stt.pto_x[tii]
            end
    
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            @turbo grd.xfm_qto.vmfr[tii] .= u_on_xfm.*(
                    (b_sr.*stt.cos_ftp_x[tii] .+ g_sr.*stt.sin_ftp_x[tii]).*stt.vt_tau_x[tii])
            @turbo grd.xfm_qto.vmto[tii] .= u_on_xfm.*(2.0.*(.-b_sr.-b_to.-b_ch./2.0).*stt.vm_to_x[tii] .+
                    (b_sr.*stt.cos_ftp_x[tii] .+ g_sr.*stt.sin_ftp_x[tii]).*stt.vf_tau_x[tii])
            @turbo grd.xfm_qto.vafr[tii] .= u_on_xfm.*(
                    (.-b_sr.*stt.sin_ftp_x[tii] .+ g_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_qto.vato[tii] .= u_on_xfm.*(
                    (b_sr.*stt.sin_ftp_x[tii] .- g_sr.*stt.cos_ftp_x[tii]).*stt.vft_tau_x[tii])
            @turbo grd.xfm_qto.tau[tii]  .= u_on_xfm.*(
                    .-(b_sr.*stt.cos_ftp_x[tii] .+ g_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau2_x[tii])
            grd.xfm_qto.phi[tii] .= grd.xfm_qto.vato[tii]
            if qG.update_acline_xfm_bins
                grd.xfm_qto.uon[tii] .= stt.qto_x[tii]
            end
    
            # loop and take the gradients of the penalties
            if qG.acflow_grad_is_soft_abs
                # softabs
                @fastmath @inbounds @simd for xfm in 1:sys.nx
                    if (stt.xfm_sfr_plus[tii][xfm] > 0.0) && (stt.xfm_sfr_plus[tii][xfm] > stt.xfm_sto_plus[tii][xfm])
                        scale_fr                     = dt*qG.acflow_grad_weight*quasiGrad.soft_abs_acflow_grad(stt.xfm_sfr_plus[tii][xfm], qG)
                        grd.zs_xfm.xfm_pfr[tii][xfm] = scale_fr*stt.xfm_pfr[tii][xfm]./stt.xfm_sfr[tii][xfm]
                        grd.zs_xfm.xfm_qfr[tii][xfm] = scale_fr*stt.xfm_qfr[tii][xfm]./stt.xfm_sfr[tii][xfm]
                        grd.zs_xfm.xfm_pto[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qto[tii][xfm] = 0.0
                    elseif (stt.xfm_sto_plus[tii][xfm] > 0.0) && (stt.xfm_sto_plus[tii][xfm] > stt.xfm_sfr_plus[tii][xfm])
                        grd.zs_xfm.xfm_pfr[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qfr[tii][xfm] = 0.0
                        scale_to                     = dt*qG.acflow_grad_weight*quasiGrad.soft_abs_acflow_grad(stt.xfm_sto_plus[tii][xfm], qG)
                        grd.zs_xfm.xfm_pto[tii][xfm] = scale_to*stt.xfm_pto[tii][xfm]./stt.xfm_sto[tii][xfm]
                        grd.zs_xfm.xfm_qto[tii][xfm] = scale_to*stt.xfm_qto[tii][xfm]./stt.xfm_sto[tii][xfm]
                    else
                        grd.zs_xfm.xfm_pfr[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qfr[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_pto[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qto[tii][xfm] = 0.0
                    end
                end
            else
                # standard
                @fastmath @inbounds @simd for xfm in 1:sys.nx
                    if (stt.xfm_sfr_plus[tii][xfm] > 0.0) && (stt.xfm_sfr_plus[tii][xfm] > stt.xfm_sto_plus[tii][xfm])
                        grd.zs_xfm.xfm_pfr[tii][xfm] = dt*cs*stt.xfm_pfr[tii][xfm]/stt.xfm_sfr[tii][xfm]
                        grd.zs_xfm.xfm_qfr[tii][xfm] = dt*cs*stt.xfm_qfr[tii][xfm]/stt.xfm_sfr[tii][xfm]
                        grd.zs_xfm.xfm_pto[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qto[tii][xfm] = 0.0
                    elseif (stt.xfm_sto_plus[tii][xfm] > 0.0) && (stt.xfm_sto_plus[tii][xfm] > stt.xfm_sfr_plus[tii][xfm])
                        grd.zs_xfm.xfm_pfr[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qfr[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_pto[tii][xfm] = dt*cs*stt.xfm_pto[tii][xfm]./stt.xfm_sto[tii][xfm]
                        grd.zs_xfm.xfm_qto[tii][xfm] = dt*cs*stt.xfm_qto[tii][xfm]./stt.xfm_sto[tii][xfm]
                    else
                        grd.zs_xfm.xfm_pfr[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qfr[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_pto[tii][xfm] = 0.0
                        grd.zs_xfm.xfm_qto[tii][xfm] = 0.0
                    end
                end
            end
        end
    end
    
    # sleep tasks
    quasiGrad.Polyester.ThreadingUtilities.sleep_all_tasks()
end

# acline -- just the flows
function update_acline_sfr_flows!(idx::quasiGrad.Index, prm::quasiGrad.Param, stt::quasiGrad.State, tii::Int8)
    # line parameters
    g_sr = prm.acline.g_sr
    b_sr = prm.acline.b_sr
    b_ch = prm.acline.b_ch
    g_fr = prm.acline.g_fr
    b_fr = prm.acline.b_fr
    g_to = prm.acline.g_to
    b_to = prm.acline.b_to    
    # organize relevant line values
    stt.vm_fr[tii] .= @view stt.vm[tii][idx.acline_fr_bus]
    stt.va_fr[tii] .= @view stt.va[tii][idx.acline_fr_bus]
    stt.vm_to[tii] .= @view stt.vm[tii][idx.acline_to_bus]
    stt.va_to[tii] .= @view stt.va[tii][idx.acline_to_bus]
    
    # tools
    @turbo stt.cos_ftp[tii] .= quasiGrad.LoopVectorization.cos_fast.(stt.va_fr[tii] .- stt.va_to[tii])
    @turbo stt.sin_ftp[tii] .= quasiGrad.LoopVectorization.sin_fast.(stt.va_fr[tii] .- stt.va_to[tii])
    @turbo stt.vff[tii]     .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_fr[tii],2)
    @turbo stt.vtt[tii]     .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_to[tii],2) 
    @turbo stt.vft[tii]     .= stt.vm_fr[tii].*stt.vm_to[tii]
    
    # evaluate the function? we always need to in order to get the grd
    #
    # active power flow -- from -> to
    @turbo stt.pfr[tii] .= (g_sr.+g_fr).*stt.vff[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    @turbo stt.acline_pfr[tii] .= stt.u_on_acline[tii].*stt.pfr[tii]
    
    # reactive power flow -- from -> to
    @turbo stt.qfr[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*stt.vff[tii] .+ (b_sr.*stt.cos_ftp[tii] .- g_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    @turbo stt.acline_qfr[tii] .= stt.u_on_acline[tii].*stt.qfr[tii]
    
    # apparent power flow -- to -> from
    @turbo stt.acline_sfr[tii] .= quasiGrad.LoopVectorization.sqrt_fast.(quasiGrad.LoopVectorization.pow_fast.(stt.acline_pfr[tii],2) .+ quasiGrad.LoopVectorization.pow_fast.(stt.acline_qfr[tii],2))
end

# xfm line flows
function update_xfm_sfr_flows!(idx::quasiGrad.Index, prm::quasiGrad.Param, stt::quasiGrad.State, tii::Int8)
    g_sr = prm.xfm.g_sr
    b_sr = prm.xfm.b_sr
    b_ch = prm.xfm.b_ch
    g_fr = prm.xfm.g_fr
    b_fr = prm.xfm.b_fr
    g_to = prm.xfm.g_to
    b_to = prm.xfm.b_to    

    # call stt
    phi      = stt.phi[tii]
    tau      = stt.tau[tii]
    u_on_xfm = stt.u_on_xfm[tii]
    
    # organize relevant line values
    stt.vm_fr_x[tii] .= stt.vm[tii][idx.xfm_fr_bus]
    stt.va_fr_x[tii] .= stt.va[tii][idx.xfm_fr_bus]
    stt.vm_to_x[tii] .= stt.vm[tii][idx.xfm_to_bus]
    stt.va_to_x[tii] .= stt.va[tii][idx.xfm_to_bus]
    
    # tools
    @turbo stt.cos_ftp_x[tii]  .= quasiGrad.LoopVectorization.cos_fast.(stt.va_fr_x[tii] .- stt.va_to_x[tii] .- phi)
    @turbo stt.sin_ftp_x[tii]  .= quasiGrad.LoopVectorization.sin_fast.(stt.va_fr_x[tii] .- stt.va_to_x[tii] .- phi)
    @turbo stt.vff_x[tii]      .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_fr_x[tii],2)
    @turbo stt.vtt_x[tii]      .= quasiGrad.LoopVectorization.pow_fast.(stt.vm_to_x[tii],2)
    @turbo stt.vft_x[tii]      .= stt.vm_fr_x[tii].*stt.vm_to_x[tii]
    @turbo stt.vt_tau_x[tii]   .= stt.vm_to_x[tii]./tau
    @turbo stt.vf_tau_x[tii]   .= stt.vm_fr_x[tii]./tau
    @turbo stt.vf_tau2_x[tii]  .= stt.vf_tau_x[tii]./tau
    @turbo stt.vff_tau2_x[tii] .= stt.vff_x[tii]./quasiGrad.LoopVectorization.pow_fast.(tau,2)
    @turbo stt.vft_tau_x[tii]  .= stt.vft_x[tii]./tau
    @turbo stt.vft_tau2_x[tii] .= stt.vft_tau_x[tii]./tau
    @turbo stt.vff_tau3_x[tii] .= stt.vff_tau2_x[tii]./tau
    
    # evaluate the function? we always need to in order to get the grd
    #
    # active power flow -- from -> to
    @turbo stt.pfr_x[tii] .= (g_sr.+g_fr).*stt.vff_tau2_x[tii] .+ (.-g_sr.*stt.cos_ftp_x[tii] .- b_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau_x[tii]
    @turbo stt.xfm_pfr[tii] .= u_on_xfm.*stt.pfr_x[tii]
    
    # reactive power flow -- from -> to
    @turbo stt.qfr_x[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*stt.vff_tau2_x[tii] .+ (b_sr.*stt.cos_ftp_x[tii] .- g_sr.*stt.sin_ftp_x[tii]).*stt.vft_tau_x[tii]
    @turbo stt.xfm_qfr[tii] .= u_on_xfm.*stt.qfr_x[tii]
    
    # apparent power flow -- from -> to
    @turbo stt.xfm_sfr[tii] .= quasiGrad.LoopVectorization.sqrt_fast.(quasiGrad.LoopVectorization.pow_fast.(stt.xfm_pfr[tii],2) .+ quasiGrad.LoopVectorization.pow_fast.(stt.xfm_qfr[tii],2))
end