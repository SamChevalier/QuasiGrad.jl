# ac line flows
function acline_flows!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
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
    for tii in prm.ts.time_keys

        # duration
        dt = prm.ts.duration[tii]

        # call statuses
        u_on_lines = stt[:u_on_acline][tii]

        # organize relevant line values
        vm_fr = stt[:vm][tii][idx.acline_fr_bus]
        va_fr = stt[:va][tii][idx.acline_fr_bus]
        vm_to = stt[:vm][tii][idx.acline_to_bus]
        va_to = stt[:va][tii][idx.acline_to_bus]
        
        # tools
        cos_ftp  = cos.(va_fr - va_to)
        sin_ftp  = sin.(va_fr - va_to)
        vff      = vm_fr.^2
        vtt      = vm_to.^2
        vft      = vm_fr.*vm_to 
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        pfr = (g_sr+g_fr).*vff + (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft
        stt[:acline_pfr][tii] = u_on_lines.*pfr
        
        # reactive power flow -- from -> to
        qfr = (-b_sr-b_fr-b_ch/2).*vff + (b_sr.*cos_ftp - g_sr.*sin_ftp).*vft
        stt[:acline_qfr][tii] = u_on_lines.*qfr
        
        # apparent power flow -- to -> from
        acline_sfr = sqrt.(stt[:acline_pfr][tii].^2 + stt[:acline_qfr][tii].^2)
        
        # active power flow -- to -> from
        pto = (g_sr+g_to).*vtt + (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft
        stt[:acline_pto][tii] = u_on_lines.*pto
        
        # reactive power flow -- to -> from
        qto = (-b_sr-b_to-b_ch/2).*vtt + (b_sr.*cos_ftp + g_sr.*sin_ftp).*vft
        stt[:acline_qto][tii] = u_on_lines.*qto

        # apparent power flow -- to -> from
        acline_sto = sqrt.(stt[:acline_pto][tii].^2 + stt[:acline_qto][tii].^2)
        
        # penalty functions and scores
        acline_sfr_plus      = acline_sfr - prm.acline.mva_ub_nom
        acline_sto_plus      = acline_sto - prm.acline.mva_ub_nom
        stt[:zs_acline][tii] = dt*cs*max.(acline_sfr_plus, acline_sto_plus, 0.0)
        
        # ====================================================== #
        # ====================================================== #
        #
        # evaluate the grd?
        if qG.eval_grad
            # Gradients: active power flow -- from -> to
            grd[:acline_pfr][:vmfr][tii] = u_on_lines.*(2*(g_sr+g_fr).*vm_fr + 
                    (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_to)
            grd[:acline_pfr][:vmto][tii] = u_on_lines.*(
                    (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_fr)
            grd[:acline_pfr][:vafr][tii] = u_on_lines.*(
                    (g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
            grd[:acline_pfr][:vato][tii] = u_on_lines.*(
                    (-g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
            grd[:acline_pfr][:uon][tii] = pfr   
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            grd[:acline_qfr][:vmfr][tii] = u_on_lines.*(2*(-b_sr-b_fr-b_ch/2).*vm_fr +
                    (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_to)
            grd[:acline_qfr][:vmto][tii] = u_on_lines.*(
                    (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_fr)
            grd[:acline_qfr][:vafr][tii] = u_on_lines.*(
                    (-b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
            grd[:acline_qfr][:vato][tii] = u_on_lines.*(
                    (b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
            grd[:acline_qfr][:uon][tii]  = qfr 
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
            grd[:acline_pto][:vmfr][tii] = u_on_lines.*( 
                    (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_to)
            grd[:acline_pto][:vmto][tii] = u_on_lines.*(2*(g_sr+g_to).*vm_to +
                    (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_fr)
            grd[:acline_pto][:vafr][tii] = u_on_lines.*(
                    (g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
            grd[:acline_pto][:vato][tii] = u_on_lines.*(
                    (-g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
            grd[:acline_pto][:uon][tii] = pto
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            grd[:acline_qto][:vmfr][tii] = u_on_lines.*(
                    (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_to)
            grd[:acline_qto][:vmto][tii] = u_on_lines.*(2*(-b_sr-b_to-b_ch/2).*vm_to +
                    (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_fr)
            grd[:acline_qto][:vafr][tii] = u_on_lines.*(
                    (-b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
            grd[:acline_qto][:vato][tii] = u_on_lines.*(
                    (b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
            grd[:acline_qto][:uon][tii] = qto  
            # ====================================================== #
            # Gradients: apparent power flow -- to -> from
            #
            # penalty function derivatives
            #=
            grd[:acline_sfr_plus][:acline_pfr][tii] = stt[:acline_pfr][tii]./acline_sfr
            grd[:acline_sfr_plus][:acline_qfr][tii] = stt[:acline_qfr][tii]./acline_sfr
            grd[:acline_sto_plus][:acline_pto][tii] = stt[:acline_pto][tii]./acline_sto
            grd[:acline_sto_plus][:acline_qto][tii] = stt[:acline_qto][tii]./acline_sto 
            max_sfst0  = [argmax([spfr,spto,0]) for (spfr,spto) in zip(acline_sfr_plus,acline_sto_plus)]
            grd[:zs_acline][:acline_sfr_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sfr_plus][tii][max_sfst0 .== 1] .= dt*cs
            grd[:zs_acline][:acline_sto_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_acline][:acline_sto_plus][tii][max_sfst0 .== 2] .= dt*cs
            =#

            # indicators
            max_sfst0  = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(acline_sfr_plus,acline_sto_plus)]
            ind_fr = max_sfst0 .== 1
            ind_to = max_sfst0 .== 2

            # flush -- set to 0 => [Not(ind_fr)] and [Not(ind_to)] was slow/memory inefficient
            grd[:zs_acline][:acline_pfr][tii] .= 0.0
            grd[:zs_acline][:acline_qfr][tii] .= 0.0
            grd[:zs_acline][:acline_pto][tii] .= 0.0
            grd[:zs_acline][:acline_qto][tii] .= 0.0

            # gradients
            grd[:zs_acline][:acline_pfr][tii][ind_fr] = (dt*cs)*stt[:acline_pfr][tii][ind_fr]./acline_sfr[ind_fr]
            grd[:zs_acline][:acline_qfr][tii][ind_fr] = (dt*cs)*stt[:acline_qfr][tii][ind_fr]./acline_sfr[ind_fr]
            grd[:zs_acline][:acline_pto][tii][ind_to] = (dt*cs)*stt[:acline_pto][tii][ind_to]./acline_sto[ind_to]
            grd[:zs_acline][:acline_qto][tii][ind_to] = (dt*cs)*stt[:acline_qto][tii][ind_to]./acline_sto[ind_to]
        end
    end
end

# ac line flows
function xfm_flows!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
    g_sr = prm.xfm.g_sr
    b_sr = prm.xfm.b_sr
    b_ch = prm.xfm.b_ch
    g_fr = prm.xfm.g_fr
    b_fr = prm.xfm.b_fr
    g_to = prm.xfm.g_to
    b_to = prm.xfm.b_to
    
    # call penalty costs
    cs = prm.vio.s_flow

    # loop over time
    for tii in prm.ts.time_keys
        
        # duration
        dt = prm.ts.duration[tii]
    
        # call stt
        phi      = stt[:phi][tii]
        tau      = stt[:tau][tii]
        u_on_xfm = stt[:u_on_xfm][tii]

        # organize relevant line values
        vm_fr      = stt[:vm][tii][idx.xfm_fr_bus]
        va_fr      = stt[:va][tii][idx.xfm_fr_bus]
        vm_to      = stt[:vm][tii][idx.xfm_to_bus]
        va_to      = stt[:va][tii][idx.xfm_to_bus]
    
        # tools
        cos_ftp  = cos.(va_fr - va_to - phi)
        sin_ftp  = sin.(va_fr - va_to - phi)
        vff      = vm_fr.^2
        vtt      = vm_to.^2
        vft      = vm_fr.*vm_to
        vt_tau   = vm_to./tau
        vf_tau   = vm_fr./tau
        vf_tau2  = vf_tau./tau
        vff_tau2 = vff./(tau.^2)
        vft_tau  = vft./tau
        vft_tau2 = vft_tau./tau
        vff_tau3 = vff_tau2./tau
        
        # evaluate the function? we always need to in order to get the grd
        #
        # active power flow -- from -> to
        pfr = (g_sr+g_fr).*vff_tau2 + (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft_tau
        stt[:xfm_pfr][tii] = u_on_xfm.*pfr
        
        # reactive power flow -- from -> to
        qfr = (-b_sr-b_fr-b_ch/2).*vff_tau2 + (b_sr.*cos_ftp - g_sr.*sin_ftp).*vft_tau
        stt[:xfm_qfr][tii] = u_on_xfm.*qfr
        
        # apparent power flow -- from -> to
        xfm_sfr = sqrt.(stt[:xfm_pfr][tii].^2 + stt[:xfm_qfr][tii].^2)
        
        # active power flow -- to -> from
        pto = (g_sr+g_to).*vtt + (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft_tau
        stt[:xfm_pto][tii] = u_on_xfm.*pto
        
        # reactive power flow -- to -> from
        qto = (-b_sr-b_to-b_ch/2).*vtt + (b_sr.*cos_ftp + g_sr.*sin_ftp).*vft_tau
        stt[:xfm_qto][tii] = u_on_xfm.*qto
        
        # apparent power flow -- to -> from
        xfm_sto = sqrt.(stt[:xfm_pto][tii].^2 + stt[:xfm_qto][tii].^2)
        
        # penalty functions and scores
        xfm_sfr_plus      = xfm_sfr - prm.xfm.mva_ub_nom
        xfm_sto_plus      = xfm_sto - prm.xfm.mva_ub_nom
        stt[:zs_xfm][tii] = dt*cs*max.(xfm_sfr_plus, xfm_sto_plus, 0.0)
        
        # ====================================================== #
        # ====================================================== #
        
        # evaluate the grd?
        if qG.eval_grad     
            # Gradients: active power flow -- from -> to
            grd[:xfm_pfr][:vmfr][tii] = u_on_xfm.*(2*(g_sr+g_fr).*vf_tau2 + 
                    (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vt_tau)
            grd[:xfm_pfr][:vmto][tii] = u_on_xfm.*(
                    (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vf_tau)
            grd[:xfm_pfr][:vafr][tii] = u_on_xfm.*(
                    (g_sr.*sin_ftp - b_sr.*cos_ftp).*vft_tau)
            grd[:xfm_pfr][:vato][tii] = u_on_xfm.*(
                    (-g_sr.*sin_ftp + b_sr.*cos_ftp).*vft_tau)
            grd[:xfm_pfr][:tau][tii] = u_on_xfm.*(-2*(g_sr+g_fr).*vff_tau3 + 
                    -(-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft_tau2)
            grd[:xfm_pfr][:phi][tii] = grd[:xfm_pfr][:vato][tii]
            grd[:xfm_pfr][:uon][tii] = pfr
    
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            grd[:xfm_qfr][:vmfr][tii] = u_on_xfm.*(2*(-b_sr-b_fr-b_ch/2).*vf_tau2 +
                    (b_sr.*cos_ftp - g_sr.*sin_ftp).*vt_tau)
            grd[:xfm_qfr][:vmto][tii] = u_on_xfm.*(
                    (b_sr.*cos_ftp - g_sr.*sin_ftp).*vf_tau)
            grd[:xfm_qfr][:vafr][tii] = u_on_xfm.*(
                    (-b_sr.*sin_ftp - g_sr.*cos_ftp).*vft_tau)
            grd[:xfm_qfr][:vato][tii] = u_on_xfm.*(
                    (b_sr.*sin_ftp + g_sr.*cos_ftp).*vft_tau)
            grd[:xfm_qfr][:tau][tii]  = u_on_xfm.*(-2*(-b_sr-b_fr-b_ch/2).*vff_tau3 +
                    -(b_sr.*cos_ftp - g_sr.*sin_ftp).*vft_tau2)
            grd[:xfm_qfr][:phi][tii]  = grd[:xfm_qfr][:vato][tii]
            grd[:xfm_qfr][:uon][tii]  = qfr
    
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            # ...
            # ====================================================== #
            # Gradients: active power flow -- to -> from
            grd[:xfm_pto][:vmfr][tii] = u_on_xfm.*( 
                    (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vt_tau)
            grd[:xfm_pto][:vmto][tii] = u_on_xfm.*(2*(g_sr+g_to).*vm_to +
                    (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vf_tau)
            grd[:xfm_pto][:vafr][tii] = u_on_xfm.*(
                    (g_sr.*sin_ftp + b_sr.*cos_ftp).*vft_tau)
            grd[:xfm_pto][:vato][tii] = u_on_xfm.*(
                    (-g_sr.*sin_ftp - b_sr.*cos_ftp).*vft_tau)
            grd[:xfm_pto][:tau][tii] = u_on_xfm.*(
                    -(-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft_tau2)
            grd[:xfm_pto][:phi][tii] = grd[:xfm_pto][:vato][tii]
            grd[:xfm_pto][:uon][tii] = pto
    
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            grd[:xfm_qto][:vmfr][tii] = u_on_xfm.*(
                    (b_sr.*cos_ftp + g_sr.*sin_ftp).*vt_tau)
            grd[:xfm_qto][:vmto][tii] = u_on_xfm.*(2*(-b_sr-b_to-b_ch/2).*vm_to +
                    (b_sr.*cos_ftp + g_sr.*sin_ftp).*vf_tau)
            grd[:xfm_qto][:vafr][tii] = u_on_xfm.*(
                    (-b_sr.*sin_ftp + g_sr.*cos_ftp).*vft_tau)
            grd[:xfm_qto][:vato][tii] = u_on_xfm.*(
                    (b_sr.*sin_ftp - g_sr.*cos_ftp).*vft_tau)
            grd[:xfm_qto][:tau][tii]  = u_on_xfm.*(
                    -(b_sr.*cos_ftp + g_sr.*sin_ftp).*vft_tau2)
            grd[:xfm_qto][:phi][tii] = grd[:xfm_qto][:vato][tii]
            grd[:xfm_qto][:uon][tii] = qto
    
            # ====================================================== #
            # Gradients: apparent power flow -- to -> from
            #
            # penalty function derivatives
            #=
            grd[:xfm_sfr_plus][:xfm_pfr][tii] = stt[:xfm_pfr][tii]./xfm_sfr
            grd[:xfm_sfr_plus][:xfm_qfr][tii] = stt[:xfm_qfr][tii]./xfm_sfr
            grd[:xfm_sto_plus][:xfm_pto][tii] = stt[:xfm_pto][tii]./xfm_sto
            grd[:xfm_sto_plus][:xfm_qto][tii] = stt[:xfm_qto][tii]./xfm_sto
    
            max_sfst0  = [argmax([spfr,spto,0]) for (spfr,spto) in zip(xfm_sfr_plus,xfm_sto_plus)]
            grd[:zs_xfm][:xfm_sfr_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_xfm][:xfm_sfr_plus][tii][max_sfst0 .== 1] .= dt*cs
            grd[:zs_xfm][:xfm_sto_plus][tii] = zeros(length(max_sfst0))
            grd[:zs_xfm][:xfm_sto_plus][tii][max_sfst0 .== 2] .= dt*cs
            =#

            # indicators
            max_sfst0  = [argmax([spfr, spto, 0.0]) for (spfr,spto) in zip(xfm_sfr_plus,xfm_sto_plus)]
            ind_fr = max_sfst0 .== 1
            ind_to = max_sfst0 .== 2

            # flush -- set to 0 => [Not(ind_fr)] and [Not(ind_to)] was slow/memory inefficient
            grd[:zs_xfm][:xfm_pfr][tii] .= 0.0
            grd[:zs_xfm][:xfm_qfr][tii] .= 0.0
            grd[:zs_xfm][:xfm_pto][tii] .= 0.0
            grd[:zs_xfm][:xfm_qto][tii] .= 0.0
    
            # gradients
            grd[:zs_xfm][:xfm_pfr][tii][ind_fr] = (dt*cs)*stt[:xfm_pfr][tii][ind_fr]./xfm_sfr[ind_fr]
            grd[:zs_xfm][:xfm_qfr][tii][ind_fr] = (dt*cs)*stt[:xfm_qfr][tii][ind_fr]./xfm_sfr[ind_fr]
            grd[:zs_xfm][:xfm_pto][tii][ind_to] = (dt*cs)*stt[:xfm_pto][tii][ind_to]./xfm_sto[ind_to]
            grd[:zs_xfm][:xfm_qto][tii][ind_to] = (dt*cs)*stt[:xfm_qto][tii][ind_to]./xfm_sto[ind_to]
        end
    end
end