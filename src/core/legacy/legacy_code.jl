# ac line flows
function acline_flows_and_grads!(eval_grad::Bool, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol, SparseMatrixCSC{Float64, Int64}}, prm::quasiGrad.Param, state::Dict{Symbol,Vector{Float64}}, idx::quasiGrad.Idx{Symbol, Any})
    # line parameters
    g_sr = prm.acline.g_sr
    b_sr = prm.acline.b_sr
    b_ch = prm.acline.b_ch
    g_fr = prm.acline.g_fr
    b_fr = prm.acline.b_fr
    g_to = prm.acline.g_to
    b_to = prm.acline.b_to

    # loop over time
    for tii in keys(state[:pb_slack])

        # call statuses
        u_on_lines = state[:u_on_acline][tii]

        # organize relevant line values
        vm_fr      = state[:vm][tii][idx.acline_fr_bus]
        va_fr      = state[:va][tii][idx.acline_fr_bus]
        vm_to      = state[:vm][tii][idx.acline_to_bus]
        va_to      = state[:va][tii][idx.acline_to_bus]
        
        # tools
        cos_ftp  = cos.(va_fr - va_to)
        sin_ftp  = sin.(va_fr - va_to)
        vff      = vm_fr.^2
        vtt      = vm_to.^2
        vft      = vm_fr.*vm_to 
        
        # evaluate the function? we always need to in order to get the grad
        #
        # active power flow -- from -> to
        state[:acline_pfr][tii] = u_on_lines.*((g_sr+g_fr).*vff + 
                (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft)
        
        # reactive power flow -- from -> to
        state[:acline_qfr][tii] = u_on_lines.*((-b_sr-b_fr-b_ch/2).*vff +
                (b_sr.*cos_ftp - g_sr.*sin_ftp).*vft)
        
        # apparent power flow -- to -> from
        state[:acline_sfr][tii] = sqrt.(state[:acline_pfr][tii].^2 + state[:acline_qfr][tii].^2)
        
        # active power flow -- to -> from
        state[:acline_pto][tii] = u_on_lines.*((g_sr+g_to).*vtt + 
                (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft)
        
        # reactive power flow -- to -> from
        state[:acline_qto][tii] = u_on_lines.*((-b_sr-b_to-b_ch/2).*vtt +
                (b_sr.*cos_ftp + g_sr.*sin_ftp).*vft)

        # apparent power flow -- to -> from
        state[:acline_sto][tii] = sqrt.(state[:acline_pto][tii].^2 + state[:acline_qto][tii].^2)
        
        # ====================================================== #
        # ====================================================== #
        #
        # evaluate the grad?
        if eval_grad == true     
                # Gradients: active power flow -- from -> to
                grad[:acline_pfr][:vmfr][tii] = u_on_lines.*(2*(g_sr+g_fr).*vm_fr + 
                        (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_to)
                grad[:acline_pfr][:vmto][tii] = u_on_lines.*(
                        (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_fr)
                grad[:acline_pfr][:vafr][tii] = u_on_lines.*(
                        (g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
                grad[:acline_pfr][:vato][tii] = u_on_lines.*(
                        (-g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
                grad[:acline_pfr][:uon][tii] = state[:acline_pfr][tii]

                # ====================================================== #
                # Gradients: reactive power flow -- from -> to
                grad[:acline_qfr][:vmfr][tii] = u_on_lines.*(2*(-b_sr-b_fr-b_ch/2).*vm_fr +
                        (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_to)
                grad[:acline_qfr][:vmto][tii] = u_on_lines.*(
                        (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_fr)
                grad[:acline_qfr][:vafr][tii] = u_on_lines.*(
                        (-b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
                grad[:acline_qfr][:vato][tii] = u_on_lines.*(
                        (b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
                grad[:acline_qfr][:uon][tii]  = state[:acline_qfr][tii]

                # ====================================================== #
                # Gradients: apparent power flow -- from -> to
                grad[:acline_sfr][:pfr][tii]  = state[:acline_pfr][tii]./state[:acline_sfr][tii]
                grad[:acline_sfr][:qfr][tii]  = state[:acline_qfr][tii]./state[:acline_sfr][tii]

                # ====================================================== #
                # Gradients: active power flow -- to -> from
                grad[:acline_pto][:vmfr][tii] = u_on_lines.*( 
                        (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_to)
                grad[:acline_pto][:vmto][tii] = u_on_lines.*(2*(g_sr+g_to).*vm_to +
                        (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_fr)
                grad[:acline_pto][:vafr][tii] = u_on_lines.*(
                        (g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
                grad[:acline_pto][:vato][tii] = u_on_lines.*(
                        (-g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
                grad[:acline_pto][:uon][tii] = state[:acline_pto][tii]

                # ====================================================== #
                # Gradients: reactive power flow -- to -> from
                grad[:acline_qto][:vmfr][tii] = u_on_lines.*(
                        (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_to)
                grad[:acline_qto][:vmto][tii] = u_on_lines.*(2*(-b_sr-b_to-b_ch/2).*vm_to +
                        (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_fr)
                grad[:acline_qto][:vafr][tii] = u_on_lines.*(
                        (-b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
                grad[:acline_qto][:vato][tii] = u_on_lines.*(
                        (b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
                grad[:acline_qto][:uon][tii] = state[:acline_qto][tii]

                # ====================================================== #
                # Gradients: apparent power flow -- to -> from
                grad[:acline_sto][:pto][tii]  = state[:acline_pto][tii]./state[:acline_sto][tii]
                grad[:acline_sto][:qto][tii]  = state[:acline_qto][tii]./state[:acline_sto][tii]

                # now, what we really want is the gradients accociated with a single
                # given variable -- place them into a sparse Jacobian matrix
                ///////////////////////////////////

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dpfr_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dpfr_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dpfr_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dpfr_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dqfr_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dqfr_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dqfr_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dqfr_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dpto_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dpto_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dpto_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dpto_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dqto_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dqto_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dqto_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dqto_dvato]
        end
        end
end

# ac line flows
function xfm_flows_and_grads!(eval_grad::Bool, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol, SparseMatrixCSC{Float64, Int64}}, prm::quasiGrad.Param, state::Dict{Symbol,Vector{Float64}}, idx::quasiGrad.Idx{Symbol, Any})
    g_sr = prm.xfm.g_sr
    b_sr = prm.xfm.b_sr
    b_ch = prm.xfm.b_ch
    g_fr = prm.xfm.g_fr
    b_fr = prm.xfm.b_fr
    g_to = prm.xfm.g_to
    b_to = prm.xfm.b_to
    
    # call state
    phi      = state[:phi]
    tau      = state[:tau]
    u_on_xfm = state[:u_on_xfm]
        
    # organize relevant line values
    vm_fr      = state[:vm][idx.xfm_fr_bus]
    va_fr      = state[:va][idx.xfm_fr_bus]
    vm_to      = state[:vm][idx.xfm_to_bus]
    va_to      = state[:va][idx.xfm_to_bus]
    
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
    
    # evaluate the function? we always need to in order to get the grad
    #
    # active power flow -- from -> to
    state[:xfm_pfr] = u_on_xfm.*((g_sr+g_fr).*vff_tau2 + 
        (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft_tau)
    
    # reactive power flow -- from -> to
    state[:xfm_qfr] = u_on_xfm.*((-b_sr-b_fr-b_ch/2).*vff_tau2 +
        (b_sr.*cos_ftp - g_sr.*sin_ftp).*vft_tau)
    
    # apparent power flow -- from -> to
    state[:xfm_sfr] = sqrt.(state[:xfm_pfr].^2 + state[:xfm_qfr].^2)
    
    # active power flow -- to -> from
    state[:xfm_pto] = u_on_xfm.*((g_sr+g_to).*vtt + 
        (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft_tau)
    
    # reactive power flow -- to -> from
    state[:xfm_qto] = u_on_xfm.*((-b_sr-b_to-b_ch/2).*vtt +
        (b_sr.*cos_ftp + g_sr.*sin_ftp).*vft_tau)
    
    # apparent power flow -- to -> from
    state[:xfm_sto] = sqrt.(state[:xfm_pto].^2 + state[:xfm_qto].^2)
    
    # ====================================================== #
    # ====================================================== #
    
    # evaluate the grad?
    if eval_grad == true     
        # Gradients: active power flow -- from -> to
        grad[:xfm_dpfr_dvmfr] = u_on_xfm.*(2*(g_sr+g_fr).*vf_tau2 + 
                (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vt_tau)
        grad[:xfm_dpfr_dvmto] = u_on_xfm.*(
                (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vf_tau)
        grad[:xfm_dpfr_dvafr] = u_on_xfm.*(
                (g_sr.*sin_ftp - b_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dpfr_dvato] = u_on_xfm.*(
                (-g_sr.*sin_ftp + b_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dpfr_dtau] = u_on_xfm.*(-2*(g_sr+g_fr).*vff_tau3 + 
                -(-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft_tau2)
        grad[:xfm_dpfr_dphi] = grad[:xfm_dpfr_dvato]
        grad[:xfm_dpfr_duon] = state[:xfm_pfr]

        # ====================================================== #
        # Gradients: reactive power flow -- from -> to
        grad[:xfm_dqfr_dvmfr] = u_on_xfm.*(2*(-b_sr-b_fr-b_ch/2).*vf_tau2 +
                (b_sr.*cos_ftp - g_sr.*sin_ftp).*vt_tau)
        grad[:xfm_dqfr_dvmto] = u_on_xfm.*(
                (b_sr.*cos_ftp - g_sr.*sin_ftp).*vf_tau)
        grad[:xfm_dqfr_dvafr] = u_on_xfm.*(
                (-b_sr.*sin_ftp - g_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dqfr_dvato] = u_on_xfm.*(
                (b_sr.*sin_ftp + g_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dqfr_dtau]  = u_on_xfm.*(-2*(-b_sr-b_fr-b_ch/2).*vff_tau3 +
                -(b_sr.*cos_ftp - g_sr.*sin_ftp).*vft_tau2)
        grad[:xfm_dqfr_dphi]  = grad[:xfm_dqfr_dvato]
        grad[:xfm_dqfr_duon]  = state[:xfm_qfr]

        # ====================================================== #
        # Gradients: apparent power flow -- from -> to
        grad[:xfm_dsfr_dpfr]  = state[:xfm_pfr]./state[:xfm_sfr]
        grad[:xfm_dsfr_dqfr]  = state[:xfm_qfr]./state[:xfm_sfr]

        # ====================================================== #
        # Gradients: active power flow -- to -> from
        grad[:xfm_dpto_dvmfr] = u_on_xfm.*( 
                (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vt_tau)
        grad[:xfm_dpto_dvmto] = u_on_xfm.*(2*(g_sr+g_to).*vm_to +
                (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vf_tau)
        grad[:xfm_dpto_dvafr] = u_on_xfm.*(
                (g_sr.*sin_ftp + b_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dpto_dvato] = u_on_xfm.*(
                (-g_sr.*sin_ftp - b_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dpto_dtau] = u_on_xfm.*(
                -(-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft_tau2)
        grad[:xfm_dpto_dphi] = grad[:xfm_dpto_dvato]
        grad[:xfm_dpto_duon] = state[:xfm_pto]

        # ====================================================== #
        # Gradients: reactive power flow -- to -> from
        grad[:xfm_dqto_dvmfr] = u_on_xfm.*(
                (b_sr.*cos_ftp + g_sr.*sin_ftp).*vt_tau)
        grad[:xfm_dqto_dvmto] = u_on_xfm.*(2*(-b_sr-b_to-b_ch/2).*vm_to +
                (b_sr.*cos_ftp + g_sr.*sin_ftp).*vf_tau)
        grad[:xfm_dqto_dvafr] = u_on_xfm.*(
                (-b_sr.*sin_ftp + g_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dqto_dvato] = u_on_xfm.*(
                (b_sr.*sin_ftp - g_sr.*cos_ftp).*vft_tau)
        grad[:xfm_dqto_dtau]  = u_on_xfm.*(
                -(b_sr.*cos_ftp + g_sr.*sin_ftp).*vft_tau2)
        grad[:xfm_dqto_dphi] = grad[:xfm_dqto_dvato]
        grad[:xfm_dqto_duon] = state[:xfm_qto]

        # ====================================================== #
        # Gradients: apparent power flow -- to -> from
        grad[:xfm_dsto_dpto]  = state[:xfm_pto]./state[:xfm_sto]
        grad[:xfm_dsto_dqto]  = state[:xfm_qto]./state[:xfm_sto]

        # now, what we really want is the gradients accociated with a single
        # given variable -- place them into a sparse Jacobian matrix
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dpfr_dvmfr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dpfr_dvmto]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dpfr_dvafr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dpfr_dvato]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dpfr_dtau]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dpfr_dphi]

        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dqfr_dvmfr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dqfr_dvmto]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dqfr_dvafr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dqfr_dvato]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dqfr_dtau]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dqfr_dphi]

        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dpto_dvmfr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dpto_dvmto]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dpto_dvafr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dpto_dvato]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dpto_dtau]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dpto_dphi]

        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dqto_dvmfr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dqto_dvmto]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dqto_dvafr]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dqto_dvato]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dpto_dtau]
        jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dpto_dphi]
    end
end
    
# ac line flows
function acline_flows!(eval_grad::Bool, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol, SparseMatrixCSC{Float64, Int64}}, prm::quasiGrad.Param, state::Dict{Symbol,Vector{Float64}}, idx::quasiGrad.Idx{Symbol, Any})
        # line parameters
        g_sr = prm.acline.g_sr
        b_sr = prm.acline.b_sr
        b_ch = prm.acline.b_ch
        g_fr = prm.acline.g_fr
        b_fr = prm.acline.b_fr
        g_to = prm.acline.g_to
        b_to = prm.acline.b_to
    
        # call penalty costs
        cs = prm[:vio_cost][:s_vio_cost]
    
        # loop over time
        for tii in keys(prm[:time_series][:duration])
    
            # duration
            dt = prm[:time_series][:duration][tii]
    
            # call statuses
            u_on_lines = state[:u_on_acline][tii]
    
            # organize relevant line values
            vm_fr      = state[:vm][tii][idx.acline_fr_bus]
            va_fr      = state[:va][tii][idx.acline_fr_bus]
            vm_to      = state[:vm][tii][idx.acline_to_bus]
            va_to      = state[:va][tii][idx.acline_to_bus]
            
            # tools
            cos_ftp  = cos.(va_fr - va_to)
            sin_ftp  = sin.(va_fr - va_to)
            vff      = vm_fr.^2
            vtt      = vm_to.^2
            vft      = vm_fr.*vm_to 
            
            # evaluate the function? we always need to in order to get the grad
            #
            # active power flow -- from -> to
            state[:acline_pfr][tii] = u_on_lines.*((g_sr+g_fr).*vff + 
                    (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft)
            
            # reactive power flow -- from -> to
            state[:acline_qfr][tii] = u_on_lines.*((-b_sr-b_fr-b_ch/2).*vff +
                    (b_sr.*cos_ftp - g_sr.*sin_ftp).*vft)
            
            # apparent power flow -- to -> from
            state[:acline_sfr][tii] = sqrt.(state[:acline_pfr][tii].^2 + state[:acline_qfr][tii].^2)
            
            # active power flow -- to -> from
            state[:acline_pto][tii] = u_on_lines.*((g_sr+g_to).*vtt + 
                    (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft)
            
            # reactive power flow -- to -> from
            state[:acline_qto][tii] = u_on_lines.*((-b_sr-b_to-b_ch/2).*vtt +
                    (b_sr.*cos_ftp + g_sr.*sin_ftp).*vft)
    
            # apparent power flow -- to -> from
            state[:acline_sto][tii] = sqrt.(state[:acline_pto][tii].^2 + state[:acline_qto][tii].^2)
            
            # penalty functions and scores
            state[:acline_sfr_plus][tii] = state[:acline_sfr][tii] - prm.acline.mva_ub_nom
            state[:acline_sto_plus][tii] = state[:acline_sto][tii] - prm.acline.mva_ub_nom
            state[:acline_s_plus][tii]   = max.(state[:acline_sfr_plus][tii], state[:acline_sto_plus][tii], 0)
            state[:zs_acline][tii]       = dt*cs*state[:acline_s_plus][tii]
            
            # ====================================================== #
            # ====================================================== #
            #
            # evaluate the grad?
            if eval_grad == true     
                    # Gradients: active power flow -- from -> to
                    grad[:acline_pfr][:vmfr][tii] = u_on_lines.*(2*(g_sr+g_fr).*vm_fr + 
                            (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_to)
                    grad[:acline_pfr][:vmto][tii] = u_on_lines.*(
                            (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vm_fr)
                    grad[:acline_pfr][:vafr][tii] = u_on_lines.*(
                            (g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
                    grad[:acline_pfr][:vato][tii] = u_on_lines.*(
                            (-g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
                    grad[:acline_pfr][:uon][tii] = state[:acline_pfr][tii]
    
                    # ====================================================== #
                    # Gradients: reactive power flow -- from -> to
                    grad[:acline_qfr][:vmfr][tii] = u_on_lines.*(2*(-b_sr-b_fr-b_ch/2).*vm_fr +
                            (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_to)
                    grad[:acline_qfr][:vmto][tii] = u_on_lines.*(
                            (b_sr.*cos_ftp - g_sr.*sin_ftp).*vm_fr)
                    grad[:acline_qfr][:vafr][tii] = u_on_lines.*(
                            (-b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
                    grad[:acline_qfr][:vato][tii] = u_on_lines.*(
                            (b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
                    grad[:acline_qfr][:uon][tii]  = state[:acline_qfr][tii]
    
                    # ====================================================== #
                    # Gradients: apparent power flow -- from -> to
                    grad[:acline_sfr][:pfr][tii]  = state[:acline_pfr][tii]./state[:acline_sfr][tii]
                    grad[:acline_sfr][:qfr][tii]  = state[:acline_qfr][tii]./state[:acline_sfr][tii]
    
                    # ====================================================== #
                    # Gradients: active power flow -- to -> from
                    grad[:acline_pto][:vmfr][tii] = u_on_lines.*( 
                            (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_to)
                    grad[:acline_pto][:vmto][tii] = u_on_lines.*(2*(g_sr+g_to).*vm_to +
                            (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vm_fr)
                    grad[:acline_pto][:vafr][tii] = u_on_lines.*(
                            (g_sr.*sin_ftp + b_sr.*cos_ftp).*vft)
                    grad[:acline_pto][:vato][tii] = u_on_lines.*(
                            (-g_sr.*sin_ftp - b_sr.*cos_ftp).*vft)
                    grad[:acline_pto][:uon][tii] = state[:acline_pto][tii]
    
                    # ====================================================== #
                    # Gradients: reactive power flow -- to -> from
                    grad[:acline_qto][:vmfr][tii] = u_on_lines.*(
                            (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_to)
                    grad[:acline_qto][:vmto][tii] = u_on_lines.*(2*(-b_sr-b_to-b_ch/2).*vm_to +
                            (b_sr.*cos_ftp + g_sr.*sin_ftp).*vm_fr)
                    grad[:acline_qto][:vafr][tii] = u_on_lines.*(
                            (-b_sr.*sin_ftp + g_sr.*cos_ftp).*vft)
                    grad[:acline_qto][:vato][tii] = u_on_lines.*(
                            (b_sr.*sin_ftp - g_sr.*cos_ftp).*vft)
                    grad[:acline_qto][:uon][tii] = state[:acline_qto][tii]
    
                    # ====================================================== #
                    # Gradients: apparent power flow -- to -> from
                    grad[:acline_sto][:pto][tii]  = state[:acline_pto][tii]./state[:acline_sto][tii]
                    grad[:acline_sto][:qto][tii]  = state[:acline_qto][tii]./state[:acline_sto][tii]
    
                    # penalty function derivatives
                    grad[:acline_sfr_plus][:acline_pfr][tii] = state[:acline_pfr][tii]./state[:acline_sfr][tii]
                    grad[:acline_sfr_plus][:acline_qfr][tii] = state[:acline_qfr][tii]./state[:acline_sfr][tii]
                    grad[:acline_sto_plus][:acline_pto][tii] = state[:acline_pto][tii]./state[:acline_sto][tii]
                    grad[:acline_sto_plus][:acline_qto][tii] = state[:acline_qto][tii]./state[:acline_sto][tii]
                    
                    max_sfst0  = [argmax([spfr,spto,0]) for (spfr,spto) in zip(state[:acline_sfr_plus][tii],state[:acline_sto_plus][tii])]
                    grad[:acline_s_plus][:acline_sfr_plus][tii] = zeros(length(max_sfst0))
                    grad[:acline_s_plus][:acline_sfr_plus][tii][max_sfst0 .== 1] .= 1
                    grad[:acline_s_plus][:acline_sto_plus][tii] = zeros(length(max_sfst0))
                    grad[:acline_s_plus][:acline_sto_plus][tii][max_sfst0 .== 2] .= 1
                    grad[:zs_acline][:acline_s_plus][tii] = dt*cs
    
                    # now, what we really want is the gradients accociated with a single
                    # given variable -- place them into a sparse Jacobian matrix
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dpfr_dvmfr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dpfr_dvmto]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dpfr_dvafr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dpfr_dvato]
    
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dqfr_dvmfr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dqfr_dvmto]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dqfr_dvafr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dqfr_dvato]
    
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dpto_dvmfr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dpto_dvmto]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dpto_dvafr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dpto_dvato]
    
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmfr]))] = grad[:acline_dqto_dvmfr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmto]))] = grad[:acline_dqto_dvmto]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vafr]))] = grad[:acline_dqto_dvafr]
                    jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vato]))] = grad[:acline_dqto_dvato]
            end
            end
    end
    
    # ac line flows
    function xfm_flows!(eval_grad::Bool, grad::Dict{Symbol,Vector{Float64}}, jac::Dict{Symbol, SparseMatrixCSC{Float64, Int64}}, prm::quasiGrad.Param, state::Dict{Symbol,Vector{Float64}}, idx::quasiGrad.Idx{Symbol, Any})
        g_sr = prm.xfm.g_sr
        b_sr = prm.xfm.b_sr
        b_ch = prm.xfm.b_ch
        g_fr = prm.xfm.g_fr
        b_fr = prm.xfm.b_fr
        g_to = prm.xfm.g_to
        b_to = prm.xfm.b_to
        
        # call state
        phi      = state[:phi]
        tau      = state[:tau]
        u_on_xfm = state[:u_on_xfm]
    
        # organize relevant line values
        vm_fr      = state[:vm][idx.xfm_fr_bus]
        va_fr      = state[:va][idx.xfm_fr_bus]
        vm_to      = state[:vm][idx.xfm_to_bus]
        va_to      = state[:va][idx.xfm_to_bus]
        
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
        
        # evaluate the function? we always need to in order to get the grad
        #
        # active power flow -- from -> to
        state[:xfm_pfr] = u_on_xfm.*((g_sr+g_fr).*vff_tau2 + 
            (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft_tau)
        
        # reactive power flow -- from -> to
        state[:xfm_qfr] = u_on_xfm.*((-b_sr-b_fr-b_ch/2).*vff_tau2 +
            (b_sr.*cos_ftp - g_sr.*sin_ftp).*vft_tau)
        
        # apparent power flow -- from -> to
        state[:xfm_sfr] = sqrt.(state[:xfm_pfr].^2 + state[:xfm_qfr].^2)
        
        # active power flow -- to -> from
        state[:xfm_pto] = u_on_xfm.*((g_sr+g_to).*vtt + 
            (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft_tau)
        
        # reactive power flow -- to -> from
        state[:xfm_qto] = u_on_xfm.*((-b_sr-b_to-b_ch/2).*vtt +
            (b_sr.*cos_ftp + g_sr.*sin_ftp).*vft_tau)
        
        # apparent power flow -- to -> from
        state[:xfm_sto] = sqrt.(state[:xfm_pto].^2 + state[:xfm_qto].^2)
        
        # ====================================================== #
        # ====================================================== #
        
        # evaluate the grad?
        if eval_grad == true     
            # Gradients: active power flow -- from -> to
            grad[:xfm_dpfr_dvmfr] = u_on_xfm.*(2*(g_sr+g_fr).*vf_tau2 + 
                    (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vt_tau)
            grad[:xfm_dpfr_dvmto] = u_on_xfm.*(
                    (-g_sr.*cos_ftp - b_sr.*sin_ftp).*vf_tau)
            grad[:xfm_dpfr_dvafr] = u_on_xfm.*(
                    (g_sr.*sin_ftp - b_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dpfr_dvato] = u_on_xfm.*(
                    (-g_sr.*sin_ftp + b_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dpfr_dtau] = u_on_xfm.*(-2*(g_sr+g_fr).*vff_tau3 + 
                    -(-g_sr.*cos_ftp - b_sr.*sin_ftp).*vft_tau2)
            grad[:xfm_dpfr_dphi] = grad[:xfm_dpfr_dvato]
            grad[:xfm_dpfr_duon] = state[:xfm_pfr]
    
            # ====================================================== #
            # Gradients: reactive power flow -- from -> to
            grad[:xfm_dqfr_dvmfr] = u_on_xfm.*(2*(-b_sr-b_fr-b_ch/2).*vf_tau2 +
                    (b_sr.*cos_ftp - g_sr.*sin_ftp).*vt_tau)
            grad[:xfm_dqfr_dvmto] = u_on_xfm.*(
                    (b_sr.*cos_ftp - g_sr.*sin_ftp).*vf_tau)
            grad[:xfm_dqfr_dvafr] = u_on_xfm.*(
                    (-b_sr.*sin_ftp - g_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dqfr_dvato] = u_on_xfm.*(
                    (b_sr.*sin_ftp + g_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dqfr_dtau]  = u_on_xfm.*(-2*(-b_sr-b_fr-b_ch/2).*vff_tau3 +
                    -(b_sr.*cos_ftp - g_sr.*sin_ftp).*vft_tau2)
            grad[:xfm_dqfr_dphi]  = grad[:xfm_dqfr_dvato]
            grad[:xfm_dqfr_duon]  = state[:xfm_qfr]
    
            # ====================================================== #
            # Gradients: apparent power flow -- from -> to
            grad[:xfm_dsfr_dpfr]  = state[:xfm_pfr]./state[:xfm_sfr]
            grad[:xfm_dsfr_dqfr]  = state[:xfm_qfr]./state[:xfm_sfr]
    
            # ====================================================== #
            # Gradients: active power flow -- to -> from
            grad[:xfm_dpto_dvmfr] = u_on_xfm.*( 
                    (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vt_tau)
            grad[:xfm_dpto_dvmto] = u_on_xfm.*(2*(g_sr+g_to).*vm_to +
                    (-g_sr.*cos_ftp + b_sr.*sin_ftp).*vf_tau)
            grad[:xfm_dpto_dvafr] = u_on_xfm.*(
                    (g_sr.*sin_ftp + b_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dpto_dvato] = u_on_xfm.*(
                    (-g_sr.*sin_ftp - b_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dpto_dtau] = u_on_xfm.*(
                    -(-g_sr.*cos_ftp + b_sr.*sin_ftp).*vft_tau2)
            grad[:xfm_dpto_dphi] = grad[:xfm_dpto_dvato]
            grad[:xfm_dpto_duon] = state[:xfm_pto]
    
            # ====================================================== #
            # Gradients: reactive power flow -- to -> from
            grad[:xfm_dqto_dvmfr] = u_on_xfm.*(
                    (b_sr.*cos_ftp + g_sr.*sin_ftp).*vt_tau)
            grad[:xfm_dqto_dvmto] = u_on_xfm.*(2*(-b_sr-b_to-b_ch/2).*vm_to +
                    (b_sr.*cos_ftp + g_sr.*sin_ftp).*vf_tau)
            grad[:xfm_dqto_dvafr] = u_on_xfm.*(
                    (-b_sr.*sin_ftp + g_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dqto_dvato] = u_on_xfm.*(
                    (b_sr.*sin_ftp - g_sr.*cos_ftp).*vft_tau)
            grad[:xfm_dqto_dtau]  = u_on_xfm.*(
                    -(b_sr.*cos_ftp + g_sr.*sin_ftp).*vft_tau2)
            grad[:xfm_dqto_dphi] = grad[:xfm_dqto_dvato]
            grad[:xfm_dqto_duon] = state[:xfm_qto]
    
            # ====================================================== #
            # Gradients: apparent power flow -- to -> from
            grad[:xfm_dsto_dpto]  = state[:xfm_pto]./state[:xfm_sto]
            grad[:xfm_dsto_dqto]  = state[:xfm_qto]./state[:xfm_sto]
    
            # now, what we really want is the gradients accociated with a single
            # given variable -- place them into a sparse Jacobian matrix
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dpfr_dvmfr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dpfr_dvmto]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dpfr_dvafr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dpfr_dvato]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dpfr_dtau]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dpfr_dphi]
    
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dqfr_dvmfr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dqfr_dvmto]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dqfr_dvafr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dqfr_dvato]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dqfr_dtau]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dqfr_dphi]
    
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dpto_dvmfr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dpto_dvmto]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dpto_dvafr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dpto_dvato]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dpto_dtau]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dpto_dphi]
    
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vmfr]))] = grad[:xfm_dqto_dvmfr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vmto]))] = grad[:xfm_dqto_dvmto]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vafr]))] = grad[:xfm_dqto_dvafr]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vato]))] = grad[:xfm_dqto_dvato]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_tau]))]  = grad[:xfm_dpto_dtau]
            jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_phi]))]  = grad[:xfm_dpto_dphi]
        end
    end
        
    

        # compute the energy cost of each device -- producers
        for dev in idx[:pr_dev]
                dev_ind = idx[:pr_dev_ind][dev]
                state[:zen_pr][tii][dev_ind] = dt*sum(prm[:pr][:c_en][ll]*max(min(state[:pr_p][tii][dev]-prm[:pr_p_cum_max][tii][dev][ll],prm[:pr_p_block][tii][dev][ll]),0) for ll in prm[:pr_p_block_inds][tii][dev]; init=0.0)
    
                # evaluate the grad?
                if eval_grad == true
                    # what is the index of the "active" block?
                    del = state[:pr_p][tii][dev] .- prm[:pr_p_cum_max][tii][dev]
                    active_block_ind = argmin(del[del .>= 0])
                    grad[:zen_pr][:pr_p][tii][dev_ind] = prm[:pr][:c_en][active_block_ind]
                end
            end
    
            # compute the energy cost of each device -- consumers
            for dev in idx[:cs_dev]
                dev_ind = idx[:cs_dev_ind][dev]
                state[:zen_cs][tii][dev_ind] = dt*sum(prm[:cs][:c_en][ll]*max(min(state[:cs_p][tii][dev]-prm[:cs_p_cum_max][tii][dev][ll],prm[:cs_p_block][tii][dev][ll]),0) for ll in prm[:cs_p_block_inds][tii][dev]; init=0.0)
    
                # evaluate the grad?
                if eval_grad == true
                    # what is the index of the "active" block?
                    del = state[:cs_p][tii][dev] .- prm[:cs_p_cum_max][tii][dev]
                    active_block_ind = argmin(del[del .>= 0])
                    grad[:zen_cs][:cs_p][tii][dev_ind] = prm[:cs][:c_en][active_block_ind]
                end
            end

# update the flow vectors
function update_all_flows!(prm::quasiGrad.Param, idx::quasiGrad.Idx, state::Dict, grad::Dict, eval_grad::Bool)
        # loop over time
        for tii in prm.ts.time_keys
            # from
            state[:ac_qfr][tii][idx.ac_line_flows] = state[:acline_qfr][tii]
            state[:ac_qfr][tii][idx.ac_xfm_flows]  = state[:xfm_qfr][tii]
    
            # to 
            state[:ac_qto][tii][idx.ac_line_flows] = state[:acline_qto][tii]
            state[:ac_qto][tii][idx.ac_xfm_flows]  = state[:xfm_qto][tii]
        end
    end

    #tkeys[ii] => Dict(                      :u_on_dev  => collect(1:sys.ndev),
    #:p_rrd_off => collect(1:sys.ndev),
    #:p_nsc     => collect(1:sys.ndev),
    #:p_rru_off => collect(1:sys.ndev),
    #:q_qru     => collect(1:sys.ndev),
    #:q_qrd     => collect(1:sys.ndev),
    #:phi       => collect(1:sys.nx),
    #:tau       => collect(1:sys.nx),
    #:dc_pto    => collect(1:sys.nldc)))


    function clip_shunt!(state::Dict)
        for tii in prm[:time_series][:time_keys]
            # comute the deviation (0 if within bounds!)
            del = min.(state[:u_step_shunt][tii] - prm.shunt.step_lb, 0.0) + max.(state[:u_step_shunt][tii] - prm.shunt.step_ub, 0.0)
        
            # clip, and save the amount clipped
            state[:u_step_shunt][tii]                  = state[:u_step_shunt][tii] - del
            state[:u_step_shunt][tii][:amount_clipped] = del
        end
    end

    
function clip_reserve_shortfall!(state::Dict)
        for tii in prm[:time_series][:time_keys]
            del_p_rgu_zonal = min.(state[:p_rgu_zonal][tii], 0.0)
            del_p_rgd_zonal = min.(state[:p_rgd_zonal][tii], 0.0)
            del_p_scr_zonal = min.(state[:p_scr_zonal][tii], 0.0)
            del_p_nsc_zonal = min.(state[:p_nsc_zonal][tii], 0.0)
            del_p_rru_zonal = min.(state[:p_rru_zonal][tii], 0.0)
            del_p_rrd_zonal = min.(state[:p_rrd_zonal][tii], 0.0)
            del_q_qru_zonal = min.(state[:q_qru_zonal][tii], 0.0)
            del_q_qrd_zonal = min.(state[:q_qrd_zonal][tii], 0.0)
    
            # clip
            state[:p_rgu_zonal][tii] = state[:p_rgu_zonal][tii] - del_p_rgu_zonal
            state[:p_rgd_zonal][tii] = state[:p_rgd_zonal][tii] - del_p_rgd_zonal
            state[:p_scr_zonal][tii] = state[:p_scr_zonal][tii] - del_p_scr_zonal
            state[:p_nsc_zonal][tii] = state[:p_nsc_zonal][tii] - del_p_nsc_zonal
            state[:p_rru_zonal][tii] = state[:p_rru_zonal][tii] - del_p_rru_zonal
            state[:p_rrd_zonal][tii] = state[:p_rrd_zonal][tii] - del_p_rrd_zonal
            state[:q_qru_zonal][tii] = state[:q_qru_zonal][tii] - del_q_qru_zonal
            state[:q_qrd_zonal][tii] = state[:q_qrd_zonal][tii] - del_q_qrd_zonal
    
            # amount clipped, and save the amount clipped
            state[:p_rgu_zonal][tii][:amount_clipped] = del_p_rgu_zonal
            state[:p_rgd_zonal][tii][:amount_clipped] = del_p_rgd_zonal
            state[:p_scr_zonal][tii][:amount_clipped] = del_p_scr_zonal
            state[:p_nsc_zonal][tii][:amount_clipped] = del_p_nsc_zonal
            state[:p_rru_zonal][tii][:amount_clipped] = del_p_rru_zonal
            state[:p_rrd_zonal][tii][:amount_clipped] = del_p_rrd_zonal
            state[:q_qru_zonal][tii][:amount_clipped] = del_q_qru_zonal
            state[:q_qrd_zonal][tii][:amount_clipped] = del_q_qrd_zonal
        end
    end

    # ### DEPRECIATED ### #
function initialize_states_grad_jac(sys)
        # state
        state = Dict(
            :acline_pfr => Float64[],
            :acline_qfr => Float64[],
            :acline_sfr => Float64[],
            :acline_pto => Float64[],
            :acline_qto => Float64[],
            :acline_sto => Float64[],
            :xfm_pfr   => Float64[],
            :xfm_qfr   => Float64[],
            :xfm_sfr   => Float64[],
            :xfm_pto   => Float64[],
            :xfm_qto   => Float64[],
            :xfm_sto   => Float64[])
    
        # grd = grad
        grd = Dict(
            # aclines
            :acline_dpfr_dvmfr => Float64[],
            :acline_dpfr_dvmto => Float64[],
            :acline_dpfr_dvafr => Float64[],
            :acline_dpfr_dvato => Float64[],
            :acline_dpfr_duon  => Float64[],
            :acline_dqfr_dvmfr => Float64[],
            :acline_dqfr_dvmto => Float64[],
            :acline_dqfr_dvafr => Float64[],
            :acline_dqfr_dvato => Float64[],
            :acline_dqfr_duon  => Float64[],
            :acline_dsfr_dpfr  => Float64[],
            :acline_dsfr_dqfr  => Float64[],
            :acline_dpto_dvmfr => Float64[],
            :acline_dpto_dvmto => Float64[],
            :acline_dpto_dvafr => Float64[],
            :acline_dpto_dvato => Float64[],
            :acline_dpto_duon  => Float64[],
            :acline_dqto_dvmfr => Float64[],
            :acline_dqto_dvmto => Float64[],
            :acline_dqto_dvafr => Float64[],
            :acline_dqto_dvato => Float64[],
            :acline_dqto_duon  => Float64[],
            :acline_dsto_dpto  => Float64[],
            :acline_dsto_dqto  => Float64[],
            # xfms
            :xfm_dpfr_dvmfr => Float64[],
            :xfm_dpfr_dvmto => Float64[],
            :xfm_dpfr_dvafr => Float64[],
            :xfm_dpfr_dvato => Float64[],
            :xfm_dpfr_dtau  => Float64[],
            :xfm_dpfr_dphi  => Float64[],
            :xfm_dpfr_duon  => Float64[],
            :xfm_dqfr_dvmfr => Float64[],
            :xfm_dqfr_dvmto => Float64[],
            :xfm_dqfr_dvafr => Float64[],
            :xfm_dqfr_dvato => Float64[],
            :xfm_dqfr_dtau  => Float64[],
            :xfm_dqfr_dphi  => Float64[],
            :xfm_dqfr_duon  => Float64[],
            :xfm_dsfr_dpfr  => Float64[],
            :xfm_dsfr_dqfr  => Float64[],
            :xfm_dpto_dvmfr => Float64[],
            :xfm_dpto_dvmto => Float64[],
            :xfm_dpto_dvafr => Float64[],
            :xfm_dpto_dvato => Float64[],
            :xfm_dpto_dtau  => Float64[],
            :xfm_dpto_dphi  => Float64[],
            :xfm_dpto_duon  => Float64[],
            :xfm_dqto_dvmfr => Float64[],
            :xfm_dqto_dvmto => Float64[],
            :xfm_dqto_dvafr => Float64[],
            :xfm_dqto_dvato => Float64[],
            :xfm_dqto_dtau  => Float64[],
            :xfm_dqto_dphi  => Float64[],
            :xfm_dqto_duon  => Float64[],
            :xfm_dsto_dpto  => Float64[],
            :xfm_dsto_dqto  => Float64[])
    
        # jac
        jac = Dict(
            :acline => spzeros(4*sys.nl, sys.nvar),
            :xfm    => spzeros(4*sys.nx, sys.nvar));
    
            # output
            return state, grd, jac
    end

    # in this file, we prepare the hard device constraints, which we pass to Gurobi
#
# note -- this is always run after clipping
function solve_Gurobi_IBR!(prm::quasiGrad.Param, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, qG::quasiGrad.QG, sys::quasiGrad.System)
        # loop over each device and solve individually -- not clear if this is faster
        # than solving one big optimization problem all at once. see legacy code for
        # a (n unfinished) version where all devices are solved at once!
        for dev in 1:sys.ndev
        
        # Build model -- solve pr/cs devices at once
        model = Model(Gurobi.Optimizer)
        empty!(model)
    
        # Model Settings
        set_optimizer_attribute(model, "OutputFlag", 0)
    
        # define the minimum set of variables we will need to solve the constraints                                                                                     -- round() the int?
        u_on_dev  = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "u_on_dev_t$(ii)",  start=stt[:u_on_dev][tkeys[ii]], Bin) for ii in 1:(sys.nT))
        p_on      = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_on_t$(ii)",      start=stt[:p_on][tkeys[ii]])          for ii in 1:(sys.nT))
        dev_q     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "dev_q_t$(ii)",     start=stt[:dev_q][tkeys[ii]])         for ii in 1:(sys.nT))
        p_rgu     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rgu_t$(ii)",     start=stt[:p_rgu][tkeys[ii]])         for ii in 1:(sys.nT))
        p_rgd     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rgd_t$(ii)",     start=stt[:p_rgd][tkeys[ii]])         for ii in 1:(sys.nT))
        p_scr     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_scr_t$(ii)",     start=stt[:p_scr][tkeys[ii]])         for ii in 1:(sys.nT))
        p_nsc     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_nsc_t$(ii)",     start=stt[:p_nsc][tkeys[ii]])         for ii in 1:(sys.nT))
        p_rru_on  = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rru_on_t$(ii)",  start=stt[:p_rru_on][tkeys[ii]])      for ii in 1:(sys.nT))
        p_rru_off = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rru_off_t$(ii)", start=stt[:p_rru_off][tkeys[ii]])     for ii in 1:(sys.nT))
        p_rrd_on  = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rrd_on_t$(ii)",  start=stt[:p_rrd_on][tkeys[ii]])      for ii in 1:(sys.nT))
        p_rrd_off = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rrd_off_t$(ii)", start=stt[:p_rrd_off][tkeys[ii]])     for ii in 1:(sys.nT))
        q_qru     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "q_qru_t$(ii)",     start=stt[:q_qru][tkeys[ii]])         for ii in 1:(sys.nT))
        q_qrd     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "q_qrd_t$(ii)",     start=stt[:q_qrd][tkeys[ii]])         for ii in 1:(sys.nT))
    
        # add a few more (implicit) variables which are necessary for solving this system
        u_su_dev = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "u_su_dev_t$(ii)", Bin) for ii in 1:(sys.nT))
        u_sd_dev = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "u_sd_dev_t$(ii)", Bin) for ii in 1:(sys.nT))
        
        # we have the affine "AffExpr" expressions (whose values are specified)
        dev_p = Dict(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
        p_su  = Dict(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
        p_sd  = Dict(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
    
        # now, we need to loop and set the affine expressions to 0
        #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
        for tii in prm.ts.time_keys
            for dev in 1:sys.ndev
                dev_p[tii][dev] = AffExpr(0.0)
                p_su[tii][dev]  = AffExpr(0.0)
                p_sd[tii][dev]  = AffExpr(0.0)
            end
        end
    
        # == define active power constraints ==
        for tii in prm.ts.time_keys
            for dev in 1:sys.ndev
                # first, get the startup power
                T_set, p_supc_set = get_supc(tii, dev, prm)
                add_to_expression!(p_su[tii][dev], sum(p_supc_set[ii]*u_su_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_set); init=0.0))
    
                # second, get the shutdown power
                T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
                add_to_expression!(p_sd[tii][dev], sum(p_sdpc_set[ii]*u_sd_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_set); init=0.0))
            end
    
            # finally, get the total power balance
            dev_p[tii] = p_on[tii] + p_su[tii] + p_sd[tii]
        end
    
        # == define reactive power constraints ==
        for tii in prm.ts.time_keys
            for dev in 1:sys.ndev
                # only a subset of devices will have a reactive power equality constraint
                if dev in idx.J_pqe
    
                    # the following (pr vs cs) are equivalent
                    if dev in idx.pr_devs
                        # producer?
                        T_supc, ~ = get_supc(tii, dev, prm)
                        T_sdpc, ~ = get_sdpc(tii, dev, prm)
                        u_sum     = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)
                        
                        # compute q -- this might be the only equality constraint (and below)
                        @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
                    else
                        # the device must be a consumer :)
                        T_supc, ~ = get_supc(tii, dev, prm)
                        T_sdpc, ~ = get_sdpc(tii, dev, prm)
                        u_sum     = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)
    
                        # compute q -- this might be the only equality constraint (and above)
                        @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
                    end
                end
            end
        end
    
        # loop over each time period and define the hard constraints
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # duration
            dt = prm.ts.duration[tii]
    
            # for now, we use "del" in the scoring function to penalize all
            # constraint violations -- thus, don't call the "c_hat" constants
            for dev in 1:sys.ndev
                # in the following constraints, we need to sum over previous constraints
                # in time -- so, say t = 10, and d = 5, then we need to sum over all time
                # state between t < 10 and and t = 5.0
                #
                # so, we need a way to get the variables associated with previous time
                # instances. Thus, we write a function which takes a given time instance
                # and a time interval (e.g., min downtime) and returns the set of relevant
                # time instances: t_set = get_mintimes(tii,interval)
    
                # careful here -- if we're in, for example, the second "tii", then the
                # start time is :t1. 
    
                # 1. Minimum downtime: zhat_mndn
                t_set                        = get_tmindn(tii, dev, prm)
                @constraint(model, u_su_dev[tii][dev] + sum(u_sd_dev[tii_inst][dev] for tii_inst in t_set; init=0.0) - 1.0 <= 0)
    
                # 2. Minimum uptime: zhat_mnup
                t_set                        = get_tminup(tii, dev, prm)
                @constraint(model, u_sd_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in t_set; init=0.0) - 1.0 <= 0)
    
                # define the previous power value (used by both up and down ramping!)
                if tii == :t1
                    # note: p0 = prm.dev.init_p[dev]
                    dev_p_previous = prm.dev.init_p[dev]
                else
                    # grab previous time
                    tii_m1 = prm.ts.time_keys[t_ind-1]
                    dev_p_previous = dev_p[tii_m1][dev] 
                end
    
                # 3. Ramping limits (up): zhat_rup
                @constraint(model, dev_p[tii][dev] - dev_p_previous
                        - dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii][dev] - u_su_dev[tii][dev])
                        +     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii][dev] + 1.0 - u_on_dev[tii][dev])) <= 0)
    
                # 4. Ramping limits (down): zhat_rd
                @constraint(model,  dev_p_previous - dev_p[tii][dev]
                        - dt*(prm.dev.p_ramp_down_ub[dev]*u_on_dev[tii][dev]
                        +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-u_on_dev[tii][dev])) <= 0)
    
                # 5. Regulation up: zhat_rgu
                @constraint(model, p_rgu[tii][dev] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[tii][dev] <= 0)
    
                # 6. Regulation down: zhat_rgd
                @constraint(model, p_rgd[tii][dev] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[tii][dev] <= 0)
    
                # 7. Synchronized reserve: zhat_scr
                @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] - prm.dev.p_syn_res_ub[dev]*u_on_dev[tii][dev] <= 0)
    
                # 8. Synchronized reserve: zhat_nsc
                @constraint(model, p_nsc[tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[tii][dev]) <= 0)
    
                # 9. Ramping reserve up (on): zhat_rruon
                @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[tii][dev] <= 0)
    
                # 10. Ramping reserve up (off): zhat_rruoff
                @constraint(model, p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0-u_on_dev[tii][dev]) <= 0)
                
                # 11. Ramping reserve down (on): zhat_rrdon
                @constraint(model, p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[tii][dev] <= 0)
    
                # 12. Ramping reserve down (off): zhat_rrdoff
                @constraint(model, p_rrd_off[tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[tii][dev]) <= 0)
                
                # Now, we must separate: producers vs consumers
                if dev in idx.pr_devs
                    # 13p. Maximum reserve limits (producers): zhat_pmax
                    @constraint(model, p_on[tii][dev] + p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii][dev] <= 0)
                
                    # 14p. Minimum reserve limits (producers): zhat_pmin
                    @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii][dev] + p_rrd_on[tii][dev] + p_rgd[tii][dev] - p_on[tii][dev] <= 0)
                    
                    # 15p. Off reserve limits (producers): zhat_pmaxoff
                    @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii][dev]) <= 0)
    
                    # get common "u_sum" terms that will be used in the subsequent four equations 
                    T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum     = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)
    
                    # 16p. Maximum reactive power reserves (producers): zhat_qmax
                    @constraint(model, dev_q[tii][dev] + q_qru[tii][dev] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0)
    
                    # 17p. Minimum reactive power reserves (producers): zhat_qmin
                    @constraint(model, q_qrd[tii][dev] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii][dev] <= 0)
    
                    # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                    if dev in idx.J_pqmax
                        @constraint(model, dev_q[tii][dev] + q_qru[tii][dev] - prm.dev.q_0_ub[dev]*u_sum
                        - prm.dev.beta_ub[dev]*dev_p[tii][dev] <= 0)
                    end 
                    
                    # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                    if dev in idx.J_pqmin
                        @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                        + prm.dev.beta_lb[dev]*dev_p[tii][dev]
                        + q_qrd[tii][dev] - dev_q[tii][dev] <= 0)
                    end
    
                # consumers
                else  # => dev in idx.cs_devs
                    # 13c. Maximum reserve limits (consumers): zhat_pmax
                    @constraint(model, p_on[tii][dev] + p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii][dev] <= 0)
    
                    # 14c. Minimum reserve limits (consumers): zhat_pmin
                    @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii][dev] + p_rru_on[tii][dev] + p_scr[tii][dev] + p_rgu[tii][dev] - p_on[tii][dev] <= 0)
                    
                    # 15c. Off reserve limits (consumers): zhat_pmaxoff
                    @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_rrd_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii][dev]) <= 0)
    
                    # get common "u_sum" terms that will be used in the subsequent four equations 
                    T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum     = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)
    
                    # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                    @constraint(model, dev_q[tii][dev] + q_qrd[tii][dev] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0)
    
                    # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                    @constraint(model, q_qru[tii][dev] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii][dev] <= 0)
                    
                    # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                    if dev in idx.J_pqmax
                        @constraint(model, dev_q[tii][dev] + q_qrd[tii][dev] - prm.dev.q_0_ub[dev]*u_sum
                        - prm.dev.beta_ub[dev]*dev_p[tii][dev] <= 0)
                    end 
    
                    # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                    if dev in idx.J_pqmin
                        @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                        + prm.dev.beta_lb[dev]*dev_p[tii][dev]
                        + q_qru[tii][dev] - dev_q[tii][dev] <= 0)
                    end
                end
            end
        end
    
        # misc penalty: maximum starts over multiple periods
        for dev in 1:sys.ndev
            # now, loop over the startup constraints
            for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
                # get the time periods: zhat_mxst
                T_su_max = get_tsumax(w_params, prm)
                @constraint(model, sum(u_su_dev[tii][dev] for tii in T_su_max; init=0.0) - w_params[3] <= 0.0)
            end
        end
    
        # now, we need to add two other sorts of constraints:
        # 1. "evolutionary" constraints which link startup and shutdown variables
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            for dev in 1:sys.ndev
                if tii == :t1
                    @constraint(model, u_on_dev[tii][dev] - prm.dev.init_on_status[dev] == u_su_dev[tii][dev] - u_sd_dev[tii][dev])
                else
                    tii_m1 = prm.ts.time_keys[t_ind-1]
                    @constraint(model, u_on_dev[tii][dev] - u_on_dev[tii_m1][dev] == u_su_dev[tii][dev] - u_sd_dev[tii][dev])
                end
                # only one can be nonzero
                @constraint(model, u_su_dev[tii][dev] + u_sd_dev[tii][dev] <= 1)
            end
        end
    
        # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "vst"
        # vst = variable states
        #
        # note -- in this loop, we also build the objective function!
         # now, let's define an objective function and solve this mf.
        # our overall objective is to round and fix some subset of 
        # integer variables. Here is our approach: find a feasible
        # solution which is as close to our Adam solution as possible.
        # next, we process the results: we identify the x% of variables
        # which had to move "the least". We fix these values and remove
        # their associated indices from vst. the end.
        #
        # afterwards, we initialize adam with the closest feasible
        # solution variable values.
        obj = AffExpr(0.0)
    #add_to_expression!(ex, 2.0, x)
    
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            for dev in 1:sys.ndev
                # if a device is *not* in the set of variables,
                # then it must be held constant!
                if dev ∉ vst[:u_on_dev][tii]
                    @constraint(model, u_on_dev[tii][dev] == stt[:u_on_dev][tii][dev])
                end
                if dev ∉ vst[:p_rrd_off][tii][dev]
                    @constraint(model, p_rrd_off[tii][dev] == stt[:p_rrd_off][tii][dev])
                end
                if dev ∉ vst[:p_nsc][tii][dev]
                    @constraint(model, p_nsc[tii][dev] == stt[:p_nsc][tii][dev])
                end
                if dev ∉ vst[:p_rru_off][tii][dev]
                    @constraint(model, p_rru_off[tii][dev] == stt[:p_rru_off][tii][dev])
                end
                if dev ∉ vst[:q_qru][tii][dev]
                    @constraint(model, q_qru[tii][dev] == stt[:q_qru][tii][dev])
                end
                if dev ∉ vst[:q_qrd][tii][dev]
                    @constraint(model, q_qrd[tii][dev] == stt[:q_qrd][tii][dev])
                end
            end
        end
    
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            for dev in 1:sys.ndev
                # define an anonymous variable which is used to
                # bound objective function error
                t = @variable(model)
                u_on_dev[tii][dev]  - stt[:u_on_dev][tii]
    
    
                p_on[tii][dev]      - stt[:p_on][tii]
                dev_q[tii][dev]     - stt[:dev_q][tii]
                p_rgu[tii][dev]     - stt[:p_rgu][tii]    
                p_rgd[tii][dev]     - stt[:p_rgd][tii]
                p_scr[tii][dev]     - stt[:p_scr][tii]
                p_nsc[tii][dev]     - stt[:p_nsc][tii]
                p_rru_on[tii][dev]  - stt[:p_rru_on][tii]
                p_rru_off[tii][dev] - stt[:p_rru_off][tii]
                p_rrd_on[tii][dev]  - stt[:p_rrd_on][tii]
                p_rrd_off[tii][dev] - stt[:p_rrd_off][tii]
                q_qru[tii][dev]     - stt[:q_qru][tii]
                q_qrd[tii][dev]     - stt[:q_qrd][tii]
            end
        end
    
    
        u_on_dev  = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "u_on_dev_t$(ii)",  start=round(stt[:u_on_dev][tkeys[ii]]), Bin) for ii in 1:(sys.nT))
        p_on      = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_on_t$(ii)",      start=stt[:p_on][tkeys[ii]]) for ii in 1:(sys.nT))
        dev_q     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "dev_q_t$(ii)",     start=stt[:dev_q][tkeys[ii]])         for ii in 1:(sys.nT))
        p_rgu     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rgu_t$(ii)",     start=stt[:p_rgu][tkeys[ii]])         for ii in 1:(sys.nT))
        p_rgd     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rgd_t$(ii)",     start=stt[:p_rgd][tkeys[ii]])         for ii in 1:(sys.nT))
        p_scr     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_scr_t$(ii)",     start=stt[:p_scr][tkeys[ii]])         for ii in 1:(sys.nT))
        p_nsc     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_nsc_t$(ii)",     start=stt[:p_nsc][tkeys[ii]])         for ii in 1:(sys.nT))
        p_rru_on  = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rru_on_t$(ii)",  start=stt[:p_rru_on][tkeys[ii]])      for ii in 1:(sys.nT))
        p_rru_off = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rru_off_t$(ii)", start=stt[:p_rru_off][tkeys[ii]])     for ii in 1:(sys.nT))
        p_rrd_on  = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rrd_on_t$(ii)",  start=stt[:p_rrd_on][tkeys[ii]])      for ii in 1:(sys.nT))
        p_rrd_off = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "p_rrd_off_t$(ii)", start=stt[:p_rrd_off][tkeys[ii]])     for ii in 1:(sys.nT))
        q_qru     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "q_qru_t$(ii)",     start=stt[:q_qru][tkeys[ii]])         for ii in 1:(sys.nT))
        q_qrd     = Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "q_qrd_t$(ii)",     start=stt[:q_qrd][tkeys[ii]])         for ii in 1:(sys.nT))
    
    
    
    
        @objective(model, Max, obj)
    end

                # local reserve penalties (consumers)
                grd[:zt][:zrgu_cs][tii] = -ones(sys.npr)
                grd[:zt][:zrgd_cs][tii] = -ones(sys.npr)
                grd[:zt][:zscr_cs][tii] = -ones(sys.npr)
                grd[:zt][:znsc_cs][tii] = -ones(sys.npr)
                grd[:zt][:zrru_cs][tii] = -ones(sys.npr)
                grd[:zt][:zrrd_cs][tii] = -ones(sys.npr)
                grd[:zt][:zqru_cs][tii] = -ones(sys.npr)
                grd[:zt][:zqrd_cs][tii] = -ones(sys.npr)

        # consumer injections -- q_on
        mgd[:p_on][tii][idx.cs[bus]] += mgd_com*
        grd[:pb_slack][bus][:dev_p][tii][idx.cs[bus]].*
        grd[:dev_p][:p_on][tii][idx.cs[bus]]

        # producer injections -- p_on
        mgd[:p_on][tii][idx.pr[bus]] += mgd_com*
        grd[:pb_slack][bus][:dev_p][tii][idx.pr[bus]].*
        grd[:dev_p][:p_on][tii][idx.pr[bus]]

        # consumer injections -- u_on_dev
        for dev in idx.cs[bus]
            # consumer: startup -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev][tii][:t_now]
            # consumer: startup -- previous u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev][tii][:t_prev]

            # consumer: shutdown -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev][tii][:t_now]
            # consumer: shutdown -- previous u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev][tii][:t_prev]
        end

        # producer injections -- u_on_dev
        for dev in idx.pr[bus]

            # producer: startup -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev][tii][:t_now]
            # producer: startup -- previous u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev][tii][:t_prev]

            # producer: shutdown -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev][tii][:t_now]
            # producer: shutdown -- previous u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev][tii][:t_prev]
        end

        # -------------------

#=
        # consumer injections -- p_on
        mgd[:p_on][tii][idx.cs[bus]] += mgd_com*
        grd[:pb_slack][bus][:dev_p][tii][idx.cs[bus]].*
        grd[:dev_p][:p_on][tii][idx.cs[bus]]

        # producer injections -- p_on
        mgd[:p_on][tii][idx.pr[bus]] += mgd_com*
        grd[:pb_slack][bus][:dev_p][tii][idx.pr[bus]].*
        grd[:dev_p][:p_on][tii][idx.pr[bus]]

        # consumer injections -- u_on_dev
        for dev in idx.cs[bus]
            # consumer: startup -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev][tii][dev]
            # consumer: startup -- previous u_on
            mgd[:u_on_dev][tii_m1][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev_prev][tii][dev]

            # consumer: shutdown -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev][tii][dev]
            # consumer: shutdown -- previous u_on
            mgd[:u_on_dev][tii_m1][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev_prev][tii][dev]
        end

        # producer injections -- u_on_dev
        for dev in idx.pr[bus]

            # producer: startup -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev][tii]
            # producer: startup -- previous u_on
            mgd[:u_on_dev][tii_m1][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_su][tii][dev].*
            grd[:p_su][:u_su_dev][tii][dev].*
            grd[:u_su_dev][:u_on_dev_prev][tii]

            # producer: shutdown -- current u_on
            mgd[:u_on_dev][tii][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev][tii]
            # producer: shutdown -- previous u_on
            mgd[:u_on_dev][tii_m1][dev] += mgd_com*
            grd[:pb_slack][bus][:dev_p][tii][dev].*
            grd[:dev_p][:p_sd][tii][dev].*
            grd[:p_sd][:u_sd_dev][tii][dev].*
            grd[:u_sd_dev][:u_on_dev_prev][tii][dev]
        end
        =#

                        # now, what we really want is the gradients accociated with a single
                # given variable -- place them into a sparse Jacobian matrix
                #=
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmfr]))] = grd[:acline_dpfr_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vmto]))] = grd[:acline_dpfr_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vafr]))] = grd[:acline_dpfr_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pfr],idx[:varstack][:t1][:acline_vato]))] = grd[:acline_dpfr_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmfr]))] = grd[:acline_dqfr_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vmto]))] = grd[:acline_dqfr_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vafr]))] = grd[:acline_dqfr_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qfr],idx[:varstack][:t1][:acline_vato]))] = grd[:acline_dqfr_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmfr]))] = grd[:acline_dpto_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vmto]))] = grd[:acline_dpto_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vafr]))] = grd[:acline_dpto_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_pto],idx[:varstack][:t1][:acline_vato]))] = grd[:acline_dpto_dvato]

                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmfr]))] = grd[:acline_dqto_dvmfr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vmto]))] = grd[:acline_dqto_dvmto]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vafr]))] = grd[:acline_dqto_dvafr]
                jac[:acline][CartesianIndex.(tuple.(idx[:jac][:acline_qto],idx[:varstack][:t1][:acline_vato]))] = grd[:acline_dqto_dvato]
                =#

                # now, what we really want is the gradients accociated with a single
                # given variable -- place them into a sparse Jacobian matrix
                #=
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vmfr]))] = grd[:xfm_dpfr_dvmfr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vmto]))] = grd[:xfm_dpfr_dvmto]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vafr]))] = grd[:xfm_dpfr_dvafr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_vato]))] = grd[:xfm_dpfr_dvato]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_tau]))]  = grd[:xfm_dpfr_dtau]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pfr],idx[:varstack][:t1][:xfm_phi]))]  = grd[:xfm_dpfr_dphi]

                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vmfr]))] = grd[:xfm_dqfr_dvmfr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vmto]))] = grd[:xfm_dqfr_dvmto]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vafr]))] = grd[:xfm_dqfr_dvafr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_vato]))] = grd[:xfm_dqfr_dvato]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_tau]))]  = grd[:xfm_dqfr_dtau]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qfr],idx[:varstack][:t1][:xfm_phi]))]  = grd[:xfm_dqfr_dphi]

                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vmfr]))] = grd[:xfm_dpto_dvmfr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vmto]))] = grd[:xfm_dpto_dvmto]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vafr]))] = grd[:xfm_dpto_dvafr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_vato]))] = grd[:xfm_dpto_dvato]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_tau]))]  = grd[:xfm_dpto_dtau]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_pto],idx[:varstack][:t1][:xfm_phi]))]  = grd[:xfm_dpto_dphi]

                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vmfr]))] = grd[:xfm_dqto_dvmfr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vmto]))] = grd[:xfm_dqto_dvmto]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vafr]))] = grd[:xfm_dqto_dvafr]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_vato]))] = grd[:xfm_dqto_dvato]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_tau]))]  = grd[:xfm_dpto_dtau]
                jac[:xfm][CartesianIndex.(tuple.(idx[:jac][:xfm_qto],idx[:varstack][:t1][:xfm_phi]))]  = grd[:xfm_dpto_dphi]
                =#
# evaluate the grd? do this in the time and zone loops
if qG.eval_grad
        # 7
        if stt[:q_qru_zonal_penalty][tii][zone] == 0.0
            grd[:q_qru_zonal_penalty][zone][:q_qru][tii][idx.dev_qzone[zone]] .= 0.0
        else
            grd[:q_qru_zonal_penalty][zone][:q_qru][tii][idx.dev_qzone[zone]] .= -1.0
        end
        # 8
        if stt[:q_qrd_zonal_penalty][tii][zone] == 0.0
            grd[:q_qrd_zonal_penalty][zone][:q_qrd][tii][idx.dev_qzone[zone]] .= 0.0
        else
            grd[:q_qrd_zonal_penalty][zone][:q_qrd][tii][idx.dev_qzone[zone]] .= -1.0
        end
    end

    # evaluate the grd? do this in the time and zone loops
    if qG.eval_grad
        # endogenous sum
        if idx.cs_pzone[zone] == []
            # in the case there are NO consumers in a zone
            grd[:p_rgu_zonal_REQ][zone][:dev_p][tii][idx.cs_pzone[zone]] .= 0.0
            grd[:p_rgd_zonal_REQ][zone][:dev_p][tii][idx.cs_pzone[zone]] .= 0.0
        else
            grd[:p_rgu_zonal_REQ][zone][:dev_p][tii][idx.cs_pzone[zone]] .= rgu_sigma[zone]
            grd[:p_rgd_zonal_REQ][zone][:dev_p][tii][idx.cs_pzone[zone]] .= rgd_sigma[zone]
        end

        # endogenous max
        if idx.pr_pzone[zone] == []
            # in the case there are NO producers in a zone
            grd[:p_scr_zonal_REQ][zone][:dev_p][tii][i_max] .= 0.0
            grd[:p_nsc_zonal_REQ][zone][:dev_p][tii][i_max] .= 0.0
        else
            i_max = idx.pr_pzone[zone][argmax([stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]])]
            grd[:p_scr_zonal_REQ][zone][:dev_p][tii][i_max] = scr_sigma[zone]
            grd[:p_nsc_zonal_REQ][zone][:dev_p][tii][i_max] = nsc_sigma[zone]
        end

        # balance equations -- compute the shortfall value gradients
        #
        # 1
        if stt[:p_rgu_zonal_penalty][tii][zone] == 0.0
            grd[:p_rgu_zonal_penalty][zone][:p_rgu_zonal_REQ][tii]              .= 0.0
            grd[:p_rgu_zonal_penalty][zone][:p_rgu][tii][idx.dev_pzone[zone]] .= 0.0
        else
            grd[:p_rgu_zonal_penalty][zone][:p_rgu_zonal_REQ][tii]               .= 1.0
            grd[:p_rgu_zonal_penalty][zone][:p_rgu][tii][idx.dev_pzone[zone]] .= -1.0
        end
        # 2
        if stt[:p_rgd_zonal_penalty][tii][zone] == 0.0
            grd[:p_rgd_zonal_penalty][zone][:p_rgd_zonal_REQ][tii]               .= 0.0
            grd[:p_rgd_zonal_penalty][zone][:p_rgd][tii][idx.dev_pzone[zone]] .= 0.0
        else
            grd[:p_rgd_zonal_penalty][zone][:p_rgd_zonal_REQ][tii]               .= 1.0
            grd[:p_rgd_zonal_penalty][zone][:p_rgd][tii][idx.dev_pzone[zone]] .= -1.0
        end
        # 3
        if stt[:p_scr_zonal_penalty][tii][zone] == 0.0
            grd[:p_scr_zonal_penalty][zone][:p_rgu_zonal_REQ][tii] .= 0.0
            grd[:p_scr_zonal_penalty][zone][:p_scr_zonal_REQ][tii] .= 0.0
            grd[:p_scr_zonal_penalty][zone][:p_rgu][tii][idx.dev_pzone[zone]] .= 0.0
            grd[:p_scr_zonal_penalty][zone][:p_scr][tii][idx.dev_pzone[zone]] .= 0.0
        else
            grd[:p_scr_zonal_penalty][zone][:p_rgu_zonal_REQ][tii] .= 1.0
            grd[:p_scr_zonal_penalty][zone][:p_scr_zonal_REQ][tii] .= 1.0
            grd[:p_scr_zonal_penalty][zone][:p_rgu][tii][idx.dev_pzone[zone]] .= -1.0
            grd[:p_scr_zonal_penalty][zone][:p_scr][tii][idx.dev_pzone[zone]] .= -1.0
        end
        # 4
        if stt[:p_nsc_zonal_penalty][tii][zone] == 0.0
            grd[:p_nsc_zonal_penalty][zone][:p_rgu_zonal_REQ][tii] .= 0.0
            grd[:p_nsc_zonal_penalty][zone][:p_scr_zonal_REQ][tii] .= 0.0
            grd[:p_nsc_zonal_penalty][zone][:p_nsc_zonal_REQ][tii] .= 0.0
            grd[:p_nsc_zonal_penalty][zone][:p_rgu][idx.dev_pzone[zone]] .= 0.0
            grd[:p_nsc_zonal_penalty][zone][:p_scr][idx.dev_pzone[zone]] .= 0.0
            grd[:p_nsc_zonal_penalty][zone][:p_nsc][idx.dev_pzone[zone]] .= 0.0
        else
            grd[:p_nsc_zonal_penalty][zone][:p_rgu_zonal_REQ][tii] .= 1.0
            grd[:p_nsc_zonal_penalty][zone][:p_scr_zonal_REQ][tii] .= 1.0
            grd[:p_nsc_zonal_penalty][zone][:p_nsc_zonal_REQ][tii] .= 1.0
            grd[:p_nsc_zonal_penalty][zone][:p_rgu][idx.dev_pzone[zone]] .= -1.0
            grd[:p_nsc_zonal_penalty][zone][:p_scr][idx.dev_pzone[zone]] .= -1.0
            grd[:p_nsc_zonal_penalty][zone][:p_nsc][idx.dev_pzone[zone]] .= -1.0
        end
        # 5
        if stt[:p_rru_zonal_penalty][tii][zone] == 0.0
            grd[:p_rru_zonal_penalty][zone][:p_rru_on][idx.dev_pzone[zone]]  = 0.0
            grd[:p_rru_zonal_penalty][zone][:p_rru_off][idx.dev_pzone[zone]] = 0.0
        else
            grd[:p_rru_zonal_penalty][zone][:p_rru_on][idx.dev_pzone[zone]]  = -1.0
            grd[:p_rru_zonal_penalty][zone][:p_rru_off][idx.dev_pzone[zone]] = -1.0
        end
        # 6
        if stt[:p_rrd_zonal_penalty][tii][zone] == 0.0
            grd[:p_rrd_zonal_penalty][zone][:p_rrd_on][idx.dev_pzone[zone]]  = 0.0
            grd[:p_rrd_zonal_penalty][zone][:p_rrd_off][idx.dev_pzone[zone]] = 0.0
        else
            grd[:p_rrd_zonal_penalty][zone][:p_rrd_on][idx.dev_pzone[zone]]  = -1.0
            grd[:p_rrd_zonal_penalty][zone][:p_rrd_off][idx.dev_pzone[zone]] = -1.0
        end
    end

    grd[:ep_max][dev][w_ind] = sign(stt[:ep_max][dev][w_ind])


    alpha = sign(stt[:ep_max][dev][w_ind])
    for tii in T_en_max
        ddev_p!(tii, prm, stt, grd, mgd, dev, alpha)

        grd[:zbase][:z_enmax]   = 1.0
        grd[:zbase][:z_enmin]   = 1.0

        grd[:nzms][:zbase]*

        stt[:ep_max][dev][w_ind]


        # mgd[:u_on_dev][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsus_dev][tii] *
    # grd[:zon_dev][:u_on_dev][tii]

        grd[:ep_max][:dev_p][dev][w_ind] += alpha*prm.ts.duration[tii]

    # min and max energy penalties
    grd[:zw_enmax][:ep_max] = prm.vio.e_dev
    grd[:zw_enmin][:ep_min] = prm.vio.e_dev

    :ep_max    => Dict(ii => zeros(prm.dev.num_W_enmax[ii]) for ii in 1:(sys.ndev)), 
    :ep_min    => Dict(ii => zeros(prm.dev.num_W_enmin[ii]) for ii in 1:(sys.ndev)),

        # power balance penalty p+/q+
        cp = prm.vio.p_bus
        cq = prm.vio.q_bus
        for tii in prm.ts.time_keys
            dt = prm.ts.duration[tii]
            grd[:zp][:pb_penalty][tii] = cp*dt
            grd[:zq][:qb_penalty][tii] = cp*dt
        end

if tii == t1
    # devices
    mgd[:u_on_dev][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_dev] *
    grd[:zsu_dev][:u_su_dev] .* grd[:u_su_dev][:u_on_dev][tii]

    # acline 
    mgd[:u_on_acline][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_acline] *
    grd[:zsu_acline][:u_su_acline] .* grd[:u_su_acline][:u_on_acline][tii]

    # xfm
    mgd[:u_on_xfm][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_xfm] *
    grd[:zsu_xfm][:u_su_xfm] .* grd[:u_su_xfm][:u_on_xfm][tii]
else
    # current time and previous time
    #
    # devices
    gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_dev] * grd[:zsu_dev][:u_su_dev]
    mgd[:u_on_dev][tii]                   += gc .* grd[:u_su_dev][:u_on_dev][tii]
    mgd[:u_on_dev][prm.ts.tmin1[tii]] += gc .* grd[:u_su_dev][:u_on_dev_prev][tii]

    # acline
    gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_acline] * grd[:zsu_acline][:u_su_acline]
    mgd[:u_on_acline][tii]                   += gc .* grd[:u_su_acline][:u_on_acline][tii]
    mgd[:u_on_acline][prm.ts.tmin1[tii]] += gc .* grd[:u_su_acline][:u_on_acline_prev][tii]

    # xfm
    gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsu_xfm] * grd[:zsu_xfm][:u_su_xfm]
    mgd[:u_on_xfm][tii]                   += gc .* grd[:u_su_xfm][:u_on_xfm][tii]
    mgd[:u_on_xfm][prm.ts.tmin1[tii]] += gc .* grd[:u_su_xfm][:u_on_xfm_prev][tii]
end

if tii == :t1
            # devices
            mgd[:u_on_dev][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_dev] *
            grd[:zsd_dev][:u_sd_dev] .* grd[:u_sd_dev][:u_on_dev][tii]

            # acline
            mgd[:u_on_acline][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_acline] *
            grd[:zsd_acline][:u_sd_acline] .* grd[:u_sd_acline][:u_on_acline][tii]

            # xfm
            mgd[:u_on_xfm][tii] += grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_xfm] *
            grd[:zsd_xfm][:u_sd_xfm] .* grd[:u_sd_xfm][:u_on_xfm][tii]
        else
            # current time and previous time
            #
            # devices
            gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_dev] * grd[:zsd_dev][:u_sd_dev]
            mgd[:u_on_dev][tii]                   += gc .* grd[:u_sd_dev][:u_on_dev][tii]
            mgd[:u_on_dev][prm.ts.tmin1[tii]] += gc .* grd[:u_sd_dev][:u_on_dev_prev][tii]

            # acline
            gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_acline] * grd[:zsd_acline][:u_sd_acline]
            mgd[:u_on_acline][tii]                   += gc .* grd[:u_sd_acline][:u_on_acline][tii]
            mgd[:u_on_acline][prm.ts.tmin1[tii]] += gc .* grd[:u_sd_acline][:u_on_acline_prev][tii]

            # xfm
            gc = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zsd_xfm] * grd[:zsd_xfm][:u_sd_xfm]
            mgd[:u_on_xfm][tii]                   += gc .* grd[:u_sd_xfm][:u_on_xfm][tii]
            mgd[:u_on_xfm][prm.ts.tmin1[tii]] += gc .* grd[:u_sd_xfm][:u_on_xfm_prev][tii]
        end
#=

    # u_on_dev -- startup
    mgd[:u_on_dev][tii][devs] = alphas.*grd[:dev_p][:p_su].*
    grd[:p_su][:u_su_dev][tii][devs].*
    grd[:u_su_dev][:u_on_dev][tii][devs]

    # u_on_dev -- shutdown
    mgd[:u_on_dev][tii][devs] = alphas.*grd[:dev_p][:p_sd].*
    grd[:p_sd][:u_sd_dev][tii][devs].*
    grd[:u_sd_dev][:u_on_dev][tii][desv]

    # if we are not at the first time interval, we need the derivative
    # with respect to the previous on/off state
    if tii != :t1
        # u_on_dev (previous time) -- startup
        mgd[:u_on_dev][prm.ts.tmin1[tii]][devs] = alphas.*grd[:dev_p][:p_su].*
        grd[:p_su][:u_su_dev][tii][devs].*
        grd[:u_su_dev][:u_on_dev_prev][tii][devs]

        # u_on_dev (previous time) -- shutdown
        mgd[:u_on_dev][prm.ts.tmin1[tii]][devs] = alphas.*grd[:dev_p][:p_sd].*
        grd[:p_sd][:u_sd_dev][tii][devs].*
        grd[:u_sd_dev][:u_on_dev_prev][tii][devs]
    end
    =#

    

for dev in idx.cs[bus]
    if dev in idx.J_pqe
        # in this case, we take the derivatives of "dev_q" wrt
        #   a) u_on_dev (a la u_sum)
        alpha = mgd_com*grd[:dev_q][:u_sum][dev] # grd[:qb_slack][:dev_q_cs] = +1
        T_supc, ~ = quasiGrad.get_supc(tii, dev, prm)
        T_sdpc, ~ = quasiGrad.get_sdpc(tii, dev, prm)
        du_sum!(tii, prm, stt, mgd, dev, alpha, T_supc, T_sdpc)

        #   b) dev_p (and its fellows)
        alpha = mgd_com*grd[:dev_q][:dev_p][dev] # grd[:qb_slack][:dev_q_cs] = +1
        ddev_p!(tii, prm, stt, grd, mgd, dev, alpha)
    else
        mgd[:dev_q][tii][dev] += mgd_com # grd[:qb_slack][:dev_q_cs] = +1
    end
end

# producers
for dev in idx.pr[bus]
    if dev in idx.J_pqe
        # in this case, we take the derivatives of "dev_q" wrt
        #   a) u_on_dev (a la u_sum)
        alpha = -mgd_com*grd[:dev_q][:u_sum][dev] # grd[:qb_slack][:dev_q_pr] = -1
        T_supc, ~ = quasiGrad.get_supc(tii, dev, prm)
        T_sdpc, ~ = quasiGrad.get_sdpc(tii, dev, prm)
        du_sum!(tii, prm, stt, mgd, dev, alpha, T_supc, T_sdpc)

        #   b) dev_p (and its fellows)
        alpha = -mgd_com*grd[:dev_q][:dev_p][tii][dev] # grd[:qb_slack][:dev_q_pr] = -1
        ddev_p!(tii, prm, stt, grd, mgd, dev, alpha)
    else
        mgd[:dev_q][tii][dev] += -mgd_com # grd[:qb_slack][:dev_q_pr] = -1
    end
end

    # consumers  
    for dev in idx.cs[bus]
        if dev in idx.J_pqe
            # in this case, we take the derivatives of "dev_q" wrt
            #   a) u_on_dev (a la u_sum)
            alpha = mgd_com*grd[:dev_q][:u_sum][dev] # grd[:qb_slack][:dev_q_cs] = +1
            T_supc, ~ = quasiGrad.get_supc(tii, dev, prm)
            T_sdpc, ~ = quasiGrad.get_sdpc(tii, dev, prm)
            du_sum!(tii, prm, stt, mgd, dev, alpha, T_supc, T_sdpc)

            #   b) dev_p (and its fellows)
            alpha = mgd_com*grd[:dev_q][:dev_p][dev] # grd[:qb_slack][:dev_q_cs] = +1
            ddev_p!(tii, prm, stt, grd, mgd, dev, alpha)
        else
            mgd[:dev_q][tii][dev] += mgd_com # grd[:qb_slack][:dev_q_cs] = +1
        end
    end

    # producers
    for dev in idx.pr[bus]
        if dev in idx.J_pqe
            # in this case, we take the derivatives of "dev_q" wrt
            #   a) u_on_dev (a la u_sum)
            alpha = -mgd_com*grd[:dev_q][:u_sum][dev] # grd[:qb_slack][:dev_q_pr] = -1
            T_supc, ~ = quasiGrad.get_supc(tii, dev, prm)
            T_sdpc, ~ = quasiGrad.get_sdpc(tii, dev, prm)
            du_sum!(tii, prm, stt, mgd, dev, alpha, T_supc, T_sdpc)

            #   b) dev_p (and its fellows)
            alpha = -mgd_com*grd[:dev_q][:dev_p][tii][dev] # grd[:qb_slack][:dev_q_pr] = -1
            ddev_p!(tii, prm, stt, grd, mgd, dev, alpha)
        else
            mgd[:dev_q][tii][dev] += -mgd_com # grd[:qb_slack][:dev_q_pr] = -1
        end
    end

    
    for ii in 1:(sys.nT)
        tii = Symbol("t"*string(ii))
        varstack[tii] = Dict(
            # locations in the varstack
            :vm            => 0*sys.nb + 0*sys.nx + 0*sys.nl .+ bus_enumerate,
            :va            => 1*sys.nb + 0*sys.nx + 0*sys.nl .+ bus_enumerate,
            :tau           => 2*sys.nb + 0*sys.nx + 0*sys.nl .+ xfm_enumerate,
            :phi           => 2*sys.nb + 1*sys.nx + 0*sys.nl .+ xfm_enumerate,
            :u_on_acline   => 2*sys.nb + 2*sys.nx + 0*sys.nl .+ acline_enumerate,
            :u_on_xfm      => 2*sys.nb + 2*sys.nb + 1*sys.nl .+ xfm_enumerate,
            # magnitude variables
            :acline_vmfr   => (ii-1)*num_var + 0*sys.nb .+ prm[:acline][:fr_bus_num],
            :acline_vmto   => (ii-1)*num_var + 0*sys.nb .+ prm[:acline][:to_bus_num],
            :xfm_vmfr      => (ii-1)*num_var + 0*sys.nb .+ prm[:xfm][:fr_bus_num],
            :xfm_vmto      => (ii-1)*num_var + 0*sys.nb .+ prm[:xfm][:to_bus_num],
            # angle variables
            :acline_vafr   => (ii-1)*num_var + 1*sys.nb .+ prm[:acline][:fr_bus_num],
            :acline_vato   => (ii-1)*num_var + 1*sys.nb .+ prm[:acline][:to_bus_num],
            :xfm_vafr      => (ii-1)*num_var + 1*sys.nb .+ prm[:xfm][:fr_bus_num],
            :xfm_vato      => (ii-1)*num_var + 1*sys.nb .+ prm[:xfm][:to_bus_num],
            # tap changing variables
            :xfm_tau       => (ii-1)*num_var + 2*sys.nb + 0*sys.nx .+ xfm_enumerate,
            # phase shifting variables
            :xfm_phi       => (ii-1)*num_var + 2*sys.nb + 1*sys.nx .+ xfm_enumerate,
            # lines status
            :acline_status => (ii-1)*num_var + 2*sys.nb + 2*sys.nx + 0*sys.nl .+ acline_enumerate,
            # xfm status
            :xfm_status    => (ii-1)*num_var + 2*sys.nb + 2*sys.nx + 1*sys.nl .+ xfm_enumerate)
    end

        # assume single time period (for now) -- get the indices associated with jacobian functions
        jac = Dict(
            :acline_pfr => 0*sys.nl .+ acline_enumerate,
            :acline_qfr => 1*sys.nl .+ acline_enumerate,
            :acline_pto => 2*sys.nl .+ acline_enumerate,
            :acline_qto => 3*sys.nl .+ acline_enumerate,
            # xfm flows
            :xfm_pfr => 0*sys.nx .+ xfm_enumerate, # + 4*sys.nl --- needed IF xfm jac and acline jac are concatenated
            :xfm_qfr => 1*sys.nx .+ xfm_enumerate, # + 4*sys.nl
            :xfm_pto => 2*sys.nx .+ xfm_enumerate, # + 4*sys.nl
            :xfm_qto => 3*sys.nx .+ xfm_enumerate) # + 4*sys.nl

            # variables stack (for now): vm -> va -> tau -> phi -> uon
    #   -> this is defined for EACH time period!!!
    #
    # create a varstack idx dictionary, which holds the key for mapping 
    # from a time period + varible vector to the state
    bus_enumerate    = collect(1:sys.nb)
    acline_enumerate = collect(1:sys.nl)
    xfm_enumerate    = collect(1:sys.nx)

    :acline_sfr => Dict(:pfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                 :qfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
    :acline_sto => Dict(:pto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                      :qto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),

                      :xfm_sfr => Dict(:pfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                      :qfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
     :xfm_sto => Dict(:pto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                      :qto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)))
                              #:zp        => Dict(:pb_penalty => Dict(tkeys[ii] => 0.0 for ii in 1:(sys.nT))),
        #:zq        => Dict(:qb_penalty => Dict(tkeys[ii] => 0.0 for ii in 1:(sys.nT))),

                    #:p_rgu_zonal_REQ => Dict(jj => Dict(:cs_p => Dict(tkeys[ii] => zeros(sys.ncs) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
            #:p_rgd_zonal_REQ => Dict(jj => Dict(:cs_p => Dict(tkeys[ii] => zeros(sys.ncs) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
            #:p_scr_zonal_REQ => Dict(jj => Dict(:pr_p => Dict(tkeys[ii] => zeros(sys.npr) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
            #:p_nsc_zonal_REQ => Dict(jj => Dict(:pr_p => Dict(tkeys[ii] => zeros(sys.npr) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
            # evaluate the grd?
            if qG.eval_grad
                # pb_slack
                grd[:pb_slack][bus][:dev_p][tii][idx.cs[bus]]                     .= 1.0
                grd[:pb_slack][bus][:dev_p][tii][idx.pr[bus]]                     .= -1.0
                grd[:pb_slack][bus][:sh_p][tii][idx.sh[bus]]                      .= 1.0 
                grd[:pb_slack][bus][:acline_pfr][tii][idx.bus_is_acline_frs[bus]] .= 1.0
                grd[:pb_slack][bus][:acline_pto][tii][idx.bus_is_acline_tos[bus]] .= 1.0
                grd[:pb_slack][bus][:xfm_pfr][tii][idx.bus_is_xfm_frs[bus]]       .= 1.0
                grd[:pb_slack][bus][:xfm_pto][tii][idx.bus_is_xfm_tos[bus]]       .= 1.0
                grd[:pb_slack][bus][:dc_pfr][tii][idx.bus_is_dc_frs[bus]]         .= 1.0
                grd[:pb_slack][bus][:dc_pto][tii][idx.bus_is_dc_tos[bus]]         .= 1.0

                # pb_slack
                grd[:qb_slack][bus][:dev_q][tii][idx.cs[bus]]                     .= 1.0
                grd[:qb_slack][bus][:dev_q][tii][idx.pr[bus]]                     .= -1.0
                grd[:qb_slack][bus][:sh_q][tii][idx.sh[bus]]                      .= 1.0 
                grd[:qb_slack][bus][:acline_qfr][tii][idx.bus_is_acline_frs[bus]] .= 1.0
                grd[:qb_slack][bus][:acline_qto][tii][idx.bus_is_acline_tos[bus]] .= 1.0
                grd[:qb_slack][bus][:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]       .= 1.0
                grd[:qb_slack][bus][:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]       .= 1.0
                grd[:qb_slack][bus][:dc_qfr][tii][idx.bus_is_dc_frs[bus]]         .= 1.0
                grd[:qb_slack][bus][:dc_qto][tii][idx.bus_is_dc_tos[bus]]         .= 1.0
            end

                #= 
    # Now, we solve each contingency at each time!
    for ctg_ii in prm.ctg.ctg_inds
        for tii in prm.ts.time_keys
            if ctg[:solver] == "LU"
                # this is the most direct, non-approximate
                # alternative: a straight LU solve
                #
                # solve Ax = b
                # b => stt[:pinj_DC][tii][bus][ctg[:nsb]], where nsb = nonslack buses
                # x => nonslack thetas
                # A => reduced admittance
                p        = stt[:pinj_DC][tii][bus][ctg[:nsb]]
                bt       = ctg[:b].*ctg[:phi]
                phi_inj  = ctg[:reduced_Et]*bt
                theta_ns = ctg[:reduced_Y][ctg_ii]\(p-phi_inj)
                p_flow   = ctg[:reduced_Y_flow][ctg_ii]*theta_ns + bt

            elseif ctg[:solver] == "pLU"
                # preconditioned LU
            elseif ctg[:solver] == "pLU with WMI"
                # use an LU solver on the base case,
                # but use the Woodbury matrix identity
                # to compute all low-rank updates
            elseif ctg[:solver] == "cg"
                # direct conjugate gradient
            elseif ctg[:solver] == "pcg"
                # preconditioned conjugate gradient
            elseif ctg[:solver] == "pcg with WMI"
                # preconditioned conjugate gradient 
                # withWoodbury matrix identity
            end
        end
    end
    =#

        # the base case, if helpful -- note: stt[:sfr_ctg][tii][end], with end = n+1, etc. does not exist, 
        # because we don't want to accidentally sum it into the upper scoring function
            #stt[:sfr_ctg][tii][end] = sqrt.(stt[:ac_qfr][tii].^2 + ctg[:pflow_k][tii][end].^2) - ctg[:s_max]
            #stt[:sto_ctg][tii][end] = sqrt.(stt[:ac_qto][tii].^2 + ctg[:pflow_k][tii][end].^2) - ctg[:s_max]
            #stt[:sp_ctg][tii][end]  = max(stt[:sfr_ctg][tii][end], stt[:sto_ctg][tii][end], 0.0)
            #stt[:zctg_s][tii][end]  = dt*prm.vio.s_flow*stt[:sp_ctg][tii][end]

            function score_zctg!(prm::quasiGrad.Param, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, qG::quasiGrad.QG, sys::quasiGrad.System)
                # loop over time
                for tii in prm.ts.time_keys
                    # loop over the contingencies
                    for ctg_ii in 1:sys.nctg
                        stt[:zctg][tii][ctg_ii] = -sum(stt[:zctg_s][tii][ctg_ii], init=0.0)
            
                        # evaluate the grd?
                        if qG.eval_grad
                            grd[:zctg][:zs][tii][ctg_ii] = -ones(length(stt[:zctg_s][tii][ctg_ii]))
                        end
                    end
            
                    # after this update, compute the minimum and average
                    stt[:zctg_min_t][tii] = minimum(stt[:zctg][tii])
                    stt[:zctg_avg_t][tii] = sum(stt[:zctg][tii])/sys.nctg
            
                    # evaluate the grd?
                    if qG.eval_grad
                        grd[:zctg_avg_t][:zctg][tii] = ones(sys.nctg)/sys.nctg
            
                        # find the biggest (smallest) violation
                        (_,ind_small) = findmin(stt[:zctg][tii])
            
                        # add 1 to the correspoding element
                        grd[:zctg_min_t][:zctg][tii]             = zeros(sys.nctg)
                        grd[:zctg_min_t][:zctg][tii][ind_small] += 1.0
                    end
                end
            
                # finally, update the sum across all time
                scr[:zctg_min] = sum(values(stt[:zctg_min_t]))
                scr[:zctg_avg] = sum(values(stt[:zctg_avg_t]))
            
                # evaluate the grd?
                if qG.eval_grad
                    for tii in prm.ts.time_keys
                        grd[:zctg_min][:zctg_min_t][tii] .= 1.0
                        grd[:zctg_avg][:zctg_avg_t][tii] .= 1.0
                    end
                end
            end
            rhs        = alpha_p_flow
            rhs_update = rhs - ones()
        #=
        # line injections -- "from" flows
        #
        # loop over each line, and compute the gradient of flows
        for line in idx.bus_is_acline_frs[bus]
            bus_fr = idx.acline_fr_bus[line]
            bus_to = idx.acline_to_bus[line]

            # make sure :)
            @assert bus_fr == bus

            # gradients
            vmfrpfr = grd[:acline_pfr][:vmfr][tii][line]
            vafrpfr = grd[:acline_pfr][:vafr][tii][line]
            uonpfr  = grd[:acline_pfr][:uon][tii][line]

            # "pfr" is also a function of the to bus voltages, so we need these
            vmtopfr = grd[:acline_pfr][:vmto][tii][line]
            vatopfr = grd[:acline_pfr][:vato][tii][line]

            # update the master grad -- pfr, at this bus and its corresponding "to" bus
            mgd[:vm][tii][bus_fr]        += -alpha[bus-1]*vmfrpfr
            mgd[:va][tii][bus_fr]        += -alpha[bus-1]*vafrpfr
            mgd[:vm][tii][bus_to]        += -alpha[bus-1]*vmtopfr
            mgd[:va][tii][bus_to]        += -alpha[bus-1]*vatopfr
            mgd[:u_on_acline][tii][line] += -alpha[bus-1]*uonpfr
        end
        # line injections -- "to" flows
        #
        # loop over each line, and compute the gradient of flows
        for line in idx.bus_is_acline_tos[bus]
            bus_to = idx.acline_to_bus[line]
            bus_fr = idx.acline_fr_bus[line]

            # make sure :)
            @assert bus_to == bus

            # gradients
            vmtopto = grd[:acline_pto][:vmto][tii][line]
            vatopto = grd[:acline_pto][:vato][tii][line]
            uonpto  = grd[:acline_pto][:uon][tii][line]

            # "pto" is also a function of the fr bus voltages, so we need these
            vmfrpto = grd[:acline_pto][:vmfr][tii][line]
            vafrpto = grd[:acline_pto][:vafr][tii][line]

            # update the master grad -- pto, at this bus and its corresponding "fr" bus
            mgd[:vm][tii][bus_to]        += -alpha[bus-1]*vmtopto
            mgd[:va][tii][bus_to]        += -alpha[bus-1]*vatopto
            mgd[:vm][tii][bus_fr]        += -alpha[bus-1]*vmfrpto
            mgd[:va][tii][bus_fr]        += -alpha[bus-1]*vafrpto
            mgd[:u_on_acline][tii][line] += -alpha[bus-1]*uonpto
        end
        
        # xfm injections -- "from" flows
        #
        # loop over each xfm, and compute the gradient of flows
        for xfm in idx.bus_is_xfm_frs[bus]
            bus_fr = idx.xfm_fr_bus[xfm]
            bus_to = idx.xfm_to_bus[xfm]

            # make sure :)
            @assert bus_fr == bus

            # gradients
            vmfrpfr = grd[:xfm_pfr][:vmfr][tii][xfm]
            vafrpfr = grd[:xfm_pfr][:vafr][tii][xfm]
            uonpfr  = grd[:xfm_pfr][:uon][tii][xfm]

            # "pfr" is also a function of the to bus voltages, so we need these
            vmtopfr = grd[:xfm_pfr][:vmto][tii][xfm]
            vatopfr = grd[:xfm_pfr][:vato][tii][xfm]

            # xfm ratios
            taupfr  = grd[:xfm_pfr][:tau][tii][xfm]
            phipfr  = grd[:xfm_pfr][:phi][tii][xfm]

            # update the master grad -- pfr, at this bus and its corresponding "to" bus
            mgd[:vm][tii][bus_fr]    += -alpha[bus-1]*vmfrpfr
            mgd[:va][tii][bus_fr]    += -alpha[bus-1]*vafrpfr
            mgd[:vm][tii][bus_to]    += -alpha[bus-1]*vmtopfr
            mgd[:va][tii][bus_to]    += -alpha[bus-1]*vatopfr
            mgd[:tau][tii][xfm]      += -alpha[bus-1]*taupfr
            mgd[:phi][tii][xfm]      += -alpha[bus-1]*phipfr
            mgd[:u_on_xfm][tii][xfm] += -alpha[bus-1]*uonpfr
        end
        # xfm injections -- "to" flows
        #
        # loop over each xfm, and compute the gradient of flows
        for xfm in idx.bus_is_xfm_tos[bus]
            bus_to = idx.xfm_to_bus[xfm]
            bus_fr = idx.xfm_fr_bus[xfm]

            # make sure :)
            @assert bus_to == bus

            # gradients
            vmtopto = grd[:xfm_pto][:vmto][tii][xfm]
            vatopto = grd[:xfm_pto][:vato][tii][xfm]
            uonpto  = grd[:xfm_pto][:uon][tii][xfm]

            # "pto" is also a function of the fr bus voltages, so we need these
            vmfrpto = grd[:xfm_pto][:vmfr][tii][xfm]
            vafrpto = grd[:xfm_pto][:vafr][tii][xfm]

            # xfm ratios
            taupto  = grd[:xfm_pto][:tau][tii][xfm]
            phipto  = grd[:xfm_pto][:phi][tii][xfm]

            # update the master grad -- pto, at this bus and its corresponding "fr" bus
            mgd[:vm][tii][bus_to]    += -alpha[bus-1]*vmtopto
            mgd[:va][tii][bus_to]    += -alpha[bus-1]*vatopto
            mgd[:vm][tii][bus_fr]    += -alpha[bus-1]*vmfrpto
            mgd[:va][tii][bus_fr]    += -alpha[bus-1]*vafrpto
            mgd[:tau][tii][xfm]      += -alpha[bus-1]*taupto
            mgd[:phi][tii][xfm]      += -alpha[bus-1]*phipto
            mgd[:u_on_xfm][tii][xfm] += -alpha[bus-1]*uonpto
        end
        =#
        :zctg_s     => Dict(tkeys[ii] => Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg) for ii in 1:(sys.nT))
        #grd[:ctg_min_update][tii] = Float64(sys.nctg)
        # check:
        # grad[:ctg_avg][tii] * grad[:ctg_min_update][tii] 
        #              = (dt*prm.vio.s_flow/sys.nctg) * (sys.nctg)
        #              = dt*prm.vio.s_flow, which is what we need to add!
# test if this contingency has the new smallest score
                #if stt[:zctg][tii][ctg_ii] < ctg_score_min
                #    ctg_ii_min      = ctg_ii
                #    ctg_score_min   = stt[:zctg][tii][ctg_ii]
                ##    xfr_inds_min    = xfr_inds    
                 #   xto_inds_min    = xto_inds
               #     xfr_alpha_min   = xfr_alpha 
                 #   xto_alpha_min   = xto_alpha
                  #  aclfr_inds_min  = aclfr_inds
                  #  aclto_inds_min  = aclto_inds
                  #  aclfr_alpha_min = aclfr_alpha
                  #  aclto_alpha_min = aclto_alpha
                #end


        # eval grad?
        #if qG.eval_grad
        #    # only run this if there is a contingency violation
        #    if ctg_ii_min != -1
        #        # now, apply the gradient on the min case -- this is tricky,
        #        # because we don't know which one is the min until we have tested
        #        # them all. But, we don't want to have to store everything to run
        #        # independent loops, so we just re-run the worst case one a second time.
        #        #
        #        # note: we use gc_min_update to quickly scale up the gradients.
        #        zctgs_grad_q_acline!(tii, idx, grd, mgd, aclfr_inds_min, aclto_inds_min, gc_min_update*aclfr_alpha_min, gc_min_update*aclto_alpha_min)
        #        zctgs_grad_q_xfm!(tii, idx, grd, mgd, xfr_inds_min, xto_inds_min, gc_min_update*xfr_alpha_min, gc_min_update*xto_alpha_min)
        #        zctgs_grad_pinj!(ctg, tii, idx, grd, mgd, gc_min_update*ctg[:dz_dpinj][tii][ctg_ii_min], sys)
        #    end
        # end

        # ... however, we use an update parameter, such that (gc_avg * gc_min_update) = gc_min_update, because,
        # 1) gc_avg is what we applied as a scalar,
        # 2) gc_min_update is what we also want to apply,
        # 3) so gc_avg * gc_min_update gives us the desired update!
        #gc_min_update = grd[:ctg_min_update][tii]

        # loop over ctgs to score
        ctg_score_min = 0.0 # use to track the minimum score -- reset each time index
        ctg_ii_min    = -1  # use to track the minimum score -- reset each time index

        # initialize structure used for the worst-case ctg
        xfr_inds_min    = Int64[]
        xto_inds_min    = Int64[]
        xfr_alpha_min   = Float64[]
        xto_alpha_min   = Float64[]
        aclfr_inds_min  = Int64[]
        aclto_inds_min  = Int64[]
        aclfr_alpha_min = Float64[]
        aclto_alpha_min = Float64[]

        for tii in prm.ts.time_keys
            # for bus in 1:sys.nb
            # pb_slack
            #    grd[:pb_slack][bus][:dev_p][tii][idx.cs[bus]]                     .= 1.0
            #    grd[:pb_slack][bus][:dev_p][tii][idx.pr[bus]]                     .= -1.0
            #    grd[:pb_slack][bus][:sh_p][tii][idx.sh[bus]]                      .= 1.0 
            #    grd[:pb_slack][bus][:acline_pfr][tii][idx.bus_is_acline_frs[bus]] .= 1.0
            #    grd[:pb_slack][bus][:acline_pto][tii][idx.bus_is_acline_tos[bus]] .= 1.0
            #    grd[:pb_slack][bus][:xfm_pfr][tii][idx.bus_is_xfm_frs[bus]]       .= 1.0
            #    grd[:pb_slack][bus][:xfm_pto][tii][idx.bus_is_xfm_tos[bus]]       .= 1.0
            #    grd[:pb_slack][bus][:dc_pfr][tii][idx.bus_is_dc_frs[bus]]         .= 1.0
            #    grd[:pb_slack][bus][:dc_pto][tii][idx.bus_is_dc_tos[bus]]         .= 1.0
            # pb_slack
            #    grd[:qb_slack][bus][:dev_q][tii][idx.cs[bus]]                     .= 1.0
            #    grd[:qb_slack][bus][:dev_q][tii][idx.pr[bus]]                     .= -1.0
            #    grd[:qb_slack][bus][:sh_q][tii][idx.sh[bus]]                      .= 1.0 
            #    grd[:qb_slack][bus][:acline_qfr][tii][idx.bus_is_acline_frs[bus]] .= 1.0
            #    grd[:qb_slack][bus][:acline_qto][tii][idx.bus_is_acline_tos[bus]] .= 1.0
            #    grd[:qb_slack][bus][:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]       .= 1.0
            #    grd[:qb_slack][bus][:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]       .= 1.0
            #    grd[:qb_slack][bus][:dc_qfr][tii][idx.bus_is_dc_frs[bus]]         .= 1.0
            #    grd[:qb_slack][bus][:dc_qto][tii][idx.bus_is_dc_tos[bus]]         .= 1.0
            #end
        # power balance
        #:pb_slack     => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)),
        #:qb_slack     => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)),

        # power balance -- this one is special, because each bus gets its own dictionary *before* the "wrt" variable
        :pb_slack => Dict(jj => Dict(:dev_p      => Dict(tkeys[ii] => quasiGrad.spzeros(sys.ndev) for ii in 1:(sys.nT)),
                                     :sh_p       => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nsh) for ii in 1:(sys.nT)),
                                     :acline_pfr => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nl) for ii in 1:(sys.nT)),
                                     :acline_pto => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nl) for ii in 1:(sys.nT)),
                                     :xfm_pfr    => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nx) for ii in 1:(sys.nT)),
                                     :xfm_pto    => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nx) for ii in 1:(sys.nT)),
                                     :dc_pfr     => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nldc) for ii in 1:(sys.nT)),
                                     :dc_pto     => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nldc) for ii in 1:(sys.nT))) for jj in 1:(sys.nb)),
        :qb_slack => Dict(jj => Dict(:dev_q      => Dict(tkeys[ii] => quasiGrad.spzeros(sys.ndev) for ii in 1:(sys.nT)),
                                     :sh_q       => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nsh) for ii in 1:(sys.nT)),
                                     :acline_qfr => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nl) for ii in 1:(sys.nT)),
                                     :acline_qto => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nl) for ii in 1:(sys.nT)),
                                     :xfm_qfr    => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nx) for ii in 1:(sys.nT)),
                                     :xfm_qto    => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nx) for ii in 1:(sys.nT)),
                                     :dc_qfr     => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nldc) for ii in 1:(sys.nT)),
                                     :dc_qto     => Dict(tkeys[ii] => quasiGrad.spzeros(sys.nldc) for ii in 1:(sys.nT))) for jj in 1:(sys.nb))
                                    # now, compute the penalized mismatches
                # nope => stt[:pb_penalty][tii][bus] = abs(stt[:pb_slack][tii][bus])
                # nope => stt[:qb_penalty][tii][bus] = abs(stt[:qb_slack][tii][bus])
                # stt[:zp][tii][bus] = stt[:pb_penalty][tii][bus]*cp*dt
                # stt[:zq][tii][bus] = stt[:qb_penalty][tii][bus]*cq*dt
        # evaluate the grd (penalties)?
        if qG.eval_grad
            if qG.pb_grad 
            # penalized mismatches and penalty gradients
            grd[:zp][:pb_slack][tii] = cp*dt*sign.(stt[:pb_slack][tii])
            grd[:zq][:qb_slack][tii] = cq*dt*sign.(stt[:qb_slack][tii])
        end
        qG = Dict(
            # penalty gradients are expensive -- only compute the gradient
            # if the constraint is violated by more than this value
            :pg_tol                   => 1e-4,
            # amount to penalize constraint violations
            :delta                    => 0*prm.vio.p_bus,
            # mainly for testing
            :eval_grad                => true,
            # amount to prioritize binary selection over continuous variables
            :binary_projection_weight => 100.0,
            # print stats at the end?
            :print_final_stats        => true,
            # mip gap for Gurobi
            :mip_gap                  => 1/100.0,
            # Gurobi time limit
            :time_lim                 => 10.0,
            # how much should Gurobi print?
            :GRB_output_flag          => 0,
            # print nzms at every adam iteration?
            :print_zms               => true,
            # ctg solver settings
            :frac_ctg_keep            => 1.0, # all are scored, but only 100*frac_ctg_keep%
                                              # are differentiated and included in the mastergrad
            :pcg_tol                  => 1e-3,
            :cutoff_level             => 2,
            :base_solver              => "pcg", # "lu", "pcg" for approx
            :ctg_solver               => "wmi", # "lu", "pcg", for approx, "wmi" for low rank updates
            :build_ctg_full           => true,  # build the full contingency matrices?
            :build_ctg_lowrank        => true,  # build the low rank contingency elements?
            #                                      -- you don't need both ^, unless for testing
            # initialize adam parameters
            :eps        => 1e-8,
            :beta1      => 0.99,
            :beta2      => 0.999,
            :alpha_0    => 0.001,
            :alpha_min  => 0.001/10.0,   # for cos decay
            :alpha_max  => 0.001/0.5,  # for cos decay
            :Ti         => 100,        # for cos decay -- at Tcurr == Ti, cos() => -1
            :step_decay => 0.999,
            # specify step size decay approach: "cos", "none", or "exponential"
            :decay_type => "cos",
            # gradient modifications -- power balance
            :pqbal_grad_type     => "soft_abs", # "standard"
            :pqbal_grad_weight_p => 10.0, # standard: prm.vio.p_bus
            :pqbal_grad_weight_q => 10.0, # standard: prm.vio.q_bus
            :pqbal_grad_eps2     => 1e-5,
            )
            # don't need => grd[:acline_sto][:pto][tii]  = stt[:acline_pto][tii]./stt[:acline_sto][tii]
            # don't need => grd[:acline_sto][:qto][tii]  = stt[:acline_qto][tii]./stt[:acline_sto][tii]   

                        # don't need => grd[:acline_sfr][:pfr][tii]  = stt[:acline_pfr][tii]./stt[:acline_sfr][tii]
            # don't need => grd[:acline_sfr][:qfr][tii]  = stt[:acline_qfr][tii]./stt[:acline_sfr][tii]  

                        # don't need => grd[:xfm_sfr][:pfr][tii]  = stt[:xfm_pfr][tii]./stt[:xfm_sfr][tii]
            # don't need => grd[:xfm_sfr][:qfr][tii]  = stt[:xfm_qfr][tii]./stt[:xfm_sfr][tii]

                        # don't need => grd[:xfm_sto][:pto][tii]  = stt[:xfm_pto][tii]./stt[:xfm_sto][tii]
            # don't need => grd[:xfm_sto][:qto][tii]  = stt[:xfm_qto][tii]./stt[:xfm_sto][tii]

:g_tv_shunt   => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))
:b_tv_shunt   => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))

        # power balance (locally saved)
        :pb_slack     => zeros(sys.nb)
        :qb_slack     => zeros(sys.nb)

# startup state bound -- each "sus_bnd" is a function of time,
# device, and the number of potential startup state "f", given as ":num_sus"
:u_sus_bnd => Dict(tkeys[ii] => Dict(dev => zeros(prm.dev.num_sus[dev]) for dev in 1:sys.ndev) for ii in 1:(sys.nT))

        # contingencies :)
        :sfr_ctg           => Dict(tkeys[ii] => Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:(sys.nctg + 1)) for ii in 1:(sys.nT))
        :sto_ctg           => Dict(tkeys[ii] => Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:(sys.nctg + 1)) for ii in 1:(sys.nT))

        qG.constraint_grad_weight*sum(sum.(values(scr[:zhat_mxst]), init=0.0), init=0.0)
        :zhat_mxst         => Dict(ii => zeros(prm.dev.num_mxst[ii]) for ii in 1:(sys.ndev))
        #:zw_enmax   => Dict(ii => zeros(prm.dev.num_W_enmax[ii]) for ii in 1:(sys.ndev)),
        #:zw_enmin   => Dict(ii => zeros(prm.dev.num_W_enmax[ii]) for ii in 1:(sys.ndev)),

        function score_z_enmin_enmax!(prm::quasiGrad.Param, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, qG::quasiGrad.QG, sys::quasiGrad.System)
            # loop over the device indices and call the device id
            for dev in 1:sys.ndev
                scr[:z_enmax][dev] = -sum(stt[:zw_enmax][dev]; init=0.0)
                scr[:z_enmin][dev] = -sum(stt[:zw_enmin][dev]; init=0.0)
            end
        end
        quasiGrad.score_z_enmin_enmax!(prm, idx, stt, grd, qG, sys)

    vio_prm     = quasiGrad.parse_json_violation(json_data)
    reserve_prm = quasiGrad.parse_json_reserve(json_data, vio_prm)

# parse the json into dicts -- violations
function parse_json_violation(json_data::Dict)
    # setup outputs direclty
    violation_param = Dict(
        :p_bus      => Float64(json_data["network"]["violation_cost"]["p_bus_vio_cost"]),
        :q_bus      => Float64(json_data["network"]["violation_cost"]["q_bus_vio_cost"]),
        :s_flow     => Float64(json_data["network"]["violation_cost"]["s_vio_cost"]),
        :e_dev      => Float64(json_data["network"]["violation_cost"]["e_vio_cost"]),
        :rgu_zonal  => Float64[],
        :rgd_zonal  => Float64[],
        :scr_zonal  => Float64[],
        :nsc_zonal  => Float64[],
        :rru_zonal  => Float64[],
        :rrd_zonal  => Float64[],
        :qru_zonal  => Float64[],
        :qrd_zonal  => Float64[])

    # output
    return violation_param
end

:acline_sfr_plus => Dict(:acline_pfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)), 
:acline_qfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
:acline_sto_plus => Dict(:acline_pto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)), 
:acline_qto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)))

:xfm_sfr_plus => Dict(:xfm_pfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
:xfm_qfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
:xfm_sto_plus => Dict(:xfm_pto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
:xfm_qto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))) 


    # shunt gradients
    grd[:g_tv_shunt][:u_step_shunt] = prm.shunt.gs
    grd[:b_tv_shunt][:u_step_shunt] = prm.shunt.bs

    :g_tv_shunt => Dict(:u_step_shunt => zeros(sys.nsh)),
    :b_tv_shunt => Dict(:u_step_shunt => zeros(sys.nsh)),

        # zones (endogenous)
        :p_rgu_zonal_REQ => Dict(jj => Dict(:dev_p => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_rgd_zonal_REQ => Dict(jj => Dict(:dev_p => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_scr_zonal_REQ => Dict(jj => Dict(:dev_p => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_nsc_zonal_REQ => Dict(jj => Dict(:dev_p => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),

        # zone power penalties
        # 1
        :p_rgu_zonal_penalty => Dict(jj => Dict(:p_rgu_zonal_REQ => Dict(tkeys[ii] => 0.0             for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_rgu_zonal_penalty => Dict(jj => Dict(:p_rgu           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        # 2
        :p_rgd_zonal_penalty => Dict(jj => Dict(:p_rgd_zonal_REQ => Dict(tkeys[ii] => 0.0             for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_rgd_zonal_penalty => Dict(jj => Dict(:p_rgd           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        # 3
        :p_scr_zonal_penalty => Dict(jj => Dict(:p_rgu_zonal_REQ => Dict(tkeys[ii] => 0.0             for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_scr_zonal_penalty => Dict(jj => Dict(:p_scr_zonal_REQ => Dict(tkeys[ii] => 0.0             for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_scr_zonal_penalty => Dict(jj => Dict(:p_rgu           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_scr_zonal_penalty => Dict(jj => Dict(:p_scr           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        # 4
        :p_nsc_zonal_penalty => Dict(jj => Dict(:p_rgu_zonal_REQ => Dict(tkeys[ii] => 0.0             for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_nsc_zonal_penalty => Dict(jj => Dict(:p_scr_zonal_REQ => Dict(tkeys[ii] => 0.0             for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_nsc_zonal_penalty => Dict(jj => Dict(:p_nsc_zonal_REQ => Dict(tkeys[ii] => 0.0             for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_nsc_zonal_penalty => Dict(jj => Dict(:p_rgu           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_nsc_zonal_penalty => Dict(jj => Dict(:p_scr           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_nsc_zonal_penalty => Dict(jj => Dict(:p_nsc           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        # 5
        :p_rru_zonal_penalty => Dict(jj => Dict(:p_rru_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_rru_zonal_penalty => Dict(jj => Dict(:p_rru_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        # 6
        :p_rrd_zonal_penalty => Dict(jj => Dict(:p_rrd_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        :p_rrd_zonal_penalty => Dict(jj => Dict(:p_rrd_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzP)),
        # 7 -- reactive
        :q_qru_zonal_penalty => Dict(jj => Dict(:q_qru => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzQ)),
        # 8 -- reactive
        :q_qrd_zonal_penalty => Dict(jj => Dict(:q_qrd => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))) for jj in 1:(sys.nzQ)),

        :z_enmax => Dict(:zw_enmax => 0.0),
        :z_enmin => Dict(:zw_enmin => 0.0),

            # min and max energy penalty scores
    grd[:z_enmax][:zw_enmax] = -1.0
    grd[:z_enmin][:zw_enmin] = -1.0

    # device active power 
    grd[:dev_p][:p_on] = 1.0
    grd[:dev_p][:p_su] = 1.0
    grd[:dev_p][:p_sd] = 1.0

    # device active power 
    grd[:dev_p][:p_on] = 1.0
    grd[:dev_p][:p_su] = 1.0
    grd[:dev_p][:p_sd] = 1.0

    # derivative of qjt with respect to "u_sum" (Jpqe)
    grd[:dev_q][:u_sum] = prm.dev.q_0 # not a function of time!

    # derivative of qjt with respect to "u_sum" (Jpqe)
    grd[:dev_q][:dev_p] = prm.dev.beta

    :dev_q => Dict(:u_sum => zeros(sys.ndev),
    :dev_p => zeros(sys.ndev)),

        # static gradients
        grd[:zsu_dev][:u_su_dev]       = prm.dev.startup_cost
        grd[:zsu_acline][:u_su_acline] = prm.acline.connection_cost
        grd[:zsu_xfm][:u_su_xfm]       = prm.xfm.connection_cost
        grd[:zsd_dev][:u_sd_dev]       = prm.dev.shutdown_cost
        grd[:zsd_acline][:u_sd_acline] = prm.acline.disconnection_cost
        grd[:zsd_xfm][:u_sd_xfm]       = prm.xfm.disconnection_cost

        :zon_dev    => Dict(:u_on_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zsu_dev    => Dict(:u_su_dev    => zeros(sys.ndev)),
        :zsu_acline => Dict(:u_su_acline => zeros(sys.nl)),
        :zsu_xfm    => Dict(:u_su_xfm    => zeros(sys.nx)),
        :zsd_dev    => Dict(:u_sd_dev    => zeros(sys.ndev)),
        :zsd_acline => Dict(:u_sd_acline => zeros(sys.nl)),
        :zsd_xfm    => Dict(:u_sd_xfm    => zeros(sys.nx)),

        :zon_dev    => Dict(:u_on_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),

    # zonal reserve costs
    for (t_ind,tii) in enumerate(prm.ts.time_keys)
        dt = prm.ts.duration[tii]
        grd[:zrgu_zonal][:p_rgu_zonal_penalty][tii] = dt*prm.vio.rgu_zonal
        grd[:zrgd_zonal][:p_rgd_zonal_penalty][tii] = dt*prm.vio.rgd_zonal
        grd[:zscr_zonal][:p_scr_zonal_penalty][tii] = dt*prm.vio.scr_zonal
        grd[:znsc_zonal][:p_nsc_zonal_penalty][tii] = dt*prm.vio.nsc_zonal
        grd[:zrru_zonal][:p_rru_zonal_penalty][tii] = dt*prm.vio.rru_zonal
        grd[:zrrd_zonal][:p_rrd_zonal_penalty][tii] = dt*prm.vio.rrd_zonal
        grd[:zqru_zonal][:q_qru_zonal_penalty][tii] = dt*prm.vio.qru_zonal
        grd[:zqrd_zonal][:q_qrd_zonal_penalty][tii] = dt*prm.vio.qrd_zonal
    end

    # power balance gradients -- not chaning! -- we don't actually use these :)
    for tii in prm.ts.time_keys
        # ctg =========
        dt = prm.ts.duration[tii]
        grd[:ctg_avg][tii]        = dt*prm.vio.s_flow/sys.nctg
        grd[:ctg_min][tii]        = dt*prm.vio.s_flow
    end

    # Device reserve costs
    for (t_ind,tii) in enumerate(prm.ts.time_keys)
        dt = prm.ts.duration[tii]
        grd[:zrgu][:p_rgu][tii]     = dt*prm.dev.p_reg_res_up_cost_tmdv[t_ind]
        grd[:zrgd][:p_rgd][tii]     = dt*prm.dev.p_reg_res_down_cost_tmdv[t_ind]
        grd[:zscr][:p_scr][tii]     = dt*prm.dev.p_syn_res_cost_tmdv[t_ind]
        grd[:znsc][:p_nsc][tii]     = dt*prm.dev.p_nsyn_res_cost_tmdv[t_ind]
        grd[:zrru][:p_rru_on][tii]  = dt*prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind]
        grd[:zrru][:p_rru_off][tii] = dt*prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind]
        grd[:zrrd][:p_rrd_on][tii]  = dt*prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind]
        grd[:zrrd][:p_rrd_off][tii] = dt*prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind]
        grd[:zqru][:q_qru][tii]     = dt*prm.dev.q_res_up_cost_tmdv[t_ind]
        grd[:zqrd][:q_qrd][tii]     = dt*prm.dev.q_res_down_cost_tmdv[t_ind]
    end

    # zon_dev, zsu_dev, zsd_dev (also for lines and xfms)
    for tii in prm.ts.time_keys
        dt = prm.ts.duration[tii]
        grd[:zon_dev][:u_on_dev][tii] = dt*prm.dev.on_cost
    end

    # zon_dev, zsu_dev, zsd_dev (also for lines and xfms)
    for tii in prm.ts.time_keys
        dt = prm.ts.duration[tii]
        grd[:zon_dev][:u_on_dev][tii] = dt*prm.dev.on_cost
    end

# deprecated
function adam_legacy(adam_prm::Dict{Symbol,Float64}, adam_states::Dict{Symbol,Vector{Float64}}, loss_grad::Vector{Float64}, idx::quasiGrad.Idx)
    adam_states[:adam_step][1]                  += 1.0
    adam_states[:alpha_decay]                    = [adam_prm[:alpha]*(adam_prm[:step_decay]^adam_states[:adam_step][1])]
    adam_states[:m][idx[:adam][:update]]         = adam_prm[:beta1].*adam_states[:m][idx[:adam][:update]] + (1-adam_prm[:beta1]).*loss_grad[idx[:adam][:update]]
    adam_states[:v][idx[:adam][:update]]         = adam_prm[:beta2].*adam_states[:v][idx[:adam][:update]] + (1-adam_prm[:beta2]).*loss_grad[idx[:adam][:update]].^2
    adam_states[:mhat][idx[:adam][:update]]      = adam_states[:m][idx[:adam][:update]]/(1-adam_prm[:beta1]^adam_states[:adam_step][1])
    adam_states[:vhat][idx[:adam][:update]]      = adam_states[:v][idx[:adam][:update]]/(1-adam_prm[:beta2]^adam_states[:adam_step][1])
    adam_states[:GO_states][idx[:adam][:update]] = adam_states[:GO_states][idx[:adam][:update]] - adam_states[:alpha_decay].*adam_states[:mhat][idx[:adam][:update]]./(sqrt.(adam_states[:vhat][idx[:adam][:update]]) .+ adam_prm[:eps])

    return adam_states
end

#=
    # ctg
    ctg = Dict(
        # aclines
        :s_max           => s_max_ctg,     # max contingency flows
        :E               => E,             # full incidence matrix
        :Er              => Er,            # reduced incidence matrix
        :Yb              => Yb,            # full Ybus (DC)    
        :Ybr             => Ybr,           # reduced Ybus (DC)
        :Yfr             => Yfr,           # reduced flow matrix (DC)
        :ctg_out_ind     => ctg_out_ind,   # for each ctg, the list of line indices
        :ctg_params      => ctg_params,    # for each ctg, the list of (negative) params
        :Ybr_k           => Ybr_k,         # if build_ctg == true, reduced admittance matrix for each ctg 
        :b               => ac_b_params,   # base case susceptance parameters
        :xfm_at_bus      => xfm_at_bus,
        :xfm_phi_scalars => xfm_phi_scalars,
        :p_slack         => p_slack,
        :p_inj           => p_inj,
        :pflow_k         => pflow_k,
        :theta_k         => theta_k,
        :sfr             => sfr,     
        :sto             => sto,     
        :sfr_vio         => sfr_vio, 
        :sto_vio         => sto_vio,  
        :dz_dpinj        => dz_dpinj,
        :ctd   => ctd,
        :ctg_to_score    => ctg_to_score,
        :alpha           => prm.ctg.alpha,
        :components      => prm.ctg.components,
        :id              => prm.ctg.id,
        :ctg_inds        => prm.ctg.ctg_inds,
        :Ybr_ChPr        => Ybr_ChPr,   # base case preconditioner (everyone uses it!)
        :v_k             => v_k,        # low rank update vectors: v*b*v'
        :b_k             => b_k,        # low rank update scalar: v*b*v'
        :u_k             => u_k,        # low rank update vector: u = Y\v
        :w_k             => w_k)        # low rank update vector: w = b*u/(1+v'*u*b)
=#

json_data = Dict{String,Any}()
open(json_file_path, "r") do io
    json_data = JSON.parse(io)
end


# loop over Gurobi projections
for ii = 1:5
    global prev_step  = 0

    for ii in 1:N_its
        # compute all states and grads
        quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

        # take an adam step
        quasiGrad.adam!(adm_step, prm, stt, upd, adm, mgd, qG)

        # plot
        if strt == 1
            zmean = scr[:nzms]
            strt = 0
        else
            display(quasiGrad.plot!(p1, [prev_step, adm[:step]],[prev_val, scr[:nzms]],label = "", xlim = [0; N_its], ylim = [-zmean/2; 1.1*zmean], linewidth=1.75, color = plot_color ))
        end

        # previous
        prev_step = adm[:step]
        prev_val  = scr[:nzms]
    end

    # apply a Gurobi projection!
    # quasiGrad.Gurobi_projection!(prm, idx, stt, grd, qG, sys, upd, GRB)
end

# read and parse the input data
jsn, prm, idx, sys = quasiGrad.load_and_parse_json(data_dir*file_name)
qG                 = quasiGrad.initialize_qG(prm)
qG.eval_grad       = true


# --------------------
cgd, GRB, grd, mgd, scr, stt = quasiGrad.initialize_states(idx, prm, sys)

# perturb stt
perturb!(stt, prm, idx, grd, sys, qG, 1.0)
            
(n_its-1) ? fix = true : fix = false
quasiGrad.snap_shunts!(prm, qG, stt, upd, fix)
end

# %%
ac_phi              = zeros(sys.nac)
ac_qfr              = zeros(sys.nac)
ac_qto              = zeros(sys.nac)
dsmax_dp_flow       = zeros(sys.nac)
dsmax_dqfr_flow     = zeros(sys.nac)
dsmax_dqto_flow     = zeros(sys.nac)
p_inj               = zeros(sys.nb)

# are we solving power flow?
skip = false # this is for a standard adam iteration
#if qG.solve_pf == true
#    # solve pf! in this case, we only iterate on voltage and phase + dc line powers + xfm and shunt params
#    if var_key ∉ [:vm, :va, :dc_pfr, :dc_qto, :dc_qfr, :tau, :phi, :u_step_shunt]
#        skip = true
#    end
#end



if plt[:plot]
    if plt[:first_plot]
        p1 = quasiGrad.plot()
        display(p1)
        label1 = "market surplus"       
        label2 = "constraint penalties" 
        label3 = "contingency violations"
    else
        label1 = ""      
        label2 = ""
        label3 = ""
    end
    v_nms_prev = 0.0 # init
    v_pen_prev = 0.0 # init
    v_ctg_prev = 0.0 # init
end


    # only plot of this isn't the first iteration
    if adm_step != 1
        # record
        v_nms = copy(scr[:nzms])                       + 1e-3  # bc log
        v_pen = copy(-scr[:zt_penalty])                + 1e-3  # bc log
        v_ctg = copy(-scr[:zctg_min] - scr[:zctg_avg]) + 1e-3  # bc log



        # plots
        display(quasiGrad.plot!([adm_step-1, adm_step],[v_nms_prev, v_nms], label = label1, xlim = [1; plt[:N_its]], ylim = [plt[:zmean]/qG.plot_scale_dn; qG.plot_scale_up*plt[:zmean]], linewidth=1.75, color = 1, yaxis=:log, xlabel = "adam iteration", ylabel = "score values (z)", legend = false))
        display(quasiGrad.plot!([adm_step-1, adm_step],[v_pen_prev, v_pen], label = label2, xlim = [1; plt[:N_its]], ylim = [plt[:zmean]/qG.plot_scale_dn; qG.plot_scale_up*plt[:zmean]], linewidth=1.75, color = 2, yaxis=:log, legend = false))
        display(quasiGrad.plot!([adm_step-1, adm_step],[v_ctg_prev, v_ctg], label = label3, xlim = [1; plt[:N_its]], ylim = [plt[:zmean]/qG.plot_scale_dn; qG.plot_scale_up*plt[:zmean]], linewidth=1.75, color = 3, yaxis=:log, legend = false))
    end



    # update! ======================
    v_nms_prev = copy( scr[:nzms])                      + 1e-3 # bc log
    v_pen_prev = copy(-scr[:zt_penalty])                + 1e-3 # bc log
    v_ctg_prev = copy(-scr[:zctg_min] - scr[:zctg_avg]) + 1e-3 # bc log
    if adm_step > 1
        # turn these off after the first plotting round
        label1     = ""      
        label2     = ""
        label3     = ""
    end

        # if this is a first plot, or adam 

        # now, set the previous scoring values
        z_prev[:zms]  = scale_z(scr[:zms])
        z_prev[:pzms] = scale_z(scr[:zms_penalized])      
        z_prev[:zhat] = scale_z(scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst])
        z_prev[:ctg]  = scale_z(scr[:zctg_min] + scr[:zctg_avg])
        z_prev[:emnx] = scale_z(scr[:emnx])
        z_prev[:zp]   = scale_z(scr[:zp])
        z_prev[:zq]   = scale_z(scr[:zq])
        z_prev[:acl]  = scale_z(scr[:acl])
        z_prev[:xfm]  = scale_z(scr[:xfm])
        z_prev[:zoud] = scale_z(scr[:zoud])
        z_prev[:zone] = scale_z(scr[:zone])
        z_prev[:rsv]  = scale_z(scr[:rsv])
        z_prev[:enpr] = scale_z(scr[:enpr])
        z_prev[:encs] = scale_z(scr[:encs])
        z_prev[:zsus] = scale_z(scr[:zsus])

end

stt[:q_qru][tii][intersect(idx.pr_devs,idx.J_pqe)] .= 0.0   # see (117)
stt[:q_qrd][tii][intersect(idx.pr_devs,idx.J_pqe)] .= 0.0   # see (118)
stt[:q_qru][tii][intersect(idx.cs_devs,idx.J_pqe)] .= 0.0   # see (127)
stt[:q_qrd][tii][intersect(idx.cs_devs,idx.J_pqe)] .= 0.0   # see (128)

# filter out the ones in Jpqe
pr_NOT_Jpqe_on_bus = setdiff(pr_devs_on_bus, idx.J_pqe)
cs_NOT_Jpqe_on_bus = setdiff(cs_devs_on_bus, idx.J_pqe)

# keep the ones in Jpqe
pr_AND_Jpqe_on_bus = intersect(pr_devs_on_bus, idx.J_pqe)
cs_AND_Jpqe_on_bus = intersect(cs_devs_on_bus, idx.J_pqe)

# solve power flow (to some degree of accuracy)
function solve_linear_pf_with_Gurobi!(Jac::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG,  stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol)
    # ask Gurobi to solve a linearize power flow. two options here:
    #   1) define device variables which are bounded, and then insert them into the power balance expression
    #   2) just define power balance bounds based on device characteristics, and then, at the end, optimally
    #      dispatch the devices via updating.
    # 
    # I am pretty sure 1) is slower, but it is (1) safer, (2) simpler, and (3) more flexible -- and it will
    # require less trouble-shooting. Plus, it will be easy to update/improve in the future! Go with it.
    #
    #
    # here is power balance:
    #
    # p_pr - p_cs - pdc = p_lines/xfm/shunt => this is typical.
    #
    # vm0 = stt[:vm][tii]
    # va0 = stt[:va][tii][2:end-1]
    #
    # bias point: msc[:pinj0, qinj0] === Y * stt[:vm, va]
    qG.Gurobi_pf_obj            = "min_dispatch_distance"
    qG.compute_pf_injs_with_Jac = true

    # build and empty the model!
    model = Model(Gurobi.Optimizer)
    @info "Running lineaized power flow across $(sys.nT) time periods."

    # loop over time
    for tii in prm.ts.time_keys

        # initialize
        init_pf = true
        run_pf  = true
        pf_cnt  = 0

        # 1. update the ideal dispatch point (active power) -- we do this just once
        quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

        # 2. update the injection bounds (upper and lower P/Q bounds) -- no longer needed
        quasiGrad.get_injection_bounds!(idx, msc, prm, stt, sys, tii)

        # 3. update y_bus and Jacobian and bias point -- this
        #    only needs to be done once per time, since xfm/shunt
        #    values are not changing between iterations
        Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

        # loop over pf solves
        while run_pf == true

            # increment
            pf_cnt += 1

            # first, rebuild the jacobian, and update the
            # base points: msc[:pinj0], msc[:qinj0]
            Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
            
            # quiet down!!!
            empty!(model)
            set_silent(model)
            quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])
            @variable(model, x_out[1:2*sys.nb])

            # assign
            dvm   = x_in[1:sys.nb]
            dva   = x_in[(sys.nb+1):end]
            dpinj = x_out[1:sys.nb]
            dqinj = x_out[(sys.nb+1):end]
            #
            # note:
            # vm   = vm0   + dvm
            # va   = va0   + dva
            # pinj = pinj0 + dpinj
            # qinj = qinj0 + dqinj
            #
            # key equation:
            #                       dPQ .== Jac*dVT
            #                       dPQ + basePQ(v) = devicePQ
            #
            #                       Jac*dVT + basePQ(v) == devicePQ
            #
            # so, we don't actually need to model dPQ explicitly (cool)
            #
            # so, the optimizer asks, how shall we tune dVT in order to produce a power perurbation
            # which, when added to the base point, lives inside the feasible device region?
            #
            # based on the result, we only have to actually update the device set points on the very
            # last power flow iteration, where we have converged.

            # now, model all nodal injections from the device/dc line side, all put in nodal_p/q
            nodal_p = Vector{AffExpr}(undef, sys.nb)
            nodal_q = Vector{AffExpr}(undef, sys.nb)
            for bus in 1:sys.nb
                nodal_p[bus] = AffExpr(0.0)
                nodal_q[bus] = AffExpr(0.0)
            end

            # create a flow variable for each dc line and sum these into the nodal vectors
            if sys.nldc == 0
                # nothing to see here
            else

                # define dc variables
                @variable(model, pdc_vars[1:sys.nldc])    # oriented so that fr = + !!
                @variable(model, qdc_fr_vars[1:sys.nldc])
                @variable(model, qdc_to_vars[1:sys.nldc])

                # bound dc power
                @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
                @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
                @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

                # now, we need to loop and set the affine expressions to 0, and then add powers
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                for bus in 1:sys.nb
                    # sum over line powers
                    nodal_p[bus] -= sum(pdc_vars[idx.bus_is_dc_frs[bus]];    init=0.0)
                    nodal_p[bus] += sum(pdc_vars[idx.bus_is_dc_tos[bus]];    init=0.0)
                    nodal_q[bus] -= sum(qdc_fr_vars[idx.bus_is_dc_frs[bus]]; init=0.0)
                    nodal_q[bus] -= sum(qdc_to_vars[idx.bus_is_dc_tos[bus]]; init=0.0)
                end
            end
            
            # next, deal with devices
            @variable(model, dev_p_vars[1:sys.ndev])
            @variable(model, dev_q_vars[1:sys.ndev])

            # call the bounds
            dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
            dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
            dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
            dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]

            # first, define p_on at this time
            p_on = dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii]

            # bound
            @constraint(model, dev_plb .<= p_on       .<= dev_pub)
            @constraint(model, dev_qlb .<= dev_q_vars .<= dev_qub)

            # apply additional bounds: J_pqe (equality constraints)
            if ~isempty(idx.J_pqe)
                @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[dev].*stt[:u_sum][tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe].*dev_p_vars[idx.J_pqe])
            end

            # apply additional bounds: J_pqmin/max (inequality constraints)
            #
            # note: when the reserve products are negelected, pr and cs constraints are the same
            #   remember: idx.J_pqmax == idx.J_pqmin
            if ~isempty(idx.J_pqmax)
                @constraint(model, dev_q_vars[idx.J_pqmax] .<= prm.dev.q_0_ub[idx.J_pqmax]*stt[:u_sum][tii][idx.J_pqmax] + prm.dev.beta_ub[idx.J_pqmax]*dev_p_vars[idx.J_pqmax])
                @constraint(model, prm.dev.q_0_lb[idx.J_pqmax]*stt[:u_sum][tii][idx.J_pqmax] + prm.dev.beta_lb[idx.J_pqmax]*dev_p_vars[idx.J_pqmax] .<= dev_q_vars[idx.J_pqmax])
            end

            # great, now just update the nodal injection vectors
            for bus in 1:sys.nb
                # sum over line powers
                nodal_p[bus] += sum(dev_p_vars[idx.pr[bus]]; init=0.0)
                nodal_p[bus] -= sum(dev_p_vars[idx.cs[bus]]; init=0.0)
                nodal_q[bus] += sum(dev_q_vars[idx.pr[bus]]; init=0.0)
                nodal_q[bus] -= sum(dev_q_vars[idx.cs[bus]]; init=0.0)
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm   .<= prm.bus.vm_ub)

            # mapping
            noref_Jac = @view Jac[:,[1:sys.nb; (sys.nb+2):end]]





            # loop, bound, and apply
            for dev in 1:sys.ndev

                # first, define p_on at this time
                p_on = dev_p_vars[dev] - stt[:p_su][tii] - stt[:p_sd][tii]

                # next, bound the active power: p_on
                @constraint(model, -prm.dc.pdc_ub    .<= p_on    .<= prm.dc.pdc_ub)







            nodal_qdc_fr = Vector{AffExpr}(undef, sys.nb)
            nodal_qdc_to = Vector{AffExpr}(undef, sys.nb)
            nodal_p_pr = Vector{AffExpr}(undef, sys.nb)
            nodal_p_cs = Vector{AffExpr}(undef, sys.nb)
            nodal_q_pr = Vector{AffExpr}(undef, sys.nb)
            nodal_q_cs = Vector{AffExpr}(undef, sys.nb)
            for bus in 1:sys.nb
                nodal_pdc_fr[bus]  = AffExpr(0.0)
                nodal_pdc_to[bus]  = AffExpr(0.0)
                nodal_qdc_fr[bus]  = AffExpr(0.0)
                nodal_qdc_to[bus]  = AffExpr(0.0)
            end


            # create a active power flow variable for each line and sum these into a nodal vector
            if sys.nldc == 0
                # nothing to see here
                nodal_pdc_fr = Vector{AffExpr}(undef, sys.nb)
                nodal_pdc_to = Vector{AffExpr}(undef, sys.nb)
                nodal_qdc_fr = Vector{AffExpr}(undef, sys.nb)
                nodal_qdc_to = Vector{AffExpr}(undef, sys.nb)
                for bus in 1:sys.nb
                    nodal_pdc_fr[bus]  = AffExpr(0.0)
                    nodal_pdc_to[bus]  = AffExpr(0.0)
                    nodal_qdc_fr[bus]  = AffExpr(0.0)
                    nodal_qdc_to[bus]  = AffExpr(0.0)
                end
            else
                # deal with the lines
                nodal_pdc_fr = Vector{AffExpr}(undef, sys.nb)
                nodal_pdc_to = Vector{AffExpr}(undef, sys.nb)
                nodal_qdc_fr = Vector{AffExpr}(undef, sys.nb)
                nodal_qdc_to = Vector{AffExpr}(undef, sys.nb)

                # define dc variables
                @variable(model, pdc_vars[1:sys.nldc])    # oriented so that fr = + !!
                @variable(model, qdc_fr_vars[1:sys.nldc])
                @variable(model, qdc_to_vars[1:sys.nldc])

                # bound dc power
                @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
                @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
                @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

                # now, we need to loop and set the affine expressions to 0, and then add powers
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                for bus in 1:sys.nb
                    nodal_pdc_fr[bus]  = AffExpr(0.0)
                    nodal_pdc_to[bus]  = AffExpr(0.0)
                    nodal_qdc_fr[bus]  = AffExpr(0.0)
                    nodal_qdc_to[bus]  = AffExpr(0.0)

                    # sum over line powers
                    nodal_pdc_fr[bus] += sum(pdc_vars[idx.bus_is_dc_frs[bus]];    init=0.0)
                    nodal_pdc_to[bus] -= sum(pdc_vars[idx.bus_is_dc_tos[bus]];    init=0.0)
                    nodal_qdc_fr[bus] += sum(qdc_fr_vars[idx.bus_is_dc_frs[bus]]; init=0.0)
                    nodal_qdc_to[bus] += sum(qdc_to_vars[idx.bus_is_dc_tos[bus]]; init=0.0)
                end
            end

            # next, deal with devices -- 
            nodal_p_pr = Vector{AffExpr}(undef, sys.nb)
            nodal_p_cs = Vector{AffExpr}(undef, sys.nb)
            nodal_q_pr = Vector{AffExpr}(undef, sys.nb)
            nodal_q_cs = Vector{AffExpr}(undef, sys.nb)


            # create P and Q power variables for constant power factor devices -- pr
            # 
            # 
            pr_and_Jpqe
            cs_and_Jpqe

            dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
            dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
            dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
            dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]


            nJpqe_pr = length(idx.pr_and_Jpqe)
            if nJpqe_pr == 0
                # nothing to see here
                nodal_p_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                nodal_q_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                for bus in 1:sys.nb
                    nodal_p_Jpqe_pr[bus] = AffExpr(0.0)
                    nodal_q_Jpqe_pr[bus] = AffExpr(0.0)
                end
            else

                # define an active power variable, but let q be implicit
                @variable(model, p_Jpqe_pr_vars[1:sys.nldc])
                q_Jpqe_pr_vars = Vector{AffExpr}(undef, nJpqe_pr)

                # loop and bound
                dev_cnt = 1
                for dev in idx.pr_and_Jpqe
                    # first, bound p_Jpqe_pr_vars
                    @constraint(model, p_Jpqe_pr_vars[dev])
                    # second, implicitly define q_Jpqe_pr_vars via equality
                    q_Jpqe_pr_vars[dev_cnt] = AffExpr(0.0)

                    # finally, bound q_Jpqe_pr_vars
                    q_Jpqe_pr[dev_cnt] += prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]
                    dev_cnt += 1
                end

                # compute reactive power
                stt[:dev_q][tii][dev] = prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]




                @variable(model, q_Jpqe_pr_vars[1:nJpqe_pr])

                # bound these variables
                @constraint(model, dev_qlb[idx.pr_and_Jpqe] .<= q_Jpqe_pr_vars .<= dev_qub[idx.pr_and_Jpqe])

                # for beta devices, compute reactive power as a function of active power
                for dev in 1:sys.ndev
                    # only a subset of devices will have a reactive power equality constraint
                    if dev in idx.J_pqe
                        # the following (pr vs cs) are equivalent
                        if dev in idx.pr_devs
                            stt[:dev_q][tii][dev] = prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]
                        else
                            stt[:dev_q][tii][dev] = prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]
                        end
                    end
                    
                    # now, populate the nodal vectors
                    nodal_p_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                    nodal_q_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                    for bus in 1:sys.nb
                        nodal_p_Jpqe_pr[bus]  = AffExpr(0.0)
                        nodal_p_Jpqe_pr[bus] += sum(...)

                        nodal_q_Jpqe_pr[bus] = AffExpr(0.0)
                        nodal_q_Jpqe_pr[bus] += sum(...)
                    end
                    
                end
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm   .<= prm.bus.vm_ub)

            # bound variables -- active power -- must include dc line constraints
            @constraint(model, msc[:plb]     .<= msc[:pinj0]   + dpinj + nodal_p_Jpqe_pr - nodal_p_Jpqe_cs - nodal_pdc_fr - nodal_pdc_to .<= msc[:pub])

            # bound variables -- reactive power -- must include beta equality links between P and Q
            @constraint(model, msc[:qlb]     .<= msc[:qinj0]   + dqinj + nodal_q_Jpqe_pr - nodal_q_Jpqe_cs .<= msc[:qub])

            # mapping
            noref_Jac = @view Jac[:,[1:sys.nb; (sys.nb+2):end]]
            @constraint(model, x_out .== noref_Jac*x_in)


            # objective: hold p (and v?) close to its initial value
            # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
            if qG.Gurobi_pf_obj == "min_dispatch_distance"
                # this finds a solution close to the dispatch point -- does not converge without v,a regularization
                obj    = AffExpr(0.0)
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                for bus in 1:sys.nb
                    tmp = @variable(model)
                    @constraint(model, msc[:pinj_ideal][bus] - (dpinj[bus] + msc[:pinj0][bus]) <= tmp)
                    @constraint(model, (dpinj[bus] + msc[:pinj0][bus]) - msc[:pinj_ideal][bus] <= tmp)
                    add_to_expression!(obj, tmp)

                    # voltage regularization
                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    # phase regularization
                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end
                end

                # this adds light regularization and causes convergence
                add_to_expression!(obj, tmp_vm)
                add_to_expression!(obj, tmp_va)

            elseif qG.Gurobi_pf_obj == "min_dispatch_perturbation"
                # this finds a solution with minimum movement -- not really needed
                # now that "min_dispatch_distance" converges
                tmp_p  = @variable(model)
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                for bus in 1:sys.nb
                    #tmp = @variable(model)
                    @constraint(model, -dpinj[bus] <= tmp_p)
                    @constraint(model,  dpinj[bus] <= tmp_p)

                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end
                    # for l1 norm: add_to_expression!(obj, tmp)
                end
                obj = tmp_p + tmp_vm + tmp_va
            else
                @warn "pf solver objective not recognized!"
            end

            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # take the norm of dv
            norm_dv = quasiGrad.norm(value.(dvm))
            
            # println("========================================================")
            println(termination_status(model),". ",primal_status(model),". objective value: ", round(objective_value(model), sigdigits = 5), "dv norm: ", round(norm_dv, sigdigits = 5))
            # println("========================================================")

            # now, update the state vector with the soluion
            stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
            stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

            # shall we terminate?
            if (norm_dv < 1e-3) || (pf_cnt == qG.max_linear_pfs)
                run_pf = false
            end
        end

    end
end

# get these at the bus level, too..
for bus = 1:sys.nb
    # are the devices consumers or producers?
    pr_devs_on_bus = dev_on_bus_inds[in.(dev_on_bus_inds,Ref(pr_inds))]
    cs_devs_on_bus = dev_on_bus_inds[in.(dev_on_bus_inds,Ref(cs_inds))]
end

# solve power flow (to some degree of accuracy)
#=
function solve_linear_pf_with_Gurobi_simple_bounds!(Jac::quasiGrad.SparseArrays.SparseMatrixCSC{Float64, Int64}, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG,  stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol)
    # ask Gurobi to solve a linearize power flow
    #
    # here is power balance:
    #
    # p_pr - p_cs - pdc - () = p_lines => this is typical.
    #
    # vm0 = stt[:vm][tii]
    # va0 = stt[:va][tii][2:end-1]
    #
    # bias point: msc[:pinj0, qinj0] === Y * stt[:vm, va]
    qG.Gurobi_pf_obj            = "min_dispatch_distance"
    qG.compute_pf_injs_with_Jac = true

    # build and empty the model!
    model = Model(Gurobi.Optimizer)
    @info "Running lineaized power flow across $(sys.nT) time periods."

    # loop over time
    for tii in prm.ts.time_keys

        # initialize
        init_pf = true
        run_pf  = true
        pf_cnt  = 0

        # 1. update the ideal dispatch point (active power)
        quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

        # 2. update the injection bounds (upper and lower P/Q bounds)
        quasiGrad.get_injection_bounds!(idx, msc, prm, stt, sys, tii)

        # 3. update y_bus and Jacobian and bias point -- this
        #    only needs to be done once per time, since xfm/shunt
        #    values are not changing between iterations
        Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

        # loop over pf solves
        while run_pf == true

            # increment
            pf_cnt += 1

            # first, rebuild the jacobian, and update the
            # base points: msc[:pinj0], msc[:qinj0]
            Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
            
            # quiet down!!!
            empty!(model)
            set_silent(model)
            quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])
            @variable(model, x_out[1:2*sys.nb])

            # assign
            dvm   = x_in[1:sys.nb]
            dva   = x_in[(sys.nb+1):end]
            dpinj = x_out[1:sys.nb]
            dqinj = x_out[(sys.nb+1):end]
            #
            # note:
            # vm   = vm0   + dvm
            # va   = va0   + dva
            # pinj = pinj0 + dpinj
            # qinj = qinj0 + dqinj

            # create a active power flow variable for each line and sum these into a nodal vector
            if sys.nldc == 0
                # nothing to see here
                nodal_pdc_fr = Vector{AffExpr}(undef, sys.nb)
                nodal_pdc_to = Vector{AffExpr}(undef, sys.nb)
                for bus in 1:sys.nb
                    nodal_pdc_fr[bus]  = AffExpr(0.0)
                    nodal_pdc_to[bus]  = AffExpr(0.0)
                end
            else
                # deal with the lines
                nodal_pdc_fr = Vector{AffExpr}(undef, sys.nb)
                nodal_pdc_to = Vector{AffExpr}(undef, sys.nb)
                @variable(model, pdc_vars[1:sys.nldc]) # oriented so that fr = + !!

                # bound dc power
                @constraint(model, -prm.dc.pdc_ub .<= pdc_vars .<= prm.dc.pdc_ub)

                # now, we need to loop and set the affine expressions to 0
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                for bus in 1:sys.nb
                    nodal_pdc_fr[bus]  = AffExpr(0.0)
                    nodal_pdc_to[bus]  = AffExpr(0.0)
                    nodal_pdc_fr[bus] += sum(pdc_vars[idx.bus_is_dc_frs[bus]]; init=0.0)
                    nodal_pdc_to[bus] -= sum(pdc_vars[idx.bus_is_dc_tos[bus]]; init=0.0)
                end
            end

            # create P and Q power variables for constant power factor devices -- pr
            # 
            # 
            pr_and_Jpqe
            cs_and_Jpqe

            dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
            dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
            dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
            dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]


            nJpqe_pr = length(idx.pr_and_Jpqe)
            if nJpqe_pr == 0
                # nothing to see here
                nodal_p_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                nodal_q_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                for bus in 1:sys.nb
                    nodal_p_Jpqe_pr[bus] = AffExpr(0.0)
                    nodal_q_Jpqe_pr[bus] = AffExpr(0.0)
                end
            else

                # define an active power variable, but let q be implicit
                @variable(model, p_Jpqe_pr_vars[1:sys.nldc])
                q_Jpqe_pr_vars = Vector{AffExpr}(undef, nJpqe_pr)

                # loop and bound
                dev_cnt = 1
                for dev in idx.pr_and_Jpqe
                    # first, bound p_Jpqe_pr_vars
                    @constraint(model, p_Jpqe_pr_vars[dev])
                    # second, implicitly define q_Jpqe_pr_vars via equality
                    q_Jpqe_pr_vars[dev_cnt] = AffExpr(0.0)

                    # finally, bound q_Jpqe_pr_vars
                    q_Jpqe_pr[dev_cnt] += prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]
                    dev_cnt += 1
                end

                # compute reactive power
                stt[:dev_q][tii][dev] = prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]




                @variable(model, q_Jpqe_pr_vars[1:nJpqe_pr])

                # bound these variables
                @constraint(model, dev_qlb[idx.pr_and_Jpqe] .<= q_Jpqe_pr_vars .<= dev_qub[idx.pr_and_Jpqe])

                # for beta devices, compute reactive power as a function of active power
                for dev in 1:sys.ndev
                    # only a subset of devices will have a reactive power equality constraint
                    if dev in idx.J_pqe
                        # the following (pr vs cs) are equivalent
                        if dev in idx.pr_devs
                            stt[:dev_q][tii][dev] = prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]
                        else
                            stt[:dev_q][tii][dev] = prm.dev.q_0[dev]*stt[:u_sum][tii][dev] + prm.dev.beta[dev]*stt[:dev_p][tii][dev]
                        end
                    end
                    
                    # now, populate the nodal vectors
                    nodal_p_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                    nodal_q_Jpqe_pr = Vector{AffExpr}(undef, sys.nb)
                    for bus in 1:sys.nb
                        nodal_p_Jpqe_pr[bus]  = AffExpr(0.0)
                        #nodal_p_Jpqe_pr[bus] += sum(...)

                        nodal_q_Jpqe_pr[bus] = AffExpr(0.0)
                        #nodal_q_Jpqe_pr[bus] += sum(...)
                    end
                    
                end
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm   .<= prm.bus.vm_ub)

            # bound variables -- active power -- must include dc line constraints
            @constraint(model, msc[:plb]     .<= msc[:pinj0]   + dpinj + nodal_p_Jpqe_pr - nodal_p_Jpqe_cs - nodal_pdc_fr - nodal_pdc_to .<= msc[:pub])

            # bound variables -- reactive power -- must include beta equality links between P and Q
            @constraint(model, msc[:qlb]     .<= msc[:qinj0]   + dqinj + nodal_q_Jpqe_pr - nodal_q_Jpqe_cs .<= msc[:qub])

            # mapping
            noref_Jac = @view Jac[:,[1:sys.nb; (sys.nb+2):end]]
            @constraint(model, x_out .== noref_Jac*x_in)


            # objective: hold p (and v?) close to its initial value
            # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
            if qG.Gurobi_pf_obj == "min_dispatch_distance"
                # this finds a solution close to the dispatch point -- does not converge without v,a regularization
                obj    = AffExpr(0.0)
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                for bus in 1:sys.nb
                    tmp = @variable(model)
                    @constraint(model, msc[:pinj_ideal][bus] - (dpinj[bus] + msc[:pinj0][bus]) <= tmp)
                    @constraint(model, (dpinj[bus] + msc[:pinj0][bus]) - msc[:pinj_ideal][bus] <= tmp)
                    add_to_expression!(obj, tmp)

                    # voltage regularization
                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    # phase regularization
                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end
                end

                # this adds light regularization and causes convergence
                add_to_expression!(obj, tmp_vm)
                add_to_expression!(obj, tmp_va)

            elseif qG.Gurobi_pf_obj == "min_dispatch_perturbation"
                # this finds a solution with minimum movement -- not really needed
                # now that "min_dispatch_distance" converges
                tmp_p  = @variable(model)
                tmp_vm = @variable(model)
                tmp_va = @variable(model)
                for bus in 1:sys.nb
                    #tmp = @variable(model)
                    @constraint(model, -dpinj[bus] <= tmp_p)
                    @constraint(model,  dpinj[bus] <= tmp_p)

                    @constraint(model, -dvm[bus] <= tmp_vm)
                    @constraint(model,  dvm[bus] <= tmp_vm)

                    if bus > 1
                        @constraint(model, -dva[bus-1] <= tmp_va)
                        @constraint(model,  dva[bus-1] <= tmp_va)
                    end
                    # for l1 norm: add_to_expression!(obj, tmp)
                end
                obj = tmp_p + tmp_vm + tmp_va
            else
                @warn "pf solver objective not recognized!"
            end

            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # take the norm of dv
            norm_dv = quasiGrad.norm(value.(dvm))
            
            # println("========================================================")
            println(termination_status(model),". ",primal_status(model),". objective value: ", round(objective_value(model), sigdigits = 5), "dv norm: ", round(norm_dv, sigdigits = 5))
            # println("========================================================")

            # now, update the state vector with the soluion
            stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
            stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

            # shall we terminate?
            if (norm_dv < 1e-3) || (pf_cnt == qG.max_linear_pfs)
                run_pf = false
            end
        end

        # apply the updated injections to the devices
    end
end
=#

# note -- this is ALWAYS run after solve_economic_dispatch()
#
# no longer needed!
function apply_economic_dispatch_projection!()
    @warn "Deprecated."

    stt[:u_on_dev]  = deepcopy(GRB[:u_on_dev])
    stt[:p_on]      = deepcopy(GRB[:p_on])
    stt[:dev_q]     = deepcopy(GRB[:dev_q])
    stt[:p_rgu]     = deepcopy(GRB[:p_rgu])
    stt[:p_rgd]     = deepcopy(GRB[:p_rgd])
    stt[:p_scr]     = deepcopy(GRB[:p_scr])
    stt[:p_nsc]     = deepcopy(GRB[:p_nsc])
    stt[:p_rru_on]  = deepcopy(GRB[:p_rru_on])
    stt[:p_rru_off] = deepcopy(GRB[:p_rru_off])
    stt[:p_rrd_on]  = deepcopy(GRB[:p_rrd_on])
    stt[:p_rrd_off] = deepcopy(GRB[:p_rrd_off])
    stt[:q_qru]     = deepcopy(GRB[:q_qru])
    stt[:q_qrd]     = deepcopy(GRB[:q_qrd])

    # update the u_sum and powers (used in clipping, so must be correct!)
    qG.run_susd_updates = true
    quasiGrad.simple_device_statuses!(idx, prm, qG, stt)
    quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
end

                # cg has failed in the past -- not sure why -- test for NaN
                if isnan(sum(ctb[t_ind])) || maximum(ctb[t_ind]) > 1e7  # => faster than any(isnan, t)
                    # LU backup
                    @info "Krylov failed -- using LU backup (ctg flows)!"
                    ctb[t_ind] = ntk.Ybr\c
                end

                function adam_line_search_initialization!(t)
                    # in this function, we find an optimzal step size for the first set of
                    # gradients, and then we "hot start" adam's momentum states
                    # loop over the keys in mgd
                    for var_key in keys(mgd)
                
                        # initialize
                        step = 1e-14
                        step_size_search = true
                
                        # evaluate the penalized market surplus value
                        z0 = f()
                
                        # copy the initial state
                        stt_vk0 = deepcopy(stt[var_key])
                        
                        while step_size_search == true
                            # loop over all time
                            for tii in prm.ts.time_keys
                                # states to update                                            
                                if var_key in keys(upd)
                                    update_subset = upd[var_key][tii]
                                    stt[var_key][tii][update_subset] = stt_vk0[tii][update_subset] - step*mgd[var_key][tii][update_subset]
                                else
                                    stt[var_key][tii] = stt_vk0[tii] - step*mgd[var_key][tii]
                                end
                
                                # now, test the penalized market surplus value
                                z_new = f()
                
                                if (z_new > z0) && (step <= 1e-9)
                                    step = step*10
                                elseif (z_new > z0) && (1e-9 < step)
                                    step = step*5.0
                                else
                                    step = step/2.0
                                end
                            end
                        end
                
                        # now that we have terminated for this variable, let's
                        # initialize the momentum states 
                    end  
                end
# cleanup reserve variables, mostly
function centralized_reserve_cleanup!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # this is, necessarily, a centralized (across devices) optimziation problem.
    #
    @warn "The centralized solver is 40% slower than the time-distributed solver (reserve_cleanup!)."
    #
    # build and empty the model!
    model = Model(Gurobi.Optimizer; add_bridges = false)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # set model properties
    set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)

    # define local time keys
    tkeys = prm.ts.time_keys

    # define the minimum set of variables we will need to solve the constraints
    p_rgu     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_rgu][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rgd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_rgd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_scr     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_scr][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_nsc     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_nsc][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rru_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_on_t$(ii)",  [dev = 1:sys.ndev], start=stt[:p_rru_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rru_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_off_t$(ii)", [dev = 1:sys.ndev], start=stt[:p_rru_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rrd_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_on_t$(ii)",  [dev = 1:sys.ndev], start=stt[:p_rrd_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rrd_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_off_t$(ii)", [dev = 1:sys.ndev], start=stt[:p_rrd_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT))
    q_qru     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qru_t$(ii)",     [dev = 1:sys.ndev], start=stt[:q_qru][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    q_qrd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qrd_t$(ii)",     [dev = 1:sys.ndev], start=stt[:q_qrd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))

    # add scoring variables and affine terms
    p_rgu_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgd_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_scr_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_nsc_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgu_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_scr_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_nsc_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    q_qru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qru_zonal_penalty_t$(ii)", [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT))
    q_qrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qrd_zonal_penalty_t$(ii)", [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT))
    
    # affine aggregation terms
    zt = AffExpr(0.0)

    # loop over all devices
    for dev in 1:sys.ndev

        # loop over each time period and define the hard constraints
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # duration
            dt = prm.ts.duration[tii]

            # 1. Minimum downtime: zhat_mndn
            # 2. Minimum uptime: zhat_mnup
            # 3. Ramping limits (up): zhat_rup
            # 4. Ramping limits (down): zhat_rd

            # 5. Regulation up: zhat_rgu
            @constraint(model, p_rgu[tii][dev] - prm.dev.p_reg_res_up_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[tii][dev] - prm.dev.p_reg_res_down_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] - prm.dev.p_syn_res_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt[:u_on_dev][tii][dev] <= 0.0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt[:u_on_dev][tii][dev]) <= 0.0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, stt[:p_on][tii][dev] + p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= 0.0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rrd_on[tii][dev] + p_rgd[tii][dev] - stt[:p_on][tii][dev] <= 0.0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, stt[:dev_q][tii][dev] + q_qru[tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= 0.0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= 0.0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt[:dev_q][tii][dev] + q_qru[tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= 0.0)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev] + 
                        prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev] + 
                        q_qrd[tii][dev] - stt[:dev_q][tii][dev] <= 0.0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, stt[:p_on][tii][dev] + p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= 0.0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rru_on[tii][dev] + p_scr[tii][dev] + p_rgu[tii][dev] - stt[:p_on][tii][dev] <= 0.0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_rrd_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= 0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, stt[:dev_q][tii][dev] + q_qrd[tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= 0.0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= 0.0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, stt[:dev_q][tii][dev] + q_qrd[tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= 0.0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev]
                    + prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev]
                    + q_qru[tii][dev] - stt[:dev_q][tii][dev] <= 0.0)
                end
            end
        end

        # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "upd"
        # upd = update states
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # if a device is *not* in the set of variables, then it must be held constant!

            if dev ∉ upd[:p_rrd_off][tii]
                @constraint(model, p_rrd_off[tii][dev] == stt[:p_rrd_off][tii][dev])
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[tii][dev] == stt[:p_nsc][tii][dev])
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[tii][dev] == stt[:p_rru_off][tii][dev])
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[tii][dev] == stt[:q_qru][tii][dev])
            end

            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[tii][dev] == stt[:q_qrd][tii][dev])
            end
        end
    end

    # loop over reserves
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # for the "endogenous" reserve requirements
        rgu_sigma = prm.reserve.rgu_sigma
        rgd_sigma = prm.reserve.rgd_sigma 
        scr_sigma = prm.reserve.scr_sigma 
        nsc_sigma = prm.reserve.nsc_sigma  

        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == rgu_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == rgd_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, scr_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[tii][zone])
                @constraint(model, nsc_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[tii][zone])
            end

            # balance equations -- compute the shortfall values
            #
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_pzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] <= p_rgu_zonal_penalty[tii][zone])
                
                @constraint(model, p_rgd_zonal_REQ[tii][zone] <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][t_ind] <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] <= p_rrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] - 
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgu_zonal_penalty[tii][zone])

                @constraint(model, p_rgd_zonal_REQ[tii][zone] - 
                                sum(p_rgd[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) - 
                                sum(p_nsc[tii][dev] for dev in idx.dev_pzone[zone]) <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][t_ind] -
                                sum(p_rru_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] -
                                sum(p_rrd_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[tii][zone])
            end
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_qzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] <= q_qrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] -
                                sum(q_qru[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] -
                                sum(q_qrd[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[tii][zone])
            end
        end
    end

    # loop -- NOTE -- we are not including start-up-state discounts -- not worth it
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]
        
        # add up
        zt_temp = 
            # local reserve penalties
            sum(dt*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*p_rgu[tii]) -   # zrgu
            sum(dt*prm.dev.p_reg_res_down_cost_tmdv[t_ind].*p_rgd[tii]) - # zrgd
            sum(dt*prm.dev.p_syn_res_cost_tmdv[t_ind].*p_scr[tii]) -      # zscr
            sum(dt*prm.dev.p_nsyn_res_cost_tmdv[t_ind].*p_nsc[tii]) -     # znsc
            sum(dt*(prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind].*p_rru_on[tii] +
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind].*p_rru_off[tii])) -   # zrru
            sum(dt*(prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind].*p_rrd_on[tii] +
                    prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind].*p_rrd_off[tii])) - # zrrd
            sum(dt*prm.dev.q_res_up_cost_tmdv[t_ind].*q_qru[tii]) -   # zqru      
            sum(dt*prm.dev.q_res_down_cost_tmdv[t_ind].*q_qrd[tii]) - # zqrd
            # zonal reserve penalties (P)
            sum(dt*prm.vio.rgu_zonal.*p_rgu_zonal_penalty[tii]) -
            sum(dt*prm.vio.rgd_zonal.*p_rgd_zonal_penalty[tii]) -
            sum(dt*prm.vio.scr_zonal.*p_scr_zonal_penalty[tii]) -
            sum(dt*prm.vio.nsc_zonal.*p_nsc_zonal_penalty[tii]) -
            sum(dt*prm.vio.rru_zonal.*p_rru_zonal_penalty[tii]) -
            sum(dt*prm.vio.rrd_zonal.*p_rrd_zonal_penalty[tii]) -
            # zonal reserve penalties (Q)
            sum(dt*prm.vio.qru_zonal.*q_qru_zonal_penalty[tii]) -
            sum(dt*prm.vio.qrd_zonal.*q_qrd_zonal_penalty[tii])

        # update zt
        add_to_expression!(zt, zt_temp)
    end

    # set the objective
    @objective(model, Max, zt)

    # solve
    optimize!(model)

    # test solution!
    soln_valid = solution_status(model)

    # did Gurobi find something valid?
    if soln_valid == true
        println("========================================================")
        println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
        println("========================================================")

        # solve, and then return the solution
        for tii in prm.ts.time_keys
            stt[:p_rgu][tii]     = copy(value.(p_rgu[tii]))
            stt[:p_rgd][tii]     = copy(value.(p_rgd[tii]))
            stt[:p_scr][tii]     = copy(value.(p_scr[tii]))
            stt[:p_nsc][tii]     = copy(value.(p_nsc[tii]))
            stt[:p_rru_on][tii]  = copy(value.(p_rru_on[tii]))
            stt[:p_rru_off][tii] = copy(value.(p_rru_off[tii]))
            stt[:p_rrd_on][tii]  = copy(value.(p_rrd_on[tii]))
            stt[:p_rrd_off][tii] = copy(value.(p_rrd_off[tii]))
            stt[:q_qru][tii]     = copy(value.(q_qru[tii]))
            stt[:q_qrd][tii]     = copy(value.(q_qrd[tii]))
        end
    else
        # warn!
        @warn "Reserve cleanup solver (LP) failed -- skip cleanup!"
        println(termination_status(model))
    end
end

# cleanup reserve variables, mostly
function centralized_soft_reserve_cleanup!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # this is, necessarily, a centralized optimziation problem.
    #
    @warn "The centralized solver is 40% slower than the time-distributed solver (soft_reserve_cleanup!)."
    #
    # build and empty the model!
    model = Model(Gurobi.Optimizer; add_bridges = false)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # define local time keys
    tkeys = prm.ts.time_keys

    # define the minimum set of variables we will need to solve the constraints
    p_rgu     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_rgu][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rgd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_rgd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_scr     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_scr][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_nsc     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_t$(ii)",     [dev = 1:sys.ndev], start=stt[:p_nsc][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rru_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_on_t$(ii)",  [dev = 1:sys.ndev], start=stt[:p_rru_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rru_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_off_t$(ii)", [dev = 1:sys.ndev], start=stt[:p_rru_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rrd_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_on_t$(ii)",  [dev = 1:sys.ndev], start=stt[:p_rrd_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT))
    p_rrd_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_off_t$(ii)", [dev = 1:sys.ndev], start=stt[:p_rrd_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT))
    q_qru     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qru_t$(ii)",     [dev = 1:sys.ndev], start=stt[:q_qru][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))
    q_qrd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qrd_t$(ii)",     [dev = 1:sys.ndev], start=stt[:q_qrd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT))

    # add scoring variables and affine terms
    p_rgu_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgd_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_scr_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_nsc_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_zonal_REQ_t$(ii)",     [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgu_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgu_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rgd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rgd_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_scr_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_scr_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_nsc_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_nsc_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rru_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    p_rrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "p_rrd_zonal_penalty_t$(ii)", [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT))
    q_qru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qru_zonal_penalty_t$(ii)", [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT))
    q_qrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, base_name = "q_qrd_zonal_penalty_t$(ii)", [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT))
    
    # affine aggregation terms
    zt             = AffExpr(0.0)
    z_penalty      = AffExpr(0.0)
    penalty_scalar = 1e6

    # loop over all devices
    for dev in 1:sys.ndev

        # loop over each time period and define the hard constraints
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # duration
            dt = prm.ts.duration[tii]

            # 1. Minimum downtime: zhat_mndn
            # 2. Minimum uptime: zhat_mnup
            # 3. Ramping limits (up): zhat_rup
            # 4. Ramping limits (down): zhat_rd

            # 5. Regulation up: zhat_rgu
            tmp_penalty_c5 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c5, penalty_scalar)
            @constraint(model, p_rgu[tii][dev] - prm.dev.p_reg_res_up_ub[dev]*stt[:u_on_dev][tii][dev] <= tmp_penalty_c5)

            # 6. Regulation down: zhat_rgd
            tmp_penalty_c6 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c6, penalty_scalar)
            @constraint(model, p_rgd[tii][dev] - prm.dev.p_reg_res_down_ub[dev]*stt[:u_on_dev][tii][dev] <= tmp_penalty_c6)

            # 7. Synchronized reserve: zhat_scr
            tmp_penalty_c7 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c7, penalty_scalar)
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] - prm.dev.p_syn_res_ub[dev]*stt[:u_on_dev][tii][dev] <= tmp_penalty_c7)

            # 8. Synchronized reserve: zhat_nsc
            tmp_penalty_c8 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c8, penalty_scalar)
            @constraint(model, p_nsc[tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= tmp_penalty_c8)

            # 9. Ramping reserve up (on): zhat_rruon
            tmp_penalty_c9 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c9, penalty_scalar)
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt[:u_on_dev][tii][dev] <= tmp_penalty_c9)

            # 10. Ramping reserve up (off): zhat_rruoff
            tmp_penalty_c10 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c10, penalty_scalar)
            @constraint(model, p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]) <= tmp_penalty_c10)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            tmp_penalty_c11 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c11, penalty_scalar)
            @constraint(model, p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt[:u_on_dev][tii][dev] <= tmp_penalty_c11)

            # 12. Ramping reserve down (off): zhat_rrdoff
            tmp_penalty_c12 = @variable(model, lower_bound = 0.0)
            add_to_expression!(z_penalty, tmp_penalty_c12, penalty_scalar)
            @constraint(model, p_rrd_off[tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt[:u_on_dev][tii][dev]) <= tmp_penalty_c12)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                tmp_penalty_c13pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c13pr, penalty_scalar)
                @constraint(model, stt[:p_on][tii][dev] + p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= tmp_penalty_c13pr)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                tmp_penalty_c14pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c14pr, penalty_scalar)
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rrd_on[tii][dev] + p_rgd[tii][dev] - stt[:p_on][tii][dev] <= tmp_penalty_c14pr)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                tmp_penalty_c15pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c15pr, penalty_scalar)
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= tmp_penalty_c15pr)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                tmp_penalty_c16pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c16pr, penalty_scalar)
                @constraint(model, stt[:dev_q][tii][dev] + q_qru[tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= tmp_penalty_c16pr)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                tmp_penalty_c17pr = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c17pr, penalty_scalar)
                @constraint(model, q_qrd[tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= tmp_penalty_c17pr)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    tmp_penalty_c18pr = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c18pr, penalty_scalar)
                    @constraint(model, stt[:dev_q][tii][dev] + q_qru[tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= tmp_penalty_c18pr)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    tmp_penalty_c19pr = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c19pr, penalty_scalar)
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev] + 
                        prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev] + 
                        q_qrd[tii][dev] - stt[:dev_q][tii][dev] <= tmp_penalty_c19pr)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                tmp_penalty_c13cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c13cs, penalty_scalar)
                @constraint(model, stt[:p_on][tii][dev] + p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev] <= tmp_penalty_c13cs)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                tmp_penalty_c14cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c14cs, penalty_scalar)
                @constraint(model, prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + p_rru_on[tii][dev] + p_scr[tii][dev] + p_rgu[tii][dev] - stt[:p_on][tii][dev] <= tmp_penalty_c14cs)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                tmp_penalty_c15cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c15cs, penalty_scalar)
                @constraint(model, stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + p_rrd_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]) <= tmp_penalty_c15cs)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                tmp_penalty_c16cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c16cs, penalty_scalar)
                @constraint(model, stt[:dev_q][tii][dev] + q_qrd[tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev] <= tmp_penalty_c16cs)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                tmp_penalty_c17cs = @variable(model, lower_bound = 0.0)
                add_to_expression!(z_penalty, tmp_penalty_c17cs, penalty_scalar)
                @constraint(model, q_qru[tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] <= tmp_penalty_c17cs)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    tmp_penalty_c18cs = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c18cs, penalty_scalar)
                    @constraint(model, stt[:dev_q][tii][dev] + q_qrd[tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev]
                    - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev] <= tmp_penalty_c18cs)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    tmp_penalty_c19cs = @variable(model, lower_bound = 0.0)
                    add_to_expression!(z_penalty, tmp_penalty_c19cs, penalty_scalar)
                    @constraint(model, prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev]
                    + prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev]
                    + q_qru[tii][dev] - stt[:dev_q][tii][dev] <= tmp_penalty_c19cs)
                end
            end
        end

        # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "upd"
        # upd = update states
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # if a device is *not* in the set of variables, then it must be held constant!

            if dev ∉ upd[:p_rrd_off][tii]
                @constraint(model, p_rrd_off[tii][dev] == stt[:p_rrd_off][tii][dev])
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[tii][dev] == stt[:p_nsc][tii][dev])
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[tii][dev] == stt[:p_rru_off][tii][dev])
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[tii][dev] == stt[:q_qru][tii][dev])
            end

            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[tii][dev] == stt[:q_qrd][tii][dev])
            end
        end
    end

    # loop over reserves
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # for the "endogenous" reserve requirements
        rgu_sigma = prm.reserve.rgu_sigma
        rgd_sigma = prm.reserve.rgd_sigma 
        scr_sigma = prm.reserve.scr_sigma 
        nsc_sigma = prm.reserve.nsc_sigma  

        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == rgu_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == rgd_sigma[zone]*sum(stt[:dev_p][tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, scr_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[tii][zone])
                @constraint(model, nsc_sigma[zone]*[stt[:dev_p][tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[tii][zone])
            end

            # balance equations -- compute the shortfall values
            #
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_pzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] <= p_rgu_zonal_penalty[tii][zone])
                
                @constraint(model, p_rgd_zonal_REQ[tii][zone] <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][t_ind] <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] <= p_rrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] - 
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgu_zonal_penalty[tii][zone])

                @constraint(model, p_rgd_zonal_REQ[tii][zone] - 
                                sum(p_rgd[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) - 
                                sum(p_nsc[tii][dev] for dev in idx.dev_pzone[zone]) <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][t_ind] -
                                sum(p_rru_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] -
                                sum(p_rrd_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[tii][zone])
            end
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_qzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] <= q_qrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] -
                                sum(q_qru[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] -
                                sum(q_qrd[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[tii][zone])
            end
        end
    end

    # loop -- NOTE -- we are not including start-up-state discounts -- not worth it
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]
        
        # add up
        zt_temp = 
            # local reserve penalties
            sum(dt*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*p_rgu[tii]) -   # zrgu
            sum(dt*prm.dev.p_reg_res_down_cost_tmdv[t_ind].*p_rgd[tii]) - # zrgd
            sum(dt*prm.dev.p_syn_res_cost_tmdv[t_ind].*p_scr[tii]) -      # zscr
            sum(dt*prm.dev.p_nsyn_res_cost_tmdv[t_ind].*p_nsc[tii]) -     # znsc
            sum(dt*(prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind].*p_rru_on[tii] +
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind].*p_rru_off[tii])) -   # zrru
            sum(dt*(prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind].*p_rrd_on[tii] +
                    prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind].*p_rrd_off[tii])) - # zrrd
            sum(dt*prm.dev.q_res_up_cost_tmdv[t_ind].*q_qru[tii]) -   # zqru      
            sum(dt*prm.dev.q_res_down_cost_tmdv[t_ind].*q_qrd[tii]) - # zqrd
            # zonal reserve penalties (P)
            sum(dt*prm.vio.rgu_zonal.*p_rgu_zonal_penalty[tii]) -
            sum(dt*prm.vio.rgd_zonal.*p_rgd_zonal_penalty[tii]) -
            sum(dt*prm.vio.scr_zonal.*p_scr_zonal_penalty[tii]) -
            sum(dt*prm.vio.nsc_zonal.*p_nsc_zonal_penalty[tii]) -
            sum(dt*prm.vio.rru_zonal.*p_rru_zonal_penalty[tii]) -
            sum(dt*prm.vio.rrd_zonal.*p_rrd_zonal_penalty[tii]) -
            # zonal reserve penalties (Q)
            sum(dt*prm.vio.qru_zonal.*q_qru_zonal_penalty[tii]) -
            sum(dt*prm.vio.qrd_zonal.*q_qrd_zonal_penalty[tii])

        # update zt
        add_to_expression!(zt, zt_temp)
    end

    # set the objective
    @objective(model, Max, zt - z_penalty)

    # solve
    optimize!(model)

    # test solution!
    soln_valid = solution_status(model)

    # did Gurobi find something valid?
    if soln_valid == true
        println("========================================================")
        println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
        println("========================================================")

        # solve, and then return the solution
        for tii in prm.ts.time_keys
            stt[:p_rgu][tii]     = copy(value.(p_rgu[tii]))
            stt[:p_rgd][tii]     = copy(value.(p_rgd[tii]))
            stt[:p_scr][tii]     = copy(value.(p_scr[tii]))
            stt[:p_nsc][tii]     = copy(value.(p_nsc[tii]))
            stt[:p_rru_on][tii]  = copy(value.(p_rru_on[tii]))
            stt[:p_rru_off][tii] = copy(value.(p_rru_off[tii]))
            stt[:p_rrd_on][tii]  = copy(value.(p_rrd_on[tii]))
            stt[:p_rrd_off][tii] = copy(value.(p_rrd_off[tii]))
            stt[:q_qru][tii]     = copy(value.(q_qru[tii]))
            stt[:q_qrd][tii]     = copy(value.(q_qrd[tii]))
        end
    else
        # warn!
        @warn "(softly constrained) Reserve cleanup solver (LP) failed -- skip cleanup!"
    end
end

# before softabs!!
            # reserve zones -- p
            for zone in 1:sys.nzP
                # g17 (zrgu_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgu_zonal] * cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone] #grd[:zrgu_zonal][:p_rgu_zonal_penalty][tii][zone]
                mgd_com = cgd.dzrgu_zonal_dp_rgu_zonal_penalty[tii][zone]
                if qG.reserve_grad_type == "standard"
                                                        # NOTE: minus
                    mgd[:p_rgu][tii][idx.dev_pzone[zone]] .-= mgd_com*sign(stt[:p_rgu_zonal_penalty][tii][zone])
                else
                # ===> requirements -- depend on active power consumption
                for dev in idx.cs_pzone[zone]
                    alpha = mgd_com*sign(stt[:p_rgu_zonal_penalty][tii][zone])*prm.reserve.rgu_sigma[zone]
                    dp_alpha!(grd, dev, tii, alpha)
                end

                # g18 (zrgd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrgd_zonal] * cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone] #grd[:zrgd_zonal][:p_rgd_zonal_penalty][tii][zone]
                mgd_com = cgd.dzrgd_zonal_dp_rgd_zonal_penalty[tii][zone]
                                                   # NOTE: minus
                mgd[:p_rgd][tii][idx.dev_pzone[zone]] .-= mgd_com*sign(stt[:p_rgd_zonal_penalty][tii][zone])
                # ===> requirements -- depend on active power consumption
                for dev in idx.cs_pzone[zone]
                    alpha = mgd_com*sign(stt[:p_rgd_zonal_penalty][tii][zone])*prm.reserve.rgd_sigma[zone]
                    dp_alpha!(grd, dev, tii, alpha)
                end

                # g19 (zscr_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zscr_zonal] * cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone] #grd[:zscr_zonal][:p_scr_zonal_penalty][tii][zone]
                mgd_com = cgd.dzscr_zonal_dp_scr_zonal_penalty[tii][zone]
                                                # NOTE: minus
                mgd[:p_rgu][tii][idx.dev_pzone[zone]] .-= mgd_com*sign(stt[:p_scr_zonal_penalty][tii][zone])
                mgd[:p_scr][tii][idx.dev_pzone[zone]] .-= mgd_com*sign(stt[:p_scr_zonal_penalty][tii][zone])
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(stt[:dev_p][tii][idx.pr_pzone[zone]])]
                    # ===> requirements -- depend on active power production/consumption!
                    for dev = i_pmax # we only take the derivative of the device which has the highest production
                        alpha = mgd_com*sign(stt[:p_scr_zonal_penalty][tii][zone])*prm.reserve.scr_sigma[zone]
                        dp_alpha!(grd, dev, tii, alpha)
                    end
                end
                if ~isempty(idx.cs_pzone[zone])
                    # only do the following if there are consumers here -- overly cautious
                    for dev in idx.cs_pzone[zone]
                        alpha = mgd_com*sign(stt[:p_scr_zonal_penalty][tii][zone])*prm.reserve.rgu_sigma[zone]
                        dp_alpha!(grd, dev, tii, alpha)
                    end
                end

                # g20 (znsc_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:znsc_zonal] * cgd.dznsc_zonal_dp_nsc_zonal_penalty[tii][zone] #grd[:znsc_zonal][:p_nsc_zonal_penalty][tii][zone]
                mgd_com = cgd.dznsc_zonal_dp_nsc_zonal_penalty[tii][zone]
                                                 # NOTE: minus
                mgd[:p_rgu][tii][idx.dev_pzone[zone]] .-= mgd_com*sign(stt[:p_nsc_zonal_penalty][tii][zone])
                mgd[:p_scr][tii][idx.dev_pzone[zone]] .-= mgd_com*sign(stt[:p_nsc_zonal_penalty][tii][zone])
                mgd[:p_nsc][tii][idx.dev_pzone[zone]] .-= mgd_com*sign(stt[:p_nsc_zonal_penalty][tii][zone])
                if ~isempty(idx.pr_pzone[zone])
                    # only do the following if there are producers here
                    i_pmax  = idx.pr_pzone[zone][argmax(stt[:dev_p][tii][idx.pr_pzone[zone]])]
                    # ===> requirements -- depend on active power production/consumption!
                    for dev in i_pmax # we only take the derivative of the device which has the highest production
                        alpha = mgd_com*sign(stt[:p_nsc_zonal_penalty][tii][zone])*prm.reserve.scr_sigma[zone]
                        dp_alpha!(grd, dev, tii, alpha)
                    end
                    for dev in i_pmax # we only take the derivative of the device which has the highest production
                        alpha = mgd_com*sign(stt[:p_nsc_zonal_penalty][tii][zone])*prm.reserve.nsc_sigma[zone]
                        dp_alpha!(grd, dev, tii, alpha)
                    end
                end
                if ~isempty(idx.cs_pzone[zone])
                    # only do the following if there are consumers here -- overly cautious
                    for dev in idx.cs_pzone[zone]
                        alpha = mgd_com*sign(stt[:p_nsc_zonal_penalty][tii][zone])*prm.reserve.rgu_sigma[zone]
                        dp_alpha!(grd, dev, tii, alpha)
                    end
                end

                # g21 (zrru_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrru_zonal] * cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone] #grd[:zrru_zonal][:p_rru_zonal_penalty][tii][zone]
                                                      # NOTE: minus
                mgd[:p_rru_on][tii][idx.dev_pzone[zone]]  .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*sign(stt[:p_rru_zonal_penalty][tii][zone])
                mgd[:p_rru_off][tii][idx.dev_pzone[zone]] .-= cgd.dzrru_zonal_dp_rru_zonal_penalty[tii][zone]*sign(stt[:p_rru_zonal_penalty][tii][zone])

                # g22 (zrrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zrrd_zonal] * cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone] #grd[:zrrd_zonal][:p_rrd_zonal_penalty][tii][zone]
                                                       # NOTE: minus
                mgd[:p_rrd_on][tii][idx.dev_pzone[zone]]  .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*sign(stt[:p_rrd_zonal_penalty][tii][zone])
                mgd[:p_rrd_off][tii][idx.dev_pzone[zone]] .-= cgd.dzrrd_zonal_dp_rrd_zonal_penalty[tii][zone]*sign(stt[:p_rrd_zonal_penalty][tii][zone])
            end

            # reserve zones -- q
            for zone in 1:sys.nzQ
                # g23 (zqru_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqru_zonal] * cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone] #grd[:zqru_zonal][:q_qru_zonal_penalty][tii][zone]
                                                  # NOTE: minus
                mgd[:q_qru][tii][idx.dev_qzone[zone]] .-= cgd.dzqru_zonal_dq_qru_zonal_penalty[tii][zone]*sign(stt[:q_qru_zonal_penalty][tii][zone])

                # g24 (zqrd_zonal):
                # OG => mgd_com = grd[:nzms][:zbase] * grd[:zbase][:zt] * grd[:zt][:zqrd_zonal] * cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone] #grd[:zqrd_zonal][:q_qrd_zonal_penalty][tii][zone]
                                                  # NOTE: minus
                mgd[:q_qrd][tii][idx.dev_qzone[zone]] .-= cgd.dzqrd_zonal_dq_qrd_zonal_penalty[tii][zone]*sign(stt[:q_qrd_zonal_penalty][tii][zone])
            end
        end


        function power_balance_old!(grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, idx::quasiGrad.Idx, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
            # call penalty cost
            cp = prm.vio.p_bus * qG.scale_c_pbus_testing
            cq = prm.vio.q_bus * qG.scale_c_qbus_testing
        
            @warn "the sum function sucks -- don't use this"
        
            # note: msc[:pb_slack] and stt[:pq][:slack] are just
            #       endlessly overwritten
        
            # loop over each time period and compute the power balance
            for tii in prm.ts.time_keys
                # duration
                dt = prm.ts.duration[tii]
        
                # loop over each bus
                for bus in 1:sys.nb
                    # active power balance: msc[:pb_slack][tii][bus] to record with time
                    msc[:pb_slack][bus] = 
                            # consumers (positive)
                            sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) +
                            # shunt
                            sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) +
                            # acline
                            sum(stt[:acline_pfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                            sum(stt[:acline_pto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                            # xfm
                            sum(stt[:xfm_pfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                            sum(stt[:xfm_pto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
                            # dcline
                            sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
                            sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
                            # producer (negative)
                           -sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0)
                    
                    # reactive power balance
                    msc[:qb_slack][bus] = 
                            # consumers (positive)
                            sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) +
                            # shunt
                            sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) +
                            # acline
                            sum(stt[:acline_qfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                            sum(stt[:acline_qto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                            # xfm
                            sum(stt[:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                            sum(stt[:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0) +
                            # dcline
                            sum(stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) + 
                            sum(stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) +
                            # producer (negative)
                           -sum(stt[:dev_q][tii][idx.pr[bus]]; init=0.0)
                end
        
                # actual mismatch penalty
                stt[:zp][tii] .= abs.(msc[:pb_slack]).*(cp*dt)
                stt[:zq][tii] .= abs.(msc[:qb_slack]).*(cq*dt)
        
                # evaluate the grad?
                if qG.eval_grad
                    if qG.pqbal_grad_type == "standard"
                        grd[:zp][:pb_slack][tii] .= (cp*dt).*sign.(msc[:pb_slack])
                        grd[:zq][:qb_slack][tii] .= (cq*dt).*sign.(msc[:qb_slack])
                    elseif qG.pqbal_grad_type == "soft_abs"
                        grd[:zp][:pb_slack][tii] .= (qG.pqbal_grad_weight_p*dt).*msc[:pb_slack]./(sqrt.(msc[:pb_slack].^2 .+ qG.pqbal_grad_eps2))
                        grd[:zq][:qb_slack][tii] .= (qG.pqbal_grad_weight_q*dt).*msc[:qb_slack]./(sqrt.(msc[:qb_slack].^2 .+ qG.pqbal_grad_eps2))
                    elseif qG.pqbal_grad_type == "quadratic_for_lbfgs"
                        grd[:zp][:pb_slack][tii] .= (cp*dt).*msc[:pb_slack]
                        grd[:zq][:qb_slack][tii] .= (cp*dt).*msc[:qb_slack]
                    else
                        println("not recognized!")
                    end
                end
            end
        end

        function device_reserve_costs_old!(prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}})
            # compute the costs associated with device reserve offers
        
            @warn "slow -- depricated"
        
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # duration
                dt = prm.ts.duration[tii]
                
                # costs
                stt[:zrgu][tii] .= dt.*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*stt[:p_rgu][tii]
                stt[:zrgd][tii] .= dt.*prm.dev.p_reg_res_down_cost_tmdv[t_ind].*stt[:p_rgd][tii]
                stt[:zscr][tii] .= dt.*prm.dev.p_syn_res_cost_tmdv[t_ind].*stt[:p_scr][tii]
                stt[:znsc][tii] .= dt.*prm.dev.p_nsyn_res_cost_tmdv[t_ind].*stt[:p_nsc][tii]
                stt[:zrru][tii] .= dt.*(prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind].*stt[:p_rru_on][tii] .+
                                        prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind].*stt[:p_rru_off][tii])
                stt[:zrrd][tii] .= dt.*(prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind].*stt[:p_rrd_on][tii] .+
                                        prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind].*stt[:p_rrd_off][tii]) 
                stt[:zqru][tii] .= dt.*prm.dev.q_res_up_cost_tmdv[t_ind].*stt[:q_qru][tii]      
                stt[:zqrd][tii] .= dt.*prm.dev.q_res_down_cost_tmdv[t_ind].*stt[:q_qrd][tii]
            end
        end



        function reserve_balance_experimental!(idx::quasiGrad.Idx, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
            # we need access to the time index itself
            for (t_ind, tii) in enumerate(prm.ts.time_keys)
                # duration
                dt = prm.ts.duration[tii]
        
                # for the "endogenous" reserve requirements
                rgu_sigma = prm.reserve.rgu_sigma
                rgd_sigma = prm.reserve.rgd_sigma 
                scr_sigma = prm.reserve.scr_sigma 
                nsc_sigma = prm.reserve.nsc_sigma  
        
                # loop over the zones (active power)
                for zone in 1:sys.nzP
        
                    # compute the reserve sums -- these are put in msc!!
                    #quasiGrad.reserve_sum!(idx, msc, :p_rgu    , stt, tii, zone, :Pz)
                    #quasiGrad.reserve_sum!(idx, msc, :p_rgd    , stt, tii, zone, :Pz)
                    #quasiGrad.reserve_sum!(idx, msc, :p_scr    , stt, tii, zone, :Pz)
                    #quasiGrad.reserve_sum!(idx, msc, :p_nsc    , stt, tii, zone, :Pz)
                    #quasiGrad.reserve_sum!(idx, msc, :p_rru_on , stt, tii, zone, :Pz)
                    #quasiGrad.reserve_sum!(idx, msc, :p_rru_off, stt, tii, zone, :Pz)
                    #quasiGrad.reserve_sum!(idx, msc, :p_rrd_on , stt, tii, zone, :Pz)
                    #quasiGrad.reserve_sum!(idx, msc, :p_rrd_off, stt, tii, zone, :Pz)
        
                    # endogenous sum
                    if isempty(idx.cs_pzone[zone])
                        # in the case there are NO consumers in a zone
                        stt[:p_rgu_zonal_REQ][tii][zone] = 0.0
                        stt[:p_rgd_zonal_REQ][tii][zone] = 0.0
                    else
                        quasiGrad.reserve_p_sum!(idx, msc, stt, tii, zone)
                        stt[:p_rgu_zonal_REQ][tii][zone] = rgu_sigma[zone]*msc[:pz_sum][zone]
                        stt[:p_rgd_zonal_REQ][tii][zone] = rgd_sigma[zone]*msc[:pz_sum][zone]
                    end
        
                    # endogenous max
                    if isempty(idx.pr_pzone[zone])
                        # in the case there are NO producers in a zone
                        stt[:p_scr_zonal_REQ][tii][zone] = 0.0
                        stt[:p_scr_zonal_REQ][tii][zone] = 0.0
                    else
                        quasiGrad.reserve_p_max!(idx, msc, stt, tii, zone)
                        stt[:p_scr_zonal_REQ][tii][zone] = scr_sigma[zone]*msc[:pz_max][zone]
                        stt[:p_nsc_zonal_REQ][tii][zone] = nsc_sigma[zone]*msc[:pz_max][zone]
                    end
        
                    # balance equations -- compute the shortfall values
                    stt[:p_rgu_zonal_penalty][tii][zone] = max(stt[:p_rgu_zonal_REQ][tii][zone] - 
                                msc[:p_rgu][zone], 0.0)
                    
                    stt[:p_rgd_zonal_penalty][tii][zone] = max(stt[:p_rgd_zonal_REQ][tii][zone] - 
                                msc[:p_rgd][zone], 0.0)
        
                    stt[:p_scr_zonal_penalty][tii][zone] = max(stt[:p_rgu_zonal_REQ][tii][zone] + 
                                stt[:p_scr_zonal_REQ][tii][zone] -
                                msc[:p_rgu][zone] -
                                msc[:p_scr][zone],0.0)
        
                    stt[:p_nsc_zonal_penalty][tii][zone] = max(stt[:p_rgu_zonal_REQ][tii][zone] + 
                                stt[:p_scr_zonal_REQ][tii][zone] +
                                stt[:p_nsc_zonal_REQ][tii][zone] -
                                msc[:p_rgu][zone] -
                                msc[:p_scr][zone] - 
                                msc[:p_nsc][zone],0.0)
        
                    stt[:p_rru_zonal_penalty][tii][zone] = max(prm.reserve.rru_min[zone][t_ind] -
                                msc[:p_rru_on][zone] - 
                                msc[:p_rru_off][zone],0.0)
        
                    stt[:p_rrd_zonal_penalty][tii][zone] = max(prm.reserve.rrd_min[zone][t_ind] -
                                msc[:p_rrd_on][zone] - 
                                msc[:p_rrd_off][zone],0.0)
                end
        
                # loop over the zones (reactive power) -- gradients are computed in the master grad
                for zone in 1:sys.nzQ
                    quasiGrad.reserve_sum!(idx, msc, :q_qru , stt, tii, zone, :Qz)
                    quasiGrad.reserve_sum!(idx, msc, :q_qrd, stt, tii, zone, :Qz)
        
                    stt[:q_qru_zonal_penalty][tii][zone] = max(prm.reserve.qru_min[zone][t_ind] -
                                msc[:q_qru][zone], 0.0)
        
                    stt[:q_qrd_zonal_penalty][tii][zone] = max(prm.reserve.qrd_min[zone][t_ind] -
                                msc[:q_qrd][zone], 0.0)
                end
        
                # finally, call the penalty costt
                crgu = prm.vio.rgu_zonal
                crgd = prm.vio.rgd_zonal
                cscr = prm.vio.scr_zonal
                cnsc = prm.vio.nsc_zonal
                crru = prm.vio.rru_zonal
                crrd = prm.vio.rrd_zonal
                cqru = prm.vio.qru_zonal
                cqrd = prm.vio.qrd_zonal
        
                # shortfall penalties -- gradients are static and taken when initialized
                stt[:zrgu_zonal][tii] .= (dt*crgu).*stt[:p_rgu_zonal_penalty][tii]
                stt[:zrgd_zonal][tii] .= (dt*crgd).*stt[:p_rgd_zonal_penalty][tii]
                stt[:zscr_zonal][tii] .= (dt*cscr).*stt[:p_scr_zonal_penalty][tii]
                stt[:znsc_zonal][tii] .= (dt*cnsc).*stt[:p_nsc_zonal_penalty][tii]
                stt[:zrru_zonal][tii] .= (dt*crru).*stt[:p_rru_zonal_penalty][tii]
                stt[:zrrd_zonal][tii] .= (dt*crrd).*stt[:p_rrd_zonal_penalty][tii]
                stt[:zqru_zonal][tii] .= (dt*cqru).*stt[:q_qru_zonal_penalty][tii]
                stt[:zqrd_zonal][tii] .= (dt*cqrd).*stt[:q_qrd_zonal_penalty][tii]
            end
        end
        
        # reserve sum
        function reserve_sum!(idx::quasiGrad.Idx, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, reserve_type::Symbol, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, tii::Symbol, zone::Int64, zone_type::Symbol)
            if zone_type == :Pz
                msc[reserve_type][zone] = 0.0
                for dev in idx.dev_pzone[zone]
                    msc[reserve_type][zone] += stt[reserve_type][tii][dev]
                end
            elseif zone_type == :Qz
                msc[reserve_type][zone] = 0.0
                for dev in idx.dev_qzone[zone]
                    msc[reserve_type][zone] += stt[reserve_type][tii][dev]
                end
            else
                @warn "wrong zone type!!"
            end
        end
        
        # reserve power sum
        function reserve_p_sum!(idx::quasiGrad.Idx, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, tii::Symbol, zone::Int64)
            msc[:pz_sum][zone] = 0.0
            for dev in idx.cs_pzone[zone]
                msc[:pz_sum][zone] += stt[:dev_p][tii][dev]
            end
        end
        
        # reserve power max
        function reserve_p_max!(idx::quasiGrad.Idx, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, tii::Symbol, zone::Int64)
            msc[:pz_max][zone] = 0.0
            for dev in idx.cs_pzone[zone]
                if stt[:dev_p][tii][dev] > msc[:pz_max][zone]
                    msc[:pz_max][zone] = copy(stt[:dev_p][tii][dev])
                end
            end
        end

        :pz_sum          => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :pz_max          => zeros(sys.nzP), # NOTE -- these are ZONAL max's!!
        :p_rgu           => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :p_rgd           => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :p_scr           => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :p_nsc           => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :p_rru_on        => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :p_rru_off       => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :p_rrd_on        => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :p_rrd_off       => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :q_qru           => zeros(sys.nzP), # NOTE -- these are ZONAL sums!!
        :q_qrd           => zeros(sys.nzP) # NOTE -- these are ZONAL sums!!



        function adam_pf!(adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, alpha::Float64, beta1::Float64, beta2::Float64, beta1_decay::Float64, beta2_decay::Float64, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
            # loop over the keys in mgd
            for var_key in keys(mgd)
                # loop over all time
                for tii in prm.ts.time_keys
                    # states to update                                            
                    if var_key in keys(upd)
                        # => (var_key in keys(upd)) ? (update = upd[var_key][tii]) : (update = Colon())
                        #    the above caused weird type instability, so we just copy and paste
                        update_subset = upd[var_key][tii]
        
                        # update adam moments
                        clipped_grad                         = clamp.(mgd[var_key][tii][update_subset], -qG.grad_max, qG.grad_max)
                        adm[var_key][:m][tii][update_subset] = beta1.*adm[var_key][:m][tii][update_subset] + (1.0-beta1).*clipped_grad
                        adm[var_key][:v][tii][update_subset] = beta2.*adm[var_key][:v][tii][update_subset] + (1.0-beta2).*clipped_grad.^2.0
                        stt[var_key][tii][update_subset]     = stt[var_key][tii][update_subset] - alpha*(adm[var_key][:m][tii][update_subset]/(1.0-beta1_decay))./(sqrt.(adm[var_key][:v][tii][update_subset]/(1.0-beta2_decay)) .+ qG.eps)
                        
                    else 
                        # update adam moments
                        clipped_grad          = clamp.(mgd[var_key][tii], -qG.grad_max, qG.grad_max)
                        adm[var_key][:m][tii] = beta1.*adm[var_key][:m][tii] + (1.0-beta1).*clipped_grad
                        adm[var_key][:v][tii] = beta2.*adm[var_key][:v][tii] + (1.0-beta2).*clipped_grad.^2.0
                        stt[var_key][tii]     = stt[var_key][tii] - alpha*(adm[var_key][:m][tii]/(1.0-beta1_decay))./(sqrt.(adm[var_key][:v][tii]/(1.0-beta2_decay)) .+ qG.eps)
                    end
                end
            end
        end

        function master_grad_zs_xfm_fastesttt!(tii::Symbol, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
            # =========== =========== =========== #
                        # zs (xfm)
            # =========== =========== =========== #
            #
            # common master grad
            # OG => mg_com =  grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zs_xfm]
            
            # common flow grads
            pfr_com = grd[:zs_xfm][:xfm_pfr][tii]
            qfr_com = grd[:zs_xfm][:xfm_qfr][tii]
            pto_com = grd[:zs_xfm][:xfm_pto][tii]
            qto_com = grd[:zs_xfm][:xfm_qto][tii]
        
            # final pfr gradients
            # OG => mgpfr   = mg_com.*pfr_com
            # ... * everything below:
            msc[:vmfrpfr_x] .= pfr_com.*grd[:xfm_pfr][:vmfr][tii]
            msc[:vmtopfr_x] .= pfr_com.*grd[:xfm_pfr][:vmto][tii]
            msc[:vafrpfr_x] .= pfr_com.*grd[:xfm_pfr][:vafr][tii]
            msc[:vatopfr_x] .= pfr_com.*grd[:xfm_pfr][:vato][tii]
            msc[:taupfr_x]  .= pfr_com.*grd[:xfm_pfr][:tau][tii]
            msc[:phipfr_x]  .= pfr_com.*grd[:xfm_pfr][:phi][tii]
            msc[:uonpfr_x]  .= pfr_com.*grd[:xfm_pfr][:uon][tii]
        
            # final qfr gradients
            # OG => mgqfr   = mg_com.*qfr_com
            # ... * everything below:
            msc[:vmfrqfr_x] .= qfr_com.*grd[:xfm_qfr][:vmfr][tii]
            msc[:vmtoqfr_x] .= qfr_com.*grd[:xfm_qfr][:vmto][tii]
            msc[:vafrqfr_x] .= qfr_com.*grd[:xfm_qfr][:vafr][tii]
            msc[:vatoqfr_x] .= qfr_com.*grd[:xfm_qfr][:vato][tii]
            msc[:tauqfr_x]  .= qfr_com.*grd[:xfm_qfr][:tau][tii]
            msc[:phiqfr_x]  .= qfr_com.*grd[:xfm_qfr][:phi][tii]
            msc[:uonqfr_x]  .= qfr_com.*grd[:xfm_qfr][:uon][tii]
        
            # final pto gradients
            # OG => mgpto   = mg_com.*pto_com
            # ... * everything below:
            msc[:vmfrpto_x] .= pto_com.*grd[:xfm_pto][:vmfr][tii]
            msc[:vmtopto_x] .= pto_com.*grd[:xfm_pto][:vmto][tii]
            msc[:vafrpto_x] .= pto_com.*grd[:xfm_pto][:vafr][tii]
            msc[:vatopto_x] .= pto_com.*grd[:xfm_pto][:vato][tii]
            msc[:taupto_x]  .= pto_com.*grd[:xfm_pto][:tau][tii]
            msc[:phipto_x]  .= pto_com.*grd[:xfm_pto][:phi][tii]
            msc[:uonpto_x]  .= pto_com.*grd[:xfm_pto][:uon][tii]
        
            # final qfr gradients
            # OG => mgqto   = mg_com.*qto_com
            # ... * everything below:
            msc[:vmfrqto_x] .= qto_com.*grd[:xfm_qto][:vmfr][tii]
            msc[:vmtoqto_x] .= qto_com.*grd[:xfm_qto][:vmto][tii]
            msc[:vafrqto_x] .= qto_com.*grd[:xfm_qto][:vafr][tii]
            msc[:vatoqto_x] .= qto_com.*grd[:xfm_qto][:vato][tii]
            msc[:tauqto_x]  .= qto_com.*grd[:xfm_qto][:tau][tii]
            msc[:phiqto_x]  .= qto_com.*grd[:xfm_qto][:phi][tii]
            msc[:uonqto_x]  .= qto_com.*grd[:xfm_qto][:uon][tii]
        
            # note: we must loop over these assignments!
            for xfm in 1:sys.nx
                # update the master grad -- pfr
                mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrpfr[xfm]
                mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtopfr[xfm]
                mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrpfr[xfm]
                mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatopfr[xfm]
                mgd[:tau][tii][xfm]                += taupfr[xfm]
                mgd[:phi][tii][xfm]                += phipfr[xfm]
                mgd[:u_on_xfm][tii][xfm]           += uonpfr[xfm]
        
                # update the master grad -- qfr
                mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqfr[xfm]
                mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqfr[xfm]
                mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqfr[xfm]
                mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqfr[xfm]
                mgd[:tau][tii][xfm]                += tauqfr[xfm]
                mgd[:phi][tii][xfm]                += phiqfr[xfm]
                mgd[:u_on_xfm][tii][xfm]           += uonqfr[xfm]
        
                # update the master grad -- pto
                mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrpto[xfm]
                mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtopto[xfm]
                mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrpto[xfm]
                mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatopto[xfm]
                mgd[:tau][tii][xfm]                += taupto[xfm]
                mgd[:phi][tii][xfm]                += phipto[xfm]
                mgd[:u_on_xfm][tii][xfm]           += uonpto[xfm]
        
                # update the master grad -- qto
                mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqto[xfm]
                mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqto[xfm]
                mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqto[xfm]
                mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqto[xfm]
                mgd[:tau][tii][xfm]                += tauqto[xfm]
                mgd[:phi][tii][xfm]                += phiqto[xfm]
                mgd[:u_on_xfm][tii][xfm]           += uonqto[xfm]
            end
        end


        function master_grad_zs_acline_fastesttt!(tii::Symbol, idx::quasiGrad.Idx, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
            # =========== =========== =========== #
                        # zs (acline flows)
            # =========== =========== =========== #
            #
            # common master grad
            # OG => mg_com =  grd[:nzms][:zbase] .* grd[:zbase][:zt] .* grd[:zt][:zs_acline]
            
            # common flow grads
            pfr_com = grd[:zs_acline][:acline_pfr][tii]
            qfr_com = grd[:zs_acline][:acline_qfr][tii]
            pto_com = grd[:zs_acline][:acline_pto][tii]
            qto_com = grd[:zs_acline][:acline_qto][tii]
        
            # final pfr gradients
            # OG => mgpfr   = mg_com.*pfr_com
            # mgpfr * everything below:
            msc[:vmfrpfr] .= pfr_com.*grd[:acline_pfr][:vmfr][tii]
            msc[:vmtopfr] .= pfr_com.*grd[:acline_pfr][:vmto][tii]
            msc[:vafrpfr] .= pfr_com.*grd[:acline_pfr][:vafr][tii]
            msc[:vatopfr] .= pfr_com.*grd[:acline_pfr][:vato][tii]
            msc[:uonpfr]  .= pfr_com.*grd[:acline_pfr][:uon][tii]
        
            # final qfr gradients
            # OG => mgqfr   = mg_com.*qfr_com
            # mgqfr * everything below:
            msc[:vmfrqfr] .= qfr_com.*grd[:acline_qfr][:vmfr][tii]
            msc[:vmtoqfr] .= qfr_com.*grd[:acline_qfr][:vmto][tii]
            msc[:vafrqfr] .= qfr_com.*grd[:acline_qfr][:vafr][tii]
            msc[:vatoqfr] .= qfr_com.*grd[:acline_qfr][:vato][tii]
            msc[:uonqfr]  .= qfr_com.*grd[:acline_qfr][:uon][tii]
        
            # final pto gradients
            # OG => mgpto   = mg_com.*pto_com
            # mgpto * everything below:
            msc[:vmfrpto] .= pto_com.*grd[:acline_pto][:vmfr][tii]
            msc[:vmtopto] .= pto_com.*grd[:acline_pto][:vmto][tii]
            msc[:vafrpto] .= pto_com.*grd[:acline_pto][:vafr][tii]
            msc[:vatopto] .= pto_com.*grd[:acline_pto][:vato][tii]
            msc[:uonpto]  .= pto_com.*grd[:acline_pto][:uon][tii]
        
            # final qfr gradients
            # OG => mgqto   = mg_com.*qto_com
            # mgqto * everything below:
            msc[:vmfrqto] .= qto_com.*grd[:acline_qto][:vmfr][tii]
            msc[:vmtoqto] .= qto_com.*grd[:acline_qto][:vmto][tii]
            msc[:vafrqto] .= qto_com.*grd[:acline_qto][:vafr][tii]
            msc[:vatoqto] .= qto_com.*grd[:acline_qto][:vato][tii]
            msc[:uonqto]  .= qto_com.*grd[:acline_qto][:uon][tii]
        
            # note: we MUST loop over these assignments! otherwise, += gets confused
            for ln in 1:sys.nl
                # update the master grad -- pfr
                mgd[:vm][tii][idx.acline_fr_bus[ln]] += msc[:vmfrpfr][ln]
                mgd[:vm][tii][idx.acline_to_bus[ln]] += msc[:vmtopfr][ln]
                mgd[:va][tii][idx.acline_fr_bus[ln]] += msc[:vafrpfr][ln]
                mgd[:va][tii][idx.acline_to_bus[ln]] += msc[:vatopfr][ln]
                mgd[:u_on_acline][tii][ln]           += msc[:uonpfr][ln]
        
                # update the master grad -- qfr
                mgd[:vm][tii][idx.acline_fr_bus[ln]] += msc[:vmfrqfr][ln]
                mgd[:vm][tii][idx.acline_to_bus[ln]] += msc[:vmtoqfr][ln]
                mgd[:va][tii][idx.acline_fr_bus[ln]] += msc[:vafrqfr][ln]
                mgd[:va][tii][idx.acline_to_bus[ln]] += msc[:vatoqfr][ln]
                mgd[:u_on_acline][tii][ln]           += msc[:uonqfr][ln]
        
                # update the master grad -- pto
                mgd[:vm][tii][idx.acline_fr_bus[ln]] += msc[:vmfrpto][ln]
                mgd[:vm][tii][idx.acline_to_bus[ln]] += msc[:vmtopto][ln]
                mgd[:va][tii][idx.acline_fr_bus[ln]] += msc[:vafrpto][ln]
                mgd[:va][tii][idx.acline_to_bus[ln]] += msc[:vatopto][ln]
                mgd[:u_on_acline][tii][ln]           += msc[:uonpto][ln]
        
                # update the master grad -- qto
                mgd[:vm][tii][idx.acline_fr_bus[ln]] += msc[:vmfrqto][ln]
                mgd[:vm][tii][idx.acline_to_bus[ln]] += msc[:vmtoqto][ln]
                mgd[:va][tii][idx.acline_fr_bus[ln]] += msc[:vafrqto][ln]
                mgd[:va][tii][idx.acline_to_bus[ln]] += msc[:vatoqto][ln]
                mgd[:u_on_acline][tii][ln]           += msc[:uonqto][ln]
            end
        end

# cleanup power flow (to some degree of accuracy)
function cleanup_pf_with_Gurobi!(idx::quasiGrad.Idx, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, qG::quasiGrad.QG,  stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # device p/q stay fixed -- just tune v, theta, and dc
    # here is power balance:
    #
    # p_pr - p_cs - pdc = p_lines/xfm/shunt => this is typical.
    #
    # vm0 = stt[:vm][tii]
    # va0 = stt[:va][tii][2:end-1]
    #
    # bias point: msc[:pinj0, qinj0] === Y * stt[:vm, va]
    qG.compute_pf_injs_with_Jac = true

    @warn "this is useless, because it doesn't enforce ramp constraints!"

    # build and empty the model!
    model = Model(Gurobi.Optimizer)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # set model properties
    quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
    quasiGrad.set_attribute(model, MOI.RelativeGapTolerance(), qG.FeasibilityTol)
    quasiGrad.set_attribute(model, MOI.AbsoluteGapTolerance(), qG.FeasibilityTol)

    @info "Running lineaized power flow cleanup across $(sys.nT) time periods."

    # loop over time
    for (t_ind, tii) in enumerate(prm.ts.time_keys)

        # initialize
        run_pf    = true
        pf_cnt    = 0 

        # 1. update the ideal dispatch points p/q -- we do this just once
        quasiGrad.ideal_dispatch!(idx, msc, stt, sys, tii)

        # 2. update y_bus and Jacobian and bias point -- this
        #    only needs to be done once per time, since xfm/shunt
        #    values are not changing between iterations
        Ybus_real, Ybus_imag = quasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii)

        # loop over pf solves
        while run_pf == true

            # increment
            pf_cnt += 1

            # first, rebuild the jacobian, and update the
            # base points: msc[:pinj0], msc[:qinj0]
            Jac = quasiGrad.build_acpf_Jac_and_pq0(msc, qG, stt, sys, tii, Ybus_real, Ybus_imag);
            
            # quiet down!!!
            empty!(model)

            # define the variables (single time index)
            @variable(model, x_in[1:(2*sys.nb - 1)])
            set_start_value.(x_in, [stt[:vm][tii]; stt[:va][tii][2:end]])

            # assign
            dvm   = x_in[1:sys.nb]
            dva   = x_in[(sys.nb+1):end]

            # note:
            # vm   = vm0   + dvm
            # va   = va0   + dva
            # pinj = pinj0 + dpinj
            # qinj = qinj0 + dqinj
            #
            # key equation:
            #                       dPQ .== Jac*dVT
            #                       dPQ + basePQ(v) = devicePQ
            #
            #                       Jac*dVT + basePQ(v) == devicePQ
            #
            # so, we don't actually need to model dPQ explicitly (cool)
            #
            # so, the optimizer asks, how shall we tune dVT in order to produce a power perurbation
            # which, when added to the base point, lives inside the feasible device region?
            #
            # based on the result, we only have to actually update the device set points on the very
            # last power flow iteration, where we have converged.

            # now, model all nodal injections from the device/dc line side, all put in nodal_p/q
            nodal_p = Vector{AffExpr}(undef, sys.nb)
            nodal_q = Vector{AffExpr}(undef, sys.nb)
            for bus in 1:sys.nb
                # now, we need to loop and set the affine expressions to 0, and then add powers
                #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
                nodal_p[bus] = AffExpr(0.0)
                nodal_q[bus] = AffExpr(0.0)
            end

            # create a flow variable for each dc line and sum these into the nodal vectors
            if sys.nldc == 0
                # nothing to see here
            else

                # define dc variables
                @variable(model, pdc_vars[1:sys.nldc])    # oriented so that fr = + !!
                @variable(model, qdc_fr_vars[1:sys.nldc])
                @variable(model, qdc_to_vars[1:sys.nldc])

                set_start_value.(pdc_vars, stt[:dc_pfr][tii])
                set_start_value.(qdc_fr_vars, stt[:dc_qfr][tii])
                set_start_value.(qdc_to_vars, stt[:dc_qto][tii])

                # bound dc power
                @constraint(model, -prm.dc.pdc_ub    .<= pdc_vars    .<= prm.dc.pdc_ub)
                @constraint(model,  prm.dc.qdc_fr_lb .<= qdc_fr_vars .<= prm.dc.qdc_fr_ub)
                @constraint(model,  prm.dc.qdc_to_lb .<= qdc_to_vars .<= prm.dc.qdc_to_ub)

                # loop and add to the nodal injection vectors
                for dcl in 1:sys.nldc
                    add_to_expression!(nodal_p[idx.dc_fr_bus[dcl]], -pdc_vars[dcl])
                    add_to_expression!(nodal_p[idx.dc_to_bus[dcl]], +pdc_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_fr_bus[dcl]], -qdc_fr_vars[dcl])
                    add_to_expression!(nodal_q[idx.dc_to_bus[dcl]], -qdc_to_vars[dcl])
                end
            end

            # next, deal with devices
            # 
            for dev in 1:sys.ndev
                if dev in idx.pr_devs
                    # producers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], stt[:dev_p][tii][dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], stt[:dev_q][tii][dev])
                else
                    # consumers
                    add_to_expression!(nodal_p[idx.device_to_bus[dev]], -stt[:dev_p][tii][dev])
                    add_to_expression!(nodal_q[idx.device_to_bus[dev]], -stt[:dev_q][tii][dev])
                end
            end

            # bound system variables ==============================================
            #
            # bound variables -- voltage
            @constraint(model, prm.bus.vm_lb - stt[:vm][tii] .<= dvm .<= prm.bus.vm_ub - stt[:vm][tii])
            # alternative: => @constraint(model, prm.bus.vm_lb .<= stt[:vm][tii] + dvm .<= prm.bus.vm_ub)

            # mapping
            JacP_noref = @view Jac[1:sys.nb,      [1:sys.nb; (sys.nb+2):end]]
            JacQ_noref = @view Jac[(sys.nb+1):end,[1:sys.nb; (sys.nb+2):end]]

            # objective: find v, theta with minimal mismatch penalty
            obj    = AffExpr(0.0)
            tmp_vm = @variable(model)
            tmp_va = @variable(model)
            @variable(model, slack_p[1:sys.nb])
            @variable(model, slack_q[1:sys.nb])

            for bus in 1:sys.nb
                # penalize mismatch
                @constraint(model, JacP_noref[bus,:]'*x_in + msc[:pinj0][bus] - nodal_p[bus] <= slack_p[bus])
                @constraint(model, nodal_p[bus] - JacP_noref[bus,:]'*x_in - msc[:pinj0][bus] <= slack_p[bus])
                @constraint(model, JacQ_noref[bus,:]'*x_in + msc[:qinj0][bus] - nodal_q[bus] <= slack_q[bus])
                @constraint(model, nodal_q[bus] - JacQ_noref[bus,:]'*x_in - msc[:qinj0][bus] <= slack_q[bus])

                # add both to the objective
                add_to_expression!(obj, slack_p[bus], 1e3)
                add_to_expression!(obj, slack_q[bus], 1e3)

                # voltage regularization
                @constraint(model, -dvm[bus] <= tmp_vm)
                @constraint(model,  dvm[bus] <= tmp_vm)

                # phase regularization
                if bus > 1
                    @constraint(model, -dva[bus-1] <= tmp_va)
                    @constraint(model,  dva[bus-1] <= tmp_va)
                end
            end

            # this adds light regularization and causes convergence
            add_to_expression!(obj, tmp_vm)
            add_to_expression!(obj, tmp_va)


            # set the objective
            @objective(model, Min, obj)

            # solve
            optimize!(model)

            # test solution!
            soln_valid = solution_status(model)

            # if we have reached our max number of tries, jsut quit after this
            if pf_cnt == qG.max_linear_pfs
                run_pf = false
            end

            # test validity
            if soln_valid == true
                # we update the voltage soluion
                stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
                stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

                # take the norm of dv
                max_dx = maximum(abs.(value.(x_in)))
                
                # println("========================================================")
                if qG.print_linear_pf_iterations == true
                    println(termination_status(model),". ",primal_status(model),". objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
                end
                # println("========================================================")
                #
                # shall we terminate?
                if (max_dx < qG.max_pf_dx) || (pf_cnt == qG.max_linear_pfs)
                    run_pf = false

                    # now, apply the updated injections to the devices
                    if sys.nldc > 0
                        stt[:dc_pfr][tii] =  value.(pdc_vars)
                        stt[:dc_pto][tii] = -value.(pdc_vars)  # also, performed in clipping
                        stt[:dc_qfr][tii] = value.(qdc_fr_vars)
                        stt[:dc_qto][tii] = value.(qdc_to_vars)
                    end
                end
            else
                # the solution is NOT valid
                @warn "Power flow cleanup failed at time $(tii)!"
            end
        end
    end
end

# soft abs derviative
function soft_abs_grad_faster(x::Vector{Float64}, qG::quasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x./(sqrt.(x.^2 .+ qG.acflow_grad_eps2))
end


function soft_abs_grad_ac_old(x::Float64, qG::quasiGrad.QG)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return x/(sqrt(x^2 + qG.acflow_grad_eps2))
end


function soft_abs_grad_ac(ind::Int64, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, qG::quasiGrad.QG, s::Symbol)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0)
    return msc[s][ind]/(sqrt(msc[s][ind]^2 + qG.acflow_grad_eps2))
end

# soft abs derviative
function soft_abs_grad_vec!(bit::Dict{Symbol, Dict{Symbol, BitVector}}, Dict{Symbol, Dict{Symbol, Vector{Float64}}}, qG::quasiGrad.QG, s_bit::Symbol, s_in::Symbol, s_out::Symbol)
    # soft_abs(x)      = sqrt(x^2 + eps^2)
    # soft_abs_grad(x) = x/sqrt(x^2 + eps^2)
    #
    # usage: instead of c*sign(max(x,0)), use c*soft_abs_grad(max(x,0))
    # usage: instead of c*abs(x), use c*soft_abs_grad(x,0) #
    msc[s_out][bit[s_bit]] .= @view msc[s_in][bit[s_bit]]#./(sqrt.((@view msc[s_in][bit[s_bit]]).^2 .+ qG.acflow_grad_eps2))
end

# correct the reactive power injections into the network
function correct_reactive_injections!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, tii::Symbol)

    # warning
    @info "note: the reactive power correction function does NOT take J pqe into account yet"
    @warn "this was a good idea, but it doesn't really work -- replaced by power flow!"

    # loop over each time period
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # at this time, compute the pr and cs upper and lower bounds across all devices
        dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
        dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]
        # note: clipping is based on the upper/lower bounds, and not
        # based on the beta linking equations -- so, we just treat
        # that as a penalty, and not as a power balance factor
        # 
        # also, compute the dc line upper and lower bounds
        dcfr_qlb = prm.dc.qdc_fr_lb
        dcfr_qub = prm.dc.qdc_fr_ub
        dcto_qlb = prm.dc.qdc_to_lb
        dcto_qub = prm.dc.qdc_to_ub

        # how does balance work?
        # 0 = qb_slack + (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)
        #
        # so, we take want to set:
        # -qb_slack = (dev_q_cs + dc_qfr + dc_qto - dev_q_pr)

        for bus in 1:sys.nb
            # reactive power balance
            qb_slack = 
                    # shunt        
                    sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) +
                    # acline
                    sum(stt[:acline_qfr][tii][idx.bus_is_acline_frs[bus]]; init=0.0) + 
                    sum(stt[:acline_qto][tii][idx.bus_is_acline_tos[bus]]; init=0.0) +
                    # xfm
                    sum(stt[:xfm_qfr][tii][idx.bus_is_xfm_frs[bus]]; init=0.0) + 
                    sum(stt[:xfm_qto][tii][idx.bus_is_xfm_tos[bus]]; init=0.0)
                    # dcline -- not included
                    # consumers (positive) -- not included
                    # producer (negative) -- not included

            # get limits
            pr_lb   = sum(dev_qlb[idx.pr[bus]]; init=0.0)
            cs_lb   = sum(dev_qlb[idx.cs[bus]]; init=0.0)
            pr_ub   = sum(dev_qub[idx.pr[bus]]; init=0.0) 
            cs_ub   = sum(dev_qub[idx.cs[bus]]; init=0.0)
            dcfr_lb = sum(dcfr_qlb[idx.bus_is_dc_frs[bus]]; init=0.0)
            dcfr_ub = sum(dcfr_qub[idx.bus_is_dc_frs[bus]]; init=0.0)
            dcto_lb = sum(dcto_qlb[idx.bus_is_dc_tos[bus]]; init=0.0)
            dcto_ub = sum(dcto_qub[idx.bus_is_dc_tos[bus]]; init=0.0) 

            # total: lb < -qb_slack < ub
            ub = cs_ub + dcfr_ub + dcto_ub - pr_lb
            lb = cs_lb + dcfr_lb + dcto_lb - pr_ub

            # test
            if -qb_slack > ub
                println("ub limit")
                #assign = ub
                # max everything out
                stt[:dev_q][tii][idx.cs[bus]]             = dev_qub[idx.cs[bus]]
                stt[:dev_q][tii][idx.pr[bus]]             = dev_qlb[idx.pr[bus]]
                stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]] = dcfr_qub[idx.bus_is_dc_frs[bus]]
                stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]] = dcfr_qub[idx.bus_is_dc_tos[bus]]

            elseif -qb_slack < lb
                println("lb limit")
                #assign = ub
                # min everything out
                stt[:dev_q][tii][idx.cs[bus]]             = dev_qlb[idx.cs[bus]]
                stt[:dev_q][tii][idx.pr[bus]]             = dev_qub[idx.pr[bus]]
                stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]] = dcfr_qlb[idx.bus_is_dc_frs[bus]]
                stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]] = dcfr_qlb[idx.bus_is_dc_tos[bus]]

            else # in the middle -- all good
                println("middle")
                lb_dist  = -qb_slack - lb
                bnd_dist = ub - lb
                scale    = lb_dist/bnd_dist

                stt[:dev_q][tii][idx.cs[bus]]             = dev_qlb[idx.cs[bus]]             + scale*(dev_qub[idx.cs[bus]] - dev_qlb[idx.cs[bus]])
                stt[:dev_q][tii][idx.pr[bus]]             = dev_qub[idx.pr[bus]]             - scale*(dev_qub[idx.pr[bus]] - dev_qlb[idx.pr[bus]])
                stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]] = dcfr_qlb[idx.bus_is_dc_frs[bus]] + scale*(dcfr_qub[idx.bus_is_dc_frs[bus]] - dcfr_qlb[idx.bus_is_dc_frs[bus]])
                stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]] = dcfr_qlb[idx.bus_is_dc_tos[bus]] + scale*(dcfr_qub[idx.bus_is_dc_tos[bus]] - dcfr_qlb[idx.bus_is_dc_tos[bus]])
            end
        end
    end
end

msc = Dict(
    :pinj_ideal => zeros(sys.nb),
    :qinj_ideal => zeros(sys.nb),
    :pb_slack   => zeros(sys.nb),
    :qb_slack   => zeros(sys.nb),
    :pub        => zeros(sys.nb),
    :plb        => zeros(sys.nb),
    :qub        => zeros(sys.nb),
    :qlb        => zeros(sys.nb),
    :pinj0      => zeros(sys.nb), # bias point
    :qinj0      => zeros(sys.nb), # bias point
    :pinj_dc    => zeros(sys.nb),
    :cos_ftp    => zeros(sys.nl),
    :sin_ftp    => zeros(sys.nl),
    :vff        => zeros(sys.nl),
    :vtt        => zeros(sys.nl),
    :vft        => zeros(sys.nl),
    :pfr        => zeros(sys.nl),
    :pto        => zeros(sys.nl),
    :qfr        => zeros(sys.nl),
    :qto        => zeros(sys.nl),
    :acline_pfr => zeros(sys.nl),
    :acline_pto => zeros(sys.nl),
    :acline_qfr => zeros(sys.nl),
    :acline_qto => zeros(sys.nl),
    :acline_sfr => zeros(sys.nl),
    :acline_sto => zeros(sys.nl),
    :acline_sfr_plus => zeros(sys.nl),
    :acline_sto_plus => zeros(sys.nl),
    :cos_ftp_x    => zeros(sys.nx),
    :sin_ftp_x    => zeros(sys.nx),
    :vff_x        => zeros(sys.nx),
    :vtt_x        => zeros(sys.nx),
    :vft_x        => zeros(sys.nx),
    :vt_tau_x     => zeros(sys.nx),
    :vf_tau_x     => zeros(sys.nx),
    :vf_tau2_x    => zeros(sys.nx),
    :vff_tau2_x   => zeros(sys.nx),
    :vft_tau_x    => zeros(sys.nx),
    :vft_tau2_x   => zeros(sys.nx),
    :vff_tau3_x   => zeros(sys.nx),
    :pfr_x        => zeros(sys.nx),
    :pto_x        => zeros(sys.nx),
    :qfr_x        => zeros(sys.nx),
    :qto_x        => zeros(sys.nx),
    :xfm_pfr_x    => zeros(sys.nx),
    :xfm_pto_x    => zeros(sys.nx),
    :xfm_qfr_x    => zeros(sys.nx),
    :xfm_qto_x    => zeros(sys.nx),
    :xfm_sfr_x    => zeros(sys.nx),
    :xfm_sto_x    => zeros(sys.nx),
    :xfm_sfr_plus_x  => zeros(sys.nx),
    :xfm_sto_plus_x  => zeros(sys.nx),
    :acline_scale_fr => zeros(sys.nl),
    :acline_scale_to => zeros(sys.nl),
    :scale_fr_x      => zeros(sys.nx),
    :scale_to_x      => zeros(sys.nx),
    :vm2_sh          => zeros(sys.nsh),
    :g_tv_shunt      => zeros(sys.nsh),
    :b_tv_shunt      => zeros(sys.nsh),
    :vmfrpfr         => zeros(sys.nl),
    :vmtopfr         => zeros(sys.nl),
    :vafrpfr         => zeros(sys.nl),
    :vatopfr         => zeros(sys.nl),
    :uonpfr          => zeros(sys.nl),
    :vmfrqfr         => zeros(sys.nl),
    :vmtoqfr         => zeros(sys.nl),
    :vafrqfr         => zeros(sys.nl),
    :vatoqfr         => zeros(sys.nl),
    :uonqfr          => zeros(sys.nl),
    :vmfrpto         => zeros(sys.nl),
    :vmtopto         => zeros(sys.nl),
    :vafrpto         => zeros(sys.nl),
    :vatopto         => zeros(sys.nl),
    :uonpto          => zeros(sys.nl),
    :vmfrqto         => zeros(sys.nl),
    :vmtoqto         => zeros(sys.nl),
    :vafrqto         => zeros(sys.nl),
    :vatoqto         => zeros(sys.nl),
    :uonqto          => zeros(sys.nl))

# in this file, we design the function which solves economic dispatch
#
# note -- this is ALWAYS run after clipping and fixing various states
function solve_economic_dispatch_with_sus!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}})
    # note: all binaries are LP relaxed (so there is not BaB-ing): 0 < b < 1
    #
    # NOTE -- we are not including start-up-state discounts -- not worth it :)

    # build and empty the model!
    model = Model(Gurobi.Optimizer; add_bridges = false)
    set_string_names_on_creation(model, false)
    set_silent(model)

    # FYI -- no startup states!
    @info "Economic dispatch neglects startup states"

    # quiet down!!!
    # alternative: => quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

    # set model properties => let this run until it finishes
        # quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
        # quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
        # quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)

    # define local time keys
    tkeys = prm.ts.time_keys

    # define the minimum set of variables we will need to solve the constraints
    u_on_dev  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:u_on_dev][tkeys[ii]][dev],  lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT)) # => base_name = "u_on_dev_t$(ii)",  
    p_on      = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_on][tkeys[ii]][dev])                                            for ii in 1:(sys.nT)) # => base_name = "p_on_t$(ii)",      
    dev_q     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:dev_q][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "dev_q_t$(ii)",     
    p_rgu     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_rgu][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_rgu_t$(ii)",     
    p_rgd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_rgd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_rgd_t$(ii)",     
    p_scr     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_scr][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_scr_t$(ii)",     
    p_nsc     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_nsc][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_nsc_t$(ii)",     
    p_rru_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_rru_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_rru_on_t$(ii)",  
    p_rru_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_rru_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_rru_off_t$(ii)", 
    p_rrd_on  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_rrd_on][tkeys[ii]][dev],  lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_rrd_on_t$(ii)",  
    p_rrd_off = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:p_rrd_off][tkeys[ii]][dev], lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "p_rrd_off_t$(ii)", 
    q_qru     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:q_qru][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "q_qru_t$(ii)",     
    q_qrd     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:q_qrd][tkeys[ii]][dev],     lower_bound = 0.0)                    for ii in 1:(sys.nT)) # => base_name = "q_qrd_t$(ii)",     

    # add a few more (implicit) variables which are necessary for solving this system
    u_su_dev  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:u_su_dev][tkeys[ii]][dev], lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT)) # => base_name = "u_su_dev_t$(ii)", 
    u_sd_dev  = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev], start=stt[:u_sd_dev][tkeys[ii]][dev], lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT)) # => base_name = "u_sd_dev_t$(ii)", 
    u_sus_dev = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [dev = 1:sys.ndev],                                       lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT)) # => base_name = "u_sus_dev_t$(ii)", 
    
    # we have the affine "AffExpr" expressions (whose values are specified)
    dev_p   = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
    p_su    = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
    p_sd    = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))
    zen_dev = Dict{Symbol, Vector{AffExpr}}(tkeys[ii] => Vector{AffExpr}(undef, sys.ndev) for ii in 1:(sys.nT))

    # now, we need to loop and set the affine expressions to 0
    #   -> see: https://jump.dev/JuMP.jl/stable/manual/expressions/
    for tii in prm.ts.time_keys
        for dev in 1:sys.ndev
            dev_p[tii][dev]   = AffExpr(0.0)
            p_su[tii][dev]    = AffExpr(0.0)
            p_sd[tii][dev]    = AffExpr(0.0)
            zen_dev[tii][dev] = AffExpr(0.0)
        end
    end

    # add scoring variables and affine terms
    p_rgu_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgu_zonal_REQ_t$(ii)",    
    p_rgd_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgd_zonal_REQ_t$(ii)",    
    p_scr_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_scr_zonal_REQ_t$(ii)",    
    p_nsc_zonal_REQ     = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_nsc_zonal_REQ_t$(ii)",    
    p_rgu_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgu_zonal_penalty_t$(ii)",
    p_rgd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rgd_zonal_penalty_t$(ii)",
    p_scr_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_scr_zonal_penalty_t$(ii)",
    p_nsc_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_nsc_zonal_penalty_t$(ii)",
    p_rru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rru_zonal_penalty_t$(ii)",
    p_rrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzP], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "p_rrd_zonal_penalty_t$(ii)",
    q_qru_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qru_zonal_penalty_t$(ii)",
    q_qrd_zonal_penalty = Dict{Symbol, Vector{quasiGrad.VariableRef}}(tkeys[ii] => @variable(model, [1:sys.nzQ], lower_bound = 0.0) for ii in 1:(sys.nT)) # => base_name = "q_qrd_zonal_penalty_t$(ii)",
    
    # affine aggregation terms
    zt      = AffExpr(0.0)
    z_enmax = AffExpr(0.0)
    z_enmin = AffExpr(0.0)

    # loop over all devices
    for dev in 1:sys.ndev

        # == define active power constraints ==
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # first, get the startup power
            T_supc     = idx.Ts_supc[dev][t_ind]     # T_set, p_supc_set = get_supc(tii, dev, prm)
            p_supc_set = idx.ps_supc_set[dev][t_ind] # T_set, p_supc_set = get_supc(tii, dev, prm)
            add_to_expression!(p_su[tii][dev], sum(p_supc_set[ii]*u_su_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_supc); init=0.0))

            # second, get the shutdown power
            T_sdpc     = idx.Ts_sdpc[dev][t_ind]     # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
            p_sdpc_set = idx.ps_sdpc_set[dev][t_ind] # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
            add_to_expression!(p_sd[tii][dev], sum(p_sdpc_set[ii]*u_sd_dev[tii_inst][dev] for (ii,tii_inst) in enumerate(T_sdpc); init=0.0))

            # finally, get the total power balance
            dev_p[tii][dev] = p_on[tii][dev] + p_su[tii][dev] + p_sd[tii][dev]
        end

        # == define reactive power constraints ==
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # only a subset of devices will have a reactive power equality constraint
            if dev in idx.J_pqe

                # the following (pr vs cs) are equivalent
                if dev in idx.pr_devs
                    # producer?
                    T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)
                    
                    # compute q -- this might be the only equality constraint (and below)
                    @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
                else
                    # the device must be a consumer :)
                    T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                    # compute q -- this might be the only equality constraint (and above)
                    @constraint(model, dev_q[tii][dev] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii][dev])
                end
            end
        end

        # loop over each time period and define the hard constraints
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # duration
            dt = prm.ts.duration[tii]

            # 1. Minimum downtime: zhat_mndn
            T_mndn = idx.Ts_mndn[dev][t_ind] # t_set = get_tmindn(tii, dev, prm)
            @constraint(model, u_su_dev[tii][dev] + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_mndn; init=0.0) - 1.0 <= 0.0)

            # 2. Minimum uptime: zhat_mnup
            T_mnup = idx.Ts_mnup[dev][t_ind] # t_set = get_tminup(tii, dev, prm)
            @constraint(model, u_sd_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_mnup; init=0.0) - 1.0 <= 0.0)

            # define the previous power value (used by both up and down ramping!)
            if tii == :t1
                # note: p0 = prm.dev.init_p[dev]
                dev_p_previous = prm.dev.init_p[dev]
            else
                # grab previous time
                tii_m1 = prm.ts.time_keys[t_ind-1]
                dev_p_previous = dev_p[tii_m1][dev]
            end

            # 3. Ramping limits (up): zhat_rup
            @constraint(model, dev_p[tii][dev] - dev_p_previous
                    - dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii][dev] - u_su_dev[tii][dev])
                    +     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii][dev] + 1.0 - u_on_dev[tii][dev])) <= 0.0)

            # 4. Ramping limits (down): zhat_rd
            @constraint(model,  dev_p_previous - dev_p[tii][dev]
                    - dt*(prm.dev.p_ramp_down_ub[dev]*u_on_dev[tii][dev]
                    +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-u_on_dev[tii][dev])) <= 0.0)

            # 5. Regulation up: zhat_rgu
            @constraint(model, p_rgu[tii][dev] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[tii][dev] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] - prm.dev.p_syn_res_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0 - u_on_dev[tii][dev]) <= 0.0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[tii][dev] <= 0.0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[tii][dev]) <= 0.0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, p_on[tii][dev] + p_rgu[tii][dev] + p_scr[tii][dev] + p_rru_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii][dev] <= 0.0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii][dev] + p_rrd_on[tii][dev] + p_rgd[tii][dev] - p_on[tii][dev] <= 0.0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_nsc[tii][dev] + p_rru_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, dev_q[tii][dev] + q_qru[tii][dev] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[tii][dev] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii][dev] <= 0.0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[tii][dev] + q_qru[tii][dev] - prm.dev.q_0_ub[dev]*u_sum
                    - prm.dev.beta_ub[dev]*dev_p[tii][dev] <= 0.0)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_sum + 
                        prm.dev.beta_lb[dev]*dev_p[tii][dev] + 
                        q_qrd[tii][dev] - dev_q[tii][dev] <= 0.0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, p_on[tii][dev] + p_rgd[tii][dev] + p_rrd_on[tii][dev] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii][dev] <= 0.0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii][dev] + p_rru_on[tii][dev] + p_scr[tii][dev] + p_rgu[tii][dev] - p_on[tii][dev] <= 0.0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, p_su[tii][dev] + p_sd[tii][dev] + p_rrd_off[tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii][dev]) <= 0.0)

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                u_sum  = u_on_dev[tii][dev] + sum(u_su_dev[tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, dev_q[tii][dev] + q_qrd[tii][dev] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0.0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[tii][dev] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii][dev] <= 0.0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[tii][dev] + q_qrd[tii][dev] - prm.dev.q_0_ub[dev]*u_sum
                    - prm.dev.beta_ub[dev]*dev_p[tii][dev] <= 0.0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                    + prm.dev.beta_lb[dev]*dev_p[tii][dev]
                    + q_qru[tii][dev] - dev_q[tii][dev] <= 0.0)
                end
            end
        end

        # misc penalty: maximum starts over multiple periods
        for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
            # get the time periods: zhat_mxst
            T_su_max = idx.Ts_su_max[dev][w_ind] #get_tsumax(w_params, prm)
            @constraint(model, sum(u_su_dev[tii][dev] for tii in T_su_max; init=0.0) - w_params[3] <= 0.0)
        end

        # now, we need to add two other sorts of constraints:
        # 1. "evolutionary" constraints which link startup and shutdown variables
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            if tii == :t1
                @constraint(model, u_on_dev[tii][dev] - prm.dev.init_on_status[dev] == u_su_dev[tii][dev] - u_sd_dev[tii][dev])
            else
                tii_m1 = prm.ts.time_keys[t_ind-1]
                @constraint(model, u_on_dev[tii][dev] - u_on_dev[tii_m1][dev] == u_su_dev[tii][dev] - u_sd_dev[tii][dev])
            end
            # only one can be nonzero
            @constraint(model, u_su_dev[tii][dev] + u_sd_dev[tii][dev] <= 1.0)
        end

        # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "upd"
        # upd = update states
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # if a device is *not* in the set of variables,
            # then it must be held constant! -- otherwise, try to hold it
            # close to its initial value
            if dev ∉ upd[:u_on_dev][tii]
                @constraint(model, u_on_dev[tii][dev] == stt[:u_on_dev][tii][dev])
            end

            if dev ∉ upd[:p_rrd_off][tii]
                @constraint(model, p_rrd_off[tii][dev] == stt[:p_rrd_off][tii][dev])
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[tii][dev] == stt[:p_nsc][tii][dev])
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[tii][dev] == stt[:p_rru_off][tii][dev])
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[tii][dev] == stt[:q_qru][tii][dev])
            end

            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[tii][dev] == stt[:q_qrd][tii][dev])
            end

            # now, deal with reactive powers, some of which are specified with equality
            # only a subset of devices will have a reactive power equality constraint
            #
            # nothing here :)
        end
    end

    # now, include a "copper plate" power balance constraint
    # loop over each time period and compute the power balance
    for tii in prm.ts.time_keys
        # duration
        dt = prm.ts.duration[tii]

        # power must balance at each time!
        sum_p   = AffExpr(0.0)
        sum_q   = AffExpr(0.0)

        # loop over each bus
        for bus in 1:sys.nb
            # active power balance:
            bus_p = +sum(dev_p[tii][dev] for dev in idx.cs[bus]; init=0.0) +
                    -sum(dev_p[tii][dev] for dev in idx.pr[bus]; init=0.0)
            add_to_expression!(sum_p, bus_p)

            # reactive power balance:
            bus_q = +sum(dev_q[tii][dev] for dev in idx.cs[bus]; init=0.0) +
                    -sum(dev_q[tii][dev] for dev in idx.pr[bus]; init=0.0)
            add_to_expression!(sum_q, bus_q)
        end

        # sum of active and reactive powers is 0
        @constraint(model, sum_p == 0.0)
        @constraint(model, sum_q == 0.0)
    end

    # ========== costs! ============= #
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # active power costs
        for dev in 1:sys.ndev
            # note -- these were sorted previously!
            cst = prm.dev.cum_cost_blocks[dev][t_ind][1][2:end]  # cost for each block (trim leading 0)
            pbk = prm.dev.cum_cost_blocks[dev][t_ind][2][2:end]  # power in each block (trim leading 0)
            nbk = length(pbk)

            # define a set of intermediate vars "p_jtm"
            p_jtm = @variable(model, [1:nbk], lower_bound = 0.0)
            @constraint(model, p_jtm .<= pbk)

            # have the blocks sum to the output power
            @constraint(model, sum(p_jtm) == dev_p[tii][dev])

            # compute the cost!
            zen_dev[tii][dev] = dt*sum(cst.*p_jtm)

            # start up states (sus) <----- fix this
            @constraint(model, sum(u_sus_dev[tii][dev]; init=0.0)  <= u_su_dev[tii][dev])
        end
    end
            
    # compute the costs associated with device reserve offers -- computed directly in the objective
    # 
    # min/max energy requirements
    for dev in 1:sys.ndev
        Wub = prm.dev.energy_req_ub[dev]
        Wlb = prm.dev.energy_req_lb[dev]

        # upper bounds
        for (w_ind, w_params) in enumerate(Wub)
            T_en_max = idx.Ts_en_max[dev][w_ind]
            zw_enmax = @variable(model, lower_bound = 0.0)
            @constraint(model, prm.vio.e_dev*(sum(prm.ts.duration[tii]*dev_p[tii][dev] for tii in T_en_max; init=0.0) - w_params[3]) <= zw_enmax)
            add_to_expression!(z_enmax, -1.0, zw_enmax)
        end

        # lower bounds
        for (w_ind, w_params) in enumerate(Wlb)
            T_en_min = idx.Ts_en_min[dev][w_ind]
            zw_enmin = @variable(model, lower_bound = 0.0)
            @constraint(model, prm.vio.e_dev*(w_params[3] - sum(prm.ts.duration[tii]*dev_p[tii][dev] for tii in T_en_min; init=0.0)) <= zw_enmin)
            add_to_expression!(z_enmin, -1.0, zw_enmin)
        end
    end

    # loop over reserves
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # loop over the zones (active power)
        for zone in 1:sys.nzP
            # endogenous sum
            if idx.cs_pzone[zone] == []
                # in the case there are NO consumers in a zone
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, p_rgu_zonal_REQ[tii][zone] == prm.reserve.rgu_sigma[zone]*sum(dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
                @constraint(model, p_rgd_zonal_REQ[tii][zone] == prm.reserve.rgd_sigma[zone]*sum(dev_p[tii][dev] for dev in idx.cs_pzone[zone]))
            end

            # endogenous max
            if idx.pr_pzone[zone] == []
                # in the case there are NO producers in a zone
                @constraint(model, p_scr_zonal_REQ[tii][zone] == 0.0)
                @constraint(model, p_nsc_zonal_REQ[tii][zone] == 0.0)
            else
                @constraint(model, prm.reserve.scr_sigma[zone]*[dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_scr_zonal_REQ[tii][zone])
                @constraint(model, prm.reserve.nsc_sigma[zone]*[dev_p[tii][dev] for dev in idx.pr_pzone[zone]] .<= p_nsc_zonal_REQ[tii][zone])
            end

            # balance equations -- compute the shortfall values
            #
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_pzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] <= p_rgu_zonal_penalty[tii][zone])
                
                @constraint(model, p_rgd_zonal_REQ[tii][zone] <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][t_ind] <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] <= p_rrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, p_rgu_zonal_REQ[tii][zone] - 
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgu_zonal_penalty[tii][zone])

                @constraint(model, p_rgd_zonal_REQ[tii][zone] - 
                                sum(p_rgd[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rgd_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) <= p_scr_zonal_penalty[tii][zone])

                @constraint(model, p_rgu_zonal_REQ[tii][zone] + 
                                p_scr_zonal_REQ[tii][zone] +
                                p_nsc_zonal_REQ[tii][zone] -
                                sum(p_rgu[tii][dev] for dev in idx.dev_pzone[zone]) -
                                sum(p_scr[tii][dev] for dev in idx.dev_pzone[zone]) - 
                                sum(p_nsc[tii][dev] for dev in idx.dev_pzone[zone]) <= p_nsc_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rru_min[zone][t_ind] -
                                sum(p_rru_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rru_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.rrd_min[zone][t_ind] -
                                sum(p_rrd_on[tii][dev]  for dev in idx.dev_pzone[zone]) - 
                                sum(p_rrd_off[tii][dev] for dev in idx.dev_pzone[zone]) <= p_rrd_zonal_penalty[tii][zone])
            end
        end

        # loop over the zones (reactive power) -- gradients are computed in the master grad
        for zone in 1:sys.nzQ
            # we want to safely avoid sum(...; init=0.0)
            if isempty(idx.dev_qzone[zone])
                # in this case, we assume all sums are 0!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] <= q_qrd_zonal_penalty[tii][zone])
            else
                # is this case, sums are what they are!!
                @constraint(model, prm.reserve.qru_min[zone][t_ind] -
                                sum(q_qru[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qru_zonal_penalty[tii][zone])

                @constraint(model, prm.reserve.qrd_min[zone][t_ind] -
                                sum(q_qrd[tii][dev] for dev in idx.dev_qzone[zone]) <= q_qrd_zonal_penalty[tii][zone])
            end
        end

        # shortfall penalties -- NOT needed explicitly (see objective)
    end

    # loop -- NOTE -- we are not including start-up-state discounts -- not worth it
    zt = AffExpr(0.0)
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]
        
        # add up
        zt_temp = @expression(model,
            # consumer revenues (POSITIVE)
            sum(zen_dev[tii][dev] for dev in idx.cs_devs) - 
            # producer costs
            sum(zen_dev[tii][dev] for dev in idx.pr_devs) - 
            # startup costs
            sum(prm.dev.startup_cost.*u_su_dev[tii]) - 
            # shutdown costs
            sum(prm.dev.shutdown_cost.*u_sd_dev[tii]) - 
            # on-costs
            sum(dt*prm.dev.on_cost.*u_on_dev[tii]) - 
            # time-dependent su costs
            sum(stt[:zsus_dev][tii]) -
            # local reserve penalties
            sum(dt*prm.dev.p_reg_res_up_cost_tmdv[t_ind].*p_rgu[tii]) -   # zrgu
            sum(dt*prm.dev.p_reg_res_down_cost_tmdv[t_ind].*p_rgd[tii]) - # zrgd
            sum(dt*prm.dev.p_syn_res_cost_tmdv[t_ind].*p_scr[tii]) -      # zscr
            sum(dt*prm.dev.p_nsyn_res_cost_tmdv[t_ind].*p_nsc[tii]) -     # znsc
            sum(dt*(prm.dev.p_ramp_res_up_online_cost_tmdv[t_ind].*p_rru_on[tii] +
                    prm.dev.p_ramp_res_up_offline_cost_tmdv[t_ind].*p_rru_off[tii])) -   # zrru
            sum(dt*(prm.dev.p_ramp_res_down_online_cost_tmdv[t_ind].*p_rrd_on[tii] +
                    prm.dev.p_ramp_res_down_offline_cost_tmdv[t_ind].*p_rrd_off[tii])) - # zrrd
            sum(dt*prm.dev.q_res_up_cost_tmdv[t_ind].*q_qru[tii]) -   # zqru      
            sum(dt*prm.dev.q_res_down_cost_tmdv[t_ind].*q_qrd[tii]) - # zqrd
            # zonal reserve penalties (P)
            sum(dt*prm.vio.rgu_zonal.*p_rgu_zonal_penalty[tii]) -
            sum(dt*prm.vio.rgd_zonal.*p_rgd_zonal_penalty[tii]) -
            sum(dt*prm.vio.scr_zonal.*p_scr_zonal_penalty[tii]) -
            sum(dt*prm.vio.nsc_zonal.*p_nsc_zonal_penalty[tii]) -
            sum(dt*prm.vio.rru_zonal.*p_rru_zonal_penalty[tii]) -
            sum(dt*prm.vio.rrd_zonal.*p_rrd_zonal_penalty[tii]) -
            # zonal reserve penalties (Q)
            sum(dt*prm.vio.qru_zonal.*q_qru_zonal_penalty[tii]) -
            sum(dt*prm.vio.qrd_zonal.*q_qrd_zonal_penalty[tii]))

        # update zt
        add_to_expression!(zt, zt_temp)
    end
    
    # set the objective
    @objective(model, Max, zt + z_enmax + z_enmin)

    # solve
    optimize!(model)

    # test solution!
    soln_valid = solution_status(model)

    # did Gurobi find something valid?
    if soln_valid == true
        println("========================================================")
        println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
        println("========================================================")

        # solve, and then return the solution
        for tii in prm.ts.time_keys
            stt[:u_on_dev][tii]  = copy.(value.(u_on_dev[tii]))
            stt[:p_on][tii]      = copy.(value.(p_on[tii]))
            stt[:dev_q][tii]     = copy.(value.(dev_q[tii]))
            stt[:p_rgu][tii]     = copy.(value.(p_rgu[tii]))
            stt[:p_rgd][tii]     = copy.(value.(p_rgd[tii]))
            stt[:p_scr][tii]     = copy.(value.(p_scr[tii]))
            stt[:p_nsc][tii]     = copy.(value.(p_nsc[tii]))
            stt[:p_rru_on][tii]  = copy.(value.(p_rru_on[tii]))
            stt[:p_rru_off][tii] = copy.(value.(p_rru_off[tii]))
            stt[:p_rrd_on][tii]  = copy.(value.(p_rrd_on[tii]))
            stt[:p_rrd_off][tii] = copy.(value.(p_rrd_off[tii]))
            stt[:q_qru][tii]     = copy.(value.(q_qru[tii]))
            stt[:q_qrd][tii]     = copy.(value.(q_qrd[tii]))
        end

        # update the u_sum and powers (used in clipping, so must be correct!)
        qG.run_susd_updates = true
        quasiGrad.simple_device_statuses!(idx, prm, qG, stt)
        quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

        # update the objective value score
        scr[:ed_obj] = objective_value(model)
    else
        # warn!
        @warn "Copper plate economic dispatch (LP) failed -- skip initialization!"
    end
end


function dcvm_initialization!(flw::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, idx::quasiGrad.Idx, ntk::quasiGrad.Ntk, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # apply vm-dcpf to the economic dispatch solution -- again, doesn't really work :)
    #
    qinj = zeros(sys.nb)   # this will be overwritten
    vmr  = zeros(sys.nb-1) # this will be overwritten

    @warn "this doesn't work"
    for tii in prm.ts.time_keys
        # first, update the xfm phase shifters (whatever they may be..)
        flw[:ac_phi][idx.ac_phi] = stt[:phi][tii]

        # loop over each bus
        for bus in 1:sys.nb
        # reactive power balance
            qinj[bus] = 
               sum(stt[:dev_q][tii][idx.pr[bus]]; init=0.0) - 
               sum(stt[:dev_q][tii][idx.cs[bus]]; init=0.0) - 
               sum(stt[:sh_q][tii][idx.sh[bus]]; init=0.0) - 
               sum(stt[:dc_qfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
               sum(stt[:dc_qto][tii][idx.bus_is_dc_tos[bus]]; init=0.0)
        end

        # solve
        if sys.nb <= qG.min_buses_for_krylov

            # voltage magnitude
            stt[:vm][tii][2:end] = ntk.Ybr\qinj[2:end] .+ 1.0 #stt[:vm][tii][1]
        else

            # solve with pcg -- vm
            quasiGrad.cg!(vmr, ntk.Ybr, qinj[2:end], abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its)
            
            # cg has failed in the past -- not sure why -- test for NaN
            if isnan(sum(vmr)) || maximum(vmr) > 1e7  # => faster than any(isnan, t)
                # LU backup
                @info "Krylov failed -- using LU backup (dcpf)!"
                vmr = ntk.Ybr\qinj[2:end]
             end

            # voltage magnitude
            stt[:vm][tii][2:end] = vmr .+ 1.0
        end
    end
end








    
#
# first, get the ctg limits
s_max_ctg = [prm.acline.mva_ub_em; prm.xfm.mva_ub_em]

# get the ordered names of all components
ac_ids = [prm.acline.id; prm.xfm.id ]

# get the ordered (negative!!) susceptances
ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]

# build the full incidence matrix: E = lines x buses
E  = quasiGrad.build_incidence(idx, prm, sys)
Er = E[:,2:end]
ErT = copy(Er')

# get the diagonal admittance matrix   => Ybs == "b susceptance"
Ybs = quasiGrad.spdiagm(ac_b_params)
Yb  = E'*Ybs*E
Ybr = Yb[2:end,2:end]  # use @view ? 

# should we precondition the base case?
#
# Note: Ybr should be sparse!! otherwise,
# the sparse preconditioner won't have any memory limits and
# will be the full Chol-decomp -- not a big deal, I guess..
if qG.base_solver == "pcg"
    if sys.nb <= qG.min_buses_for_krylov
        # too few buses -- use LU
        @warn "Not enough buses for Krylov! using LU."
        # => Ybr_ChPr = quasiGrad.I
        Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
    else
        # test for negative reactances -- @info "Preconditioning is disabled."
        if minimum(ac_b_params) < 0.0
            # Amrit Pandey: "watch out for negatvive reactance! You will lose
            #                pos-sem-def of the Cholesky preconditioner."
            abs_b    = abs.(ac_b_params)
            abs_Ybr  = (E'*quasiGrad.spdiagm(abs_b)*E)[2:end,2:end] 
            Ybr_ChPr = quasiGrad.CholeskyPreconditioner(abs_Ybr, qG.cutoff_level)
        else
            Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
        end
    end
else
    # => Ybr_ChPr = quasiGrad.I
    Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);
end

# should we build the cholesky decomposition of the base case
# admittance matrix? we build this to compute high-fidelity
# solutions of the rank-1 update matrices ldlt(A)
if qG.build_basecase_cholesky
    #Ybr_Ch = quasiGrad.cholesky(Ybr)
    Ybr_Ch = quasiGrad.ldlt(Ybr)
else
    Ybr_Ch = quasiGrad.I
end

# get the flow matrix
Yfr  = Ybs*Er
YfrT = copy(Yfr')

# build the low-rank contingecy updates
#
# base: Y_b*theta_b = p
# ctg:  Y_c*theta_c = p
#       Y_c = Y_b + uk'*uk
ctg_out_ind = Dict(ctg_ii => Vector{Int64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
ctg_params  = Dict(ctg_ii => Vector{Float64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)

# should we build the full ctg matrices?
if qG.build_ctg_full == true
    nac   = sys.nac
    Ybr_k = Dict(ctg_ii => quasiGrad.spzeros(nac,nac) for ctg_ii in 1:sys.nctg)
else
    # build something small of the correct data type
    Ybr_k = Dict(1 => quasiGrad.spzeros(1,1))
end

# and/or, should we build the low rank ctg elements?
if qG.build_ctg_lowrank == true
    # no need => v_k = Dict(ctg_ii => quasiGrad.spzeros(nac) for ctg_ii in 1:sys.nctg)
    # no need => b_k = Dict(ctg_ii => 0.0 for ctg_ii in 1:sys.nctg)
    u_k = Dict(ctg_ii => zeros(sys.nb-1) for ctg_ii in 1:sys.nctg) # Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
    g_k = Dict(ctg_ii => 0.0             for ctg_ii in 1:sys.nctg)
    # if the "w_k" formulation is wanted => w_k = Dict(ctg_ii => quasiGrad.spzeros(sys.nb-1) for ctg_ii in 1:sys.nctg)
else
    v_k = 0
    b_k = 0
end

for ctg_ii in 1:sys.nctg
    # components
    cmpnts = prm.ctg.components[ctg_ii]
    for (cmp_ii,cmp) in enumerate(cmpnts)
        # get the cmp index and b
        cmp_index = findfirst(x -> x == cmp, ac_ids) 
        cmp_b     = -ac_b_params[cmp_index] # negative, because we subtract it out

        # output
        ctg_out_ind[ctg_ii][cmp_ii] = cmp_index
        ctg_params[ctg_ii][cmp_ii]  = cmp_b

        # -> y_diag[cmp_index] = sqrt(cmp_b)
            # we record these in ctg
            # ctg_out_ind[ctg_ii]
    end

    # next, should we build the actual, full ctg matrix?
    if qG.build_ctg_full == true
        # direct construction..
        #
        # NOTE: this is written assuming multiple elements can be
        # simultaneously outaged
        Ybs_k = copy(Ybs)
        Ybs_k[CartesianIndex.(tuple.(ctg_out_ind[ctg_ii],ctg_out_ind[ctg_ii]))] .= 0.0
        Ybr_k[ctg_ii] = Er'*Ybs_k*Er
    end

    # and/or, should we build the low rank ctg elements?
    if qG.build_ctg_lowrank == true
        # .. vs low rank
        #
        # NOTE: this is written assuming only ONE element
        # can be outaged

        # no need to save:
            # v_k[ctg_ii] =  Er[ctg_out_ind[ctg_ii][1],:]
            # b_k[ctg_ii] = -ac_b_params[ctg_out_ind[ctg_ii][1]]
        v_k = Er[ctg_out_ind[ctg_ii][1],:]
        b_k = -ac_b_params[ctg_out_ind[ctg_ii][1]]
        #
        # construction: 
        # 
        # Ybr_k[ctg_ii] = ctg[:Ybr] + v*beta*v'
        #               = ctg[:Ybr] + vLR_k[ctg_ii]*beta*vLR_k[ctg_ii]
        #
        # if v, b saved:
            # u_k[ctg_ii] = Ybr\Array(v_k[ctg_ii])
            # w_k[ctg_ii] = b_k[ctg_ii]*u_k[ctg_ii]/(1+(v_k[ctg_ii]'*u_k[ctg_ii])*b_k[ctg_ii])
        # LU fac => u_k[ctg_ii] = Ybr\Vector(v_k)
            # this is very slow -- we need to us cg and then enforce sparsity!
            # Float64.(Vector(v_k)) is not needed! cg can handle sparse :)
            # quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl=Ybr_ChPr)
        # enforce sparsity -- should be sparse anyways
            # u_k[ctg_ii][abs.(u_k[ctg_ii]) .< 1e-8] .= 0.0

        # we want to sparsify a high-fidelity solution:
        # uk_d = Ybr_Ch\v_k[ctg_ii]
        # quasiGrad.cg!(u_k[ctg_ii], Ybr, Vector(Float64.(v_k)), abstol = qG.pcg_tol, Pl=Ybr_ChPr)
        # u_k[ctg_ii] = Ybr\Vector(v_k)
        # u_k[ctg_ii] = C\Vector(v_k)
        if qG.build_basecase_cholesky
            u_k_local = (Ybr_Ch\v_k)[:]
        else
            u_k_local = Ybr\Vector(v_k)
        end
        # sparsify
        abs_u_k           = abs.(u_k_local)
        u_k_ii_SmallToBig = sortperm(abs_u_k)
        bit_vec           = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> (1.0 - qG.accuracy_sparsify_lr_updates)
        # edge case is caught! bit_vec will never be empty. Say, abs_u_k[u_k_ii_SmallToBig] = [0,0,1], then we have
        # bit_vec = cumsum(abs_u_k[u_k_ii_SmallToBig])/sum(abs_u_k) .> 0.01%, say => bit_vec = [0,0,1] 
        # 
        # also, we use ".>" because we only want to include all elements that contribute to meeting the stated accuracy goal
        u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]] = u_k_local[u_k_ii_SmallToBig[bit_vec]]
        # this is ok, since u_k and w_k have the same sparsity pattern
        # => for the "w_k" formulation: w_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]] = b_k*u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]]/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii][u_k_ii_SmallToBig[bit_vec]]))*b_k)
        g_k[ctg_ii] = b_k/(1.0+(quasiGrad.dot(v_k,u_k[ctg_ii]))*b_k)
    end
end

# initialize ctg state
tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

# build the phase angle solution dict -- this will have "sys.nb-1" angles for each solution,
# since theta1 = 0, and it will have n_ctg+1 solutions, because the base case solution will be
# optionally saved at the end.. similar for pflow_k
# theta_k       = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1)) for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# pflow_k       = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nac))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# theta_k       = Dict(tkeys[ii] => [zeros(sys.nb-1) for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# pflow_k       = Dict(tkeys[ii] => [zeros(sys.nac)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))
# this is the gradient solution assuming a base case admittance (it is then rank 1 corrected to dz_dpinj)
# ctd = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 
# ctd = Dict(tkeys[ii] => [zeros(sys.nb-1)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT))   
# this is the gradient solution, corrected from ctd
# dz_dpinj      = Dict(tkeys[ii] => [Vector{Float64}(undef,(sys.nb-1))  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 
# dz_dpinj      = Dict(tkeys[ii] => [zeros(sys.nb-1)  for jj in 1:(sys.nctg+1)] for ii in 1:(sys.nT)) 

# "local" storage for apparent power flows (not needed across time)
# sfr     = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
# sto     = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
# sfr_vio = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)
# sto_vio = Dict(ctg_ii => zeros(sys.nac) for ctg_ii in 1:sys.nctg)   

# phase shift derivatives
#   => consider power injections:  pinj = (p_pr-p_cs-p_sh-p_fr_dc-p_to_dc-alpha*slack) + Er^T*phi*b
#      => Er^T*phi*b
# ~ skip the reference bus! -- fr_buses = positive in the incidence matrix; to_buses = negative..
xfm_at_bus      = Dict(bus => vcat(idx.bus_is_xfm_frs[bus],idx.bus_is_xfm_tos[bus]) for bus in 2:sys.nb)
xfm_at_bus_sign = Dict(bus => vcat(idx.bus_is_xfm_frs[bus],-idx.bus_is_xfm_tos[bus]) for bus in 2:sys.nb)
xfm_phi_scalars = Dict(bus => ac_b_params[xfm_at_bus[bus] .+ sys.nl].*sign.(xfm_at_bus_sign[bus]) for bus in 2:sys.nb)

# compute the constant acline Ybus matrix
Ybus_acline_real, Ybus_acline_imag = quasiGrad.initialize_acline_Ybus(idx, prm, sys)

# %% question if Ybr is NOT PSD, does the preconditioner still work pretty well?
#
# case 1: yes psd
Ybs      = quasiGrad.spdiagm(ac_b_params)
Yb       = E'*Ybs*E
Ybr      = Yb[2:end,2:end]  # use @view ? 
Ybr_ChPr = quasiGrad.CholeskyPreconditioner(Ybr, qG.cutoff_level);

# solve
b     = randn(616)
x     = randn(616)
t_ind = 1
_, ch = quasiGrad.cg!(x, Ybr, b, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr, maxiter = qG.max_pcg_its, log = true)
ch

# %% ==================
t_ind = 1

ac_b_params_neg      = copy(ac_b_params)
#ac_b_params_neg[5]   = -ac_b_params_neg[5]
#ac_b_params_neg[12]  = -ac_b_params_neg[12]
#ac_b_params_neg[18]  = -ac_b_params_neg[18]
#ac_b_params_neg[22]  = -ac_b_params_neg[22]
#ac_b_params_neg[55]  = -ac_b_params_neg[55]
#ac_b_params_neg[58]  = -ac_b_params_neg[58]
#ac_b_params_neg[59]  = -ac_b_params_neg[59]
#ac_b_params_neg[100] = -ac_b_params_neg[100]
#ac_b_params_neg[end] = -ac_b_params_neg[end]

Ybs_neg      = quasiGrad.spdiagm(ac_b_params_neg)
Yb_neg       = E'*Ybs_neg*E
Ybr_neg      = Yb_neg[2:end,2:end]  # use @view ? 
abs_b        = abs.(ac_b_params_neg)
abs_Ybr      = (E'*quasiGrad.spdiagm(abs_b)*E)[2:end,2:end] 
Ybr_ChPr_neg = quasiGrad.CholeskyPreconditioner(abs_Ybr, qG.cutoff_level);
Ybr_diag     = quasiGrad.DiagonalPreconditioner(Ybr_neg);
Ybr_ldl      = lldl(Ybr_neg, memory = qG.cutoff_level)

# solve
b = randn(616); 
x = copy(b)
@time _, ch = quasiGrad.cg!(x, Ybr_neg, b, abstol = qG.pcg_tol, Pl=Ybr_ChPr, maxiter = 1000, log = true)
println(ch)

x = copy(b)
@time _, ch = quasiGrad.cg!(x, Ybr_neg, b, abstol = qG.pcg_tol, Pl=Ybr_diag, maxiter = 1000, log = true)
println(ch)

x = copy(b)
@time _, ch = quasiGrad.cg!(x, Ybr_neg, b, abstol = qG.pcg_tol, Pl=Ybr_ldl, maxiter = 1000, log = true)
println(ch)

# %%

@btime lldl(Ybr_neg, memory = qG.cutoff_level)

function zctgs_grad_q_xfm!(tii::Symbol, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, xfr_inds::Vector{Int64}, xto_inds::Vector{Int64}, xfr_alpha::Vector{Float64}, xto_alpha::Vector{Float64})
    # so, this function takes and applies the gradient of
    # zctgs (at transformers) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # xfr_inds = xfms which are violated on the from side!
    # xto_inds = xfms which are violated on the to side!
    # xfr_alpha = associated partial
    # xto_alpha = associated partial
    #   example: xfms 1 (max overload on to), 2 and 3 (max overload on frm)
    #   xfr_inds = [3]
    #   xto_inds = [1]

    vmfrqfr = xfr_alpha.*grd[:xfm_qfr][:vmfr][tii][xfr_inds]
    vmtoqfr = xfr_alpha.*grd[:xfm_qfr][:vmto][tii][xfr_inds]
    vafrqfr = xfr_alpha.*grd[:xfm_qfr][:vafr][tii][xfr_inds]
    vatoqfr = xfr_alpha.*grd[:xfm_qfr][:vato][tii][xfr_inds]
    tauqfr  = xfr_alpha.*grd[:xfm_qfr][:tau][tii][xfr_inds]
    phiqfr  = xfr_alpha.*grd[:xfm_qfr][:phi][tii][xfr_inds]
    uonqfr  = xfr_alpha.*grd[:xfm_qfr][:uon][tii][xfr_inds]

    # final qfr gradients
    vmfrqto = xto_alpha.*grd[:xfm_qto][:vmfr][tii][xto_inds]
    vmtoqto = xto_alpha.*grd[:xfm_qto][:vmto][tii][xto_inds]
    vafrqto = xto_alpha.*grd[:xfm_qto][:vafr][tii][xto_inds]
    vatoqto = xto_alpha.*grd[:xfm_qto][:vato][tii][xto_inds]
    tauqto  = xto_alpha.*grd[:xfm_qto][:tau][tii][xto_inds]
    phiqto  = xto_alpha.*grd[:xfm_qto][:phi][tii][xto_inds]
    uonqto  = xto_alpha.*grd[:xfm_qto][:uon][tii][xto_inds]

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(xfr_inds)
        # update the master grad -- qfr
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqfr[ii]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqfr[ii]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqfr[ii]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqfr[ii]
        mgd[:tau][tii][xfm]                += tauqfr[ii]
        mgd[:phi][tii][xfm]                += phiqfr[ii]
        mgd[:u_on_xfm][tii][xfm]           += uonqfr[ii]
    end

    # note: we must loop over these assignments!
    for (ii,xfm) in enumerate(xto_inds)
        # update the master grad -- qto
        mgd[:vm][tii][idx.xfm_fr_bus[xfm]] += vmfrqto[ii]
        mgd[:vm][tii][idx.xfm_to_bus[xfm]] += vmtoqto[ii]
        mgd[:va][tii][idx.xfm_fr_bus[xfm]] += vafrqto[ii]
        mgd[:va][tii][idx.xfm_to_bus[xfm]] += vatoqto[ii]
        mgd[:tau][tii][xfm]                += tauqto[ii]
        mgd[:phi][tii][xfm]                += phiqto[ii]
        mgd[:u_on_xfm][tii][xfm]           += uonqto[ii]
    end
end

function zctgs_grad_q_acline!(tii::Symbol, idx::quasiGrad.Idx, grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, aclfr_inds::Vector{Int64}, aclto_inds::Vector{Int64}, aclfr_alpha::Vector{Float64}, aclto_alpha::Vector{Float64})
    # so, this function takes and applies the gradient of
    # zctgs (at acline) with repsect to reactive power
    # variables (i.e., all variables on a line which affect
    # reactive power flows).
    #
    # We compute and apply gradients for "fr" and "to" lines
    # as necessary. what are the incoming variables?
    #
    # more comments in the xfm function
    vmfrqfr = aclfr_alpha.*grd[:acline_qfr][:vmfr][tii][aclfr_inds]
    vmtoqfr = aclfr_alpha.*grd[:acline_qfr][:vmto][tii][aclfr_inds]
    vafrqfr = aclfr_alpha.*grd[:acline_qfr][:vafr][tii][aclfr_inds]
    vatoqfr = aclfr_alpha.*grd[:acline_qfr][:vato][tii][aclfr_inds]
    uonqfr  = aclfr_alpha.*grd[:acline_qfr][:uon][tii][aclfr_inds]

    # final qfr gradients
    vmfrqto = aclto_alpha.*grd[:acline_qto][:vmfr][tii][aclto_inds]
    vmtoqto = aclto_alpha.*grd[:acline_qto][:vmto][tii][aclto_inds]
    vafrqto = aclto_alpha.*grd[:acline_qto][:vafr][tii][aclto_inds]
    vatoqto = aclto_alpha.*grd[:acline_qto][:vato][tii][aclto_inds]
    uonqto  = aclto_alpha.*grd[:acline_qto][:uon][tii][aclto_inds]

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(aclfr_inds)
        # update the master grad -- qfr
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqfr[ii]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqfr[ii]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqfr[ii]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqfr[ii]
        mgd[:u_on_acline][tii][ln]           += uonqfr[ii]
    end

    # note: we must loop over these assignments!
    for (ii,ln) in enumerate(aclto_inds)
        # update the master grad -- qto
        mgd[:vm][tii][idx.acline_fr_bus[ln]] += vmfrqto[ii]
        mgd[:vm][tii][idx.acline_to_bus[ln]] += vmtoqto[ii]
        mgd[:va][tii][idx.acline_fr_bus[ln]] += vafrqto[ii]
        mgd[:va][tii][idx.acline_to_bus[ln]] += vatoqto[ii]
        mgd[:u_on_acline][tii][ln]           += uonqto[ii]
    end
end




function clip_for_feasibility!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # sequentially clip -- order does not matter
    #
    @warn "this isn't totally validated.. or used."
    # note: "clamp" is much faster than the alternatives!
    clip_onoff_binaries!(prm, stt)
    clip_reserves!(prm, stt)
    clip_pq!(prm, qG, stt)

    # target the problematic one
    clip_17c!(idx, prm, qG, stt, sys)
end

function clip_17c!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System)
    # loop over time/devices and look for violations
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        for dev in 1:sys.ndev
            if dev in idx.cs_devs
                val = stt[:q_qru][tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev]
                if val > 0.0
                    # is this case, q is below its lower bound!
                    #
                    # first, try to clip :q_qru, since this is safe
                    stt[:q_qru][tii][dev] = max(prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev], 0.0)

                    # did this work?
                    if stt[:q_qru][tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev] > 0
                        # if not, clip stt[:dev_q] (generally, not safe, but desperate times..)
                        stt[:dev_q][tii][dev] = stt[:q_qru][tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev]
                    end
                end
            end
        end
    end
end

function get_largest_indices(Dict{Symbol, Dict{Symbol, Vector{Float64}}}, bit::Dict{Symbol, Dict{Symbol, BitVector}}, s1::Symbol, s2::Symbol)
    for ii in 1:length(msc[s1])
        if (msc[s1][ii] >= msc[s2][ii]) && (msc[s1][ii] > 0.0)
            bit[s1][ii] = 1
            bit[s2][ii] = 0
        elseif msc[s2][ii] > 0.0 # no need to check v2[ii] > v1[ii]
            bit[s2][ii] = 1
            bit[s1][ii] = 0
        else
            bit[s1][ii] = 0  
            bit[s2][ii] = 0
        end
    end
end

# final, manual projection
function final_projection___BROKEN!(idx::quasiGrad.Idx, prm::quasiGrad.Param, qG::quasiGrad.QG, stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, sys::quasiGrad.System, upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}}; final_projection::Bool=false)
    
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # duration
        dt = prm.ts.duration[tii]

        # for now, we use "del" in the scoring function to penalize all
        # constraint violations -- thus, don't call the "c_hat" constants

        for dev in 1:sys.ndev
            # 1. Minimum downtime -- nothing to do here
            # 2. Minimum uptime  -- nothing to do here

            # define the previous power value (used by both up and down ramping!)
            if tii == :t1
                # note: p0 = prm.dev.init_p[dev]
                dev_p_previous = prm.dev.init_p[dev]
            else
                # grab previous time
                dev_p_previous = stt[:dev_p][prm.ts.tmin1[tii]][dev] 
            end

            # 3. Ramping limits (up) -- zhat_rup
            cvio = stt[:dev_p][tii][dev] - dev_p_previous
                    - dt*(prm.dev.p_ramp_up_ub[dev]     *(stt[:u_on_dev][tii][dev] - stt[:u_su_dev][tii][dev])
                    +     prm.dev.p_startup_ramp_ub[dev]*(stt[:u_su_dev][tii][dev] + 1.0 - stt[:u_on_dev][tii][dev]))
            
            if cvio > quasiGrad.eps_constr
                # clip
                if stt[:dev_p][tii][dev] > cvio
                    stt[:dev_p][tii][dev] -= cvio
                elseif tii != :tii
                    # last resort
                    stt[:dev_p][prm.ts.tmin1[tii]][dev] -= cvio
                end
            end


            # 4. Ramping limits (down)
            cvio = max(dev_p_previous - stt[:dev_p][tii][dev]
                    - dt*(prm.dev.p_ramp_down_ub[dev]*stt[:u_on_dev][tii][dev]
                    +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-stt[:u_on_dev][tii][dev])),0.0)
            stt[:zhat_rd][tii][dev] = dt* cvio


            # 5. Regulation up
            cvio                     = max(stt[:p_rgu][tii][dev] - prm.dev.p_reg_res_up_ub[dev]*stt[:u_on_dev][tii][dev], 0.0)
            stt[:zhat_rgu][tii][dev] = dt* cvio

            # 6. Regulation down
            cvio                     = max(stt[:p_rgd][tii][dev] - prm.dev.p_reg_res_down_ub[dev]*stt[:u_on_dev][tii][dev], 0.0)
            stt[:zhat_rgd][tii][dev] = dt* cvio

            # 7. Synchronized reserve
            cvio                     = max(stt[:p_rgu][tii][dev] + stt[:p_scr][tii][dev] - prm.dev.p_syn_res_ub[dev]*stt[:u_on_dev][tii][dev], 0.0)
            stt[:zhat_scr][tii][dev] = dt* cvio


            # 8. Synchronized reserve
            cvio                     = max(stt[:p_nsc][tii][dev] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - stt[:u_on_dev][tii][dev]), 0.0)
            stt[:zhat_nsc][tii][dev] = dt* cvio

            # 9. Ramping reserve up (on)
            cvio                       = max(stt[:p_rgu][tii][dev] + stt[:p_scr][tii][dev] + stt[:p_rru_on][tii][dev] - prm.dev.p_ramp_res_up_online_ub[dev]*stt[:u_on_dev][tii][dev], 0.0)
            stt[:zhat_rruon][tii][dev] = dt* cvio


            # 10. Ramping reserve up (off)
            cvio                        = max(stt[:p_nsc][tii][dev] + stt[:p_rru_off][tii][dev] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0-stt[:u_on_dev][tii][dev]), 0.0)
            stt[:zhat_rruoff][tii][dev] = dt* cvio

            # 11. Ramping reserve down (on)
            cvio                       = max(stt[:p_rgd][tii][dev] + stt[:p_rrd_on][tii][dev] - prm.dev.p_ramp_res_down_online_ub[dev]*stt[:u_on_dev][tii][dev], 0.0)
            stt[:zhat_rrdon][tii][dev] = dt* cvio

            # 12. Ramping reserve down (off)
            cvio                        = max(stt[:p_rrd_off][tii][dev] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-stt[:u_on_dev][tii][dev]), 0.0)
            stt[:zhat_rrdoff][tii][dev] = dt* cvio

            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers)
                cvio                      = max(stt[:p_on][tii][dev] + stt[:p_rgu][tii][dev] + stt[:p_scr][tii][dev] + stt[:p_rru_on][tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev], 0.0)
                stt[:zhat_pmax][tii][dev] = dt* cvio

                # 14p. Minimum reserve limits (producers)
                cvio                      = max(prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + stt[:p_rrd_on][tii][dev] + stt[:p_rgd][tii][dev] - stt[:p_on][tii][dev], 0.0)
                stt[:zhat_pmin][tii][dev] = dt* cvio
                
                # 15p. Off reserve limits (producers)
                cvio                         = max(stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + stt[:p_nsc][tii][dev] + stt[:p_rru_off][tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]), 0.0)
                stt[:zhat_pmaxoff][tii][dev] = dt* cvio

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][t_ind] # => get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][t_ind] # => get_sdpc(tii, dev, prm)
                stt[:u_sum][tii][dev] = stt[:u_on_dev][tii][dev] + sum(stt[:u_su_dev][tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(stt[:u_sd_dev][tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                # 16p. Maximum reactive power reserves (producers)
                cvio                      = max(stt[:dev_q][tii][dev] + stt[:q_qru][tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev], 0.0)
                stt[:zhat_qmax][tii][dev] = dt* cvio

                # 17p. Minimum reactive power reserves (producers)
                cvio                      = max(stt[:q_qrd][tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev], 0.0)
                stt[:zhat_qmin][tii][dev] = dt* cvio

                # 18p. Linked maximum reactive power reserves (producers)
                if dev in idx.J_pqmax
                    cvio                           = max(stt[:dev_q][tii][dev] + stt[:q_qru][tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev] - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev], 0.0)
                    stt[:zhat_qmax_beta][tii][dev] = dt* cvio

                end 
                
                # 19p. Linked minimum reactive power reserves (producers)
                if dev in idx.J_pqmin
                    cvio                           = max(prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev] + prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev] + stt[:q_qrd][tii][dev] - stt[:dev_q][tii][dev], 0.0)
                    stt[:zhat_qmin_beta][tii][dev] = dt* cvio
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers)
                cvio                      = max(stt[:p_on][tii][dev] + stt[:p_rgd][tii][dev] + stt[:p_rrd_on][tii][dev] - prm.dev.p_ub[dev][t_ind]*stt[:u_on_dev][tii][dev], 0.0)
                stt[:zhat_pmax][tii][dev] = dt* cvio

                # 14c. Minimum reserve limits (consumers)
                cvio                      = max(prm.dev.p_lb[dev][t_ind]*stt[:u_on_dev][tii][dev] + stt[:p_rru_on][tii][dev] + stt[:p_scr][tii][dev] + stt[:p_rgu][tii][dev] - stt[:p_on][tii][dev], 0.0)
                stt[:zhat_pmin][tii][dev] = dt* cvio

                # 15c. Off reserve limits (consumers)
                cvio                         = max(stt[:p_su][tii][dev] + stt[:p_sd][tii][dev] + stt[:p_rrd_off][tii][dev] - prm.dev.p_ub[dev][t_ind]*(1.0 - stt[:u_on_dev][tii][dev]), 0.0)
                stt[:zhat_pmaxoff][tii][dev] = dt* cvio

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][t_ind] # => get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][t_ind] # => get_sdpc(tii, dev, prm)
                stt[:u_sum][tii][dev] = stt[:u_on_dev][tii][dev] + sum(stt[:u_su_dev][tii_inst][dev] for tii_inst in T_supc; init=0.0) + sum(stt[:u_sd_dev][tii_inst][dev] for tii_inst in T_sdpc; init=0.0)

                # 16c. Maximum reactive power reserves (consumers)
                cvio                      = max(stt[:dev_q][tii][dev] + stt[:q_qrd][tii][dev] - prm.dev.q_ub[dev][t_ind]*stt[:u_sum][tii][dev], 0.0)
                stt[:zhat_qmax][tii][dev] = dt* cvio

                # 17c. Minimum reactive power reserves (consumers)
                cvio                      = max(stt[:q_qru][tii][dev] + prm.dev.q_lb[dev][t_ind]*stt[:u_sum][tii][dev] - stt[:dev_q][tii][dev], 0.0)
                stt[:zhat_qmin][tii][dev] = dt* cvio

                # 18c. Linked maximum reactive power reserves (consumers)
                if dev in idx.J_pqmax
                    cvio                           = max(stt[:dev_q][tii][dev] + stt[:q_qrd][tii][dev] - prm.dev.q_0_ub[dev]*stt[:u_sum][tii][dev] - prm.dev.beta_ub[dev]*stt[:dev_p][tii][dev], 0.0)
                    stt[:zhat_qmax_beta][tii][dev] = dt* cvio
                end 

                # 19c. Linked minimum reactive power reserves (consumers)
                if dev in idx.J_pqmin
                    cvio                           = max(prm.dev.q_0_lb[dev]*stt[:u_sum][tii][dev] + prm.dev.beta_lb[dev]*stt[:dev_p][tii][dev] + stt[:q_qru][tii][dev] - stt[:dev_q][tii][dev], 0.0)
                    stt[:zhat_qmin_beta][tii][dev] = dt* cvio
                end
            end
        end
    end
end