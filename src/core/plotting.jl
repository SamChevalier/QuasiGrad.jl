function run_adam_with_plotting!(
        adm::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}},
        ax::quasiGrad.Makie.Axis,
        cgd::quasiGrad.Cgd,
        ctb::Vector{Vector{Float64}},
        ctd::Vector{Vector{Float64}},
        fig::quasiGrad.Makie.Figure,
        flw::Dict{Symbol, Vector{Float64}},
        grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, 
        idx::quasiGrad.Idx,
        mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
        msc::Dict{Symbol, Vector{Float64}},
        ntk::quasiGrad.Ntk,
        plt::Dict{Symbol, Integer},
        prm::quasiGrad.Param,
        qG::quasiGrad.QG,
        scr::Dict{Symbol, Float64},
        stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
        sys::quasiGrad.System,
        upd::Dict{Symbol, Dict{Symbol, Vector{Int64}}},
        wct::Vector{Vector{Int64}},
        z_plt::Dict{Symbol, Dict{Symbol, Float64}})

    # initialize
    adm_step    = 0
    beta1       = qG.beta1
    beta2       = qG.beta2
    beta1_decay = 1.0
    beta2_decay = 1.0
    run_adam    = true
    
    # flush adam at each restart
    #if qG.flush_adam == true
    #    quasiGrad.flush_adam!(adm, mgd, prm, upd)
    #end

    # add Gurobi Projection line?
    if !plt[:first_plot]
        # add a dark vertical line
        quasiGrad.Makie.lines!(ax, [plt[:global_adm_step], plt[:global_adm_step]], [-20, 20], color = :black, linestyle = :dot, linewidth = 3.0)
    end

    # start the timer!
    adam_start = time()

    # loop over adam steps
    while run_adam
        # increment
        adm_step += 1
        plt[:global_adm_step] += 1 # for plotting

        # what type of step decay should we employ?
        if qG.decay_type == "cos"
            alpha = qG.alpha_min + 0.5*(qG.alpha_max - qG.alpha_min)*(1+cos((adm_step/qG.Ti)*pi))
        elseif qG.decay_type == "exponential"
            alpha = qG.alpha_0*(qG.step_decay^adm_step)
        else
            @assert qG.decay_type == "none"
            alpha = copy(qG.alpha_0)
        end

        # decay beta
        beta1_decay = beta1_decay*beta1
        beta2_decay = beta2_decay*beta2

        # compute all states and grads
        quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

        # take an adam step
        quasiGrad.adam!(adm, beta1, beta2, beta1_decay, beta2_decay, mgd, prm, qG, stt, upd)

        # stopping criteria
        if qG.adam_stopper == "time"
            if time() - adam_start >= qG.adam_max_time
                run_adam = false
            end
        elseif qG.adam_stopper == "iterations"
            if adm_step >= qG.adam_max_its
                run_adam = false
            end
        else
            # uh-oh -- no stopper!
        end

        # plot the progress
        quasiGrad.update_plot!(adm_step, ax, fig, plt, qG, scr, z_plt)
        display(fig)
    end

    # one last clip + state computation -- no grad needed!
    qG.eval_grad = false
    quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

    # turn it back on
    qG.eval_grad = true
end


# initialize the plot
function initialize_plot(
    cgd::quasiGrad.Cgd,
    ctb::Vector{Vector{Float64}}, 
    ctd::Vector{Vector{Float64}}, 
    flw::Dict{Symbol, Vector{Float64}}, 
    grd::Dict{Symbol, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}, 
    idx::quasiGrad.Idx, 
    mgd::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    msc::Dict{Symbol, Vector{Float64}}, 
    ntk::quasiGrad.Ntk, 
    plt::Dict{Symbol, Integer}, 
    prm::quasiGrad.Param, 
    qG::quasiGrad.QG, 
    scr::Dict{Symbol, Float64}, 
    stt::Dict{Symbol, Dict{Symbol, Vector{Float64}}}, 
    sys::quasiGrad.System, 
    wct::Vector{Vector{Int64}})
    
    # first, make sure scores are updated!
    qG.eval_grad = false
    quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)
    qG.eval_grad = true

    # now, initialize
    fig = quasiGrad.Makie.Figure(resolution=(1200, 600), fontsize=18) 
    ax  = quasiGrad.Makie.Axis(fig[1, 1], xlabel = "adam iteration", ylabel = "score values (z)", title = "quasiGrad")
    quasiGrad.Makie.xlims!(ax, [1, plt[:N_its]])

    # set ylims -- this is tricky, since we use "1000" as zero, so the scale goes,
    # -10^4 -10^3 0 10^3 10^4...
    min_y     = (-log10(abs(scr[:zms]) + 0.1) - 1.0) + 3.0
    min_y_int = ceil(min_y)

    max_y     = (+log10(scr[:ed_obj]  + 0.1) + 0.25) - 3.0
    max_y_int = floor(max_y)

    # since "1000" is our reference -- see scaling function notes
    y_vec = collect((min_y_int):(max_y_int))
    quasiGrad.Makie.ylims!(ax, [min_y, max_y])
    tick_name = String[]
    for yv in y_vec
        if yv == 0
            push!(tick_name,"0")
        elseif yv < 0
            push!(tick_name,"-10^"*string(Int(abs(yv - 3.0))))
        else
            push!(tick_name,"+10^"*string(Int(abs(yv + 3.0))))
        end
    end
    ax.yticks = (y_vec, tick_name)
    display(fig)

    # define current and previous dicts
    z_plt = Dict(:prev => Dict(
                            :zms  => 0.0,
                            :pzms => 0.0,       
                            :zhat => 0.0,
                            :ctg  => 0.0,
                            :zp   => 0.0,
                            :zq   => 0.0,
                            :acl  => 0.0,
                            :xfm  => 0.0,
                            :zoud => 0.0,
                            :zone => 0.0,
                            :rsv  => 0.0,
                            :enpr => 0.0,
                            :encs => 0.0,
                            :emnx => 0.0,
                            :zsus => 0.0),
                :now => Dict(
                            :zms  => 0.0,
                            :pzms => 0.0,     
                            :zhat => 0.0,
                            :ctg  => 0.0,
                            :zp   => 0.0,
                            :zq   => 0.0,
                            :acl  => 0.0,
                            :xfm  => 0.0,
                            :zoud => 0.0,
                            :zone => 0.0,
                            :rsv  => 0.0,
                            :enpr => 0.0,
                            :encs => 0.0,
                            :emnx => 0.0,
                            :zsus => 0.0))

    # output
    return ax, fig, z_plt
end

# update the plot
function update_plot!(adm_step::Int64, ax::quasiGrad.Makie.Axis, fig::quasiGrad.Makie.Figure, plt::Dict{Symbol, Integer}, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, z_plt::Dict{Symbol, Dict{Symbol, Float64}})
    #
    # is this the first plot? if adm_step == 1, then we don't plot (just update)
    if adm_step > 1 # !(plt[:first_plot] || adm_step == 1)
        # first, set the current values
        z_plt[:now][:zms]  = scale_z(scr[:zms])
        z_plt[:now][:pzms] = scale_z(scr[:zms_penalized])      
        z_plt[:now][:zhat] = scale_z(scr[:zt_penalty] - qG.delta*scr[:zhat_mxst])
        z_plt[:now][:ctg]  = scale_z(scr[:zctg_min] + scr[:zctg_avg])
        z_plt[:now][:emnx] = scale_z(scr[:emnx])
        z_plt[:now][:zp]   = scale_z(scr[:zp])
        z_plt[:now][:zq]   = scale_z(scr[:zq])
        z_plt[:now][:acl]  = scale_z(scr[:acl])
        z_plt[:now][:xfm]  = scale_z(scr[:xfm])
        z_plt[:now][:zoud] = scale_z(scr[:zoud])
        z_plt[:now][:zone] = scale_z(scr[:zone])
        z_plt[:now][:rsv]  = scale_z(scr[:rsv])
        z_plt[:now][:enpr] = scale_z(scr[:enpr])
        z_plt[:now][:encs] = scale_z(scr[:encs])
        z_plt[:now][:zsus] = scale_z(scr[:zsus])

        # x-axis
        if plt[:global_adm_step] > plt[:N_its]
            plt[:N_its] = plt[:N_its] + 150
            quasiGrad.Makie.xlims!(ax, [1, plt[:N_its]])
        end

        # now, plot!
        #
        # add an economic dipatch upper bound
        l0 = quasiGrad.Makie.lines!(ax, [0, 1e4], [log10(scr[:ed_obj]) - 3.0, log10(scr[:ed_obj]) - 3.0], color = :coral1, linestyle = :dash, linewidth = 5.0)

        l1  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zms],  z_plt[:now][:zms] ], color = :cornflowerblue, linewidth = 4.5)
        l2  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:pzms], z_plt[:now][:pzms]], color = :mediumblue,     linewidth = 3.0)

        l3  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zhat], z_plt[:now][:zhat]], color = :goldenrod1, linewidth = 2.0)

        l4  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:ctg] , z_plt[:now][:ctg] ], color = :lightslateblue)

        l5  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zp]  , z_plt[:now][:zp]  ], color = :firebrick, linewidth = 3.5)
        l6  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zq]  , z_plt[:now][:zq]  ], color = :salmon1,   linewidth = 2.0)

        l7  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:acl] , z_plt[:now][:acl] ], color = :darkorange1, linewidth = 3.5)
        l8  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:xfm] , z_plt[:now][:xfm] ], color = :orangered1,  linewidth = 2.0)
        
        l9  = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zoud], z_plt[:now][:zoud]], color = :grey95, linewidth = 3.5, linestyle = :solid)
        l10 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zone], z_plt[:now][:zone]], color = :gray89, linewidth = 3.0, linestyle = :dot)
        l11 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:rsv] , z_plt[:now][:rsv] ], color = :gray75, linewidth = 2.5, linestyle = :dash)
        l12 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:emnx], z_plt[:now][:emnx]], color = :grey38, linewidth = 2.0, linestyle = :dashdot)
        l13 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:zsus], z_plt[:now][:zsus]], color = :grey0,  linewidth = 1.5, linestyle = :dashdotdot)

        l14 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:enpr], z_plt[:now][:enpr]], color = :forestgreen, linewidth = 3.5)
        l15 = quasiGrad.Makie.lines!(ax, [plt[:global_adm_step]-1.01, plt[:global_adm_step]], [z_plt[:prev][:encs], z_plt[:now][:encs]], color = :darkgreen,   linewidth = 2.0)
        
        if plt[:first_plot] == true  # this will occur only once
            plt[:first_plot] = false # toggle

            # define trace lables
            label = Dict(
                :zms  => "market surplus",
                :pzms => "penalized market surplus",       
                :zhat => "constraint penalties", 
                :ctg  => "contingency penalties",
                :zp   => "active power balance",
                :zq   => "reactive power balance",
                :acl  => "acline flow",  
                :xfm  => "xfm flow", 
                :zoud => "on/up/down costs",
                :zone => "zonal reserve penalties",
                :rsv  => "local reserve penalties",
                :enpr => "energy costs (pr)",   
                :encs => "energy revenues (cs)",   
                :emnx => "min/max energy violations",
                :zsus => "start-up state discount",
                :ed   => "economic dispatch (bound)")

            # build legend ==================
            quasiGrad.Makie.Legend(fig[1, 2], [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15],
                            [label[:ed],  label[:zms],  label[:pzms], label[:zhat], label[:ctg], label[:zp],   label[:zq],   label[:acl],
                             label[:xfm], label[:zoud], label[:zone], label[:rsv], label[:emnx], label[:zsus], label[:enpr], 
                             label[:encs]],
                             halign = :right, valign = :top, framevisible = false)
        end

        # display the figure
        if mod(adm_step,plt[:disp_freq]) == 0
            # display(fig) => this does nothing!!
            sleep(1e-10) # I don't know why this needs to be here..
        end
    end

    # update the previous values!
    z_plt[:prev][:zms]  = scale_z(scr[:zms])
    z_plt[:prev][:pzms] = scale_z(scr[:zms_penalized])      
    z_plt[:prev][:zhat] = scale_z(scr[:zt_penalty] - qG.delta*scr[:zhat_mxst])
    z_plt[:prev][:ctg]  = scale_z(scr[:zctg_min] + scr[:zctg_avg])
    z_plt[:prev][:emnx] = scale_z(scr[:emnx])
    z_plt[:prev][:zp]   = scale_z(scr[:zp])
    z_plt[:prev][:zq]   = scale_z(scr[:zq])
    z_plt[:prev][:acl]  = scale_z(scr[:acl])
    z_plt[:prev][:xfm]  = scale_z(scr[:xfm])
    z_plt[:prev][:zoud] = scale_z(scr[:zoud])
    z_plt[:prev][:zone] = scale_z(scr[:zone])
    z_plt[:prev][:rsv]  = scale_z(scr[:rsv])
    z_plt[:prev][:enpr] = scale_z(scr[:enpr])
    z_plt[:prev][:encs] = scale_z(scr[:encs])
    z_plt[:prev][:zsus] = scale_z(scr[:zsus])
end

# function to rescale scores for plotting :)
function scale_z(z::Float64)
    sgn  = sign(z .+ 1e-6)
    absz = abs(z)
    if absz < 1000.0 # clip
        absz = 1000.0
    end
    if sgn < 0
        # shift up two
        zs = sgn*log10(absz) + 3.0
    else
        # shift down two
        zs = sgn*log10(absz) - 3.0
        # +10^5 => 2
        # +10^4 => 1
        # -10^1/2/3 = +10^1/2/3 => 0
        # -10^4 => -1
        # -10^5 => -2
    end

    # output
    return zs
end