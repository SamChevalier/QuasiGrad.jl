using quasiGrad
using Revise

# files -- 1576 system
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"
path = tfp*"C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"

# solve ED
InFile1 = path
jsn = quasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn)

# write locally
qG.write_location   = "local"
qG.eval_grad        = true
qG.always_solve_ctg = true
qG.skip_ctg_eval    = false

# turn off all printing
qG.print_zms                     = false # print zms at every adam iteration?
qG.print_final_stats             = false # print stats at the end?
qG.print_lbfgs_iterations        = false
qG.print_projection_success      = false
qG.print_linear_pf_iterations    = false
qG.print_reserve_cleanup_success = false

# solve
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% initialize the data-log
nz       = 20000
data_log = Dict(:zms  => zeros(nz),
                :pzms => zeros(nz), 
                :zhat => zeros(nz), 
                :ctg  => zeros(nz), 
                :emnx => zeros(nz), 
                :zp   => zeros(nz), 
                :zq   => zeros(nz), 
                :acl  => zeros(nz), 
                :xfm  => zeros(nz), 
                :zoud => zeros(nz), 
                :zone => zeros(nz), 
                :rsv  => zeros(nz), 
                :enpr => zeros(nz), 
                :encs => zeros(nz), 
                :zsus => zeros(nz)) 

# solve with adam
stt = deepcopy(stt0);

# ===============
vm_t0      = 2.5e-5
va_t0      = 2.5e-5
phi_t0     = 2.5e-5
tau_t0     = 2.5e-5
dc_t0      = 1e-2
power_t0   = 1e-2
reserve_t0 = 1e-2
bin_t0     = 1e-2 # bullish!!!
qG.alpha_t0 = Dict(
               :vm     => vm_t0,
               :va     => va_t0,
               # xfm
               :phi    => phi_t0,
               :tau    => tau_t0,
               # dc
               :dc_pfr => dc_t0,
               :dc_qto => dc_t0,
               :dc_qfr => dc_t0,
               # powers
               :dev_q  => power_t0,
               :p_on   => power_t0,
               # reserves
               :p_rgu     => reserve_t0,
               :p_rgd     => reserve_t0,
               :p_scr     => reserve_t0,
               :p_nsc     => reserve_t0,
               :p_rrd_on  => reserve_t0,
               :p_rrd_off => reserve_t0,
               :p_rru_on  => reserve_t0,
               :p_rru_off => reserve_t0,
               :q_qrd     => reserve_t0,
               :q_qru     => reserve_t0,
               # bins
               :u_on_xfm     => bin_t0,
               :u_on_dev     => bin_t0,
               :u_step_shunt => bin_t0,
               :u_on_acline  => bin_t0)
vm_tf      = 5e-7# 2e-6#
va_tf      = 5e-7# 2e-6#
phi_tf     = 5e-7# 2e-6#
tau_tf     = 5e-7# 2e-6#
dc_tf      = 2e-4 # 1e-3#
power_tf   = 2e-4 # 1e-3#
reserve_tf = 2e-4 # 1e-3#
bin_tf     = 2e-4 # 1e-3#
qG.alpha_tf = Dict(
                :vm     => vm_tf,
                :va     => va_tf,
                # xfm
                :phi    => phi_tf,
                :tau    => tau_tf,
                # dc
                :dc_pfr => dc_tf,
                :dc_qto => dc_tf,
                :dc_qfr => dc_tf,
                # powers
                :dev_q  => power_tf,
                :p_on   => power_tf,
                # reserves
                :p_rgu     => reserve_tf,
                :p_rgd     => reserve_tf,
                :p_scr     => reserve_tf,
                :p_nsc     => reserve_tf,
                :p_rrd_on  => reserve_tf,
                :p_rrd_off => reserve_tf,
                :p_rru_on  => reserve_tf,
                :p_rru_off => reserve_tf,
                :q_qrd     => reserve_tf,
                :q_qru     => reserve_tf,
                # bins
                :u_on_xfm     => bin_tf,
                :u_on_dev     => bin_tf,
                :u_step_shunt => bin_tf,
                :u_on_acline  => bin_tf)

qG.print_zms     = true
qG.adam_max_time = 300.0
quasiGrad.run_adam_with_data_collection!(adm, cgd, ctg, data_log, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% ==============
using Makie
using GLMakie
# using CairoMakie
GLMakie.activate!()

# now, initialize
fig = Makie.Figure(resolution=(1200, 600), fontsize=26) 
ax  = Makie.Axis(fig[1, 1], xlabel = "adam iteration", ylabel = "score values (z)", xlabelfont = :bold, ylabelfont = :bold)
Makie.xlims!(ax, [1, qG.adm_step+1])

# set ylims -- this is tricky, since we use "1000" as zero, so the scale goes,
# -10^4 -10^3 0 10^3 10^4... data_log[:pzms][1]
min_y     = (-log10(abs(-1e8) + 0.1) - 1.0) + 3.0
min_y_int = ceil(min_y)

max_y     = (+log10(scr[:ed_obj]  + 0.1) + 0.25) - 3.0
max_y_int = floor(max_y)

# since "1000" is our reference -- see scaling function notes
y_vec = collect((min_y_int):2:(max_y_int))
Makie.ylims!(ax, [min_y, max_y])
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
ax.xticks = [2500, 5000, 7500, 10000]
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

# update the plot
l0 = Makie.lines!(ax, [0, 1e5], [log10(scr[:ed_obj]) - 3.0, log10(scr[:ed_obj]) - 3.0], color = :coral1, linestyle = :dash, linewidth = 7.0)

l1  = Makie.lines!(ax, 1:qG.adm_step, data_log[:zms][1:qG.adm_step],  color = :cornflowerblue, linewidth = 10.0)
l2  = Makie.lines!(ax, 1:qG.adm_step, data_log[:pzms][1:qG.adm_step], color = :mediumblue,     linewidth = 3.0)
l3  = Makie.lines!(ax, 1:qG.adm_step, data_log[:zhat][1:qG.adm_step], color = :goldenrod1, linewidth = 2.0)
l4  = Makie.lines!(ax, 1:qG.adm_step, data_log[:ctg][1:qG.adm_step], color = :lightslateblue)
l5  = Makie.lines!(ax, 1:qG.adm_step, data_log[:zp][1:qG.adm_step], color = :firebrick, linewidth = 3.5)
l6  = Makie.lines!(ax, 1:qG.adm_step, data_log[:zq][1:qG.adm_step], color = :salmon1,   linewidth = 2.0)
l7  = Makie.lines!(ax, 1:qG.adm_step, data_log[:acl][1:qG.adm_step], color = :darkorange1, linewidth = 3.5)
l8  = Makie.lines!(ax, 1:qG.adm_step, data_log[:xfm][1:qG.adm_step], color = :orangered1,  linewidth = 2.0)
l9  = Makie.lines!(ax, 1:qG.adm_step, data_log[:zoud][1:qG.adm_step], color = :grey95, linewidth = 3.5, linestyle = :solid)
l10 = Makie.lines!(ax, 1:qG.adm_step, data_log[:zone][1:qG.adm_step], color = :gray89, linewidth = 3.0, linestyle = :dot)
l11 = Makie.lines!(ax, 1:qG.adm_step, data_log[:rsv][1:qG.adm_step],  color = :gray75, linewidth = 2.5, linestyle = :dash)
l12 = Makie.lines!(ax, 1:qG.adm_step, data_log[:emnx][1:qG.adm_step], color = :grey38, linewidth = 2.0, linestyle = :dashdot)
l13 = Makie.lines!(ax, 1:qG.adm_step, data_log[:zsus][1:qG.adm_step], color = :grey0,  linewidth = 1.5, linestyle = :dashdotdot)
l14 = Makie.lines!(ax, 1:qG.adm_step, data_log[:enpr][1:qG.adm_step], color = :forestgreen, linewidth = 3.5)
l15 = Makie.lines!(ax, 1:qG.adm_step, data_log[:encs][1:qG.adm_step], color = :darkgreen,   linewidth = 2.0)

Makie.lines!(ax, [0, 1e5], [log10(scr[:ed_obj]) - 3.0, log10(scr[:ed_obj]) - 3.0], color = :coral1, linestyle = :dash, linewidth = 7.0)

# define trace lables
label = Dict(
    :zms  => "market surplus",
    :pzms => "penalized market surplus",       
    :zhat => "constraint penalties", 
    :ctg  => "contingency penalties",
    :zp   => "active power penalty",
    :zq   => "reactive power penalty",
    :acl  => "acline flow penalty",  
    :xfm  => "xfm flow penalty", 
    :zoud => "on/up/down costs",
    :zone => "zonal reserve penalties",
    :rsv  => "local reserve penalties",
    :enpr => "energy costs (pr)",   
    :encs => "energy revenues (cs)",   
    :emnx => "min/max energy violations",
    :zsus => "start-up state discount",
    :ed   => "economic dispatch (bound)")

# build legend ==================
Makie.Legend(fig[1, 2], [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15],
                [label[:ed],  label[:zms],  label[:pzms], label[:zhat], label[:ctg], label[:zp],   label[:zq],   label[:acl],
                    label[:xfm], label[:zoud], label[:zone], label[:rsv], label[:emnx], label[:zsus], label[:enpr], 
                    label[:encs]],
                    halign = :right, valign = :top, framevisible = false)







# %% quasiGrad plot
#
# solve with adam
stt = deepcopy(stt0);

# ===============
vm_t0      = 2.5e-5
va_t0      = 2.5e-5
phi_t0     = 2.5e-5
tau_t0     = 2.5e-5
dc_t0      = 1e-2
power_t0   = 1e-2
reserve_t0 = 1e-2
bin_t0     = 1e-2 # bullish!!!
qG.alpha_t0 = Dict(
               :vm     => vm_t0,
               :va     => va_t0,
               # xfm
               :phi    => phi_t0,
               :tau    => tau_t0,
               # dc
               :dc_pfr => dc_t0,
               :dc_qto => dc_t0,
               :dc_qfr => dc_t0,
               # powers
               :dev_q  => power_t0,
               :p_on   => power_t0,
               # reserves
               :p_rgu     => reserve_t0,
               :p_rgd     => reserve_t0,
               :p_scr     => reserve_t0,
               :p_nsc     => reserve_t0,
               :p_rrd_on  => reserve_t0,
               :p_rrd_off => reserve_t0,
               :p_rru_on  => reserve_t0,
               :p_rru_off => reserve_t0,
               :q_qrd     => reserve_t0,
               :q_qru     => reserve_t0,
               # bins
               :u_on_xfm     => bin_t0,
               :u_on_dev     => bin_t0,
               :u_step_shunt => bin_t0,
               :u_on_acline  => bin_t0)
vm_tf      = 5e-7# 2e-6#
va_tf      = 5e-7# 2e-6#
phi_tf     = 5e-7# 2e-6#
tau_tf     = 5e-7# 2e-6#
dc_tf      = 2e-4 # 1e-3#
power_tf   = 2e-4 # 1e-3#
reserve_tf = 2e-4 # 1e-3#
bin_tf     = 2e-4 # 1e-3#
qG.alpha_tf = Dict(
                :vm     => vm_tf,
                :va     => va_tf,
                # xfm
                :phi    => phi_tf,
                :tau    => tau_tf,
                # dc
                :dc_pfr => dc_tf,
                :dc_qto => dc_tf,
                :dc_qfr => dc_tf,
                # powers
                :dev_q  => power_tf,
                :p_on   => power_tf,
                # reserves
                :p_rgu     => reserve_tf,
                :p_rgd     => reserve_tf,
                :p_scr     => reserve_tf,
                :p_nsc     => reserve_tf,
                :p_rrd_on  => reserve_tf,
                :p_rrd_off => reserve_tf,
                :p_rru_on  => reserve_tf,
                :p_rru_off => reserve_tf,
                :q_qrd     => reserve_tf,
                :q_qru     => reserve_tf,
                # bins
                :u_on_xfm     => bin_tf,
                :u_on_dev     => bin_tf,
                :u_step_shunt => bin_tf,
                :u_on_acline  => bin_tf)

qG.print_zms     = true
qG.adam_max_time = 400.0
using Makie
using GLMakie
# using CairoMakie
GLMakie.activate!()

# now, initialize
fig = Makie.Figure(resolution=(1200, 600), fontsize=26) 
ax  = Makie.Axis(fig[1, 1], xlabel = "adam iteration", ylabel = "score values (z)", xlabelfont = :bold, ylabelfont = :bold)
Makie.xlims!(ax, [1, qG.adm_step+1])

# set ylims -- this is tricky, since we use "1000" as zero, so the scale goes,
# -10^4 -10^3 0 10^3 10^4... data_log[:pzms][1]
min_y     = (-log10(abs(-1e8) + 0.1) - 1.0) + 3.0
min_y_int = ceil(min_y)

max_y     = (+log10(scr[:ed_obj]  + 0.1) + 0.25) - 3.0
max_y_int = floor(max_y)

# since "1000" is our reference -- see scaling function notes
y_vec = collect((min_y_int):2:(max_y_int))
Makie.ylims!(ax, [min_y, max_y])
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
ax.xticks = [2500, 5000, 7500, 10000]
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

# %% ==============
stt = deepcopy(stt0);
run_adam_and_plot!(ax, fig, z_plt, adm, cgd, ctg, data_log, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)






# %% =======================
function run_adam_and_plot!(ax::Makie.Axis, fig::Makie.Figure, z_plt::Dict{Symbol, Dict{Symbol, Float64}}, adm::quasiGrad.Adam, cgd::quasiGrad.ConstantGrad, ctg::quasiGrad.Contingency, data_log::Dict{Symbol, Vector{Float64}}, flw::quasiGrad.Flow, grd::quasiGrad.Grad, idx::quasiGrad.Index, mgd::quasiGrad.MasterGrad, ntk::quasiGrad.Network, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}; clip_pq_based_on_bins::Bool=false)
    # NOTE -- "clip_pq_based_on_bins = true" is only used once all binaries have been fixed!
    #         so, use in on the very last adam iteration after binaries have been set.
    # 
    # here we go!
    @info "Running adam for $(qG.adam_max_time) seconds!"
    fp = true

    # flush adam just once!
    quasiGrad.flush_adam!(adm, flw, prm, upd)

    # loop and solve adam twice: once for an initialization, and once for a true run
    qG.skip_ctg_eval = false

    # re-initialize
    qG.adm_step      = 0
    qG.beta1_decay   = 1.0
    qG.beta2_decay   = 1.0
    qG.one_min_beta1 = 1.0 - qG.beta1 # here for testing, in case beta1 is changed before a run
    qG.one_min_beta2 = 1.0 - qG.beta2 # here for testing, in case beta2 is changed before a run
    run_adam         = true

    # start the timer!
    adam_start = time()
    # loop over adam steps
    while run_adam

        # increment
        qG.adm_step += 1

        # step decay
        quasiGrad.adam_step_decay!(qG, time(), adam_start, adam_start+qG.adam_max_time)

        # decay beta and pre-compute
        qG.beta1_decay         = qG.beta1_decay*qG.beta1
        qG.beta2_decay         = qG.beta2_decay*qG.beta2
        qG.one_min_beta1_decay = (1.0-qG.beta1_decay)
        qG.one_min_beta2_decay = (1.0-qG.beta2_decay)

        # update weight parameters?
        if qG.apply_grad_weight_homotopy == true
            quasiGrad.update_penalties!(prm, qG, time(), adam_start, adam_start+qG.adam_max_time)
        end

        # compute all states and grads
        quasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, clip_pq_based_on_bins=clip_pq_based_on_bins)

        # take an adam step
        quasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
        GC.safepoint()

        update_plot!(ax, fig, qG, scr, z_plt, fp)
        display(fig)

        # stop?
        run_adam = quasiGrad.adam_termination(adam_start, qG, run_adam, qG.adam_max_time)

        # log adam data for plotting
        # => quasiGrad.log_data(data_log, qG, scr)
    end

end

function update_plot!(ax::Makie.Axis, fig::Makie.Figure, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, z_plt::Dict{Symbol, Dict{Symbol, Float64}}, fp::Bool)
    #
    # is this the first plot? if adm_step == 1, then we don't plot (just update)
    if qG.adm_step > 1 # !(plt[:first_plot] || adm_step == 1)
        # first, set the current values
        z_plt[:now][:zms]  = quasiGrad.scale_z(scr[:zms])
        z_plt[:now][:pzms] = quasiGrad.scale_z(scr[:zms_penalized])      
        z_plt[:now][:zhat] = quasiGrad.scale_z(scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst])
        z_plt[:now][:ctg]  = quasiGrad.scale_z(scr[:zctg_min] + scr[:zctg_avg])
        z_plt[:now][:emnx] = quasiGrad.scale_z(scr[:emnx])
        z_plt[:now][:zp]   = quasiGrad.scale_z(scr[:zp])
        z_plt[:now][:zq]   = quasiGrad.scale_z(scr[:zq])
        z_plt[:now][:acl]  = quasiGrad.scale_z(scr[:acl])
        z_plt[:now][:xfm]  = quasiGrad.scale_z(scr[:xfm])
        z_plt[:now][:zoud] = quasiGrad.scale_z(scr[:zoud])
        z_plt[:now][:zone] = quasiGrad.scale_z(scr[:zone])
        z_plt[:now][:rsv]  = quasiGrad.scale_z(scr[:rsv])
        z_plt[:now][:enpr] = quasiGrad.scale_z(scr[:enpr])
        z_plt[:now][:encs] = quasiGrad.scale_z(scr[:encs])
        z_plt[:now][:zsus] = quasiGrad.scale_z(scr[:zsus])

        # now, plot!
        #
        # add an economic dipatch upper bound
        l0 = Makie.lines!(ax, [0, 1e4], [log10(scr[:ed_obj]) - 3.0, log10(scr[:ed_obj]) - 3.0], color = :coral1, linestyle = :dash, linewidth = 5.0)

        l1  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:zms],  z_plt[:now][:zms] ], color = :cornflowerblue, linewidth = 4.5)
        l2  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:pzms], z_plt[:now][:pzms]], color = :mediumblue,     linewidth = 3.0)

        l3  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:zhat], z_plt[:now][:zhat]], color = :goldenrod1, linewidth = 2.0)

        l4  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:ctg] , z_plt[:now][:ctg] ], color = :lightslateblue)

        l5  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:zp]  , z_plt[:now][:zp]  ], color = :firebrick, linewidth = 3.5)
        l6  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:zq]  , z_plt[:now][:zq]  ], color = :salmon1,   linewidth = 2.0)

        l7  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:acl] , z_plt[:now][:acl] ], color = :darkorange1, linewidth = 3.5)
        l8  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:xfm] , z_plt[:now][:xfm] ], color = :orangered1,  linewidth = 2.0)
        
        l9  = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:zoud], z_plt[:now][:zoud]], color = :grey95, linewidth = 3.5, linestyle = :solid)
        l10 = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:zone], z_plt[:now][:zone]], color = :gray89, linewidth = 3.0, linestyle = :dot)
        l11 = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:rsv] , z_plt[:now][:rsv] ], color = :gray75, linewidth = 2.5, linestyle = :dash)
        l12 = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:emnx], z_plt[:now][:emnx]], color = :grey38, linewidth = 2.0, linestyle = :dashdot)
        l13 = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:zsus], z_plt[:now][:zsus]], color = :grey0,  linewidth = 1.5, linestyle = :dashdotdot)

        l14 = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:enpr], z_plt[:now][:enpr]], color = :forestgreen, linewidth = 3.5)
        l15 = Makie.lines!(ax, [qG.adm_step-1.01, qG.adm_step], [z_plt[:prev][:encs], z_plt[:now][:encs]], color = :darkgreen,   linewidth = 2.0)
        
        if fp == true
            fp = false # toggle

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
            Makie.Legend(fig[1, 2], [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15],
                            [label[:ed],  label[:zms],  label[:pzms], label[:zhat], label[:ctg], label[:zp],   label[:zq],   label[:acl],
                             label[:xfm], label[:zoud], label[:zone], label[:rsv], label[:emnx], label[:zsus], label[:enpr], 
                             label[:encs]],
                             halign = :right, valign = :top, framevisible = false)
        end

        # display the figure
        sleep(1e-10) 
    end

    # update the previous values!
    z_plt[:prev][:zms]  = quasiGrad.scale_z(scr[:zms])
    z_plt[:prev][:pzms] = quasiGrad.scale_z(scr[:zms_penalized])      
    z_plt[:prev][:zhat] = quasiGrad.scale_z(scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst])
    z_plt[:prev][:ctg]  = quasiGrad.scale_z(scr[:zctg_min] + scr[:zctg_avg])
    z_plt[:prev][:emnx] = quasiGrad.scale_z(scr[:emnx])
    z_plt[:prev][:zp]   = quasiGrad.scale_z(scr[:zp])
    z_plt[:prev][:zq]   = quasiGrad.scale_z(scr[:zq])
    z_plt[:prev][:acl]  = quasiGrad.scale_z(scr[:acl])
    z_plt[:prev][:xfm]  = quasiGrad.scale_z(scr[:xfm])
    z_plt[:prev][:zoud] = quasiGrad.scale_z(scr[:zoud])
    z_plt[:prev][:zone] = quasiGrad.scale_z(scr[:zone])
    z_plt[:prev][:rsv]  = quasiGrad.scale_z(scr[:rsv])
    z_plt[:prev][:enpr] = quasiGrad.scale_z(scr[:enpr])
    z_plt[:prev][:encs] = quasiGrad.scale_z(scr[:encs])
    z_plt[:prev][:zsus] = quasiGrad.scale_z(scr[:zsus])
end

# %% ===
gr()

c1 = 165/256
c2 = 42/256
c3 = 42/256
redd = RGB(c1,c2,c3)

x     = -1:0.01:1
beta  = exp.(4.0.*x)./(0.6 .+ exp.(4.0.*x))
log_stp_ratio = log10(1e-2/1e-5)
stp = -beta.*log_stp_ratio .+ log10.(1e-2)
# => Plots.plot(x, stp)
Plots.plot((x .+ 1.0)./2.0, 10 .^ stp, yaxis=:log, ylabel = "adam step size", xlabel = "normalized wallclock time (sec)", yguidefont = font(12,:black), xguidefont = font(12,:black), linewidth = 3.5, color = :black, xtickfontsize = 10, ytickfontsize = 10, label = "α", legend=:bottomleft, yticks =[10^-2, 10^-3, 10^-4, 10^-5], ylim = [10^-5, 10^-2], legendfontsize=12)

beta  = exp.(5.0.*x)./(1.0.+ exp.(5.0.*x))
eps2_t0                 = 1e-2
eps2_tf                 = 1e-10
log_eps2_ratio          = log10.(eps2_t0./eps2_tf)
esp2_update             = 10.0 .^ (.-beta.*log_eps2_ratio .+ log10.(eps2_t0))

Plots.plot!(twinx(), (x .+ 1.0)./2.0, esp2_update, yaxis=:log, ylabel = "soft-abs and soft-ReLU ϵ value", linewidth = 3.5, color = redd, yguidefont = font(12,redd), ytickfontcolor = redd, ytickfontsize = 10, label = "ϵ", yticks =[10^-2, 10^-4, 10^-6, 10^-8, 10^-10], legendfontsize=12)
p = Plots.plot!(size=(600,300))

# %% ===
x     = -1:0.001:1
l1 = 0.5*sqrt.(x.^2 .+ 0.05)
l2 = 0.8*sqrt.(x.^2 .+ 0.005)
l3 = sqrt.(x.^2 .+ 0.00001)

Plots.plot(x, l1, label = "β=0.5, ϵ=0.05", linewidth = 2.5, xtickfontsize = 10, ytickfontsize = 10, legendfontsize=12)
Plots.plot!(x, l2, label = "β=0.8, ϵ=0.005", linewidth = 2.5)
Plots.plot!(x, l3, label = "β=1.0, ϵ=0.00001", yguidefont = font(12,:black), xguidefont = font(12,:black), ylabel = "β * soft-abs(x)", xlabel = "normalized input x", linewidth = 2.5, legend=:top)
p = Plots.plot!(size=(600,325))
