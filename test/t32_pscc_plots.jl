using quasiGrad
using Revise

# files -- 1576 system
tfp = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"

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
nz       = 10000
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

#for tii in prm.ts.time_keys
#    stt.va[tii] .= 0.0
#end

# ===============
vm_t0      = 5e-5
va_t0      = 5e-5
phi_t0     = 5e-5
tau_t0     = 5e-5
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
vm_tf      = 1e-7
va_tf      = 1e-7
phi_tf     = 1e-7
tau_tf     = 1e-7
dc_tf      = 1e-5 
power_tf   = 1e-5 
reserve_tf = 1e-5 
bin_tf     = 1e-5 # bullish!!!
qG.alpha_tf = Dict(
                :vm    => vm_tf,
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

# now, initialize
fig = Makie.Figure(resolution=(1200, 600), fontsize=18) 
ax  = Makie.Axis(fig[1, 1], xlabel = "adam iteration", ylabel = "score values (z)", title = "quasiGrad")
Makie.xlims!(ax, [1, qG.adm_step+1])

# set ylims -- this is tricky, since we use "1000" as zero, so the scale goes,
# -10^4 -10^3 0 10^3 10^4... data_log[:pzms][1]
min_y     = (-log10(abs(-1e8) + 0.1) - 1.0) + 3.0
min_y_int = ceil(min_y)

max_y     = (+log10(scr[:ed_obj]  + 0.1) + 0.25) - 3.0
max_y_int = floor(max_y)

# since "1000" is our reference -- see scaling function notes
y_vec = collect((min_y_int):(max_y_int))
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
l0 = Makie.lines!(ax, [0, 1e4], [log10(scr[:ed_obj]) - 3.0, log10(scr[:ed_obj]) - 3.0], color = :coral1, linestyle = :dash, linewidth = 5.0)

l1  = Makie.lines!(ax, 1:qG.adm_step, data_log[:zms][1:qG.adm_step],  color = :cornflowerblue, linewidth = 4.5)
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