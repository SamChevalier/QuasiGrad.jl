using quasiGrad
using Revise

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/C3S0N00014/scenario_003.json"
jsn  = quasiGrad.load_json(path)

# %% initialize
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn, perturb_states=false);

# run copper plate ED
quasiGrad.economic_dispatch_initialization!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct)

# initialize plot
plt = Dict(:plot            => false,
           :first_plot      => true,
           :N_its           => 150,
           :global_adm_step => 0,
           :disp_freq       => 5)

# initialize
ax, fig, z_plt  = quasiGrad.initialize_plot(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, plt, prm, qG, scr, stt, sys, wct)

# run adam with plotting
vmva_scale    = 1e-4
xfm_scale     = 1e-4
dc_scale      = 2e-3
power_scale   = 2e-3
reserve_scale = 2e-3
bin_scale     = 2e-3
qG.alpha_0 = Dict(
                :vm     => vmva_scale,
                :va     => vmva_scale,
                # xfm
                :phi    => xfm_scale,
                :tau    => xfm_scale,
                # dc
                :dc_pfr => dc_scale,
                :dc_qto => dc_scale,
                :dc_qfr => dc_scale,
                # powers -- decay this!!
                :dev_q  => power_scale,
                :p_on   => power_scale,
                # reserves
                :p_rgu     => reserve_scale,
                :p_rgd     => reserve_scale,
                :p_scr     => reserve_scale,
                :p_nsc     => reserve_scale,
                :p_rrd_on  => reserve_scale,
                :p_rrd_off => reserve_scale,
                :p_rru_on  => reserve_scale,
                :p_rru_off => reserve_scale,
                :q_qrd     => reserve_scale,
                :q_qru     => reserve_scale,
                # bins
                :u_on_xfm     => bin_scale,
                :u_on_dev     => bin_scale,
                :u_step_shunt => bin_scale,
                :u_on_acline  => bin_scale)

qG.print_freq                  = 10
qG.num_threads                 = 6
qG.apply_grad_weight_homotopy  = false
qG.take_adam_pf_steps          = false
qG.pqbal_grad_type             = "soft_abs" 
qG.pqbal_grad_eps2             = 1e-1

qG.constraint_grad_is_soft_abs = false
qG.acflow_grad_is_soft_abs     = false
qG.reserve_grad_is_soft_abs    = false
qG.skip_ctg_eval               = false
qG.beta1                       = 0.9
qG.beta2                       = 0.99
qG.pqbal_grad_eps2             = 1e-8
qG.adam_max_time               = 100.0

quasiGrad.run_adam_with_plotting!(adm, ax, bit, cgd, ctb, ctd, fig, flw, grd, idx, mgd, msc, 
                                  ntk, plt, prm, qG, scr, stt, sys, upd, wct, z_plt)
