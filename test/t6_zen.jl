# test the energy penalties with a modified
include("../src/quasiGrad_dual.jl")

# %% data files -- upper
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073_zen_mod/"
file_name = "scenario_002_zen_mod.json"

# %% data files -- lower
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073_zen_min_mod/"
file_name = "scenario_002_zen_min_mod.json"

# load
jsn = quasiGrad.load_json(data_path*file_name);

# %% initialize
adm, cgd, GRB, grd, idx, mgd, ntk, prm, qG, scr, stt, 
sys, upd, flw, dz_dpinj_base, theta_k_base, worst_ctgs = 
    quasiGrad.base_initialization(jsn, true, 0.25);

# compute states
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, 
                                        dz_dpinj_base, theta_k_base, worst_ctgs)
quasiGrad.solve_Gurobi_projection!(GRB, idx, prm, qG, stt, sys, upd)
quasiGrad.quasiGrad.apply_Gurobi_projection!(GRB, idx, prm, stt)
quasiGrad.update_states_and_grads!(cgd, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, 
                                        dz_dpinj_base, theta_k_base, worst_ctgs)

# print the solution zen_costs
zenmax  = sum(values(scr[:z_enmax]), init=0.0)
zenmin  = sum(values(scr[:z_enmin]), init=0.0)
println(zenmax)
println(zenmin)

# %% assess the gradient ============
epsilon = 1e-3
tii = :t1
ind = 2
quasiGrad.flush_gradients!(grd, mgd, prm, sys)
quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
for dev in 1:sys.ndev
    scr[:z_enmax][dev] = -sum(stt[:zw_enmax][dev]; init=0.0)
    scr[:z_enmin][dev] = -sum(stt[:zw_enmin][dev]; init=0.0)
end

for (t_ind, tii) in enumerate(prm.ts.time_keys)
    for dev in 1:sys.ndev
        quasiGrad.apply_dev_q_grads!(tii, t_ind, prm, qG, idx, stt, grd, mgd, dev, grd[:dx][:dq][tii][dev])
        quasiGrad.apply_dev_p_grads!(tii, t_ind, prm, qG, stt, grd, mgd, dev, grd[:dx][:dp][tii][dev])
    end
end
z0   = copy(sum(values(scr[:z_enmax]), init=0.0) + sum(values(scr[:z_enmin]), init=0.0))
dzdx = copy(mgd[:p_on][tii][ind])

# update device power
stt[:p_on][tii][ind] += epsilon
stt[:dev_p][tii] = stt[:p_on][tii] + stt[:p_su][tii] + stt[:p_sd][tii]
quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
for dev in 1:sys.ndev
    scr[:z_enmax][dev] = -sum(stt[:zw_enmax][dev]; init=0.0)
    scr[:z_enmin][dev] = -sum(stt[:zw_enmin][dev]; init=0.0)
end
zp       = copy(sum(values(scr[:z_enmax]), init=0.0) + sum(values(scr[:z_enmin]), init=0.0))
dzdx_num = (zp - z0)/epsilon
println(dzdx)
println(dzdx_num)

# %% write a solution :)
soln_dict = quasiGrad.prepare_solution(prm, stt, sys)
quasiGrad.write_solution(data_dir*file_name, qG, soln_dict, scr)