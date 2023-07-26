using quasiGrad
using Revise

#include("./test_functions.jl")

path    = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
InFile1 = path
jsn     = quasiGrad.load_json(InFile1)

# %%
adm, bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, upd, wct = quasiGrad.base_initialization(jsn);

# %%
@time quasiGrad.acline_flows!(grd, idx, msc, prm, qG, stt, sys)

# %%
#@btime 

@btime quasiGrad.acline_flows_st!(bit, grd, idx, msc_im, prm, qG, stt_im, sys)

# %%
@btime stt.acline_pfr[:t1] .= stt.u_on_acline[:t1].*msc.pfr[:t1];
@btime stt_im.acline_pfr[1] .= stt_im.u_on_acline[1].*msc.pfr[:t1];

# %%
uv = UInt8.(1:(sys.nT))
vv1 = [zeros(sys.nb) for ii in 1:(sys.nT)]
vv2 = [zeros(sys.nb) for ii in uv]

# %%
stt_im = quasiGrad.State([ones(sys.nb) for ii in 1:sys.nT],         
    [copy(prm.bus.init_va) for ii in 1:sys.nT],
    [zeros(sys.nl) for ii in 1:sys.nT],            
    [zeros(sys.nl) for ii in 1:sys.nT],            
    [zeros(sys.nl) for ii in 1:sys.nT],            
    [zeros(sys.nl) for ii in 1:sys.nT],            
    [prm.acline.init_on_status for ii in 1:sys.nT],
    [zeros(sys.nl)  for ii in 1:sys.nT],           
    [zeros(sys.nl)  for ii in 1:sys.nT],           
    [copy(prm.xfm.init_phi) for ii in 1:sys.nT],
    [copy(prm.xfm.init_tau) for ii in 1:sys.nT],
    [zeros(sys.nx)   for ii in 1:sys.nT],       
    [zeros(sys.nx)   for ii in 1:sys.nT],       
    [zeros(sys.nx)   for ii in 1:sys.nT],       
    [zeros(sys.nx)   for ii in 1:sys.nT],       
    [prm.xfm.init_on_status for ii in 1:sys.nT],
    [zeros(sys.nx) for ii in 1:sys.nT],        
    [zeros(sys.nx) for ii in 1:sys.nT],        
    [Vector{Float64}(undef,(sys.nac)) for ii in 1:sys.nT], 
    [Vector{Float64}(undef,(sys.nac)) for ii in 1:sys.nT], 
    [zeros(sys.nac) for ii in 1:sys.nT], 
    [copy(prm.dc.init_pdc_fr) for ii in 1:sys.nT], 
    [copy(-prm.dc.init_pdc_fr) for ii in 1:sys.nT], 
    [copy(prm.dc.init_qdc_fr) for ii in 1:sys.nT], 
    [copy(prm.dc.init_qdc_to) for ii in 1:sys.nT], 
    [zeros(sys.nsh) for ii in 1:sys.nT], 
    [zeros(sys.nsh) for ii in 1:sys.nT], 
    [zeros(sys.nsh) for ii in 1:sys.nT], 
    [ones(sys.ndev) for ii in 1:sys.nT],  
    [zeros(sys.ndev) for ii in 1:sys.nT],  
    [zeros(sys.ndev) for ii in 1:sys.nT],  
    [ones(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.nctg) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.nl) for ii in 1:sys.nT], 
    [zeros(sys.nx) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.nl) for ii in 1:sys.nT], 
    [zeros(sys.nx) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.nl) for ii in 1:sys.nT], 
    [zeros(sys.nx) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.nb) for ii in 1:sys.nT], 
    [zeros(sys.nb) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzQ) for ii in 1:sys.nT], 
    [zeros(sys.nzQ) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzP) for ii in 1:sys.nT], 
    [zeros(sys.nzQ) for ii in 1:sys.nT], 
    [zeros(sys.nzQ) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.ndev) for ii in 1:sys.nT], 
    [zeros(sys.nb) for ii in 1:sys.nT]);
# %% =======
msc_im = quasiGrad.Msc(
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nsh) for ii in 1:(sys.nT)],
    [zeros(sys.nsh) for ii in 1:(sys.nT)],
    [zeros(sys.nsh) for ii in 1:(sys.nT)],
    [zeros(maximum(prm.dev.num_sus)) for ii in 1:(sys.nT)],
    [zeros(maximum(prm.dev.num_sus)) for ii in 1:(sys.nT)]);

# %% can we loop over a struct?
syms = 


# %% =======
msc_mut = quasiGrad.MMsc(
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nb) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nl) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nx) for ii in 1:(sys.nT)],
    [zeros(sys.nsh) for ii in 1:(sys.nT)],
    [zeros(sys.nsh) for ii in 1:(sys.nT)],
    [zeros(sys.nsh) for ii in 1:(sys.nT)],
    [zeros(maximum(prm.dev.num_sus)) for ii in 1:(sys.nT)],
    [zeros(maximum(prm.dev.num_sus)) for ii in 1:(sys.nT)]);
# %%  =============
nT    = 18
nd    = 100
nb    = 100
t_ind_vec = 1:18
t_ind_int = collect(t_ind_vec)
t_ind_int_fast = UInt8.(collect(t_ind_vec))
t_ind_vec_fast = UInt8.(t_ind_vec)

tkeys = [Symbol("t"*string(ii)) for ii in 1:nT]

# stt -- use initial values
stt = Dict(
    # network -- set all network voltages to there given initial values
    :vm              => Dict(tkeys[ii] => ones(nb)  for ii in 1:(nT)), # this is a flat start
    :va              => Dict(tkeys[ii] => ones(nb)  for ii in 1:(nT)))

stt_d = Dict(
        # network -- set all network voltages to there given initial values
        1              => [ones(nb)  for ii in 1:(nT)], # this is a flat start
        2              => [ones(nb)  for ii in 1:(nT)])

map = Dict(:vm => 1,
           :va => 2)

stt_v = Dict(
    # network -- set all network voltages to there given initial values
    :vm => [ones(nb)  for ii in 1:(nT)], # this is a flat start
    :va => [ones(nb)  for ii in 1:(nT)])

stt_v_all = 
    # network -- set all network voltages to there given initial values
    [[ones(nb)  for ii in 1:(nT)], # this is a flat start
    [ones(nb)  for ii in 1:(nT)]]

# network -- set all network voltages to there given initial values
# stt_v_all = zeros(2,nT,nb) #[[ones(nb)  for ii in 1:(nT)] for jj in 1:1]

stt_v_vm = [ones(nb)  for ii in 1:(nT)]

# %% test timing
dev = 5

@btime sum(stt.vm[tii][dev] for tii in tkeys)
@btime sum(stt_v[:vm][tii][dev] for tii in t_ind_vec)

# %%
mm = map_s(Dict(:vm => 1,:va => 2))
# %%
mm2 = map_inds2(1,2)
# %%

include("./test_functions.jl")

# %%

@btime f_t_symbols(stt, dev, tkeys)

# %%

@btime f_t_inds(stt_v, dev, t_ind_vec)
@btime f_t_inds_vm(stt_v_vm, dev, t_ind_vec)

t = 1

# %%
@time t = mm2.vm;
@time t;



# %%
@btime f_t_inds_vall(stt_v_all, dev, t_ind_int, mm2)
@btime f_t_inds_vall_d(stt_d, dev, t_ind_int, mm2)

# %%
v_st_func     = v_st([ones(nb)  for ii in 1:(nT)],[ones(nb)  for ii in 1:(nT)])
v_st_func_mut = v_st_mut([ones(nb)  for ii in 1:(nT)],[ones(nb)  for ii in 1:(nT)])
# %%

@btime f_t_inds_vst(v_st_func, dev, t_ind_int, mm2);
@btime f_t_inds_vst_mut(v_st_func_mut, dev, t_ind_int, mm2);

# %%
@btime f_t_inds_vst_rep(v_st_func, dev, t_ind_int, mm2);
@btime f_t_inds_vst_mut_rep(v_st_func_mut, dev, t_ind_int, mm2);


# %%
@btime f_t_inds_vec(stt_v, dev, t_ind_int)
@btime f_t_inds_vec_fast(stt_v, dev, t_ind_int_fast)

# %%
struct map_s
    stt_map::Dict{Symbol, Int64}
end

struct map_inds2
    vm::Int64
    va::Int64
end

# %%
struct v_st
    vm::Vector{Vector{Float64}}
    va::Vector{Vector{Float64}}
end

# %%
mutable struct v_st_mut
    vm::Vector{Vector{Float64}}
    va::Vector{Vector{Float64}}
end