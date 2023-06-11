# cd
cd(dirname(@__FILE__))

# quasiGrad
include("./src/quasiGrad_dual.jl")

# Play!
data_path = "./test/data/c3/C3S0_20221208/D1/C3S0N00003/"
file_name = "scenario_003.json"

data_path = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
file_name = "scenario_002.json"

# %%
g = Dict("t1" => 1, "t2" => 5)

sum(v for (_,v) in g)


# %%
startup_states = Vector{Float64}.([device[ind]["startup_states"] for ind in device_inds])

# %%
cd(dirname(@__FILE__))

include("./src/quasiGrad_dual.jl")

data_path = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
file_name = "scenario_002.json"
json_file = data_path*file_name

json_data = Dict{String,Any}()
open(json_file, "r") do io
    json_data = quasiGrad.JSON.parse(io)
end

# build the system struct
sys = quasiGrad.build_sys(json_data)


# %%

json_data, prm, idx, ID, sys = quasiGrad.load_and_parse_json(data_path*file_name);
#acline_params, acline_mappings, acline_ids = quasiGrad.parse_json_acline(json_data);

# %%
bus_inds              = keys(json_data["network"]["bus"])
buses                 = json_data["network"]["bus"]
bus_id                = [buses[ind]["uid"] for ind in bus_inds]
tt  = [buses[ind]["active_reserve_uids"] for ind in bus_inds]
# %%
for ii = 1:73
    println(buses[ii]["active_reserve_uids"])
end

# %%
sum((ii["device_type"]=="producer") for ii in json_data["network"]["simple_dispatchable_device"])

# %%
d = Dict(name => Vector{Int64}() for name in 1:100)

#for ii in 1:73
#    println(json_data["network"]["shunt"][ii]["bs"])
#    println(ii)
#end

# %%
keys(json_data["network"]["two_winding_transformer"])

#getidx.(Ref(json_data["network"]["ac_line"]))

#
#[json_data["network"]["ac_line"][i]["r"] for i in 1:N]


# %%
N_lines = 105

@btime[json_data["network"]["ac_line"][i]["fr_bus"] for i in 1:N_lines];

tt = keys(json_data["network"]["ac_line"])
@btime ty = [json_data["network"]["ac_line"][i]["fr_bus"] for i in tt];

@btime tt2 = [value["fr_bus"] for value in values(json_data["network"]["ac_line"])];

# %%
#for value in values(json_data["network"]["ac_line"])
#    println(value)
#end
#joinpath(case_directory, "case.json")

@btime tt2 = [[value["r"];value["x"];value["b"];parse(Int64,value["fr_bus"]);parse(Int64,value["to_bus"])] for value in values(json_data["network"]["ac_line"])];

# %%
#@btime 
tt2 = [a=[value["r"]]; b=[value["x"]] for value in values(json_data["network"]["ac_line"])];


# %%
@btime tta = [value["r"] for value in values(json_data["network"]["ac_line"])];
@btime ttb = [value["x"] for value in values(json_data["network"]["ac_line"])];
@btime ttc = [value["b"] for value in values(json_data["network"]["ac_line"])];
@btime ttd = [value["fr_bus"] for value in values(json_data["network"]["ac_line"])];
@btime tte = [value["to_bus"] for value in values(json_data["network"]["ac_line"])];

# %%
@time ttd = [value["fr_bus"] for value in values(json_data["network"]["ac_line"])];
@time parse.(Int64,ttd);
# %% @btime parse.(Int64,tte);

@btime ttd = [parse(Int64,value["fr_bus"]) for value in values(json_data["network"]["ac_line"])];
@btime ttd = [parse(Int16,value["fr_bus"]) for value in values(json_data["network"]["ac_line"])];

# %%

tt2 = [[value["r"];value["x"];value["b"];parse(Int64,value["fr_bus"]);parse(Int64,value["to_bus"])] for value in values(json_data["network"]["ac_line"])];

# %%

@btime [ac_line["r"] for ac_line in values(json_data["network"]["ac_line"])]
@btime Float32.([ac_line["r"] for ac_line in values(json_data["network"]["ac_line"])])

# %%
@btime findall(x -> x .== "C34",line_id);
@btime "C34" .== line_id;
@btime findfirst(x -> x=="C34",line_id)
@btime findfirst(x -> x.=="C34",line_id)

# %%
d = Dict{Symbol,Float64}(:sam => [1.2; 3.4])

# %%
I = [1, 4, 3, 5]; J = [4, 7, 18, 9]; V = [1.4, 2.3, -5, 3];
S = sparse(I,J,V)

# %%

varstack_idx = Dict()
    for ii in 1:3
        varstack_idx[ii] = "sam"
    end

# %% 
for jac[:acline][idx[:jac][:acline_p_fr],idx[:varstack][:t1][:acline_vmfr]] = grad[:acline_dpfr_dvmfr]

# %%

CartesianIdx.(Tuple.([1,2]))

M[ii,jj] = jj for (ii,jj) in (idx[:jac][:acline_p_fr],idx[:varstack][:t1][:acline_vmfr])

# %%
inds = [[1,2], [2,3],[4,4]];
A = reshape(1:16,4,4);

A[CartesianIdx.(Tuple.(inds))]
# %%

ind = [idx[:jac][:acline_p_fr], idx[:varstack][:t1][:acline_vmfr]]'

# %%
Tuple.([[3;5;7]',[1;2;4]'])

# %%
Tuple.([[3 5],[1 2]])

# %%
tt = [4,6]
x  = [4,5,6]
Int64.(indexin(tt, x))

bus_id_to_number(id) = Int64.(indexin(id, bus_id))

# %% test!
using BenchmarkTools

# %%
n  = 2000
A  = randn(n,n) 
V  = randn(n,n)
@btime A[1:end-1,:]*V[:,1]
@btime A[1:end-1,:]*@view(V[:,1])
@btime @view(A[1:end-1,:])*@view(V[:,1])
t = 1

# %%
for dev = 1:208
    tii = :t4
    println(quasiGrad.get_tminup(tii, dev, prm))
end


# %%
n    = 100000
val  = randn(n)
minv = randn(n)
maxv = randn(n)
dev = min.(val - minv,0.0) + max.(val - maxv,0.0)

@btime dev + val;
@btime min.(max.(val,minv),maxv);

# %%

Ts_mndn = [[Vector{Float64}(undef, 42) for _ = 1:(sys.nT)] for _ = 1:100]
@btime Ts_mndn[1][1] = zeros(5)

# %%
@btime [[Vector{Float64}(undef, Int(sys.nT/2)) for t = 1:(sys.nT)] for d = 1:sys.ndev];
@btime [[Vector{Symbol}(undef, Int(sys.nT/2)) for tii in prm.ts.time_key_ind] for d = 1:sys.ndev];

# %%
#maximum([state[:pr_p][tii][dev] for dev in idx[:cs_zone][zone]])

[ [1.0,2.0,3.0,4.0][dev] for dev in []] == 0

# %%
a = [1.3,5.4,-4]
b = [2.1,4.4,-5]
c = [0.0,0.0,0.0]

t = [argmax([a[ii],b[ii],0]) for ii in 1:length(a)]

t2 = [argmax([aa,bb,0]) for (aa,bb) in zip(a,b)]

# %%
M = [1 1 0; 0 1 0; 0 0 1]

# %%
A = randn(1000,1000)

@btime B = @view A[2:end,2:end]
@btime C = A[2:end,2:end]

# %% Test
using Preconditioners
using SparseArrays
using LinearAlgebra

# %%
A  = sprand(10, 10, 0.25)
A  = A + A' + 10*I
p = CholeskyPreconditioner(A, 1);
A_approx =  ((p.ldlt.L + I)*sparse(diagm(p.ldlt.D))*(p.ldlt.L' + I))
norm(A_approx - A[p.ldlt.P, p.ldlt.P])

# %% not even close?
norm(A_approx - A[p.ldlt.P, p.ldlt.P])

# Diagonal preconditioner
#p = @btime CholeskyPreconditioner(A, 0);
#p = @btime cholesky(A);

#@btime p = CholeskyPreconditioner(A, 0);
#@btime p = LinearAlgebra.Cholesky(MA);

# %%

t1 = ((p.ldlt.L)*sparse(diagm(p.ldlt.D))*(p.ldlt.L'))
t2 = A[p.ldlt.P, p.ldlt.P]

println(nnz(t1))
println(nnz(t2))
norm(p \ b - (A) \ b, Inf)

#((p.ldlt.L)*sparse(diagm(p.ldlt.D))*(p.ldlt.L'))\ b

# %%
b = ones(100)
C = CholeskyPreconditioner(A, 0)
norm(C \ b - (A) \ b, Inf)

# %%
C \ b

((C.ldlt.L)*diagm(C.ldlt.D)*(C.ldlt.L'))\ b

# %%
tt = LinearAlgebra.cholesky(A)

# %%
sparse(tt.L) * sparse(tt.L)' ≈ A[C.p, C.p]

# %%

Int64.([device[ind]["initial_status"]["on_status"] for ind in device_inds])

# %%
device_inds      = keys(json_data["network"]["simple_dispatchable_device"])
device           = json_data["network"]["simple_dispatchable_device"]
device_id        = [device[ind]["uid"] for ind in device_inds]
bus              = [device[ind]["bus"] for ind in device_inds]
device_type      = [device[ind]["device_type"] for ind in device_inds]

# %%
#tik = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

# state
y = Dict(tik[ii] => Float64[] for ii in 1:10)

# %%
for bus = 1:sys.nb
    # get the devices tied to this bus
    bus_id          = prm.bus.id[bus]
    dev_on_bus_inds = findall(x -> x == bus_id, prm[:device][:bus])

    # are the devices consumers or producers?
    pr_devs_on_bus = dev_on_bus_inds[in.(dev_on_bus_inds,Ref(pr_inds))]
    cs_devs_on_bus = dev_on_bus_inds[in.(dev_on_bus_inds,Ref(cs_inds))]

    # update dictionaries
    bus_to_pr[bus] = pr_devs_on_bus
    bus_to_cs[bus] = cs_devs_on_bus
end

# %%
t = Dict(:pb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)),
         :qb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)))

# %%
# define the time elements
tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]

# %% state -- use initial values
@btime state#

# %%
for (t_ind, tii) in enumerate(prm[:time_series][:time_keys])
    println(t_ind)
    println(tii)
end

# %%
t = randn(100000)
t2 = 4.3
@btime push!(t,t2);

t = randn(100000)
t2 = 4.3
@btime append!(t,t2);

# %%
F = prm.dev.startup_states[1]

for f in F
    println(f)
end


# %%
tt = [[Vector{Float64}(undef, 4) for i = 1:5] for i = 1:6]
length()

tttt = [[[Vector{Float64}(undef, length(prm.dev.cost[dev][tii])) for vecs = 1:3] for tii = 1:5] for dev = 1:3]

# %% Vector{Vector{Vector{Vector{Float64}}}}
cst  = [0, 1.0, 5.0, 10.0]
pbk  = [0, 1.0, 2.0, 3.0]
pcm  = [0, 1.0, 3.0, 6.0]
p    = 3.5
dt   = 1.0
cost = dt*sum(cst[ii]*max(min(p - pcm[ii-1], pbk[ii]), 0.0) for ii in 2:4)




# %% --------- ctg

build_ctg = true
# note, the reference bus is always bus #1
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

# get the diagonal admittance matrix   => Ybs == "b susceptance"
Ybs = quasiGrad.spdiagm(ac_b_params)
Yb  = E'*Ybs*E
Ybr = @view Yb[2:end,2:end]

# get the flow matrix
Yfr = Ybs*Er

# build the low-rank contingecy updates
#
# base: Y_b*theta_b = p
# ctg:  Y_c*theta_c = p
#       Y_c = Y_b + uk'*uk
ctg_vectors = Dict(ctg_ii => Vector{Int64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)
ctg_params  = Dict(ctg_ii => Vector{Float64}(undef, length(prm.ctg.components[ctg_ii])) for ctg_ii in 1:sys.nctg)

# should we build the full ctg matrices?
if build_ctg == true
    nac   = sys.nl + sys.nx
    Ybr_k = Dict(ctg_ii => quasiGrad.spzeros(nac,nac) for ctg_ii in 1:sys.nctg)
else
    Ybr_k = 0
end

for ctg_ii in 1:sys.nctg
    # components
    cmpnts = prm.ctg.components[ctg_ii]
    for (cmp_ii,cmp) in enumerate(cmpnts)
        # get the cmp index and b
        cmp_index         = findfirst(x -> x == cmp, ac_ids) 
        cmp_b             = -ac_b_params[cmp_index]

        # output
        ctg_vectors[ctg_ii][cmp_ii] = cmp_index
        ctg_params[ctg_ii][cmp_ii]  = cmp_b

        #y_diag[cmp_index] = sqrt(cmp_b)
        # we record these in ctg
        # ctg_vectors[ctg_ii]
    end

    # next, should we build the actual, full ctg matrix?
    if build_ctg == true
        Ybs_k = copy(Ybs)
        Ybs_k[ctg_vectors[ctg_ii],ctg_vectors[ctg_ii]] .= 0
        Ybr_k[ctg_ii] = Er'*Ybs_k*Er

        println(Ybs_k)
    end
end

# %%
@btime t1 = [Vector{Float64}(undef,1000) for ii in 1:1000];
@btime t2 = Dict(ii => Vector{Float64}(undef,1000) for ii in 1:1000);
@btime t3 = Dict(ii => zeros(1000) for ii in 1:1000);

# %% -- show that my implementation of WMI works
A = randn(4,4)
A = A + A'
v = randn(4,1)
u = randn(4,1)

m1 = inv(A + v*v' + u*u')
m2 = inv(A) - inv(A)*v*v'*inv(A)/(1+(v'*inv(A)*v)[1]) - inv(A)*u*u'*inv(A)/(1+(u'*inv(A)*u)[1])


# %% ------------
d = Dict("outer" => [Dict("line1" => "hello",
                          "line2" => "test")])
#JSON.print("d.json", d)

# %%
open("d.json", "w") do io
    JSON.print(io, soln_dict)
end

# %% test su/sd defintions
uont   = 0
uontm1 = 1
du     = uont - uontm1

su =  max.(uont - uontm1, 0)
sd = -min.(uont - uontm1, 0)

du_def =  su - sd

println(du)
println(du_def)

# %% test

a = collect(1:1000000)
@btime filter!(x->x!=463230,a);

a = collect(1:1000000)
@btime deleteat!(a, a .== 463230);

# %%

a = collect(1:1000000)
@btime tt = findfirst(x -> x == 463230, a)
@btime deleteat!(a,tt);

# %%
for dev in reverse(1:sys.ndev)
    println(dev)
end



dev_ind = findfirst(x -> x == dev, a)
update_states[tii][:u_on_dev]
update_states[tii]
@btime tt = findfirst(x -> x == 463230, a)
@btime deleteat!(a,tt);

# 4. rounded and fixed (i.e., -- ibr) =======================================
[:u_on_dev]

# good way to find states to fix/remove!
@btime tt = findfirst(x -> x == 463230, a)
@btime deleteat!(a,tt);


# %% Gurobi
model = Model(quasiGrad.Gurobi.Optimizer)
empty!(model)

# Model Settings
# set_optimizer_attribute(model, "OutputFlag", 0)
tkeys = [:t1, :t2, :t3]
#vrb = Dict{Symbol, Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "u_on_dev_t$(ii)") for ii in 1:3)
vrb = Dict{Symbol, Dict{Symbol, quasiGrad.Vector{quasiGrad.JuMP.VariableRef}}}(:u_on_dev => Dict(tkeys[ii] => @variable(model,[1:sys.ndev],base_name = "u_on_dev_t$(ii)") for ii in 1:3))

# %% loop over each time period
tii = :t1

# duration
dt = prm.ts.duration[tii]

# devices
#for dev in 1:sys.ndev
dev = 1
cst = prm.dev.cum_cost_blocks[dev][t_ind][1]  # cost for each block (leading with 0)
pbk = prm.dev.cum_cost_blocks[dev][t_ind][2]  # power in each block (leading with 0)
pcm = prm.dev.cum_cost_blocks[dev][t_ind][3]  # accumulated power for each block!
nbk = length(pbk)

println(nbk)

# get the cost!
stt[:zen_dev][tii][dev] = dt*sum(cst[ii]*max(min(stt[:dev_p][tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)

println(stt[:zen_dev][tii][dev])

#end

# %%

c = 0.0
for tii in prm.ts.time_keys
    for dev in idx.cs_devs
        c += stt[:zen_dev][tii][dev]
    end

    if tii != 1
        #stt[:p_on][tii][1] = 0
    end
end

println(c)


# %%
ct = 0
for (t_ind,tii) in enumerate(prm.ts.time_keys)
    # duration
    dt = prm.ts.duration[tii]

    # devices
    dev = 1
    cst = prm.dev.cum_cost_blocks[dev][t_ind][1]  # cost for each block (leading with 0)
    pbk = prm.dev.cum_cost_blocks[dev][t_ind][2]  # power in each block (leading with 0)
    pcm = prm.dev.cum_cost_blocks[dev][t_ind][3]  # accumulated power for each block!
    nbk = length(pbk)

    # get the cost!
    ct += dt*sum(cst[ii]*max(min(stt[:dev_p][tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)
    println(ct)

end

#
0.0
0.05496935335313412
0.05496935335313412
0.05496935335313412
0.05496935335313412
0.1549693533531341

0.0
1000.0
2500.0
10000.0
50000.0
100000.0

0.275
0.2815417215550038
0.2425
0.2425

t1 = 0.15630834431100077*100000.0 + 0.056308344311000756*50000.0 + 0.05496935335313412*1000.0 + 0.010091939940597677*2500.0
t2 = 0.15630834431100077*100000.0 + 0.056308344311000756*50000.0 + 0.05496935335313412*1000.0 + (0.016591939940597628+4.172155500381525e-5)*2500.0
t3 = 0.15630834431100077*100000.0 + 0.056308344311000756*50000.0 + 0.032561293293731766*1000.0
t4 = 0.1549693533531341*100000.0 + 0.05496935335313412*50000.0 + 0.032561293293731766*1000.0

0.25*(t1 + t2 + t3 + t4)

# %% test the derivative
epsilon = 0.0000001
x = randn(1)[1]
f = -min(x,0)

dfdx     = -sign(f)
dfdx_num = (-min(x+epsilon,0)-f)/epsilon

println(dfdx)
println(dfdx_num)

# %%
cst = prm.dev.cum_cost_blocks[dev][t_ind][1]  # cost for each block (leading with 0)
pbk = prm.dev.cum_cost_blocks[dev][t_ind][2]  # power in each block (leading with 0)
pcm = prm.dev.cum_cost_blocks[dev][t_ind][3]  # accumulated power for each block!
nbk = length(pbk)
dt = 1

# get the cost!
stt[:zen_dev][tii][dev] = dt*sum(cst[ii]*max(min(stt[:dev_p][tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)
qG.eval_grad = true

# evaluate the grd?
if qG.eval_grad
    # what is the index of the "active" block?
    del = stt[:dev_p][tii][dev] .- pcm
    active_block_ind = argmin(del[del .>= 0])
    grd[:zen_dev][:dev_p][tii][dev] = dt*cst[active_block_ind + 1] # where + 1 is due to the leading 0
end


# %% test vector calls
v = randn(1000000)
t = collect(1:1000000)

@btime v[1:1000000];
@btime v[t];

# %%
@btime v[:];
@btime v[Colon()];

# %%
dict_test = Dict(:t1 => 1.0,
                 :t2 => 1.0)

for (k,v) in dict_test
    dict_test[k] = 5
end

# %% plot tests -- penalties
v = -2:0.001:2
plot(v,v.^2)
plot!(v,abs.(v))
plot!(v,abs.(v).^1.25)

# %% test dict speed
v1 = randn(1000000)
v2 = Dict(ii => v1[ii] for ii in 1:1000000)

vkeys = [Symbol("t"*string(ii)) for ii in 1:1000000]
v3    = Dict(vkeys[ii] => v1[ii] for ii in 1:1000000)

# %%
@btime v1[502045]^2;
@btime v2[502045]^2;
@btime v3[:t502045]^2;

# %% time
tnan = [sum(stt[:dev_p][tt]) for tt in prm.ts.time_keys]
ti = findall(isnan.(tnan))

# %% space
pid = findall(isnan.(stt[:dev_p][:t14]))

# %%
for tii in prm.ts.time_keys
    stt[:zt][tii] = 
    # consumer and revenues
    sum(stt[:zen_dev][tii][dev] for dev in idx.cs_devs) - 
    # producer costs
    sum(stt[:zen_dev][tii][dev] for dev in idx.pr_devs) - 
    # startup costs
    sum(stt[:zsu_dev][tii]) - 
    sum(stt[:zsu_acline][tii]) - 
    sum(stt[:zsu_xfm][tii]) - 
    # shutdown costs
    sum(stt[:zsd_dev][tii]) - 
    sum(stt[:zsd_acline][tii]) - 
    sum(stt[:zsd_xfm][tii]) - 
    # on-costs
    sum(stt[:zon_dev][tii]) - 
    # time-dependent su costs
    sum(stt[:zsus_dev][tii]) - 
    # ac branch overload costs
    sum(stt[:zs_acline][tii]) - 
    sum(stt[:zs_xfm][tii]) - 
    # local reserve penalties
    sum(stt[:zrgu][tii]) -
    sum(stt[:zrgd][tii]) -
    sum(stt[:zscr][tii]) -
    sum(stt[:znsc][tii]) -
    sum(stt[:zrru][tii]) -
    sum(stt[:zrrd][tii]) -
    sum(stt[:zqru][tii]) -
    sum(stt[:zqrd][tii]) -
    # power mismatch penalties
    sum(stt[:zp][tii]) -
    sum(stt[:zq][tii]) -
    # zonal reserve penalties (P)
    sum(stt[:zrgu_zonal][tii]) -
    sum(stt[:zrgd_zonal][tii]) -
    sum(stt[:zscr_zonal][tii]) -
    sum(stt[:znsc_zonal][tii]) -
    sum(stt[:zrru_zonal][tii]) -
    sum(stt[:zrrd_zonal][tii]) -
    # zonal reserve penalties (Q)
    sum(stt[:zqru_zonal][tii]) -
    sum(stt[:zqrd_zonal][tii])
end

# %%

del = qG.delta
for tii in prm.ts.time_keys
    stt[:zt][tii] = 
    # - 
    # penalized constraints
    -del*sum(stt[:zhat_mndn][tii]) - 
    del*sum(stt[:zhat_mnup][tii]) - 
    del*sum(stt[:zhat_rup][tii]) - 
    del*sum(stt[:zhat_rd][tii]) - 
    del*sum(stt[:zhat_rgu][tii]) - 
    del*sum(stt[:zhat_rgd][tii]) - 
    del*sum(stt[:zhat_scr][tii]) - 
    del*sum(stt[:zhat_nsc][tii]) - 
    del*sum(stt[:zhat_rruon][tii]) - 
    del*sum(stt[:zhat_rruoff][tii]) -
    del*sum(stt[:zhat_rrdon][tii]) -
    del*sum(stt[:zhat_rrdoff][tii]) -
    # common set of pr and cs constraint variables (see below)
    del*sum(stt[:zhat_pmax][tii]) - 
    del*sum(stt[:zhat_pmin][tii]) - 
    del*sum(stt[:zhat_pmaxoff][tii]) - 
    del*sum(stt[:zhat_qmax][tii]) - 
    del*sum(stt[:zhat_qmin][tii]) - 
    del*sum(stt[:zhat_qmax_beta][tii]) - 
    del*sum(stt[:zhat_qmin_beta][tii])
end

# %%
t = 1:10000
@btime sum(t)
@btime sum(t[ii] for ii in 1:10000);

# %%
for ii in 1:0
    println(ii)
end

# %% 
t = randn(100000)
@btime y = abs.(t).^1.25;
@btime y = sqrt.(t.^2 .+ 0.01);

@btime y = 1.25*abs.(t).^0.25.*sign.(t);
@btime y = t./sqrt.(t.^2 .+ 0.01);
m = 1;

# %%

# loop over each time period
for (t_ind,tii) in enumerate(prm.ts.time_keys)
    # duration
    dt = prm.ts.duration[tii]

    # devices
    for dev in 1:sys.ndev
        cst = prm.dev.cum_cost_blocks[dev][t_ind][1]  # cost for each block (leading with 0)
        pbk = prm.dev.cum_cost_blocks[dev][t_ind][2]  # power in each block (leading with 0)
        pcm = prm.dev.cum_cost_blocks[dev][t_ind][3]  # accumulated power for each block!
        nbk = length(pbk)

        # get the cost!
        println(dev)
        println(tii)
        stt[:zen_dev][tii][dev] = dt*sum(cst[ii]*max(min(stt[:dev_p][tii][dev] - pcm[ii-1], pbk[ii]), 0.0)  for ii in 2:nbk; init=0.0)
        
        # evaluate the grd? 
        #
        # WARNING -- this will break if stt[:dev_p] > pcm[end]! It will
        #            mean the device power is out of bounds, and this will
        #            call a price curve which does not exist.
        #                  ~ clipping will fix ~
        if qG.eval_grad
            # what is the index of the "active" block?
            del = stt[:dev_p][tii][dev] .- pcm
            # we add in a tiny bit of tollerance here for the
            active_block_ind = argmin(del[del .>= 1e-6])
            grd[:zen_dev][:dev_p][tii][dev] = dt*cst[active_block_ind + 1] # where + 1 is due to the leading 0
        end
    end
end

# %%
using BenchmarkTools

# %%
tkeys = [Symbol("t"*string(ii)) for ii in 1:100]
@btime y1 = Dict(tkeys[ii] => [Vector{Float64}(undef,(10000)) for jj in 1:100] for ii in 1:100);  
@btime y2 = Dict(tkeys[ii] => [zeros(10000)                   for jj in 1:100] for ii in 1:100);            

# %%
GRB = Dict(
        :u_on_dev  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_on      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :dev_q     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_rgu     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_rgd     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_scr     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_nsc     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_rru_on  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_rru_off => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_rrd_on  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_rrd_off => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :q_qru     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :q_qrd     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)))

# %%
@btime y = Vector{Float64}(undef,(100000));

# %%
tt = Dict(dev => zeros(prm.dev.num_sus[dev]) for dev in 1:sys.ndev)

ty = [Vector{Float64}(undef,(prm.dev.num_sus[dev])) for dev in 1:sys.ndev]

# %%
    # define the time elements
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]
    stt = Dict(
        # network -- set all network voltages to there given initial values
        :vm              => Dict(tkeys[ii] => copy(prm.bus.init_vm) for ii in 1:(sys.nT)),
        :va              => Dict(tkeys[ii] => copy(prm.bus.init_va) for ii in 1:(sys.nT)),
        # aclines
        :acline_pfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :acline_qfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :acline_pto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :acline_qto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :u_on_acline     => Dict(tkeys[ii] => ones(sys.nl)                    for ii in 1:(sys.nT)),
        :u_su_acline     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        :u_sd_acline     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nl)) for ii in 1:(sys.nT)),
        # xfms
        :phi          => Dict(tkeys[ii] => copy(prm.xfm.init_phi)      for ii in 1:(sys.nT)),
        :tau          => Dict(tkeys[ii] => copy(prm.xfm.init_tau)      for ii in 1:(sys.nT)),
        :xfm_pfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :xfm_qfr      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :xfm_pto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :xfm_qto      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :u_on_xfm     => Dict(tkeys[ii] => ones(sys.nx)                    for ii in 1:(sys.nT)),
        :u_su_xfm     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        :u_sd_xfm     => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nx)) for ii in 1:(sys.nT)),
        # all reactive ac flows -- used for ctgs
        :ac_qfr       => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nac)) for ii in 1:(sys.nT)),
        :ac_qto       => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nac)) for ii in 1:(sys.nT)),
        # all line phase shifts -- used for ctgs
        :ac_phi       => Dict(tkeys[ii]  => zeros(sys.nac) for ii in 1:(sys.nT)),
        # dc lines
        :dc_pfr       => Dict(tkeys[ii] => copy(prm.dc.init_pdc_fr)  for ii in 1:(sys.nT)),
        :dc_pto       => Dict(tkeys[ii] => copy(-prm.dc.init_pdc_fr) for ii in 1:(sys.nT)),
        :dc_qfr       => Dict(tkeys[ii] => copy(prm.dc.init_qdc_fr)  for ii in 1:(sys.nT)),
        :dc_qto       => Dict(tkeys[ii] => copy(prm.dc.init_qdc_to)  for ii in 1:(sys.nT)),
        # shunts
        :sh_p         => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nsh)) for ii in 1:(sys.nT)),
        :sh_q         => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.nsh)) for ii in 1:(sys.nT)),
        :u_step_shunt => Dict(tkeys[ii] => copy(prm.shunt.init_step)    for ii in 1:(sys.nT)),
        # producing and consuming devices
        :u_on_dev   => Dict(tkeys[ii] => ones(sys.ndev)                    for ii in 1:(sys.nT)),
        :dev_p      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)), 
        :dev_q      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        # devices variables
        :u_su_dev  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :u_sd_dev  => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)),
        :p_on      => Dict(tkeys[ii] => zeros(sys.ndev)                   for ii in 1:(sys.nT)), 
        :p_su      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)), 
        :p_sd      => Dict(tkeys[ii] => Vector{Float64}(undef,(sys.ndev)) for ii in 1:(sys.nT)), 
        # device powers
        :p_rgu     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rgd     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_scr     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_nsc     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rru_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rru_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rrd_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :p_rrd_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :q_qru     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :q_qrd     => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # scoring and penalties
        :zctg       => Dict(tkeys[ii] => zeros(sys.nctg) for ii in 1:(sys.nT)),    
        # revenues -- with devices, lump consumers and producers together
        :zen_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # startup costs
        :zsu_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zsu_acline => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)), 
        :zsu_xfm    => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
        # shutdown costs
        :zsd_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zsd_acline => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),  
        :zsd_xfm    => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
        # on-costs
        :zon_dev    => Dict(tkeys[ii]  => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        # time-dependent su costs
        :zsus_dev   => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        # ac branch overload costs
        :zs_acline  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
        :zs_xfm     => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
        # local reserve penalties (producers)
        :zrgu       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zrgd       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zscr       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :znsc       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zrru       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zrrd       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zqru       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :zqrd       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        # power mismatch penalties
        :zp          => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)),
        :zq          => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT)),
        # zonal reserve penalties (P)
        :zrgu_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zrgd_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zscr_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :znsc_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zrru_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :zrrd_zonal   => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        # zonal reserve penalties (Q)
        :zqru_zonal   => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        :zqrd_zonal   => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        # endogenous zonal state
        :p_rgu_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rgd_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_scr_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_nsc_zonal_REQ => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        # actual zonal state
        :p_rgu_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rgd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_scr_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_nsc_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :p_rrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT)),
        :q_qru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        :q_qrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT)),
        # penalized constraints
        :zhat_mndn         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_mnup         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rup          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rd           => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rgu          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rgd          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_scr          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_nsc          => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rruon        => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rruoff       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rrdon        => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_rrdoff       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        # more penalized constraints
        :zhat_pmax         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_pmin         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_pmaxoff      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmax         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmin         => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmax_beta    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)), 
        :zhat_qmin_beta    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)))

# %% test types
typeof(idx.dev_qzone)

# %% ==========

# build the system struct
# include("./src/core/structs.jl")
include("./src/io/read_data.jl")
include("./src/core/initializations.jl")

# %%parse the network elements
dc_prm = parse_json_dc(jsn)



#=

# %%
prm.dc.dcline_inds   prm.dc.dcline_inds
prm.dc.fr_bus        prm.dc.fr_bus
prm.dc.to_bus        prm.dc.to_bus    
prm.dc.id            prm.dc.id       
prm.dc.pdc_ub        prm.dc.pdc_ub     
prm.dc.qdc_fr_lb     prm.dc.qdc_fr_lb  
prm.dc.qdc_to_lb     prm.dc.qdc_to_lb  
prm.dc.qdc_fr_ub     prm.dc.qdc_fr_ub  
prm.dc.qdc_to_ub     prm.dc.qdc_to_ub  
prm.dc.init_pdc_fr   prm.dc.init_pdc_fr
prm.dc.init_qdc_fr   prm.dc.init_qdc_fr
prm.dc.init_qdc_to   prm.dc.init_qdc_to








# %%

struct Dc
    dcline_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}
    fr_bus::Vector{String}
    to_bus::Vector{String}
    id::Vector{String}
    pdc_ub::Vector{Float64}
    qdc_fr_lb::Vector{Float64}
    qdc_to_lb::Vector{Float64}
    qdc_fr_ub::Vector{Float64}
    qdc_to_ub::Vector{Float64}
    init_pdc_fr::Vector{Float64}
    init_qdc_fr::Vector{Float64}
    init_qdc_to::Vector{Float64}
end

# %%
prm.bus.bus_num         prm.bus.bus_num
prm.bus.id              prm.bus.id            
prm.bus.vm_ub           prm.bus.vm_ub         
prm.bus.vm_lb           prm.bus.vm_lb         
prm.bus.base_nom_volt   prm.bus.base_nom_volt 
prm.bus.init_vm         prm.bus.init_vm       
prm.bus.init_va         prm.bus.init_va      
prm.bus.active_rsvid    prm.bus.active_rsvid  
prm.bus.reactive_rsvid  prm.bus.reactive_rsvid


struct Bus
    bus_num::Vector{Int64}
    id::Vector{String}
    vm_ub::Vector{Float64}
    vm_lb::Vector{Float64}
    base_nom_volt::Vector{Float64}
    init_vm::Vector{Float64}
    init_va::Vector{Float64}
    active_rsvid::Vector{Vector{String}}
    reactive_rsvid::Vector{Vector{String}}
end

# %%
prm.xfm.id                  prm.xfm.id                 
prm.xfm.g_sr               prm.xfm.g_sr               
prm.xfm.b_sr               prm.xfm.b_sr               
prm.xfm.b_ch               prm.xfm.b_ch               
prm.xfm.g_fr               prm.xfm.g_fr                
prm.xfm.b_fr               prm.xfm.b_fr               
prm.xfm.g_to               prm.xfm.g_to               
prm.xfm.b_to               prm.xfm.b_to               
prm.xfm.tm_lb              prm.xfm.tm_lb              
prm.xfm.tm_ub              prm.xfm.tm_ub              
prm.xfm.ta_lb              prm.xfm.ta_lb              
prm.xfm.ta_ub              prm.xfm.ta_ub              
prm.xfm.init_on_status     prm.xfm.init_on_status    
prm.xfm.init_tau           prm.xfm.init_tau           
prm.xfm.init_phi           prm.xfm.init_phi           
prm.xfm.mva_ub_em          prm.xfm.mva_ub_em          
prm.xfm.mva_ub_nom         prm.xfm.mva_ub_nom         
prm.xfm.disconnection_cost prm.xfm.disconnection_cost 
prm.xfm.connection_cost    prm.xfm.connection_cost    
prm.xfm.xfm_inds           prm.xfm.xfm_inds           
prm.xfm.fr_bus             prm.xfm.fr_bus             
prm.xfm.to_bus             prm.xfm.to_bus             
prm.xfm.J_fpd              prm.xfm.J_fpd              
prm.xfm.J_fwr              prm.xfm.J_fwr  

struct Xfm
    id::Vector{String}
    g_sr::Vector{Float64}
    b_sr::Vector{Float64}
    b_ch::Vector{Float64}
    g_fr::Vector{Float64}
    b_fr::Vector{Float64}
    g_to::Vector{Float64}
    b_to::Vector{Float64}
    tm_lb::Vector{Float64}
    tm_ub::Vector{Float64}
    ta_lb::Vector{Float64}
    ta_ub::Vector{Float64}
    init_on_status::Vector{Float64}
    init_tau::Vector{Float64}
    init_phi::Vector{Float64}
    mva_ub_em::Vector{Float64}
    mva_ub_nom::Vector{Float64}
    disconnection_cost::Vector{Float64}
    connection_cost::Vector{Float64}
    xfm_inds::Vector{Int64}
    fr_bus::Vector{String}
    to_bus::Vector{String}
    J_fpd::Vector{Int64}
    J_fwr::Vector{Int64}
end

# %%
prm.acline.id

prm.acline.id                 prm.acline.id               
prm.acline.g_sr               prm.acline.g_sr              
prm.acline.b_sr               prm.acline.b_sr               
prm.acline.b_ch               prm.acline.b_ch               
prm.acline.g_fr               prm.acline.g_fr               
prm.acline.b_fr               prm.acline.b_fr                
prm.acline.g_to               prm.acline.g_to               
prm.acline.b_to               prm.acline.b_to               
prm.acline.init_on_status     prm.acline.init_on_status     
prm.acline.mva_ub_em          prm.acline.mva_ub_em          
prm.acline.mva_ub_nom         prm.acline.mva_ub_nom         
prm.acline.disconnection_cost prm.acline.disconnection_cost 
prm.acline.connection_cost    prm.acline.connection_cost    
prm.acline.line_inds          prm.acline.line_inds          
prm.acline.fr_bus             prm.acline.fr_bus             
prm.acline.to_bus             prm.acline.to_bus  

struct Acline
    id::Vector{String}
    g_sr::Vector{Float64}
    b_sr::Vector{Float64}
    b_ch::Vector{Float64}
    g_fr::Vector{Float64}
    b_fr::Vector{Float64}
    g_to::Vector{Float64}
    b_to::Vector{Float64}
    init_on_status::Vector{Float64}
    mva_ub_em::Vector{Float64}
    mva_ub_nom::Vector{Float64}
    disconnection_cost::Vector{Float64}
    connection_cost::Vector{Float64}
    line_inds::LinearIndices{1, Tuple{Base.OneTo{Int64}}}
    fr_bus::Vector{String}
    to_bus::Vector{String}
end

# %%
prm.reserve.pzone_inds  prm.reserve.pzone_inds
prm.reserve.qzone_inds  prm.reserve.qzone_inds
prm.reserve.id_pzone    prm.reserve.id_pzone 
prm.reserve.id_qzone    prm.reserve.id_qzone 
prm.reserve.rgu_sigma   prm.reserve.rgu_sigma 
prm.reserve.rgd_sigma   prm.reserve.rgd_sigma 
prm.reserve.scr_sigma   prm.reserve.scr_sigma 
prm.reserve.nsc_sigma   prm.reserve.nsc_sigma 
prm.reserve.rru_min     prm.reserve.rru_min   
prm.reserve.rrd_min     prm.reserve.rrd_min   
prm.reserve.qru_min     prm.reserve.qru_min   
prm.reserve.qrd_min     prm.reserve.qrd_min   

struct Reserve
    pzone_inds::Vector{Int64}
    qzone_inds::Vector{Int64}
    id_pzone::Vector{String}
    id_qzone::Vector{String}
    rgu_sigma::Vector{Float64}
    rgd_sigma::Vector{Float64}
    scr_sigma::Vector{Float64}
    nsc_sigma::Vector{Float64}
    rru_min::Vector{Vector{Float64}}
    rrd_min::Vector{Vector{Float64}}
    qru_min::Vector{Vector{Float64}}
    qrd_min::Vector{Vector{Float64}}
end

# %%
prm.vio.p_bus      prm.vio.p_bus  
prm.vio.q_bus      prm.vio.q_bus   
prm.vio.s_flow     prm.vio.s_flow  
prm.vio.e_dev      prm.vio.e_dev  
prm.vio.rgu_zonal  prm.vio.rgu_zonal
prm.vio.rgd_zonal  prm.vio.rgd_zonal
prm.vio.scr_zonal  prm.vio.scr_zonal
prm.vio.nsc_zonal  prm.vio.nsc_zonal
prm.vio.rru_zonal  prm.vio.rru_zonal
prm.vio.rrd_zonal  prm.vio.rrd_zonal
prm.vio.qru_zonal  prm.vio.qru_zonal
prm.vio.qrd_zonal  prm.vio.qrd_zonal

struct Violation
    p_bus::Float64
    q_bus::Float64
    s_flow::Float64
    e_dev::Float64
    rgu_zonal::Vector{Float64}
    rgd_zonal::Vector{Float64}
    scr_zonal::Vector{Float64}
    nsc_zonal::Vector{Float64}
    rru_zonal::Vector{Float64}
    rrd_zonal::Vector{Float64}
    qru_zonal::Vector{Float64}
    qrd_zonal::Vector{Float64}
end

# %%
prm.ts.time_keys        prm.ts.time_keys
prm.ts.tmin1            prm.ts.tmin1           
prm.ts.duration         prm.ts.duration        
prm.ts.start_time       prm.ts.start_time      
prm.ts.end_time         prm.ts.end_time        
prm.ts.start_time_dict  prm.ts.start_time_dict
prm.ts.end_time_dict    prm.ts.end_time_dict  
prm.ts.time_key_ind     prm.ts.time_key_ind

struct Timeseries
    time_keys::Vector{Symbol}     
    tmin1::Dict{Symbol, Symbol}          
    duration::Dict{Symbol, Float64}       
    start_time::Vector{Float64}     
    end_time::Vector{Float64}
    start_time_dict::Dict{Symbol, Float64}
    end_time_dict::Dict{Symbol, Float64}  
    time_key_ind::Dict{Symbol, Int64}   
end

# %%
prm.dev.device_inds                    prm.dev.device_inds                    
prm.dev.id                             prm.dev.id                            
prm.dev.bus                            prm.dev.bus                           
prm.dev.device_type                    prm.dev.device_type                    
prm.dev.startup_cost                   prm.dev.startup_cost                   
prm.dev.startup_states                 prm.dev.startup_states                 
prm.dev.num_sus                        prm.dev.num_sus                        
prm.dev.shutdown_cost                  prm.dev.shutdown_cost                  
prm.dev.startups_ub                    prm.dev.startups_ub                    
prm.dev.num_mxst                       prm.dev.num_mxst                       
prm.dev.energy_req_ub                  prm.dev.energy_req_ub                  
prm.dev.energy_req_lb                  prm.dev.energy_req_lb                  
prm.dev.num_W_enmax                    prm.dev.num_W_enmax                    
prm.dev.num_W_enmin                    prm.dev.num_W_enmin                    
prm.dev.on_cost                        prm.dev.on_cost                        
prm.dev.down_time_lb                   prm.dev.down_time_lb                   
prm.dev.in_service_time_lb             prm.dev.in_service_time_lb             
prm.dev.p_ramp_up_ub                   prm.dev.p_ramp_up_ub                   
prm.dev.p_ramp_down_ub                 prm.dev.p_ramp_down_ub                 
prm.dev.p_startup_ramp_ub              prm.dev.p_startup_ramp_ub              
prm.dev.p_shutdown_ramp_ub             prm.dev.p_shutdown_ramp_ub             
prm.dev.init_on_status                 prm.dev.init_on_status                 
prm.dev.init_p                         prm.dev.init_p                         
prm.dev.init_q                         prm.dev.init_q                         
prm.dev.init_accu_down_time            prm.dev.init_accu_down_time            
prm.dev.init_accu_up_time              prm.dev.init_accu_up_time              
prm.dev.q_linear_cap                   prm.dev.q_linear_cap                   
prm.dev.q_bound_cap                    prm.dev.q_bound_cap                    
prm.dev.q_0                            prm.dev.q_0                           
prm.dev.beta                           prm.dev.beta                          
prm.dev.q_0_ub                         prm.dev.q_0_ub                         
prm.dev.q_0_lb                         prm.dev.q_0_lb                         
prm.dev.beta_ub                        prm.dev.beta_ub                       
prm.dev.beta_lb                        prm.dev.beta_lb                       
prm.dev.J_pqe                          prm.dev.J_pqe                          
prm.dev.J_pqmax                        prm.dev.J_pqmax                        
prm.dev.p_reg_res_up_ub                prm.dev.p_reg_res_up_ub                
prm.dev.p_reg_res_down_ub              prm.dev.p_reg_res_down_ub              
prm.dev.p_syn_res_ub                   prm.dev.p_syn_res_ub                   
prm.dev.p_nsyn_res_ub                  prm.dev.p_nsyn_res_ub                  
prm.dev.p_ramp_res_up_online_ub        prm.dev.p_ramp_res_up_online_ub        
prm.dev.p_ramp_res_down_online_ub      prm.dev.p_ramp_res_down_online_ub      
prm.dev.p_ramp_res_up_offline_ub       prm.dev.p_ramp_res_up_offline_ub       
prm.dev.p_ramp_res_down_offline_ub     prm.dev.p_ramp_res_down_offline_ub     
prm.dev.on_status_ub                   prm.dev.on_status_ub                   
prm.dev.on_status_lb                   prm.dev.on_status_lb                   
prm.dev.p_ub                           prm.dev.p_ub                           
prm.dev.p_lb                           prm.dev.p_lb                           
prm.dev.q_ub                           prm.dev.q_ub                           
prm.dev.q_lb                           prm.dev.q_lb                           
prm.dev.cost                           prm.dev.cost                           
prm.dev.cum_cost_blocks                prm.dev.cum_cost_blocks                
prm.dev.p_reg_res_up_cost              prm.dev.p_reg_res_up_cost              
prm.dev.p_reg_res_down_cost            prm.dev.p_reg_res_down_cost            
prm.dev.p_syn_res_cost                 prm.dev.p_syn_res_cost                 
prm.dev.p_nsyn_res_cost                prm.dev.p_nsyn_res_cost                
prm.dev.p_ramp_res_up_online_cost      prm.dev.p_ramp_res_up_online_cost      
prm.dev.p_ramp_res_down_online_cost    prm.dev.p_ramp_res_down_online_cost    
prm.dev.p_ramp_res_up_offline_cost     prm.dev.p_ramp_res_up_offline_cost     
prm.dev.p_ramp_res_down_offline_cost   prm.dev.p_ramp_res_down_offline_cost   
prm.dev.q_res_up_cost                  prm.dev.q_res_up_cost                  
prm.dev.q_res_down_cost                prm.dev.q_res_down_cost               


# %% -- 
idx.acline_fr_bus     idx.acline_fr_bus    
idx.acline_to_bus     idx.acline_to_bus    
idx.xfm_fr_bus        idx.xfm_fr_bus       
idx.xfm_to_bus        idx.xfm_to_bus        
idx.ac_line_flows     idx.ac_line_flows    
idx.ac_xfm_flows      idx.ac_xfm_flows      
idx.ac_phi            idx.ac_phi            
idx.bus_is_acline_frs idx.bus_is_acline_frs 
idx.bus_is_acline_tos idx.bus_is_acline_tos 
idx.bus_is_xfm_frs    idx.bus_is_xfm_frs    
idx.bus_is_xfm_tos    idx.bus_is_xfm_tos    
idx.bus_is_dc_frs     idx.bus_is_dc_frs     
idx.bus_is_dc_tos     idx.bus_is_dc_tos     
idx.J_pqe             idx.J_pqe             
idx.J_pqmax           idx.J_pqmax           
idx.J_pqmin           idx.J_pqmin           
idx.J_fpd             idx.J_fpd             
idx.J_fwr             idx.J_fwr            
idx.pr                idx.pr                
idx.cs                idx.cs                
idx.sh                idx.sh                
idx.shunt_bus         idx.shunt_bus         
idx.pr_devs           idx.pr_devs           
idx.cs_devs           idx.cs_devs           
idx.pr_pzone          idx.pr_pzone          
idx.cs_pzone          idx.cs_pzone          
idx.dev_pzone         idx.dev_pzone         
idx.pr_qzone          idx.pr_qzone          
idx.cs_qzone          idx.cs_qzone          
idx.dev_qzone         idx.dev_qzone
=#

# %%
    # grd = grad
    tkeys = [Symbol("t"*string(ii)) for ii in 1:(sys.nT)]
    grd = Dict(
        # aclines
        :acline_pfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_qfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_pto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_qto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :zs_acline => Dict(:acline_pfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                           :acline_qfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                           :acline_pto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                           :acline_qto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        # xfms
        :xfm_pfr =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_qfr =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_pto =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_qto =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :zs_xfm => Dict(:xfm_pfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                        :xfm_qfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                        :xfm_pto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                        :xfm_qto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        # shunts
        :sh_p => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                      :g_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
        :sh_q => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                      :b_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
        :zp         => Dict(:pb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
        :zq         => Dict(:qb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
        # devices
        :zrgu       => Dict(:p_rgu => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zrgd       => Dict(:p_rgd => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zscr       => Dict(:p_scr => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :znsc       => Dict(:p_nsc => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zrru       => Dict(:p_rru_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                            :p_rru_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zrrd       => Dict(:p_rrd_on  => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                            :p_rrd_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zqru       => Dict(:q_qru => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zqrd       => Dict(:q_qrd => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        # zones
        :zrgu_zonal => Dict(:p_rgu_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP)  for ii in 1:(sys.nT))),
        :zrgd_zonal => Dict(:p_rgd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP)  for ii in 1:(sys.nT))),
        :zscr_zonal => Dict(:p_scr_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP)  for ii in 1:(sys.nT))),
        :znsc_zonal => Dict(:p_nsc_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP)  for ii in 1:(sys.nT))),
        :zrru_zonal => Dict(:p_rru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP)  for ii in 1:(sys.nT))),
        :zrrd_zonal => Dict(:p_rrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP)  for ii in 1:(sys.nT))),
        :zqru_zonal => Dict(:q_qru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ)  for ii in 1:(sys.nT))),
        :zqrd_zonal => Dict(:q_qrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ)  for ii in 1:(sys.nT))),
        :zen_dev    => Dict(:dev_p               => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        # zon_dev, zsu_dev, zsd_dev (also for lines and xfms)
        # ...
        # these next two dictionaries take the derivatives of the su/sd states with respect to
        # both the current on status, and the previous on status
        :u_su_dev => Dict(:u_on_dev            => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                          :u_on_dev_prev       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :u_sd_dev => Dict(:u_on_dev            => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                          :u_on_dev_prev       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :u_su_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :u_sd_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :u_su_xfm => Dict(:u_on_xfm            => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                          :u_on_xfm_prev       => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :u_sd_xfm => Dict(:u_on_xfm            => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                          :u_on_xfm_prev       => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        # these two following elements are unique -- they serve to collect all of the 
        # coefficients applied to the same partial derivatives (e.g.,
        # a1*dxdp, a2*dxdp, a3*dxdp => dxdp[tii][dev] = a1+a2+a3)
        :dx => Dict(:dp => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                    :dq => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))))

                    # %%
                    grd = Dict(
                        # aclines
                        :acline_pfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
                        :acline_qfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
                        :acline_pto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
                        :acline_qto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
                        :zs_acline => Dict(:acline_pfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                           :acline_qfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                           :acline_pto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                           :acline_qto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
                        # xfms
                        :xfm_pfr =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
                        :xfm_qfr =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
                        :xfm_pto =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
                        :xfm_qto =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
                        :zs_xfm => Dict(:xfm_pfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                        :xfm_qfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                        :xfm_pto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                        :xfm_qto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
                        # shunts
                        :sh_p => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                                      :g_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
                        :sh_q => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                                      :b_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
                        :zp         => Dict(:pb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
                        :zq         => Dict(:qb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
                        # energy costs
                        :zen_dev    => Dict(:dev_p               => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
                        # zon_dev, zsu_dev, zsd_dev (also for lines and xfms)
                        # ...
                        # these next two dictionaries take the derivatives of the su/sd states with respect to
                        # both the current on status, and the previous on status
                        :u_su_dev => Dict(:u_on_dev            => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                                          :u_on_dev_prev       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
                        :u_sd_dev => Dict(:u_on_dev            => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                                          :u_on_dev_prev       => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
                        :u_su_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
                        :u_sd_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
                        :u_su_xfm => Dict(:u_on_xfm            => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                          :u_on_xfm_prev       => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
                        :u_sd_xfm => Dict(:u_on_xfm            => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                                          :u_on_xfm_prev       => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
                        # these two following elements are unique -- they serve to collect all of the 
                        # coefficients applied to the same partial derivatives (e.g.,
                        # a1*dxdp, a2*dxdp, a3*dxdp => dxdp[tii][dev] = a1+a2+a3)
                        :dx => Dict(:dp => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                                    :dq => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))))

# %%
ctg_avg = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.s_flow/sys.nctg for ii in 1:(sys.nT))
ctg_min = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.s_flow          for ii in 1:(sys.nT))

# device on costs
dzon_dev_du_on_dev = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.dev.on_cost for ii in 1:(sys.nT))

# device reserve gradients
dzrgu_dp_rgu     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_reg_res_up_cost,ii)            for ii in 1:(sys.nT))
dzrgd_dp_rgd     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_reg_res_down_cost,ii)          for ii in 1:(sys.nT))
dzscr_dp_scr     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_syn_res_cost,ii)               for ii in 1:(sys.nT))
dznsc_dp_nsc     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_nsyn_res_cost,ii)              for ii in 1:(sys.nT))
dzrru_dp_rru_on  = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_up_online_cost,ii)    for ii in 1:(sys.nT))
dzrru_dp_rru_off = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_up_offline_cost,ii)   for ii in 1:(sys.nT))
dzrrd_dp_rrd_on  = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_down_online_cost,ii)  for ii in 1:(sys.nT))
dzrrd_dp_rrd_off = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.p_ramp_res_down_offline_cost,ii) for ii in 1:(sys.nT))
dzqru_dq_qru     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.q_res_up_cost,ii)                for ii in 1:(sys.nT))
dzqrd_dq_qrd     = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*getindex.(prm.dev.q_res_down_cost,ii)              for ii in 1:(sys.nT))

# zonal gradients
dzrgu_zonal_dp_rgu_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rgu_zonal  for ii in 1:(sys.nT))
dzrgd_zonal_dp_rgd_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rgd_zonal  for ii in 1:(sys.nT))
dzscr_zonal_dp_scr_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.scr_zonal  for ii in 1:(sys.nT))
dznsc_zonal_dp_nsc_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.nsc_zonal  for ii in 1:(sys.nT))
dzrru_zonal_dp_rru_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rru_zonal  for ii in 1:(sys.nT))
dzrrd_zonal_dp_rrd_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.rrd_zonal  for ii in 1:(sys.nT))
dzqru_zonal_dq_qru_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.qru_zonal  for ii in 1:(sys.nT))
dzqrd_zonal_dq_qrd_zonal_penalty = Dict(tkeys[ii] => prm.ts.duration[tkeys[ii]]*prm.vio.qrd_zonal  for ii in 1:(sys.nT))

# %% ===============================
    # grd = grad
    grd = Dict(
        # aclines
        :acline_pfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_qfr => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_pto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_qto => Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_sfr_plus => Dict(:acline_pfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)), 
                                 :acline_qfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :acline_sto_plus => Dict(:acline_pto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)), 
                                 :acline_qto => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :zs_acline => Dict(:acline_sfr_plus => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                           :acline_sto_plus => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        # xfms
        :xfm_pfr =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_qfr =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_pto =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_qto =>  Dict(:vmfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vmto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vafr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :vato => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :phi  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :tau  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                            :uon  => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_sfr_plus => Dict(:xfm_pfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
                              :xfm_qfr => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :xfm_sto_plus => Dict(:xfm_pto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)), 
                              :xfm_qto => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))), 
        :zs_xfm => Dict(:xfm_sfr_plus => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                        :xfm_sto_plus => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        # shunts
        :sh_p => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                      :g_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
        :sh_q => Dict(:vm         => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT)),
                      :b_tv_shunt => Dict(tkeys[ii] => zeros(sys.nsh) for ii in 1:(sys.nT))),
        :g_tv_shunt => Dict(:u_step_shunt => zeros(sys.nsh)),
        :b_tv_shunt => Dict(:u_step_shunt => zeros(sys.nsh)),
        :zp         => Dict(:pb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
        :zq         => Dict(:qb_slack => Dict(tkeys[ii] => zeros(sys.nb) for ii in 1:(sys.nT))),
        # devices
        :zrgu       => Dict(:p_rgu => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zrgd       => Dict(:p_rgd => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zscr       => Dict(:p_scr => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :znsc       => Dict(:p_nsc => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zrru       => Dict(:p_rru_on => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                            :p_rru_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zrrd       => Dict(:p_rrd_on => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                            :p_rrd_off => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zqru       => Dict(:q_qru => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zqrd       => Dict(:q_qrd => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        # zones
        :zrgu_zonal => Dict(:p_rgu_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT))),
        :zrgd_zonal => Dict(:p_rgd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT))),
        :zscr_zonal => Dict(:p_scr_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT))),
        :znsc_zonal => Dict(:p_nsc_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT))),
        :zrru_zonal => Dict(:p_rru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT))),
        :zrrd_zonal => Dict(:p_rrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzP) for ii in 1:(sys.nT))),
        :zqru_zonal => Dict(:q_qru_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT))),
        :zqrd_zonal => Dict(:q_qrd_zonal_penalty => Dict(tkeys[ii] => zeros(sys.nzQ) for ii in 1:(sys.nT))),
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
        # scores -- NOTE: many of these are just common gradient terms
        #                 and do not represent the full vectro of gradients.
        #                 For example, grd[:zbase][:zt] is a vector in reality,
        #                 but every value in that vector is 1, so we jsut use a
        #                 common gradient term.
        :nzms => Dict(:zbase    => 0.0,
                      :zctg_min => 0.0,
                      :zctg_avg => 0.0),
        :zbase => Dict(:zt        => 0.0,
                       :z_enmax   => 0.0,
                       :z_enmin   => 0.0,
                       :zhat_mxst => 0.0),
        :zt    => Dict( :zen_dev => zeros(sys.ndev),
                        :zsu_dev => 0.0,
                        :zsu_acline => 0.0,
                        :zsu_xfm => 0.0,
                        :zsd_dev => 0.0,
                        :zsd_acline => 0.0,
                        :zsd_xfm   => 0.0,
                        :zon_dev   => 0.0,
                        :zsus_dev  => 0.0,
                        :zs_acline => 0.0,
                        :zs_xfm    => 0.0,
                        :zrgu => 0.0,
                        :zrgd => 0.0,
                        :zscr => 0.0,
                        :znsc => 0.0,
                        :zrru => 0.0,
                        :zrrd => 0.0,
                        :zqru => 0.0,
                        :zqrd => 0.0,
                        :zp   => 0.0,
                        :zq   => 0.0,
                        :zrgu_zonal => 0.0,
                        :zrgd_zonal => 0.0,
                        :zscr_zonal => 0.0,
                        :znsc_zonal => 0.0,
                        :zrru_zonal => 0.0,
                        :zrrd_zonal => 0.0,
                        :zqru_zonal => 0.0,
                        :zqrd_zonal => 0.0,
                        :zhat_mndn  => 0.0,
                        :zhat_mnup  => 0.0,
                        :zhat_rup => 0.0,
                        :zhat_rd  => 0.0,
                        :zhat_rgu => 0.0,
                        :zhat_rgd => 0.0,
                        :zhat_scr => 0.0,
                        :zhat_nsc => 0.0,
                        :zhat_rruon   => 0.0,
                        :zhat_rruoff  => 0.0,
                        :zhat_rrdon   => 0.0,
                        :zhat_rrdoff  => 0.0,
                        :zhat_pmax    => 0.0,
                        :zhat_pmin    => 0.0,    
                        :zhat_pmaxoff => 0.0,  
                        :zhat_qmax    => 0.0,     
                        :zhat_qmin    => 0.0,     
                        :zhat_qmax_beta => 0.0,
                        :zhat_qmin_beta => 0.0),
        :z_enmax => Dict(:zw_enmax => 0.0),
        :z_enmin => Dict(:zw_enmin => 0.0),
        :zen_dev => Dict(:dev_p => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :dev_p => Dict(:p_on => 0.0,
                       :p_su => 0.0,
                       :p_sd => 0.0),
        :dev_q => Dict(:u_sum => zeros(sys.ndev),
                       :dev_p => zeros(sys.ndev)),
        # zon_dev, zsu_dev, zsd_dev (also for lines and xfms)
        :zon_dev    => Dict(:u_on_dev    => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :zsu_dev    => Dict(:u_su_dev    => zeros(sys.ndev)),
        :zsu_acline => Dict(:u_su_acline => zeros(sys.nl)),
        :zsu_xfm    => Dict(:u_su_xfm    => zeros(sys.nx)),
        :zsd_dev    => Dict(:u_sd_dev    => zeros(sys.ndev)),
        :zsd_acline => Dict(:u_sd_acline => zeros(sys.nl)),
        :zsd_xfm    => Dict(:u_sd_xfm    => zeros(sys.nx)),
        # these next two dictionaries take the derivatives of the su/sd states with respect to
        # both the current on status, and the previous on status
        :u_su_dev => Dict(:u_on_dev      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                          :u_on_dev_prev => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :u_sd_dev => Dict(:u_on_dev      => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
                          :u_on_dev_prev => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT))),
        :u_su_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :u_sd_acline => Dict(:u_on_acline      => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT)),
                             :u_on_acline_prev => Dict(tkeys[ii] => zeros(sys.nl) for ii in 1:(sys.nT))),
        :u_su_xfm => Dict(:u_on_xfm      => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                          :u_on_xfm_prev => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :u_sd_xfm => Dict(:u_on_xfm      => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT)),
                          :u_on_xfm_prev => Dict(tkeys[ii] => zeros(sys.nx) for ii in 1:(sys.nT))),
        :ctg_avg => Dict(tkeys[ii] => 0.0 for ii in 1:(sys.nT)),
        :ctg_min => Dict(tkeys[ii] => 0.0 for ii in 1:(sys.nT)),
        # these two following elements are unique -- they serve to collect all of the 
        # coefficients applied to the same partial derivatives (e.g.,
        # a1*dxdp, a2*dxdp, a3*dxdp => dxdp[tii][dev] = a1+a2+a3)
        :dxdp => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)),
        :dxdq => Dict(tkeys[ii] => zeros(sys.ndev) for ii in 1:(sys.nT)))
        # %%

        grd[:nzms][:zbase]    = -1.0
        grd[:nzms][:zctg_min] = -1.0
        grd[:nzms][:zctg_avg] = -1.0
    
        # zbase: see score_zbase!()
        grd[:zbase][:zt]        = 1.0
        grd[:zbase][:z_enmax]   = 1.0
        grd[:zbase][:z_enmin]   = 1.0
        grd[:zbase][:zhat_mxst] = -qG.delta
    
        # zt: see score_zt!()
        #
        # consumer revenues and costs
        grd[:zt][:zen_dev][idx.cs_devs] = +ones(sys.ncs)
        grd[:zt][:zen_dev][idx.pr_devs] = -ones(sys.npr)
        # startup costs
        grd[:zt][:zsu_dev]    = -1.0
        grd[:zt][:zsu_acline] = -1.0
        grd[:zt][:zsu_xfm]    = -1.0
        # shutdown costs
        grd[:zt][:zsd_dev]    = -1.0
        grd[:zt][:zsd_acline] = -1.0
        grd[:zt][:zsd_xfm]    = -1.0
        # on-costs
        grd[:zt][:zon_dev]    = -1.0
        # time-dependent su costs
        grd[:zt][:zsus_dev]   = -1.0
        # ac line overload costs
        grd[:zt][:zs_acline]  = -1.0
        grd[:zt][:zs_xfm]     = -1.0
        # local reserve penalties (producers and consumers)
        grd[:zt][:zrgu] = -1.0
        grd[:zt][:zrgd] = -1.0
        grd[:zt][:zscr] = -1.0
        grd[:zt][:znsc] = -1.0
        grd[:zt][:zrru] = -1.0
        grd[:zt][:zrrd] = -1.0
        grd[:zt][:zqru] = -1.0
        grd[:zt][:zqrd] = -1.0
        # power mismatch penalties
        grd[:zt][:zp] = -1.0
        grd[:zt][:zq] = -1.0
        # zonal reserve penalties (P)
        grd[:zt][:zrgu_zonal] = -1.0
        grd[:zt][:zrgd_zonal] = -1.0
        grd[:zt][:zscr_zonal] = -1.0
        grd[:zt][:znsc_zonal] = -1.0
        grd[:zt][:zrru_zonal] = -1.0
        grd[:zt][:zrrd_zonal] = -1.0
        # zonal reserve penalties (Q)
        grd[:zt][:zqru_zonal] = -1.0
        grd[:zt][:zqrd_zonal] = -1.0
    
        grd[:zt][:zhat_mndn]   = -qG.delta
        grd[:zt][:zhat_mnup]   = -qG.delta
        grd[:zt][:zhat_rup]    = -qG.delta
        grd[:zt][:zhat_rd]     = -qG.delta
        grd[:zt][:zhat_rgu]    = -qG.delta
        grd[:zt][:zhat_rgd]    = -qG.delta
        grd[:zt][:zhat_scr]    = -qG.delta
        grd[:zt][:zhat_nsc]    = -qG.delta
        grd[:zt][:zhat_rruon]  = -qG.delta
        grd[:zt][:zhat_rruoff] = -qG.delta
        grd[:zt][:zhat_rrdon]  = -qG.delta
        grd[:zt][:zhat_rrdoff] = -qG.delta
        # common set of pr and cs constraint variables (see below)
        grd[:zt][:zhat_pmax]      = -qG.delta
        grd[:zt][:zhat_pmin]      = -qG.delta
        grd[:zt][:zhat_pmaxoff]   = -qG.delta
        grd[:zt][:zhat_qmax]      = -qG.delta
        grd[:zt][:zhat_qmin]      = -qG.delta
        grd[:zt][:zhat_qmax_beta] = -qG.delta
        grd[:zt][:zhat_qmin_beta] = -qG.delta

# %% =======
include("./src/core/structs.jl")
include("./src/core/initializations.jl")

ctg, flw, ntk = initialize_ctg(sys, prm, qG, idx)



include("../src/quasiGrad_dual.jl")
include("../src/core/structs.jl")
include("./test_functions.jl")

# ===============
using Pkg
Pkg.activate(".")

using JSON
using JuMP
using Plots
using Gurobi
using Statistics
using SparseArrays
using InvertedIndices

# call this first
include("../src/core/structs.jl")
include("../src/core/shunts.jl")
include("../src/core/devices.jl")
include("../src/io/read_data.jl")
include("../src/core/ac_flow.jl")
include("../src/core/scoring.jl")
include("../src/core/clipping.jl")
include("../src/io/write_data.jl")
include("../src/core/reserves.jl")
include("../src/core/opt_funcs.jl")
include("../src/scripts/solver.jl")
include("../src/core/master_grad.jl")
include("../src/core/contingencies.jl")
include("../src/core/power_balance.jl")
include("../src/core/initializations.jl")
include("../src/core/projection.jl")

# load things
    #data_dir  = "./test/data/c3/C3S0_20221208/D1/C3S0N00003/"
    #file_name = "scenario_003.json"
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
file_name = "scenario_002.json"

# read and parse the input data
jsn, prm, idx, sys = quasiGrad.load_and_parse_json(data_dir*file_name)
qG                 = quasiGrad.initialize_qG(prm)
qG.eval_grad       = true

# reset -- to help with numerical conditioning of the market surplus function 
# (so that we can take its derivative numerically)
qG.scale_c_pbus_testing  = 0.00001
qG.scale_c_qbus_testing  = 0.00001
qG.scale_c_sflow_testing = 0.02

# previous:
    #prm.vio.q_bus  = 11.0
    #prm.vio.p_bus  = 11.0
    #prm.vio.e_dev  = 12.0
    #prm.vio.s_flow = 13.0

# --------------------
cgd, GRB, grd, mgd, scr, stt = quasiGrad.initialize_states(idx, prm, sys)

# perturb stt
perturb!(stt, prm, idx, grd, sys, qG, 1.0)

# initialize static gradients
qG.delta = 50.0
quasiGrad.clip_all!(prm, qG, stt)

# initialize the states which adam will update -- the rest are fixed
adm = quasiGrad.initialize_adam_states(sys)
upd = quasiGrad.identify_update_states(prm, idx, stt, sys)
ctg, flw, ntk = quasiGrad.initialize_ctg(sys, prm, qG, idx)

# %%
p1         = quasiGrad.plot()
display(quasiGrad.plot!(p1, [adm_step-1, adm_step],[v_nms_prev, v_nms], label = label1, xlim = [0; N_its], ylim = [zmean/qG.plot_scale_dn; qG.plot_scale_up*zmean], linewidth=1.75, color = 1))

# %% ===================
bin_vec = [stt[:u_on_dev][tii] for tii in prm.ts.time_keys]

# %% ===================

for tii in prm.ts.time_keys
    bin_inds = (1:sys.ndev) .+ (prm.ts.time_key_ind[tii]-1)*sys.ndev
    println(tii)
    bin_vec[bin_inds] = stt[:u_on_dev][tii]
end

# %% ==================
pct_round   = 75.0
bin_vec_del = zeros(sys.nT*sys.ndev)

# loop and concatenate
for tii in prm.ts.time_keys
    bin_inds              = (1:sys.ndev) .+ (prm.ts.time_key_ind[tii]-1)*sys.ndev
    bin_vec_del[bin_inds] = stt[:u_on_dev][tii] - GRB[:u_on_dev][tii]
end

# sort and find the binaries that are closest to Gurobi's solution
most_sim_to_lest_sim = sortperm(abs.(bin_vec_del))

# which ones to we fix?
num_bin_fix = Int64(round(sys.nT*sys.ndev*pct_round/100.0))
bins_to_fix = most_sim_to_lest_sim[1:num_bin_fix]

# now, we loop over time and check for each binary in "bins_to_fix"
for tii in prm.ts.time_keys
    bin_inds          = (1:sys.ndev) .+ (prm.ts.time_key_ind[tii]-1)*sys.ndev
    local_bins_to_fix = findall(bin_inds .∈ [bins_to_fix])

    # reset, for safety
    upd[:u_on_dev][tii] = collect(1:sys.ndev)

    # now, for "bin_inds" which are to be fixed, delete them
    deleteat!(upd[:u_on_dev][tii],local_bins_to_fix)
end


# %%
v  = randn(100000)
v
#v2 = randn(100000)
#v3 = abs.(v2)
# %%
clamp!(v,-0.5,0.5)

#del = min.(stt[:vm][tii] - prm.bus.vm_lb  , 0.0) + max.(stt[:vm][tii] - prm.bus.vm_ub, 0.0)
#stt[:vm][tii] = stt[:vm][tii] - del

for (solver_iteration, pct_round) in enumerate(pcts_to_round)
    println(pct_round)
end






# %% load the json
data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
file_name = "scenario_002.json"

include("MyJulia1.jl")
InFile1               = data_dir*file_name
TimeLimitInSeconds    = 100.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

MyJulia1(InFile1, TimeLimitInSeconds, Division, NetworkModel, AllowSwitching)

# %%
# initialize: dev => time => set of time indices
# note: we use Int(sys.nT/2) because we don't want to initialize
#       the structure with more or less memory than needed.. super heuristic.
Ts_mndn     = [[Vector{Symbol}(undef, Int(sys.nT/2))                                                 for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
Ts_mnup     = [[Vector{Symbol}(undef, Int(sys.nT/2))                                                 for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
Ts_sdpc     = [[Vector{Symbol}(undef, Int(sys.nT/2))                                                 for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
ps_sdpc_set = [[Vector{Float64}(undef, Int(sys.nT/2))                                                for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
Ts_supc     = [[Vector{Symbol}(undef, Int(sys.nT/2))                                                 for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
ps_supc_set = [[Vector{Float64}(undef, Int(sys.nT/2))                                                for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
Ts_sus_jft  = [[[Vector{Symbol}(undef, Int(sys.nT/2)) for i in 1:prm.dev.num_sus[dev]]               for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
Ts_sus_jf   = [[[Vector{Symbol}(undef, Int(sys.nT/2)) for i in 1:prm.dev.num_sus[dev]]               for t in prm.ts.time_key_ind] for d in 1:sys.ndev]
Ts_en_max   = [[Vector{Symbol}(undef, Int(sys.nT/2))  for i in 1:length(prm.dev.energy_req_ub[dev])]                               for d in 1:sys.ndev]
Ts_en_min   = [[Vector{Symbol}(undef, Int(sys.nT/2))  for i in 1:length(prm.dev.energy_req_lb[dev])]                               for d in 1:sys.ndev]
Ts_su_max   = [[Vector{Symbol}(undef, Int(sys.nT/2))  for i in 1:length(prm.dev.startups_ub[dev])]                                 for d in 1:sys.ndev]

# loop over devices
for dev in 1:sys.ndev

    # loop over time
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # up and down times
        Ts_mndn[dev][t_ind] = quasiGrad.get_tmindn(tii, dev, prm)
        Ts_mnup[dev][t_ind] = quasiGrad.get_tminup(tii, dev, prm)

        # startup/down power curves
        Ts_sdpc[dev][t_ind], ps_sdpc_set[dev][t_ind] = quasiGrad.get_sdpc(tii, dev, prm)
        Ts_supc[dev][t_ind], ps_supc_set[dev][t_ind] = quasiGrad.get_supc(tii, dev, prm)

        # loop over sus (i.e., f in F)
        for ii in 1:prm.dev.num_sus[dev]
            Ts_sus_jft[dev][t_ind][ii], Ts_sus_jf[dev][t_ind][ii] = quasiGrad.get_tsus_sets(tii, dev, prm, stt, ii)
        end
    end

    # => Wub = prm.dev.energy_req_ub[dev]
    # => Wlb = prm.dev.energy_req_lb[dev]
    # max energy
    for (w_ind, w_params) in enumerate(prm.dev.energy_req_ub[dev])
        Ts_en_max[dev][w_ind] = quasiGrad.get_tenmax(w_params, prm)
    end

    # min energy
    for (w_ind, w_params) in enumerate(prm.dev.energy_req_lb[dev])
        Ts_en_min[dev][w_ind] = quasiGrad.get_tenmin(w_params, prm)
    end

    # max start ups
    for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
        Ts_su_max[dev][w_ind] = quasiGrad.get_tsumax(w_params, prm)
    end
end


# %%
# initialize: dev => time => set of time indices
# note: we use Int(sys.nT/2) because we don't want to initialize
#       the structure with more or less memory than needed.. super heuristic.
Ts_mndn     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
Ts_mnup     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
Ts_sdpc     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
ps_sdpc_set = [[Vector{Float64}(undef, Int64(round(sys.nT/2)))                                                for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
Ts_supc     = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))                                                 for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
ps_supc_set = [[Vector{Float64}(undef, Int64(round(sys.nT/2)))                                                for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
Ts_sus_jft  = [[[Vector{Symbol}(undef, Int64(round(sys.nT/2))) for i in 1:prm.dev.num_sus[dev]]               for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
Ts_sus_jf   = [[[Vector{Symbol}(undef, Int64(round(sys.nT/2))) for i in 1:prm.dev.num_sus[dev]]               for t in prm.ts.time_key_ind] for dev in 1:sys.ndev]
Ts_en_max   = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))  for i in 1:length(prm.dev.energy_req_ub[dev])]                               for dev in 1:sys.ndev]
Ts_en_min   = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))  for i in 1:length(prm.dev.energy_req_lb[dev])]                               for dev in 1:sys.ndev]
Ts_su_max   = [[Vector{Symbol}(undef, Int64(round(sys.nT/2)))  for i in 1:length(prm.dev.startups_ub[dev])]                                 for dev in 1:sys.ndev]

# loop over devices
for dev in 1:sys.ndev

    # loop over time
    for (t_ind, tii) in enumerate(prm.ts.time_keys)
        # up and down times
        Ts_mndn[dev][t_ind] = get_tmindn(tii, dev, prm)
        Ts_mnup[dev][t_ind] = get_tminup(tii, dev, prm)

        # startup/down power curves
        Ts_sdpc[dev][t_ind], ps_sdpc_set[dev][t_ind] = get_sdpc(tii, dev, prm)
        Ts_supc[dev][t_ind], ps_supc_set[dev][t_ind] = get_supc(tii, dev, prm)

        # loop over sus (i.e., f in F)
        for ii in 1:prm.dev.num_sus[dev]
            Ts_sus_jft[dev][t_ind][ii], Ts_sus_jf[dev][t_ind][ii] = get_tsus_sets(tii, dev, prm, ii)
        end
    end

    # => Wub = prm.dev.energy_req_ub[dev]
    # => Wlb = prm.dev.energy_req_lb[dev]
    # max energy
    for (w_ind, w_params) in enumerate(prm.dev.energy_req_ub[dev])
        Ts_en_max[dev][w_ind] = get_tenmax(w_params, prm)
    end

    # min energy
    for (w_ind, w_params) in enumerate(prm.dev.energy_req_lb[dev])
        Ts_en_min[dev][w_ind] = get_tenmin(w_params, prm)
    end

    # max start ups
    for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
        Ts_su_max[dev][w_ind] = get_tsumax(w_params, prm)
    end
end

# %%
A = quasiGrad.sprand(10, 10, 0.5);
A = A * A' + quasiGrad.spdiagm(0 => rand(10));
LLDL = lldl(A, memory = 5)

# %%
# call data
device_inds      = keys(json_data["network"]["simple_dispatchable_device"])
device           = json_data["network"]["simple_dispatchable_device"]
device_id        = [device[ind]["uid"] for ind in device_inds]
bus              = [device[ind]["bus"] for ind in device_inds]
device_type      = [device[ind]["device_type"] for ind in device_inds]
ndev             = length(device_inds)

# costs -- some of the time interval "ints" are set to "float"
startup_cost       = Float64.([device[ind]["startup_cost"] for ind in device_inds])
startup_states     = Vector{Vector{Float64}}.([device[ind]["startup_states"] for ind in device_inds])
shutdown_cost      = Float64.([device[ind]["shutdown_cost"] for ind in device_inds])
startups_ub        = Vector{Vector{Float64}}.([device[ind]["startups_ub"] for ind in device_inds])
energy_req_ub      = Vector{Vector{Float64}}.([device[ind]["energy_req_ub"] for ind in device_inds])
energy_req_lb      = Vector{Vector{Float64}}.([device[ind]["energy_req_lb"] for ind in device_inds])
on_cost            = Float64.([device[ind]["on_cost"] for ind in device_inds])
down_time_lb       = Float64.([device[ind]["down_time_lb"] for ind in device_inds])
in_service_time_lb = Float64.([device[ind]["in_service_time_lb"] for ind in device_inds])

# Reserve attributes
p_ramp_up_ub       = Float64.([device[ind]["p_ramp_up_ub"] for ind in device_inds])
p_ramp_down_ub     = Float64.([device[ind]["p_ramp_down_ub"] for ind in device_inds])
p_startup_ramp_ub  = Float64.([device[ind]["p_startup_ramp_ub"] for ind in device_inds])
p_shutdown_ramp_ub = Float64.([device[ind]["p_shutdown_ramp_ub"] for ind in device_inds])

# initialization
init_on_status      = Float64.([device[ind]["initial_status"]["on_status"] for ind in device_inds])
init_p              = Float64.([device[ind]["initial_status"]["p"] for ind in device_inds])
init_q              = Float64.([device[ind]["initial_status"]["q"] for ind in device_inds])
init_accu_down_time = Float64.([device[ind]["initial_status"]["accu_down_time"] for ind in device_inds])
init_accu_up_time   = Float64.([device[ind]["initial_status"]["accu_up_time"] for ind in device_inds])

# extra parameterts
q_linear_cap  = Bool.([device[ind]["q_linear_cap"] for ind in device_inds])
q_bound_cap   = Bool.([device[ind]["q_bound_cap"] for ind in device_inds])
iq_linear_cap = Int64.(q_linear_cap)
iq_bound_cap  = Int64.(q_bound_cap)

# conditional reactive devices
q_linear_cap_true  = findall(x -> x == true, q_linear_cap)
q_bound_cap_true   = findall(x -> x == true, q_bound_cap)

# initialize
q_0     = zeros(Float64,ndev)
beta    = zeros(Float64,ndev)
q_0_ub  = zeros(Float64,ndev)
q_0_lb  = zeros(Float64,ndev)
beta_ub = zeros(Float64,ndev)
beta_lb = zeros(Float64,ndev)

# populate
q_0[q_linear_cap_true]    = Float64.([device[ind]["q_0"]     for ind in q_linear_cap_true])
beta[q_linear_cap_true]   = Float64.([device[ind]["beta"]    for ind in q_linear_cap_true])
q_0_ub[q_bound_cap_true]  = Float64.([device[ind]["q_0_ub"]  for ind in q_bound_cap_true])
q_0_lb[q_bound_cap_true]  = Float64.([device[ind]["q_0_lb"]  for ind in q_bound_cap_true])
beta_ub[q_bound_cap_true] = Float64.([device[ind]["beta_ub"] for ind in q_bound_cap_true])
beta_lb[q_bound_cap_true] = Float64.([device[ind]["beta_lb"] for ind in q_bound_cap_true])

# reserve attributes
p_reg_res_up_ub            = Float64.([device[ind]["p_reg_res_up_ub"] for ind in device_inds]) 
p_reg_res_down_ub          = Float64.([device[ind]["p_reg_res_down_ub"] for ind in device_inds])
p_syn_res_ub               = Float64.([device[ind]["p_syn_res_ub"] for ind in device_inds])
p_nsyn_res_ub              = Float64.([device[ind]["p_nsyn_res_ub"] for ind in device_inds])
p_ramp_res_up_online_ub    = Float64.([device[ind]["p_ramp_res_up_online_ub"] for ind in device_inds])
p_ramp_res_down_online_ub  = Float64.([device[ind]["p_ramp_res_down_online_ub"] for ind in device_inds])
p_ramp_res_up_offline_ub   = Float64.([device[ind]["p_ramp_res_up_offline_ub"] for ind in device_inds])
p_ramp_res_down_offline_ub = Float64.([device[ind]["p_ramp_res_down_offline_ub"] for ind in device_inds])

# time series inputs -- device attributes
ts_device_inds = keys(json_data["time_series_input"]["simple_dispatchable_device"])
ts_device      = json_data["time_series_input"]["simple_dispatchable_device"]
ts_device_id   = [ts_device[ind]["uid"] for ind in ts_device_inds]

# now, for each device in "ts_device_uids", we need to get its index in "device_id"
ts_device_inds_adjusted = Int64.(indexin(ts_device_id, device_id))

on_status_ub = Vector{Float64}.([ts_device[ind]["on_status_ub"] for ind in ts_device_inds_adjusted])
on_status_lb = Vector{Float64}.([ts_device[ind]["on_status_lb"] for ind in ts_device_inds_adjusted])
p_ub         = Vector{Float64}.([ts_device[ind]["p_ub"] for ind in ts_device_inds_adjusted])
p_lb         = Vector{Float64}.([ts_device[ind]["p_lb"] for ind in ts_device_inds_adjusted])
q_ub         = Vector{Float64}.([ts_device[ind]["q_ub"] for ind in ts_device_inds_adjusted])
q_lb         = Vector{Float64}.([ts_device[ind]["q_lb"] for ind in ts_device_inds_adjusted])
cost         = Vector{Vector{Vector{Float64}}}.([ts_device[ind]["cost"] for ind in ts_device_inds_adjusted])

# time series inputs -- time series reserve attributes
p_reg_res_up_cost            = Vector{Float64}.([ts_device[ind]["p_reg_res_up_cost"] for ind in ts_device_inds_adjusted])
p_reg_res_down_cost          = Vector{Float64}.([ts_device[ind]["p_reg_res_down_cost"] for ind in ts_device_inds_adjusted])
p_syn_res_cost               = Vector{Float64}.([ts_device[ind]["p_syn_res_cost"] for ind in ts_device_inds_adjusted]) 
p_nsyn_res_cost              = Vector{Float64}.([ts_device[ind]["p_nsyn_res_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_up_online_cost    = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_online_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_down_online_cost  = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_online_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_up_offline_cost   = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_offline_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_down_offline_cost = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_offline_cost"] for ind in ts_device_inds_adjusted])
q_res_up_cost                = Vector{Float64}.([ts_device[ind]["q_res_up_cost"] for ind in ts_device_inds_adjusted])
q_res_down_cost              = Vector{Float64}.([ts_device[ind]["q_res_down_cost"] for ind in ts_device_inds_adjusted])

# get the number of startup states for each device
num_dev_sus = [length(dev_sus) for dev_sus in startup_states]

# get the number of minimum/maximum energy reqs states for each device
num_W_enmin = [length(dev_reqs) for dev_reqs in energy_req_lb]
num_W_enmax = [length(dev_reqs) for dev_reqs in energy_req_ub]

# get the number of maximum startup reqs for each device
num_mxst = [length(startups) for startups in startups_ub]

# update cost structures
#cum_cost_blocks = compute_cost_curves(cost,device_type)

# %%
json_data        = jsn

device_inds      = keys(json_data["network"]["simple_dispatchable_device"])
device           = json_data["network"]["simple_dispatchable_device"]
device_id        = [device[ind]["uid"] for ind in device_inds]
bus              = [device[ind]["bus"] for ind in device_inds]
device_type      = [device[ind]["device_type"] for ind in device_inds]
ndev             = length(device_inds)

# costs -- some of the time interval "ints" are set to "float"
startup_cost       = Float64.([device[ind]["startup_cost"] for ind in device_inds])
startup_states     = Vector{Vector{Float64}}.([device[ind]["startup_states"] for ind in device_inds])
shutdown_cost      = Float64.([device[ind]["shutdown_cost"] for ind in device_inds])
startups_ub        = Vector{Vector{Float64}}.([device[ind]["startups_ub"] for ind in device_inds])
energy_req_ub      = Vector{Vector{Float64}}.([device[ind]["energy_req_ub"] for ind in device_inds])
energy_req_lb      = Vector{Vector{Float64}}.([device[ind]["energy_req_lb"] for ind in device_inds])
on_cost            = Float64.([device[ind]["on_cost"] for ind in device_inds])
down_time_lb       = Float64.([device[ind]["down_time_lb"] for ind in device_inds])
in_service_time_lb = Float64.([device[ind]["in_service_time_lb"] for ind in device_inds])

# Reserve attributes
p_ramp_up_ub       = Float64.([device[ind]["p_ramp_up_ub"] for ind in device_inds])
p_ramp_down_ub     = Float64.([device[ind]["p_ramp_down_ub"] for ind in device_inds])
p_startup_ramp_ub  = Float64.([device[ind]["p_startup_ramp_ub"] for ind in device_inds])
p_shutdown_ramp_ub = Float64.([device[ind]["p_shutdown_ramp_ub"] for ind in device_inds])

# initialization
init_on_status      = Float64.([device[ind]["initial_status"]["on_status"] for ind in device_inds])
init_p              = Float64.([device[ind]["initial_status"]["p"] for ind in device_inds])
init_q              = Float64.([device[ind]["initial_status"]["q"] for ind in device_inds])
init_accu_down_time = Float64.([device[ind]["initial_status"]["accu_down_time"] for ind in device_inds])
init_accu_up_time   = Float64.([device[ind]["initial_status"]["accu_up_time"] for ind in device_inds])

# extra parameterts
q_linear_cap  = Bool.([device[ind]["q_linear_cap"] for ind in device_inds])
q_bound_cap   = Bool.([device[ind]["q_bound_cap"] for ind in device_inds])
iq_linear_cap = Int64.(q_linear_cap)
iq_bound_cap  = Int64.(q_bound_cap)

# conditional reactive devices
q_linear_cap_true  = findall(x -> x == true, q_linear_cap)
q_bound_cap_true   = findall(x -> x == true, q_bound_cap)

# initialize
q_0     = zeros(Float64,ndev)
beta    = zeros(Float64,ndev)
q_0_ub  = zeros(Float64,ndev)
q_0_lb  = zeros(Float64,ndev)
beta_ub = zeros(Float64,ndev)
beta_lb = zeros(Float64,ndev)

# populate
q_0[q_linear_cap_true]    = Float64.([device[ind]["q_0"]     for ind in q_linear_cap_true])
beta[q_linear_cap_true]   = Float64.([device[ind]["beta"]    for ind in q_linear_cap_true])
q_0_ub[q_bound_cap_true]  = Float64.([device[ind]["q_0_ub"]  for ind in q_bound_cap_true])
q_0_lb[q_bound_cap_true]  = Float64.([device[ind]["q_0_lb"]  for ind in q_bound_cap_true])
beta_ub[q_bound_cap_true] = Float64.([device[ind]["beta_ub"] for ind in q_bound_cap_true])
beta_lb[q_bound_cap_true] = Float64.([device[ind]["beta_lb"] for ind in q_bound_cap_true])

# reserve attributes
p_reg_res_up_ub            = Float64.([device[ind]["p_reg_res_up_ub"] for ind in device_inds]) 
p_reg_res_down_ub          = Float64.([device[ind]["p_reg_res_down_ub"] for ind in device_inds])
p_syn_res_ub               = Float64.([device[ind]["p_syn_res_ub"] for ind in device_inds])
p_nsyn_res_ub              = Float64.([device[ind]["p_nsyn_res_ub"] for ind in device_inds])
p_ramp_res_up_online_ub    = Float64.([device[ind]["p_ramp_res_up_online_ub"] for ind in device_inds])
p_ramp_res_down_online_ub  = Float64.([device[ind]["p_ramp_res_down_online_ub"] for ind in device_inds])
p_ramp_res_up_offline_ub   = Float64.([device[ind]["p_ramp_res_up_offline_ub"] for ind in device_inds])
p_ramp_res_down_offline_ub = Float64.([device[ind]["p_ramp_res_down_offline_ub"] for ind in device_inds])

# time series inputs -- device attributes
ts_device_inds = keys(json_data["time_series_input"]["simple_dispatchable_device"])
ts_device      = json_data["time_series_input"]["simple_dispatchable_device"]
ts_device_id   = [ts_device[ind]["uid"] for ind in ts_device_inds]

# now, for each device in "ts_device_uids", we need to get its index in "device_id"
ts_device_inds_adjusted = Int64.(indexin(ts_device_id, device_id))

on_status_ub = Vector{Float64}.([ts_device[ind]["on_status_ub"] for ind in ts_device_inds_adjusted])
on_status_lb = Vector{Float64}.([ts_device[ind]["on_status_lb"] for ind in ts_device_inds_adjusted])
p_ub         = Vector{Float64}.([ts_device[ind]["p_ub"] for ind in ts_device_inds_adjusted])
p_lb         = Vector{Float64}.([ts_device[ind]["p_lb"] for ind in ts_device_inds_adjusted])
q_ub         = Vector{Float64}.([ts_device[ind]["q_ub"] for ind in ts_device_inds_adjusted])
q_lb         = Vector{Float64}.([ts_device[ind]["q_lb"] for ind in ts_device_inds_adjusted])
cost         = Vector{Vector{Vector{Float64}}}.([ts_device[ind]["cost"] for ind in ts_device_inds_adjusted])

# time series inputs -- time series reserve attributes
p_reg_res_up_cost            = Vector{Float64}.([ts_device[ind]["p_reg_res_up_cost"] for ind in ts_device_inds_adjusted])
p_reg_res_down_cost          = Vector{Float64}.([ts_device[ind]["p_reg_res_down_cost"] for ind in ts_device_inds_adjusted])
p_syn_res_cost               = Vector{Float64}.([ts_device[ind]["p_syn_res_cost"] for ind in ts_device_inds_adjusted]) 
p_nsyn_res_cost              = Vector{Float64}.([ts_device[ind]["p_nsyn_res_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_up_online_cost    = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_online_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_down_online_cost  = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_online_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_up_offline_cost   = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_offline_cost"] for ind in ts_device_inds_adjusted])
p_ramp_res_down_offline_cost = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_offline_cost"] for ind in ts_device_inds_adjusted])
q_res_up_cost                = Vector{Float64}.([ts_device[ind]["q_res_up_cost"] for ind in ts_device_inds_adjusted])
q_res_down_cost              = Vector{Float64}.([ts_device[ind]["q_res_down_cost"] for ind in ts_device_inds_adjusted])

# get the number of startup states for each device
num_dev_sus = [length(dev_sus) for dev_sus in startup_states]

# get the number of minimum/maximum energy reqs states for each device
num_W_enmin = [length(dev_reqs) for dev_reqs in energy_req_lb]
num_W_enmax = [length(dev_reqs) for dev_reqs in energy_req_ub]

# get the number of maximum startup reqs for each device
num_mxst = [length(startups) for startups in startups_ub]

# %%
cost[127][42]

issorted(round.(getindex.(cost[dev][tii],1), digits=3))


# %%
t = randn(100000)
@btime any(isnan, t)
@btime sum(t) == isnan

# %%
d = 4

if d == 1
    dvn = "D1/"
elseif d == 2
    dvn = "D2/"
end

# %%
for bus in 1:sys.nb
    # active power balance
    stt[:p_inj][tii][bus] = 
    sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0) - 
    sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) - 
    sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) - 
    sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
    sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) - 
    ntk.alpha*p_slack
end
# %%
# also, we need to update the flows on all lines! and the phase shift
stt[:ac_qfr][tii][idx.ac_line_flows] = stt[:acline_qfr][tii]
stt[:ac_qfr][tii][idx.ac_xfm_flows]  = stt[:xfm_qfr][tii]
stt[:ac_qto][tii][idx.ac_line_flows] = stt[:acline_qto][tii]
stt[:ac_qto][tii][idx.ac_xfm_flows]  = stt[:xfm_qto][tii]
stt[:ac_phi][tii][idx.ac_phi]        = stt[:phi][tii]

# %%
p  =  stt[:p_inj][tii][2:end]
bt = -stt[:ac_phi][tii].*ntk.b

# now, we have p_inj = Yb*theta + E'*bt
c = p - ntk.Er'*bt

# solve!
if qG.base_solver == "lu"
    ctg[:theta_k][tii][end]  = ntk.Ybr\c
elseif qG.base_solver == "pcg"
    # solve with a hot start!
    #
    # note: ctg[:theta_k][tii][end] is modified in place,
    # and it represents the base case solution
    quasiGrad.cg!(ctg[:theta_k][tii][end], ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)
else
    println("base case solve type not recognized :)")
end

# %%

# store the base case solutions in the last entry of theta_k, pflow_k
#   theta   = ctg[:theta_k][tii][end]
#   pflow_k = ctg[:pflow_k][tii][end]
ctg[:pflow_k][tii][end] = ntk.Yfr*ctg[:theta_k][tii][end] + bt

# loop over contingencies
for ctg_ii in 1:sys.nctg
    # Here, we must solve theta_k = Ybr_k\c
    if qG.ctg_solver == "lu"
        ctg[:theta_k][tii][ctg_ii] = ntk.Ybr_k[ctg_ii]\c
    elseif qG.ctg_solver == "pcg"
        # two things to note: 1) we hot start with the base case solution => copy()
        #                     2) we precondition with the base case preconditioner
        ctg[:theta_k][tii][ctg_ii] = copy(ctg[:theta_k][tii][end])
        quasiGrad.cg!(ctg[:theta_k][tii][ctg_ii], ntk.Ybr_k[ctg_ii], c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)
    elseif qG.ctg_solver == "wmi"
        # now, we need to solve the following:
        # (Yb + v*b*v')x = c
        #
        # we already know x0 = Yb\c, so let's use it!
        #
        # wmi :)
        ctg[:theta_k][tii][ctg_ii] = ctg[:theta_k][tii][end] - ntk.u_k[ctg_ii]*(ntk.w_k[ctg_ii]'*c)
    else
        println("contingency solve type not recognized :)")
    end

    # compute flows
    #
    # NOTE: ctg[:pflow_k][tii][ctg_ii] contains the flow on the outaged line --
    #       -- this will be dealt with when computing the flows and gradients
    ctg[:pflow_k][tii][ctg_ii] = ntk.Yfr*ctg[:theta_k][tii][ctg_ii]  + bt
      # alternative: -> ctg[:pflow_k][tii][ctg_ii][ntk.ctg_out_ind[ctg_ii]] .= 0
end

# %% ======

tt = randn(2)
c = [-0.012128116082390526, 0.14487188391760947]
# quasiGrad.cg!(ctg[:theta_k][tii][end], ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)
quasiGrad.cg!(tt, ntk.Ybr, c, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)

# %%
    # loop over each device and solve individually -- not clear if this is faster
    # than solving one big optimization problem all at once. see legacy code for
    # a (n unfinished) version where all devices are solved at once!
    model = Model(Gurobi.Optimizer)
    
    # loop over all devices
    for dev in 1:sys.ndev
        
        # empty the model!
        empty!(model)

        # quiet down!!!
        quasiGrad.set_optimizer_attribute(model, "OutputFlag", qG.GRB_output_flag)

        # set model properties
        quasiGrad.set_optimizer_attribute(model, "FeasibilityTol", qG.FeasibilityTol)
        quasiGrad.set_optimizer_attribute(model, "MIPGap",         qG.mip_gap)
        quasiGrad.set_optimizer_attribute(model, "TimeLimit",      qG.time_lim)

        # define local time keys
        tkeys = prm.ts.time_keys

        # define the minimum set of variables we will need to solve the constraints                                                       -- round() the int?
        u_on_dev  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "u_on_dev_t$(ii)",  start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true)       for ii in 1:(sys.nT))
        p_on      = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_on_t$(ii)",      start=stt[:p_on][tkeys[ii]][dev])                         for ii in 1:(sys.nT))
        dev_q     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "dev_q_t$(ii)",     start=stt[:dev_q][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT))
        p_rgu     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_rgu_t$(ii)",     start=stt[:p_rgu][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT))
        p_rgd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_rgd_t$(ii)",     start=stt[:p_rgd][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT))
        p_scr     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_scr_t$(ii)",     start=stt[:p_scr][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT))
        p_nsc     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_nsc_t$(ii)",     start=stt[:p_nsc][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT))
        p_rru_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_rru_on_t$(ii)",  start=stt[:p_rru_on][tkeys[ii]][dev],  lower_bound = 0.0) for ii in 1:(sys.nT))
        p_rru_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_rru_off_t$(ii)", start=stt[:p_rru_off][tkeys[ii]][dev], lower_bound = 0.0) for ii in 1:(sys.nT))
        p_rrd_on  = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_rrd_on_t$(ii)",  start=stt[:p_rrd_on][tkeys[ii]][dev],  lower_bound = 0.0) for ii in 1:(sys.nT))
        p_rrd_off = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "p_rrd_off_t$(ii)", start=stt[:p_rrd_off][tkeys[ii]][dev], lower_bound = 0.0) for ii in 1:(sys.nT))
        q_qru     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "q_qru_t$(ii)",     start=stt[:q_qru][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT))
        q_qrd     = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "q_qrd_t$(ii)",     start=stt[:q_qrd][tkeys[ii]][dev],     lower_bound = 0.0) for ii in 1:(sys.nT))

        # add a few more (implicit) variables which are necessary for solving this system
        u_su_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "u_su_dev_t$(ii)", start=stt[:u_su_dev][tkeys[ii]][dev], binary=true) for ii in 1:(sys.nT))
        u_sd_dev = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "u_sd_dev_t$(ii)", start=stt[:u_sd_dev][tkeys[ii]][dev], binary=true) for ii in 1:(sys.nT))
        
        # we have the affine "AffExpr" expressions (whose values are specified)
        dev_p = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))
        p_su  = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))
        p_sd  = Dict(tkeys[ii] => AffExpr(0.0) for ii in 1:(sys.nT))

        # == define active power constraints ==
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # first, get the startup power
            T_supc     = idx.Ts_supc[dev][t_ind]     # T_set, p_supc_set = get_supc(tii, dev, prm)
            p_supc_set = idx.ps_supc_set[dev][t_ind] # T_set, p_supc_set = get_supc(tii, dev, prm)
            add_to_expression!(p_su[tii], sum(p_supc_set[ii]*u_su_dev[tii_inst] for (ii,tii_inst) in enumerate(T_supc); init=0.0))

            # second, get the shutdown power
            T_sdpc     = idx.Ts_sdpc[dev][t_ind]     # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
            p_sdpc_set = idx.ps_sdpc_set[dev][t_ind] # T_set, p_sdpc_set = get_sdpc(tii, dev, prm)
            add_to_expression!(p_sd[tii], sum(p_sdpc_set[ii]*u_sd_dev[tii_inst] for (ii,tii_inst) in enumerate(T_sdpc); init=0.0))

            # finally, get the total power balance
            dev_p[tii] = p_on[tii] + p_su[tii] + p_sd[tii]
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
                    u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)
                    
                    # compute q -- this might be the only equality constraint (and below)
                    @constraint(model, dev_q[tii] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii])
                else
                    # the device must be a consumer :)
                    T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                    T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                    u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                    # compute q -- this might be the only equality constraint (and above)
                    @constraint(model, dev_q[tii] == prm.dev.q_0[dev]*u_sum + prm.dev.beta[dev]*dev_p[tii])
                end
            end
        end

        # loop over each time period and define the hard constraints
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # duration
            dt = prm.ts.duration[tii]

            # 1. Minimum downtime: zhat_mndn
            T_mndn = idx.Ts_mndn[dev][t_ind] # t_set = get_tmindn(tii, dev, prm)
            @constraint(model, u_su_dev[tii] + sum(u_sd_dev[tii_inst] for tii_inst in T_mndn; init=0.0) - 1.0 <= 0)

            # 2. Minimum uptime: zhat_mnup
            T_mnup = idx.Ts_mnup[dev][t_ind] # t_set = get_tminup(tii, dev, prm)
            @constraint(model, u_sd_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_mnup; init=0.0) - 1.0 <= 0)

            # define the previous power value (used by both up and down ramping!)
            if tii == :t1
                # note: p0 = prm.dev.init_p[dev]
                dev_p_previous = prm.dev.init_p[dev]
            else
                # grab previous time
                tii_m1 = prm.ts.time_keys[t_ind-1]
                dev_p_previous = dev_p[tii_m1]
            end

            # 3. Ramping limits (up): zhat_rup
            @constraint(model, dev_p[tii] - dev_p_previous
                    - dt*(prm.dev.p_ramp_up_ub[dev]     *(u_on_dev[tii] - u_su_dev[tii])
                    +     prm.dev.p_startup_ramp_ub[dev]*(u_su_dev[tii] + 1.0 - u_on_dev[tii])) <= 0)

            # 4. Ramping limits (down): zhat_rd
            @constraint(model,  dev_p_previous - dev_p[tii]
                    - dt*(prm.dev.p_ramp_down_ub[dev]*u_on_dev[tii]
                    +     prm.dev.p_shutdown_ramp_ub[dev]*(1.0-u_on_dev[tii])) <= 0)

            # 5. Regulation up: zhat_rgu
            @constraint(model, p_rgu[tii] - prm.dev.p_reg_res_up_ub[dev]*u_on_dev[tii] <= 0)

            # 6. Regulation down: zhat_rgd
            @constraint(model, p_rgd[tii] - prm.dev.p_reg_res_down_ub[dev]*u_on_dev[tii] <= 0)

            # 7. Synchronized reserve: zhat_scr
            @constraint(model, p_rgu[tii] + p_scr[tii] - prm.dev.p_syn_res_ub[dev]*u_on_dev[tii] <= 0)

            # 8. Synchronized reserve: zhat_nsc
            @constraint(model, p_nsc[tii] - prm.dev.p_nsyn_res_ub[dev]*(1.0 - u_on_dev[tii]) <= 0)

            # 9. Ramping reserve up (on): zhat_rruon
            @constraint(model, p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ramp_res_up_online_ub[dev]*u_on_dev[tii] <= 0)

            # 10. Ramping reserve up (off): zhat_rruoff
            @constraint(model, p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ramp_res_up_offline_ub[dev]*(1.0-u_on_dev[tii]) <= 0)
            
            # 11. Ramping reserve down (on): zhat_rrdon
            @constraint(model, p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ramp_res_down_online_ub[dev]*u_on_dev[tii] <= 0)

            # 12. Ramping reserve down (off): zhat_rrdoff
            @constraint(model, p_rrd_off[tii] - prm.dev.p_ramp_res_down_offline_ub[dev]*(1-u_on_dev[tii]) <= 0)
            
            # Now, we must separate: producers vs consumers
            if dev in idx.pr_devs
                # 13p. Maximum reserve limits (producers): zhat_pmax
                @constraint(model, p_on[tii] + p_rgu[tii] + p_scr[tii] + p_rru_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0)
            
                # 14p. Minimum reserve limits (producers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rrd_on[tii] + p_rgd[tii] - p_on[tii] <= 0)
                
                # 15p. Off reserve limits (producers): zhat_pmaxoff
                @constraint(model, p_su[tii] + p_sd[tii] + p_nsc[tii] + p_rru_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0)

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                u_sum     = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                # 16p. Maximum reactive power reserves (producers): zhat_qmax
                @constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0)

                # 17p. Minimum reactive power reserves (producers): zhat_qmin
                @constraint(model, q_qrd[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0)

                # 18p. Linked maximum reactive power reserves (producers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[tii] + q_qru[tii] - prm.dev.q_0_ub[dev]*u_sum
                    - prm.dev.beta_ub[dev]*dev_p[tii] <= 0)
                end 
                
                # 19p. Linked minimum reactive power reserves (producers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                    + prm.dev.beta_lb[dev]*dev_p[tii]
                    + q_qrd[tii] - dev_q[tii] <= 0)
                end

            # consumers
            else  # => dev in idx.cs_devs
                # 13c. Maximum reserve limits (consumers): zhat_pmax
                @constraint(model, p_on[tii] + p_rgd[tii] + p_rrd_on[tii] - prm.dev.p_ub[dev][t_ind]*u_on_dev[tii] <= 0)

                # 14c. Minimum reserve limits (consumers): zhat_pmin
                @constraint(model, prm.dev.p_lb[dev][t_ind]*u_on_dev[tii] + p_rru_on[tii] + p_scr[tii] + p_rgu[tii] - p_on[tii] <= 0)
                
                # 15c. Off reserve limits (consumers): zhat_pmaxoff
                @constraint(model, p_su[tii] + p_sd[tii] + p_rrd_off[tii] - prm.dev.p_ub[dev][t_ind]*(1.0 - u_on_dev[tii]) <= 0)

                # get common "u_sum" terms that will be used in the subsequent four equations 
                T_supc = idx.Ts_supc[dev][t_ind] # T_supc, ~ = get_supc(tii, dev, prm) T_supc     = idx.Ts_supc[dev][t_ind] #T_supc, ~ = get_supc(tii, dev, prm)
                T_sdpc = idx.Ts_sdpc[dev][t_ind] # T_sdpc, ~ = get_sdpc(tii, dev, prm) T_sdpc, ~ = get_sdpc(tii, dev, prm)
                u_sum  = u_on_dev[tii] + sum(u_su_dev[tii_inst] for tii_inst in T_supc; init=0.0) + sum(u_sd_dev[tii_inst] for tii_inst in T_sdpc; init=0.0)

                # 16c. Maximum reactive power reserves (consumers): zhat_qmax
                @constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_ub[dev][t_ind]*u_sum <= 0)

                # 17c. Minimum reactive power reserves (consumers): zhat_qmin
                @constraint(model, q_qru[tii] + prm.dev.q_lb[dev][t_ind]*u_sum - dev_q[tii] <= 0)
                
                # 18c. Linked maximum reactive power reserves (consumers): zhat_qmax_beta
                if dev in idx.J_pqmax
                    @constraint(model, dev_q[tii] + q_qrd[tii] - prm.dev.q_0_ub[dev]*u_sum
                    - prm.dev.beta_ub[dev]*dev_p[tii] <= 0)
                end 

                # 19c. Linked minimum reactive power reserves (consumers): zhat_qmin_beta
                if dev in idx.J_pqmin
                    @constraint(model, prm.dev.q_0_lb[dev]*u_sum
                    + prm.dev.beta_lb[dev]*dev_p[tii]
                    + q_qru[tii] - dev_q[tii] <= 0)
                end
            end
        end

        # misc penalty: maximum starts over multiple periods
        for (w_ind, w_params) in enumerate(prm.dev.startups_ub[dev])
            # get the time periods: zhat_mxst
            T_su_max = idx.Ts_su_max[dev][w_ind] #get_tsumax(w_params, prm)
            @constraint(model, sum(u_su_dev[tii] for tii in T_su_max; init=0.0) - w_params[3] <= 0.0)
        end

        # now, we need to add two other sorts of constraints:
        # 1. "evolutionary" constraints which link startup and shutdown variables
        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            if tii == :t1
                @constraint(model, u_on_dev[tii] - prm.dev.init_on_status[dev] == u_su_dev[tii] - u_sd_dev[tii])
            else
                tii_m1 = prm.ts.time_keys[t_ind-1]
                @constraint(model, u_on_dev[tii] - u_on_dev[tii_m1] == u_su_dev[tii] - u_sd_dev[tii])
            end
            # only one can be nonzero
            @constraint(model, u_su_dev[tii] + u_sd_dev[tii] <= 1)
        end

        # 2. constraints which hold constant variables from moving
            # a. must run
            # b. planned outages
            # c. pre-defined fixed values (e.g., q_qru = 0 for devs in J_pqe)
            # d. other states which are fixed from previous IBR rounds
            #       note: all of these are relfected in "upd"
        # upd = update states
        #
        # note -- in this loop, we also build the objective function!
        # now, let's define an objective function and solve this mf.
        # our overall objective is to round and fix some subset of 
        # integer variables. Here is our approach: find a feasible
        # solution which is as close to our Adam solution as possible.
        # next, we process the results: we identify the x% of variables
        # which had to move "the least". We fix these values and remove
        # their associated indices from upd. the end.
        #
        # afterwards, we initialize adam with the closest feasible
        # solution variable values.
        obj = AffExpr(0.0)

        for (t_ind, tii) in enumerate(prm.ts.time_keys)
            # if a device is *not* in the set of variables,
            # then it must be held constant! -- otherwise, try to hold it
            # close to its initial value
            if dev ∉ upd[:u_on_dev][tii]
                @constraint(model, u_on_dev[tii] == stt[:u_on_dev][tii][dev])
            else
                # add it to the objective function
                tmp = @variable(model)
                @constraint(model, u_on_dev[tii]  - stt[:u_on_dev][tii][dev] <= tmp)
                @constraint(model, stt[:u_on_dev][tii][dev] - u_on_dev[tii]  <= tmp)
                add_to_expression!(obj, tmp, qG.binary_projection_weight)
            end

            if dev ∉ upd[:p_rrd_off][tii]
                @constraint(model, p_rrd_off[tii] == stt[:p_rrd_off][tii][dev])
            else
                # add it to the objective function
                tmp = @variable(model)
                @constraint(model, p_rrd_off[tii] - stt[:p_rrd_off][tii][dev] <= tmp)
                @constraint(model, stt[:p_rrd_off][tii][dev] - p_rrd_off[tii] <= tmp)
                add_to_expression!(obj, tmp)
            end

            if dev ∉ upd[:p_nsc][tii]
                @constraint(model, p_nsc[tii] == stt[:p_nsc][tii][dev])
            else
                # add it to the objective function
                tmp = @variable(model)
                @constraint(model, p_nsc[tii]  - stt[:p_nsc][tii][dev] <= tmp)
                @constraint(model, stt[:p_nsc][tii][dev] - p_nsc[tii] <= tmp)
                add_to_expression!(obj, tmp)
            end

            if dev ∉ upd[:p_rru_off][tii]
                @constraint(model, p_rru_off[tii] == stt[:p_rru_off][tii][dev])
            else
                # add it to the objective function
                tmp = @variable(model)
                @constraint(model, p_rru_off[tii]  - stt[:p_rru_off][tii][dev] <= tmp)
                @constraint(model, stt[:p_rru_off][tii][dev] - p_rru_off[tii]  <= tmp)
                add_to_expression!(obj, tmp)
            end

            if dev ∉ upd[:q_qru][tii]
                @constraint(model, q_qru[tii] == stt[:q_qru][tii][dev])
            else
                # add it to the objective function
                tmp = @variable(model)
                @constraint(model, q_qru[tii]  - stt[:q_qru][tii][dev] <= tmp)
                @constraint(model, stt[:q_qru][tii][dev] - q_qru[tii]  <= tmp)
                add_to_expression!(obj, tmp)
            end
            if dev ∉ upd[:q_qrd][tii]
                @constraint(model, q_qrd[tii] == stt[:q_qrd][tii][dev])
            else
                # add it to the objective function
                tmp = @variable(model)
                @constraint(model, q_qrd[tii]  - stt[:q_qrd][tii][dev] <= tmp)
                @constraint(model, stt[:q_qrd][tii][dev] - q_qrd[tii]  <= tmp)
                add_to_expression!(obj, tmp)
            end

            # now, deal with reactive powers, some of which are specified with equality
            # only a subset of devices will have a reactive power equality constraint
            if dev ∉ idx.J_pqe

                # add it to the objective function
                tmp = @variable(model)
                @constraint(model, dev_q[tii]  - stt[:dev_q][tii][dev] <= tmp)
                @constraint(model, stt[:dev_q][tii][dev] - dev_q[tii]  <= tmp)
                add_to_expression!(obj, tmp)
            end

            # and now the rest -- none of which are in fixed sets
            #
            # p_on
            tmp = @variable(model)
            @constraint(model, p_on[tii]  - stt[:p_on][tii][dev] <= tmp)
            @constraint(model, stt[:p_on][tii][dev] - p_on[tii]  <= tmp)
            add_to_expression!(obj, tmp)
            
            # p_rgu 
            tmp = @variable(model)
            @constraint(model, p_rgu[tii]  - stt[:p_rgu][tii][dev] <= tmp)
            @constraint(model, stt[:p_rgu][tii][dev] - p_rgu[tii]  <= tmp)
            add_to_expression!(obj, tmp)
            
            # p_rgd
            tmp = @variable(model)
            @constraint(model, p_rgd[tii]  - stt[:p_rgd][tii][dev] <= tmp)
            @constraint(model, stt[:p_rgd][tii][dev] - p_rgd[tii]  <= tmp)
            add_to_expression!(obj, tmp)

            # p_scr
            tmp = @variable(model)
            @constraint(model, p_scr[tii]  - stt[:p_scr][tii][dev] <= tmp)
            @constraint(model, stt[:p_scr][tii][dev] - p_scr[tii]  <= tmp)
            add_to_expression!(obj, tmp)

            # p_rru_on
            tmp = @variable(model)
            @constraint(model, p_rru_on[tii]  - stt[:p_rru_on][tii][dev] <= tmp)
            @constraint(model, stt[:p_rru_on][tii][dev] - p_rru_on[tii]  <= tmp)
            add_to_expression!(obj, tmp)

            # p_rrd_on
            tmp = @variable(model)
            @constraint(model, p_rrd_on[tii]  - stt[:p_rrd_on][tii][dev] <= tmp)
            @constraint(model, stt[:p_rrd_on][tii][dev] - p_rrd_on[tii]  <= tmp)
            add_to_expression!(obj, tmp)
        end

        # set the objective
        @objective(model, Min, obj)

        # solve
        optimize!(model)
        println("========================================================")
        println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
        println("========================================================")

        # solve, and then return the solution
        for tii in prm.ts.time_keys
            GRB[:u_on_dev][tii][dev]  = value(u_on_dev[tii])
            GRB[:p_on][tii][dev]      = value(p_on[tii])
            GRB[:dev_q][tii][dev]     = value(dev_q[tii])
            GRB[:p_rgu][tii][dev]     = value(p_rgu[tii])
            GRB[:p_rgd][tii][dev]     = value(p_rgd[tii])
            GRB[:p_scr][tii][dev]     = value(p_scr[tii])
            GRB[:p_nsc][tii][dev]     = value(p_nsc[tii])
            GRB[:p_rru_on][tii][dev]  = value(p_rru_on[tii])
            GRB[:p_rru_off][tii][dev] = value(p_rru_off[tii])
            GRB[:p_rrd_on][tii][dev]  = value(p_rrd_on[tii])
            GRB[:p_rrd_off][tii][dev] = value(p_rrd_off[tii])
            GRB[:q_qru][tii][dev]     = value(q_qru[tii])
            GRB[:q_qrd][tii][dev]     = value(q_qrd[tii])
        end
    end
# %%
json_data = jsn;

    # call data
    device_inds      = keys(json_data["network"]["simple_dispatchable_device"])
    device           = json_data["network"]["simple_dispatchable_device"]
    device_id        = [device[ind]["uid"] for ind in device_inds]
    bus              = [device[ind]["bus"] for ind in device_inds]
    device_type      = [device[ind]["device_type"] for ind in device_inds]
    ndev             = length(device_inds)

    # costs -- some of the time interval "ints" are set to "float"
    startup_cost       = Float64.([device[ind]["startup_cost"] for ind in device_inds])
    startup_states     = Vector{Vector{Float64}}.([device[ind]["startup_states"] for ind in device_inds])
    shutdown_cost      = Float64.([device[ind]["shutdown_cost"] for ind in device_inds])
    startups_ub        = Vector{Vector{Float64}}.([device[ind]["startups_ub"] for ind in device_inds])
    energy_req_ub      = Vector{Vector{Float64}}.([device[ind]["energy_req_ub"] for ind in device_inds])
    energy_req_lb      = Vector{Vector{Float64}}.([device[ind]["energy_req_lb"] for ind in device_inds])
    on_cost            = Float64.([device[ind]["on_cost"] for ind in device_inds])
    down_time_lb       = Float64.([device[ind]["down_time_lb"] for ind in device_inds])
    in_service_time_lb = Float64.([device[ind]["in_service_time_lb"] for ind in device_inds])

    # Reserve attributes
    p_ramp_up_ub       = Float64.([device[ind]["p_ramp_up_ub"] for ind in device_inds])
    p_ramp_down_ub     = Float64.([device[ind]["p_ramp_down_ub"] for ind in device_inds])
    p_startup_ramp_ub  = Float64.([device[ind]["p_startup_ramp_ub"] for ind in device_inds])
    p_shutdown_ramp_ub = Float64.([device[ind]["p_shutdown_ramp_ub"] for ind in device_inds])

    # initialization
    init_on_status      = Float64.([device[ind]["initial_status"]["on_status"] for ind in device_inds])
    init_p              = Float64.([device[ind]["initial_status"]["p"] for ind in device_inds])
    init_q              = Float64.([device[ind]["initial_status"]["q"] for ind in device_inds])
    init_accu_down_time = Float64.([device[ind]["initial_status"]["accu_down_time"] for ind in device_inds])
    init_accu_up_time   = Float64.([device[ind]["initial_status"]["accu_up_time"] for ind in device_inds])

    # extra parameterts
    q_linear_cap  = Bool.([device[ind]["q_linear_cap"] for ind in device_inds])
    q_bound_cap   = Bool.([device[ind]["q_bound_cap"] for ind in device_inds])
    iq_linear_cap = Int64.(q_linear_cap)
    iq_bound_cap  = Int64.(q_bound_cap)

    # conditional reactive devices
    q_linear_cap_true  = findall(x -> x == true, q_linear_cap)
    q_bound_cap_true   = findall(x -> x == true, q_bound_cap)

    # initialize
    q_0     = zeros(Float64,ndev)
    beta    = zeros(Float64,ndev)
    q_0_ub  = zeros(Float64,ndev)
    q_0_lb  = zeros(Float64,ndev)
    beta_ub = zeros(Float64,ndev)
    beta_lb = zeros(Float64,ndev)

    # populate
    q_0[q_linear_cap_true]    = Float64.([device[ind]["q_0"]     for ind in q_linear_cap_true])
    beta[q_linear_cap_true]   = Float64.([device[ind]["beta"]    for ind in q_linear_cap_true])
    q_0_ub[q_bound_cap_true]  = Float64.([device[ind]["q_0_ub"]  for ind in q_bound_cap_true])
    q_0_lb[q_bound_cap_true]  = Float64.([device[ind]["q_0_lb"]  for ind in q_bound_cap_true])
    beta_ub[q_bound_cap_true] = Float64.([device[ind]["beta_ub"] for ind in q_bound_cap_true])
    beta_lb[q_bound_cap_true] = Float64.([device[ind]["beta_lb"] for ind in q_bound_cap_true])

    # reserve attributes
    p_reg_res_up_ub            = Float64.([device[ind]["p_reg_res_up_ub"] for ind in device_inds]) 
    p_reg_res_down_ub          = Float64.([device[ind]["p_reg_res_down_ub"] for ind in device_inds])
    p_syn_res_ub               = Float64.([device[ind]["p_syn_res_ub"] for ind in device_inds])
    p_nsyn_res_ub              = Float64.([device[ind]["p_nsyn_res_ub"] for ind in device_inds])
    p_ramp_res_up_online_ub    = Float64.([device[ind]["p_ramp_res_up_online_ub"] for ind in device_inds])
    p_ramp_res_down_online_ub  = Float64.([device[ind]["p_ramp_res_down_online_ub"] for ind in device_inds])
    p_ramp_res_up_offline_ub   = Float64.([device[ind]["p_ramp_res_up_offline_ub"] for ind in device_inds])
    p_ramp_res_down_offline_ub = Float64.([device[ind]["p_ramp_res_down_offline_ub"] for ind in device_inds])

    # time series inputs -- device attributes
    ts_device_inds = keys(json_data["time_series_input"]["simple_dispatchable_device"])
    ts_device      = json_data["time_series_input"]["simple_dispatchable_device"]
    ts_device_id   = [ts_device[ind]["uid"] for ind in ts_device_inds]

    # now, for each device in "ts_device_uids", we need to get its index in "device_id"
    # wrong => ts_device_inds_adjusted = Int64.(indexin(ts_device_id, device_id))
    ts_device_inds_adjusted = Int64.(indexin(device_id, ts_device_id))

    on_status_ub = Vector{Float64}.([ts_device[ind]["on_status_ub"] for ind in ts_device_inds_adjusted])
    on_status_lb = Vector{Float64}.([ts_device[ind]["on_status_lb"] for ind in ts_device_inds_adjusted])
    p_ub         = Vector{Float64}.([ts_device[ind]["p_ub"]         for ind in ts_device_inds_adjusted])
    p_lb         = Vector{Float64}.([ts_device[ind]["p_lb"]         for ind in ts_device_inds_adjusted])
    q_ub         = Vector{Float64}.([ts_device[ind]["q_ub"]         for ind in ts_device_inds_adjusted])
    q_lb         = Vector{Float64}.([ts_device[ind]["q_lb"]         for ind in ts_device_inds_adjusted])
    cost         = Vector{Vector{Vector{Float64}}}.([ts_device[ind]["cost"] for ind in ts_device_inds_adjusted])

    # time series inputs -- time series reserve attributes
    p_reg_res_up_cost            = Vector{Float64}.([ts_device[ind]["p_reg_res_up_cost"] for ind in ts_device_inds_adjusted])
    p_reg_res_down_cost          = Vector{Float64}.([ts_device[ind]["p_reg_res_down_cost"] for ind in ts_device_inds_adjusted])
    p_syn_res_cost               = Vector{Float64}.([ts_device[ind]["p_syn_res_cost"] for ind in ts_device_inds_adjusted]) 
    p_nsyn_res_cost              = Vector{Float64}.([ts_device[ind]["p_nsyn_res_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_up_online_cost    = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_online_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_down_online_cost  = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_online_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_up_offline_cost   = Vector{Float64}.([ts_device[ind]["p_ramp_res_up_offline_cost"] for ind in ts_device_inds_adjusted])
    p_ramp_res_down_offline_cost = Vector{Float64}.([ts_device[ind]["p_ramp_res_down_offline_cost"] for ind in ts_device_inds_adjusted])
    q_res_up_cost                = Vector{Float64}.([ts_device[ind]["q_res_up_cost"] for ind in ts_device_inds_adjusted])
    q_res_down_cost              = Vector{Float64}.([ts_device[ind]["q_res_down_cost"] for ind in ts_device_inds_adjusted])

    # get the number of startup states for each device
    num_dev_sus = [length(dev_sus) for dev_sus in startup_states]

    # get the number of minimum/maximum energy reqs states for each device
    num_W_enmin = [length(dev_reqs) for dev_reqs in energy_req_lb]
    num_W_enmax = [length(dev_reqs) for dev_reqs in energy_req_ub]

    # get the number of maximum startup reqs for each device
    num_mxst = [length(startups) for startups in startups_ub]

    # update cost structures
    cum_cost_blocks = quasiGrad.compute_cost_curves(cost,device_type)
    
# %% setup outputs

    device_param = quasiGrad.Device(
        device_inds,
        device_id,
        bus,
        device_type,
        startup_cost,
        startup_states,
        num_dev_sus,
        shutdown_cost,
        startups_ub,
        num_mxst,
        energy_req_ub,
        energy_req_lb,
        num_W_enmax,
        num_W_enmin,
        on_cost,
        down_time_lb,
        in_service_time_lb,
        p_ramp_up_ub,
        p_ramp_down_ub,
        p_startup_ramp_ub,
        p_shutdown_ramp_ub,
        init_on_status, 
        init_p,
        init_q,
        init_accu_down_time,
        init_accu_up_time,
        iq_linear_cap,
        iq_bound_cap,
        q_0,
        beta,
        q_0_ub,
        q_0_lb,
        beta_ub,
        beta_lb,
        q_linear_cap_true,
        q_bound_cap_true,
        p_reg_res_up_ub,
        p_reg_res_down_ub,
        p_syn_res_ub,
        p_nsyn_res_ub,
        p_ramp_res_up_online_ub,
        p_ramp_res_down_online_ub,
        p_ramp_res_up_offline_ub,
        p_ramp_res_down_offline_ub,
        on_status_ub,
        on_status_lb,
        p_ub,
        p_lb,
        q_ub,
        q_lb,
        cost,
        cum_cost_blocks,
        p_reg_res_up_cost,
        p_reg_res_down_cost,
        p_syn_res_cost,
        p_nsyn_res_cost,
        p_ramp_res_up_online_cost,
        p_ramp_res_down_online_cost,
        p_ramp_res_up_offline_cost,
        p_ramp_res_down_offline_cost,
        q_res_up_cost,
        q_res_down_cost)







# %% ===========
# this script solves and scores 
#
# loop over each time period and compute the power injections
#
# This step is contingency invariant -- i.e., each ctg will use this information
for tii in prm.ts.time_keys
    # get the slack at this time
    p_slack = 
    sum(stt[:dev_p][tii][idx.pr_devs]) -
    sum(stt[:dev_p][tii][idx.cs_devs]) - 
    sum(stt[:sh_p][tii])

    # loop over each bus
    for bus in 1:sys.nb
        # active power balance
        stt[:p_inj][tii][bus] = 
        sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0) - 
        sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) - 
        sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) - 
        sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
        sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) - 
        ntk.alpha*p_slack
    end
    
    # also, we need to update the flows on all lines! and the phase shift
    stt[:ac_qfr][tii][idx.ac_line_flows] = stt[:acline_qfr][tii]
    stt[:ac_qfr][tii][idx.ac_xfm_flows]  = stt[:xfm_qfr][tii]
    stt[:ac_qto][tii][idx.ac_line_flows] = stt[:acline_qto][tii]
    stt[:ac_qto][tii][idx.ac_xfm_flows]  = stt[:xfm_qto][tii]
    stt[:ac_phi][tii][idx.ac_phi]        = stt[:phi][tii]
end

# %% solve across all contingencies
qG.pcg_tol = 1e-2

@time quasiGrad.lowrank_update_all_ctg_solutions!(ctg, ntk, prm, qG, stt, sys)

# %% now that we have solved the ctg, we need to score it and compute the flows
#
# reset 

scr[:zctg_min] = 0.0
scr[:zctg_avg] = 0.0
# loop over time
for tii in prm.ts.time_keys
    # duration
    dt = prm.ts.duration[tii]

    # compute the shared, common gradient terms (time dependent)
    # gc_avg = grd[:nzms][:zctg_avg] * grd[:zctg_avg][:zctg_avg_t] * grd[:zctg_avg_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
    #        = (-1)                  *                (1)          *       (1/sys.nctg)        *          (-1)       *   dt*prm.vio.s_flow
    #        = dt*prm.vio.s_flow/sys.nctg
    gc_avg   = cgd.ctg_avg[tii] * qG.scale_c_sflow_testing
    
    # gc_min = (-1)                  *                (1)          *            (1)          *          (-1)       *   dt*prm.vio.s_flow
    #        = grd[:nzms][:zctg_min] * grd[:zctg_min][:zctg_min_t] * grd[:zctg_min_t][:zctg] * grd[:zctg][:zctg_s] * grd[:zctg_s][:s_ctg][tii]
    gc_min   = cgd.ctg_min[tii] * qG.scale_c_sflow_testing

    # loop over ctgs
    for ctg_ii in 1:sys.nctg
        flw[:sfr][ctg_ii]                = sqrt.(stt[:ac_qfr][tii].^2 + ctg[:pflow_k][tii][ctg_ii].^2)
        flw[:sto][ctg_ii]                = sqrt.(stt[:ac_qto][tii].^2 + ctg[:pflow_k][tii][ctg_ii].^2)
        flw[:sfr_vio][ctg_ii]            = flw[:sfr][ctg_ii] - ntk.s_max
        flw[:sto_vio][ctg_ii]            = flw[:sto][ctg_ii] - ntk.s_max

        # make sure there are no penalties on lines that are out-aged!
        flw[:sfr_vio][ctg_ii][ntk.ctg_out_ind[ctg_ii]] .= 0
        flw[:sto_vio][ctg_ii][ntk.ctg_out_ind[ctg_ii]] .= 0
        smax_vio = max.(flw[:sfr_vio][ctg_ii], flw[:sto_vio][ctg_ii], 0.0)

        # compute the penalties: "stt[:zctg_s][tii][ctg_ii]" -- if want to keep
        zctg_s = dt*prm.vio.s_flow*smax_vio * qG.scale_c_sflow_testing

        # each contingency, at each time, gets a score:
        stt[:zctg][tii][ctg_ii] = -sum(zctg_s, init=0.0)
    end

    # now that we have scored the contingencies, let's only include the 
    # gradients associated with the worst X% of them (at this time!!!)
    wct = sortperm(stt[:zctg][tii])[ntk.ctg_to_score]
    wct = sortperm(stt[:zctg][tii])[1:500]
    
    # loop over ctgs
    for ctg_ii in wct
        # evaluate the gradients?
        if qG.eval_grad
            # What are the gradients? build indicators
            gamma_fr   = (flw[:sfr_vio][ctg_ii] .> 0) .&& (flw[:sfr_vio][ctg_ii] .> flw[:sto_vio][ctg_ii])
            gamma_to   = (flw[:sto_vio][ctg_ii] .> 0) .&& (flw[:sto_vio][ctg_ii] .> flw[:sfr_vio][ctg_ii])

            # build the grads
            dsmax_dqfr_flow           = zeros(sys.nac)
            dsmax_dqto_flow           = zeros(sys.nac)
            dsmax_dp_flow             = zeros(sys.nac)
            dsmax_dp_flow[gamma_fr]   = ctg[:pflow_k][tii][ctg_ii][gamma_fr]./flw[:sfr][ctg_ii][gamma_fr]
            dsmax_dp_flow[gamma_to]   = ctg[:pflow_k][tii][ctg_ii][gamma_to]./flw[:sto][ctg_ii][gamma_to]
            dsmax_dqfr_flow[gamma_fr] = stt[:ac_qfr][tii][gamma_fr]./flw[:sfr][ctg_ii][gamma_fr]
            dsmax_dqto_flow[gamma_to] = stt[:ac_qto][tii][gamma_to]./flw[:sto][ctg_ii][gamma_to]

            # is this the worst ctg of the lot? (most negative!)
            if ctg_ii == wct[1]
                gc = copy(gc_avg) + copy(gc_min)
            else
                gc = copy(gc_avg)
            end

            # first, deal with the reactive power flows -- these are functions
            # of line variables (v, theta, phi, tau, u_on)
            #
            # acline
            aclfr_inds  = findall(!iszero,gamma_fr[1:sys.nl])
            aclto_inds  = findall(!iszero,gamma_to[1:sys.nl])
            aclfr_alpha = gc*(dsmax_dqfr_flow[1:sys.nl][aclfr_inds])
            aclto_alpha = gc*(dsmax_dqto_flow[1:sys.nl][aclto_inds])
            quasiGrad.zctgs_grad_q_acline!(tii, idx, grd, mgd, aclfr_inds, aclto_inds, aclfr_alpha, aclto_alpha)
            # xfm
            xfr_inds  = findall(!iszero,gamma_fr[(sys.nl+1):sys.nac])
            xto_inds  = findall(!iszero,gamma_to[(sys.nl+1):sys.nac])
            xfr_alpha = gc*(dsmax_dqfr_flow[(sys.nl+1):sys.nac][xfr_inds])
            xto_alpha = gc*(dsmax_dqto_flow[(sys.nl+1):sys.nac][xto_inds])
            quasiGrad.zctgs_grad_q_xfm!(tii, idx, grd, mgd, xfr_inds, xto_inds, xfr_alpha, xto_alpha)

            # now, the fun one: active power injection + xfm phase shift!!
            alpha_p_flow_phi = gc*dsmax_dp_flow
            #
            # 
            # alpha_p_flow_phi is the derivative of znms with repsect to the
            # active power flow vector in a given contingency at a given time

            # get the derivative of znms wrt active power injection
            # NOTE: ntk.Yfr = Ybs*Er, so ntk.Yfr^T = Er^T*Ybs
            #   -> techincally, we need Yfr_k, where the admittance
            #      at the outaged line has been drive to 0, but we
            #      can safely use Yfr, since alpha_p_flow_phi["k"] = 0
            #      (this was enforced ~50 or so lines above)
            # NOTE #2 -- this does NOT include the reference bus!
            #            we skip this gradient :)
            rhs = ntk.Yfr'*alpha_p_flow_phi

            # time to solve for dz_dpinj -- two options here:
            #   1. solve with ntk.Ybr_k, but we didn't actually build this,
            #      and we didn't build its preconditioner either..
            #   2. solve with ntk.Ybr, and then use a rank 1 update! Let's do
            #      this instead :) we'll do this in-loop for each ctg at each time.
            quasiGrad.lowrank_update_single_ctg_gradient!(ctg, ctg_ii, ntk, qG, rhs, tii)
            
            # now, we have the gradient of znms wrt all nodal injections/xfm phase shifts!!!
            # except the slack bus... time to apply these gradients into 
            # the master grad at all buses except the slack bus.
            #
            # update the injection gradient to account for slack!
            #   alternative direct solution: 
            #       -> ctg[:dz_dpinj][tii][ctg_ii] = (quasiGrad.I-ones(sys.nb-1)*ones(sys.nb-1)'/(sys.nb))*(ntk.Ybr_k[ctg_ii]\(ntk.Yfr'*alpha_p_flow))
            ctg[:dz_dpinj][tii][ctg_ii] = ctg[:dz_dpinj][tii][ctg_ii] .- sum(ctg[:dz_dpinj][tii][ctg_ii])/sys.nb
            quasiGrad.zctgs_grad_pinj!(ctg[:dz_dpinj][tii][ctg_ii], grd, idx, mgd, ntk, prm, sys, tii)

        end
    end

    # across each contingency, we get the average, and we get the min
    # => stt[:zctg_min_t][tii] = minimum(stt[:zctg][tii])
    # => stt[:zctg_avg_t][tii] = sum(stt[:zctg][tii])/sys.nctg
    scr[:zctg_min] += minimum(stt[:zctg][tii])
    scr[:zctg_avg] += sum(stt[:zctg][tii])/sys.nctg
end


# %% ==========
using BenchmarkTool
# %% ==========
v  = randn(616)
rhs = randn(616)
ctg[:ctd][tii][ctg_ii] = copy(v)
qG.pcg_tol = 1e-3
@btime quasiGrad.cg!(ctg[:ctd][tii][ctg_ii], ntk.Ybr, rhs, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)

ctg[:ctd][tii][ctg_ii] = copy(v)
qG.pcg_tol = 1e-4
@btime quasiGrad.cg!(ctg[:ctd][tii][ctg_ii], ntk.Ybr, rhs, abstol = qG.pcg_tol, Pl=ntk.Ybr_ChPr)

ctg[:ctd][tii][ctg_ii] = copy(v)
qG.pcg_tol = 1e-4
@btime quasiGrad.cg!(ctg[:ctd][tii][ctg_ii], ntk.Ybr, rhs, abstol = qG.pcg_tol)

@btime ntk.Ybr\rhs
# %%
@btime ctg[:theta_k][tii][end] - ntk.u_k[ctg_ii]*(ntk.w_k[ctg_ii]'*rhs)
@btime ctg[:theta_k][tii][ctg_ii] = ctg[:theta_k][tii][end] - ntk.u_k[ctg_ii]*(ntk.w_k[ctg_ii]'*rhs)

# %%
yy = ctg[:theta_k][tii][end] - ntk.u_k[ctg_ii]*(ntk.w_k[ctg_ii]'*rhs)

@btime ctg[:theta_k][tii][end] = yy
@btime ctg[:theta_k][tii][end] = yy2

# %%
v1 = Vector(ntk.u_k[ctg_ii])
v2 = Vector(ntk.w_k[ctg_ii])
v3 = Vector(rhs)

@btime ctg[:theta_k][tii][end] - ntk.u_k[ctg_ii]*(ntk.w_k[ctg_ii]'*rhs)
@btime ctg[:theta_k][tii][ctg_ii] = ctg[:theta_k][tii][end] - ntk.u_k[ctg_ii]*(ntk.w_k[ctg_ii]'*rhs)
@btime ctg[:theta_k][tii][ctg_ii] = ctg[:theta_k][tii][end] - v1*(v2'*v3)

# %%
@btime ctg[:theta_k][tii][ctg_ii] = ones(616)

# %%
quasiGrad.dot(y,x);

@btime quasiGrad.dot(y,x)
@btime y'*x

# %%
v_k = ntk.Ybr\ntk.Er[1,:]

# %% =========
for (t_ind, tii) in enumerate(prm.ts.time_keys)
    for ctg_ii = 1:563
        ctg[:ctd][tii][ctg_ii] = zeros(616)
        ctg[:theta_k][tii][ctg_ii]       = zeros(616)
    end
end

# %%
# this script solves and scores 
#
# loop over each time period and compute the power injections
#
# This step is contingency invariant -- i.e., each ctg will use this information
for tii in prm.ts.time_keys
    # get the slack at this time
    p_slack = 
    sum(stt[:dev_p][tii][idx.pr_devs]) -
    sum(stt[:dev_p][tii][idx.cs_devs]) - 
    sum(stt[:sh_p][tii])

    # loop over each bus
    for bus in 1:sys.nb
        # active power balance
        stt[:p_inj][tii][bus] = 
        sum(stt[:dev_p][tii][idx.pr[bus]]; init=0.0) - 
        sum(stt[:dev_p][tii][idx.cs[bus]]; init=0.0) - 
        sum(stt[:sh_p][tii][idx.sh[bus]]; init=0.0) - 
        sum(stt[:dc_pfr][tii][idx.bus_is_dc_frs[bus]]; init=0.0) - 
        sum(stt[:dc_pto][tii][idx.bus_is_dc_tos[bus]]; init=0.0) - 
        ntk.alpha*p_slack
    end
    
    # also, we need to update the flows on all lines! and the phase shift
    stt[:ac_qfr][tii][idx.ac_line_flows] = stt[:acline_qfr][tii]
    stt[:ac_qfr][tii][idx.ac_xfm_flows]  = stt[:xfm_qfr][tii]
    stt[:ac_qto][tii][idx.ac_line_flows] = stt[:acline_qto][tii]
    stt[:ac_qto][tii][idx.ac_xfm_flows]  = stt[:xfm_qto][tii]
    stt[:ac_phi][tii][idx.ac_phi]        = stt[:phi][tii]
end

quasiGrad.lowrank_update_all_ctg_solutions!(ctg, ntk, prm, qG, stt, sys)

# %% ------------
p   = -0.0
del = p .- prm.dev.cum_cost_blocks[1][1][3]
argmin(del[del .>= 0.0])

# %%
dev = findall(x -> x .== "Gen Bus 509 #1", prm.dev.id)[1]

# %%
T_mnup                    = idx.Ts_mnup[dev][t_ind]
cvio                      = max(stt[:u_sd_dev][tii][dev] + sum(stt[:u_su_dev][tii_inst][dev] for tii_inst in T_mnup; init=0.0) - 1.0 , 0.0)
stt[:zhat_mnup][tii][dev] = dt*cvio^2

# %%

mr = quasiGrad.get_tmr(dev, prm)

# %%
for tii in prm.ts.time_keys
    println(stt[:u_on_dev][tii][dev])
end
# %%
for dev in reverse(1:sys.ndev)
    println(dev)
end

# %% test!
using BenchmarkTools

# %%
tt1 = randn(10000)
tt2 = randn(10000)
tt3 = randn(10000)
@btime tt1 .= 0
@btime tt2 = zeros(10000)
@btime tt3 = 0*tt3

# %%

tt1 = randn(5000,5000)
tt2 = [randn(5000) for _ in 1:5000]
z   = zeros(5000)

@btime tt1[:,1] = z
@btime tt2[1]   = z

update_states_and_grads!(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, 
                          ac_phi, ac_qfr, ac_qto, dsmax_dp_flow, dsmax_dqfr_flow,
                          dsmax_dqto_flow, ctd, p_inj, ctb, wct)

update_states_and_grads!(cgd, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, 
                          ac_phi, ac_qfr, ac_qto, dsmax_dp_flow, dsmax_dqfr_flow,
                          dsmax_dqto_flow, ctd, p_inj, ctb, wct)

# %%
#
# define local time keys
tkeys = prm.ts.time_keys
model = Model(Gurobi.Optimizer)
dev = 1

vars_new = Dict(dev => Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "u_on_dev_t$(ii)",  start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true)  for ii in 1:(sys.nT)) for dev in 1:sys.ndev)

#vars = Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "u_on_dev_t$(ii)",  start=stt[:u_on_dev][tkeys[ii]][dev],  binary=true)  for ii in 1:(sys.nT))

# %%
u_on_dev1  = Dict(tkeys[ii] => Dict{Symbol, quasiGrad.JuMP.VariableRef}(tkeys[ii] => @variable(model, base_name = "u_on_dev_t$(ii)_d$(dev)",  start=stt[:u_on_dev][tkeys[ii]][dev],  lower_bound = 0.0, upper_bound = 1.0) for dev in 1:sys.ndev) for ii in 1:(sys.nT))

# %%

u_on_dev4  = Dict{Symbol, Vector{quasiGrad.JuMP.VariableRef}}(tkeys[ii] => @variable(model, base_name = "u_on_dev_t$(ii)", [dev = 1:sys.ndev], start=stt[:u_on_dev][tkeys[ii]][dev],  lower_bound = 0.0, upper_bound = 1.0) for ii in 1:(sys.nT))

# %% 
t = (1
+ 1)

# %%
using BenchmarkTools
# %%

@btime zeros(100000)                   
@btime Vector{Float64}(undef,100000)

# %%
@time y = randn(100) + randn(100);
@time y = randn(100) .+ sum(t[Int64[]]);

# %%

@btime mgd[:vm][:t1] .= 0.0
@btime mgd[:vm][:t1] .= zeros(sys.nb);
# %%

@btime quasiGrad.flush_gradients!(grd, mgd, prm, sys);
# %%

# %%
using BenchmarkTools
t = randn(100000)
f(x) = for ii in eachindex(x); x[ii] = 0.0; end

# %%

@btime t .= 0.0;

t = randn(100000)

@btime f(t);

# %%
# 1:1 mapping, from device number, to its bus
device_to_bus = zeros(Int64, sys.ndev)

for bus = 1:sys.nb
    # get the devices tied to this bus
    bus_id             = prm.bus.id[bus]
    dev_on_bus_inds    = findall(x -> x == bus_id, prm.dev.bus)

    # broadcast
    device_to_bus[dev_on_bus_inds] .= bus
end

# %%
tii = :t1

@time stt[:p_su][tii]
@time @view stt[:p_su][tii]

# %%
@btime worst_ctgs1 = 1:10000
@btime worst_ctgs2 = collect(1:10000)

t = randn(100000)
@btime t[worst_ctgs1];
@btime t[worst_ctgs2];

# %%
using StaticArrays

a  = rand(10000)
b  = rand(10000)
sa = SizedVector{10000}(a)
sb = SizedVector{10000}(b)

f(x,y) = quasiGrad.dot(x,y)

@btime f(sa,a);
@btime f(a,b);










# %%
quasiGrad.update_states_and_grads!(bit, cgd, ctb, ctd, flw, grd, idx, mgd, msc, ntk, prm, qG, scr, stt, sys, wct)

# %%

t_ind = 1
tii   = prm.ts.time_keys[t_ind]

residual = zeros(2*sys.nb)
quasiGrad.power_flow_residual!(idx, residual, stt, sys, tii)

# %% ========= %%%%%%%%%%%%%%%%%
model = Model(Gurobi.Optimizer)

t_ind = 1
tii = prm.ts.time_keys[t_ind]

# initialize
run_pf  = true
pf_cnt  = 0

# 1. update the ideal dispatch point (active power) -- we do this just once
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
    set_silent(model)

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
    @variable(model, dev_p_vars[1:sys.ndev])
    @variable(model, dev_q_vars[1:sys.ndev])

    set_start_value.(dev_p_vars, stt[:dev_p][tii])
    set_start_value.(dev_q_vars, stt[:dev_q][tii])

    # call the bounds
    dev_plb = stt[:u_on_dev][tii].*prm.dev.p_lb_tmdv[t_ind]
    dev_pub = stt[:u_on_dev][tii].*prm.dev.p_ub_tmdv[t_ind]
    dev_qlb = stt[:u_sum][tii].*prm.dev.q_lb_tmdv[t_ind]
    dev_qub = stt[:u_sum][tii].*prm.dev.q_ub_tmdv[t_ind]

    # first, define p_on at this time
    # => p_on = dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii]

    # bound
    @constraint(model, dev_plb + stt[:p_su][tii] + stt[:p_sd][tii] .<= dev_p_vars .<= dev_pub + stt[:p_su][tii] + stt[:p_sd][tii])
    # alternative: => @constraint(model, dev_plb .<= dev_p_vars - stt[:p_su][tii] - stt[:p_sd][tii] .<= dev_pub)
    @constraint(model, dev_qlb .<= dev_q_vars .<= dev_qub)


    # apply additional bounds: J_pqe (equality constraints)
    if ~isempty(idx.J_pqe)
        @constraint(model, dev_q_vars[idx.J_pqe] - prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt[:u_sum][tii][idx.J_pqe])
        # alternative: @constraint(model, dev_q_vars[idx.J_pqe] .== prm.dev.q_0[idx.J_pqe]*stt[:u_sum][tii][idx.J_pqe] + prm.dev.beta[idx.J_pqe]*dev_p_vars[idx.J_pqe])
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
    for dev in 1:sys.ndev
        if dev in idx.pr_devs
            # producers
            add_to_expression!(nodal_p[idx.device_to_bus[dev]], dev_p_vars[dev])
            add_to_expression!(nodal_q[idx.device_to_bus[dev]], dev_q_vars[dev])
        else
            # consumers
            add_to_expression!(nodal_p[idx.device_to_bus[dev]], -dev_p_vars[dev])
            add_to_expression!(nodal_q[idx.device_to_bus[dev]], -dev_q_vars[dev])
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

    # balance p and q
    #=
    add_to_expression!.(nodal_p, -msc[:pinj0])
    add_to_expression!.(nodal_q, -msc[:qinj0])
    # constrain
    @constraint(model, JacP_noref*x_in .== nodal_p)
    @constraint(model, JacQ_noref*x_in .== nodal_q)
    # remove these values!
    add_to_expression!.(nodal_p, msc[:pinj0])
    add_to_expression!.(nodal_q, msc[:qinj0])
    =#
    
    # alternative: => @constraint(model, JacP_noref*x_in + msc[:pinj0] .== nodal_p)
    # alternative: => @constraint(model, JacQ_noref*x_in + msc[:qinj0] .== nodal_q)
    @constraint(model, JacP_noref*x_in + msc[:pinj0] .== nodal_p)
    @constraint(model, JacQ_noref*x_in + msc[:qinj0] .== nodal_q)

    # objective: hold p (and v?) close to its initial value
    # => || msc[:pinj_ideal] - (p0 + dp) || + regularization
    if qG.Gurobi_pf_obj == "min_dispatch_distance"
        # this finds a solution close to the dispatch point -- does not converge without v,a regularization
        obj    = AffExpr(0.0)
        tmp_vm = @variable(model)
        tmp_va = @variable(model)
        for bus in 1:sys.nb
            tmp = @variable(model)
            # => @constraint(model, msc[:pinj_ideal][bus] - nodal_p[bus] <= tmp)
            # => @constraint(model, nodal_p[bus] - msc[:pinj_ideal][bus] <= tmp)
            #
            @constraint(model, msc[:pinj_ideal][bus] - nodal_p[bus] <= tmp)
            @constraint(model, nodal_p[bus] - msc[:pinj_ideal][bus] <= tmp)
            # slightly faster:
            #=
            add_to_expression!(nodal_p[bus], -msc[:pinj_ideal][bus])
            @constraint(model,  nodal_p[bus] <= tmp)
            @constraint(model, -nodal_p[bus] <= tmp)
            add_to_expression!(obj, tmp)
            =#

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
            @constraint(model, -JacP_noref[bus,:]*x_in <= tmp_p)
            @constraint(model,  JacP_noref[bus,:]*x_in <= tmp_p)

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
    max_dx = maximum(value.(x_in))
    
    # println("========================================================")
    println(termination_status(model),". ",primal_status(model),". objective value: ", round(objective_value(model), sigdigits = 5), ". max dx: ", round(max_dx, sigdigits = 5))
    # println("========================================================")

    # now, update the state vector with the soluion
    stt[:vm][tii]        = stt[:vm][tii]        + value.(dvm)
    stt[:va][tii][2:end] = stt[:va][tii][2:end] + value.(dva)

    # shall we terminate?
    if (maximum(value.(x_in)) < qG.max_pf_dx) || (pf_cnt == qG.max_linear_pfs)
        run_pf = false

        # now, apply the updated injections to the devices
        stt[:dev_p][tii]  = value.(dev_p_vars)
        stt[:p_on][tii]   = stt[:dev_p][tii] - stt[:p_su][tii] - stt[:p_sd][tii]
        stt[:dev_q][tii]  = value.(dev_q_vars)
        if sys.nldc > 0
            stt[:dc_pfr][tii] = value.(pdc_vars)
            stt[:dc_qfr][tii] = value.(qdc_fr_vars)
            stt[:dc_qto][tii] = value.(qdc_to_vars)
        end
    end
end