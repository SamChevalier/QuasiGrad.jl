using quasiGrad
using Plots
include("../src/scripts/solver.jl")

# =============
path = "../GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"

#path = "../GO3_testcases/C3S1_20221222/D1/C3S1N00600/scenario_001.json"
#path = "../GO3_testcases/C3S1_20221222/D2/C3S1N00600/scenario_001.json"

# %% ===============
InFile1               = path
NewTimeLimitInSeconds = 60000.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# run
compute_quasiGrad_solution(InFile1, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching)


# %%
y = 1000*randn(1000)

yt = sign.(y).*log10.(abs.(y).+1)

# %% plot(y)
values = [0, 5, 10, 15, 20]
labels = ["zero", "five", "ten", "fifteen", "twenty"]

lines(0..20, sin, axis = (xticks = (values, labels),))

plot(yt)
yticks!([-1:1:1;], ["10^{-1}", "zero", "max"])


# 1. update plots
# 2. economic dispatch :)
# 3. approximate power flow


# %%
sum(sum(stt[:zen_dev][tii][idx.cs_devs] for tii in prm.ts.time_keys))