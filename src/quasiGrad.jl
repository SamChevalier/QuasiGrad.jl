module quasiGrad

# don't call Pkg, don't activate, and don't use Plots
    # using Pkg
    # Pkg.activate(".")
    # using Plots

using JSON
using JuMP
using Gurobi
using Random
using Statistics
using SparseArrays
using LinearAlgebra
using InvertedIndices
using Preconditioners
using IterativeSolvers
using LimitedLDLFactorizations
# don't call => using LimitedLDLFactorizations

# call this first
include("./core/structs.jl")

# and the rest :)
include("./core/shunts.jl")
include("./core/devices.jl")
include("./core/cleanup.jl")
include("./io/read_data.jl")
include("./core/ac_flow.jl")
include("./core/scoring.jl")
include("./core/clipping.jl")
include("./io/write_data.jl")
include("./core/reserves.jl")
include("./core/homotopy.jl")
include("./scripts/solver.jl")
include("./core/projection.jl")
include("./core/master_grad.jl")
include("./core/optimization.jl")
include("./core/contingencies.jl")
include("./core/power_balance.jl")
include("./core/initializations.jl")
include("./core/economic_dispatch.jl")

# use plotting?
# => using Makie
# => using GLMakie
# => include("./core/plotting.jl")

# define static data constants
const eps_int    = 1e-8::Float64
const eps_time   = 1e-6::Float64
const eps_constr = 1e-8::Float64
const eps_beta   = 1e-6::Float64
const eps_susd   = 1e-6::Float64
const d_unit     = 5e-3::Float64

# export?
export JuMP

end
