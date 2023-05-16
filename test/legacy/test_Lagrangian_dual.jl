# quasiGrad
include("../src/quasiGrad_dual.jl")

using JuMP
using Gurobi

# %% test the Lagrange dual formulation
nvars  = 250
ncons  = 200
A      = randn(ncons,nvars)
gamma  = randn(nvars)
b      = randn(ncons)

# %% min (x-gamma)'(x-gamma)
# st  Ax-b \le 0
model = Model(Gurobi.Optimizer)
empty!(model)
@variable(model, x[1:nvars])
@constraint(model, A*x .<= b)
@objective(model, Min, (x-gamma)'*(x-gamma))
optimize!(model)

plot(value.(x))
objective_value(model)

# %% Compare to the Lagrange dual
# max: -0.25*lam'*A*A'*lam + lam'(A*gamma -b), lam \ge 0

# loop, step, and project
function solve_ld(lambda, A, b, gamma, its, ncons)
    lambda = 0*ones(ncons)
    f      = 0
    lb     = []
    for ii in 1:its
        grad   = (-0.5*lambda'*A*(A'))' + (A*gamma - b)
        lambda = lambda + 0.5*grad
        # println(lambda)
        lambda = max.(lambda,0)
        #println(lambda[1:5])
        f      = -0.25*lambda'*A*A'*lambda + lambda'*(A*gamma - b)
        println(f)
        push!(lb,f)
        #println(lambda[1:5])
        #println("==========")
        sleep(0.15)
    end

    # output
    xv = gamma - 0.5*A'*lambda
    return xv, f, lb
end

# test
its    = 100
lambda = 0*ones(ncons)
xval, fval, lb = solve_ld(lambda, A, b, gamma, its, ncons)

# %% Compare bounds
plot(ones(its).*objective_value(model), ylims = (0,100))
plot!(lb)

# %% Compare x-vals
plot(value.(x))
plot!(xval)

