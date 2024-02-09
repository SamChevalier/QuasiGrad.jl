# QuasiGrad.jl

QuasiGrad.jl is a parallelized, Adam-based solver for reserve and security constrained AC unit commitment problems. Solver details are sketeched out in a recent [pre-print](https://arxiv.org/pdf/2310.06650.pdf) and the associated [SI](https://samchevalier.github.io/docs/SI.pdf).

The QuasiGrad solver recently competed in the [3rd Grid Optimizaiton Competition](https://gocompetition.energy.gov/challenges/challenge-3), scoring reasonably well in the day-ahead market clearing problem. QuasiGrad's internal gradient-based solver (Adam) can easily be substituted for other ML-inspired solvers (e.g., AdaGrad, AdaDelta, RMSProp, etc.).

## Installation
QuasiGrad can be installed using the Julia package manager via
```
] add QuasiGrad
```
Julia 1.9 or higher is reccomended for use with QuasiGrad. Julia should be launched with as many CPU threads as you wish to dedicate to the solve. More threads, more parallelization.

## Usage
This package is a work-in-progress, and contributions are more than welcome. A valid Gurobi license is needed to run the solver (with some effort, this solver can be swapped out). Note: no QuasiGrad functions are exported to Julia's namespace.

Out-of-the-box usage can be acheived by running the following "division 1" example (it solves reserve+security constrained AC Unit Commitment on the 617 bus system).

```
using QuasiGrad

QuasiGrad_root = dirname(dirname(pathof(QuasiGrad)))
test_file = joinpath(QuasiGrad_root,"data","C3E3.1_20230629","D1","C3E3N00617D1","scenario_001.json")

# define some test parameters
NewTimeLimitInSeconds = 600.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

# run the test
QuasiGrad.compute_quasiGrad_solution_d1(test_file, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
```
## Questions
Please contact [Sam Chevalier](https://samchevalier.github.io/) (schevali@uvm.edu) with any questions.
