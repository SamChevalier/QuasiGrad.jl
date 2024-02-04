# QuasiGrad.jl

QuasiGrad.jl is a parallelized, adam-based solver for reserve and security constrained AC unit commitment problems. Solver details are sketeched out in a recent [pre-print](https://arxiv.org/pdf/2310.06650.pdf) and the associated [SI](https://samchevalier.github.io/docs/SI.pdf).

The QuasiGrad solver recently competed in the [3rd Grid Optimizaiton Competition](https://gocompetition.energy.gov/challenges/challenge-3), scoring reasonably well in the day-ahead market clearing problem. QuasiGrad's internal gradient-based solver (Adam) can easily be substituted for other ML-inspired solvers (e.g., AdaGrad, AdaDelta, RMSProp, etc.).

## Installation
QuasiGrad can be installed using the Julia package manager via
```julia
Pkg.add(url="https://github.com/SamChevalier/QuasiGrad")
```
Alternatively, this repo can be cloned locally for more interative usage.

## Usage
This package is a work-in-progress, and contributions are more than welcome. Out-of-the-box usage, however, can be acheived by running the example file in the test folder. The workhorse function is

```julia
QuasiGrad.compute_QuasiGrad_solution_d1(InFile, NewTimeLimitInSeconds, Division, NetworkModel, AllowSwitching; post_process=true)
```
## Questions
Please contact [Sam Chevalier](https://samchevalier.github.io/) (schevali@uvm.edu) with any questions.