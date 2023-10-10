# quasiGrad.jl

quasiGrad.jl is a parallelized, adam-based solver for reserve and security constrained AC unit commitment problems. The quasiGrad solver recently competed in the [3rd Grid Optimizaiton Competition](https://gocompetition.energy.gov/challenges/challenge-3), scoring reasonably well in the day-ahead market clearing problem. quasiGrad's internal gradient-based solver (Adam) can easily be substituted for other ML-inspired solvers (e.g., AdaGrad, AdaDelta, RMSProp, etc.).

## Installation
quasiGrad can be installed using the Julia package manager with

```julia
] add PowerModels
```
