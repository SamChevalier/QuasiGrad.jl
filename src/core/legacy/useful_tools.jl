# thread looping and tracking threadid()
Threads.@threads for i = 1:10
    a[i] = Threads.threadid()
end

Sys.cpu_info()

# %% ===
@btime Threads.threadid()

# %% ===
using LinearAlgebra
LinearAlgebra.BLAS.get_num_threads()
BLAS.get_num_threads()
BLAS.set_num_threads(1)