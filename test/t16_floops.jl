using quasiGrad
using FLoops

# count threads
Threads.nthreads()
Threads.nthreads()

# how many threads are available?
Sys.CPU_THREADS

# %% now:
function ff()
    s = 0
    @floop ThreadedEx() for x in 1:300000
            s += x
        end
    return s
end

# %% now:
function f()
    s = 0
    for x in 1:300000
        s += x
    end
    return s
end

# %%
s_fast = ff()
s_slow = f()

@time println(s_fast)
@time println(s_slow)