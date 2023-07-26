# thread looping and tracking threadid()
Threads.@threads for i = 1:10
    a[i] = Threads.threadid()
end