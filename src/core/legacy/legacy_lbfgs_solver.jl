# note: this is a full on lbfgs solver! i.e., to replace adam
function initialize_lbfgs(mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, sys::quasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # first, how many historical gradients do we keep? 
    # 2 < n < 21 according to Wright

    @info "this is general lbfgs solver -- it does not work well"

    num_lbfgs_to_keep = 8

    # define the mapping indices which put gradient and state
    # information into aggregated forms -- to be populated!
    tkeys = prm.ts.time_keys

    lbfgs_map = Dict(
        :vm            => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT),       
        :va            => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT),           
        :tau           => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT),            
        :phi           => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :dc_pfr        => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :dc_qfr        => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :dc_qto        => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :u_on_acline   => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT),  
        :u_on_xfm      => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT),  
        :u_step_shunt  => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT),
        :u_on_dev      => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_on          => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :dev_q         => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_rgu         => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_rgd         => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_scr         => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_nsc         => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_rru_on      => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_rrd_on      => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_rru_off     => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :p_rrd_off     => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :q_qru         => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT), 
        :q_qrd         => Dict(tkeys[ii] => Int64[] for ii in 1:sys.nT))

    # next, how many lbfgs states are there? let's base this on "upd"
    n_lbfgs = 0
    for var_key in keys(mgd)
        for tii in prm.ts.time_keys
            if var_key in keys(upd)
                n = length(upd[var_key][tii]) # number
                if n == 0
                    lbfgs_map[var_key][tii] = Int64[]
                else
                    lbfgs_map[var_key][tii] = collect(1:n) .+ n_lbfgs
                end

                # update the total number
                n_lbfgs += n
            else
                n = length(mgd[var_key][tii])
                if n == 0
                    lbfgs_map[var_key][tii] = Int64[]
                else
                    lbfgs_map[var_key][tii] = collect(1:n) .+ n_lbfgs
                end

                # update the total number
                n_lbfgs += n
            end
        end
    end

    # build the lbfgs dict
    lbfgs = Dict(:x_now      => zeros(n_lbfgs),
                 :x_new      => zeros(n_lbfgs),
                 :x_prev     => zeros(n_lbfgs),
                 :gradf_now  => zeros(n_lbfgs),
                 :gradf_prev => zeros(n_lbfgs),
                 :alpha      => zeros(num_lbfgs_to_keep),
                 :rho        => zeros(num_lbfgs_to_keep))

    # build the lbfgs dict
    lbfgs_diff = Dict(:s => [zeros(n_lbfgs) for _ in 1:num_lbfgs_to_keep],
                      :y => [zeros(n_lbfgs) for _ in 1:num_lbfgs_to_keep])

    # step size
    lbfgs_step = Dict(:nzms_prev   => 0.0,
                      :beta1_decay => 1.0,
                      :beta2_decay => 1.0,
                      :m           => 0.0,   
                      :v           => 0.0,   
                      :mhat        => 0.0,
                      :vhat        => 0.0,
                      :step        => 1.0,
                      :alpha_0     => 1.0)
    # => lbfgs_step = 1.0

    # indices to track where previous differential vectors are stored --
    # lbfgs_idx[1] is always the most recent data, and lbfgs_idx[end] is the oldest
    lbfgs_idx = Int64.(zeros(num_lbfgs_to_keep))

    return lbfgs, lbfgs_diff, lbfgs_idx, lbfgs_map, lbfgs_step
end

# lbfgs
function lbfgs!(lbfgs::Dict{Symbol, Vector{Float64}}, lbfgs_diff::Dict{Symbol, Vector{Vector{Float64}}}, lbfgs_idx::Vector{Int64}, lbfgs_map::Dict{Symbol, Dict{Symbol, Vector{Int64}}}, lbfgs_step::Dict{Symbol, Float64}, mgd::quasiGrad.MasterGrad, prm::quasiGrad.Param, qG::quasiGrad.QG, scr::Dict{Symbol, Float64}, stt::quasiGrad.State, upd::Dict{Symbol, Vector{Vector{Int64}}})
    # NOTE: based on testing on May 10 or so, lbfgs does NOT outperform adam,
    #       More fundamentally, it has a problem: the states "lbfgs[:x_now]",
    #       etc. need to modified after binaries are fixed. Right now, they
    #       are not. (i.e., some of the lbfgs states need to be removed).
    #
    # note: lbfgs_idx is a set of ordered indices, where the first is the most
    #       recent step information, and the last is the oldest step information
    #       in the following order: (k-1), (k-2)
    #
    # prepare the lbfgs structures -- x and gradf
    for var_key in keys(mgd)
        for tii in prm.ts.time_keys
            # states to update
            if var_key in keys(upd)
                update_subset = upd[var_key][tii]
                lbfgs[:x_now][lbfgs_map[var_key][tii]]     = copy(stt.var_key[tii][update_subset]) # no update_subset needed on lbfgs side
                lbfgs[:gradf_now][lbfgs_map[var_key][tii]] = copy(mgd[var_key][tii][update_subset]) # no update_subset needed on lbfgs side
            else
                lbfgs[:x_now][lbfgs_map[var_key][tii]]     = copy(stt.var_key[tii])
                lbfgs[:gradf_now][lbfgs_map[var_key][tii]] = copy(mgd[var_key][tii])
            end
        end
    end

    # if this is the very first iteration, just take a gradient step
    if sum(lbfgs_idx) == 0
        # if this is the very first iteration, just take a gradient step
        lbfgs[:x_new] = lbfgs[:x_now] - 1e-10*lbfgs[:gradf_now]

        # pass x back into the state vector
        for var_key in keys(mgd)
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    update_subset = upd[var_key][tii]
                    stt.var_key[tii][update_subset] = lbfgs[:x_new][lbfgs_map[var_key][tii]] # no update_subset needed
                else
                    stt.var_key[tii]                = lbfgs[:x_new][lbfgs_map[var_key][tii]]
                end
            end
        end

        # update the lbfgs states and grads
        lbfgs[:x_prev]     = copy(lbfgs[:x_now])
        lbfgs[:gradf_prev] = copy(lbfgs[:gradf_now])

        # now, let's initialize lbfgs_idx
        lbfgs_idx[1] = 1
    else
        # udpdate the state difference
        idx_km1 = lbfgs_idx[1]
        lbfgs_diff[:s][idx_km1] = lbfgs[:x_now]     - lbfgs[:x_prev]
        lbfgs_diff[:y][idx_km1] = lbfgs[:gradf_now] - lbfgs[:gradf_prev]
        lbfgs[:rho][idx_km1]    = 1/(quasiGrad.dot(lbfgs_diff[:s][idx_km1],lbfgs_diff[:y][idx_km1]))

        # now, double-loop and compute lbfgs values
        q = copy.(lbfgs[:gradf_now])
        for ii in lbfgs_idx[lbfgs_idx .!= 0] # k-1, k-2, ..., k-m
            lbfgs[:alpha][ii] = lbfgs[:rho][ii]*quasiGrad.dot(lbfgs_diff[:s][ii], q)
            q                -= lbfgs[:alpha][ii]*lbfgs_diff[:y][ii]
        end
        
        # set "r", which will be H*grad
        r = q*quasiGrad.dot(lbfgs_diff[:s][idx_km1],lbfgs_diff[:y][idx_km1])/quasiGrad.dot(lbfgs_diff[:y][idx_km1], lbfgs_diff[:y][idx_km1])
        
        # compute H*grad
        for ii in reverse(lbfgs_idx[lbfgs_idx .!= 0]) # k-m, k-m+1, ..., k-1
            # skip beta -- defined implicitly below
            r += lbfgs_diff[:s][ii]*(lbfgs[:alpha][ii] - lbfgs[:rho][ii]*quasiGrad.dot(lbfgs_diff[:y][ii], r))
        end

        if sum(lbfgs_idx) == 1
            # this is the first step, so just use 0.1
            lbfgs_step[:step] = 0.1
        else
            # decay beta
            lbfgs_step[:beta1_decay] = lbfgs_step[:beta1_decay]*qG.beta1
            lbfgs_step[:beta2_decay] = lbfgs_step[:beta2_decay]*qG.beta2

            # have the STEP take a step with adam!
            grad              = (scr[:nzms] - lbfgs_step[:nzms_prev])/lbfgs_step[:step]
            lbfgs_step[:m]    = qG.beta1.*lbfgs_step[:m] + (1.0-qG.beta1).*grad
            lbfgs_step[:v]    = qG.beta2.*lbfgs_step[:v] + (1.0-qG.beta2).*grad.^2.0
            lbfgs_step[:mhat] = lbfgs_step[:m]/(1.0-lbfgs_step[:beta1_decay])
            lbfgs_step[:vhat] = lbfgs_step[:v]/(1.0-lbfgs_step[:beta2_decay])
            lbfgs_step[:step] = lbfgs_step[:step] - lbfgs_step[:alpha_0]*lbfgs_step[:mhat]/(sqrt.(lbfgs_step[:vhat]) .+ qG.eps)
        end

        # lbfgs step
        lbfgs[:x_new] = lbfgs[:x_now] - lbfgs_step[:step]*r

        # pass x back into the state vector
        for var_key in keys(mgd)
            for tii in prm.ts.time_keys
                # states to update
                if var_key in keys(upd)
                    update_subset = upd[var_key][tii]
                    stt.var_key[tii][update_subset] = lbfgs[:x_new][lbfgs_map[var_key][tii]] # no update_subset needed
                else
                    stt.var_key[tii]                = lbfgs[:x_new][lbfgs_map[var_key][tii]]
                end
            end
        end

        # update the lbfgs states and grads
        lbfgs[:x_prev]     = copy(lbfgs[:x_now])
        lbfgs[:gradf_prev] = copy(lbfgs[:gradf_now])

        # finally, update the lbfgs indices -- rule: lbfgs_idx[1] is where 
        # we write the newest data, and every next index is successively
        # older data -- oldest data gets bumped when the dataset if full.
        #
        # v = [data(0), -, -]  => lbfgs_idx = [1,0,0]
        #
        # v = [data(0), data(1), -]  => lbfgs_idx = [2,1,0]
        #
        # v = [data(0), data(1), data(2)]  => lbfgs_idx = [3,2,1]
        # 
        # v = [data(3), data(1), data(2)]  => lbfgs_idx = [1,3,2]
        #
        # v = [data(3), data(4), data(2)]  => lbfgs_idx = [2,1,3]
        #
        # ....
        #
        # so, 1 becomes 2, 2 becomes 3, etc. :
        if 0 ∈ lbfgs_idx
            circshift!(lbfgs_idx, -1)
            lbfgs_idx[1] = lbfgs_idx[2] + 1
        else
            circshift!(lbfgs_idx, -1)
        end
    end
end
