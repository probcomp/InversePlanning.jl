import GenParticleFilters: ParticleFilterView

export probvec, logprobvec

"""
    probvec(pf::ParticleFilterView, addr)

Returns a probability vector for values of `addr` in the particle filter.
"""
function probvec(pf::ParticleFilterView, addr)
    pmap = proportionmap(pf, addr)
    keys = collect(keys(pmap))
    if hasmethod(isless, Tuple{eltype(keys), eltype(keys)})
        sort!(keys)
    end
    return [pmap[k] for k in keys]
end

function probvec(pf::ParticleFilterView, addr, support)
    pmap = proportionmap(pf, addr)
    return [get(pmap, k, 0.0) for k in support]
end

"""
    logprobvec(pf::ParticleFilterView, addr)

Returns a vector of log probabilities for values of `addr` in the particle filter.
"""
function logprobvec(pf::ParticleFilterView, addr)
    vals = [tr[addr] for tr in get_traces(pf)]
    log_probs = Dict{eltype(vals), Float64}()
    for (v, lp) in zip(vals, get_log_norm_weights(pf))
        log_probs[v] = logsumexp(get(log_probs, v, -Inf), lp)
    end
    if hasmethod(isless, Tuple{eltype(vals), eltype(vals)})
        sort!(vals)
    end
    return [log_probs[v] for v in vals]
end

function logprobvec(pf::ParticleFilterView, addr, support)
    vals = [tr[addr] for tr in get_traces(pf)]
    log_probs = Dict{eltype(vals), Float64}()
    for (v, lp) in zip(vals, get_log_norm_weights(pf))
        log_probs[v] = logsumexp(get(log_probs, v, -Inf), lp)
    end
    return [get(log_probs, k, -Inf) for k in support]
end
