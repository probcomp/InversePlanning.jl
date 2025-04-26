export RejuvenationKernel
export NullKernel, SequentialKernel, MixtureKernel
export ReplanKernel, ConsecutiveReplanKernel
export InitGoalKernel, RecentGoalKernel, ConsecutiveGoalKernel
export NestedISKernel, NestedISReplanKernel, NestedISRecentGoalKernel

"Abstract type for SIPS rejuvenation kernels."
abstract type RejuvenationKernel end

"""
    is_reweight_kernel(kernel::RejuvenationKernel)

Returns `true` if the kernel is a reweighting kernel. 
"""
is_reweight_kernel(kernel::RejuvenationKernel) = false

"""
    NullKernel()

Null rejuvenation kernel that always returns original trace.
"""
struct NullKernel <: RejuvenationKernel end

(kernel::NullKernel)(trace::Trace) = trace, false

"""
    SequentialKernel(subkernels...)

Composite kernel that applies each subkernel in sequential order.
"""
struct SequentialKernel{Ks <: Tuple} <: RejuvenationKernel
    subkernels::Ks
end

SequentialKernel(subkernel::Union{Function,RejuvenationKernel}) =
    SequentialKernel((subkernel,))
SequentialKernel(subkernel, subkernels...) =
    SequentialKernel((subkernel, subkernels...))

is_reweight_kernel(kernel::SequentialKernel) =
    any(is_reweight_kernel, kernel.subkernels)

function (kernel::SequentialKernel)(trace::Trace)
    if is_reweight_kernel(kernel)
        weight = 0.0
        for k in kernel.subkernels
            trace, accept_or_weight = k(trace)
            is_reweight_kernel(k) && (weight += accept_or_weight)
            return trace, weight
        end
    else
        accept = false
        for k in kernel.subkernels
            trace, accept = k(trace)
        end
        return trace, accept
    end
end

"""
    MixtureKernel(probs::Vector{Float64}, subkernels::Tuple)

Composite kernel that applies each subkernel with a certain probability.
"""
struct MixtureKernel{Ks <: Tuple} <: RejuvenationKernel
    probs::Vector{Float64}
    subkernels::Ks
    function MixtureKernel(probs, subkernels::Ks) where {Ks <: Tuple}
        @assert length(probs) == length(subkernels)
        @assert sum(probs) ≈ 1.0
        @assert !any(is_reweight_kernel, subkernels)
        return new{Ks}(probs, subkernels)
    end
end

MixtureKernel(probs, subkernels) =
    MixtureKernel(probs, Tuple(collect(subkernels)))
MixtureKernel(probs, k::Union{Function,RejuvenationKernel}, ks...) =
    MixtureKernel(probs, (k, ks...))

function (kernel::MixtureKernel)(trace::Trace)
    idx = categorical(kernel.probs)
    return kernel.subkernels[idx](trace)
end

"""
    ReplanKernel(n::Int=1)

Performs a single Metropolis-Hastings resimulation move on the agent's planning
steps for the past `n` steps.
"""
struct ReplanKernel <: RejuvenationKernel
    n::Int
end

ReplanKernel() = ReplanKernel(1)

function (kernel::ReplanKernel)(trace::Trace)
    n_steps = Gen.get_args(trace)[1]
    start = max(n_steps-kernel.n+1, 1)
    sel = select((:timestep => t => :agent => :plan for t in start:n_steps)...)
    return mh(trace, sel)
end

"""
    ConsecutiveReplanKernel(n::Int, order::Symbol=:forward)

Performs `n` consecutive Metropolis-Hastings resimulation moves on the agent's
planning steps. The `order` argument determines whether the moves are performed
in `:forward` (earliest first) or `:backward` order (most recent first).
"""
function ConsecutiveReplanKernel(n::Int, order::Symbol=:forward)
    @assert order in (:forward, :backward)
    if order == :forward
        subkernels = ntuple(i -> ReplanKernel(n-i+1), n)
    else
        subkernels = ntuple(i -> ReplanKernel(i), n)
    end
    return SequentialKernel(subkernels)
end

"""
    InitGoalKernel()

Perform a Metropolis-Hastings resimulation move on the agent's initial goal.
"""
struct InitGoalKernel <: RejuvenationKernel end

function (kernel::InitGoalKernel)(trace::Trace)
    # Rejuvenate goal and all downstream agent choices
    n_steps = Gen.get_args(trace)[1]
    goal_addrs = (:timestep => t => :agent => :goal for t in 1:n_steps)
    plan_addrs = (:timestep => t => :agent => :plan for t in 1:n_steps)
    sel = select(:init => :agent => :goal, goal_addrs..., plan_addrs...)
    return mh(trace, sel)
end

"""
    RecentGoalKernel(n::Int=1)

Performs Metropolis-Hastings resimulation moves on the agent's goals for the
past `n` steps.
"""
struct RecentGoalKernel <: RejuvenationKernel
    n::Int
end

RecentGoalKernel() = RecentGoalKernel(1)

function (kernel::RecentGoalKernel)(trace::Trace)
    n_steps = Gen.get_args(trace)[1]
    start = max(n_steps-kernel.n+1, 1)
    goal_addrs = (:timestep => t => :agent => :goal for t in start:n_steps)
    plan_addrs = (:timestep => t => :agent => :plan for t in start:n_steps)
    sel = select(goal_addrs..., plan_addrs...)
    return mh(trace, sel)
end

"""
    ConsecutiveGoalKernel(n::Int, order::Symbol=:forward)

Performs `n` consecutive Metropolis-Hastings resimulation moves on the agent's
goals. The `order` argument determines whether the moves are performed in
`:forward` (earliest first) or `:backward` order (most recent first).
"""
function ConsecutiveGoalKernel(n::Int, order::Symbol=:forward)
    @assert order in (:forward, :backward)
    if order == :forward
        subkernels = ntuple(i -> RecentGoalKernel(n-i+1), n)
    else
        subkernels = ntuple(i -> RecentGoalKernel(i), n)
    end
    return SequentialKernel(subkernels)
end

"""
    NestedISKernel(sel::Selection, m::Int)

Performs a nested importance sampling rejuvenation move with `m` importance
samples on the addresses selected by `sel`.

This kernel is intended to undo the default proposal used to generate the 
choices in `sel` by computing an appropriate weight update. As such, it should
only be used once after the default proposal to ensure local optimality.
"""
struct NestedISKernel{S <: Selection} <: RejuvenationKernel
    sel::S
    m::Int
end

is_reweight_kernel(kernel::NestedISKernel) = true

function (kernel::NestedISKernel)(trace::Trace)
    new_trace = trace
    log_total_weight = 0.0
    # Generate m candidate traces via importance sampling, and select one
    for _ in 1:kernel.m
        cand_trace, log_weight, _ = regenerate(trace, kernel.sel)
        log_total_weight = logsumexp(log_total_weight, log_weight)
        if bernoulli(exp(log_weight - log_total_weight))
            new_trace = cand_trace
        end
    end
    # Compute weight update as if we are undoing the default proposal
    log_mean_weight = log_total_weight - log(kernel.m)
    return new_trace, log_mean_weight
end

"""
    NestedISReplanKernel(m::Int, n::Int=1)

Performs a nested importance sampling rejuvenation move on the agent's planning
steps for the past `n` steps, where `m` is the number of samples.

This kernel uses `NestedISKernel` to perform the rejuvenation move. As such, it
should only be used once after the default proposal was used to generate the
past `n` planning steps, without any resampling in that period.
"""
struct NestedISReplanKernel <: RejuvenationKernel
    m::Int
    n::Int
end

NestedISReplanKernel(m::Int) = NestedISReplanKernel(m, 1)

is_reweight_kernel(kernel::NestedISReplanKernel) = true

function (kernel::NestedISReplanKernel)(trace::Trace)
    n_steps = Gen.get_args(trace)[1]
    start = max(n_steps-kernel.n+1, 1)
    sel = select((:timestep => t => :agent => :plan for t in start:n_steps)...)
    nested_is = NestedISKernel(sel, kernel.m)
    return nested_is(trace)
end

"""
    NestedISRecentGoalKernel(m::Int, n::Int=1)

Performs a nested importance sampling rejuvenation move on the agent's goals for
the past `n` steps, where `m` is the number of samples.

This kernel uses `NestedISKernel` to perform the rejuvenation move. As such, it
should only be used once after the default proposal was used to generate the
past `n` goals and planning steps, without any resampling in that period.
"""
struct NestedISRecentGoalKernel <: RejuvenationKernel
    m::Int
    n::Int
end

NestedISRecentGoalKernel(m::Int) = NestedISRecentGoalKernel(m, 1)

is_reweight_kernel(kernel::NestedISRecentGoalKernel) = true

function (kernel::NestedISRecentGoalKernel)(trace::Trace)
    n_steps = Gen.get_args(trace)[1]
    start = max(n_steps-kernel.n+1, 1)
    goal_addrs = (:timestep => t => :agent => :goal for t in start:n_steps)
    plan_addrs = (:timestep => t => :agent => :plan for t in start:n_steps)
    sel = select(goal_addrs..., plan_addrs...)
    nested_is = NestedISKernel(sel, kernel.m)
    return nested_is(trace)
end
