## Action distributions and model configurations ##

export ActState, ActConfig
export DetermActConfig, EpsilonGreedyActConfig
export BoltzmannActConfig, BoltzmannMixtureActConfig
export HierarchicalEpsilonActConfig, HierarchicalBoltzmannActConfig
export ReplanMixtureActConfig
export BoltzmannReplanMixtureActConfig, HBoltzmannReplanMixtureActConfig
export EpsilonReplanMixtureActConfig, HEpsilonReplanMixtureActConfig
export CommunicativeActConfig
export policy_dist

"""
    ActState(action::Term, [metadata])
    ActState(act_state, [metadata])

Generic action state, which combines either an `action` term or a nested
`act_state` with optional `metadata`.
"""
struct ActState{T, U}
    act_state::T
    metadata::U
end

ActState(act_state) = ActState(act_state, ())
ActState{T}(act_state::T, metadata) where {T} =
    ActState{T, typeof(metadata)}(act_state, metadata)

Base.convert(::Type{Term}, act_state::ActState) = 
    convert(Term, act_state.act_state)

"""
    ActConfig

Configuration of an agent's action model.

# Fields

$(FIELDS)
"""
struct ActConfig{T,U,V}
    "Initializer with arguments `(agent_state, env_state, init_args...)`."
    init::T
    "Trailing arguments to initializer."
    init_args::U
    "Transition function with arguments `(t, act_state, agent_state, env_state, step_args...)`."
    step::GenerativeFunction
    "Trailing arguments to transition function."
    step_args::V
end

# Deterministic action selection #

"""
    DetermActConfig()

Constructs an `ActConfig` which deterministically selects the planned best
action for the current state, give the current plan or policy.
"""
function DetermActConfig()
    return ActConfig(PDDL.no_op, (), determ_act_step, ())
end

"""
    determ_act_step(t, act_state, agent_state, env_state)

Deterministic action selection from the current plan or policy. Returns
a no-op action if the goal has been achieved.
"""
@gen function determ_act_step(t, act_state, agent_state, env_state)
    # Planning solution is assumed to be a deterministic plan or policy
    plan_state = agent_state.plan_state::PlanState
    act = {:act} ~ policy_dist(plan_state.sol, env_state)
    return act
end

# Epsilon greedy action selection #

"""
    EpsilonGreedyActConfig(domain::Domain, epsilon::Real,
                           default=RandomPolicy(domain))

Constructs an `ActConfig` which selects a random action `epsilon` of the time
and otherwise selects the best/planned action.

If a `default` policy is provided, then actions will be selected from this
policy when the goal is unreachable or has been achieved.
"""
function EpsilonGreedyActConfig(domain::Domain, epsilon::Real,
                                default = RandomPolicy(domain))
    return ActConfig(PDDL.no_op, (), eps_greedy_act_step,
                     (domain, epsilon, default))
end

"""
    eps_greedy_act_step(t, act_state, agent_state, env_state,
                        domain, epsilon, default)

Samples an available action uniformly at random `epsilon` of the time, otherwise
selects the best action(s) (tie-breaking randomly if there is more than one).
"""
@gen function eps_greedy_act_step(
    t, act_state, agent_state, env_state,
    domain::Domain, epsilon::Real, default::Solution
)
    sol = agent_state.plan_state.sol
    # Check if goal is unreachable or has been achieved
    is_done = (sol isa NullSolution ||
               (sol isa PathSearchSolution && sol.status == :success &&
                env_state == sol.trajectory[end]))
    # Specify policy based on whether goal is unreachable or achieved
    policy = is_done ? default : EpsilonGreedyPolicy(domain, sol, epsilon)
    act = {:act} ~ policy_dist(policy, env_state)
    return act
end

# Hierarchical epsilon greedy action selection #

"""
    HierarchicalEpsilonActConfig(domain, epsilons, [prior_weights],
                                 [default=RandomPolicy(domain)])

Constructs an `ActConfig` which samples actions according to a hierarchical
epsilon-greedy policy, given a categorical prior over epsilon. The prior can be
specified by a list of `epsilons` and optional `prior_weights` which sum to 1.0.

After each action is sampled or observed, the weights for each epsilon are
automatically updated via Bayes rule. This policy thus functions as
a Rao-Blackwellized version of the joint distribution over epsilons and
actions, where the epsilon variable has been marginalized out.

If a `default` policy is provided, then actions will be selected from this
policy when the goal is unreachable or has been achieved.
"""
function HierarchicalEpsilonActConfig(
    domain::Domain,
    epsilons::AbstractVector{<:Real},
    prior_weights::AbstractVector{<:Real} =
        ones(length(epsilons)) ./ length(epsilons),
    default = RandomPolicy(domain)
)
    epsilons = convert(Vector{Float64}, epsilons)
    prior_weights = convert(Vector{Float64}, prior_weights)
    init_act_state = ActState{Term}(PDDL.no_op, prior_weights)
    return ActConfig(init_act_state, (), h_epsilon_act_step,
                     (domain, epsilons, default))
end

"""
    h_epsilon_act_step(t, act_state, agent_state, env_state,
                       domain, epsilons, default)

Samples actions according to a hierarchical epsilon-greedy policy, then updates
the distribution over epsilons via Bayes rule.

The input `act_state` is expected to contain the epsilon weights in its
`metadata` field, and the output `act_state` will contain the updated weights
in the same field.
"""
@gen function h_epsilon_act_step(
    t, act_state::ActState, agent_state, env_state,
    domain::Domain, epsilons::Vector{Float64}, default::Solution
)
    plan_state = agent_state.plan_state::PlanState
    weights = act_state.metadata
    # Check if goal is unreachable or has been achieved
    sol = agent_state.plan_state.sol
    is_done = (sol isa NullSolution ||
               (sol isa PathSearchSolution && sol.status == :success &&
                env_state == sol.trajectory[end]))
    if is_done # Sample action according to default policy
        policy = default
    else # Sample action according to current mixture weights
        policy = EpsilonMixturePolicy(domain, plan_state.sol, epsilons, weights)
    end
    act = {:act} ~ policy_dist(policy, env_state)
    # Skip weight update if action node was intervened upon
    if act.name == DO_SYMBOL || is_done
        return ActState{Term}(act, weights)
    end
    # Update distribution over mixture weights
    new_weights = SymbolicPlanners.get_mixture_weights(policy, env_state, act)
    total_weight = sum(new_weights)
    if total_weight == 0 || isnan(total_weight)
        new_weights = ones(length(epsilons)) ./ length(epsilons)
    else
        new_weights = new_weights ./ total_weight
    end
    return ActState{Term}(act, new_weights)
end

# Boltzmann action selection #

"""
    BoltzmannActConfig(temperature::Real, [epsilon=0.0])

Constructs an `ActConfig` which samples actions according to the Boltzmann
distribution over action Q-values with a specified `temperature`. To prevent 
zero-probability actions, an `epsilon` value can be specified, creating a
mixture between Boltzmann and epsilon-greedy policies.
"""
function BoltzmannActConfig(temperature::Real, epsilon::Real=0.0)
    return ActConfig(PDDL.no_op, (), boltzmann_act_step,
                     (temperature, epsilon))
end

"""
    boltzmann_act_step(t, act_state, agent_state, env_state,
                       temperature, epsilon = 0.0)

Samples actions according to the Boltzmann distribution over action values with
a specified `temperature` and optional `epsilon` value.
"""
@gen function boltzmann_act_step(t, act_state, agent_state, env_state,
                                 temperature::Real, epsilon::Real=0.0)
    plan_state = agent_state.plan_state::PlanState
    policy = BoltzmannPolicy(plan_state.sol, temperature, epsilon)
    act = {:act} ~ policy_dist(policy, env_state)
    return act
end

# Boltzmann mixture action selection #

"""
    BoltzmannMixtureActConfig(temperatures, [weights, epsilon=0.0])

Constructs an `ActConfig` which samples actions according to a mixture of
Boltzmann distributions over action Q-values, given a vector of `temperatures`
and optional `weights` which sum to 1.0. If `weights` is not specified, then
all weights are equal.

To prevent zero-probability actions, an `epsilon` value can be specified, as in
[`BoltzmannActConfig`](@ref).
"""
function BoltzmannMixtureActConfig(
    temperatures::AbstractVector{<:Real},
    weights::AbstractVector{<:Real} =
        ones(length(temperatures)) ./ length(temperatures),
    epsilon::Real=0.0
)
    temperatures = convert(Vector{Float64}, temperatures)
    weights = convert(Vector{Float64}, weights)
    return ActConfig(PDDL.no_op, (), boltzmann_mixture_act_step,
                     (temperatures, weights, epsilon))
end

"""
    boltzmann_mixture_act_step(t, act_state, agent_state, env_state,
                               temperatures, weights, epsilon = 0.0)

Samples actions according to a mixture of Boltzmann distributions over action
values with the specified `temperatures` and mixture `weights`, and optional
`epsilon` value.
"""
@gen function boltzmann_mixture_act_step(
    t, act_state, agent_state, env_state,
    temperatures::Vector{Float64}, weights::Vector{Float64}, epsilon::Real=0.0
)
    plan_state = agent_state.plan_state::PlanState
    policy = BoltzmannMixturePolicy(plan_state.sol, temperatures,
                                    weights, epsilon)
    act = {:act} ~ policy_dist(policy, env_state)
    return act
end

# Hierarchical Boltzmann action selection #

"""
    HierarchicalBoltzmannActConfig(temperatures, [prior_weights, epsilon])
    HierarchicalBoltzmannActConfig(temperatures, prior::Gen.Distribution,
                                   prior_args::Tuple, [epsilon])

Constructs an `ActConfig` which samples actions according to a hierarchical
Boltzmann policy, given a categorical prior over the temperature of the policy.
The prior can be specified by a list of `temperatures`, and optional
`prior_weights` which sum to 1.0.

Alternatively, a continuous univariate `prior` and `prior_args` can be provided,
which will be used to construct a categorical prior over the temperatures by 
normalizing the probability densities at each specified temperature.

After each action is sampled or observed, the temperature weights are
automatically updated via Bayes rule. This policy thus functions as
a Rao-Blackwellized version of the joint distribution over temperatures and
actions, where the temperature variable has been marginalized out.

To prevent zero-probability actions, an `epsilon` value can be specified, as in
[`BoltzmannActConfig`](@ref).
"""
function HierarchicalBoltzmannActConfig(
    temperatures::AbstractVector{<:Real},
    prior_weights::AbstractVector{<:Real} =
        ones(length(temperatures)) ./ length(temperatures),
    epsilon::Real=0.0
)
    temperatures = convert(Vector{Float64}, temperatures)
    prior_weights = convert(Vector{Float64}, prior_weights)
    init_act_state = ActState{Term}(PDDL.no_op, prior_weights)
    return ActConfig(init_act_state, (), h_boltzmann_act_step,
                     (temperatures, epsilon))
end

function HierarchicalBoltzmannActConfig(
    temperatures::AbstractVector{<:Real},
    prior::Gen.Distribution, prior_args::Tuple, epsilon::Real=0.0
)
    temperatures = convert(Vector{Float64}, temperatures)
    prior_weights = map(temperatures) do temp
        logpdf(prior, temp, prior_args...)
    end |> softmax
    init_act_state = ActState{Term}(PDDL.no_op, prior_weights)
    return ActConfig(init_act_state, (), h_boltzmann_act_step,
                     (temperatures, epsilon))
end

"""
    h_boltzmann_act_step(t, act_state, agent_state, env_state, temperatures)

Samples actions according to a hierarchical Boltzmann policy, then updates
the distribution over temperatures via Bayes rule.

The input `act_state` is expected to contain the temperature weights in its
`metadata` field, and the output `act_state` will contain the updated weights
in the same field.
"""
@gen function h_boltzmann_act_step(
    t, act_state::ActState, agent_state, env_state,
    temperatures::Vector{Float64}, epsilon::Real = 0.0
)
    plan_state = agent_state.plan_state::PlanState
    weights = act_state.metadata
    # Sample action according to current mixture weights
    policy = BoltzmannMixturePolicy(plan_state.sol, temperatures,
                                    weights, epsilon)
    act = {:act} ~ policy_dist(policy, env_state)
    # Skip weight update if action node was intervened upon
    if act.name == DO_SYMBOL
        return ActState{Term}(act, weights)
    end
    # Update distribution over mixture weights
    new_weights = SymbolicPlanners.get_mixture_weights(policy, env_state, act)
    total_weight = sum(new_weights)
    if total_weight == 0 || isnan(total_weight)
        new_weights = ones(length(temperatures)) ./ length(temperatures)
    else
        new_weights = new_weights ./ total_weight
    end
    return ActState{Term}(act, new_weights)
end

# Replan mixture action selection #

"""
    ReplanMixtureActConfig(subpolicy_type, subpolicy_args, [subweights])

Constructs an `ActConfig` which samples actions from a mixture of sub-solutions
produced by `ReplanMixturePolicyConfig`. Each sub-solution is wrapped in a
randomized sub-policy of `subpolicy_type` with arguments `subpolicy_args`.

The probability of the sampled action under each sub-policy is passed to the
next planning step, enabling the local posterior over sub-policies to be updated
at each step by `ReplanMixturePolicyConfig`.

If `subweights` is specified, then the sub-policies are assumed to have 
hierarchical priors over their parameters (e.g. temperature). These
distributions are updated for each sub-policy after an action is chosen.
"""
function ReplanMixtureActConfig(
    subpolicy_type::Type{<:PolicySolution},
    subpolicy_args::NamedTuple,
    subweights::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    if !isnothing(subweights)
        metadata = (act_logprobs=0.0, subweights=subweights)
    else
        metadata = (act_logprobs=0.0,)
    end
    init_act_state = ActState{Term}(PDDL.no_op, metadata)
    return ActConfig(init_act_state, (), replan_mixture_act_step,
                     (subpolicy_type, subpolicy_args))
end

"""
    BoltzmannReplanMixtureActConfig(temperature, [epsilon])

Convenience constructor for [`ReplanMixtureActConfig`](@ref) with a 
Boltzmann policy.
"""
function BoltzmannReplanMixtureActConfig(temperature::Real, epsilon::Real=0.0)
    args = (temperature=temperature, epsilon=epsilon)
    return ReplanMixtureActConfig(BoltzmannPolicy, args)
end

"""
    HBoltzmannReplanMixtureActConfig(temperatures, [prior_weights, epsilon])

Convenience constructor for [`ReplanMixtureActConfig`](@ref) with a
hierarchical Boltzmann policy.
"""
function HBoltzmannReplanMixtureActConfig(
    temperatures::AbstractVector{<:Real}, 
    prior_weights::AbstractVector{<:Real} =
        ones(length(temperatures)) ./ length(temperatures),
    epsilon::Real=0.0
)
    args = (temperatures=temperatures, epsilon=epsilon)
    return ReplanMixtureActConfig(BoltzmannMixturePolicy, args, prior_weights)
end

"""
    EpsilonReplanMixtureActConfig(domain, epsilon)

Convenience constructor for [`ReplanMixtureActConfig`](@ref) with an 
epsilon-greedy policy.
"""
function EpsilonReplanMixtureActConfig(domain::Domain, epsilon::Real)
    args = (domain=domain, epsilon=epsilon)
    return ReplanMixtureActConfig(EpsilonGreedyPolicy, args)
end

"""
    HEpsilonReplanMixtureActConfig(domain, epsilons, [prior_weights])

Convenience constructor for [`ReplanMixtureActConfig`](@ref) with a 
hierarchical epsilon-greedy policy.
"""
function HEpsilonReplanMixtureActConfig(
    domain::Domain, epsilons::AbstractVector{<:Real},
    prior_weights::AbstractVector{<:Real} =
        ones(length(epsilons)) ./ length(epsilons)
)
    args = (domain=domain, epsilon=epsilon)
    return ReplanMixtureActConfig(EpsilonMixturePolicy, args, prior_weights)
end

"""
    replan_mixture_act_step(t, act_state, agent_state, env_state,
                            subpolicy_type, subpolicy_args)

Samples actions from a mixture of sub-solutions produced by
`ReplanMixturePolicyConfig`. Each sub-solution is wrapped in a randomized
sub-policy of `subpolicy_type` with arguments `subpolicy_args`.

The probability of the sampled action under each sub-policy is passed to the
next planning step, enabling the local posterior over sub-policies to be updated
at each step by `ReplanMixturePolicyConfig`.

If `subweights` is specified in the `act_state.metadata`, then the sub-policies
are assumed to have hierarchical priors over their parameters (e.g. temperature).
These distributions are updated for each sub-policy after an action is chosen.
"""
@gen function replan_mixture_act_step(
    t, act_state::ActState, agent_state, env_state,
    subpolicy_type, subpolicy_args
)
    plan_state = agent_state.plan_state::PlanState
    sol = plan_state.sol::MixturePolicy
    # Extract sub-mixture weights from previous action state
    subweights = get(act_state.metadata, :subweights, nothing)
    if t == 1 && subweights isa AbstractVector{<:Real}
        subweights = fill(subweights, length(sol.policies))
    end
    # Update sub-mixture weights if there was replanning or refinement
    if !isnothing(subweights) && (plan_state.metadata.replan != 1)
        prev_weights = plan_state.metadata.prev_weights
        marginal_subweights = prev_weights' * subweights
        subweights = fill(marginal_subweights, length(subweights))
    end        
    # Construct new mixture policy by mixing subpolicies
    policy = remix_policy(sol, subpolicy_type, subpolicy_args, subweights)
    # Sample action from mixture policy
    act = {:act} ~ policy_dist(policy, env_state)
    # Compute probability of action under each sub-policy
    if !isnothing(subweights) # Hierarchical sub-policies
        if act.name == DO_SYMBOL
            # Avoid conditioning on intervened actions
            act_logprobs = zeros(length(policy.policies))
            metadata = (act_logprobs=act_logprobs, subweights=subweights)
            return ActState{Term}(act, metadata)
        end
        # Also update sub-mixture weights
        act_logprobs = fill(-Inf, length(policy.policies))
        new_subweights = copy(subweights)
        for i in eachindex(policy.policies)
            policy.weights[i] <= 0 && continue
            joint_probs = SymbolicPlanners.get_mixture_weights(
                policy.policies[i], env_state, act, normalize=false
            )
            act_prob = sum(joint_probs)
            act_logprobs[i] = log(act_prob)
            new_subweights[i] = joint_probs ./ act_prob
        end
        metadata = (act_logprobs=act_logprobs, subweights=new_subweights)
        return ActState{Term}(act, metadata)
    else # Non-hierarchical sub-policies
        if act.name == DO_SYMBOL
            # Avoid conditioning on intervened actions
            act_logprobs = zeros(length(policy.policies))
            metadata = (act_logprobs=act_logprobs,)
            return ActState{Term}(act, metadata)
        end
        act_logprobs = map(policy.policies) do subpolicy
            log(SymbolicPlanners.get_action_prob(subpolicy, env_state, act))
        end
        metadata = (act_logprobs=act_logprobs,)
        return ActState{Term}(act, metadata)
    end
end

"Wrap sub-solutions of a mixture policy within randomized sub-policies." 
function remix_policy(
    mixture::MixturePolicy,
    subpolicy_type, args::NamedTuple, subweights = nothing
)
    if subpolicy_type === BoltzmannPolicy
        subpolicies = map(mixture.policies) do subsol
            BoltzmannPolicy(subsol, args.temperature, args.epsilon)
        end
        return MixturePolicy(subpolicies, mixture.weights)
    elseif subpolicy_type === BoltzmannMixturePolicy
        if isnothing(subweights) # Non-hierarchical case
            subpolicies = map(mixture.policies) do subsol
                BoltzmannMixturePolicy(subsol, args.temperatures,
                                       args.weights, args.epsilon)
            end
            return MixturePolicy(subpolicies, mixture.weights)
        else # Hierarchical case
            @assert length(mixture.policies) == length(subweights)
            subpolicies = map(zip(mixture.policies, subweights)) do (subsol, ws)
                BoltzmannMixturePolicy(subsol, args.temperatures,
                                       ws, args.epsilon)
            end
            return MixturePolicy(subpolicies, mixture.weights)
        end
    elseif subpolicy_type === EpsilonGreedyPolicy
        subpolicies = map(mixture.policies) do subsol
            EpsilonGreedyPolicy(args.domain, subsol, args.epsilon)
        end
        return MixturePolicy(subpolicies, mixture.weights)
    elseif subpolicy_type === EpsilonMixturePolicy
        if isnothing(subweights) # Non-hierarchical case
            subpolicies = map(mixture.policies) do subsol
                EpsilonMixturePolicy(args.domain, subsol,
                                     args.epsilon, args.weights)
            end
            return MixturePolicy(subpolicies, mixture.weights)
        else # Hierarchical case
            @assert length(mixture.policies) == length(subweights)
            subpolicies = map(zip(mixture.policies, subweights)) do (subsol, ws)
                EpsilonMixturePolicy(args.domain, subsol, args.epsilon, ws)
            end
            return MixturePolicy(subpolicies, mixture.weights)
        end
    end
end

# Joint communication / action model #

"""
    CommunicativeActConfig(
        act_config::ActConfig,
        utterance_model::GenerativeFunction,
        utterance_args::Tuple = ()
    )

Constructs an `ActConfig` which samples an action and utterance jointly,
combining a (non-communicative) `act_config` with an `utterance_model`. At each
step, a `new_act` is sampled according to `act_config.step`:

    new_act ~ act_config.step(t, act_state.act_state, agent_state, env_state,
                              act_step_args...)

Then an `utterance` is sampled from the `utterance_model` given `new_act`:

    utterance ~ utterance_model(t, act_state, agent_state, env_state,
                                new_act, utterance_args...)

The `utterance_model` may return an `utterance` as a string, or a
(`utterance`, `history`) tuple. The communication `history` is passed to the
next step in the `metadata` field of an `ActState`, but defaults to `nothing` if
not returned by the `utterance_model`.
"""
function CommunicativeActConfig(
    act_config::ActConfig, utterance_model::GenerativeFunction,
    utterance_args::Tuple = ()
)
    act_init = act_config.init
    act_init_args = act_config.init_args
    act_step = act_config.step
    act_step_args = act_config.step_args
    return ActConfig(
        communicative_act_init,
        (act_init, act_init_args, utterance_model, utterance_args),
        communicative_act_step,
        (act_step, act_step_args, utterance_model, utterance_args)
    )
end

@gen function communicative_act_init(
    agent_state, env_state,
    act_init, act_init_args,
    utterance_model, utterance_args
)
    # Sample action
    new_act = {*} ~ maybe_sample(act_init, (agent_state, env_state,
                                            act_init_args...))
    # Sample utterance
    init_act_state = ActState(new_act, (utterance="", history=nothing))
    result = {*} ~ utterance_model(0, init_act_state, agent_state, env_state,
                                   new_act, utterance_args...)
    if isa(result, Tuple)
        utterance, history = result
        return ActState(new_act, (utterance=utterance, history=history))
    else
        utterance = result
        return ActState(new_act, (utterance=utterance, history=nothing))
    end
end

@gen function communicative_act_step(
    t, act_state::ActState, agent_state, env_state,
    act_step, act_step_args,
    utterance_model, utterance_args
)
    # Sample action
    new_act = {*} ~ act_step(t, act_state.act_state, agent_state, env_state,
                             act_step_args...)
    # Sample utterance
    result = {*} ~ utterance_model(t, act_state, agent_state, env_state,
                                   new_act, utterance_args...)
    if isa(result, Tuple)
        utterance, history = result
        return ActState(new_act, (utterance=utterance, history=history))
    else
        utterance = result
        return ActState(new_act, (utterance=utterance, history=nothing))
    end
end

# Do operator #

const DO_SYMBOL = :do

function do_op(act::Term)
    return Compound(DO_SYMBOL, [act])
end

# Policy distribution #

struct PolicyDistribution <: Gen.Distribution{Term} end

"""
    policy_dist(policy, state)

Gen `Distribution` that samples an action from a SymbolicPlanners `policy`
given the current `state`. If no actions are available, returns `PDDL.no_op`
with probability 1.0.
"""
const policy_dist = PolicyDistribution()

(d::PolicyDistribution)(args...) = Gen.random(d, args...)

@inline function Gen.random(::PolicyDistribution, policy, state)
    act = SymbolicPlanners.rand_action(policy, state)
    return ismissing(act) ? PDDL.no_op::Term : act::Term
end

@inline function Gen.logpdf(::PolicyDistribution, act::Term, policy, state)
    return (act.name == PDDL.no_op.name || act.name == DO_SYMBOL) ?
        0.0 : log(SymbolicPlanners.get_action_prob(policy, state, act))
end

# Always return no-op for null solutions
@inline Gen.random(::PolicyDistribution, ::NullSolution, state) =
    PDDL.no_op
@inline Gen.logpdf(::PolicyDistribution, act::Term, ::NullSolution, state) =
    (act.name == PDDL.no_op.name || act.name == DO_SYMBOL) ? 0.0 : -Inf

Gen.logpdf_grad(::PolicyDistribution, act::Term, policy, state) =
    (nothing, nothing, nothing)
Gen.has_output_grad(::PolicyDistribution) =
    false
Gen.has_argument_grads(::PolicyDistribution) =
    (false, false)
