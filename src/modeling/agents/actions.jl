## Action distributions and model configurations ##

export ActState, ActConfig
export DetermActConfig, EpsilonGreedyActConfig, BoltzmannActConfig
export BoltzmannMixtureActConfig, HierarchicalBoltzmannActConfig
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
selects the best action.    
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

# Boltzmann action selection #

"""
    BoltzmannActConfig(temperature::Real)

Constructs an `ActConfig` which samples actions according to the Boltzmann
distribution over action Q-values with a specified `temperature`.
"""
function BoltzmannActConfig(temperature::Real)
    return ActConfig(PDDL.no_op, (), boltzmann_act_step, (temperature,))
end

"""
    boltzmann_act_step(t, act_state, agent_state, env_state, temperature)

Samples actions according to the Boltzmann distribution over action values with
a specified `temperature`
"""
@gen function boltzmann_act_step(t, act_state, agent_state, env_state,
                                 temperature::Real)
    plan_state = agent_state.plan_state::PlanState
    policy = BoltzmannPolicy(plan_state.sol, temperature)
    act = {:act} ~ policy_dist(policy, env_state)
    return act
end

# Boltzmann mixture action selection #

"""
    BoltzmannMixtureActConfig(temperatures, [weights])

Constructs an `ActConfig` which samples actions according to a mixture of
Boltzmann distributions over action Q-values, given a vector of `temperatures`
and optional `weights` which sum to 1.0. If `weights` is not specified, then
all weights are equal.
"""
function BoltzmannMixtureActConfig(
    temperatures::AbstractVector{<:Real},
    weights::AbstractVector{<:Real} =
        ones(length(temperatures)) ./ length(temperatures)
)
    temperatures = convert(Vector{Float64}, temperatures)
    weights = convert(Vector{Float64}, weights)
    return ActConfig(PDDL.no_op, (), boltzmann_mixture_act_step,
                     (temperatures, weights))
end

"""
    boltzmann_mixture_act_step(t, act_state, agent_state, env_state,
                               temperatures, weights)

Samples actions according to a mixture of Boltzmann distributions over action
values with the specified `temperatures` and mixture `weights`.
"""
@gen function boltzmann_mixture_act_step(
    t, act_state, agent_state, env_state,
    temperatures::Vector{Float64}, weights::Vector{Float64}
)
    plan_state = agent_state.plan_state::PlanState
    policy = BoltzmannMixturePolicy(plan_state.sol, temperatures, weights)
    act = {:act} ~ policy_dist(policy, env_state)
    return act
end

# Hierarchical Boltzmann action selection #

"""
    HierarchicalBoltzmannActConfig(temperatures, [prior_weights])
    HierarchicalBoltzmannActConfig(temperatures, prior::Gen.Distribution,
                                   prior_args::Tuple)

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
"""
function HierarchicalBoltzmannActConfig(
    temperatures::AbstractVector{<:Real},
    prior_weights::AbstractVector{<:Real} =
        ones(length(temperatures)) ./ length(temperatures)
)
    temperatures = convert(Vector{Float64}, temperatures)
    prior_weights = convert(Vector{Float64}, prior_weights)
    init_act_state = ActState(PDDL.no_op, prior_weights)
    return ActConfig(init_act_state, (), h_boltzmann_act_step,
                     (temperatures,))
end

function HierarchicalBoltzmannActConfig(
    temperatures::AbstractVector{<:Real},
    prior::Gen.Distribution, prior_args::Tuple
)
    temperatures = convert(Vector{Float64}, temperatures)
    prior_weights = map(temperatures) do temp
        logpdf(prior, temp, prior_args...)
    end |> softmax
    init_act_state = ActState(PDDL.no_op, prior_weights)
    return ActConfig(init_act_state, (), h_boltzmann_act_step,
                     (temperatures,))
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
    temperatures::Vector{Float64}
)
    plan_state = agent_state.plan_state::PlanState
    weights = act_state.metadata
    # Sample action according to current mixture weights
    policy = BoltzmannMixturePolicy(plan_state.sol, temperatures, weights)
    act = {:act} ~ policy_dist(policy, env_state)
    # Skip weight update if action node was intervened upon
    if act.name == DO_SYMBOL
        return ActState(act, weights)
    end
    # Update distribution over mixture weights
    new_weights = map(zip(temperatures, weights)) do (T, w)
        pol = BoltzmannPolicy(plan_state.sol, T)
        return w * SymbolicPlanners.get_action_prob(pol, env_state, act)
    end
    new_weights = new_weights ./ sum(new_weights)
    return ActState(act, new_weights)
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
    return ismissing(act) ? PDDL.no_op : act
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
