## Planning states and configurations ##

export PlanState
export PlanConfig, DetermReplanConfig, ReplanConfig
export ReplanPolicyConfig, ReplanMixturePolicyConfig

import SymbolicPlanners: NullSpecification, NullGoal

"""
    PlanState

Represents the plan or policy of an agent at a point in time.

# Fields

$(FIELDS)
"""
struct PlanState{T}
    "Initial timestep of the current plan."
    init_step::Int
    "Solution returned by the planner."
    sol::Solution
    "Specification that the solution is intended to satisfy."
    spec::Specification
    "Additional metadata."
    metadata::T
end

PlanState() =
    PlanState(0, NullSolution(), NullGoal(), ())
PlanState(init_step, sol, spec) =
    PlanState(init_step, sol, spec, ())

"Returns whether an action is planned for step `t` at a `belief_state`."
has_action(plan_state::PlanState, t::Int, belief_state) =
    has_action(plan_state.sol, t - plan_state.init_step + 1, belief_state)

has_action(sol::NullSolution, t::Int, state::State) =
    false
has_action(sol::NullPolicy, t::Int, state::State) =
    false
has_action(sol::PolicySolution, t::Int, state::State) =
    !ismissing(SymbolicPlanners.get_action(sol, state))
has_action(sol::OrderedSolution, t::Int, state::State) =
    ((0 < t <= length(sol) && sol[t] == state) ||
     !isnothing(findfirst(==(state), sol)))
has_action(sol::PathSearchSolution, t::Int, state::State) =
    !ismissing(SymbolicPlanners.get_action(sol, state))
has_action(sol::MultiSolution, t::Int, state::State) =
    has_action(sol.selector(sol.solutions, state), t, state)
has_action(sol::MixturePolicy, t::Int, state::State) =
    all(has_action(p, t, state) for p in sol.policies)

"Returns whether there are cached action value for step `t` at a `belief_state`."
has_cached_action(plan_state::PlanState, t::Int, belief_state) =
    has_cached_action(plan_state.sol, t - plan_state.init_step + 1, belief_state)
has_cached_action(sol::Solution, t::Int, state::State) =
    has_action(sol, t, state)
has_cached_action(sol::MultiSolution, t::Int, state::State) =
    has_cached_action(sol.selector(sol.solutions, state), t, state)
has_cached_action(sol::PolicySolution, t::Int, state::State) =
    SymbolicPlanners.has_cached_action_values(sol, state)
has_cached_action(sol::MixturePolicy, t::Int, state::State) =
    all(has_cached_action(p, t, state) for p in sol.policies)

"Returns whether the solution has a search node for `belief_state`."
has_search_node(plan_state::PlanState, belief_state) = 
    has_search_node(plan_state.sol, belief_state)
has_search_node(sol::Solution, state::State) =
    false
has_search_node(sol::PathSearchSolution, state::State) =
    SymbolicPlanners.is_expanded(state, sol)
has_search_node(sol::ReusableTreePolicy, state::State) =
    SymbolicPlanners.is_expanded(state, sol.search_sol)
has_search_node(sol::MixturePolicy, state::State) =
    all(has_search_node(p, state) for p in sol.policies)

"Check whether planning is required due to a goal change, etc."
function must_plan(plan_state::PlanState, goal_spec::Specification)
    return (plan_state.sol isa NullSolution || plan_state.sol isa NullPolicy ||
            (plan_state.spec !== goal_spec && plan_state.spec != goal_spec))
end

"Check whether replanning is required according to `replan_cond`."
function must_replan(
    replan_cond::Symbol, t::Int,
    plan_state::Union{PlanState, Solution}, belief_state::State
)
    if replan_cond == :unplanned
        # Enforce recomputation if action is unplanned
        return !has_action(plan_state, t, belief_state)
    elseif replan_cond == :uncached
        # Enforce recomputation if action is uncached
        return !has_cached_action(plan_state, t, belief_state)
    elseif replan_cond == :unexpanded
        # Enforce recomputation if current state is unexpanded
        return !has_search_node(plan_state, belief_state)
    else
        error("Invalid replan condition: $replan_cond")
    end
end

"""
    PlanConfig

Planning configuration for an agent model.

# Fields

$(FIELDS)
"""
struct PlanConfig{T,U,V}
    "Initializer with arguments `(belief_state, goal_state, init_args...)`."
    init::T
    "Trailing arguments to initializer."
    init_args::U
    "Transition function with arguments `(t, plan_state, belief_state, goal_state, act_state, step_args...)`."
    step::GenerativeFunction
    "Trailing arguments to transition function."
    step_args::V
end

"""
    default_plan_init(belief_state, goal_state, [init_metadata])

Default plan initialization, which returns a `PlanState` with a `NullSolution`
and the initial goal specification (i.e. no planning is done on the zeroth
timestep), with optional `init_metadata`.
"""
function default_plan_init(belief_state, goal_state, init_metadata=())
    spec = convert(Specification, goal_state)
    return PlanState(0, NullSolution(), spec, init_metadata)
end

"""
    step_plan_init(belief_state, goal_state,
                   plan_step, step_args, init_metadata)

Reuses the `plan_step` function to initialize the plan at the zeroth timestep.
An initial plan state is constructed with a `NullSolution` and the initial
goal specification and `init_metadata`, and then `plan_step` is called with
this initial plan state and `step_args`.
"""
@gen function step_plan_init(belief_state, goal_state, plan_step,
                             step_args, init_metadata)
    spec = convert(Specification, goal_state)
    plan_state = PlanState(0, NullSolution(), spec, init_metadata)
    plan_state = {*} ~ plan_step(0, plan_state, belief_state,
                                 goal_state, PDDL.no_op, step_args...)
    return plan_state
end

"""
    StaticPlanConfig(init=PlanState(), init_args=())

Constructs a `PlanConfig` that never updates the initial plan.
"""
function StaticPlanConfig(init=default_plan_init, init_args=())
    return PlanConfig(init, init_args, static_plan_step, ())
end

"""
    static_plan_step(t, plan_state, belief_state, goal_state, act_state)

Plan transition that returns the previous plan state without modification.
"""
@gen static_plan_step(t::Int, plan_state, belief_state, goal_state, act_state) =
    plan_state

# Deterministic (re)planning configuration #

"""
    DetermReplanConfig(domain::Domain, planner::Planner; plan_at_init=false)

Constructs a `PlanConfig` that deterministically replans only when necessary.
If `plan_at_init` is true, then the initial plan is computed at timestep zero.
"""
function DetermReplanConfig(domain::Domain, planner::Planner;
                            plan_at_init::Bool = false)
    if plan_at_init
        init = step_plan_init
        init_args = (determ_replan_step, (domain, planner), ())
    else
        init = default_plan_init
        init_args = ()
    end
    return PlanConfig(init, init_args, determ_replan_step, (domain, planner))
end

"""
    determ_replan_step(t, plan_state, belief_state, goal_state, domain, planner)

Deterministic replanning step, which only replans when an unexpected state
is encountered. Replanning is either a full horizon search or goes on until
a fixed budget.
"""
@gen function determ_replan_step(
    t::Int, plan_state::PlanState, belief_state::State, goal_state, act_state,
    domain::Domain, planner::Planner
)   
    spec = convert(Specification, goal_state)
    # Return original plan if an action is already computed
    if (has_action(plan_state, t, belief_state) &&
        (plan_state.spec === spec || plan_state.spec == spec))
        return plan_state
    else # Otherwise replan from the current belief state
        sol = planner(domain, belief_state, spec)
        return PlanState(t, sol, spec)
    end
end

# Stochastic (re)planning configuration #

"""
    ReplanConfig(
        domain::Domain, planner::Planner;
        plan_at_init::Bool = false,
        prob_replan::Real=0.1,
        prob_refine::Real=0.0,
        replan_period::Int=1,
        rand_budget::Bool = true,
        budget_var::Symbol = default_budget_var(planner),
        budget_dist::Distribution = shifted_neg_binom,
        budget_dist_args::Tuple = (2, 0.05, 1)
    )

Constructs a `PlanConfig` that may stochastically replan at each timestep.
If `plan_at_init` is true, then the initial plan is computed at timestep zero.
"""
function ReplanConfig(
    domain::Domain, planner::Planner;
    plan_at_init::Bool = false,
    prob_replan::Real = 0.1,
    prob_refine::Real = 0.0,
    replan_period::Int = 1,
    rand_budget::Bool = true,
    budget_var::Symbol = default_budget_var(planner),
    budget_dist::Distribution = shifted_neg_binom,
    budget_dist_args::Tuple = (2, 0.05, 1)
)
    step_args = (domain, planner, prob_replan, prob_refine, replan_period,
                 rand_budget, budget_var, budget_dist, budget_dist_args)
    if plan_at_init
        init = step_plan_init
        init_args = (replan_step, step_args, ())
    else
        init = default_plan_init
        init_args = ()
    end
    return PlanConfig(init, init_args, replan_step, step_args)
end

default_budget_var(::Planner) = :max_time
default_budget_var(::ForwardPlanner) = :max_nodes

"""
    replan_step(
        t, plan_state, belief_state, goal_state, act_state,
        domain, planner, prob_replan=0.1, prob_refine=0.0, replan_period=1,
        rand_budget=true, budget_var=:max_nodes,
        budget_dist=shifted_neg_binom, budget_dist_args=(2, 0.95, 1)
    )

Replanning step for fully-ordered planners. After each `replan_period`, a
decision is made to either replan from scratch or refine the previous search
solution. If `rand_budget` is true, replanning or refinement is performed up to
a randomly sampled maximum resource budget.
"""
@gen function replan_step(
    t::Int, plan_state::PlanState, belief_state::State, goal_state, act_state,
    domain::Domain, planner::Planner,
    prob_replan::Real=0.1,
    prob_refine::Real=0.0,
    replan_period::Int=1,
    rand_budget::Bool=true,
    budget_var::Symbol=:max_nodes,
    budget_dist::Distribution=shifted_neg_binom,
    budget_dist_args::Tuple=(2, 0.05, 1)
)   
    spec = convert(Specification, goal_state)
    if must_plan(plan_state, spec)
        # Plan from scratch if no solution exists or goal changes
        prob_replan = 1.0
        prob_refine = 0.0
    elseif must_replan(:unplanned, t, plan_state, belief_state)
        # Enforce recomputation if action is unplanned
        prob_recompute = prob_replan + prob_refine
        if prob_recompute <= 0.0 || !has_search_node(plan_state, belief_state)
            prob_replan = 1.0
            prob_refine = 0.0
        else
            prob_replan = prob_replan / prob_recompute
            prob_refine = prob_refine / prob_recompute
        end
    elseif t % replan_period != 0
        # Otherwise only replan/refine periodically
        prob_refine = 0.0
        prob_replan = 0.0
    end
    # Sample whether to replan or refine
    probs = [1-(prob_replan+prob_refine), prob_replan, prob_refine]
    replan = {:replan} ~ categorical(probs)
    if rand_budget # Sample planning resource budget
        budget = {:budget} ~ budget_dist(budget_dist_args...)
    end
    # Decide whether to replan or refine
    if replan == 1 # Return original plan
        return plan_state
    elseif replan == 2 # Replan from the current belief state
        if rand_budget # Set new resource budget
            planner = copy(planner)
            setproperty!(planner, budget_var, budget)
        end
        # Compute and return new plan
        sol = planner(domain, belief_state, spec)
        return PlanState(t, sol, spec)
    elseif replan == 3 # Refine existing solution
        if rand_budget # Set new resource budget
            planner = copy(planner)
            setproperty!(planner, budget_var, budget)
        end
        # Refine existing solution
        sol = copy(plan_state.sol)
        refine!(sol, planner, domain, belief_state, spec)
        return PlanState(plan_state.init_step, sol, spec)
    end
end

"""
    ReplanPolicyConfig(
        domain::Domain, planner::Planner;
        plan_at_init::Bool = false,
        prob_replan::Real = 0.05,
        prob_refine::Real = 0.2,
        replan_period::Int = 1,
        replan_cond::Symbol = :unplanned,
        rand_budget::Bool = true,
        budget_var::Symbol = default_budget_var(planner),
        budget_dist::Distribution = shifted_neg_binom,
        budget_dist_args::Tuple = (2, 0.05, 1)
    )

Constructs a `PlanConfig` that may stochastically recompute or refine a policy
at regular intervals (controlled by `replan_period`). If `plan_at_init` is true,
then the initial plan is computed at timestep zero.
"""
function ReplanPolicyConfig(
    domain::Domain, planner::Planner;
    plan_at_init::Bool = false,
    prob_replan::Real = 0.05,
    prob_refine::Real = 0.2,
    replan_period::Int = 1,
    replan_cond::Symbol = :unplanned,
    rand_budget::Bool = true,
    budget_var::Symbol = default_budget_var(planner),
    budget_dist::Distribution = shifted_neg_binom,
    budget_dist_args::Tuple = (2, 0.05, 1)
)
    step_args = (
        domain, planner, prob_replan, prob_refine, replan_period, replan_cond,
        rand_budget, budget_var, budget_dist, budget_dist_args
    )
    if plan_at_init
        init = step_plan_init
        init_args = (replan_policy_step, step_args, ())
    else
        init = default_plan_init
        init_args = ()
    end
    return PlanConfig(init, init_args, replan_policy_step, step_args)
end

default_budget_var(::RealTimeDynamicPlanner) = :max_depth
default_budget_var(::RealTimeHeuristicSearch) = :max_nodes
default_budget_var(::AlternatingRealTimeHeuristicSearch) = :max_nodes

"""
    replan_policy_step(
        t, plan_state, belief_state, goal_state, act_state,
        domain, planner, prob_replan=0.05, prob_refine=0.2, replan_period=1,
        replan_cond=:unplanned, rand_budget=true, budget_var=:max_nodes,
        budget_dist=shifted_neg_binom, budget_dist_args=(2, 0.95, 1)
    )

Replanning step for policy-based planners. After each `replan_period`, a
decision is made whether to refine the existing policy or replan from scratch.
If `rand_budget` is true, policy computation or refinement is performed up to
a randomly sampled maximum resource budget.
"""
@gen function replan_policy_step(
    t::Int, plan_state::PlanState, belief_state::State, goal_state, act_state,
    domain::Domain, planner::Planner,
    prob_replan::Real=0.05,
    prob_refine::Real=0.2,
    replan_period::Int=1,
    replan_cond::Symbol=:unplanned,
    rand_budget::Bool=true,
    budget_var::Symbol=:max_nodes,
    budget_dist::Distribution=shifted_neg_binom,
    budget_dist_args::Tuple=(2, 0.05, 1)
)
    spec = convert(Specification, goal_state)
    if must_plan(plan_state, spec)
        # Plan from scratch if no solution exists or goal changes
        prob_replan = 1.0
        prob_refine = 0.0
    elseif must_replan(replan_cond, t, plan_state, belief_state)
        # Enforce recomputation if action is unplanned / uncached
        prob_recompute = prob_replan + prob_refine
        if prob_recompute <= 0.0
            prob_replan = 1.0
            prob_refine = 0.0
        else
            prob_replan = prob_replan / prob_recompute
            prob_refine = prob_refine / prob_recompute
        end
    elseif t % replan_period != 0
        # Otherwise only replan/refine periodically
        prob_refine = 0.0
        prob_replan = 0.0
    end
    # Sample whether to replan or refine
    probs = [1-(prob_replan+prob_refine), prob_replan, prob_refine]
    replan = {:replan} ~ categorical(probs)
    if rand_budget # Sample planning resource budget
        budget = {:budget} ~ budget_dist(budget_dist_args...)
    end
    # Decide whether to replan or refine
    if replan == 1 # Return original plan
        return plan_state
    elseif replan == 2 # Replan from the current belief state
        if rand_budget # Set new resource budget
            planner = copy(planner)
            setproperty!(planner, budget_var, budget)
        end
        # Compute and return new plan
        sol = planner(domain, belief_state, spec)
        return PlanState(t, sol, spec)
    elseif replan == 3 # Refine existing solution
        if rand_budget # Set new resource budget
            planner = copy(planner)
            setproperty!(planner, budget_var, budget)
        end
        # Refine existing solution
        sol = copy(plan_state.sol)
        refine!(sol, planner, domain, belief_state, spec)
        return PlanState(plan_state.init_step, sol, spec)
    end
end

"""
    ReplanMixturePolicyConfig(
        domain::Domain, planner::Planner;
        plan_at_init::Bool = false,
        prob_replan::Real = 0.05,
        prob_refine::Real = 0.2,
        replan_period::Int = 1,
        replan_cond::Symbol = :unplanned,
        budget_var::Symbol = default_budget_var(planner),
        budget_dist_support = [8, 16, 32, 64, 128],
        budget_dist_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    )

Constructs a `PlanConfig` that may stochastically recompute or refine a policy
at regular intervals (controlled by `replan_period`), with marginalization 
over the search budget that exploits incremental planning for efficiency.
This configuration should be used with a `ReplanMixtureActConfig` only.
    
When refining the previous policy, a delayed sample of the last search budget
is drawn, before mixing over the new search budget. A delayed sample is also
drawn if one of the previous sub-policies does not satisfy the replanning
condition. If `plan_at_init` is true, then the initial policy is computed
at timestep zero.
"""
function ReplanMixturePolicyConfig(
    domain::Domain, planner::Planner;
    plan_at_init::Bool = false,
    prob_replan::Real = 0.05,
    prob_refine::Real = 0.2,
    replan_period::Int = 1,
    replan_cond::Symbol = :unplanned,
    budget_var::Symbol = default_budget_var(planner),
    budget_dist_support::AbstractVector{<:Integer} = [8, 16, 32, 64, 128],
    budget_dist_probs::AbstractVector{<:Real} = [0.2, 0.2, 0.2, 0.2, 0.2]
)
    step_args = (
        domain, planner, prob_replan, prob_refine, replan_period, replan_cond,
        budget_var, budget_dist_support, budget_dist_probs
    )
    metadata = (replan=0, budget_idx=0, prev_budget_probs=budget_dist_probs)
    if plan_at_init
        init = step_plan_init
        init_args = (replan_mixture_policy_step, step_args, metadata)
    else
        init = default_plan_init
        init_args = (metadata,)
    end
    return PlanConfig(init, init_args, replan_mixture_policy_step, step_args)
end

"""
    replan_mixture_policy_step(
        t, plan_state, belief_state, goal_state, act_state,
        domain, planner, prob_replan, prob_refine, replan_period, replan_cond,
        budget_var, budget_dist_support, budget_dist_probs
    )

Replanning mixture model for policy-based planners, with marginalization over
the search budget that exploits incremental planning for efficiency.

After each `replan_period`, a decision is made whether to refine the existing
policy or replan from scratch. When refining, a delayed sample of the last
search budget is drawn, before mixing over the new search budget. A delayed
sample is also drawn if one of the previous sub-policies does not satisfy the
replanning condition.
"""
@gen function replan_mixture_policy_step(
    t::Int, plan_state::PlanState, belief_state::State, goal_state, act_state,
    domain::Domain, planner::Planner,
    prob_replan::Real=0.05,
    prob_refine::Real=0.2,
    replan_period::Int=1,
    replan_cond::Symbol=:unplanned,
    budget_var::Symbol=:max_nodes,
    budget_dist_support::AbstractVector{<:Integer} = [8, 16, 32, 64, 128],
    budget_dist_probs::AbstractVector{<:Real} = [0.2, 0.2, 0.2, 0.2, 0.2]
)
    # Update mixture weights (i.e budget probs) based on most recent action
    if plan_state.sol isa NullSolution || !(act_state isa ActState) 
        budget_probs = budget_dist_probs
    else
        act_logprobs = act_state.metadata.act_logprobs
        prev_budget_probs = plan_state.sol.weights
        budget_probs = softmax(log.(prev_budget_probs) .+ act_logprobs)
    end
    budget_idx = 0 # Set budget index to 0 when there is no sampling
    budget_sampled = false
    spec = convert(Specification, goal_state)
    if must_plan(plan_state, spec)
        # Plan with certainty if no solution exists or goal changes
        prob_replan = 1.0
        prob_refine = 0.0
    elseif must_replan(replan_cond, t, plan_state, belief_state)
        # Draw delayed sample of budget index from previous replanning step
        budget_idx = {:forced_budget_idx} ~ categorical(budget_probs)
        selected_sol = plan_state.sol.policies[forced_budget_idx]
        # Collapse budget probabilities to sampled index
        budget_sampled = true
        budget_probs = zero(budget_probs)
        budget_probs[budget_idx] = 1.0
        # Enforce recomputation if action is unplanned / uncached
        if must_replan(replan_cond, t, prev_sol, belief_state)
            prob_recompute = prob_replan + prob_refine
            if prob_recompute <= 0.0
                prob_replan = 1.0
                prob_refine = 0.0
            else
                prob_replan = prob_replan / prob_recompute
                prob_refine = prob_refine / prob_recompute
            end
        elseif t % replan_period != 0
            prob_refine = 0.0
            prob_replan = 0.0
        end
    elseif t % replan_period != 0
        # Otherwise only replan/refine periodically
        prob_refine = 0.0
        prob_replan = 0.0
    end
    # Sample whether to replan or refine
    probs = [1-(prob_replan+prob_refine), prob_replan, prob_refine]
    replan = {:replan} ~ categorical(probs)
    # Decide whether to replan or refine
    if replan == 1 # Return original plan with updated mixture weights
        sol = MixturePolicy(plan_state.sol.policies, budget_probs)
        metadata = (replan=replan, prev_weights=budget_probs,
                    budget_idx=budget_idx)
        return PlanState(plan_state.init_step, sol, plan_state.spec, metadata)
    elseif replan == 2 # Replan from the current belief state
        # Incrementally compute fresh solutions for all search budgets
        @assert issorted(budget_dist_support)
        planner = copy(planner)
        setproperty!(planner, budget_var, first(budget_dist_support))
        init_subsol = planner(domain, belief_state, spec)
        subsols = [init_subsol]
        for i in 2:length(budget_dist_support)
            budget_diff = budget_dist_support[i] - budget_dist_support[i-1]
            setproperty!(planner, budget_var, budget_diff)
            subsol = refine(last(subsols), planner, domain, belief_state, spec)
            push!(subsols, subsol)
        end
        # Reset mixture weights to prior budget probabilities
        sol = MixturePolicy(subsols, budget_dist_probs)
        metadata = (replan=replan, prev_weights=budget_probs,
                    budget_idx=budget_idx)
        return PlanState(t, sol, spec, metadata)
    elseif replan == 3 # Refine existing solution
        # Draw delayed sample of budget index if not already drawn
        if !budget_sampled
            budget_idx = {:refine_budget_idx} ~ categorical(budget_probs)
            selected_sol = plan_state.sol.policies[budget_idx]
            # Collapse budget probabilities to sampled index
            budget_sampled = true
            budget_probs = zero(budget_probs)
            budget_probs[budget_idx] = 1.0
        end
        # Refine selected previous solution for each possible budget
        planner = copy(planner)
        @assert issorted(budget_dist_support)
        subsols = Vector{eltype(selected_sol)}()
        for i in 1:length(budget_dist_support)
            budget_diff =
                budget_dist_support[i] - get(budget_dist_support, i-1, 0)
            setproperty!(planner, budget_var, budget_diff)
            prev_sol = i == 1 ? selected_sol : subsols[i-1]
            subsol = refine(prev_sol, planner, domain, belief_state, spec)
            push!(subsols, subsol)
        end
        # Reset mixture weights to prior budget probabilities
        sol = MixturePolicy(subsols, budget_dist_probs)
        metadata = (replan=replan, prev_weights=budget_probs,
                    budget_idx=budget_idx)
        return PlanState(plan_state.init_step, sol, spec, metadata)
    end
end
