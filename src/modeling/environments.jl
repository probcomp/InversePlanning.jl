## Environment models ##

export EnvConfig, StaticEnvConfig, PDDLEnvConfig

"""
    EnvConfig <: ModelConfig

Configuration of an environment model, specifying how the environment changes
in response to the agent's actions.

# Fields

$(FIELDS)

# Constructors
"""
struct EnvConfig{T,U,V}
    "Initializer with arguments `(init_args...)`."
    init::T
    "Trailing arguments to initializer."
    init_args::U
    "Transition function with arguments `(t, env_state, act_state, step_args...)`."
    step::GenerativeFunction
    "Trailing arguments to transition function."
    step_args::V
end

# Static environment model #

"""
    StaticEnvConfig(init=nothing, init_args=())

Constructs a `EnvConfig` where the environment state never changes.
"""
function StaticEnvConfig(init=nothing, init_args=())
    return EnvConfig(init, init_args, static_env_step, ())
end

@add_constructor_doc(
    EnvConfig, StaticEnvConfig,
    "Models a static environment that never changes."
)

"""
    static_env_step(t, env_state, action)

Static environment transition that returns previous state unmodified.
"""
@gen static_env_step(t, env_state, action) = env_state


# PDDL environment model #

"""
    PDDLEnvConfig(domain::Domain, init_env_state::State)
    PDDLEnvConfig(domain::Domain, state_prior, state_prior_args=())

Constructs a deterministic `EnvConfig` from a PDDL domain and initial state.
A `state_prior` can also be specified instead of a `state`.
"""
function PDDLEnvConfig(domain::Domain, init, init_args::Tuple=())
    return EnvConfig(init, init_args, pddl_env_step, (domain,))
end

@add_constructor_doc(
    EnvConfig, PDDLEnvConfig,
    "Models a PDDL environment with deterministic dynamics."
)

"""
    pddl_env_step(t, env_state::State, act_state, domain::Domain)

PDDL environment transition function with deterministic dynamics.
"""
@gen function pddl_env_step(t, env_state::State, act_state, domain::Domain)
    act = convert(Term, act_state) # Convert to action term to be safe
    if act.name == DO_SYMBOL && act isa Compound
        act = act.args[1]
    end
    if act.name == Symbol("--") || !available(domain, env_state, act)
        return env_state
    else
        return transition(domain, env_state, act, check=false)
    end
end
