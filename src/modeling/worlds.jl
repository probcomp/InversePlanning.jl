export WorldState, WorldConfig
export world_init, world_step, world_model
export get_world_timestep, get_world_states, get_world_state
export get_agent_states, get_act_states, get_env_states, get_obs_states
export get_agent_state, get_act_state, get_env_state, get_obs_state

"""
    WorldState

Represents the state the world (agent + environment) at a point in time.

# Fields

$(FIELDS)
"""
struct WorldState{A,X,E,O}
    "Agent state."
    agent_state::A
    "Action state."
    act_state::X
    "Environment state."
    env_state::E
    "Observed state."
    obs_state::O
end

"""
    WorldConfig

Configuration of a world model, including agent, environment, and observation
model configurations.

# Fields

$(FIELDS)    
"""
@kwdef struct WorldConfig{A,E,O}
  "Agent model configuration."
  agent_config::A = AgentConfig()
  "Environment model configuration."
  env_config::E = StaticEnvConfig()
  "Observation model configuration."
  obs_config::O = PerfectObsConfig()
end

"""
    world_init(config::WorldConfig)

Initialize world state by sampling from the initializers.
"""
@gen function world_init(config)
    # (Optionally) sample from prior over world configurations
    config::WorldConfig = {:world_config} ~ maybe_sample(config)
    # (Optionally) sample from priors over sub-model configurations
    @unpack agent_config, env_config, obs_config = config
    agent_config = {:agent_config} ~ maybe_sample(agent_config)
    env_config = {:env_config} ~ maybe_sample(env_config)
    obs_config = {:obs_config} ~ maybe_sample(obs_config)
    act_config = agent_config.act_config
    # Initialize environment, observation, agent, and action states
    env_state = {:env} ~ maybe_sample(env_config.init, env_config.init_args)
    obs_state = {:obs} ~ maybe_sample(
        obs_config.init, (env_state, obs_config.init_args...)
    )
    agent_state = {:agent} ~ agent_init(agent_config, env_state)
    act_state = {:act} ~ maybe_sample(
        act_config.init, (agent_state, env_state, act_config.init_args...)
    )
    # Return sampled state and configuration
    state = WorldState(agent_state, act_state, env_state, obs_state)
    config = WorldConfig(agent_config, env_config, obs_config)
    return state, config
end

"""
    world_step(t::Int, world_state::WorldState, config::WorldConfig)

Models transition dynamics at step `t` in a world model.
"""
@gen function world_step(t::Int, world_state::WorldState, config::WorldConfig)
    # Unpack agent state and configuration
    @unpack agent_state, act_state, env_state, obs_state = world_state
    @unpack agent_config, env_config, obs_config = config
    # Unpack sub-configurations
    @unpack act_config = agent_config
    act_step, act_step_args = act_config.step, act_config.step_args
    env_step, env_step_args = env_config.step, env_config.step_args
    obs_step, obs_step_args = obs_config.step, obs_config.step_args
    # Advance the agent by one step
    agent_state = {:agent} ~ agent_step(t, agent_state, act_state, env_state,
                                        agent_config)
    # Sample the agent's actions in response to the previous environment state
    act_state = {:act} ~ act_step(t, act_state, agent_state, env_state,
                                  act_step_args...)
    # Run the environment transition dynamics forward
    env_state = {:env} ~ env_step(t, env_state, act_state,
                                  env_step_args...)
    # Sample an observation, given the current environment state
    obs_state = {:obs} ~ obs_step(t, obs_state, env_state,
                                  obs_step_args...)
    # Pass the full state to the next step
    return WorldState(agent_state, act_state, env_state, obs_state)
end

"Unfold combinator applied to `world_step`."
world_unfold = Unfold(world_step)

"""
    world_model(n_steps::Int, config::WorldConfig)

Models the time-evolution of a world (agent + environment) for `n_steps`.
"""
@gen (static) function world_model(n_steps::Int, config)
    world_state, config = {:init} ~ world_init(config)
    world_trajectory = {:timestep} ~ world_unfold(n_steps, world_state, config)
    return world_trajectory
end

"""
    get_world_timestep(world_trace::Trace)

Returns the timestep of a world trace.
"""
get_world_timestep(world_trace::Trace) = Gen.get_args(world_trace)[1]

"""
    get_world_states(world_trace::Trace, include_init=true)

Extracts world states from a world trace. Includes the initial state by default.
"""
function get_world_states(world_trace::Trace, include_init::Bool=true)
    if include_init
        return [world_trace[:init][1]; get_retval(world_trace)]
    else
        return collect(get_retval(world_trace))
    end
end

"""
    get_world_state(world_trace::Trace, t::Int)

Extracts a world state from a world trace at time `t`.
"""
function get_world_state(world_trace::Trace, t::Int)
    return t == 0 ? world_trace[:init][1] : get_retval(world_trace)[t]
end

"""
    get_agent_states(world_trace::Trace, include_init=false)

Extracts agent states from a world trace. Excludes the initial state by default.
"""
get_agent_states(world_trace::Trace, include_init::Bool=true) =
    getproperty.(get_world_states(world_trace, include_init), :agent_state)

"""
    get_agent_state(world_trace::Trace, t::Int)

Extracts an agent state from a world trace at time `t`.
"""
get_agent_state(world_trace::Trace, t::Int) =
    getproperty(get_world_state(world_trace, t), :agent_state)

"""
    get_act_states(world_trace::Trace, include_init=false)

Extracts action states from a world trace. Excludes the initial state by default.
"""
get_act_states(world_trace::Trace, include_init::Bool=false) =
    getproperty.(get_world_states(world_trace, include_init), :act_state)

"""
    get_act_state(world_trace::Trace, t::Int)

Extracts an action state from a world trace at time `t`.
"""
get_act_state(world_trace::Trace, t::Int) =
    getproperty(get_world_state(world_trace, t), :act_state)

"""
    get_env_states(world_trace::Trace, include_init=true)

Extracts environment states from a world trace. Includes the initial state
by default.
"""
get_env_states(world_trace::Trace, include_init::Bool=true) =
    getproperty.(get_world_states(world_trace, include_init), :env_state)

"""
    get_env_state(world_trace::Trace, t::Int)

Extracts an environment state from a world trace at time `t`.
"""
get_env_state(world_trace::Trace, t::Int) =
    getproperty(get_world_state(world_trace, t), :env_state)

"""
    get_obs_states(world_trace::Trace, include_init=true)

Extracts the observation states from a world trace. Includes the initial state
by default.
"""
get_obs_states(world_trace::Trace, include_init::Bool=true) =
    getproperty.(get_world_states(world_trace, include_init), :obs_state)

"""
    get_obs_state(world_trace::Trace, t::Int)

Extracts an observation state from a world trace at time `t`.
"""
get_obs_state(world_trace::Trace, t::Int) =
    getproperty(get_world_state(world_trace, t), :obs_state)
