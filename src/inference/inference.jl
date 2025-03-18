import Gen: ParticleFilterState

export SequentialInversePlanSearch, SIPS
export sips_init, sips_run, sips_step!

include("utils.jl")
include("choicemaps.jl")
include("rejuvenate.jl")
include("callbacks.jl")

"""
    SequentialInversePlanSearch(world_config::WorldConfig; options...)
    SIPS(world_config::WorldConfig; options...)

Constructs a sequential inverse plan search (SIPS) particle filtering algorithm
for the agent-environment model defined by `world_config`.

# Arguments

$(TYPEDFIELDS)
"""
@kwdef struct SequentialInversePlanSearch{W <: WorldConfig, P, K}
    "Configuration of the world model to perform inference over."
    world_config::W
    "Proposal generative function `q(trace, obs)` for update step (default: `nothing`)"
    step_proposal::P = nothing
    "Number of children extended from each parent particle in the update step (default: `1`)."
    step_resize_count::Int = 1
    "Downsampling method after resizing: `[:blockwise, :multinomial, :residual, :stratified]`."
    step_resize_method::Symbol = :blockwise
    "Trigger condition for resampling particles: `[:none, :periodic, :always, :ess]` (default: `:none`)."
    resample_cond::Symbol = :none
    "Resampling method: `[:multinomial, :residual, :stratified]` (default: `:residual`)."
    resample_method::Symbol = :residual
    "Trigger condition for rejuvenating particles `[:none, :periodic, :always, :ess] (default: `:none`)`."
    rejuv_cond::Symbol = :none
    "Rejuvenation kernel (default: `NullKernel()`)."
    rejuv_kernel::K = NullKernel()
    "Effective sample size threshold fraction for resampling and/or rejuvenation (default: `0.25`)."
    ess_threshold::Float64 = 0.25
    "Period for resampling and rejuvenation (default: `1`)."
    period::Int = 1
end

const SIPS = SequentialInversePlanSearch
@doc (@doc SequentialInversePlanSearch) SIPS

SIPS(world_config; kwargs...) = SIPS(; world_config=world_config, kwargs...)

"""
    (::SIPS)(n_particles, observations, [timesteps]; kwargs...)
    (::SIPS)(n_particles, t_obs_iter; kwargs...)

Run a SIPS particle filter given a series of observation choicemaps and
timesteps, or an iterator over timestep-observation pairs. Returns the final
particle filter state.
"""
(sips::SIPS)(args...; kwargs...) = sips_run(sips, args...; kwargs...)

"Gets current model timestep from a particle filter state."
function get_model_timestep(pf_state::ParticleFilterState)
    return Gen.get_args(get_traces(pf_state)[1])[1]
end

"Decides whether to resample or rejuvenate based on trigger conditions."
function sips_trigger_cond(sips::SIPS, cond::Symbol,
                           t::Int, pf_state::ParticleFilterState)
    if cond == :always
        return true
    elseif cond == :periodic
        return mod(t, sips.period) == 0
    elseif cond == :ess
        n_particles = length(get_traces(pf_state))
        return get_ess(pf_state) < (n_particles * sips.ess_threshold)
    end
    return false
end

"""
    sips_init(sips::SIPS, n_particles::Int; kwargs...)

SIPS particle filter initialization. Constructs and returns a
`ParticleFilterState` by sampling traces from `world_model`. Initialization 
can be customized by passing keyword arguments to `sips_init`.

# Keyword Arguments

- `init_timestep = 0`: Initial timestep.
- `init_obs = EmptyChoiceMap()`: Initial observation choicemap.
- `init_strata = nothing`: Initial strata for stratified initialization.
- `init_proposal = nothing`: Initial proposal distribution.
- `init_proposal_args = ()`: Arguments to initial proposal.
- `callback = nothing`: Callback function to run at initialization.
"""
function sips_init(
    sips::SIPS, n_particles::Int;
    init_timestep::Int = 0,
    init_obs::ChoiceMap=EmptyChoiceMap(),
    init_strata=nothing,
    init_proposal=nothing,
    init_proposal_args=(),
    callback = nothing
)
    args = (init_timestep, sips.world_config)
    if isnothing(init_strata)
        if isnothing(init_proposal)
            pf_state = pf_initialize(world_model, args, init_obs, n_particles)
        else
            pf_state = pf_initialize(world_model, args, init_obs,
                                     init_proposal, init_proposal_args,
                                     n_particles)
        end       
    else
        if isnothing(init_proposal)
            pf_state = pf_initialize(world_model, args, init_obs, init_strata,
                                     n_particles)
        else
            pf_state = pf_initialize(world_model, args, init_obs, init_strata,
                                     init_proposal, init_proposal_args,
                                     n_particles)
        end
    end
    isnothing(callback) || callback(init_timestep, init_obs, pf_state)
    return pf_state
end

"""
    sips_step!(pf_state, sips::SIPS, t::Int, observations::ChoiceMap;
               callback = nothing, cb_schedule = :step)

SIPS particle filter step. Updates the particle filter state by extending 
the traces to timestep `t`, conditioning on new `observations`, and optionally
resampling and rejuvenating.

# Keyword Arguments

- `callback = nothing`: Callback function to run at each step.
- `cb_schedule = :step`: Callback schedule: `[:step, :substep]`. If `:step`,
  the callback is run once per step. If `:substep`, the callback is run after
  each substep (updating, resampling, and rejuvenating).
"""
function sips_step!(
    pf_state::ParticleFilterState, sips::SIPS,
    t::Int, observations::ChoiceMap=EmptyChoiceMap();
    callback = nothing, cb_schedule::Symbol = :step
)
    # Optionally resize particle filter
    n_particles = length(pf_state.traces)
    if sips.step_resize_count > 1
        pf_replicate!(pf_state, sips.step_resize_count)
    end
    # Update particle filter with new observations
    argdiffs = (UnknownChange(), NoChange())
    if isnothing(sips.step_proposal)
        pf_update!(pf_state, (t, sips.world_config), argdiffs, observations)
    else
        pf_update!(pf_state, (t, sips.world_config), argdiffs, observations,
                   sips.step_proposal, (observations,))
    end
    # If we replicated particles, resize down to original number
    if sips.step_resize_count > 1
        if sips.step_resize_method == :blockwise
            pf_dereplicate!(pf_state, sips.step_resize_count, method=:sample)
        else
            pf_resize!(pf_state, n_particles, sips.step_resize_method)
        end
    end
    if cb_schedule == :substep
        isnothing(callback) || callback(t, observations, pf_state)
    end
    # Optionally resample
    if sips_trigger_cond(sips, sips.resample_cond, t, pf_state)
        pf_resample!(pf_state, sips.resample_method)
        if cb_schedule == :substep
            isnothing(callback) || callback(t, observations, pf_state)
        end
    end
    # Optionally rejuvenate
    if sips_trigger_cond(sips, sips.rejuv_cond, t, pf_state)
        pf_rejuvenate!(pf_state, sips.rejuv_kernel)
        if cb_schedule == :substep
            isnothing(callback) || callback(t, observations, pf_state)
        end
    end
    # Run per-step callback
    if cb_schedule == :step
        isnothing(callback) || callback(t, observations, pf_state)
    end
    return pf_state
end

"""
    sips_run(sips, n_particles, observations, [timesteps]; kwargs...)
    sips_run(sips, n_particles, t_obs_iter; kwargs...)

Run a SIPS particle filter, given a series of observations and timesteps, or
an iterator over timestep-observation pairs. Returns the final particle filter
state.

# Keyword Arguments

- `init_args = Dict{Symbol, Any}()`: Keyword arguments to `sips_init`.
- `callback = nothing`: Callback function to run at each timestep.
- `cb_schedule = :step`: Callback schedule: `[:step, :substep]`. If `:step`,
  the callback is run once per step. If `:substep`, the callback is run after
  each substep (updating, resampling, and rejuvenating).
"""
function sips_run(
    sips::SIPS, n_particles::Int, t_obs_iter;
    init_args = Dict{Symbol, Any}(),
    callback = nothing,
    cb_schedule::Symbol = :step
)
    # Extract initial observation from iterator
    if first(t_obs_iter)[1] == 0
        _, init_obs = first(t_obs_iter)
        if !(init_args isa Dict{Symbol, Any})
            init_args = Dict{Symbol, Any}(pairs(init_args))
        end
        init_args[:init_timestep] = 0
        init_args[:init_obs] = init_obs
        t_obs_iter = Iterators.drop(t_obs_iter, 1)
    end
    # Initialize particle filter
    pf_state = sips_init(sips, n_particles; callback, init_args...)
    # Iterate over timesteps and observations
    for (t::Int, obs::ChoiceMap) in t_obs_iter
        pf_state = sips_step!(pf_state, sips, t, obs;
                              callback, cb_schedule)
    end
    # Return final particle filter state
    return pf_state
end

function sips_run(
    sips::SIPS, n_particles::Int,
    observations::AbstractVector{<:ChoiceMap},
    timesteps=nothing;
    kwargs...
)
    if isnothing(timesteps) && !isempty(observations)
        init_obs = first(observations)
        if has_submap(init_obs, :init) && !has_submap(init_obs, :timestep)
            timesteps = 0:length(observations)-1
        else
            timesteps = 1:length(observations)
        end
    end
    return sips_run(sips, n_particles, zip(timesteps, observations); kwargs...)
end
