## Model visualization code ##

export TraceRenderer
export render_trace, render_trace!
export anim_trace, anim_trace!

using Makie: Observable
using PDDLViz: Animation, maybe_observe, is_displayed

"""
    TraceRenderer(renderer::PDDLViz.Renderer; kwargs...)

Renderer for visualizing world model traces, using PDDLViz to render
environment states and planning solutions in a PDDL domain.

# Arguments

$(TYPEDFIELDS)
"""
@kwdef mutable struct TraceRenderer{R <: Renderer}
    "PDDLViz renderer."
    renderer::R
    "Whether to show observed or true environment states."
    show_observed::Bool = false
    "Whether to show the past state trajectory."
    show_past::Bool = false
    "Whether to show the future state trajectory."
    show_future::Bool = false
    "Whether to show the current planning solution."
    show_sol::Bool = false
    "Number of past states to show."
    n_past::Int = 10
    "Number of future states to show."
    n_future::Int = 10
    "Past trajectory rendering options."
    past_options::Dict{Symbol, Any} = Dict()
    "Future trajectory rendering options."
    future_options::Dict{Symbol, Any} = Dict()
    "Solution rendering options."
    sol_options::Dict{Symbol, Any} = Dict()
end

TraceRenderer(renderer::R; kwargs...) where {R <: Renderer} =
    TraceRenderer{R}(; renderer=renderer, kwargs...)

"""
    (r::TraceRenderer)(domain, world_trace, [t]; kwargs...)

Renders a trace sampled from [`world_model`](@ref) at time `t`. Both
`world_trace` and `t` can be `Observable` values. If `t` is not provided,
the last timestep of `world_trace` is used.
"""
(r::TraceRenderer)(domain::Domain, world_trace, t; kwargs...) =
    render_trace(r, domain, world_trace, t; kwargs...)

(r::TraceRenderer)(domain::Domain, world_trace; kwargs...) =
    render_trace(r, domain, world_trace; kwargs...)

"""
    (r::TraceRenderer)(canvas, domain, world_trace, [t]; kwargs...)

Renders a trace sampled from [`world_model`](@ref) at time `t` on an existing
canvas. Both `world_trace` and `t` can be `Observable` values. If `t` is not
provided, the last timestep of `world_trace` is used.
"""
(r::TraceRenderer)(canvas::Canvas, domain::Domain, world_trace, t; kwargs...) =
    render_trace!(canvas, r, domain, world_trace, t; kwargs...)

(r::TraceRenderer)(canvas::Canvas, domain::Domain, world_trace; kwargs...) =
    render_trace!(canvas, r, domain, world_trace; kwargs...)


"""
    render_trace(renderer::TraceRenderer,
                 domain::Domain, world_trace, [t]; kwargs...)

Renders a trace sampled from [`world_model`](@ref) at time `t`. Both
`world_trace` and `t` can be `Observable` values. If `t` is not provided,
the last timestep of `world_trace` is used.
"""
function render_trace(
    renderer::TraceRenderer, domain::Domain, world_trace, t; kwargs...
)
    canvas = PDDLViz.new_canvas(renderer.renderer)
    return render_trace!(canvas, renderer, domain, world_trace, t; kwargs...)
end

function render_trace(
    renderer::TraceRenderer, domain::Domain, world_trace; kwargs...
)
    canvas = PDDLViz.new_canvas(renderer.renderer)
    return render_trace!(canvas, renderer, domain, world_trace; kwargs...)
end

"""
    render_trace!(canvas::Canvas, renderer::TraceRenderer,
                  domain::Domain, world_trace, [t]; kwargs...)

Renders a trace sampled from [`world_model`](@ref) at time `t` on an existing
canvas. Both `world_trace` and `t` can be `Observable` values. If `t` is not
provided, the last timestep of `world_trace` is used.
"""
function render_trace!(
    canvas::Canvas, renderer::TraceRenderer,
    domain::Domain, world_trace, t;
    kwargs...
)
    world_trace = maybe_observe(world_trace)
    t = maybe_observe(t)
    return render_trace!(canvas, renderer, domain, world_trace, t; kwargs...)
end

function render_trace!(
    canvas::Canvas, renderer::TraceRenderer,
    domain::Domain, world_trace::Trace;
    kwargs...
)
    world_trace = Observable(world_trace)
    return render_trace!(canvas, renderer, domain, world_trace; kwargs...)
end

function render_trace!(
    canvas::Canvas, renderer::TraceRenderer,
    domain::Domain, world_trace::Observable,
    t::Observable = @lift(get_world_timestep($world_trace));
    interactive::Bool = false, kwargs...
)
    env_state = @lift renderer.show_observed ?
        get_obs_state($world_trace, $t) : get_env_state($world_trace, $t)
    render_state!(canvas, renderer.renderer, domain, env_state; kwargs...)
    # Render past trajectory
    if renderer.show_past
        past_states = @lift begin
            trajectory = renderer.show_observed ?
                get_obs_states($world_trace) : get_env_states($world_trace)
            trajectory[max(1, $t - renderer.n_past + 1):($t + 1)]
        end            
        render_trajectory!(
            canvas, renderer.renderer, domain, past_states;
            renderer.past_options...
        )
    end
    if renderer.show_future || renderer.show_sol
        sol = Observable(get_agent_state(world_trace[], t[]).plan_state.sol)
        onany(world_trace, t) do world_trace, t
            new_sol = get_agent_state(world_trace, t).plan_state.sol
            (new_sol isa NullSolution || new_sol === sol[]) && return
            sol[] = new_sol
        end
        # Render current solution
        if renderer.show_sol && !(sol[] isa NullSolution)
            render_sol!(
                canvas, renderer.renderer, domain, env_state, sol;
                renderer.sol_options...
            )
        end
        # Render future trajectory
        if renderer.show_future 
            plan_state = @lift get_agent_state($world_trace, $t).plan_state
            future_states = Observable([env_state[]])
            onany(env_state, plan_state) do env_state, plan_state
                empty!(future_states[])
                push!(future_states[], env_state)
                for _ in 1:renderer.n_future
                    if plan_state.sol isa NullSolution break end
                    act = best_action(plan_state.sol, env_state)
                    if ismissing(act) break end
                    env_state = transition(domain, env_state, act)
                    push!(future_states[], env_state)
                    if is_goal(plan_state.spec, domain, env_state) break end
                end
                notify(future_states)
            end
            notify(plan_state)
            render_trajectory!(
                canvas, renderer.renderer, domain, future_states;
                renderer.future_options...
            )
        end
    end
    # Add callback for interactivity
    if interactive
        on(events(canvas.figure).keyboardbutton) do event
            # Skip if no keys are pressed
            event.action == Keyboard.press || return
            # Skip if window not in focus
            events(canvas.figure).hasfocus[] || return
            # Update timestep if left or right arrow keys are pressed
            if event.key == Keyboard.left
                t[] = max(0, t[] - 1)
            elseif event.key == Keyboard.right
                max_t = get_world_timestep(world_trace[])
                t[] = min(max_t, t[] + 1)
            end
        end
    end
    return canvas
end

"""
    anim_trace([path], renderer::TraceRenderer, domain, world_trace;
               format="mp4", framerate=5, show=false, record_init=true,
               options...)

    anim_trace!([anim|path], canvas,
                renderer::TraceRenderer, domain, world_trace;
                format="mp4", framerate=5, show=false, record_init=true,
                options...)

Animates a trace sampled from [`world_model`](@ref). Returns an
[`Animation`](@ref) object, which can be saved or displayed.

If an `anim` is provided as the first argument, frames are added to the existing 
animation. Otherwise, a new animation is created. If `path` is provided,
the animation is saved to that file, and `path` is returned.
"""
function anim_trace(
    renderer::TraceRenderer, domain::Domain, world_trace;
    show::Bool=false, kwargs...
)
    canvas = PDDLViz.new_canvas(renderer.renderer)
    return anim_trace!(canvas, renderer, domain, world_trace;
                       show, kwargs...)
end

function anim_trace(path::AbstractString, args...; kwargs...)
    format = lstrip(splitext(path)[2], '.')
    save(path, anim_trace(args...; format=format, kwargs...))
end

function anim_trace!(
    canvas::Canvas, renderer::TraceRenderer,
    domain::Domain, world_trace;
    format="mp4", framerate=5, show::Bool=is_displayed(canvas),
    showrate=framerate, record_init=true, options...
)
    # Display canvas if `show` is true
    show && !is_displayed(canvas) && display(canvas)
    # Initialize animation
    record_args = filter(Dict(options)) do (k, v)
        k in (:compression, :profile, :pixel_format, :loop)
    end
    anim = Animation(canvas.figure; visible=is_displayed(canvas),
                     format, framerate, record_args...)
    # Record animation
    anim_trace!(anim, canvas, renderer, domain, world_trace;
                show, showrate, record_init, options...)
    return anim
end

function anim_trace!(
    anim::Animation, canvas::Canvas, renderer::TraceRenderer,
    domain::Domain, world_trace;
    show::Bool=is_displayed(canvas), showrate=5, record_init=true, options...
)
    # Display canvas if `show` is true
    show && !is_displayed(canvas) && display(canvas)
    # Initialize animation and record initial frame
    max_t = get_world_timestep(world_trace)
    t = Observable(max_t)
    render_trace!(canvas, renderer, domain, world_trace, t; options...)
    t[] = 0
    record_init && recordframe!(anim)
    # Construct recording callback
    function record_callback(canvas::Canvas)
        recordframe!(anim)
        !show && return
        notify(canvas.state)
        sleep(1/showrate)
    end
    # Iterate over rest of trace
    for _t in 1:max_t
        t[] = _t
        record_callback(canvas)
    end
    return anim
end

function anim_trace!(path::AbstractString, args...; kwargs...)
    format = lstrip(splitext(path)[2], '.')
    save(path, anim_trace!(args...; format=format, kwargs...))
end

@doc (@doc anim_trace) anim_trace!
