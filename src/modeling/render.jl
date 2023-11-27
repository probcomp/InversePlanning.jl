## Model visualization code ##

export TraceRenderer
export render_trace, render_trace!
export anim_trace, anim_trace!

using Makie: Observable
using PDDLViz: Animation, maybe_observe, is_displayed

"""
    TraceRenderer(renderer::PDDLViz.Renderer; options...)

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
    "Whether to show a title."
    show_title::Bool = true
    "Number of past states to render."
    n_past::Int = 10
    "Number of future states to render."
    n_future::Int = 10
    "Past trajectory rendering options."
    past_options::Dict{Symbol, Any} = Dict()
    "Future trajectory rendering options."
    future_options::Dict{Symbol, Any} = Dict()
    "Solution rendering options."
    sol_options::Dict{Symbol, Any} = Dict()
    "Title rendering function of the form `(trace, t, weight) -> String`."
    title_fn::Function = (trace, t, weight) -> "t = $t"
    "Title rendering options."
    title_options::Dict{Symbol, Any} = Dict(
        :valign => :bottom,
        :fontsize => 20,
        :padding => (0, 0, 5, 0)
    )
end

TraceRenderer(renderer::R; options...) where {R <: Renderer} =
    TraceRenderer{R}(; renderer=renderer, options...)

"""
    (r::TraceRenderer)(domain, world_trace, [t, weight]; options...)

Renders a trace sampled from [`world_model`](@ref) at time `t`, with optional 
`weight` metadata. All of these arguments can be `Observable` values. If `t` is
not provided, the last timestep of `world_trace` is used.
"""
function (r::TraceRenderer)(
    domain::Domain, world_trace, t = nothing, weight = 1.0; options...
)
    return render_trace(r, domain, world_trace, t, weight; options...)
end

"""
    (r::TraceRenderer)(canvas, domain, world_trace, [t, weight]; options...)

Renders a trace sampled from [`world_model`](@ref) at time `t` on an existing
canvas, with optional `weight` metadata. All of these arguments can be
`Observable` values. If `t` is not provided, the last timestep of `world_trace`
is used.
"""
function (r::TraceRenderer)(
    canvas::Canvas, domain::Domain, world_trace, t, weight = 1.0; options...
)
    return render_trace!(canvas, r, domain, world_trace, t, weight; options...)
end

"""
    render_trace(renderer::TraceRenderer,
                 domain::Domain, world_trace, [t, weight]; options...)

Renders a trace sampled from [`world_model`](@ref) at time `t`, with optional 
`weight` metadata. All of these arguments can be `Observable` values. If `t` is
not provided, the last timestep of `world_trace` is used.
"""
function render_trace(
    renderer::TraceRenderer,
    domain::Domain, world_trace, t = nothing, weight = 1.0; options...
)
    canvas = PDDLViz.new_canvas(renderer.renderer)
    return render_trace!(canvas, renderer,
                         domain, world_trace, t, weight; options...)
end

"""
    render_trace!(canvas::Canvas, renderer::TraceRenderer,
                  domain::Domain, world_trace, [t, weight]; options...)

Renders a trace sampled from [`world_model`](@ref) at time `t` on an existing
canvas, with optional `weight` metadata. All of these arguments can be
`Observable` values. If `t` is not provided, the last timestep of `world_trace`
is used.
"""
function render_trace!(
    canvas::Canvas, renderer::TraceRenderer,
    domain::Domain, world_trace, t = nothing, weight = 1.0;
    interactive::Bool = false, options...
)
    # Convert to observables
    world_trace = maybe_observe(world_trace)
    t = isnothing(t) ?
        @lift(get_world_timestep($world_trace)) : maybe_observe(t)
    weight = maybe_observe(weight)
    # Render current environment or observation state
    env_state = @lift renderer.show_observed ?
        get_obs_state($world_trace, $t) : get_env_state($world_trace, $t)
    render_state!(canvas, renderer.renderer, domain, env_state; options...)
    # Render past trajectory
    if renderer.show_past
        past_states = @lift begin
            trajectory = renderer.show_observed ?
                get_obs_states($world_trace) : get_env_states($world_trace)
            trajectory[max(1, $t - renderer.n_past + 1):($t + 1)]
        end
        past_options = copy(renderer.past_options)
        map!(values(past_options)) do f
            f isa Function ? @lift(f($world_trace, $t, $weight)) : f
        end
        render_trajectory!(
            canvas, renderer.renderer, domain, past_states;
            past_options...
        )
    end
    if renderer.show_future || renderer.show_sol
        # Render solution that starts at current environment state
        if renderer.show_sol
            t_max = get_world_timestep(world_trace[])
            t_next = min(t_max, t[]+1)
            sol = Observable(get_agent_state(world_trace[], t_next).plan_state.sol)
            onany(world_trace, t) do world_trace, t
                t_max = get_world_timestep(world_trace)
                t_next = min(t_max, t+1)
                new_sol = get_agent_state(world_trace, t_next).plan_state.sol
                (new_sol isa NullSolution || new_sol === sol[]) && return
                sol[] = new_sol
            end
            if !(sol[] isa NullSolution)
                sol_env_state = @lift get_env_state($world_trace, $t)
                sol_options = copy(renderer.sol_options)
                map!(values(sol_options)) do f
                    f isa Function ? @lift(f($world_trace, $t, $weight)) : f
                end    
                render_sol!(
                    canvas, renderer.renderer, domain, sol_env_state, sol;
                    sol_options...
                )
            end
        end
        # Render planned future trajectory
        if renderer.show_future
            future_states = Observable([env_state[]])
            onany(world_trace, t) do world_trace, t
                empty!(future_states[])
                t_max = get_world_timestep(world_trace)
                t_next = min(t_max, t+1)
                env_state = get_env_state(world_trace, t)
                plan_state = get_agent_state(world_trace, t_next).plan_state
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
            notify(world_trace)
            future_options = copy(renderer.future_options)
            map!(values(future_options)) do f
                f isa Function ? @lift(f($world_trace, $t, $weight)) : f
            end
            render_trajectory!(
                canvas, renderer.renderer, domain, future_states;
                future_options...
            )
        end
    end
    # Add title
    if renderer.show_title
        title = @lift renderer.title_fn($world_trace, $t, $weight)
        title_options = copy(renderer.title_options)
        map!(values(title_options)) do f
            f isa Function ? @lift(f($world_trace, $t, $weight)) : f
        end
        label = Label(canvas.layout[1, 1:end, Top()], title; title_options...)
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
                t_max = get_world_timestep(world_trace[])
                t[] = min(t_max, t[] + 1)
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
    show::Bool=false, options...
)
    canvas = PDDLViz.new_canvas(renderer.renderer)
    return anim_trace!(canvas, renderer, domain, world_trace;
                       show, options...)
end

function anim_trace(path::AbstractString, args...; options...)
    format = lstrip(splitext(path)[2], '.')
    save(path, anim_trace(args...; format=format, options...))
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
    t_max = get_world_timestep(world_trace)
    t = Observable(t_max)
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
    for _t in 1:t_max
        t[] = _t
        record_callback(canvas)
    end
    return anim
end

function anim_trace!(path::AbstractString, args...; options...)
    format = lstrip(splitext(path)[2], '.')
    save(path, anim_trace!(args...; format=format, options...))
end

@doc (@doc anim_trace) anim_trace!
