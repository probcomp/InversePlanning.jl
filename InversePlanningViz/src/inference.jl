## Inference visualization code ##

export PlotCallback, BarPlotCallback, SeriesPlotCallback
export RenderStateCallback, RenderTracesCallback, RecordCallback

import DataStructures: OrderedDict

"""
    PlotCallback

Callback that plots data from the particle filter state at each timestep. Can
be configured to plot data produced by a logger function (similar to
[`DataLoggerCallback`](@ref)), or data that has already been logged by 
a [`DataLoggerCallback`](@ref).
"""
mutable struct PlotCallback{P, T, U} <: SIPSCallback
    plot_type::P
    grid_pos::GridPosition
    logger::T
    converter::U
    kwargs::Dict
    data_source::Any
    data_obs::Union{Observable, Nothing}
    has_plot::Bool
end

"""
    PlotCallback(
        plot::Union{Type, Function},
        [plot_location::Union{GridPosition, Figure}],
        logger::Function, converter = identity; kwargs...
    )

Constructs a `PlotCallback` that plots data from the particle filter state
using the specified `plot` (e.g. `scatter`, `barplot`, `series`) at the
specified `plot_location` (a grid position or figure). If `plot_location` is
not specified, the plot is added to the first position of a new figure.

The `logger` function is used to extract data from the particle filter state,
and the `converter` function is used to convert the data to a format that can be
plotted (e.g. concatenating a vector of vectors into a matrix).

Keyword arguments are passed to the plotting function.
"""
function PlotCallback(
    plot_type_or_fn::Union{Type, Function},
    grid_pos_or_fig::Union{GridPosition, Figure},
    logger::Function, converter = identity; kwargs...    
)
    plot_type = plot_type_or_fn isa Type ?
        plot_type_or_fn : Combined{plot_type_or_fn}
    grid_pos = grid_pos_or_fig isa GridPosition ?
        grid_pos_or_fig : grid_pos_or_fig[1, 1]
    kwargs = Dict(kwargs...)
    data_source = nothing
    data_obs = nothing
    has_plot = false
    return PlotCallback(plot_type, grid_pos, logger, converter,
                        kwargs, data_source, data_obs, has_plot)
end

function PlotCallback(
    plot_type_or_fn::Union{Type, Function},
    logger::Function, converter = identity; kwargs...    
)
    return PlotCallback(plot_type_or_fn, Figure(), logger, converter; kwargs...)
end

"""
    PlotCallback(
        plot::Union{Type, Function},
        [plot_location::Union{GridPosition, Figure}],
        logger_cb::DataLoggerCallback, logger_var::Symbol,
        converter = identity; kwargs...
    )

Constructs a `PlotCallback` that plots data stored in a `DataLoggerCallback`
using the specified `plot` (e.g. `scatter`, `barplot`, `series`) at the
specified `plot_location` (a grid position or figure). 

The `logger_var` is the name of the variable in the `DataLoggerCallback` that
contains the data to be plotted. The `converter` function is used to convert
the data to a format that can be plotted.

Keyword arguments are passed to the plotting function. In addition, 
`legend_title` and `legend_args` can be specified to add an `axislegend` 
to the plot.
"""
function PlotCallback(
    plot_type_or_fn::Union{Type, Function},
    grid_pos_or_fig::Union{GridPosition, Figure},
    logger_cb::DataLoggerCallback, logger_var::Symbol,
    converter = identity; kwargs...
)
    plot_type = plot_type_or_fn isa Type ?
        plot_type_or_fn : Combined{plot_type_or_fn}
    grid_pos = grid_pos_or_fig isa GridPosition ?
        grid_pos_or_fig : grid_pos_or_fig[1, 1]
    logger = nothing
    kwargs = Dict(kwargs...)
    data_source = logger_cb.data[logger_var]
    data_obs = nothing
    has_plot = false
    return PlotCallback(plot_type, grid_pos, logger, converter,
                        kwargs, data_source, data_obs, has_plot)
end

function PlotCallback(
    plot_type_or_fn::Union{Type, Function},
    logger_cb::DataLoggerCallback, logger_var::Symbol,
    converter = identity; kwargs...
)
    return PlotCallback(
        plot_type_or_fn, Figure(), logger_cb, logger_var,
        converter, kwargs...
    )
end

"A convenience function for creating a `PlotCallback` for a `barplot`."
BarPlotCallback(args...; kwargs...) =
    PlotCallback(BarPlot, args...; kwargs...)

"A convenience function for creating a `PlotCallback` for a `series` plot."
SeriesPlotCallback(args...; kwargs...) =
    PlotCallback(Series, args...; kwargs...)

function (cb::PlotCallback)(t::Int, obs, pf_state)
    if !isnothing(cb.logger) # Update data using logger if one is defined 
        if applicable(cb.logger, t, obs, pf_state)
            val = cb.logger(t, obs, pf_state)
        elseif applicable(cb.logger, t, pf_state)
            val = cb.logger(t, pf_state)
        elseif applicable(cb.logger, pf_state)
            val = cb.logger(pf_state)
        else
            error("Logger cannot be called on arguments.")
        end
        if isnothing(cb.data_obs)
            cb.data_obs = Observable(cb.converter(val))
        else
            cb.data_obs[] = cb.converter(val)
        end
    else # Otherwise update data from data source
        if isnothing(cb.data_source)
            error("No data source defined.")
        end
        if isnothing(cb.data_obs)
            cb.data_obs = Observable(cb.converter(cb.data_source))
        else
            cb.data_obs[] = cb.converter(cb.data_source)
        end
    end
    # Create plot if one doesn't already exist
    if !cb.has_plot
        kwargs = copy(cb.kwargs)
        legend_title = get(kwargs, :legend_title, nothing)
        legend_args = get(kwargs, :legend_args, ())
        delete!(kwargs, :legend_title)
        delete!(kwargs, :legend_args)
        if isempty(contents(cb.grid_pos))
            ax, _ = plot(cb.plot_type, cb.grid_pos, cb.data_obs; kwargs...)
            if legend_title !== nothing
                axislegend(ax, ax, legend_title; legend_args...)
            end
        else
            delete!(kwargs, :axis)
            plot!(cb.plot_type, cb.grid_pos, cb.data_obs; kwargs...)
        end
        cb.has_plot = true
    end
    # Reset limits for axis
    ax = contents(cb.grid_pos)[1]
    reset_limits!(ax)
    # Return figure associated with grid position
    return cb.grid_pos.layout.parent
end

"""
    RenderStateCallback(
        renderer::Renderer, [output], domain::Domain;
        trajectory=nothing, overlay=nothing, kwargs...
    )

Callback that renders each new PDDL `State` observed in the process of
inference. A `Renderer` must be provided, along with a `Domain`. The `output`
can be `Canvas`, `GridPosition`, or `Figure`. By default, a new `Canvas` is
created. 

If a `trajectory` is specified, this is used as ground truth sequence of states
to render. Otherwise, this callback will look up the observed state from the 
particle filter, but this may sometimes diverge from ground truth.

An `overlay` function can be provided to render additional graphics on top of
the domain. This function should have the signature:

    overlay(canvas::Canvas, renderer::Renderer, domain::Domain,
            t::Int, obs::ChoiceMap, pf_state::ParticleFilterState)
"""
struct RenderStateCallback{T <: Renderer, U} <: SIPSCallback
    renderer::T
    canvas::Canvas
    domain::Domain
    trajectory::Union{Nothing, Vector{<:State}}
    overlay::U
    kwargs::Dict
end

function RenderStateCallback(
    renderer::Renderer, canvas::Canvas, domain::Domain;
    trajectory = nothing, overlay = nothing, kwargs...
)
    kwargs = Dict(kwargs...)
    return RenderStateCallback(renderer, canvas, domain,
                            trajectory, overlay, kwargs)
end

function RenderStateCallback(
    renderer::Renderer, domain::Domain;
    trajectory = nothing, overlay = nothing, kwargs...
)
    canvas = PDDLViz.new_canvas(renderer)
    kwargs = Dict(kwargs...)
    return RenderStateCallback(renderer, canvas, domain,
                               trajectory, overlay, kwargs)
end

function RenderStateCallback(
    renderer::Renderer, output::Union{GridPosition,Figure}, domain::Domain;
    trajectory = nothing, overlay = nothing, kwargs...
)
    canvas = PDDLViz.new_canvas(renderer, output)
    kwargs = Dict(kwargs...)
    return RenderStateCallback(renderer, canvas, domain,
                               trajectory, overlay, kwargs)
end

function (cb::RenderStateCallback)(t::Int, obs, pf_state)
    # Extract state from ground-truth trajectory or particle filter
    if cb.trajectory === nothing
        obs_addr = t > 0 ? (:timestep => t => :obs) : (:init => :obs)
        state = pf_state.traces[1][obs_addr]
    else
        state = cb.trajectory[t+1]
    end
    # Initialize or update animation
    if cb.canvas.state === nothing
        anim_initialize!(cb.canvas, cb.renderer, cb.domain, state; cb.kwargs...)
    else
        anim_transition!(cb.canvas, cb.renderer, cb.domain,
                         state, PDDL.no_op, t; cb.kwargs...)
    end
    # Overlay additional graphics
    if cb.overlay !== nothing
        cb.overlay(cb.canvas, cb.renderer, cb.domain, t, obs, pf_state)
    end
    # Return canvas
    return cb.canvas
end

"""
    TraceCanvas(figure::Figure)

Container for figure, traces and weights rendered by a `RenderTracesCallback`.
"""
mutable struct TraceCanvas
    figure::Figure
    title::Observable{String}
    t::Observable{Int}
    traces::Vector{Observable}
    weights::Vector{Observable}
end

function TraceCanvas(figure::Figure)
    title = Observable("")
    t = Observable(-1)
    traces = Vector{Observable}()
    weights = Vector{Observable}()
    return TraceCanvas(figure, title, t, traces, weights)
end

"""
    RenderTracesCallback(
        renderer::TraceRenderer, figure::Figure, domain::Domain;
        kwargs...
    )

Callback that renders a set of traces and weights from a particle filter state
using a `TraceRenderer`.

# Keyword Arguments

- `selector`: Function `pf_state -> (traces, weights)` that selects and
    transforms traces and weights from a particle filter state.
- `title_fn`: Function `(t, obs, pf) -> String` that generates a figure title.
- `title_options`: Keyword arguments passed to `Label` for the title.

All other keyword arguments are passed to the `TraceRenderer` when rendering.
"""
struct RenderTracesCallback{T <: TraceRenderer} <: SIPSCallback
    renderer::T
    canvas::TraceCanvas
    domain::Domain
    selector::Function
    title_fn::Function
    title_options::Dict
    kwargs::Dict
end

function RenderTracesCallback(
    renderer::TraceRenderer, figure::Figure, domain::Domain;
    selector = nothing,
    title_fn = (t, obs, pf) -> "t = $t",
    title_options = Dict(
        :fontsize => 22,
        :valign => :bottom
    ),
    kwargs...
)
    canvas = TraceCanvas(figure)
    kwargs = Dict(kwargs...)
    if isnothing(selector)
        selector = pf -> begin
            idxs = sortperm(get_log_weights(pf), rev=true)[1:9]
            return get_traces(pf)[idxs], get_log_weights(pf)[idxs]
        end
    end
    return RenderTracesCallback(renderer, canvas, domain,
                                selector, title_fn, title_options, kwargs)
end

function (cb::RenderTracesCallback)(t::Int, obs, pf_state)
    # Extract traces and weights to render from particle filter state
    new_traces, new_weights = cb.selector(pf_state)
    # Initialize or update traces
    canvas = cb.canvas
    if canvas.t[] < 0 || isempty(canvas.traces)
        canvas.t[] = t
        empty!(canvas.traces)
        empty!(canvas.weights)
        for (tr, w) in zip(new_traces, new_weights)
            push!(canvas.traces, Observable(tr))
            push!(canvas.weights, Observable(w))
        end
        cb.renderer(canvas.figure, cb.domain,
                    canvas.traces, canvas.t, canvas.weights; cb.kwargs...)
        canvas.title[] = cb.title_fn(t, obs, pf_state)
        Label(canvas.figure[0, :], canvas.title; cb.title_options...)
        rowsize!(canvas.figure.layout, 0, Auto())
    else
        canvas.t.val = t
        for i in 1:length(new_traces)
            canvas.traces[i].val = new_traces[i]
            canvas.weights[i].val = new_weights[i]
        end
        notify(canvas.t)
        canvas.title[] = cb.title_fn(t, obs, pf_state)
    end
    # Return figure
    return canvas.figure
end

"""
    RecordCallback(figure::Figure; kwargs...)

Callback that records frames from a figure to an animation object. Keyword
arguments are passed to the `Animation` constructor, which can be used to 
configure the animation `format`, `framerate` (default: `5`), etc. See 
`Makie.record` for more details.
"""
struct RecordCallback <: SIPSCallback
    figure::Figure
    animation::PDDLViz.Animation
end

function RecordCallback(figure::Figure; framerate=5, visible=true, kwargs...)
    animation = PDDLViz.Animation(figure; framerate=framerate,
                                  visible=visible, kwargs...)
    RecordCallback(figure, animation)
end

RecordCallback(canvas::Canvas; kwargs...) =
    RecordCallback(canvas.figure; kwargs...)

function (cb::RecordCallback)(t::Int, obs, pf_state)
    # If figure is displayed, sleep for long enough to update
    if !isempty(cb.figure.scene.current_screens)
        sleep(0.05)
    end
    # Record frame to animation object
    recordframe!(cb.animation)
end
