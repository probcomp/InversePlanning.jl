## Modeling code ##

"""
    ModelConfig

Abstract type for model configurations.
"""
abstract type ModelConfig end

function Base.show(io::IO, ::MIME"text/plain", config::ModelConfig)
    indent = get(io, :indent, "")
    show_fields = Tuple(
        f for f in fieldnames(typeof(config))
        if isa(getfield(config, f), Union{ModelConfig, NamedTuple})
    )
    show_struct(io, config; indent = indent, show_fields = show_fields)
end

"""
    ModelState

Abstract type for the state of a state-space model.
"""
abstract type ModelState end

function Base.show(io::IO, state::ModelState)
    show_fields = Tuple(f for f in fieldnames(typeof(state))
                        if isa(getfield(state, f), ModelState))
    show_struct_compact(io, state, show_fields = show_fields)
end

function Base.show(io::IO, ::MIME"text/plain", state::ModelState)
    indent = get(io, :indent, "")
    show_fields = Tuple(f for f in fieldnames(typeof(state))
                        if isa(getfield(state, f), ModelState))
    show_struct(io, state; indent = indent, show_fields = show_fields)
end

include("utils.jl")
include("agents/agents.jl")
include("environments.jl")
include("observations.jl")
include("worlds.jl")
