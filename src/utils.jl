"Convert vector of unnormalized scores to probabiities."
function softmax(scores)
    T = eltype(scores)
    isempty(scores) && return Vector{T}()
    ws = exp.(scores .- maximum(scores))
    z = sum(ws)
    return isnan(z) ? ones(T, length(scores)) ./ length(scores) : ws ./ z
end

"Compact formatting of types."
function compact_type_str(str::AbstractString, max_type_param_length::Int = 80)
    m = match(r"^([\@\w]+)\{(.+)\}$", str)
    if !isnothing(m) && length(m.captures[2]) > max_type_param_length
        str = m.captures[1] * "{...}"
    end
    return str
end
compact_type_str(T::Type, max_type_param_length::Int = 80) =
    compact_type_str(repr(T), max_type_param_length)
compact_type_str(@nospecialize(x), max_type_param_length::Int = 80) =
    compact_type_str(typeof(T), max_type_param_length)

"Pretty printing for lists."
function print_list(io::IO, list::AbstractArray,
                    n_used::Int = 0, formatter = identity)
    indent = get(io, :indent, "")
    n_lines, _ = displaysize(io)
    n_lines -= n_used
    if length(list) > n_lines
        for x in @view(list[1:(n_lines÷2-1)])
            println(io)
            print(io, indent, "  ", formatter(x))
        end
        println(io)
        print(io, indent, "  ", "⋮")
        for x in @view(list[end-(n_lines÷2)+1:end])
            println(io)
            print(io, indent, "  ", formatter(x))
        end
    else
        for x in list
            println(io)
            print(io, indent, "  ", formatter(x))
        end
    end
    return nothing
end

"Pretty printing for data structures, adapted from `Base.dump`."
function show_struct(
    io::IO, @nospecialize(x);
    indent = "", show_fields = (), show_fields_compact = (), show_list = (),
    max_type_param_length::Int = (displaysize(io)[2]*3)÷4 -length(indent)
)
    T = typeof(x)
    T_str = compact_type_str(T, max_type_param_length)
    if isa(x, Function)
        print(io, x, " (function of type ", T_str, ")")
    else
        print(io, T_str)
    end
    nf = nfields(x)
    if nf > 0
        for field in 1:nf
            println(io)
            fname = fieldname(T, field)
            print(io, indent, "  ", string(fname), ": ")
            if !isdefined(x, field)
                print(io, Base.undef_ref_str)
                continue
            end
            val = getfield(x, field)
            if fname in show_fields
                if isa(val, NamedTuple)
                    show_struct(io, val; indent = indent * "  ")
                else
                    io_ctx = IOContext(io, :indent => indent * "  ")
                    show(io_ctx, MIME"text/plain"(), val)
                end
            elseif fname in show_fields_compact
                io_ctx = IOContext(io, :compact => true)
                show(io_ctx, val)
            elseif fname in show_list && isa(val, AbstractArray)
                summary(io, val)
                io_ctx = IOContext(io, :indent => indent * "  ")
                print_list(io_ctx, val, nf+1)
            elseif isbits(val) || isa(val, Symbol) || isa(val, String)
                show(io, val)
            elseif isa(val, DynamicDSLFunction)
                show(io, val.julia_function)
            else
                new_max = max_type_param_length - 2
                print(io, compact_type_str(summary(val), new_max))
            end
        end
    elseif !isa(x, Function)
        print(io, " ")
        show(io, x)
    end
    nothing
end

"Compact pretty printing for data structures."
function show_struct_compact(
    io::IO, @nospecialize(x);
    show_fields = (),
    max_type_param_length::Int = displaysize(io)[2]÷4
)
    T = typeof(x)
    T_str = compact_type_str(T, max_type_param_length)
    print(io, T_str)
    print(io, "(")
    nf = nfields(x)
    for field in 1:nf
        if !isdefined(x, field)
            print(io, Base.undef_ref_str)
            continue
        end
        fname = fieldname(T, field)
        val = getfield(x, field)
        if fname in show_fields
            io_ctx = IOContext(io, :compact => true)
            show(io_ctx, val)
        elseif isbits(val) || isa(val, Symbol) || isa(val, String)
            show(io, val)
        else
            print(io, compact_type_str(summary(val), max_type_param_length))
            if nfields(val) > 0
                print(io, "(...)")
            else
                print(io, "()")
            end
        end
        if field < nf
            print(io, ", ")
        end
    end
    print(io, ")")
    nothing
end

"Adds documentation for a constructor to the documentation for a type."
macro add_constructor_doc(type, constructor, docstr)
    quote
        Base.with_logger(Base.NullLogger()) do
            @doc """
            $(Base.doc($type)) * `$($constructor)`: $($docstr)
            """
            $type
        end
    end
end