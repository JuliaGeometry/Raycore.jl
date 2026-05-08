# ============================================================================
# Structure of Arrays (SoA) Utilities
# ============================================================================
# Macros and helpers for efficient SoA data access in GPU kernels.
# SoA layout is critical for GPU memory coalescing.

"""
    @get field1, field2, ... = soa[idx]

Macro to extract multiple fields from a Structure of Arrays (SoA) at index `idx`.

# Example
```julia
ray_queue = (ray=[r1, r2, r3], pixel_x=[1, 2, 3], pixel_y=[4, 5, 6])
@get ray, pixel_x, pixel_y = ray_queue[2]
# Expands to:
# ray = ray_queue.ray[2]
# pixel_x = ray_queue.pixel_x[2]
# pixel_y = ray_queue.pixel_y[2]
```
"""
macro get(expr)
    if expr.head != :(=)
        error("@get expects assignment syntax: @get field1, field2 = soa[idx]")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    # Parse left side (field names)
    if lhs isa Symbol
        fields = [lhs]
    elseif lhs.head == :tuple
        fields = lhs.args
    else
        error("@get left side must be field names or tuple of field names")
    end

    # Parse right side (soa[idx])
    if rhs.head != :ref
        error("@get right side must be array indexing: soa[idx]")
    end
    soa = rhs.args[1]
    idx = rhs.args[2]

    # Generate field extraction code
    assignments = [:($(esc(field)) = $(esc(soa)).$(field)[$(esc(idx))]) for field in fields]

    return Expr(:block, assignments...)
end

"""
    @set soa[idx] = (field1=val1, field2=val2, ...)

Macro to set multiple fields in a Structure of Arrays (SoA) at index `idx`.
Expects named tuple syntax on the right side.

# Example
```julia
ray_queue = (ray=Vector{Ray}(undef, 10), pixel_x=zeros(Int32, 10))
@set ray_queue[1] = (ray=my_ray, pixel_x=Int32(5))
# Expands to:
# ray_queue.ray[1] = my_ray
# ray_queue.pixel_x[1] = Int32(5)
```
"""
macro set(expr)
    if expr.head != :(=)
        error("@set expects assignment syntax: @set soa[idx] = (field1=val1, ...)")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    # Parse left side (soa[idx])
    if lhs.head != :ref
        error("@set left side must be array indexing: soa[idx]")
    end
    soa = lhs.args[1]
    idx = lhs.args[2]

    # Parse right side (named tuple or parameters)
    assignments = []
    if rhs.head == :tuple || rhs.head == :parameters
        for arg in rhs.args
            if arg isa Expr && arg.head == :(=)
                field = arg.args[1]
                val = arg.args[2]
                push!(assignments, :($(esc(soa)).$(field)[$(esc(idx))] = $(esc(val))))
            else
                error("@set expects named parameters: @set soa[idx] = (field=value, ...)")
            end
        end
    else
        error("@set expects a tuple with named fields: @set soa[idx] = (field=value, ...)")
    end

    return Expr(:block, assignments...)
end

"""
    similar_soa(template_array, T::Type, num_elements) -> NamedTuple

Create a Structure of Arrays (SoA) layout for type `T` with `num_elements` entries.
Uses `template_array` to determine the array type (Array, ROCArray, etc.).
"""
function similar_soa(template, ::Type{T}, num_elements) where T
    fields = [f => similar(template, fieldtype(T, f), num_elements) for f in fieldnames(T)]
    return (; fields...)
end
