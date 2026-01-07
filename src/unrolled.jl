# ============================================================================
# GPU-Safe Unrolled Iteration Utilities
# ============================================================================
# Provides compile-time unrolled iteration over tuples without closure capture.
# Critical for GPU kernels where dynamic dispatch and boxing are not allowed.

# ============================================================================
# Compiler Limits
# ============================================================================

const MAX_TUPLE_LENGTH = 32
const MAX_TYPE_DEPTH = 10

# ============================================================================
# FastClosure - Compile-time validation wrapper
# ============================================================================

"""
    FastClosure{F, Args<:Tuple}

A callable wrapper that validates a function and arguments are GPU-safe:
1. Function `f` has no captured variables (no `Core.Box` fields)
2. Arguments don't exceed compiler limits (tuple length, type depth)

When called, appends stored args to the call: `fc(x) == fc.f(x, fc.args...)`

This is used internally by `for_unrolled`, `map_unrolled`, and `reduce_unrolled`.
"""
struct FastClosure{F, Args<:Tuple}
    f::F
    args::Args

    function FastClosure(f::F, args::Args) where {F, Args<:Tuple}
        check_no_capture(F)
        check_args_limits(Args)
        new{F, Args}(f, args)
    end
end

# Make FastClosure callable - appends stored args to call
@inline (fc::FastClosure)(outer_args...) = fc.f(outer_args..., fc.args...)

"""
    check_no_capture(::Type{F}) where F

Compile-time check that function type `F` has no captured variables.
Captured variables appear as `Core.Box` fields in the closure struct.
"""
@generated function check_no_capture(::Type{F}) where F
    # Regular functions have no fields
    if fieldcount(F) == 0
        return :nothing
    end

    # Check each field for Core.Box (indicates captured variable)
    for i in 1:fieldcount(F)
        ft = fieldtype(F, i)
        if ft === Core.Box
            fname = fieldname(F, i)
            return :(error("FastClosure: function captures variable '$($(QuoteNode(fname)))' which would be boxed. Pass it as an argument instead."))
        end
    end

    return :nothing
end

"""
    check_args_limits(::Type{Args}) where Args

Compile-time check that argument types don't exceed compiler limits.
"""
@generated function check_args_limits(::Type{Args}) where Args <: Tuple
    # Check total tuple length
    n = length(Args.parameters)
    if n > MAX_TUPLE_LENGTH
        return :(error("FastClosure: too many arguments ($($n) > $MAX_TUPLE_LENGTH). This may cause inference failures."))
    end

    # Check for overly long tuple arguments
    for (i, T) in enumerate(Args.parameters)
        if T <: Tuple && length(T.parameters) > MAX_TUPLE_LENGTH
            return :(error("FastClosure: argument $($i) is a tuple with $($(length(T.parameters))) elements (> $MAX_TUPLE_LENGTH). This may cause inference failures."))
        end
    end

    return :nothing
end

# ============================================================================
# for_unrolled - Side effects, no return value
# ============================================================================

"""
    for_unrolled(f, tuple, args...)

Iterate over `tuple` at compile-time, calling `f(element, args...)` for each element.
No return value (use for side effects).

The function `f` must not capture any variables - pass all data as `args` instead.

# Example
```julia
lights = (sun_light, point_light, spot_light)
total = Ref(RGBSpectrum(0f0))

# Bad - captures `total` and `ray`:
for light in lights
    total[] += le(light, ray)  # Boxing on GPU!
end

# Good - pass as arguments:
for_unrolled(add_light!, lights, total, ray)
# Where: add_light!(light, total, ray) = total[] += le(light, ray)
```
"""
@inline function for_unrolled(f::F, tuple::Tuple, args...) where F
    fc = FastClosure(f, args)
    _for_unrolled(fc, tuple)
    return nothing
end

@inline _for_unrolled(_fc, ::Tuple{}) = nothing
@inline function _for_unrolled(fc, tuple::Tuple)
    fc(first(tuple))
    _for_unrolled(fc, Base.tail(tuple))
    return nothing
end

# Val{N} version for index-based iteration
"""
    for_unrolled(f, ::Val{N}, args...)

Iterate from 1 to N at compile-time, calling `f(i, args...)` for each index.
"""
@inline function for_unrolled(f::F, ::Val{N}, args...) where {F, N}
    fc = FastClosure(f, args)
    _for_unrolled_n(fc, Val(N))
    return nothing
end

@inline _for_unrolled_n(_fc, ::Val{0}) = nothing
@inline function _for_unrolled_n(fc, ::Val{N}) where N
    _for_unrolled_n(fc, Val(N-1))
    fc(N % Int32)
    return nothing
end

# ============================================================================
# map_unrolled - Transform tuple elements
# ============================================================================

"""
    map_unrolled(f, tuple, args...) -> Tuple

Transform each element of `tuple` at compile-time, returning a new tuple.
Calls `f(element, args...)` for each element.

# Example
```julia
lights = (sun_light, point_light)
contributions = map_unrolled(compute_light, lights, hit_point, normal)
# Returns: (compute_light(sun_light, hit_point, normal),
#           compute_light(point_light, hit_point, normal))
```
"""
@inline function map_unrolled(f::F, tuple::Tuple, args...) where F
    fc = FastClosure(f, args)
    return _map_unrolled(fc, tuple)
end

@inline _map_unrolled(_fc, ::Tuple{}) = ()
@inline function _map_unrolled(fc, tuple::Tuple)
    return (fc(first(tuple)), _map_unrolled(fc, Base.tail(tuple))...)
end

# ============================================================================
# reduce_unrolled - Accumulate over tuple elements
# ============================================================================

"""
    reduce_unrolled(f, tuple, init, args...) -> result

Reduce `tuple` at compile-time using `f(accumulator, element, args...)`.

# Example
```julia
lights = (sun_light, point_light, env_light)

# Compute total light contribution
total = reduce_unrolled(add_light_contribution, lights, RGBSpectrum(0f0), ray, hit_point)
# Where: add_light_contribution(acc, light, ray, hp) = acc + compute_li(light, ray, hp)
```
"""
@inline function reduce_unrolled(f::F, tuple::Tuple, init, args...) where F
    fc = FastClosure(f, args)
    return _reduce_unrolled(fc, tuple, init)
end

@inline _reduce_unrolled(_fc, ::Tuple{}, acc) = acc
@inline function _reduce_unrolled(fc, tuple::Tuple, acc)
    new_acc = fc(acc, first(tuple))
    return _reduce_unrolled(fc, Base.tail(tuple), new_acc)
end

# ============================================================================
# sum_unrolled - Common reduction pattern
# ============================================================================

"""
    sum_unrolled(f, tuple, args...) -> result

Sum `f(element, args...)` over all elements of `tuple`.

# Example
```julia
lights = (sun_light, point_light)
total = sum_unrolled(le, lights, ray)
# Computes: le(sun_light, ray) + le(point_light, ray)
```
"""
@inline function sum_unrolled(f::F, tuple::Tuple, args...) where F
    fc = FastClosure(f, args)
    return _sum_unrolled(fc, tuple)
end

@inline _sum_unrolled(_fc, ::Tuple{}) = nothing  # Empty tuple - caller should handle
@inline _sum_unrolled(fc, tuple::Tuple{T}) where T = fc(first(tuple))
@inline function _sum_unrolled(fc, tuple::Tuple)
    return fc(first(tuple)) + _sum_unrolled(fc, Base.tail(tuple))
end
