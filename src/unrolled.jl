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
Any closure field indicates a captured variable which should be passed as an argument instead.
`Core.Box` fields are especially problematic (heap-allocated, type-unstable).
"""
@generated function check_no_capture(::Type{F}) where F
    # Regular functions have no fields - OK
    if fieldcount(F) == 0
        return :nothing
    end

    # Any field on a closure = captured variable
    # Collect all captured variable names
    captured_names = [fieldname(F, i) for i in 1:fieldcount(F)]
    boxed_names = [fieldname(F, i) for i in 1:fieldcount(F) if fieldtype(F, i) === Core.Box]

    if !isempty(boxed_names)
        # Boxed captures are the worst - definitely error
        names_str = join(boxed_names, ", ")
        return :(error("FastClosure: function captures boxed variable(s): " * $names_str * ". Pass as argument(s) instead."))
    else
        # Non-boxed captures: still problematic for GPU, error with helpful message
        names_str = join(captured_names, ", ")
        return :(error("FastClosure: function captures variable(s): " * $names_str * ". Pass as argument(s) instead to ensure GPU compatibility."))
    end
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

# Use @generated to avoid tuple splatting which causes allocations
@generated function _map_unrolled(fc, tup::T) where T <: Tuple
    N = length(T.parameters)
    exprs = [:(fc(tup[$i])) for i in 1:N]
    return :(($(exprs...),))
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

# ============================================================================
# getindex_unrolled - Select element by runtime index, apply function
# ============================================================================

"""
    getindex_unrolled(f, tuple, idx::Int32, args...) -> result

Select element at runtime index `idx` from `tuple` and apply `f(element, args...)`.
Uses unrolled if-branches for GPU compatibility - no dynamic dispatch.

The index is 1-based. If idx is out of bounds, returns `f(tuple[1], args...)` as fallback.

# Example
```julia
lights = (sun_light, point_light, env_light)
light_idx = Int32(2)

# Sample from the selected light
sample = getindex_unrolled(sample_light, lights, light_idx, point, lambda, u)
# Equivalent to: sample_light(point_light, point, lambda, u)
```
"""
@inline function getindex_unrolled(f::F, tuple::Tuple, idx::Int32, args...) where F
    fc = FastClosure(f, args)
    return _getindex_unrolled(fc, tuple, idx)
end

# Generated function creates unrolled if-branches for type stability
@generated function _getindex_unrolled(fc, tuple::T, idx::Int32) where T <: Tuple
    N = length(T.parameters)

    if N == 0
        # Empty tuple - shouldn't happen, but return nothing
        return :(error("getindex_unrolled: empty tuple"))
    end

    # Build unrolled if-else chain
    # Start from the last index and work backwards to build nested if-else
    expr = :(fc(tuple[$N]))  # Default/fallback case

    for i in (N-1):-1:1
        expr = quote
            if idx == Int32($i)
                fc(tuple[$i])
            else
                $expr
            end
        end
    end

    return expr
end
