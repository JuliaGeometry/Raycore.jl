# GPU Optimization Guidelines for Julia

This document covers common pitfalls that cause GPU kernel compilation failures or poor performance in Julia, and provides patterns to avoid them.

## The Core Problem: Dynamic Dispatch on GPU

GPUs require fully static, type-stable code. Any operation that requires runtime type dispatch will fail to compile or cause `ijl_get_nth_field_checked` errors. Common causes:

1. **Closure boxing** - captured variables become heap-allocated `Core.Box`
2. **Heterogeneous tuple iteration** - `for x in tuple` with mixed types
3. **Type inference limits** - tuples > 32 elements, deeply nested types
4. **Abstract field access** - accessing fields through abstract types

## Compiler Limits to Know

| Constant | Default | Effect when exceeded |
|----------|---------|---------------------|
| `MAX_TUPLETYPE_LEN` | ~32 | Tuple types lose precise inference |
| `MAX_TYPE_DEPTH` | varies | Nested types get widened to supertypes |
| `MAX_UNION_SPLITTING` | ~4 | Union types use dynamic dispatch |
| Inlining threshold | ~100 cycles | Functions won't inline |

## Pattern 1: Avoid `for` Loops Over Heterogeneous Tuples

### Bad - Causes boxing and dynamic dispatch
```julia
function sum_lights(lights::Tuple, ray)
    result = RGBSpectrum(0f0)
    for light in lights  # Creates dynamic iteration!
        result += le(light, ray)
    end
    return result
end
```

### Good - Recursive tuple traversal (compile-time unrolling)
```julia
@inline sum_lights(::Tuple{}, ray) = RGBSpectrum(0f0)
@inline function sum_lights(lights::Tuple, ray)
    return le(first(lights), ray) + sum_lights(Base.tail(lights), ray)
end
```

### Better - Use `for_unrolled` with explicit arguments
```julia
# Avoids closure capture entirely by passing all data as arguments
result = for_unrolled(sum_light_contribution, lights, ray, initial_value)
```

## Pattern 2: Avoid Closure Capture

### Bad - Variable capture causes boxing
```julia
function process(data, threshold)
    # `threshold` gets boxed because it's captured
    map(x -> x > threshold ? x : zero(x), data)
end
```

### Good - Use `let` block to create immutable binding
```julia
function process(data, threshold)
    f = let t = threshold
        x -> x > t ? x : zero(x)
    end
    map(f, data)
end
```

### Better - Avoid closures entirely, pass as argument
```julia
function process(data, threshold)
    map((x, t) -> x > t ? x : zero(x), data, Ref(threshold))
end

# Or use a functor
struct ThresholdFilter{T}
    threshold::T
end
(f::ThresholdFilter)(x) = x > f.threshold ? x : zero(x)
```

## Pattern 3: Use `for_unrolled` for GPU-Safe Iteration

The `for_unrolled` function provides compile-time loop unrolling without closure capture:

```julia
# Instead of:
for i in 1:N
    process(data[i], extra_arg)
end

# Use:
for_unrolled(process_item, Val(N), data, extra_arg)
# Where process_item(i, data, extra_arg) is your function
```

### With Tuples (heterogeneous types)
```julia
# Bad: for light in lights
# Good:
result = for_unrolled(
    accumulate_light,  # function(elem, acc, ray) -> new_acc
    lights,            # tuple to iterate
    RGBSpectrum(0f0),  # initial accumulator
    ray                # extra arguments...
)
```

## Pattern 4: Ensure Type Stability

### Check with `@code_warntype`
```julia
@code_warntype my_kernel_function(args...)
# Look for:
# - `Any` types (red in color terminals)
# - `Core.Box` (captured variables)
# - `Union{...}` with many types
```

### Use JET.jl for deeper analysis
```julia
using JET
@report_opt my_function(args...)
```

## Pattern 5: Use Concrete Types in Structs

### Bad - Abstract field types
```julia
struct Scene
    lights::Vector{Light}  # Abstract element type
end
```

### Good - Parameterized concrete types
```julia
struct Scene{L<:Tuple}
    lights::L  # Concrete tuple type, e.g., Tuple{SunLight, PointLight}
end
```

## Pattern 6: Avoid Runtime Allocations

### Bad - Creates intermediate arrays
```julia
function compute(points)
    distances = [norm(p) for p in points]  # Allocates!
    return sum(distances)
end
```

### Good - Fuse operations
```julia
function compute(points)
    total = 0f0
    for p in points
        total += norm(p)
    end
    return total
end
```

### For small fixed-size data, use tuples/StaticArrays
```julia
# Heap allocated (bad for GPU registers)
coords = [1.0f0, 2.0f0, 3.0f0]

# Stack allocated (good)
coords = (1.0f0, 2.0f0, 3.0f0)
coords = SVector{3, Float32}(1, 2, 3)
```

## Pattern 7: Use 32-bit Integers

```julia
# Bad - promotes to Int64
idx = blockIdx().x - 1

# Good - stays Int32
idx = blockIdx().x - Int32(1)

# Helper function
gpu_int(x) = x % Int32
```

## Quick Reference: GPU-Safe Alternatives

| Avoid | Use Instead |
|-------|-------------|
| `for x in heterogeneous_tuple` | Recursive functions or `for_unrolled` |
| `x -> f(x, captured_var)` | `let` blocks or pass args explicitly |
| `Vector{AbstractType}` | `Tuple` or `Vector{ConcreteType}` |
| Dynamic `if typeof(x) == ...` | Multiple dispatch |
| `Int64` literals | `Int32` / `gpu_int()` |
| `Array` in kernel | `Tuple` or `SVector` |

## Debugging GPU Compilation Errors

When you see errors like:
- `ijl_get_nth_field_checked` - Dynamic field access (boxing)
- `jl_apply_generic` - Dynamic dispatch
- `jl_gc_*` - Heap allocation attempted

Steps to debug:
1. Identify the function mentioned in the stack trace
2. Run `@code_warntype` on that function
3. Look for `Core.Box`, `Any`, or large `Union` types
4. Apply the patterns above to make the code type-stable

## See Also

- [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [CUDA.jl Performance Tips](https://cuda.juliagpu.org/stable/tutorials/performance/)
- [Julia Issue #15276](https://github.com/JuliaLang/julia/issues/15276) - Closure boxing
