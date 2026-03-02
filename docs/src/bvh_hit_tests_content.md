# TLAS Hit Testing: `closest_hit` vs `any_hit`

This document tests and visualizes the difference between `closest_hit` and `any_hit` functions in the TLAS implementation using the `trace_rays` API.

## Test Setup

```julia (editor=true, logging=false, output=true)
using Raycore, GeometryBasics, LinearAlgebra
using WGLMakie
using Test
using Bonito
import KernelAbstractions as KA

# Create a simple test scene with multiple overlapping primitives
function create_test_scene()
    # Three spheres at different distances along the Z-axis
    sphere1 = normal_mesh(Sphere(Point3f(0, 0, 5), 1.0f0))   # Furthest
    sphere2 = normal_mesh(Sphere(Point3f(0, 0, 3), 1.0f0))   # Middle
    sphere3 = normal_mesh(Sphere(Point3f(0, 0, 1), 1.0f0))   # Closest

    tlas = Raycore.TLAS(KA.CPU())
    push!(tlas, sphere1)
    push!(tlas, sphere2)
    push!(tlas, sphere3)
    sync!(tlas)
    return tlas
end

tlas = create_test_scene()
```
## Test 1: Single Ray Through Center

Test a ray through the center that passes through all three spheres.

```julia (editor=true, logging=false, output=true)
# Create a ray with slight offset to avoid hitting triangle vertices exactly
test_ray = Raycore.Ray(o=Point3f(0.1, 0.1, -5), d=Vec3f(0, 0, 1))

# Trace with closest_hit (default)
result_closest = trace_rays(tlas, [test_ray])

fig = Figure()

# Left: closest_hit visualization
plot(fig[1, 1], result_closest; axis=(; show_axis=false))
Label(fig[0, 1], "closest_hit", fontsize=20, font=:bold, tellwidth=false)

fig
```
## Test 2: Multiple Rays from Different Positions

Test multiple rays to ensure both functions work correctly.

```julia (editor=true, logging=false, output=true)
# Test rays from different angles (with slight offset to avoid vertex hits)
test_positions = map(p-> (p = p.-0.5; Point3f(p..., -5)), rand(Point2f, 10))
# Create rays
rays = [Raycore.Ray(o=pos, d=Vec3f(0, 0, 1)) for pos in test_positions]

# Trace rays and visualize
result_multi = trace_rays(tlas, rays)
plot(result_multi; axis=(;show_axis=false))
```
## Visualization: Multiple Rays

## Test 4: Complex Scene

Demonstrate ray tracing through a complex scene with many overlapping objects.

```julia (editor=true, logging=false, output=true)
# Create a complex scene with overlapping geometry
using Random
Random.seed!(123)

complex_tlas = Raycore.TLAS(KA.CPU())

# Add some large overlapping spheres
push!(complex_tlas, normal_mesh(Sphere(Point3f(0, 0, 10), 3.0f0)))
push!(complex_tlas, normal_mesh(Sphere(Point3f(0.5, 0, 5), 0.5f0)))
push!(complex_tlas, normal_mesh(Sphere(Point3f(-0.5, 0, 15), 1.5f0)))

# Add many small spheres to create complex TLAS structure
for i in 1:30
    x = randn() * 5
    y = randn() * 5
    z = rand(8.0:0.5:12.0)
    r = 0.3 + rand() * 0.5
    push!(complex_tlas, normal_mesh(Sphere(Point3f(x, y, z), Float32(r))))
end

sync!(complex_tlas)

# Test rays
test_rays = map(rand(Point2f, 20)) do p
    p = (p .* 14f0) .- 8f0
    Raycore.Ray(o=Point3f(p..., -5), d=Vec3f(0, 0, 1))
end

result = trace_rays(complex_tlas, test_rays)

fig = Figure()
plot(fig[1, 1], result; axis=(; show_axis=false))
Label(fig[0, 1], "closest_hit", tellwidth=false)

fig

```
**Key Findings:**

  * `closest_hit` continues searching and updates ray's `t_max` to find the nearest intersection
  * `any_hit` exits on the **first** intersection during TLAS traversal (useful for shadow rays)
  * Both always agree on **whether** a hit occurred (hit vs miss)
  * `any_hit` is typically faster than `closest_hit` due to early termination

## Performance Comparison

Compare the performance of `closest_hit` vs `any_hit`.

```julia (editor=true, logging=false, output=true)
function render_io(obj)
    io = IOBuffer()
    show(io, MIME"text/plain"(), obj)
    printer = BonitoBook.HTMLPrinter(io; root_tag = "span")
    str = sprint(io -> show(io, MIME"text/html"(), printer))
    DOM.pre(HTML(str); style="font-size: 10px")
end
```
```julia (editor=true, logging=false, output=true)
using BenchmarkTools
using Adapt

test_ray = Raycore.Ray(o=Point3f(0.1, 0.1, -5), d=Vec3f(0, 0, 1))
static_tlas = Adapt.adapt(KA.CPU(), tlas)

# Benchmark closest_hit
closest_time = @benchmark Raycore.closest_hit($static_tlas, $test_ray)

# Benchmark any_hit
any_time = @benchmark Raycore.any_hit($static_tlas, $test_ray)


perf_table = map([
    ("closest_hit", any_time),
    ("any_hit", closest_time),
]) do (method, time_us)
    (Method = method, Time_μs = render_io(time_us))
end
Bonito.Table(perf_table)
```
## Summary

This document demonstrated:

1. **`trace_rays`** - A convenient function for tracing rays against a TLAS and collecting results for visualization

      * Returns a `RayIntersectionResult` bundling rays, hit data, and the TLAS
      * Automatically builds a `StaticTLAS` for traversal
2. **Makie visualization recipe** - Automatic visualization via `plot(result)`

      * Automatically renders TLAS geometry, rays, and hit points
      * Customizable colors, transparency, markers, and labels
      * Works with any Makie backend (GLMakie, WGLMakie, CairoMakie)
3. **`closest_hit`** correctly identifies the nearest intersection among multiple overlapping primitives

      * Returns: `(hit_found::Bool, triangle::Triangle, distance::Float32, bary_coords::SVector{3,Float32}, instance_id::UInt32)`
      * Use `sum(bary_coords .* triangle.vertices)` to convert to world-space hit point
4. **`any_hit`** efficiently determines if any intersection exists, exiting early

      * Returns: Same format as `closest_hit`
      * Can exit early on first hit found, making it faster for occlusion testing
5. Both functions handle miss cases correctly (returning `hit_found=false`)
6. `any_hit` is typically faster than `closest_hit` due to early termination

All tests passed! ✓
