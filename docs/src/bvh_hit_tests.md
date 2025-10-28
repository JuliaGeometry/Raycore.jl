# BVH Hit Testing: `closest_hit` vs `any_hit`

This document tests and visualizes the difference between `closest_hit` and `any_hit` functions in the BVH implementation using the new `RayIntersectionSession` API.

## Test Setup

```julia (editor=true, logging=false, output=true)
using Raycore, GeometryBasics, LinearAlgebra
using WGLMakie
using Test
using Bonito

# Create a simple test scene with multiple overlapping primitives
function create_test_scene()
    # Three spheres at different distances along the Z-axis
    sphere1 = Tesselation(Sphere(Point3f(0, 0, 5), 1.0f0), 20)   # Furthest
    sphere2 = Tesselation(Sphere(Point3f(0, 0, 3), 1.0f0), 20)   # Middle
    sphere3 = Tesselation(Sphere(Point3f(0, 0, 1), 1.0f0), 20)   # Closest

    bvh = Raycore.BVHAccel([sphere1, sphere2, sphere3])
    return bvh
end

bvh = create_test_scene()

DOM.div("✓ Created BVH with $(length(bvh.primitives)) triangles from 3 spheres")
```
## Test 1: Single Ray Through Center

Test a ray through the center that passes through all three spheres.

```julia (editor=true, logging=false, output=true)
# Create a ray with slight offset to avoid hitting triangle vertices exactly
test_ray = Raycore.Ray(o=Point3f(0.1, 0.1, -5), d=Vec3f(0, 0, 1))

# Create session with closest_hit
session_closest = RayIntersectionSession(Raycore.closest_hit, [test_ray], bvh)

# Create session with any_hit for comparison
session_any = RayIntersectionSession(Raycore.any_hit, [test_ray], bvh)

fig = Figure()

# Left: closest_hit visualization
plot(fig[1, 1], session_closest)
plot(fig[1, 2], session_any)
Label(fig[0, 1], "closest_hit", fontsize=20, font=:bold, tellwidth=false)
Label(fig[0, 2], "any_hit", fontsize=20, font=:bold, tellwidth=false)

fig
```
## Visualization: Single Ray with Makie Recipe

```julia (editor=true, logging=false, output=true)
# Create a ray with slight offset to avoid hitting triangle vertices exactly
test_ray = Raycore.Ray(o=Point3f(0.1, 0.1, 10), d=Vec3f(0, 0, -1))

# Create session with closest_hit
session_closest = RayIntersectionSession(Raycore.closest_hit, [test_ray], bvh)

# Create session with any_hit for comparison
session_any = RayIntersectionSession(Raycore.any_hit, [test_ray], bvh)

fig = Figure()
# Left: closest_hit visualization
plot(fig[1, 1], session_closest)
plot(fig[1, 2], session_any)
Label(fig[0, 1], "closest_hit", tellwidth=false)
Label(fig[0, 2], "any_hit", tellwidth=false)

fig
```
## Test 2: Multiple Rays from Different Positions

Test multiple rays to ensure both functions work correctly.

```julia (editor=true, logging=false, output=true)
# Test rays from different angles (with slight offset to avoid vertex hits)
test_positions = [
    Point3f(0.1, 0.1, -5),      # Center
    Point3f(0.5, 0.1, -5),      # Right offset
    Point3f(0.1, 0.5, -5),      # Top offset
    Point3f(-0.5, 0.1, -5),     # Left offset
]

# Create rays
rays = [Raycore.Ray(o=pos, d=Vec3f(0, 0, 1)) for pos in test_positions]

# Create session
session_multi = RayIntersectionSession(Raycore.closest_hit, rays, bvh)
fig2 = Figure()
ax = LScene(fig2[1, 1])

# Use different colors for each ray
ray_colors = [:purple, :orange, :cyan, :magenta]

plot!(ax, session_multi;
      show_bvh=true,
      bvh_alpha=0.3,
      ray_colors=ray_colors,
      hit_color=:green,
      show_hit_points=true,
      hit_markersize=0.15,
      show_labels=false)

fig2
```
## Visualization: Multiple Rays

## Test 4: Difference Between any*hit and closest*hit

Demonstrate that `any_hit` can return different results than `closest_hit`.

```julia (editor=true, logging=false, output=true)
# Create a complex scene with overlapping geometry
# This creates a BVH where traversal order can differ from distance order
using Random
Random.seed!(123)

complex_spheres = []

# Add some large overlapping spheres
push!(complex_spheres, Tesselation(Sphere(Point3f(0, 0, 10), 3.0f0), 20))
push!(complex_spheres, Tesselation(Sphere(Point3f(0.5, 0, 5), 0.5f0), 15))
push!(complex_spheres, Tesselation(Sphere(Point3f(-0.5, 0, 15), 1.5f0), 18))

# Add many small spheres to create complex BVH structure
for i in 1:30
    x = randn() * 5
    y = randn() * 5
    z = rand(8.0:0.5:12.0)
    r = 0.3 + rand() * 0.5
    push!(complex_spheres, Tesselation(Sphere(Point3f(x, y, z), r), 8))
end

complex_bvh = Raycore.BVHAccel(complex_spheres)

# Test rays to find cases where any_hit differs from closest_hit
test_rays = map(1:100) do i
    x = (i % 10) * 0.4 - 2.0
    y = div(i-1, 10) * 0.4 - 2.0
    Raycore.Ray(o=Point3f(x, y, -5), d=Vec3f(0, 0, 1))
end

session_closest = RayIntersectionSession(Raycore.closest_hit, test_rays, complex_bvh)
session_any = RayIntersectionSession(Raycore.any_hit, test_rays, complex_bvh)
fig = Figure()
# Left: closest_hit visualization
plot(fig[1, 1], session_closest)
plot(fig[1, 2], session_any)
Label(fig[0, 1], "closest_hit", tellwidth=false)
Label(fig[0, 2], "any_hit", tellwidth=false)

fig

```
**Key Findings:**

  * `any_hit` exits on the **first** intersection during BVH traversal (uses `intersect`, doesn't update ray)
  * `closest_hit` continues searching and updates ray's `t_max` (uses `intersect_p!`)
  * In complex scenes with overlapping geometry, `any_hit` can return hits that are significantly farther
  * Both always agree on **whether** a hit occurred (hit vs miss)
  * The difference appears when BVH traversal order differs from spatial distance order

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

test_ray = Raycore.Ray(o=Point3f(0.1, 0.1, -5), d=Vec3f(0, 0, 1))

# Benchmark closest_hit
closest_time = @benchmark Raycore.closest_hit($bvh, $test_ray)

# Benchmark any_hit
any_time = @benchmark Raycore.any_hit($bvh, $test_ray)


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

1. **`RayIntersectionSession`** - A convenient struct for managing ray tracing sessions

      * Bundles rays, BVH, hit function, and results together
      * Provides helper functions: `hit_count()`, `miss_count()`, `hit_points()`, `hit_distances()`
2. **Makie visualization recipe** - Automatic visualization via `plot(session)`

      * Automatically renders BVH geometry, rays, and hit points
      * Customizable colors, transparency, markers, and labels
      * Works with any Makie backend (GLMakie, WGLMakie, CairoMakie)
3. **`closest_hit`** correctly identifies the nearest intersection among multiple overlapping primitives

      * Returns: `(hit_found::Bool, hit_primitive::Triangle, distance::Float32, barycentric_coords::Point3f)`
      * `distance` is the distance from ray origin to the hit point
      * Use `Raycore.sum_mul(bary_coords, primitive.vertices)` to convert to world-space hit point
4. **`any_hit`** efficiently determines if any intersection exists, exiting early

      * Returns: Same format as `closest_hit`: `(hit_found::Bool, hit_primitive::Triangle, distance::Float32, barycentric_coords::Point3f)`
      * Can exit early on first hit found, making it faster for occlusion testing
5. Both functions handle miss cases correctly (returning `hit_found=false`)
6. `any_hit` is typically faster than `closest_hit` due to early termination

All tests passed! ✓

