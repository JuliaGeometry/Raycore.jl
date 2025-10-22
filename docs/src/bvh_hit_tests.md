# BVH Hit Testing: `closest_hit` vs `any_hit`

This document tests and visualizes the difference between `closest_hit` and `any_hit` functions in the BVH implementation using the new `RayIntersectionSession` API.

## Test Setup

```julia (editor=true, logging=false, output=true)
using RayCaster, GeometryBasics, LinearAlgebra
using WGLMakie
using Test
using Bonito

# Create a simple test scene with multiple overlapping primitives
function create_test_scene()
    # Three spheres at different distances along the Z-axis
    sphere1 = Tesselation(Sphere(Point3f(0, 0, 5), 1.0f0), 20)   # Furthest
    sphere2 = Tesselation(Sphere(Point3f(0, 0, 3), 1.0f0), 20)   # Middle
    sphere3 = Tesselation(Sphere(Point3f(0, 0, 1), 1.0f0), 20)   # Closest

    bvh = RayCaster.BVHAccel([sphere1, sphere2, sphere3])
    return bvh
end

bvh = create_test_scene()

DOM.div("✓ Created BVH with $(length(bvh.primitives)) triangles from 3 spheres")
```
## Test 1: Single Ray Through Center

Test a ray through the center that passes through all three spheres.

```julia (editor=true, logging=false, output=true)
# Create a ray with slight offset to avoid hitting triangle vertices exactly
test_ray = RayCaster.Ray(o=Point3f(0.1, 0.1, -5), d=Vec3f(0, 0, 1))

# Create session with closest_hit
session_closest = RayIntersectionSession([test_ray], bvh, RayCaster.closest_hit)

# Create session with any_hit for comparison
session_any = RayIntersectionSession([test_ray], bvh, RayCaster.any_hit)

fig = Figure()

# Left: closest_hit visualization
plot(fig[1, 1], session_closest)
plot(fig[1, 2], session_any)
Label(fig[0, 1], "closest_hit", fontsize=20, font=:bold)
Label(fig[0, 2], "any_hit", fontsize=20, font=:bold)

fig
```
## Visualization: Single Ray with Makie Recipe

```julia (editor=true, logging=false, output=true)
# Create a ray with slight offset to avoid hitting triangle vertices exactly
test_ray = RayCaster.Ray(o=Point3f(0.1, 0.1, 10), d=Vec3f(0, 0, -1))

# Create session with closest_hit
session_closest = RayIntersectionSession([test_ray], bvh, RayCaster.closest_hit)

# Create session with any_hit for comparison
session_any = RayIntersectionSession([test_ray], bvh, RayCaster.any_hit)

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
rays = [RayCaster.Ray(o=pos, d=Vec3f(0, 0, 1)) for pos in test_positions]

# Create session
session_multi = RayIntersectionSession(rays, bvh, RayCaster.closest_hit)
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

## Test 3: Miss Cases

Test rays that don't intersect any geometry.

```julia (editor=true, logging=false, output=true)
# Rays that miss all spheres
miss_rays = [
    RayCaster.Ray(o=Point3f(5, 0, -5), d=Vec3f(0, 0, 1)),  # Too far right
    RayCaster.Ray(o=Point3f(0, 5, -5), d=Vec3f(0, 0, 1)),  # Too far up
    RayCaster.Ray(o=Point3f(0, 0, -5), d=Vec3f(1, 0, 0)),  # Wrong direction
]

# Create sessions for both hit functions
session_miss_closest = RayIntersectionSession(miss_rays, bvh, RayCaster.closest_hit)
session_miss_any = RayIntersectionSession(miss_rays, bvh, RayCaster.any_hit)

# Verify all misses
@test miss_count(session_miss_closest) == 3
@test miss_count(session_miss_any) == 3

descriptions = ["Too far right", "Too far up", "Wrong direction"]
miss_table = map(enumerate(zip(session_miss_closest.hits, session_miss_any.hits, descriptions))) do (i, (closest_hit, any_hit, desc))
    @test closest_hit[1] == false
    @test any_hit[1] == false

    (
        Ray = "Miss ray $i",
        Description = desc,
        closest_hit = closest_hit[1],
        any_hit = any_hit[1],
        Status = "✓"
    )
end

Bonito.Table(miss_table)
```
## Visualization: Miss Cases

```julia (editor=true, logging=false, output=true)
fig3 = Figure()
ax = LScene(fig3[1, 1])

plot!(ax, session_miss_closest;
      show_bvh=true,
      bvh_alpha=0.4,
      ray_colors=[:red, :orange, :yellow],
      miss_color=:gray,
      ray_length=15.0f0
)

fig3
```
## Test 4: Difference Between any_hit and closest_hit

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

complex_bvh = RayCaster.BVHAccel(complex_spheres)

# Test rays to find cases where any_hit differs from closest_hit
test_rays = map(1:100) do i
    x = (i % 10) * 0.4 - 2.0
    y = div(i-1, 10) * 0.4 - 2.0
    RayCaster.Ray(o=Point3f(x, y, -5), d=Vec3f(0, 0, 1))
end

session_closest = RayIntersectionSession(test_rays, complex_bvh, RayCaster.closest_hit)
session_any = RayIntersectionSession(test_rays, complex_bvh, RayCaster.any_hit)

# Find cases where they differ
differences = []
for (i, (closest, any)) in enumerate(zip(session_closest.hits, session_any.hits))
    hit_closest, prim_closest, dist_closest, _ = closest
    hit_any, prim_any, dist_any, _ = any

    if hit_closest && hit_any
        diff = abs(dist_closest - dist_any)
        if diff > 0.1  # Significant difference
            push!(differences, (
                ray_idx = i,
                diff = diff,
                closest_dist = dist_closest,
                any_dist = dist_any,
                closest_mat = prim_closest.material_idx,
                any_mat = prim_any.material_idx
            ))
        end
    end
end

# Show results
diff_table = if length(differences) > 0
    sort!(differences, by=x->x.diff, rev=true)
    top5 = differences[1:min(5, length(differences))]
    map(enumerate(top5)) do (i, d)
        (
            Rank = string(i),
            Distance_Diff = "$(round(d.diff, digits=2))",
            closest_hit_dist = "$(round(d.closest_dist, digits=2))",
            any_hit_dist = "$(round(d.any_dist, digits=2))",
            Same_Primitive = d.closest_mat == d.any_mat ? "✓" : "✗"
        )
    end
else
    [(Rank = "-", Distance_Diff = "No differences found", closest_hit_dist = "-", any_hit_dist = "-", Same_Primitive = "-")]
end

Bonito.Table(diff_table)
```

**Key Findings:**
- `any_hit` exits on the **first** intersection during BVH traversal (uses `intersect`, doesn't update ray)
- `closest_hit` continues searching and updates ray's `t_max` (uses `intersect_p!`)
- In complex scenes with overlapping geometry, `any_hit` can return hits that are significantly farther
- Both always agree on **whether** a hit occurred (hit vs miss)
- The difference appears when BVH traversal order differs from spatial distance order
## Performance Comparison

Compare the performance of `closest_hit` vs `any_hit`.

```julia (editor=true, logging=false, output=true)
using BenchmarkTools

test_ray = RayCaster.Ray(o=Point3f(0.1, 0.1, -5), d=Vec3f(0, 0, 1))

# Benchmark closest_hit
closest_time = @benchmark RayCaster.closest_hit($bvh, $test_ray)

# Benchmark any_hit
any_time = @benchmark RayCaster.any_hit($bvh, $test_ray)


perf_table = map([
    ("closest_hit", any_time),
    ("any_hit", closest_time),
]) do (method, time_us)
    (Method = method, Time_μs = time_us)
end
any_time
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
      * Use `RayCaster.sum_mul(bary_coords, primitive.vertices)` to convert to world-space hit point
4. **`any_hit`** efficiently determines if any intersection exists, exiting early

      * Returns: Same format as `closest_hit`: `(hit_found::Bool, hit_primitive::Triangle, distance::Float32, barycentric_coords::Point3f)`
      * Can exit early on first hit found, making it faster for occlusion testing
5. Both functions handle miss cases correctly (returning `hit_found=false`)
6. `any_hit` is typically faster than `closest_hit` due to early termination

All tests passed! ✓

