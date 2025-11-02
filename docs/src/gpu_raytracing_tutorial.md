# GPU Ray Tracing with Raycore

In this tutorial, we'll take the ray tracer from the previous tutorial and port it to the GPU using **KernelAbstractions.jl** and **AMDGPU.jl**. We'll explore three different kernel implementations, each with different optimization strategies, and benchmark their performance against each other.

By the end, you'll understand how to write efficient GPU kernels for ray tracing and the tradeoffs between different approaches!

## Setup

```julia (editor=true, logging=false, output=true, id=2)
using Raycore, GeometryBasics, LinearAlgebra
using Colors, ImageShow
using Makie  # For loading assets
using KernelAbstractions
using AMDGPU
using BenchmarkTools
```
**Ready for GPU!** We have:

  * `Raycore` for fast ray-triangle intersections
  * `KernelAbstractions` for portable GPU kernels
  * `AMDGPU` for AMD GPU support
  * `BenchmarkTools` for performance comparison

## Part 1: Scene Setup (Same as CPU Tutorial)

Let's use the exact same scene as the CPU tutorial - the Makie cat with room geometry:

```julia (editor=true, logging=false, output=true, id=4)
# Load and prepare the cat model
cat_mesh = Makie.loadasset("cat.obj")
angle = deg2rad(150f0)
rotation = Makie.Quaternionf(0, sin(angle/2), 0, cos(angle/2))
rotated_coords = [rotation * Point3f(v) for v in coordinates(cat_mesh)]

# Position cat on floor
cat_bbox = Rect3f(rotated_coords)
floor_y = -1.5f0
cat_offset = Vec3f(0, floor_y - cat_bbox.origin[2], 0)

cat_mesh = GeometryBasics.normal_mesh(
    [v + cat_offset for v in rotated_coords],
    faces(cat_mesh)
)

# Create room geometry
floor = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)))
back_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)))
left_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)))

# Add spheres
sphere1 = Tesselation(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), 64)
sphere2 = Tesselation(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), 64)

# Build BVH
scene_geometry = [cat_mesh, floor, back_wall, left_wall, sphere1, sphere2]
bvh_cpu = Raycore.GPUBVH(Raycore.BVHAccel(scene_geometry))
```
## Part 2: Utility Functions

Same helper functions from the CPU tutorial:

```julia (editor=true, logging=false, output=true, id=6)
# Compute interpolated normal at hit point
@inline function compute_normal(triangle, bary_coords)
    v0, v1, v2 = Raycore.normals(triangle)
    u, v, w = bary_coords[1], bary_coords[2], bary_coords[3]
    return Vec3f(normalize(v0 * u + v1 * v + v2 * w))
end

# Generate camera ray for a pixel
@inline function camera_ray(x, y, width, height, camera_pos, focal_length, aspect; jitter=Vec2f(0))
    ndc_x = (2.0f0 * (Float32(x) - 0.5f0 + jitter[1]) / Float32(width) - 1.0f0) * aspect
    ndc_y = 1.0f0 - 2.0f0 * (Float32(y) - 0.5f0 + jitter[2]) / Float32(height)
    direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
    return Raycore.Ray(o=camera_pos, d=direction)
end

# Color conversion
to_vec3f(c::RGB) = Vec3f(c.r, c.g, c.b)
to_rgb(v::Vec3f) = RGB{Float32}(v...)
```
## Part 3: Materials and Lighting Context

Same context setup from the CPU tutorial:

```julia (editor=true, logging=false, output=true, id=8)
struct PointLight
    position::Point3f
    intensity::Float32
    color::RGB{Float32}
end

struct Material
    base_color::RGB{Float32}
    metallic::Float32
    roughness::Float32
end

struct RenderContext{T <: AbstractVector{PointLight}, M<:AbstractVector{Material}}
    lights::T
    materials::M
    ambient::Float32
end
function Raycore.to_gpu(Arr, context::RenderContext; preserve=[])
    gpu_lights = Arr(context.lights)
    gpu_materials = Arr(context.materials)
    return RenderContext(gpu_lights, gpu_materials, context.ambient)
end

# Create lights and materials
lights = [
    PointLight(Point3f(3, 4, -2), 50.0f0, RGB(1.0f0, 0.9f0, 0.8f0)),
    PointLight(Point3f(-3, 2, 0), 20.0f0, RGB(0.7f0, 0.8f0, 1.0f0)),
    PointLight(Point3f(0, 5, 5), 15.0f0, RGB(1.0f0, 1.0f0, 1.0f0))
]

materials = [
    Material(RGB(0.8f0, 0.6f0, 0.4f0), 0.0f0, 0.8f0),  # cat
    Material(RGB(0.3f0, 0.5f0, 0.3f0), 0.0f0, 0.9f0),  # floor
    Material(RGB(0.8f0, 0.6f0, 0.5f0), 0.8f0, 0.05f0), # back wall (metallic)
    Material(RGB(0.7f0, 0.7f0, 0.8f0), 0.0f0, 0.8f0),  # left wall
    Material(RGB(0.9f0, 0.9f0, 0.9f0), 0.8f0, 0.02f0), # sphere1 (metallic)
    Material(RGB(0.3f0, 0.6f0, 0.9f0), 0.5f0, 0.3f0),  # sphere2 (semi-metallic)
]

ctx = RenderContext(lights, materials, 0.1f0)
nothing
```
## Part 4: Lighting Functions (GPU Compatible)

Reusable lighting functions that work on both CPU and GPU:

```julia (editor=true, logging=false, output=true, id=10)
# Compute lighting from all lights
@inline function compute_multi_light(bvh, lights, ambient, point, normal, mat)
    base_color = to_vec3f(mat.base_color)
    total_color = base_color * ambient

    for i in 1:length(lights)
        light = lights[i]
        light_vec = light.position - point
        light_dist = norm(light_vec)
        light_dir = light_vec / light_dist

        diffuse = max(0.0f0, dot(normal, light_dir))

        # Shadow test
        shadow_ray = Raycore.Ray(o=point + normal * 0.001f0, d=light_dir)
        shadow_hit, _, hit_dist, _ = Raycore.any_hit(bvh, shadow_ray)

        if !shadow_hit || hit_dist >= light_dist
            attenuation = light.intensity / (light_dist * light_dist)
            light_contrib = to_vec3f(light.color) * (diffuse * attenuation)
            total_color += base_color .* light_contrib
        end
    end

    return total_color
end

# Compute reflections
@inline function compute_reflection(bvh, lights, ambient, materials, sky_color,
                                     hit_point, normal, mat, ray)
    # Direct lighting
    direct_color = compute_multi_light(bvh, lights, ambient, hit_point, normal, mat)

    # Reflections for metallic surfaces
    if mat.metallic > 0.0f0
        wo = -ray.d
        reflect_dir = Raycore.reflect(wo, normal)

        # Cast reflection ray
        reflect_ray = Raycore.Ray(o=hit_point + normal * 0.001f0, d=reflect_dir)
        refl_hit, refl_tri, refl_dist, refl_bary = Raycore.closest_hit(bvh, reflect_ray)

        reflection_color = if refl_hit
            refl_point = reflect_ray.o + reflect_ray.d * refl_dist
            refl_normal = compute_normal(refl_tri, refl_bary)
            refl_mat = materials[refl_tri.material_idx]
            compute_multi_light(bvh, lights, ambient, refl_point, refl_normal, refl_mat)
        else
            to_vec3f(sky_color)
        end

        direct_color = direct_color * (1.0f0 - mat.metallic) + reflection_color * mat.metallic
    end

    return direct_color
end

```
## Part 5: GPU Kernel Version 1 - Basic Naive Approach

The simplest GPU kernel - one thread per pixel:

```julia (editor=true, logging=false, output=true, id=12)
import KernelAbstractions as KA
# Basic kernel: one thread per pixel, straightforward implementation
@kernel function raytrace_kernel_v1!(
        img, gpubvh, @Const(lights), @Const(materials),
        width, height, camera_pos, focal_length, aspect, ambient, sky_color
    )
    # Get pixel coordinates
    idx = @index(Global, Linear)

    # Convert linear index to 2D coordinates
    x = ((idx - 1) % width) + 1
    y = ((idx - 1) ÷ width) + 1

    if x <= width && y <= height
        # Generate camera ray
        ray = camera_ray(x, y, width, height, camera_pos, focal_length, aspect)


        # Trace ray using GPU-optimized traversal
        hit_found, triangle, distance, bary_coords = Raycore.closest_hit(gpubvh, ray)

        color = if hit_found
            hit_point = ray.o + ray.d * distance
            normal = compute_normal(triangle, bary_coords)
            mat = materials[triangle.material_idx]

            # Compute lighting with reflections
            compute_reflection(gpubvh, lights, ambient, materials, sky_color,
                               hit_point, normal, mat, ray)
        else
            to_vec3f(sky_color)
        end

        # Write to image
        img[y, x] = to_rgb(color)
    end
end


# New launcher: array-based (backend-agnostic) tracer for kernel v1
# This accepts the image and all scene arrays on the caller side. The caller may pass
# CPU arrays (Array) or GPU arrays (ROCArray) — KernelAbstractions will pick the right backend
# based on the `img` array backend.
function trace_gpu_v1(kernel, img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0,0.7f0,1.0f0))
    height, width = size(img)
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    backend = KA.get_backend(img)
    kernel! = kernel(backend)

    kernel!(
        img, bvh, ctx.lights, ctx.materials,
        Int32(width), Int32(height),
        camera_pos, focal_length, aspect,
        ctx.ambient, sky_color,
        ndrange=width * height
    )
    KA.synchronize(backend)
    return img
end
```
Let's test kernel v1:

```julia (editor=true, logging=false, output=true, id=0)

```
```julia (editor=true, logging=false, output=true, id=27)
using Raycore: to_gpu
img = fill(RGBf(0, 0, 0), 512, 512)
pres = []
img_gpu = ROCArray(img)
bvh_gpu = to_gpu(ROCArray, bvh_cpu)
ctx_gpu = to_gpu(ROCArray, ctx)
img_v1 = trace_gpu_v1(raytrace_kernel_v1!, img_gpu, bvh_gpu, ctx_gpu)
Array(img_v1)
```
```julia (editor=true, logging=false, output=true, id=14)
img = fill(RGBf(0, 0, 0), 512, 512)
img_v1 = trace_gpu_v1(raytrace_kernel_v1!, img, bvh_cpu, ctx)
```
**First GPU render!** This is the simplest approach - one thread per pixel with no optimization.

## Part 6: GPU Kernel Version 2 - 2D Workgroups

Better thread organization using 2D workgroups for improved memory access patterns:

```julia (editor=true, logging=false, output=true, id=16)
# Kernel v2: 2D workgroups for better memory access patterns
@kernel function raytrace_kernel_v2!(
    img, nodes, triangles, original_tris, max_node_prims, @Const(lights), @Const(materials),
    width, height, camera_pos, focal_length, aspect, ambient, sky_color
)
    # Get 2D pixel coordinates directly
    x = @index(Global, NTuple)[1]
    y = @index(Global, NTuple)[2]

    if x <= width && y <= height
        # Generate camera ray
        ray = camera_ray(x, y, width, height, camera_pos, focal_length, aspect)

        # Reconstruct GPUBVH-like struct on-device
        gpubvh = Raycore.GPUBVH(nodes, triangles, original_tris, max_node_prims)

        # Trace ray using GPU-optimized traversal
        hit_found, triangle, distance, bary_coords = Raycore.closest_hit(gpubvh, ray)

        color = if hit_found

        else
            to_vec3f(sky_color)
        end

        # Write to image
        img[y, x] = to_rgb(color)
    end
end


# Convenience launcher that prepares and uploads arrays for GPU use (returns img and preserve)
function trace_v2_from_bvh(img, bvh, ctx; camera_pos=Point3f(0,-0.9,-2.5), fov=45.0f0, sky_color=RGB{Float32}(0.5f0,0.7f0,1.0f0), workgroup_size=(16,16))
    nodes, triangles, original_tris, max_node_prims = prepare_gpubvh_arrays(bvh)
    nodes_gpu, triangles_gpu, original_tris_gpu, lights_gpu, materials_gpu, preserve =
        upload_gpubvh_to_gpu(nodes, triangles, original_tris, ctx.lights, ctx.materials; ArrayType=ROCArray)

    img_out = trace_v2_arrays(img, nodes_gpu, triangles_gpu, original_tris_gpu, max_node_prims, lights_gpu, materials_gpu, ctx;
                              camera_pos=camera_pos, fov=fov, sky_color=sky_color, workgroup_size=workgroup_size)
    return img_out, preserve
end
```
Test kernel v2:

```julia (editor=true, logging=false, output=true, id=18)
img_v2 = trace_gpu_v2(bvh_cpu, ctx)
```
**Better memory access!** 2D workgroups improve cache locality and memory coalescing.

## Part 7: GPU Kernel Version 3 - Tiled Rendering with Shared Memory

Advanced optimization: process tiles of rays together, using shared memory for better performance:

```julia (editor=true, logging=false, output=true, id=20)
# Kernel v3: Tiled rendering approach
@kernel function raytrace_kernel_v3!(
    img, bvh, @Const(lights), @Const(materials),
    width, height, camera_pos, focal_length, aspect, ambient, sky_color
)
    # Get tile coordinates
    tile_x = @index(Group, NTuple)[1]
    tile_y = @index(Group, NTuple)[2]

    # Get local thread coordinates within tile
    local_x = @index(Local, NTuple)[1]
    local_y = @index(Local, NTuple)[2]

    # Compute global pixel coordinates
    x = (tile_x - 1) * @groupsize()[1] + local_x
    y = (tile_y - 1) * @groupsize()[2] + local_y

    if x <= width && y <= height
        # Generate camera ray
        ray = camera_ray(x, y, width, height, camera_pos, focal_length, aspect)

        # Trace ray
        hit_found, triangle, distance, bary_coords = Raycore.closest_hit(bvh, ray)

        color = if hit_found
            hit_point = ray.o + ray.d * distance
            normal = compute_normal(triangle, bary_coords)
            mat = materials[triangle.material_idx]

            # Compute lighting with reflections
            compute_reflection(bvh, lights, ambient, materials, sky_color,
                             hit_point, normal, mat, ray)
        else
            to_vec3f(sky_color)
        end

        # Write to image
        img[y, x] = to_rgb(color)
    end
end

# Launch function for kernel v3
function trace_gpu_v3(bvh, ctx;
                      width=700, height=300,
                      camera_pos=Point3f(0, -0.9, -2.5),
                      fov=45.0f0,
                      sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
                      tile_size=(16, 16))

    # Allocate GPU arrays
    img = ROCArray{RGB{Float32}}(undef, height, width)

    # Transfer data to GPU
    preserve = []
    # Convert CPU BVH to GPU-optimized GPUBVH (pre-transforms triangles)
    gpubvh = Raycore.GPUBVH(bvh)
    bvh_gpu = Raycore.to_gpu(ROCArray, gpubvh; preserve=preserve)
    lights_gpu = ROCArray(ctx.lights)
    materials_gpu = ROCArray(ctx.materials)

    # Camera parameters
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    # Launch kernel with explicit workgroup size
    backend = KA.get_backend(img)
    kernel! = raytrace_kernel_v3!(backend, tile_size)

    kernel!(img, bvh_gpu, lights_gpu, materials_gpu,
            Int32(width), Int32(height), camera_pos, focal_length, aspect,
            ctx.ambient, sky_color,
            ndrange=(width, height))

    KA.synchronize(backend)

    # Transfer back to CPU
    return Array(img)
end
```
Test kernel v3:

```julia (editor=true, logging=false, output=true, id=22)
img_v3 = trace_gpu_v3(bvh_cpu, ctx)
```
**Tiled rendering!** Better workgroup utilization and potential for shared memory optimizations.

## Part 8: Performance Benchmarks

Now let's compare the performance of all three kernels plus the CPU version:

```julia (editor=true, logging=false, output=true, id=24)
# Benchmark parameters
bench_width = 700
bench_height = 300

println("═"^60)
println("Ray Tracing Performance Benchmark")
println("Resolution: $(bench_width)x$(bench_height) = $(bench_width * bench_height) pixels")
println("═"^60)
println()

# Benchmark GPU Kernel v1
println("GPU Kernel v1 (Basic Naive):")
bench_v1 = @benchmark trace_gpu_v1($bvh_cpu, $ctx, width=$bench_width, height=$bench_height) samples=10 seconds=30
display(bench_v1)
println()

# Benchmark GPU Kernel v2
println("GPU Kernel v2 (2D Workgroups):")
bench_v2 = @benchmark trace_gpu_v2($bvh_cpu, $ctx, width=$bench_width, height=$bench_height) samples=10 seconds=30
display(bench_v2)
println()

# Benchmark GPU Kernel v3
println("GPU Kernel v3 (Tiled Rendering):")
bench_v3 = @benchmark trace_gpu_v3($bvh_cpu, $ctx, width=$bench_width, height=$bench_height) samples=10 seconds=30
display(bench_v3)
println()
```
**Performance comparison summary:**

```julia (editor=true, logging=false, output=true, id=26)
using Statistics

# Extract median times
time_v1 = median(bench_v1.times) / 1e6  # Convert to milliseconds
time_v2 = median(bench_v2.times) / 1e6
time_v3 = median(bench_v3.times) / 1e6

# Calculate speedups
baseline = time_v1
speedup_v2 = time_v1 / time_v2
speedup_v3 = time_v1 / time_v3

println("═"^60)
println("Performance Summary")
println("═"^60)
println()
println("Kernel v1 (Basic):           $(round(time_v1, digits=2)) ms")
println("Kernel v2 (2D Workgroups):   $(round(time_v2, digits=2)) ms  ($(round(speedup_v2, digits=2))x)")
println("Kernel v3 (Tiled):           $(round(time_v3, digits=2)) ms  ($(round(speedup_v3, digits=2))x)")
println()

# Performance metrics
pixels_total = bench_width * bench_height
fps_v1 = 1000.0 / time_v1
fps_v2 = 1000.0 / time_v2
fps_v3 = 1000.0 / time_v3

mrays_v1 = (pixels_total / 1e6) * fps_v1
mrays_v2 = (pixels_total / 1e6) * fps_v2
mrays_v3 = (pixels_total / 1e6) * fps_v3

println("Frame Rates:")
println("  Kernel v1: $(round(fps_v1, digits=2)) FPS")
println("  Kernel v2: $(round(fps_v2, digits=2)) FPS")
println("  Kernel v3: $(round(fps_v3, digits=2)) FPS")
println()
println("Ray Throughput:")
println("  Kernel v1: $(round(mrays_v1, digits=2)) MRays/s")
println("  Kernel v2: $(round(mrays_v2, digits=2)) MRays/s")
println("  Kernel v3: $(round(mrays_v3, digits=2)) MRays/s")
println("═"^60)
```
## Summary

We successfully ported our ray tracer to the GPU and explored three different kernel implementations:

**Kernel v1 - Basic Naive:**

  * One thread per pixel with linear indexing
  * Simplest implementation
  * Good baseline for comparison
  * No special optimizations

**Kernel v2 - 2D Workgroups:**

  * 2D thread indexing for better memory access patterns
  * Improved cache locality
  * Better memory coalescing
  * Configurable workgroup size

**Kernel v3 - Tiled Rendering:**

  * Explicit tile-based processing
  * Better workgroup utilization
  * Foundation for shared memory optimizations
  * More control over thread organization

**Key Insights:**

1. **KernelAbstractions.jl** provides portable GPU code that works across different backends (AMD, NVIDIA, etc.)
2. **@Const** annotations help the compiler optimize read-only data access
3. **2D indexing** improves memory access patterns for image-based workloads
4. **Workgroup size** tuning can significantly impact performance
5. **GPU ray tracing** can provide substantial speedups for complex scenes

**Next Steps:**

  * Add multi-sampling/anti-aliasing support
  * Implement more complex materials and effects
  * Optimize BVH traversal for GPU
  * Add path tracing for global illumination
  * Explore wavefront path tracing techniques

Happy GPU ray tracing!
