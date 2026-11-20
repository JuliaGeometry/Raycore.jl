# Dynamic Scene Example - Animated TLAS with Transform Updates
#
# This example demonstrates efficient dynamic scene updates using TLAS.
# Instead of rebuilding the entire acceleration structure each frame,
# we update instance transforms and refit the TLAS - much faster!

using Revise
using Raycore
using KernelAbstractions
using GeometryBasics, Colors, LinearAlgebra
import Makie
using Makie: RGBf
import KernelAbstractions as KA
using FileIO, ImageCore

# Load helper functions
include("raytracing-core.jl")
include("wavefront-renderer.jl")

"""
Create a rotation matrix around the Y axis.
"""
function rotation_y(angle::Float32)::Mat4f
    c, s = cos(angle), sin(angle)
    Mat4f(
        c,  0, s, 0,
        0,  1, 0, 0,
        -s, 0, c, 0,
        0,  0, 0, 1
    )
end

"""
Create a translation matrix.
"""
function translation(x::Float32, y::Float32, z::Float32)::Mat4f
    Mat4f(
        1, 0, 0, x,
        0, 1, 0, y,
        0, 0, 1, z,
        0, 0, 0, 1
    )
end

"""
Create a scale matrix.
"""
function scaling(sx::Float32, sy::Float32, sz::Float32)::Mat4f
    Mat4f(
        sx, 0,  0,  0,
        0,  sy, 0,  0,
        0,  0,  sz, 0,
        0,  0,  0,  1
    )
end

"""
Create a dynamic scene with objects that can be animated.
Returns `(tlas, handles, ctx)` — `handles[i]` is the `TLASHandle` for geometry `i`.
"""
function create_dynamic_scene()
    sphere1  = Tesselation(Sphere(Point3f(0, 0, 0), 0.5f0), 32)
    sphere2  = Tesselation(Sphere(Point3f(0, 0, 0), 0.4f0), 32)
    cube     = normal_mesh(Rect3f(Vec3f(-0.4f0), Vec3f(0.8f0)))
    floor    = normal_mesh(Rect3f(Vec3f(-4, -1, -4), Vec3f(8, 0.01, 8)))
    back_wall = normal_mesh(Rect3f(Vec3f(-4, -1, 4), Vec3f(8, 4, 0.01)))

    materials = [
        Material(RGB(0.9f0, 0.2f0, 0.2f0), 0.3f0, 0.4f0, 1.0f0, 0.0f0),
        Material(RGB(0.2f0, 0.9f0, 0.2f0), 0.5f0, 0.2f0, 1.0f0, 0.0f0),
        Material(RGB(0.2f0, 0.2f0, 0.9f0), 0.0f0, 0.6f0, 1.0f0, 0.0f0),
        Material(RGB(0.4f0, 0.4f0, 0.4f0), 0.0f0, 0.9f0, 1.0f0, 0.0f0),
        Material(RGB(0.8f0, 0.7f0, 0.6f0), 0.0f0, 0.8f0, 1.0f0, 0.0f0),
    ]

    println("Building initial TLAS...")
    tlas = Raycore.TLAS(KernelAbstractions.CPU())
    handles = [push!(tlas, normal_mesh(g)) for g in [sphere1, sphere2, cube, floor, back_wall]]
    Raycore.sync!(tlas)
    println("TLAS created with $(Raycore.n_instances(tlas)) instances")

    lights = default_lights()
    ctx = RenderContext(lights, materials, 0.1f0)

    return tlas, handles, ctx
end

"""
Update transforms for frame t (0 to 1 for one animation cycle).
"""
function update_scene!(tlas::Raycore.TLAS, handles::Vector, t::Float32)
    orbit_radius = 1.5f0
    orbit_angle  = t * 2f0 * Float32(pi)
    Raycore.update_transform!(tlas, handles[1],
        translation(orbit_radius * cos(orbit_angle), 0.0f0, orbit_radius * sin(orbit_angle)))

    bounce_height = 0.5f0 + 0.8f0 * abs(sin(t * 4f0 * Float32(pi)))
    Raycore.update_transform!(tlas, handles[2], translation(-1.5f0, bounce_height, 0.0f0))

    cube_x = 1.0f0 * sin(t * 2f0 * Float32(pi))
    Raycore.update_transform!(tlas, handles[3],
        translation(cube_x, 0.0f0, -1.0f0) * rotation_y(t * 3f0 * Float32(pi)))

    # handles[4] (floor) and handles[5] (wall) are static — no update needed
    Raycore.sync!(tlas)
end

"""
Render a single frame.
"""
function render_frame!(img, tlas, ctx)
    renderer = WavefrontRenderer(img, tlas, ctx)
    render!(renderer)
end

# ==============================================================================
# Main Animation Loop
# ==============================================================================

println("\n" * "="^70)
println("Dynamic Scene Example - Animated TLAS")
println("="^70)

# Create scene
tlas, ctx = create_dynamic_scene()

# Animation parameters
num_frames = 60
width, height = 400, 300

println("\nRendering $num_frames frames...")
println("  Resolution: $(width)x$(height)")

# Render animation frames
frames = Vector{Matrix{RGB{Float32}}}(undef, num_frames)

# Warmup
img_warmup = fill(RGBf(0, 0, 0), height, width)
update_scene!(tlas, 0.0f0)
render_frame!(img_warmup, tlas, ctx)

# Time the animation
total_time = @elapsed begin
    for frame in 1:num_frames
        t = Float32((frame - 1) / num_frames)

        # Update transforms and refit TLAS
        update_scene!(tlas, t)

        # Render frame
        img = fill(RGBf(0, 0, 0), height, width)
        render_frame!(img, tlas, ctx)
        frames[frame] = img

        if frame % 10 == 0
            println("  Frame $frame/$num_frames")
        end
    end
end

avg_fps = num_frames / total_time
println("\nAnimation complete!")
println("  Total time: $(round(total_time, digits=2))s")
println("  Average FPS: $(round(avg_fps, digits=1))")

# Save first, middle, and last frames
println("\nSaving sample frames...")
save("dynamic_frame_001.png", map(clamp01nan, frames[1]))
save("dynamic_frame_030.png", map(clamp01nan, frames[30]))
save("dynamic_frame_060.png", map(clamp01nan, frames[60]))
println("Saved: dynamic_frame_001.png, dynamic_frame_030.png, dynamic_frame_060.png")

# Benchmark refit vs rebuild
println("\n" * "="^70)
println("Benchmarking: Refit vs Rebuild")
println("="^70)

using BenchmarkTools

# Benchmark refit only
refit_time = @belapsed begin
    update_scene!($tlas, 0.5f0)
end

println("  Refit time: $(round(refit_time * 1000, digits=3)) ms")

# For comparison: time to rebuild entire TLAS
geometries = [
    Tesselation(Sphere(Point3f(0, 0, 0), 0.5f0), 32),
    Tesselation(Sphere(Point3f(0, 0, 0), 0.4f0), 32),
    normal_mesh(Rect3f(Vec3f(-0.4f0), Vec3f(0.8f0))),
    normal_mesh(Rect3f(Vec3f(-4, -1, -4), Vec3f(8, 0.01, 8))),
    normal_mesh(Rect3f(Vec3f(-4, -1, 4), Vec3f(8, 4, 0.01))),
]

rebuild_time = @belapsed begin
    Raycore.TLAS($geometries, (mesh_idx, tri_idx) -> UInt32(mesh_idx))
end

println("  Rebuild time: $(round(rebuild_time * 1000, digits=3)) ms")
println("  Speedup: $(round(rebuild_time / refit_time, digits=1))x faster with refit!")
