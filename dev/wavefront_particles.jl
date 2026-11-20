# ==============================================================================
# Particle Simulation - TLAS Instancing Demo with 10k+ Particles
# ==============================================================================
#
# This example demonstrates massive instancing with TLAS:
# - 10,000+ sphere particles sharing a single BLAS
# - Dynamic transforms (position updates each frame) via the handle-based
#   `update_transforms!` + `sync!` API
# - Material changes based on particle velocity (heating effect)
# - Efficient TLAS refit instead of full rebuild
#

using Revise
using Raycore
using Raycore: Mat4f
using KernelAbstractions
using GeometryBasics, Colors, LinearAlgebra
import Makie
using Makie: RGBf
import KernelAbstractions as KA
using ImageCore
using FileIO

# Load helper functions
include("raytracing-core.jl")
include("wavefront-renderer.jl")

# ==============================================================================
# Particle System
# ==============================================================================

struct Particle
    position::Point3f
    velocity::Vec3f
    radius::Float32
end

"""
Particle system - works on CPU (Vector) or GPU (ROCArray, CuArray).
The transforms field is used for TLAS updates.
"""
struct ParticleSystem{ArrParticle, ArrMat4}
    particles::ArrParticle
    transforms::ArrMat4
    bounds_min::Point3f
    bounds_max::Point3f
    gravity::Vec3f
    damping::Float32
end

"""Create a particle system with random initial positions and velocities."""
function ParticleSystem(n_particles::Int;
        bounds_min=Point3f(-50, -50, 0),
        bounds_max=Point3f(50, 50, 100),
        radius_range=(0.3f0, 0.8f0))

    particles = Particle[]

    for _ in 1:n_particles
        pos = Point3f(
            bounds_min[1] + rand(Float32) * (bounds_max[1] - bounds_min[1]),
            bounds_min[2] + rand(Float32) * (bounds_max[2] - bounds_min[2]),
            bounds_min[3] + rand(Float32) * (bounds_max[3] - bounds_min[3])
        )
        vel = Vec3f(
            (rand(Float32) - 0.5f0) * 20,
            (rand(Float32) - 0.5f0) * 20,
            rand(Float32) * 30 + 10
        )
        r = radius_range[1] + rand(Float32) * (radius_range[2] - radius_range[1])
        push!(particles, Particle(pos, vel, r))
    end

    # Build initial transforms
    transforms = [translation(Vec3f(p.position)) * scale(p.radius) for p in particles]

    ParticleSystem(particles, transforms, bounds_min, bounds_max, Vec3f(0, 0, -30), 0.98f0)
end

"""Convert particle system to GPU."""
function Raycore.to_gpu(ArrayType, ps::ParticleSystem)
    ParticleSystem(
        ArrayType(ps.particles),
        ArrayType(ps.transforms),
        ps.bounds_min,
        ps.bounds_max,
        ps.gravity,
        ps.damping
    )
end

# ==============================================================================
# GPU Kernels for ParticleSystem
# ==============================================================================

"""GPU kernel: Physics step - update particles with gravity and boundary bouncing."""
KA.@kernel function particle_physics_kernel!(
    particles,
    bounds_min::Point3f,
    bounds_max::Point3f,
    gravity::Vec3f,
    damping::Float32,
    dt::Float32
)
    i = @index(Global, Linear)
    @inbounds begin
        p = particles[i]
        pos = Vec3f(p.position...)
        vel = p.velocity
        r = p.radius

        new_vel = (vel + gravity * dt) * damping
        new_pos = pos + new_vel * dt

        # Bounce off boundaries
        if new_pos[1] - r < bounds_min[1]
            new_pos = Vec3f(bounds_min[1] + r, new_pos[2], new_pos[3])
            new_vel = Vec3f(-new_vel[1] * 0.8f0, new_vel[2], new_vel[3])
        elseif new_pos[1] + r > bounds_max[1]
            new_pos = Vec3f(bounds_max[1] - r, new_pos[2], new_pos[3])
            new_vel = Vec3f(-new_vel[1] * 0.8f0, new_vel[2], new_vel[3])
        end

        if new_pos[2] - r < bounds_min[2]
            new_pos = Vec3f(new_pos[1], bounds_min[2] + r, new_pos[3])
            new_vel = Vec3f(new_vel[1], -new_vel[2] * 0.8f0, new_vel[3])
        elseif new_pos[2] + r > bounds_max[2]
            new_pos = Vec3f(new_pos[1], bounds_max[2] - r, new_pos[3])
            new_vel = Vec3f(new_vel[1], -new_vel[2] * 0.8f0, new_vel[3])
        end

        if new_pos[3] - r < bounds_min[3]
            new_pos = Vec3f(new_pos[1], new_pos[2], bounds_min[3] + r)
            new_vel = Vec3f(new_vel[1], new_vel[2], -new_vel[3] * 0.8f0)
        elseif new_pos[3] + r > bounds_max[3]
            new_pos = Vec3f(new_pos[1], new_pos[2], bounds_max[3] - r)
            new_vel = Vec3f(new_vel[1], new_vel[2], -new_vel[3] * 0.8f0)
        end

        particles[i] = Particle(Point3f(new_pos...), new_vel, r)
    end
end

"""GPU kernel: Build transform matrices from particles."""
KA.@kernel function build_transforms_kernel!(
    transforms,
    @Const(particles)
)
    i = @index(Global, Linear)
    @inbounds begin
        p = particles[i]
        pos = p.position
        r = p.radius
        # Translation * Scale matrix (column-major)
        transforms[i] = Mat4f(
            r,      0,      0,      0,
            0,      r,      0,      0,
            0,      0,      r,      0,
            pos[1], pos[2], pos[3], 1
        )
    end
end

"""Step particle physics (works on CPU or GPU via KernelAbstractions)."""
function step!(ps::ParticleSystem, dt::Float32)
    n = length(ps.particles)
    backend = KA.get_backend(ps.particles)
    kernel! = particle_physics_kernel!(backend)
    kernel!(ps.particles, ps.bounds_min, ps.bounds_max, ps.gravity, ps.damping, dt, ndrange=n)
    return nothing
end

"""Build transforms from current particle state."""
function build_transforms!(ps::ParticleSystem)
    n = length(ps.particles)
    backend = KA.get_backend(ps.particles)
    kernel! = build_transforms_kernel!(backend)
    kernel!(ps.transforms, ps.particles, ndrange=n)
    return nothing
end

"""
Update sphere material color based on the average particle speed.
Particles all share a single material slot (face_meta=1), so the
heat-color effect drives a single material rather than per-particle.
"""
function update_sphere_material!(materials, ps::ParticleSystem; max_speed::Float32=50.0f0)
    # Pull velocity of one representative particle from the GPU
    p = Array(ps.particles[1:1])[1]
    vel = p.velocity
    speed = sqrt(vel[1]^2 + vel[2]^2 + vel[3]^2)
    t = clamp(speed / max_speed, 0.0f0, 1.0f0)

    color = if t < 0.25f0
        s = t / 0.25f0
        RGB{Float32}(0.1f0, 0.2f0 + 0.5f0 * s, 0.8f0)
    elseif t < 0.5f0
        s = (t - 0.25f0) / 0.25f0
        RGB{Float32}(0.1f0 + 0.6f0 * s, 0.7f0, 0.8f0 - 0.6f0 * s)
    elseif t < 0.75f0
        s = (t - 0.5f0) / 0.25f0
        RGB{Float32}(0.7f0 + 0.3f0 * s, 0.7f0 - 0.4f0 * s, 0.2f0 - 0.1f0 * s)
    else
        s = (t - 0.75f0) / 0.25f0
        RGB{Float32}(1.0f0, 0.3f0 + 0.7f0 * s, 0.1f0 + 0.9f0 * s)
    end

    new_mat = Material(color, 0.6f0, 0.3f0, 1.0f0, 0.0f0)
    # Slot 1 is the sphere material (see create_particle_scene)
    materials[1:1] .= [new_mat]
    return nothing
end

"""
Full GPU update: physics -> transforms -> TLAS update -> sphere material refresh.

`tlas`           — the mutable TLAS owning the sphere instances.
`sphere_handle`  — the TLASHandle returned when the sphere was pushed.
"""
function update_gpu!(renderer_gpu, tlas, sphere_handle, ps::ParticleSystem,
                     dt::Float32; max_speed::Float32=50.0f0)
    # 1. Physics step
    step!(ps, dt)

    # 2. Build transforms from new positions (writes into ps.transforms)
    build_transforms!(ps)

    # 3. Stage new transforms for the sphere instance group
    Raycore.update_transforms!(tlas, sphere_handle, ps.transforms)

    # 4. Commit: refit the TLAS BVH in place (no topology change → fast refit path)
    Raycore.sync!(tlas)

    # 5. Refresh the sphere's material to reflect speed
    update_sphere_material!(renderer_gpu.ctx.materials, ps; max_speed=max_speed)

    return nothing
end

# ==============================================================================
# Scene Creation with Instanced Spheres
# ==============================================================================

"""Create a translation matrix."""
function translation(v::Union{Vec3f, Point3f})::Mat4f
    Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        v[1], v[2], v[3], 1
    )
end

"""Create a uniform scale matrix."""
function scale(s::Float32)::Mat4f
    Mat4f(
        s, 0, 0, 0,
        0, s, 0, 0,
        0, 0, s, 0,
        0, 0, 0, 1
    )
end

"""Map velocity magnitude to a heat color (blue -> red -> yellow -> white)."""
function velocity_to_color(speed::Float32, max_speed::Float32=50.0f0)::RGB{Float32}
    t = clamp(speed / max_speed, 0.0f0, 1.0f0)
    if t < 0.25f0
        s = t / 0.25f0
        RGB{Float32}(0.1f0, 0.2f0 + 0.5f0 * s, 0.8f0)
    elseif t < 0.5f0
        s = (t - 0.25f0) / 0.25f0
        RGB{Float32}(0.1f0 + 0.6f0 * s, 0.7f0, 0.8f0 - 0.6f0 * s)
    elseif t < 0.75f0
        s = (t - 0.5f0) / 0.25f0
        RGB{Float32}(0.7f0 + 0.3f0 * s, 0.7f0 - 0.4f0 * s, 0.2f0 - 0.1f0 * s)
    else
        s = (t - 0.75f0) / 0.25f0
        RGB{Float32}(1.0f0, 0.3f0 + 0.7f0 * s, 0.1f0 + 0.9f0 * s)
    end
end

"""
Tag every face of `mesh` with the same metadata value `meta`.

Used so that all triangles of a given BLAS look up the same material slot
(`materials[meta]`) inside the renderer's `tri.metadata` path.
"""
function mesh_with_const_meta(mesh::GeometryBasics.Mesh, meta::UInt32)
    fs = decompose(TriangleFace{UInt32}, mesh)
    n_faces = length(fs)
    face_meta = fill(meta, n_faces)
    return GeometryBasics.mesh(mesh; face_meta=GeometryBasics.per_face(face_meta, mesh))
end

"""
Create a TLAS with instanced spheres for the particle system.

Architecture:
- Single unit-sphere BLAS, instanced once per particle (one `push!` with all
  initial transforms returns one `TLASHandle` covering the whole batch).
- Three static wall BLASes (floor, back wall, left wall) at identity transform.
- Each BLAS has `face_meta` set to a constant so the renderer's
  `materials[tri.metadata]` lookup stays in-bounds:
    1 → all sphere triangles
    2 → floor
    3 → back wall
    4 → left wall

Returns `(tlas, sphere_handle, ctx, materials)`. Hand `sphere_handle` to
`update_transforms!` each frame.
"""
function create_particle_scene(ps::ParticleSystem; backend=KA.CPU())
    n_particles = length(ps.particles)
    println("Creating particle scene with $n_particles particles...")

    # ----- Geometry -----
    unit_sphere = normal_mesh(Tesselation(Sphere(Point3f(0), 1.0f0), 16))
    sphere_tagged = mesh_with_const_meta(unit_sphere, UInt32(1))

    bmin = ps.bounds_min
    bmax = ps.bounds_max
    pad = 10.0f0
    wall_thickness = 1.0f0

    floor_mesh = normal_mesh(Rect3f(
        Vec3f(bmin[1] - pad, bmin[2] - pad, bmin[3] - wall_thickness),
        Vec3f(bmax[1] - bmin[1] + 2pad, bmax[2] - bmin[2] + 2pad, wall_thickness)
    ))
    back_wall_mesh = normal_mesh(Rect3f(
        Vec3f(bmin[1] - pad, bmin[2] - pad - wall_thickness, bmin[3]),
        Vec3f(bmax[1] - bmin[1] + 2pad, wall_thickness, bmax[3] - bmin[3] + pad)
    ))
    left_wall_mesh = normal_mesh(Rect3f(
        Vec3f(bmin[1] - pad - wall_thickness, bmin[2] - pad, bmin[3]),
        Vec3f(wall_thickness, bmax[2] - bmin[2] + 2pad, bmax[3] - bmin[3] + pad)
    ))
    floor_tagged    = mesh_with_const_meta(floor_mesh,    UInt32(2))
    back_tagged     = mesh_with_const_meta(back_wall_mesh, UInt32(3))
    left_tagged     = mesh_with_const_meta(left_wall_mesh, UInt32(4))

    # ----- TLAS -----
    println("Building TLAS on backend $backend...")
    tlas = Raycore.TLAS(backend)

    # Initial sphere transforms (CPU; push! adapts to backend internally)
    initial_transforms = Mat4f[
        translation(Vec3f(p.position)) * scale(p.radius)
        for p in ps.particles
    ]

    sphere_handle = push!(tlas, sphere_tagged, initial_transforms)
    push!(tlas, floor_tagged)
    push!(tlas, back_tagged)
    push!(tlas, left_tagged)

    Raycore.sync!(tlas)
    println("  TLAS instances: $(Raycore.n_instances(tlas))  geometries: $(Raycore.n_geometries(tlas))")

    # ----- Materials (one slot per face_meta tag) -----
    p1 = ps.particles[1]
    sphere_mat = Material(velocity_to_color(norm(p1.velocity)), 0.6f0, 0.3f0, 1.0f0, 0.0f0)
    floor_mat  = Material(RGB(0.9f0, 0.9f0, 0.92f0), 1.0f0, 0.05f0, 1.0f0, 0.0f0)
    back_mat   = Material(RGB(0.8f0, 0.75f0, 0.7f0), 0.9f0, 0.1f0,  1.0f0, 0.0f0)
    left_mat   = Material(RGB(0.95f0, 0.95f0, 0.95f0), 1.0f0, 0.02f0, 1.0f0, 0.0f0)
    materials  = [sphere_mat, floor_mat, back_mat, left_mat]

    # ----- Lights -----
    lights = [
        PointLight(Point3f(0, 0, 120),    3000.0f0,  RGB(1.0f0, 0.98f0, 0.95f0)),
        PointLight(Point3f(80, 70, 50),   15000.0f0, RGB(1.0f0, 1.0f0, 1.0f0)),
        PointLight(Point3f(-60, 40, 60),  8000.0f0,  RGB(0.9f0, 0.92f0, 1.0f0)),
    ]

    ctx = RenderContext(lights, materials, 0.25f0)
    return tlas, sphere_handle, ctx, materials
end

# ==============================================================================
# Main Animation
# ==============================================================================

println("\n" * "="^70)
println("Particle Simulation - 10k Instanced Spheres Demo")
println("="^70)

# Create particle system
n_particles = 10_000
println("\nInitializing $n_particles particles...")
ps = ParticleSystem(n_particles;
    bounds_min=Point3f(-40, -40, 0),
    bounds_max=Point3f(40, 40, 80)
)

# Backend selection — TLAS arrays must live on the same backend the renderer
# uses, otherwise refit kernels and intersect kernels would talk to different
# memory spaces.
using AMDGPU: ROCBackend
backend = ROCBackend()

# Create scene on the GPU backend so refits update arrays in place that the
# renderer's StaticTLAS reads.
tlas, sphere_handle, ctx, materials = create_particle_scene(ps; backend=backend)

# Animation parameters
num_frames = 120
dt = 1.0f0 / 30.0f0
width, height = 1920, 1080
frames_dir = "particle_frames"

camera_pos = Point3f(70, 55, 45)
camera_lookat = Point3f(0, 0, 30)

println("\nRendering $num_frames frames...")
println("  Resolution: $(width)x$(height)")
println("  Particles: $n_particles")
println("  Output: $frames_dir/")

isdir(frames_dir) && rm(frames_dir; recursive=true)
mkpath(frames_dir)

# Create renderer
img = fill(RGBf(0, 0, 0), height, width)
renderer = WavefrontRenderer(img, tlas, ctx;
    camera_pos=camera_pos,
    camera_lookat=camera_lookat,
    camera_up=Vec3f(0, 0, 1),
    fov=50.0f0,
    sky_color=RGB{Float32}(0.05f0, 0.05f0, 0.1f0),
    samples_per_pixel=8
);
renderer_gpu = to_gpu(ROCArray, renderer);

# Warmup
println("\nWarmup render...")
Array(render!(renderer_gpu))

# GPU particle system
ps_gpu = to_gpu(ROCArray, ps)
println("GPU particle system created")

# Animation loop
println("\nRendering animation...")
total_time = @elapsed begin
    for frame in 1:num_frames
        update_gpu!(renderer_gpu, tlas, sphere_handle, ps_gpu, dt)

        fill!(renderer_gpu.framebuffer, RGBf(0, 0, 0))
        render!(renderer_gpu)

        filename = joinpath(frames_dir, "frame_$(lpad(frame, 4, '0')).png")
        save(filename, Array(map(clamp01nan, renderer_gpu.framebuffer)))

        if frame % 20 == 0 || frame == 1
            println("  Frame $frame/$num_frames")
        end
    end
end

avg_fps = num_frames / total_time
println("\nAnimation complete!")
println("  Total time: $(round(total_time, digits=2))s")
println("  Average FPS: $(round(avg_fps, digits=2))")
println("  Time per frame: $(round(total_time/num_frames*1000, digits=1))ms")

# Create video
println("\nCreating video...")
using FFMPEG_jll
video_output = "particles.mp4"
run(`$(FFMPEG_jll.ffmpeg()) -y -framerate 30 -i $(frames_dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 $video_output`)
println("Video saved: $video_output")

# Performance stats
println("\n" * "="^70)
println("Performance Statistics (GPU)")
println("="^70)

using BenchmarkTools

gpu_physics_time = @belapsed step!($ps_gpu, $dt)
println("  GPU Physics step: $(round(gpu_physics_time * 1000, digits=3)) ms")

gpu_transform_time = @belapsed build_transforms!($ps_gpu)
println("  GPU Transform build: $(round(gpu_transform_time * 1000, digits=3)) ms")

gpu_update_time = @belapsed update_gpu!($renderer_gpu, $tlas, $sphere_handle, $ps_gpu, $dt)
println("  GPU Full update: $(round(gpu_update_time * 1000, digits=3)) ms")

gpu_render_time = @belapsed begin
    fill!($renderer_gpu.framebuffer, RGBf(0, 0, 0))
    render!($renderer_gpu)
end
println("  GPU Render: $(round(gpu_render_time * 1000, digits=1)) ms")

total_frame_time = gpu_update_time + gpu_render_time
println("\n  Total frame time: $(round(total_frame_time * 1000, digits=1)) ms")
println("  Theoretical max FPS: $(round(1.0 / total_frame_time, digits=1))")

# Per-mesh stats from the mutable TLAS
sphere_blas = tlas.blas_storage[1]
floor_blas  = tlas.blas_storage[2]
println("\n  Total triangles: $(n_particles * length(sphere_blas.primitives) + length(floor_blas.primitives))")
println("  Unique BLAS geometries: $(length(tlas.blas_storage))")
println("  Memory saved by instancing: ~$(round(n_particles * length(sphere_blas.primitives) * 48 / 1024 / 1024, digits=1)) MB")

println("\nDone!")
