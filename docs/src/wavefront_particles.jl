# ==============================================================================
# Particle Simulation - TLAS Instancing Demo with 10k+ Particles
# ==============================================================================
#
# This example demonstrates massive instancing with TLAS:
# - 10,000+ sphere particles sharing a single BLAS
# - Dynamic transforms (position updates each frame)
# - Material changes based on particle velocity (heating effect)
# - Efficient TLAS refit instead of full rebuild
#

using Revise
using Raycore
using Raycore: update_instance_transform!, refit_tlas!, Mat4f, TLAS, BLAS
using Raycore: build_blas, build_tlas, InstanceDescriptor, Triangle
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

# CPU step! removed - we use GPU kernels for everything now

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

        # Apply gravity and damping
        new_vel = (vel + gravity * dt) * damping

        # Update position
        new_pos = pos + new_vel * dt

        # Bounce off boundaries (unrolled for GPU efficiency)
        # X axis
        if new_pos[1] - r < bounds_min[1]
            new_pos = Vec3f(bounds_min[1] + r, new_pos[2], new_pos[3])
            new_vel = Vec3f(-new_vel[1] * 0.8f0, new_vel[2], new_vel[3])
        elseif new_pos[1] + r > bounds_max[1]
            new_pos = Vec3f(bounds_max[1] - r, new_pos[2], new_pos[3])
            new_vel = Vec3f(-new_vel[1] * 0.8f0, new_vel[2], new_vel[3])
        end

        # Y axis
        if new_pos[2] - r < bounds_min[2]
            new_pos = Vec3f(new_pos[1], bounds_min[2] + r, new_pos[3])
            new_vel = Vec3f(new_vel[1], -new_vel[2] * 0.8f0, new_vel[3])
        elseif new_pos[2] + r > bounds_max[2]
            new_pos = Vec3f(new_pos[1], bounds_max[2] - r, new_pos[3])
            new_vel = Vec3f(new_vel[1], -new_vel[2] * 0.8f0, new_vel[3])
        end

        # Z axis
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

"""GPU kernel: Update materials based on particle velocities."""
KA.@kernel function update_materials_kernel!(
    materials,
    @Const(particles),
    max_speed::Float32
)
    i = @index(Global, Linear)
    @inbounds begin
        vel = particles[i].velocity
        speed = sqrt(vel[1]^2 + vel[2]^2 + vel[3]^2)

        # Velocity to heat color
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

        # Noise for metallic/roughness variety
        noise = sin(Float32(i) * 0.1f0) * 0.5f0 + 0.5f0
        metallic = 0.3f0 + noise * 0.6f0
        roughness = 0.2f0 + (1.0f0 - noise) * 0.4f0

        materials[i] = Material(color, metallic, roughness, 1.0f0, 0.0f0)
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

"""Update materials based on particle velocities."""
function update_materials!(materials, ps::ParticleSystem; max_speed::Float32=50.0f0)
    n = length(ps.particles)
    backend = KA.get_backend(ps.particles)
    kernel! = update_materials_kernel!(backend)
    kernel!(materials, ps.particles, max_speed, ndrange=n)
    return nothing
end

"""
Full GPU update: physics -> transforms -> TLAS update -> materials.
All operations stay on GPU with no CPU roundtrips.
"""
function update_gpu!(renderer_gpu, ps::ParticleSystem, dt::Float32; backend, max_speed::Float32=50.0f0)
    n = length(ps.particles)
    # 1. Physics step
    step!(ps, dt)

    # 2. Build transforms from new positions
    build_transforms!(ps)

    # 3. Update TLAS instance transforms
    Raycore.update_instance_transforms!(renderer_gpu.bvh, ps.transforms, n)

    # 4. Refit TLAS AABBs
    Raycore.refit_tlas!(renderer_gpu.bvh; backend=backend)

    # 5. Update materials based on velocities
    update_materials!(renderer_gpu.ctx.materials, ps; max_speed=max_speed)

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
        # Blue to cyan
        s = t / 0.25f0
        RGB{Float32}(0.1f0, 0.2f0 + 0.5f0 * s, 0.8f0)
    elseif t < 0.5f0
        # Cyan to green-yellow
        s = (t - 0.25f0) / 0.25f0
        RGB{Float32}(0.1f0 + 0.6f0 * s, 0.7f0, 0.8f0 - 0.6f0 * s)
    elseif t < 0.75f0
        # Yellow to orange-red
        s = (t - 0.5f0) / 0.25f0
        RGB{Float32}(0.7f0 + 0.3f0 * s, 0.7f0 - 0.4f0 * s, 0.2f0 - 0.1f0 * s)
    else
        # Red to bright white-yellow (hot!)
        s = (t - 0.75f0) / 0.25f0
        RGB{Float32}(1.0f0, 0.3f0 + 0.7f0 * s, 0.1f0 + 0.9f0 * s)
    end
end

"""Convert a mesh to triangles with a given metadata value.
Filters out degenerate (zero-area) triangles that can cause rendering artifacts.
UV-sphere tessellations create degenerate triangles at poles where vertices coincide."""
function mesh_to_triangles(mesh, metadata::UInt32; min_area::Float32=1.0f-7)
    tri_mesh = Raycore.to_triangle_mesh(mesh)
    triangles = Raycore.Triangle{UInt32}[]
    vertices = tri_mesh.vertices

    for i in 1:div(length(tri_mesh.indices), 3)
        # Get vertex positions for this triangle
        i0 = tri_mesh.indices[3*(i-1) + 1]
        i1 = tri_mesh.indices[3*(i-1) + 2]
        i2 = tri_mesh.indices[3*(i-1) + 3]

        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        # Check triangle area to filter degenerate triangles
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = norm(cross(Vec3f(edge1...), Vec3f(edge2...))) / 2

        if area >= min_area
            push!(triangles, Raycore.Triangle(tri_mesh, i, metadata))
        end
    end
    return triangles
end

"""
Create a TLAS with instanced spheres for the particle system.
All particles share a single unit sphere BLAS.
"""
function create_particle_scene(ps::ParticleSystem)
    n_particles = length(ps.particles)
    println("Creating particle scene with $n_particles particles...")

    # Create a single unit sphere BLAS (shared by all particles)
    unit_sphere = normal_mesh(Tesselation(Sphere(Point3f(0), 1.0f0), 16))
    sphere_triangles = mesh_to_triangles(unit_sphere, UInt32(1))
    sphere_blas = build_blas(sphere_triangles)
    println("  Unit sphere BLAS: $(length(sphere_blas.primitives)) triangles")

    # Box dimensions with some padding
    bmin = ps.bounds_min
    bmax = ps.bounds_max
    pad = 10.0f0
    wall_thickness = 1.0f0

    # Create floor BLAS (bottom)
    floor_mesh = normal_mesh(Rect3f(
        Vec3f(bmin[1] - pad, bmin[2] - pad, bmin[3] - wall_thickness),
        Vec3f(bmax[1] - bmin[1] + 2pad, bmax[2] - bmin[2] + 2pad, wall_thickness)
    ))
    floor_triangles = mesh_to_triangles(floor_mesh, UInt32(n_particles + 1))
    floor_blas = build_blas(floor_triangles)

    # Create back wall BLAS (negative Y side)
    back_wall_mesh = normal_mesh(Rect3f(
        Vec3f(bmin[1] - pad, bmin[2] - pad - wall_thickness, bmin[3]),
        Vec3f(bmax[1] - bmin[1] + 2pad, wall_thickness, bmax[3] - bmin[3] + pad)
    ))
    back_wall_triangles = mesh_to_triangles(back_wall_mesh, UInt32(n_particles + 2))
    back_wall_blas = build_blas(back_wall_triangles)

    # Create left wall BLAS (negative X side)
    left_wall_mesh = normal_mesh(Rect3f(
        Vec3f(bmin[1] - pad - wall_thickness, bmin[2] - pad, bmin[3]),
        Vec3f(wall_thickness, bmax[2] - bmin[2] + 2pad, bmax[3] - bmin[3] + pad)
    ))
    left_wall_triangles = mesh_to_triangles(left_wall_mesh, UInt32(n_particles + 3))
    left_wall_blas = build_blas(left_wall_triangles)

    # BLAS array: [sphere, floor, back_wall, left_wall]
    # Note: front and right walls omitted so camera can see inside the box
    blas_array = [sphere_blas, floor_blas, back_wall_blas, left_wall_blas]

    # Create instances for each particle
    instances = InstanceDescriptor[]

    # Particle instances (BLAS index 1 = sphere)
    for (i, p) in enumerate(ps.particles)
        transform = translation(Vec3f(p.position)) * scale(p.radius)
        inv_transform = scale(1.0f0 / p.radius) * translation(-Vec3f(p.position))

        inst = InstanceDescriptor(
            UInt32(1),           # blas_index (1-indexed)
            UInt32(i),           # instance_id (for material lookup)
            transform,
            inv_transform,
            UInt32(0)            # flags
        )
        push!(instances, inst)
    end

    # Wall instances (BLAS indices 2-4: floor, back_wall, left_wall)
    for wall_idx in 2:4
        wall_inst = InstanceDescriptor(
            UInt32(wall_idx),
            UInt32(n_particles + wall_idx - 1),
            Mat4f(I),
            Mat4f(I),
            UInt32(0)
        )
        push!(instances, wall_inst)
    end

    println("  Total instances: $(length(instances))")

    # Build TLAS
    tlas = build_tlas(blas_array, instances)
    println("  TLAS nodes: $(length(tlas.nodes))")

    # Create materials (one per particle + walls)
    materials = Material[]
    for (i, p) in enumerate(ps.particles)
        speed = norm(p.velocity)
        color = velocity_to_color(speed)
        # Add noise to metallic based on particle index for variety
        noise = sin(Float32(i) * 0.1f0) * 0.5f0 + 0.5f0  # 0 to 1
        metallic = 0.3f0 + noise * 0.6f0  # Range 0.3 to 0.9
        roughness = 0.2f0 + (1.0f0 - noise) * 0.4f0  # Inverse relationship with metallic
        push!(materials, Material(color, metallic, roughness, 1.0f0, 0.0f0))
    end
    # Floor material - mirror-like
    push!(materials, Material(RGB(0.9f0, 0.9f0, 0.92f0), 1.0f0, 0.05f0, 1.0f0, 0.0f0))
    # Back wall - metallic with slight color
    push!(materials, Material(RGB(0.8f0, 0.75f0, 0.7f0), 0.9f0, 0.1f0, 1.0f0, 0.0f0))
    # Left wall - mirror
    push!(materials, Material(RGB(0.95f0, 0.95f0, 0.95f0), 1.0f0, 0.02f0, 1.0f0, 0.0f0))

    # Lights - key light from above, fill light from camera direction, soft rim light
    lights = [
        PointLight(Point3f(0, 0, 120), 3000.0f0, RGB(1.0f0, 0.98f0, 0.95f0)),    # Key light (softer)
        PointLight(Point3f(80, 70, 50), 15000.0f0, RGB(1.0f0, 1.0f0, 1.0f0)),      # Fill from camera direction
        PointLight(Point3f(-60, 40, 60), 8000.0f0, RGB(0.9f0, 0.92f0, 1.0f0)),     # Soft rim light
    ]

    ctx = RenderContext(lights, materials, 0.25f0)  # Higher ambient for softer shadows

    return tlas, ctx, materials
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

# Create scene
tlas, ctx, materials = create_particle_scene(ps)

# Animation parameters
num_frames = 120
dt = 1.0f0 / 30.0f0
width, height = 1920, 1080
frames_dir = "particle_frames"

# Camera setup - closer to the action
camera_pos = Point3f(70, 55, 45)
camera_lookat = Point3f(0, 0, 30)

println("\nRendering $num_frames frames...")
println("  Resolution: $(width)x$(height)")
println("  Particles: $n_particles")
println("  Output: $frames_dir/")

# Create output directory
isdir(frames_dir) && rm(frames_dir; recursive=true)
mkpath(frames_dir)

# Create renderer
img = fill(RGBf(0, 0, 0), height, width)
renderer = WavefrontRenderer(img, tlas, ctx;
    camera_pos=camera_pos,
    camera_lookat=camera_lookat,
    camera_up=Vec3f(0, 0, 1),
    fov=50.0f0,
    sky_color=RGB{Float32}(0.05f0, 0.05f0, 0.1f0),  # Dark sky
    samples_per_pixel=8
);
renderer_gpu = to_gpu(ROCArray, renderer);
# Warmup
println("\nWarmup render...")
Array(render!(renderer_gpu))
# CPU  6.464 s (17243741 allocations: 2.01 GiB)
# GPU 590.115 ms (1310 allocations: 52.91 KiB)

# Create GPU particle system
using AMDGPU: ROCBackend
ps_gpu = to_gpu(ROCArray, ps)
println("GPU particle system created")

# Animation loop (fully GPU-accelerated)
println("\nRendering animation...")
total_time = @elapsed begin
    for frame in 1:num_frames
        # Full GPU update: physics -> transforms -> TLAS -> materials
        update_gpu!(renderer_gpu, ps_gpu, dt; backend=ROCBackend())

        # Render
        fill!(renderer_gpu.framebuffer, RGBf(0, 0, 0))
        render!(renderer_gpu)

        # Save frame (only GPU->CPU transfer is the framebuffer)
        filename = joinpath(frames_dir, "frame_$(lpad(frame, 4, '0')).png")
        save(filename, Array(map(clamp01nan, renderer_gpu.framebuffer)))

        if frame % 20 == 0 || frame == 1
            # For progress reporting, we can skip velocity stats or do a quick GPU reduction
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

# Benchmark GPU physics step
gpu_physics_time = @belapsed step!($ps_gpu, $dt)
println("  GPU Physics step: $(round(gpu_physics_time * 1000, digits=3)) ms")

# Benchmark GPU transform building
gpu_transform_time = @belapsed build_transforms!($ps_gpu)
println("  GPU Transform build: $(round(gpu_transform_time * 1000, digits=3)) ms")

# Benchmark full GPU update (physics + transforms + TLAS + materials)
gpu_update_time = @belapsed update_gpu!($renderer_gpu, $ps_gpu, $dt; backend=ROCBackend())
println("  GPU Full update: $(round(gpu_update_time * 1000, digits=3)) ms")

# Benchmark GPU render
gpu_render_time = @belapsed begin
    fill!($renderer_gpu.framebuffer, RGBf(0, 0, 0))
    render!($renderer_gpu)
end
println("  GPU Render: $(round(gpu_render_time * 1000, digits=1)) ms")

# Total frame time
total_frame_time = gpu_update_time + gpu_render_time
println("\n  Total frame time: $(round(total_frame_time * 1000, digits=1)) ms")
println("  Theoretical max FPS: $(round(1.0 / total_frame_time, digits=1))")

println("\n  Total triangles: $(n_particles * length(tlas.blas_array[1].primitives) + length(tlas.blas_array[2].primitives))")
println("  Unique BLAS geometries: $(length(tlas.blas_array))")
println("  Memory saved by instancing: ~$(round(n_particles * length(tlas.blas_array[1].primitives) * 48 / 1024 / 1024, digits=1)) MB")

println("\nDone!")
