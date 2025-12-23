# ==============================================================================
# Animated Lego Figure - TLAS Dynamic Scene Demo
# ==============================================================================
#
# This example demonstrates the power of instanced BVH (TLAS/BLAS) for animated scenes.
# Each body part is a separate BLAS instance that can be transformed independently.
# Animation updates only require refitting the TLAS - no geometry rebuild needed!
#
# Based on the Lego figure model by Kevin-Mattheus-Moerman
# https://twitter.com/KMMoerman/status/1417759722963415041

using Revise
using Raycore
using Raycore: update_instance_transform!, refit_tlas!, Mat4f
using KernelAbstractions
using GeometryBasics, Colors, LinearAlgebra
using FileIO, MeshIO
import Makie
using Makie: RGBf
import KernelAbstractions as KA
using ImageCore

# Load helper functions
include("raytracing-core.jl")
include("wavefront-renderer.jl")

const IDENTITY = Mat4f(I)

# ==============================================================================
# Transform Utilities
# ==============================================================================

"""Create a rotation matrix around an arbitrary axis (Rodrigues' formula).
Note: Mat4f uses column-major storage, so values are specified column by column."""
function rotation_axis(axis::Vec3f, angle::Float32)::Mat4f
    axis = normalize(axis)
    c, s = cos(angle), sin(angle)
    t = 1.0f0 - c
    x, y, z = axis[1], axis[2], axis[3]

    # Column-major: each group of 4 values is one column
    Mat4f(
        t*x*x + c,     t*x*y + s*z,   t*x*z - s*y,   0,  # column 1
        t*x*y - s*z,   t*y*y + c,     t*y*z + s*x,   0,  # column 2
        t*x*z + s*y,   t*y*z - s*x,   t*z*z + c,     0,  # column 3
        0,             0,             0,             1   # column 4
    )
end

"""Create a translation matrix.
Note: Mat4f uses column-major storage, translation goes in column 4."""
function translation(v::Union{Vec3f, Point3f})::Mat4f
    # Column-major: translation in column 4, rows 1-3
    Mat4f(
        1, 0, 0, 0,      # column 1
        0, 1, 0, 0,      # column 2
        0, 0, 1, 0,      # column 3
        v[1], v[2], v[3], 1  # column 4
    )
end

# ==============================================================================
# Lego Figure Configuration
# ==============================================================================

# Part colors (matching original RPRMakie example)
const LEGO_COLORS = Dict(
    "eyes_mouth" => RGB(0.0f0, 0.0f0, 0.0f0),
    "belt" => RGB(0.0f0, 0.0f0, 0.35f0),
    "arm_right" => RGB(0.0f0, 0.6f0, 0.15f0),
    "arm_left" => RGB(0.0f0, 0.6f0, 0.15f0),
    "hand_right" => RGB(1.0f0, 0.85f0, 0.0f0),
    "hand_left" => RGB(1.0f0, 0.85f0, 0.0f0),
    "leg_right" => RGB(0.2f0, 0.4f0, 0.9f0),
    "leg_left" => RGB(0.2f0, 0.4f0, 0.9f0),
    "torso" => RGB(0.84f0, 0.06f0, 0.15f0),
    "head" => RGB(1.0f0, 0.85f0, 0.0f0),
)

# Joint pivot points (in local mesh coordinates)
const JOINT_ORIGINS = Dict(
    "arm_right" => Point3f(0.1427, -6.2127, 5.7342),
    "arm_left" => Point3f(0.1427, 6.2127, 5.7342),
    "leg_right" => Point3f(0, -1, -8.2),
    "leg_left" => Point3f(0, 1, -8.2),
)

# Rotation axes for joints
const ROTATION_AXES = Dict(
    "arm_right" => Vec3f(0.0, -0.9828, 0.1848),
    "arm_left" => Vec3f(0.0, 0.9828, 0.1848),
    "leg_right" => Vec3f(0, -1, 0),
    "leg_left" => Vec3f(0, 1, 0),
)

# Order of parts for instance indexing
const PART_ORDER = [
    "torso", "head", "eyes_mouth",
    "arm_right", "hand_right",
    "arm_left", "hand_left",
    "belt", "leg_right", "leg_left",
    "floor"
]

# Parent relationships: child => parent
const PART_PARENTS = Dict(
    "head" => "torso",
    "eyes_mouth" => "head",
    "arm_right" => "torso",
    "hand_right" => "arm_right",
    "arm_left" => "torso",
    "hand_left" => "arm_left",
    "belt" => "torso",
    "leg_right" => "belt",
    "leg_left" => "belt",
)

# ==============================================================================
# Scene Creation
# ==============================================================================

"""Load a lego part mesh from Makie assets."""
function load_lego_part(name::String)
    path = Makie.assetpath("lego_figure_$name.stl")
    mesh = load(path)
    return normal_mesh(mesh)
end

"""
Create the lego scene with TLAS.
Returns (tlas, ctx, materials)
"""
function create_lego_scene()
    println("Loading lego figure parts...")

    geometries = []
    materials = Material[]

    # Load all body parts
    for part_name in PART_ORDER[1:end-1]  # Exclude floor
        mesh = load_lego_part(part_name)
        push!(geometries, mesh)

        color = get(LEGO_COLORS, part_name, RGB(0.5f0, 0.5f0, 0.5f0))
        # Slight metallic sheen on plastic
        mat = Material(color, 0.1f0, 0.4f0, 1.0f0, 0.0f0)
        push!(materials, mat)

        println("  Loaded $part_name: $(length(faces(mesh))) triangles")
    end

    # Add floor
    floor_mesh = normal_mesh(Rect3f(Vec3f(-100, -100, -2), Vec3f(200, 200, 2)))
    push!(geometries, floor_mesh)
    push!(materials, Material(RGB(0.95f0, 0.95f0, 0.95f0), 0.0f0, 0.9f0, 1.0f0, 0.0f0))

    println("\nBuilding TLAS...")
    tlas = Raycore.TLAS(geometries, (mesh_idx, tri_idx) -> UInt32(mesh_idx))
    println("  Instances: $(length(tlas.instances))")

    # Create lights
    lights = [
        PointLight(Point3f(50, 0, 100), 8000.0f0, RGB(1.0f0, 0.95f0, 0.9f0)),
        PointLight(Point3f(-30, 40, 60), 3000.0f0, RGB(0.8f0, 0.85f0, 1.0f0)),
        PointLight(Point3f(0, -50, 80), 2000.0f0, RGB(1.0f0, 1.0f0, 1.0f0)),
    ]

    ctx = RenderContext(lights, materials, 0.15f0)

    return tlas, ctx
end

"""Get the instance index for a named part."""
part_index(name::String) = findfirst(==(name), PART_ORDER)

"""
Compute rotation around a joint pivot point.
Returns transform that rotates around the pivot.
"""
function joint_rotation(pivot::Point3f, axis::Vec3f, angle::Float32)::Mat4f
    # Translate to pivot, rotate, translate back
    translation(Vec3f(pivot)) * rotation_axis(axis, angle) * translation(-Vec3f(pivot))
end

"""
Update all instance transforms for the walking animation.

joint_angles: Dict mapping joint names to rotation angles
figure_pos: Overall figure position (x translation for walking)
"""
function update_walking_pose!(tlas, joint_angles::Dict{String, Float32}, figure_pos::Vec3f)
    # Compute transforms for each part, respecting hierarchy
    transforms = Dict{String, Mat4f}()

    # Base transform: lift figure by 20 units (matches RPRMakie example) and translate
    base_transform = translation(figure_pos + Vec3f(0, 0, 20))

    # Torso gets base transform only
    transforms["torso"] = base_transform

    # Process parts in order (parents before children due to PART_ORDER)
    for part_name in PART_ORDER[1:end-1]  # Exclude floor
        if part_name == "torso"
            continue  # Already handled
        end

        # Get parent transform
        parent_name = get(PART_PARENTS, part_name, "torso")
        parent_transform = get(transforms, parent_name, base_transform)

        # Start with parent transform
        transform = parent_transform

        # Add joint rotation if this part has one
        if haskey(JOINT_ORIGINS, part_name) && haskey(joint_angles, part_name)
            pivot = JOINT_ORIGINS[part_name]
            axis = ROTATION_AXES[part_name]
            angle = joint_angles[part_name]
            transform = transform * joint_rotation(pivot, axis, angle)
        end

        transforms[part_name] = transform
    end

    # Apply transforms to TLAS instances
    for (i, part_name) in enumerate(PART_ORDER)
        if part_name == "floor"
            update_instance_transform!(tlas, i, IDENTITY)
        else
            update_instance_transform!(tlas, i, transforms[part_name])
        end
    end

    refit_tlas!(tlas)
end

"""
Generate the walking angle sequence following the RPRMakie reference.

Returns a vector of angles: [0→max, max→0, 0→-max, -max→0]
All limbs use the SAME angle - the rotation axes have opposite signs for
left/right limbs, so applying the same angle creates natural alternating motion.
"""
function generate_walk_cycle()::Vector{Float32}
    rot_joints_by = 0.25f0 * Float32(pi)
    animation_strides = 10

    a1 = collect(LinRange(0.0f0, rot_joints_by, animation_strides))
    return Vector{Float32}(vcat(
        a1,
        reverse(a1[1:end-1]),
        -a1[2:end],
        reverse(-a1[1:end-1])
    ))
end

"""
Get joint angles for a specific frame in the walk cycle.
All limbs get the same angle - the axes handle the opposition.
"""
function walking_angles(frame_idx::Int, angle_sequence::Vector{Float32})::Dict{String, Float32}
    idx = mod1(frame_idx, length(angle_sequence))
    angle = angle_sequence[idx]

    return Dict{String, Float32}(
        "arm_right" => angle,
        "arm_left" => angle,
        "leg_right" => angle,
        "leg_left" => angle,
    )
end

# ==============================================================================
# Main Animation
# ==============================================================================

println("\n" * "="^70)
println("Animated Lego Figure - TLAS Dynamic Scene Demo")
println("="^70)

# Create scene with adjusted lighting (dimmer for better visuals)
tlas, ctx_original = create_lego_scene()

# Create dimmer lighting context
lights_dim = [
    PointLight(Point3f(50, 0, 100), 5000.0f0, RGB(1.0f0, 0.95f0, 0.9f0)),
    PointLight(Point3f(-30, 40, 60), 2000.0f0, RGB(0.8f0, 0.85f0, 1.0f0)),
    PointLight(Point3f(0, -50, 80), 1500.0f0, RGB(1.0f0, 1.0f0, 1.0f0)),
]
ctx = RenderContext(lights_dim, ctx_original.materials, 0.12f0)

# Generate walk cycle angles (following RPRMakie reference)
angle_sequence = generate_walk_cycle()
nsteps = length(angle_sequence)

# Animation parameters - match RPRMakie
total_translation = 50.0f0
num_frames = nsteps  # One frame per angle step
width, height = 1920, 1080
frames_dir = "lego_walk_frames"

# Camera positioned like RPRMakie: Vec3f(100, 30, 80) looking at Vec3f(0, 0, -10)
# Figure walks in +X direction toward the camera
camera_pos = Point3f(100, 30, 80)
camera_lookat = Point3f(0, 0, 10)  # Look at figure height

println("\nRendering $num_frames frames...")
println("  Resolution: $(width)x$(height)")
println("  Output: $frames_dir/")
println("  Walk cycle: $nsteps steps")

# Create output directory
rm(frames_dir; recursive=true)
mkpath(frames_dir)

# Set initial pose
update_walking_pose!(tlas, walking_angles(1, angle_sequence), Vec3f(0, 0, 0))

# Warmup render
println("\nWarmup render...")
img_warmup = fill(RGBf(0, 0, 0), height, width)
renderer = WavefrontRenderer(img_warmup, tlas, ctx;
    camera_pos=camera_pos,
    camera_lookat=camera_lookat,
    camera_up=Vec3f(0, 0, 1),
    fov=40.0f0,
    sky_color=RGB{Float32}(0.6f0, 0.75f0, 0.95f0),
    samples_per_pixel=8
)
render!(renderer)

# Animation: figure walks forward toward camera (like RPRMakie)
println("\nRendering animation...")
translations = collect(LinRange(0.0f0, total_translation, nsteps))

total_time = @elapsed begin
    for frame in 1:num_frames
        # Get angle and translation for this frame
        angles = walking_angles(frame, angle_sequence)
        pos_x = translations[frame]

        # Update pose
        update_walking_pose!(tlas, angles, Vec3f(pos_x, 0, 0))

        # Render frame with fixed camera (figure walks toward it)
        fill!(img_warmup, RGBf(0, 0, 0))
        render!(renderer)

        # Save frame with padded number
        filename = joinpath(frames_dir, "frame_$(lpad(frame, 4, '0')).png")
        save(filename, map(clamp01nan, img_warmup))

        if frame % 10 == 0 || frame == 1
            println("  Frame $frame/$num_frames")
        end
    end
end

avg_fps = num_frames / total_time
println("\nAnimation complete!")
println("  Total time: $(round(total_time, digits=2))s")
println("  Average FPS: $(round(avg_fps, digits=2))")

# Create video using FFMPEG
println("\nCreating video...")
using FFMPEG_jll
video_output = "lego_walk.mp4"
run(`$(FFMPEG_jll.ffmpeg()) -y -framerate 30 -i $(frames_dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p $video_output`)
println("Video saved: $video_output")

# Performance comparison
println("\n" * "="^70)
println("Performance: TLAS Refit vs Full Rebuild")
println("="^70)

using BenchmarkTools

# Benchmark pose update + refit
refit_time = @belapsed begin
    angles = walking_angles(15, $angle_sequence)
    update_walking_pose!($tlas, angles, Vec3f(15, 0, 0))
end

println("  Pose update + refit: $(round(refit_time * 1000, digits=3)) ms")

# Benchmark full rebuild
geometries_rebuild = vcat(
    [load_lego_part(p) for p in PART_ORDER[1:end-1]],
    [normal_mesh(Rect3f(Vec3f(-100, -100, -2), Vec3f(200, 200, 2)))]
)

rebuild_time = @belapsed begin
    Raycore.TLAS($geometries_rebuild, (mesh_idx, tri_idx) -> UInt32(mesh_idx))
end

println("  Full TLAS rebuild: $(round(rebuild_time * 1000, digits=3)) ms")
println("  Speedup: $(round(rebuild_time / refit_time, digits=1))x faster with refit!")

println("\nDone!")
