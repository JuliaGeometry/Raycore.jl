using Revise
using Raycore
using Raycore: to_gpu
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics, Colors, LinearAlgebra
import Makie
using Makie: RGBf
import KernelAbstractions as KA
using ImageShow
using BenchmarkTools

# Load helper functions
include("raytracing-core.jl")
include("wavefront-renderer.jl")

function example_scene_tlas(; glass_cat=false)
    cat_mesh = Makie.loadasset("cat.obj")
    angle = deg2rad(150f0)
    rotation = Makie.Quaternionf(0, sin(angle/2), 0, cos(angle/2))
    rotated_coords = [rotation * Point3f(v) for v in coordinates(cat_mesh)]

    # Get bounding box and translate cat to sit on the floor
    cat_bbox = Rect3f(rotated_coords)
    floor_y = -1.5f0
    cat_offset = Vec3f(0, floor_y - cat_bbox.origin[2], 0)

    cat_mesh = GeometryBasics.normal_mesh(
        [v + cat_offset for v in rotated_coords],
        GeometryBasics.faces(cat_mesh)
    )

    # Create a simple room: floor, back wall, and side wall
    floor = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)))
    back_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)))
    left_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)))

    # Add a couple of spheres for visual interest
    sphere1 = Tesselation(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), 64)
    sphere2 = Tesselation(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), 64)

    # Material: base_color, metallic, roughness, ior, transmission
    cat_material = if glass_cat
        Material(RGB(0.95f0, 1.0f0, 0.95f0), 0.0f0, 0.0f0, 1.5f0, 1.0f0)
    else
        Material(RGB(0.8f0, 0.6f0, 0.4f0), 0.0f0, 0.8f0, 1.0f0, 0.0f0)
    end

    # (geometry, material) pairs
    scene = [
        (cat_mesh,   cat_material),
        (floor,      Material(RGB(0.3f0, 0.5f0, 0.3f0), 0.0f0, 0.9f0, 1.0f0, 0.0f0)),
        (back_wall,  Material(RGB(0.8f0, 0.6f0, 0.5f0), 0.8f0, 0.05f0, 1.0f0, 0.0f0)),
        (left_wall,  Material(RGB(0.7f0, 0.7f0, 0.8f0), 0.0f0, 0.8f0, 1.0f0, 0.0f0)),
        (sphere1,    Material(RGB(0.9f0, 0.9f0, 0.9f0), 0.8f0, 0.02f0, 1.0f0, 0.0f0)),
        (sphere2,    Material(RGB(0.3f0, 0.6f0, 0.9f0), 0.5f0, 0.3f0, 1.0f0, 0.0f0)),
    ]

    geometries = [g for (g, _) in scene]
    materials = [m for (_, m) in scene]

    println("\nBuilding TLAS (instanced BVH)...")
    println("  Each mesh becomes its own BLAS with a single instance")

    # Use TLAS instead of BVH - drop-in replacement!
    tlas = Raycore.TLAS(geometries, (mesh_idx, tri_idx) -> UInt32(mesh_idx))

    println("✓ TLAS built:")
    println("    Instances: $(length(tlas.instances))")
    println("    BLAS array: $(length(tlas.blas_array))")
    println("    TLAS nodes: $(length(tlas.nodes))")
    println("    Root AABB: $(tlas.root_aabb)")

    lights = default_lights()
    ctx = RenderContext(lights, materials, 0.1f0)
    return tlas, ctx
end

println("\n" * "="^70)
println("Creating scene with TLAS...")
println("="^70)

tlas, ctx = example_scene_tlas()

println("\n" * "="^70)
println("Rendering image...")
println("="^70)

# ibvh = Raycore.InstancedBVH(geom)
begin
    img = fill(RGBf(0, 0, 0), 400, 720)
    # Use original camera parameters to match pre-look-at benchmark image
    renderer = WavefrontRenderer(img, tlas, ctx;
        camera_pos=Point3f(0, -0.9, -2.5),
        camera_lookat=Point3f(0, -0.9, 10),  # Look in +Z direction
        camera_up=Vec3f(0, 1, 0),            # Y up
        fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
        samples_per_pixel=4
    )
    @btime render!(renderer)
end
using FileIO, ImageCore

save("wavefront_instanced.png", map(clamp01nan, img))
