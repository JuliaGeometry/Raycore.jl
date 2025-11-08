using Raycore, GeometryBasics, LinearAlgebra
using Colors, ImageShow
import KernelAbstractions as KA
function compute_normal(triangle, bary_coords)
    v0, v1, v2 = Raycore.normals(triangle)
    u, v, w = bary_coords[1], bary_coords[2], bary_coords[3]
    return Vec3f(normalize(v0 * u + v1 * v + v2 * w))
end

# Generate camera ray for a pixel with optional jitter
function camera_ray(x, y, width, height, camera_pos, focal_length, aspect; jitter=Vec2f(0))
    ndc_x = (2.0f0 * (Float32(x) - 0.5f0 + jitter[1]) / Float32(width) - 1.0f0) * aspect
    ndc_y = 1.0f0 - 2.0f0 * (Float32(y) - 0.5f0 + jitter[2]) / Float32(height)
    direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
    return Raycore.Ray(o=camera_pos, d=direction)
end

# Convert between color representations
to_vec3f(c::RGB) = Vec3f(c.r, c.g, c.b)
to_rgb(v::Vec3f) = RGB{Float32}(v...)

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

struct RenderContext{L<:AbstractVector{PointLight},M<:AbstractVector{Material}}
    lights::L
    materials::M
    ambient::Float32
end

function Raycore.to_gpu(Arr, ctx::RenderContext)
    return RenderContext(to_gpu(Arr, ctx.lights), to_gpu(Arr, ctx.materials), ctx.ambient)
end

function render_context()
    # Create lights and materials
    lights = [
        PointLight(Point3f(3, 4, -2), 50.0f0, RGB(1.0f0, 0.9f0, 0.8f0)),
        PointLight(Point3f(-3, 2, 0), 20.0f0, RGB(0.7f0, 0.8f0, 1.0f0)),
        PointLight(Point3f(0, 5, 5), 15.0f0, RGB(1.0f0, 1.0f0, 1.0f0))
    ]

    materials = [
        Material(RGB(0.8f0, 0.6f0, 0.4f0), 0.0f0, 0.8f0),  # cat
        Material(RGB(0.3f0, 0.5f0, 0.3f0), 0.0f0, 0.9f0),  # floor
        Material(RGB(0.8f0, 0.6f0, 0.5f0), 0.8f0, 0.05f0),  # back wall
        Material(RGB(0.7f0, 0.7f0, 0.8f0), 0.0f0, 0.8f0),  # left wall
        Material(RGB(0.9f0, 0.9f0, 0.9f0), 0.8f0, 0.02f0),  # sphere1 - metallic
        Material(RGB(0.3f0, 0.6f0, 0.9f0), 0.5f0, 0.3f0),  # sphere2 - semi-metallic
    ]

    return RenderContext(lights, materials, 0.1f0)
end

function compute_light(
    bvh, point, normal, light_pos, light_intensity, light_color; shadow_samples=1)

    light_vec = light_pos - point
    light_dist = norm(light_vec)
    light_dir = light_vec / light_dist

    diffuse = max(0.0f0, dot(normal, light_dir))

    # Shadow testing with optional soft shadows
    shadow_factor = 0.0f0
    light_radius = 0.2f0  # Size of area light for soft shadows

    for _ in 1:shadow_samples
        # For shadow_samples=1, this is just the light position (hard shadow)
        # For shadow_samples>1, we sample random points on a disk (soft shadow)
        if shadow_samples > 1
            # Random point on disk perpendicular to light direction
            offset = (rand(Vec3f) .* 2.0f0 .- 1.0f0) * light_radius
            offset = offset - light_dir * dot(offset, light_dir)
            shadow_target = light_pos + offset
        else
            shadow_target = light_pos
        end

        shadow_vec = shadow_target - point
        shadow_dist = norm(shadow_vec)
        shadow_dir = normalize(shadow_vec)

        shadow_ray = Raycore.Ray(o=point + normal * 0.001f0, d=shadow_dir)
        shadow_hit, _, hit_dist, _ = Raycore.any_hit(bvh, shadow_ray)

        if !shadow_hit || hit_dist >= shadow_dist
            shadow_factor += 1.0f0
        end
    end
    shadow_factor /= shadow_samples

    # Compute final light contribution
    attenuation = light_intensity / (light_dist * light_dist)
    return to_vec3f(light_color) * (diffuse * attenuation * shadow_factor)
end

function shadow_kernel(bvh, ctx, tri, dist, bary, ray; shadow_samples=1)
    hit_point = ray.o + ray.d * dist
    normal = compute_normal(tri, bary)
    # Single point light
    light_pos = Point3f(3, 4, -2)
    light_intensity = 50.0f0
    light_color = RGB{Float32}(1.0f0, 0.9f0, 0.8f0)
    # Hard shadows (shadow_samples=1)
    light_contrib = compute_light(
        bvh, hit_point, normal, light_pos, light_intensity, light_color;
        shadow_samples=shadow_samples
    )
    ambient = 0.1f0

    brightness = ambient .+ light_contrib
    return to_rgb(brightness)
end

function compute_multi_light(bvh, ctx, point, normal, mat; shadow_samples=1)
    base_color = to_vec3f(mat.base_color)
    total_color = base_color * ctx.ambient

    for light in ctx.lights
        light_contrib = compute_light(bvh, point, normal, light.position, light.intensity, light.color, shadow_samples=shadow_samples)
        total_color += base_color .* light_contrib
    end

    return total_color
end
function reflective_kernel(bvh, ctx, tri, dist, bary, ray, sky_color, shadow_samples=8)
    hit_point = ray.o + ray.d * dist
    normal = compute_normal(tri, bary)
    mat = ctx.materials[tri.material_idx]

    # Direct lighting with soft shadows
    direct_color = compute_multi_light(bvh, ctx, hit_point, normal, mat, shadow_samples=shadow_samples)

    # Reflections for metallic surfaces
    if mat.metallic > 0.0f0
        wo = -ray.d
        reflect_dir = Raycore.reflect(wo, normal)

        # Optional roughness
        if mat.roughness > 0.0f0
            offset = (rand(Vec3f) .* 2.0f0 .- 1.0f0) * mat.roughness
            reflect_dir = normalize(reflect_dir + offset)
        end

        # Cast reflection ray
        reflect_ray = Raycore.Ray(o=hit_point + normal * 0.001f0, d=reflect_dir)
        refl_hit, refl_tri, refl_dist, refl_bary = Raycore.closest_hit(bvh, reflect_ray)

        reflection_color = if refl_hit
            refl_point = reflect_ray.o + reflect_ray.d * refl_dist
            refl_normal = compute_normal(refl_tri, refl_bary)
            refl_mat = ctx.materials[refl_tri.material_idx]
            compute_multi_light(bvh, ctx, refl_point, refl_normal, refl_mat, shadow_samples=shadow_samples)
        else
            to_vec3f(sky_color)
        end

        direct_color = direct_color * (1.0f0 - mat.metallic) + reflection_color * mat.metallic
    end

    return to_rgb(direct_color)
end

function example_scene()
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
        faces(cat_mesh)
    )

    # Create a simple room: floor, back wall, and side wall
    floor = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)))
    back_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)))
    left_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)))

    # Add a couple of spheres for visual interest
    sphere1 = Tesselation(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), 64)
    sphere2 = Tesselation(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), 64)

    # Build our BVH acceleration structure
    scene_geometry = [cat_mesh, floor, back_wall, left_wall, sphere1, sphere2]
    bvh = Raycore.BVH(scene_geometry)
    return bvh, render_context()
end

function sample_light(bvh, ctx, width, height, camera_pos, focal_length, aspect, x, y, sky_color)
    jitter = rand(Vec2f)
    ray = camera_ray(x, y, width, height, camera_pos, focal_length, aspect; jitter)
    hit_found, tri, dist, bary = Raycore.closest_hit(bvh, ray)
    if hit_found
        color = reflective_kernel(bvh, ctx, tri, dist, bary, ray, RGB(0.5f0, 0.7f0, 1.0f0), 8)
        return to_vec3f(color)
    end
    return to_vec3f(sky_color)
end


import KernelAbstractions as KA
# Basic kernel: one thread per pixel, straightforward implementation
@kernel function raytrace_kernel_v1!(
        img, bvh, ctx,
        width, height, camera_pos, focal_length, aspect, sky_color
    )
    # Get pixel coordinates
    idx = @index(Global, Linear)
    # Convert linear index to 2D coordinates
    x = ((idx - 1) % width) + 1
    y = ((idx - 1) ÷ width) + 1
    if x <= width && y <= height
        # Generate camera ray
        color = Vec3f(0, 0, 0)
        for i in 1:10
            color = color .+ sample_light(bvh, ctx, width, height, camera_pos, focal_length, aspect, x, y, sky_color)
        end
        img[y, x] = (to_rgb(color) ./ 10f0)
    end
end


# New launcher: array-based (backend-agnostic) tracer for kernel v1
# This accepts the image and all scene arrays on the caller side. The caller may pass
# CPU arrays (Array) or GPU arrays (ROCArray) — KernelAbstractions will pick the right backend
# based on the `img` array backend.
function trace_gpu_v1(kernel, img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
        ndrange=length(img)
    )
    height, width = size(img)
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    backend = KA.get_backend(img)
    kernel! = kernel(backend)

    kernel!(
        img, bvh, ctx,
        Int32(width), Int32(height),
        camera_pos, focal_length, aspect,
        sky_color,
        ndrange=ndrange
    )
    KA.synchronize(backend)
    return img
end
# Helper function to plot kernel benchmark comparisons
function plot_kernel_benchmarks(benchmarks, labels; title="GPU Kernel Performance")
    # Extract median times in milliseconds
    times = [median(b.times) / 1e6 for b in benchmarks]

    # Sort by performance (fastest first)
    sorted_indices = sortperm(times)
    sorted_times = times[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Generate colors for bars based on number of benchmarks
    n = length(benchmarks)
    colors = Makie.resample_cmap(:viridis, n)

    # Create performance visualization
    fig = Figure(size=(700, 400))

    # Single plot: Execution time comparison (sorted by performance)
    ax = Axis(fig[1, 1],
        title=title,
        xlabel="Kernel Configuration",
        ylabel="Time (ms)",
        xticks=(1:length(sorted_labels), sorted_labels))

    barplot!(ax, 1:length(sorted_times), sorted_times, color=colors[sorted_indices])

    # Add value labels
    for (i, t) in enumerate(sorted_times)
        text!(ax, i, t, text="$(round(t, digits=1))ms",
              align=(:center, :bottom), fontsize=12)
    end

    # Highlight the winner (fastest)
    scatter!(ax, [1], [sorted_times[1]],
             color=:gold, markersize=30, marker=:star5)

    return fig, times, sorted_indices
end

# using Raycore: to_gpu
# bvh, ctx = example_scene()
# img = fill(RGBf(0, 0, 0), 512, 512)
# pres = []
# img_gpu = ROCArray(img);
# bvh_gpu = to_gpu(ROCArray, bvh; preserve=pres);
# ctx_gpu = to_gpu(ROCArray, ctx; preserve=pres);
# img_v1 = trace_gpu_v1(raytrace_kernel_v1!, img_gpu, bvh_gpu, ctx_gpu);
# Array(img_v1)
