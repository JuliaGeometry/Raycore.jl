# Ray Tracing in one Hour

Analougus to the famous [Ray Tracing in one Weekend](https://raytracing.github.io/), this tutorial uses Raycore to do the hard work of performant ray triangle intersection and therefore get a high performing ray tracer in a much shorter time. We'll start with the absolute basics and progressively add features until we have a ray tracer that produces beautiful images with shadows, materials, and reflections.

## Setup

```julia (editor=true, logging=false, output=true)
using Raycore, GeometryBasics, LinearAlgebra
using Colors, ImageShow, WGLMakie
using MeshIO
using BenchmarkTools
```
**Ready to go!** We have:

  * `Raycore` for fast ray-triangle intersections
  * `GeometryBasics` for geometry primitives
  * `Colors` and `ImageShow` for displaying rendered images
  * `MeshIO` for loading the cat data

## Part 1: Our Scene, The Makie Cat

Let's create a fun scene that we'll use throughout this tutorial.

```julia (editor=true, logging=false, output=true)
# Load the cat model and rotate it to face the camera
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
f, ax, pl = plot(bvh; axis=(; show_axis=false))
```
Set the camera to something better:

```julia (editor=true, logging=false, output=true)
cam = cameracontrols(ax.scene)
cam.eyeposition[] = [0, 1.0, -4]
cam.lookat[] = [0, 0, 2]
cam.upvector[] = [0.0, 1, 0.0]
cam.fov[] = 45.0
update_cam!(ax.scene, cam)
nothing
```
## Part 2: Helper Functions - Building Blocks

Let's define reusable helper functions we'll use throughout:

```julia (editor=true, logging=false, output=true)
# Compute interpolated normal at hit point
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
```
## Part 3: The Simplest Ray Tracer - Depth Visualization

We're using one main function to shoot rays for each pixel. For simplicity, we already added multisampling and simple multi threading, to enjoy smoother images and faster rendering times throughout the tutorial. Read the GPU tutorial how to further improve the performance of this simple, not yet optimal kernel.

```julia (editor=true, logging=false, output=true)
function trace(f, bvh; width=700, height=300,
               camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
               sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
               samples=1, ctx=nothing)
    img = Matrix{RGB{Float32}}(undef, height, width)
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    Threads.@threads for y in 1:height
        for x in 1:width
            color_sum = Vec3f(0)
            for _ in 1:samples
                jitter = samples > 1 ? rand(Vec2f) : Vec2f(0)
                # Calculate the ray shooting from the camera pixel into the scene
                ray = camera_ray(x, y, width, height, camera_pos, focal_length, aspect; jitter)
                hit_found, triangle, distance, bary_coords = Raycore.closest_hit(bvh, ray)
                color = if hit_found
                    to_vec3f(f(bvh, ctx, triangle, distance, bary_coords, ray))
                else
                    to_vec3f(sky_color)
                end
                color_sum += color
            end

            img[y, x] = to_rgb(color_sum / samples)
        end
    end
    return img
end

# Visualize depth
depth_kernel(bvh, ctx, tri, dist, bary, ray) = RGB(1.0f0 - min(dist / 10.0f0, 1.0f0))
```
```julia (editor=true, logging=false, output=true)
@time trace(depth_kernel, bvh, samples=16)
```
**First render!** Depth visualization shows distance to surfaces. **Much faster with threading and smoother with multi-sampling!**

## Part 5: Lighting with Hard Shadows

Let's add lighting and shadows using a reusable lighting function:

```julia (editor=true, logging=false, output=true)
# Reusable lighting function with optional shadow sampling
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

trace(shadow_kernel, bvh, samples=4)
```
**Hard shadows working!** Scene has realistic lighting with sharp shadow edges.

## Part 6: Soft Shadows

Now let's make shadows more realistic by sampling the light as an area light:

```julia (editor=true, logging=false, output=true)
trace((args...)-> shadow_kernel(args...; shadow_samples=8), bvh, samples=8)
```
**Soft shadows!** Much more realistic with smooth penumbra edges.

## Part 7: Materials and Multiple Lights

Time to add color and multiple lights:

```julia (editor=true, logging=false, output=true)
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

struct RenderContext
    lights::Vector{PointLight}
    materials::Vector{Material}
    ambient::Float32
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
    Material(RGB(0.8f0, 0.6f0, 0.5f0), 0.8f0, 0.05f0),  # back wall
    Material(RGB(0.7f0, 0.7f0, 0.8f0), 0.0f0, 0.8f0),  # left wall
    Material(RGB(0.9f0, 0.9f0, 0.9f0), 0.8f0, 0.02f0),  # sphere1 - metallic
    Material(RGB(0.3f0, 0.6f0, 0.9f0), 0.5f0, 0.3f0),  # sphere2 - semi-metallic
]

ctx = RenderContext(lights, materials, 0.1f0)
nothing
```
```julia (editor=true, logging=false, output=true)
# Compute lighting from all lights - reusing our compute_light function!
function compute_multi_light(bvh, ctx, point, normal, mat; shadow_samples=1)
    base_color = to_vec3f(mat.base_color)
    total_color = base_color * ctx.ambient

    for light in ctx.lights
        light_contrib = compute_light(bvh, point, normal, light.position, light.intensity, light.color, shadow_samples=shadow_samples)
        total_color += base_color .* light_contrib
    end

    return total_color
end

function material_kernel(bvh, ctx, tri, dist, bary, ray)
    hit_point = ray.o + ray.d * dist
    normal = compute_normal(tri, bary)
    mat = ctx.materials[tri.material_idx]

    color = compute_multi_light(bvh, ctx, hit_point, normal, mat, shadow_samples=2)
    return to_rgb(color)
end

trace(material_kernel, bvh, samples=4, ctx=ctx)
```
**Colorful scene with soft shadows from multiple lights!** Each object has its own material.

## Part 8: Reflections

Add simple reflections for metallic surfaces:

```julia (editor=true, logging=false, output=true)
function reflective_kernel(bvh, ctx, tri, dist, bary, ray, sky_color)
    hit_point = ray.o + ray.d * dist
    normal = compute_normal(tri, bary)
    mat = ctx.materials[tri.material_idx]

    # Direct lighting with soft shadows
    direct_color = compute_multi_light(bvh, ctx, hit_point, normal, mat, shadow_samples=8)

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
            compute_multi_light(bvh, ctx, refl_point, refl_normal, refl_mat, shadow_samples=1)
        else
            to_vec3f(sky_color)
        end

        direct_color = direct_color * (1.0f0 - mat.metallic) + reflection_color * mat.metallic
    end

    return to_rgb(direct_color)
end

img = trace(bvh, samples=16, ctx=ctx) do bvh, ctx, tri, dist, bary, ray
    reflective_kernel(bvh, ctx, tri, dist, bary, ray, RGB(0.5f0, 0.7f0, 1.0f0))
end
```
```julia (editor=true, logging=false, output=true)
# Tone mapping functions
luminosity(c::RGB{T}) where {T} = (max(c.r, c.g, c.b) + min(c.r, c.g, c.b)) / 2.0f0

function avg_lum(rgb_m, δ::Number=1f-10)
    cumsum = 0.0f0
    for pix in rgb_m
        cumsum += log10(δ + luminosity(pix))
    end
    return 10^(cumsum / (prod(size(rgb_m))))
end

function tone_mapping(img; a=0.5f0, y=1.0f0, lum=avg_lum(img, 1f-10))
    img_normalized = img .* a .* (1.0f0 / lum)
    img_01 = map(col->mapc(c-> clamp(c, 0f0, 1f0), col), img_normalized)
    ycorrected = map(col->mapc(c-> c^(1f0 / y), col), img_01)
    return ycorrected
end

tone_mapping(img, a=0.38, y=1.0)
```
For performance type stability is a must! We can use JET to test if a function is completely type stable, which we also test in the Raycore tests for all functions.

```julia (editor=true, logging=false, output=true)
using JET

# Get test data
test_ray = camera_ray(350, 150, 700, 300, Point3f(0, -0.9, -2.5), 1.0f0 / tan(deg2rad(45.0f0 / 2)), Float32(700/300))
hit_found, test_tri, test_dist, test_bary = Raycore.closest_hit(bvh, test_ray)

# Check kernel type stability (filter to Main module to ignore Base internals)
@test_opt depth_kernel(bvh, ctx, test_tri, test_dist, test_bary, test_ray)
@test_opt shadow_kernel(bvh, ctx, test_tri, test_dist, test_bary, test_ray)
@test_opt material_kernel(bvh, ctx, test_tri, test_dist, test_bary, test_ray)
@test_opt reflective_kernel(bvh, ctx, test_tri, test_dist, test_bary, test_ray, RGB(0.5f0, 0.7f0, 1.0f0))
nothing
```
## Summary

We built a complete ray tracer with:

**Core Features:**

  * BVH acceleration for fast ray-scene intersections
  * Perspective camera with configurable FOV
  * Smooth shading from interpolated normals
  * Multi-light system with distance attenuation
  * **Soft shadows** using area light sampling (via `compute_light` with `shadow_samples`)
  * Material system (base color, metallic, roughness)
  * Reflections with optional roughness
  * ACES tone mapping for HDR

**Performance:**

  * Multi-threading for parallel rendering (introduced early!)
  * Multi-sampling for anti-aliasing (introduced early!)
  * Type-stable kernels for optimal performance
  * **Modular, reusable `compute_light` function** - works for both hard and soft shadows

**Key Raycore Functions:**

  * `Raycore.BVH(meshes)` - Build acceleration structure
  * `Raycore.Ray(o=origin, d=direction)` - Create ray
  * `Raycore.closest_hit(bvh, ray)` - Find nearest intersection
  * `Raycore.any_hit(bvh, ray)` - Test for any intersection
  * `Raycore.reflect(wo, normal)` - Compute reflection direction

**Key Pattern:** The `compute_light` function is reusable across the entire tutorial:

  * `shadow_samples=1` → hard shadows
  * `shadow_samples=4` → soft shadows

This shows how a well-designed function can handle multiple use cases cleanly!

