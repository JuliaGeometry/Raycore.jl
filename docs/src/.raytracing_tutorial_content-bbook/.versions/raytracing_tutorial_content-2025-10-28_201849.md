# Ray Tracing with Raycore: Building a Real Ray Tracer

In this tutorial, we'll build a simple but complete ray tracer from scratch using Raycore. We'll start with the absolute basics and progressively add features until we have a ray tracer that produces beautiful images with shadows and materials.

By the end, you'll have a working ray tracer that can render complex scenes!

## Setup

```julia (editor=true, logging=false, output=true)
using Raycore, GeometryBasics, LinearAlgebra
using Colors, ImageShow
using Makie  # For loading assets
using BenchmarkTools
```
**Ready to go!** We have:

  * `Raycore` for fast ray-triangle intersections
  * `GeometryBasics` for geometry primitives
  * `Colors` and `ImageShow` for displaying rendered images

## Part 1: Our Scene - A Playful Cat

Let's create a fun scene that we'll use throughout this tutorial. We'll load a cat model and place it in a simple room.

```julia (editor=true, logging=false, output=true)
# Load the cat model and rotate it to face the camera
cat_mesh = Makie.loadasset("cat.obj")
# Rotate 150 degrees around Y axis so cat faces camera at an angle
angle = deg2rad(150f0)
rotation = Makie.Quaternionf(0, sin(angle/2), 0, cos(angle/2))
rotated_coords = [rotation * Point3f(v) for v in coordinates(cat_mesh)]

# Get bounding box and translate cat to sit on the floor
cat_bbox = Rect3f(rotated_coords)
floor_y = -1.5f0
cat_offset = Vec3f(0, floor_y - cat_bbox.origin[2], 0)  # Translate so bottom sits on floor

cat_mesh = GeometryBasics.normal_mesh(
    [v + cat_offset for v in rotated_coords],
    faces(cat_mesh)
)

# Create a simple room: floor, back wall, and side wall
floor = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)))
back_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)))
left_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)))

# Add a couple of spheres for visual interest (also on the floor)
sphere1 = Tesselation(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), 64)
sphere2 = Tesselation(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), 64)

# Build our BVH acceleration structure
scene_geometry = [cat_mesh, floor, back_wall, left_wall, sphere1, sphere2]
bvh = Raycore.BVHAccel(scene_geometry)
```
**Scene created!**

  * Cat model with triangulated geometry
  * Room geometry: 3 walls
  * 2 decorative spheres
  * BVH built for fast ray traversal

## Part 2: The Simplest Ray Tracer - Binary Hit Detection

Let's start super simple: for each pixel, we shoot a ray and color it based on whether we hit something or not.

```julia (editor=true, logging=false, output=true)
# Trace helper - runs a callback for each pixel
function trace(f, bvh; width=700, height=300, camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
               sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0))
    img = Matrix{RGB{Float32}}(undef, height, width)

    # Precompute camera parameters
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    for y in 1:height, x in 1:width
        # Generate camera ray
        ndc_x = (2.0f0 * x / width - 1.0f0) * aspect
        ndc_y = 1.0f0 - 2.0f0 * y / height
        direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
        ray = Raycore.Ray(o=camera_pos, d=direction)

        # Ray-scene intersection
        hit_found, triangle, distance, bary_coords = Raycore.closest_hit(bvh, ray)

        # Let the callback decide the color (pass sky_color for misses)
        img[y, x] = hit_found ? f(triangle, distance, bary_coords, ray) : sky_color
    end

    return img
end

# Binary kernel - white if hit
binary_kernel(triangle, distance, bary_coords, ray) = RGB(1.0f0, 1.0f0, 1.0f0)

trace(binary_kernel, bvh, sky_color=RGB(0.0f0, 0.0f0, 0.0f0))
```
**Our first render!** Pure silhouette - you can see the cat and spheres.

## Part 3: Adding Depth - Distance-Based Shading

Let's make it more interesting by coloring based on distance (depth map).

```julia (editor=true, logging=false, output=true)
function depth_kernel(triangle, distance, bary_coords, ray)
    # Map distance to grayscale (closer = brighter)
    normalized_depth = clamp(1.0f0 - (distance - 2.0f0) / 8.0f0, 0.0f0, 1.0f0)
    RGB(normalized_depth, normalized_depth, normalized_depth)
end

trace(depth_kernel, bvh)
```
**Depth perception!** Now we can see the 3D structure - closer objects are brighter.

## Part 4: Surface Normals - The Foundation of Lighting

To do proper lighting, we need surface normals. Let's compute and visualize them.

```julia (editor=true, logging=false, output=true)
# Helper to interpolate normals using barycentric coordinates
function compute_normal(triangle, bary_coords)
    n1, n2, n3 = triangle.normals
    u, v, w = bary_coords
    normalize(Vec3f(u * n1 + v * n2 + w * n3))
end

function normal_kernel(triangle, distance, bary_coords, ray)
    normal = compute_normal(triangle, bary_coords)
    # Map normal components [-1,1] to color [0,1]
    RGB((normal .+ 1.0f0) ./ 2.0f0...)
end

trace(normal_kernel, bvh)
```
**Surface normals visualized!** Each color channel represents a normal component:

  * Red = X direction
  * Green = Y direction
  * Blue = Z direction

## Part 5: Basic Lighting - Diffuse Shading

Now we can add a light source and compute simple diffuse (Lambertian) shading!

```julia (editor=true, logging=false, output=true)
light_pos = Point3f(3, 4, -2)
light_intensity = 50.0f0

function diffuse_kernel(triangle, distance, bary_coords, ray)
    # Compute hit point and normal
    hit_point = ray.o + ray.d * distance
    normal = compute_normal(triangle, bary_coords)

    # Light direction and distance
    light_dir = light_pos - hit_point
    light_distance = norm(light_dir)
    light_dir = normalize(light_dir)

    # Diffuse shading (Lambertian)
    diffuse = max(0.0f0, dot(normal, light_dir))

    # Light attenuation (inverse square law)
    attenuation = light_intensity / (light_distance * light_distance)
    color = diffuse * attenuation

    RGB(color, color, color)
end

trace(diffuse_kernel, bvh)
```
**Let there be light!** Our scene now has proper shading based on surface orientation.

## Part 6: Adding Shadows - Shadow Rays

Time to add realism with shadows using Raycore's `any_hit` for fast occlusion testing.

```julia (editor=true, logging=false, output=true)
ambient = 0.1f0  # Ambient lighting to prevent pure black shadows

function shadow_kernel(triangle, distance, bary_coords, ray)
    hit_point = ray.o + ray.d * distance
    normal = compute_normal(triangle, bary_coords)

    # Light direction
    light_dir = light_pos - hit_point
    light_distance = norm(light_dir)
    light_dir = normalize(light_dir)

    # Diffuse shading
    diffuse = max(0.0f0, dot(normal, light_dir))

    # Shadow ray - offset slightly to avoid self-intersection
    shadow_ray_origin = hit_point + normal * 0.001f0
    shadow_ray = Raycore.Ray(o=shadow_ray_origin, d=light_dir)

    # Check if path to light is blocked
    shadow_hit, _, shadow_dist, _ = Raycore.any_hit(bvh, shadow_ray)
    in_shadow = shadow_hit && shadow_dist < light_distance

    # Final color
    color = if in_shadow
        ambient  # Only ambient in shadow
    else
        attenuation = light_intensity / (light_distance * light_distance)
        ambient + diffuse * attenuation
    end

    RGB(color, color, color)
end

trace(shadow_kernel, bvh)
```
**Shadows!** Notice how objects cast shadows on each other, adding depth and realism.

## Part 7: Multiple Lights

Let's add multiple lights to make the scene more interesting! We'll define a RenderContext to hold lights and materials:

```julia (editor=true, logging=false, output=true)
# Define a simple point light structure
struct PointLight
    position::Point3f
    intensity::Float32
    color::RGB{Float32}
end

# Material structure (for later use)
struct Material
    base_color::RGB{Float32}
    metallic::Float32
    roughness::Float32
end

# Render context holds all scene data
struct RenderContext
    bvh::Raycore.BVHAccel
    lights::Vector{PointLight}
    materials::Vector{Material}
    ambient::Float32
end

# Create multiple lights
lights = [
    PointLight(Point3f(3, 4, -2), 50.0f0, RGB(1.0f0, 0.9f0, 0.8f0)),    # Warm main light
    PointLight(Point3f(-3, 2, 0), 20.0f0, RGB(0.7f0, 0.8f0, 1.0f0)),   # Cool fill light
    PointLight(Point3f(0, 5, 5), 15.0f0, RGB(1.0f0, 1.0f0, 1.0f0))     # White back light
]

# Materials (will use these in Part 8)
materials = [
    Material(RGB(0.8f0, 0.6f0, 0.4f0), 0.0f0, 0.8f0),  # 1: Cat
    Material(RGB(0.3f0, 0.5f0, 0.3f0), 0.0f0, 0.9f0),  # 2: Floor
    Material(RGB(0.7f0, 0.7f0, 0.8f0), 0.0f0, 0.8f0),  # 3: Back wall
    Material(RGB(0.8f0, 0.7f0, 0.7f0), 0.0f0, 0.8f0),  # 4: Left wall
    Material(RGB(0.95f0, 0.64f0, 0.54f0), 1.0f0, 0.1f0),  # 5: Sphere 1 - metallic
    Material(RGB(0.8f0, 0.8f0, 0.9f0), 1.0f0, 0.0f0)   # 6: Sphere 2 - mirror
]

# Create render context
ctx = RenderContext(bvh, lights, materials, 0.1f0)
```
Now we need a new trace function that works with RenderContext:

```julia (editor=true, logging=false, output=true)
# Trace with RenderContext
function trace_ctx(f, ctx::RenderContext; width=700, height=300,camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
                   sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0))
    img = Matrix{RGB{Float32}}(undef, height, width)

    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    for y in 1:height, x in 1:width
        ndc_x = (2.0f0 * x / width - 1.0f0) * aspect
        ndc_y = 1.0f0 - 2.0f0 * y / height
        direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
        ray = Raycore.Ray(o=camera_pos, d=direction)

        hit_found, triangle, distance, bary_coords = Raycore.closest_hit(ctx.bvh, ray)
        img[y, x] = hit_found ? f(ctx, triangle, distance, bary_coords, ray) : sky_color
    end

    return img
end

function multi_light_kernel(ctx, triangle, distance, bary_coords, ray)
    hit_point = ray.o + ray.d * distance
    normal = compute_normal(triangle, bary_coords)

    # Start with ambient (grayscale)
    total_color = Vec3f(ctx.ambient, ctx.ambient, ctx.ambient)

    # Accumulate contribution from each light
    for light in ctx.lights
        light_vec = light.position - hit_point
        light_distance = norm(light_vec)
        light_dir = light_vec / light_distance

        diffuse = max(0.0f0, dot(normal, light_dir))

        shadow_ray = Raycore.Ray(o=hit_point + normal * 0.001f0, d=light_dir)
        shadow_hit, _, shadow_dist, _ = Raycore.any_hit(ctx.bvh, shadow_ray)
        in_shadow = shadow_hit && shadow_dist < light_distance

        if !in_shadow
            attenuation = light.intensity / (light_distance * light_distance)
            light_col = Vec3f(light.color.r, light.color.g, light.color.b)
            total_color += light_col * (diffuse * attenuation)
        end
    end

    RGB{Float32}(total_color...)
end

trace_ctx(multi_light_kernel, ctx)
```
**Multiple lights!** The scene now has three different colored lights creating a more dynamic lighting environment.

## Part 8: Colored Materials with Multiple Lights

Now let's combine materials with our multiple lights!

```julia (editor=true, logging=false, output=true)
function material_multi_light_kernel(ctx, triangle, distance, bary_coords, ray)
    hit_point = ray.o + ray.d * distance
    normal = compute_normal(triangle, bary_coords)

    # Get material from context
    mat = ctx.materials[triangle.material_idx]
    base_color = Vec3f(mat.base_color.r, mat.base_color.g, mat.base_color.b)

    # Start with ambient
    total_color = base_color * ctx.ambient

    # Accumulate contribution from each light
    for light in ctx.lights
        light_vec = light.position - hit_point
        light_distance = norm(light_vec)
        light_dir = light_vec / light_distance

        diffuse = max(0.0f0, dot(normal, light_dir))

        # Shadow test
        shadow_ray = Raycore.Ray(o=hit_point + normal * 0.001f0, d=light_dir)
        shadow_hit, _, shadow_dist, _ = Raycore.any_hit(ctx.bvh, shadow_ray)
        in_shadow = shadow_hit && shadow_dist < light_distance

        if !in_shadow
            attenuation = light.intensity / (light_distance * light_distance)
            light_col = Vec3f(light.color.r, light.color.g, light.color.b)
            total_color += base_color .* light_col * (diffuse * attenuation)
        end
    end

    RGB{Float32}(total_color...)
end

trace_ctx(material_multi_light_kernel, ctx)
```
**Colored materials!**

  * Orange/tan cat
  * Green floor
  * Light blue back wall
  * Pink side wall
  * Red and blue spheres

## Part 9: Reflective Materials - Mirrors and Metals

The materials we defined in Part 7 already have metallic and roughness properties. Let's use them for reflections!

```julia (editor=true, logging=false, output=true)
# Helper: compute direct lighting with multiple lights
function compute_multi_light(ctx, point, normal, mat)
    base_color = Vec3f(mat.base_color.r, mat.base_color.g, mat.base_color.b)

    # Start with ambient
    total_color = base_color * ctx.ambient

    for light in ctx.lights
        light_vec = light.position - point
        light_distance = norm(light_vec)
        light_dir = light_vec / light_distance

        diffuse = max(0.0f0, dot(normal, light_dir))

        # Shadow test
        shadow_ray = Raycore.Ray(o=point + normal * 0.001f0, d=light_dir)
        shadow_hit, _, shadow_dist, _ = Raycore.any_hit(ctx.bvh, shadow_ray)
        in_shadow = shadow_hit && shadow_dist < light_distance

        if !in_shadow
            attenuation = light.intensity / (light_distance * light_distance)
            light_col = Vec3f(light.color.r, light.color.g, light.color.b)
            total_color += base_color .* light_col * (diffuse * attenuation)
        end
    end

    return RGB{Float32}(total_color...)
end

function reflective_kernel(ctx, triangle, distance, bary_coords, ray, sky_color)
    hit_point = ray.o + ray.d * distance
    normal = compute_normal(triangle, bary_coords)
    mat = ctx.materials[triangle.material_idx]

    # Compute direct lighting (diffuse component)
    direct_color = compute_multi_light(ctx, hit_point, normal, mat)

    # Add reflection for metallic materials
    if mat.metallic > 0.0f0
        # Compute reflection direction: reflect outgoing direction about normal
        # Note: ray.d points toward surface, but reflect() expects outgoing direction
        wo = -ray.d  # outgoing direction (away from surface)
        reflect_dir = Raycore.reflect(wo, normal)

        # Add roughness by perturbing reflection direction
        if mat.roughness > 0.0f0
            # Simple roughness: add random offset in tangent space
            random_offset = (rand(Vec3f) .* 2.0f0 .- 1.0f0) * mat.roughness
            reflect_dir = normalize(reflect_dir + random_offset)
        end

        # Cast reflection ray (offset to avoid self-intersection)
        reflect_ray = Raycore.Ray(o=hit_point + normal * 0.001f0, d=reflect_dir)
        refl_hit, refl_tri, refl_dist, refl_bary = Raycore.closest_hit(ctx.bvh, reflect_ray)

        # Get reflection color
        reflection_color = if refl_hit
            refl_point = reflect_ray.o + reflect_ray.d * refl_dist
            refl_normal = compute_normal(refl_tri, refl_bary)
            refl_mat = ctx.materials[refl_tri.material_idx]

            # Compute lighting for reflected surface
            compute_multi_light(ctx, refl_point, refl_normal, refl_mat)
        else
            sky_color
        end

        # Blend between diffuse and reflection based on metallic parameter
        return direct_color * (1.0f0 - mat.metallic) + reflection_color * mat.metallic
    else
        # Pure diffuse material
        return direct_color
    end
end

trace_ctx(ctx) do ctx, triangle, distance, bary_coords, ray
    reflective_kernel(ctx, triangle, distance, bary_coords, ray, RGB(0.5f0, 0.7f0, 1.0f0))
end
```
**Reflective materials!** The spheres now have metallic properties:

  * One smooth copper-colored metal with slight roughness
  * One perfect mirror reflecting the scene

Notice how reflections capture both the scene geometry and lighting!

## Part 10: Multi-threading for Performance

Let's add multi-threading to make our ray tracer much faster!

```julia (editor=true, logging=false, output=true)

```
```julia (editor=true, logging=false, output=true)
using BenchmarkTools
function trace_ctx_threaded(f, ctx::RenderContext; width=400, height=300, camera_pos=Point3f(0, 1, -2.5), fov=45.0f0,
                            sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0))
    img = Matrix{RGB{Float32}}(undef, height, width)

    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    Threads.@threads for y in 1:height
        for x in 1:width
            ndc_x = (2.0f0 * x / width - 1.0f0) * aspect
            ndc_y = 1.0f0 - 2.0f0 * y / height
            direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
            ray = Raycore.Ray(o=camera_pos, d=direction)

            hit_found, triangle, distance, bary_coords = Raycore.closest_hit(ctx.bvh, ray)
            img[y, x] = hit_found ? f(ctx, triangle, distance, bary_coords, ray) : sky_color
        end
    end

    return img
end

# Benchmark single-threaded vs multi-threaded
b1 = @belapsed trace_ctx(material_multi_light_kernel, ctx, width=800, height=600);

b2 = @belapsed trace_ctx_threaded(material_multi_light_kernel, ctx, width=800, height=600);
md"""
Threads: $(Threads.nthreads())

Single: $(b1)

Multi: $(b2)
"""
```
**Performance boost with threading!** The speedup should be close to the number of CPU cores.

Notice how we can reuse the same kernel function with both `trace_ctx()` and `trace_ctx_threaded()` - this is great for composability!

## Part 11: Multi-Sampling for Anti-Aliasing

Let's add multiple samples per pixel with jittered camera rays for smooth anti-aliasing:

```julia (editor=true, logging=false, output=true)
function trace_ctx_sampled(f, ctx::RenderContext; 
        width=700, height=300, 
        camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0, 
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0), 
        samples=4)
    img = Matrix{RGB{Float32}}(undef, height, width)

    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))
    pixel_size = 1.0f0 / width

    Threads.@threads for y in 1:height
        for x in 1:width
            # Accumulate multiple samples per pixel using Vec3f for math
            color_sum = Vec3f(0.0f0, 0.0f0, 0.0f0)

            for _ in 1:samples
                jitter_x = (rand(Float32) - 0.5f0) * pixel_size
                jitter_y = (rand(Float32) - 0.5f0) * pixel_size

                ndc_x = (2.0f0 * (x + jitter_x) / width - 1.0f0) * aspect
                ndc_y = 1.0f0 - 2.0f0 * (y + jitter_y) / height
                direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
                ray = Raycore.Ray(o=camera_pos, d=direction)

                hit_found, triangle, distance, bary_coords = Raycore.closest_hit(ctx.bvh, ray)

                color = if hit_found
                    result = f(ctx, triangle, distance, bary_coords, ray)
                    Vec3f(result.r, result.g, result.b)
                else
                    Vec3f(sky_color.r, sky_color.g, sky_color.b)
                end

                color_sum += color
            end

            # Average samples and convert back to RGB
            avg = color_sum / samples
            img[y, x] = RGB{Float32}(avg...)
        end
    end

    return img
end

# Render with 16 samples per pixel for smooth anti-aliasing
@time trace_ctx_sampled(ctx, samples=64) do ctx, triangle, distance, bary_coords, ray
    reflective_kernel(ctx, triangle, distance, bary_coords, ray, RGB(0.5f0, 0.7f0, 1.0f0))
end
```
**Anti-aliased render!** 32 samples per pixel with jittered camera rays eliminate jagged edges.

## Summary - What We Built

We created a complete ray tracer that includes:

### Features Implemented

1. **Camera system** - Perspective projection with configurable FOV
2. **Ray-scene intersection** - Using Raycore's BVH for fast traversal
3. **Surface normals** - Smooth shading from vertex normals
4. **Diffuse lighting** - Lambertian shading with distance attenuation
5. **Hard shadows** - Using `any_hit` for efficient occlusion testing
6. **Simple materials** - Per-object color assignment
7. **Multi-threading** - Parallel rendering across CPU cores
8. **Callback-based API** - Flexible `trace()` function for experimentation

### Next Steps

To make this into a full path tracer (like `Trace`), you would add:

  * **Recursive ray tracing** - Reflections and refractions
  * **Multiple light sources** - Area lights, environment lighting
  * **Advanced materials** - Specular, glossy, transparent
  * **Sampling** - Multiple samples per pixel for anti-aliasing
  * **Better normal interpolation** - Proper barycentric interpolation
  * **GPU support** - Using KernelAbstractions.jl

The `Trace` package implements all of these features and more!

### Key Raycore Functions Used

  * `Raycore.BVHAccel(meshes)` - Build acceleration structure
  * `Raycore.Ray(o=origin, d=direction)` - Create ray
  * `Raycore.closest_hit(bvh, ray)` - Find nearest intersection
  * `Raycore.any_hit(bvh, ray)` - Test for any intersection (fast!)
  * `Raycore.vertices(triangle)` - Get triangle vertex positions
  * `Raycore.normals(triangle)` - Get triangle vertex normals

Happy ray tracing!

