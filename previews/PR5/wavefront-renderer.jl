"""
Full-Featured Wavefront Path Tracer (SoA version)
Includes shadows, reflections, and roughness

This version uses Struct-of-Arrays (SoA) layout to work around
OpenCL/SPIR-V limitations with Array-of-Structs field access.
"""

using Raycore, GeometryBasics, LinearAlgebra
using Colors
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA
using StructArrays

# Helper function to convert StructArray to GPU-compatible NamedTuple
function Raycore.to_gpu(Arr, data::StructArrays.StructVector; preserve=[])
    fields = map(propertynames(data)) do pname
        pname => Raycore.to_gpu(Arr, getproperty(data, pname); preserve=preserve)
    end
    return (; fields...)
end

# ============================================================================
# Work Queue Structures
# ============================================================================

struct PrimaryRayWork
    ray::Raycore.Ray
    pixel_x::Int32
    pixel_y::Int32
    sample_idx::Int32
end

struct PrimaryHitWork{Tri}
    hit_found::Bool
    tri::Tri
    dist::Float32
    bary::Vec3f
    ray::Raycore.Ray
    pixel_x::Int32
    pixel_y::Int32
    sample_idx::Int32
end

struct ShadowRayWork
    ray::Raycore.Ray
    hit_idx::Int32  # Index back to the hit that generated this shadow ray
    light_idx::Int32
end

struct ShadowResult
    visible::Bool  # true if light is visible (no occlusion)
    hit_idx::Int32
    light_idx::Int32
end

struct ReflectionRayWork
    ray::Raycore.Ray
    hit_idx::Int32  # Index back to primary hit
end

struct ReflectionHitWork{T}
    hit_found::Bool
    tri_idx::T  # Store triangle index instead of full triangle
    dist::Float32
    bary::Vec3f
    ray::Raycore.Ray
    primary_hit_idx::Int32
end

struct ShadedResult
    color::Vec3f
    pixel_x::Int32
    pixel_y::Int32
    sample_idx::Int32
end

# ============================================================================
# Stage 1: Generate Primary Camera Rays
# ============================================================================

@kernel function generate_primary_rays!(
    @Const(width), @Const(height),
    @Const(camera_pos), @Const(focal_length), @Const(aspect),
    @Const(samples_per_pixel),
    ray_queue
)
    idx = @index(Global, Linear)
    total_pixels = width * height

    if idx <= total_pixels
        x = Int32(((idx - 1) % width) + 1)
        y = Int32(((idx - 1) ÷ width) + 1)

        for s in Int32(1):samples_per_pixel
            ray_idx = (idx - 1) * samples_per_pixel + s
            jitter = rand(Vec2f)

            ndc_x = (2.0f0 * (Float32(x) - 0.5f0 + jitter[1]) / Float32(width) - 1.0f0) * aspect
            ndc_y = 1.0f0 - 2.0f0 * (Float32(y) - 0.5f0 + jitter[2]) / Float32(height)
            direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))

            ray = Raycore.Ray(o=camera_pos, d=direction)
            @inbounds ray_queue[ray_idx] = PrimaryRayWork(ray, x, y, s)
        end
    end
end

# ============================================================================
# Stage 2: Intersect Primary Rays
# ============================================================================

@kernel function intersect_primary_rays!(
    @Const(bvh),
    @Const(ray_queue),
    hit_queue
)
    idx = @index(Global, Linear)

    if idx <= length(ray_queue)
        @inbounds ray_work = ray_queue[idx]
        hit_found, tri, dist, bary = Raycore.closest_hit(bvh, ray_work.ray)
        bary_vec = Vec3f(bary[1], bary[2], bary[3])

        @inbounds hit_queue[idx] = PrimaryHitWork(
            hit_found, tri, dist, bary_vec,
            ray_work.ray, ray_work.pixel_x, ray_work.pixel_y, ray_work.sample_idx
        )
    end
end

# ============================================================================
# Stage 3: Generate Shadow Rays (for all hits × all lights)
# ============================================================================

@kernel function generate_shadow_rays!(
    @Const(hit_queue),
    @Const(ctx),
    shadow_ray_queue,
    @Const(num_lights)
)
    idx = @index(Global, Linear)

    if idx <= length(hit_queue)
        @inbounds hit_work = hit_queue[idx]

        if hit_work.hit_found
            # Compute hit point and normal
            hit_point = hit_work.ray.o + hit_work.ray.d * hit_work.dist
            v0, v1, v2 = Raycore.normals(hit_work.tri)
            u, v, w = hit_work.bary[1], hit_work.bary[2], hit_work.bary[3]
            normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

            # Generate shadow ray for each light
            for light_idx in Int32(1):num_lights
                shadow_ray_idx = (idx - 1) * num_lights + light_idx
                light = ctx.lights[light_idx]

                # Shadow ray towards light with proper bias to avoid shadow acne
                # Use larger bias (0.01) and compute t_max from offset origin
                shadow_bias = 0.01f0
                shadow_origin = hit_point + normal * shadow_bias
                light_vec = light.position - shadow_origin
                shadow_dir = normalize(light_vec)
                light_dist = norm(light_vec)
                shadow_ray = Raycore.Ray(o=shadow_origin, d=shadow_dir, t_max=light_dist)

                @inbounds shadow_ray_queue[shadow_ray_idx] = ShadowRayWork(shadow_ray, Int32(idx), Int32(light_idx))
            end
        else
            # Sky hit - no shadow rays needed, but we need to fill the queue
            for light_idx in Int32(1):num_lights
                shadow_ray_idx = (idx - 1) * num_lights + light_idx
                # Dummy shadow ray
                dummy_ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
                @inbounds shadow_ray_queue[shadow_ray_idx] = ShadowRayWork(dummy_ray, Int32(idx), Int32(light_idx))
            end
        end
    end
end

# ============================================================================
# Stage 4: Test Shadow Rays (occlusion test)
# ============================================================================

@kernel function test_shadow_rays!(
    @Const(bvh),
    @Const(shadow_ray_queue),
    shadow_result_queue
)
    idx = @index(Global, Linear)

    if idx <= length(shadow_ray_queue)
        @inbounds shadow_work = shadow_ray_queue[idx]

        # Test for occlusion
        # any_hit respects ray.t_max and only returns hits before the light
        # So if we get a hit, something is blocking the light
        if shadow_work.ray.t_max > 0.0f0
            hit_found, _, _, _ = Raycore.any_hit(bvh, shadow_work.ray)
            visible = !hit_found  # Visible only if no obstruction
        else
            visible = false  # Dummy ray (sky hits)
        end

        @inbounds shadow_result_queue[idx] = ShadowResult(visible, shadow_work.hit_idx, shadow_work.light_idx)
    end
end

# ============================================================================
# Stage 5: Shade Primary Hits with Shadow Information
# ============================================================================

@kernel function shade_primary_hits!(
    @Const(hit_queue),
    @Const(ctx),
    @Const(shadow_results),
    @Const(sky_color),
    @Const(num_lights),
    shading_queue
)
    idx = @index(Global, Linear)

    if idx <= length(hit_queue)
        @inbounds hit_work = hit_queue[idx]

        if hit_work.hit_found
            # Compute hit point and normal
            hit_point = hit_work.ray.o + hit_work.ray.d * hit_work.dist
            v0, v1, v2 = Raycore.normals(hit_work.tri)
            u, v, w = hit_work.bary[1], hit_work.bary[2], hit_work.bary[3]
            normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

            # Get material
            mat = ctx.materials[hit_work.tri.material_idx]
            base_color = Vec3f(mat.base_color.r, mat.base_color.g, mat.base_color.b)

            # Start with ambient
            total_color = base_color * ctx.ambient

            # Add contribution from each light using shadow test results
            for light_idx in Int32(1):num_lights
                shadow_idx = (idx - 1) * num_lights + light_idx
                @inbounds shadow_result = shadow_results[shadow_idx]

                if shadow_result.visible
                    light = ctx.lights[light_idx]
                    light_vec = light.position - hit_point
                    light_dist = norm(light_vec)
                    light_dir = light_vec / light_dist

                    diffuse = max(0.0f0, dot(normal, light_dir))
                    attenuation = light.intensity / (light_dist * light_dist)

                    light_color = Vec3f(light.color.r, light.color.g, light.color.b)
                    total_color += base_color .* (light_color * (diffuse * attenuation))
                end
            end

            @inbounds shading_queue[idx] = ShadedResult(total_color, hit_work.pixel_x, hit_work.pixel_y, hit_work.sample_idx)
        else
            # Sky color
            sky_vec = Vec3f(sky_color.r, sky_color.g, sky_color.b)
            @inbounds shading_queue[idx] = ShadedResult(sky_vec, hit_work.pixel_x, hit_work.pixel_y, hit_work.sample_idx)
        end
    end
end

# ============================================================================
# Stage 6: Generate Reflection Rays (for metallic materials)
# ============================================================================

@kernel function generate_reflection_rays!(
    @Const(hit_queue),
    @Const(ctx),
    reflection_ray_soa,  # SoA: NamedTuple with .ray and .hit_idx arrays
    active_count  # Output: number of active reflection rays
)
    idx = @index(Global, Linear)

    if idx <= length(hit_queue)
        @inbounds hit_work = hit_queue[idx]

        if hit_work.hit_found
            mat = ctx.materials[hit_work.tri.material_idx]

            if mat.metallic > 0.0f0
                # Compute reflection
                hit_point = hit_work.ray.o + hit_work.ray.d * hit_work.dist
                v0, v1, v2 = Raycore.normals(hit_work.tri)
                u, v, w = hit_work.bary[1], hit_work.bary[2], hit_work.bary[3]
                normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

                wo = -hit_work.ray.d
                reflect_dir = Raycore.reflect(wo, normal)

                # Add roughness
                if mat.roughness > 0.0f0
                    offset = (rand(Vec3f) .* 2.0f0 .- 1.0f0) * mat.roughness
                    reflect_dir = normalize(reflect_dir + offset)
                end

                # Use proper bias to avoid self-intersection
                reflect_ray = Raycore.Ray(o=hit_point + normal * 0.01f0, d=reflect_dir)

                # Write to SoA
                @inbounds reflection_ray_soa.ray[idx] = reflect_ray
                @inbounds reflection_ray_soa.hit_idx[idx] = Int32(idx)
            else
                # No reflection - dummy ray
                dummy_ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
                @inbounds reflection_ray_soa.ray[idx] = dummy_ray
                @inbounds reflection_ray_soa.hit_idx[idx] = Int32(idx)
            end
        else
            # Sky hit - no reflection
            dummy_ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
            @inbounds reflection_ray_soa.ray[idx] = dummy_ray
            @inbounds reflection_ray_soa.hit_idx[idx] = Int32(idx)
        end
    end
end

# ============================================================================
# Stage 7: Intersect Reflection Rays (SoA version)
# ============================================================================

@kernel function intersect_reflection_rays!(
    @Const(bvh),
    @Const(reflection_ray_soa),  # NamedTuple with .ray and .hit_idx arrays
    reflection_hit_soa           # NamedTuple with separate field arrays
)
    idx = @index(Global, Linear)

    if idx <= length(reflection_ray_soa.ray)
        # Access from SoA: fields are separate arrays
        @inbounds ray = reflection_ray_soa.ray[idx]
        @inbounds hit_idx = reflection_ray_soa.hit_idx[idx]

        if ray.t_max > 0.0f0
            hit_found, tri, dist, bary = Raycore.closest_hit(bvh, ray)
            bary_vec = Vec3f(bary[1], bary[2], bary[3])

            # Write to SoA output: each field is a separate array
            @inbounds reflection_hit_soa.hit_found[idx] = hit_found
            @inbounds reflection_hit_soa.tri_idx[idx] = tri.primitive_idx % Int32
            @inbounds reflection_hit_soa.dist[idx] = dist
            @inbounds reflection_hit_soa.bary[idx] = bary_vec
            @inbounds reflection_hit_soa.ray[idx] = ray
            @inbounds reflection_hit_soa.primary_hit_idx[idx] = hit_idx
        else
            # Dummy hit
            @inbounds reflection_hit_soa.hit_found[idx] = false
            @inbounds reflection_hit_soa.tri_idx[idx] = Int32(0)
            @inbounds reflection_hit_soa.dist[idx] = 0.0f0
            @inbounds reflection_hit_soa.bary[idx] = Vec3f(0,0,0)
            @inbounds reflection_hit_soa.ray[idx] = ray
            @inbounds reflection_hit_soa.primary_hit_idx[idx] = hit_idx
        end
    end
end

# ============================================================================
# Stage 8: Shade Reflection Hits and Blend into Primary
# ============================================================================

@kernel function shade_reflections_and_blend!(
    @Const(hit_queue),
    @Const(reflection_hit_soa),  # SoA version: NamedTuple of arrays
    @Const(bvh_triangles),        # Need original triangles for material lookup
    @Const(ctx),
    @Const(sky_color),
    shading_queue  # In/out: update with reflection contribution
)
    idx = @index(Global, Linear)

    if idx <= length(hit_queue)
        @inbounds primary_hit = hit_queue[idx]

        if primary_hit.hit_found
            mat = ctx.materials[primary_hit.tri.material_idx]

            if mat.metallic > 0.0f0
                # Access from SoA: each field is a separate array
                @inbounds refl_hit_found = reflection_hit_soa.hit_found[idx]
                @inbounds refl_tri_idx = reflection_hit_soa.tri_idx[idx]
                @inbounds refl_dist = reflection_hit_soa.dist[idx]
                @inbounds refl_bary = reflection_hit_soa.bary[idx]
                @inbounds refl_ray = reflection_hit_soa.ray[idx]

                # Compute reflection color (simplified - no recursive shadows for performance)
                reflection_color = if refl_hit_found
                    refl_point = refl_ray.o + refl_ray.d * refl_dist

                    # Get triangle from BVH using stored index
                    @inbounds refl_tri = bvh_triangles[refl_tri_idx]
                    v0, v1, v2 = Raycore.normals(refl_tri)
                    u, v, w = refl_bary[1], refl_bary[2], refl_bary[3]
                    refl_normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

                    refl_mat = ctx.materials[refl_tri.material_idx]
                    refl_base_color = Vec3f(refl_mat.base_color.r, refl_mat.base_color.g, refl_mat.base_color.b)

                    # Simplified lighting (ambient + first light only, no shadows)
                    refl_color = refl_base_color * ctx.ambient

                    if length(ctx.lights) > 0
                        light = ctx.lights[1]
                        light_vec = light.position - refl_point
                        light_dist = norm(light_vec)
                        light_dir = normalize(light_vec)
                        diffuse = max(0.0f0, dot(refl_normal, light_dir))
                        attenuation = light.intensity / (light_dist * light_dist)
                        light_color = Vec3f(light.color.r, light.color.g, light.color.b)
                        refl_color += refl_base_color .* (light_color * (diffuse * attenuation))
                    end

                    refl_color
                else
                    Vec3f(sky_color.r, sky_color.g, sky_color.b)
                end

                # Blend with primary color
                @inbounds primary_color = shading_queue[idx].color
                blended_color = primary_color * (1.0f0 - mat.metallic) + reflection_color * mat.metallic

                # Update shading result
                @inbounds shading_queue[idx] = ShadedResult(
                    blended_color,
                    primary_hit.pixel_x,
                    primary_hit.pixel_y,
                    primary_hit.sample_idx
                )
            end
        end
    end
end

# ============================================================================
# Stage 9: Accumulate Final Image
# ============================================================================

@kernel function accumulate_final!(
    @Const(shading_queue),
    @Const(samples_per_pixel),
    img,
    sample_accumulator
)
    idx = @index(Global, Linear)
    if idx <= length(shading_queue)
        @inbounds result = shading_queue[idx]
        @inbounds sample_accumulator[idx] = result.color
    end
end

@kernel function finalize_image!(
    @Const(sample_accumulator),
    @Const(samples_per_pixel),
    img
)
    idx = @index(Global, Linear)
    height, width = size(img)

    if idx <= width * height
        x = Int32(((idx - 1) % width) + 1)
        y = Int32(((idx - 1) ÷ width) + 1)

        color_sum = Vec3f(0, 0, 0)
        for s in 1:samples_per_pixel
            sample_idx = (idx - 1) * samples_per_pixel + s
            @inbounds color_sum = color_sum .+ sample_accumulator[sample_idx]
        end

        avg_color = color_sum ./ Float32(samples_per_pixel)
        @inbounds img[y, x] = RGB{Float32}(avg_color[1], avg_color[2], avg_color[3])
    end
end

# ============================================================================
# Main Wavefront Tracer (Full Features)
# ============================================================================

function similar_soa(img, T, num_elements)
    fields = [f => similar(img, fieldtype(T, f), num_elements) for f in fieldnames(T)]
    return (; fields...)
end


# ============================================================================
# WavefrontRenderer - Struct to hold all buffers for wavefront rendering
# ============================================================================

"""
    WavefrontRenderer

A renderer struct that contains all buffers needed for wavefront path tracing.
Supports different array types (Array, ROCArray, CLArray, etc.) and can be
converted to GPU using `to_gpu(ArrayType, renderer)`.

# Fields
- `framebuffer`: Output image buffer
- `bvh`: BVH acceleration structure
- `ctx`: Scene context (materials, lights)
- Camera parameters: `camera_pos`, `fov`, `sky_color`, `samples_per_pixel`
- Work queues for each wavefront stage
"""
struct WavefrontRenderer{ImgArr <: AbstractMatrix, BVH, Ctx}
    # Image dimensions
    width::Int32
    height::Int32

    # Image and scene
    framebuffer::ImgArr
    bvh::BVH
    ctx::Ctx

    # Camera parameters
    camera_pos::Point3f
    fov::Float32
    sky_color::RGB{Float32}
    samples_per_pixel::Int32

    # Work queues
    primary_ray_queue::AbstractVector
    primary_hit_queue::AbstractVector
    shadow_ray_queue::AbstractVector
    shadow_result_queue::AbstractVector
    reflection_ray_soa::NamedTuple
    reflection_hit_soa::NamedTuple
    shading_queue::AbstractVector
    sample_accumulator::AbstractVector
    active_count::AbstractVector
end

"""
    WavefrontRenderer(img, bvh, ctx; camera_pos, fov, sky_color, samples_per_pixel)

Create a WavefrontRenderer with all necessary buffers allocated for the given image size and scene.
"""
function WavefrontRenderer(
        img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5),
        fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
        samples_per_pixel=4
    )
    height, width = size(img)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_lights = Int32(length(ctx.lights))
    num_shadow_rays = num_rays * num_lights

    # Allocate work queues
    primary_ray_queue = similar(img, PrimaryRayWork, num_rays)
    primary_hit_queue = similar(img, PrimaryHitWork{eltype(bvh.original_triangles)}, num_rays)
    shadow_ray_queue = similar(img, ShadowRayWork, num_shadow_rays)
    shadow_result_queue = similar(img, ShadowResult, num_shadow_rays)

    # Stages 6 & 7: Use SoA layout directly (OpenCL compatibility)
    reflection_ray_soa = similar_soa(img, ReflectionRayWork, num_rays)
    reflection_hit_soa = similar_soa(img, ReflectionHitWork{Int32}, num_rays)

    shading_queue = similar(img, ShadedResult, num_rays)
    sample_accumulator = similar(img, Vec3f, num_rays)
    active_count = similar(img, Int32, 1)

    return WavefrontRenderer(
        Int32(width), Int32(height),
        img, bvh, ctx,
        camera_pos, fov, sky_color, Int32(samples_per_pixel),
        primary_ray_queue, primary_hit_queue,
        shadow_ray_queue, shadow_result_queue,
        reflection_ray_soa, reflection_hit_soa,
        shading_queue, sample_accumulator, active_count
    )
end

"""
    to_gpu(ArrayType, renderer::WavefrontRenderer)

Convert a WavefrontRenderer to use a different array type (e.g., ROCArray, CLArray).
This creates a new renderer with all buffers converted to the target array type.
"""
function Raycore.to_gpu(Arr, renderer::WavefrontRenderer)
    # Convert image
    img = Arr(renderer.framebuffer)

    # Convert BVH and context
    bvh_gpu = Raycore.to_gpu(Arr, renderer.bvh)
    ctx_gpu = Raycore.to_gpu(Arr, renderer.ctx)

    # Recreate renderer with new array type
    return WavefrontRenderer(
        img, bvh_gpu, ctx_gpu;
        camera_pos=renderer.camera_pos,
        fov=renderer.fov,
        sky_color=renderer.sky_color,
        samples_per_pixel=Int(renderer.samples_per_pixel)
    )
end

"""
    trace_wavefront_full!(renderer::WavefrontRenderer)

Execute the full wavefront path tracing pipeline using the provided renderer.
The result is stored in `renderer.framebuffer`.
"""
function render!(renderer::WavefrontRenderer)
    width = Int(renderer.width)
    height = Int(renderer.height)
    samples_per_pixel = Int(renderer.samples_per_pixel)

    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(renderer.fov / 2))

    backend = KA.get_backend(renderer.framebuffer)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_lights = Int32(length(renderer.ctx.lights))
    num_shadow_rays = num_rays * num_lights

    # Stage 1: Generate primary rays
    gen_kernel! = generate_primary_rays!(backend)
    gen_kernel!(
        renderer.width, renderer.height,
        renderer.camera_pos, focal_length, aspect,
        renderer.samples_per_pixel,
        renderer.primary_ray_queue,
        ndrange=num_pixels
    )
    # Stage 2: Intersect primary rays
    intersect_kernel! = intersect_primary_rays!(backend)
    intersect_kernel!(
        renderer.bvh,
        renderer.primary_ray_queue,
        renderer.primary_hit_queue,
        ndrange=num_rays
    )

    # Stage 3: Generate shadow rays
    shadow_gen_kernel! = generate_shadow_rays!(backend)
    shadow_gen_kernel!(
        renderer.primary_hit_queue,
        renderer.ctx,
        renderer.shadow_ray_queue,
        num_lights,
        ndrange=num_rays
    )
    # Stage 4: Test shadow rays
    shadow_test_kernel! = test_shadow_rays!(backend)
    shadow_test_kernel!(
        renderer.bvh,
        renderer.shadow_ray_queue,
        renderer.shadow_result_queue,
        ndrange=num_shadow_rays
    )

    # Stage 5: Shade primary hits with shadows
    shade_kernel! = shade_primary_hits!(backend)
    shade_kernel!(
        renderer.primary_hit_queue,
        renderer.ctx,
        renderer.shadow_result_queue,
        renderer.sky_color,
        num_lights,
        renderer.shading_queue,
        ndrange=num_rays
    )

    # Stage 6: Generate reflection rays (SoA)
    refl_gen_kernel! = generate_reflection_rays!(backend)
    refl_gen_kernel!(
        renderer.primary_hit_queue,
        renderer.ctx,
        renderer.reflection_ray_soa,
        renderer.active_count,
        ndrange=num_rays
    )

    # Stage 7: Intersect reflection rays (SoA)
    refl_intersect_kernel! = intersect_reflection_rays!(backend)
    refl_intersect_kernel!(
        renderer.bvh,
        renderer.reflection_ray_soa,
        renderer.reflection_hit_soa,
        ndrange=num_rays
    )

    # Stage 8: Shade reflections and blend (using SoA)
    refl_shade_kernel! = shade_reflections_and_blend!(backend)
    refl_shade_kernel!(
        renderer.primary_hit_queue,
        renderer.reflection_hit_soa,
        renderer.bvh.original_triangles,
        renderer.ctx,
        renderer.sky_color,
        renderer.shading_queue,
        ndrange=num_rays
    )

    # Stage 9: Accumulate final image
    accum_kernel! = accumulate_final!(backend)
    accum_kernel!(
        renderer.shading_queue,
        renderer.samples_per_pixel,
        renderer.framebuffer,
        renderer.sample_accumulator,
        ndrange=num_rays
    )

    final_kernel! = finalize_image!(backend)
    final_kernel!(
        renderer.sample_accumulator,
        renderer.samples_per_pixel,
        renderer.framebuffer,
        ndrange=num_pixels
    )
    KA.synchronize(backend)

    return renderer.framebuffer
end

function trace_wavefront_full(
        img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0,0.7f0,1.0f0),
        samples_per_pixel=4
    )
    height, width = size(img)
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    backend = KA.get_backend(img)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_lights = Int32(length(ctx.lights))
    num_shadow_rays = num_rays * num_lights

    # Allocate work queues
    primary_ray_queue = similar(img, PrimaryRayWork, num_rays)
    primary_hit_queue = similar(img, PrimaryHitWork{eltype(bvh.original_triangles)}, num_rays)
    shadow_ray_queue = similar(img, ShadowRayWork, num_shadow_rays)
    shadow_result_queue = similar(img, ShadowResult, num_shadow_rays)

    # Stages 6 & 7: Use SoA layout directly (OpenCL compatibility)
    reflection_ray_soa = similar_soa(img, ReflectionRayWork, num_rays)
    reflection_hit_soa = similar_soa(img, ReflectionHitWork{Int32}, num_rays)

    shading_queue = similar(img, ShadedResult, num_rays)
    sample_accumulator = similar(img, Vec3f, num_rays)

    # Stage 1: Generate primary rays
    gen_kernel! = generate_primary_rays!(backend)
    gen_kernel!(width, height, camera_pos, focal_length, aspect, Int32(samples_per_pixel), primary_ray_queue, ndrange=num_pixels)
    KA.synchronize(backend)

    # Stage 2: Intersect primary rays
    intersect_kernel! = intersect_primary_rays!(backend)
    intersect_kernel!(bvh, primary_ray_queue, primary_hit_queue, ndrange=num_rays)
    KA.synchronize(backend)

    # Stage 3: Generate shadow rays
    shadow_gen_kernel! = generate_shadow_rays!(backend)
    shadow_gen_kernel!(primary_hit_queue, ctx, shadow_ray_queue, num_lights, ndrange=num_rays)
    # Stage 4: Test shadow rays
    shadow_test_kernel! = test_shadow_rays!(backend)
    shadow_test_kernel!(bvh, shadow_ray_queue, shadow_result_queue, ndrange=num_shadow_rays)

    # Stage 5: Shade primary hits with shadows
    shade_kernel! = shade_primary_hits!(backend)
    shade_kernel!(primary_hit_queue, ctx, shadow_result_queue, sky_color, num_lights, shading_queue, ndrange=num_rays)

    # Stage 6: Generate reflection rays (SoA)
    refl_gen_kernel! = generate_reflection_rays!(backend)
    active_count = similar(img, Int32, 1)
    refl_gen_kernel!(primary_hit_queue, ctx, reflection_ray_soa, active_count, ndrange=num_rays)
    KA.synchronize(backend)

    # Stage 7: Intersect reflection rays (SoA)
    refl_intersect_kernel! = intersect_reflection_rays!(backend)
    refl_intersect_kernel!(bvh, reflection_ray_soa, reflection_hit_soa, ndrange=num_rays)
    KA.synchronize(backend)

    # Stage 8: Shade reflections and blend (using SoA)
    refl_shade_kernel! = shade_reflections_and_blend!(backend)
    refl_shade_kernel!(primary_hit_queue, reflection_hit_soa, bvh.original_triangles, ctx, sky_color, shading_queue, ndrange=num_rays)

    # Stage 9: Accumulate final image
    accum_kernel! = accumulate_final!(backend)
    accum_kernel!(shading_queue, Int32(samples_per_pixel), img, sample_accumulator, ndrange=num_rays)

    final_kernel! = finalize_image!(backend)
    final_kernel!(sample_accumulator, Int32(samples_per_pixel), img, ndrange=num_pixels)
    KA.synchronize(backend)
    return img
end
