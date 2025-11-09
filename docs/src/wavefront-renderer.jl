using Raycore, GeometryBasics, LinearAlgebra
using Raycore: gpu_int
using Colors
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA
using Statistics
# ============================================================================
# SoA Access Macros
# ============================================================================

"""
    @get field1, field2, ... = soa[idx]

Macro to extract multiple fields from a Structure of Arrays (SoA) at index `idx`.
"""
macro get(expr)
    if expr.head != :(=)
        error("@get expects assignment syntax: @get field1, field2 = soa[idx]")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    # Parse left side (field names)
    if lhs isa Symbol
        fields = [lhs]
    elseif lhs.head == :tuple
        fields = lhs.args
    else
        error("@get left side must be field names or tuple of field names")
    end

    # Parse right side (soa[idx])
    if rhs.head != :ref
        error("@get right side must be array indexing: soa[idx]")
    end
    soa = rhs.args[1]
    idx = rhs.args[2]

    # Generate field extraction code
    assignments = [:($(esc(field)) = $(esc(soa)).$(field)[$(esc(idx))]) for field in fields]

    return Expr(:block, assignments...)
end

"""
    @set soa[idx] = (field1=val1, field2=val2, ...)

Macro to set multiple fields in a Structure of Arrays (SoA) at index `idx`.
Expects named tuple syntax on the right side.
"""
macro set(expr)
    if expr.head != :(=)
        error("@set expects assignment syntax: @set soa[idx] = (field1=val1, ...)")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    # Parse left side (soa[idx])
    if lhs.head != :ref
        error("@set left side must be array indexing: soa[idx]")
    end
    soa = lhs.args[1]
    idx = lhs.args[2]

    # Parse right side (named tuple or parameters)
    assignments = []
    if rhs.head == :tuple || rhs.head == :parameters
        for arg in rhs.args
            if arg isa Expr && arg.head == :(=)
                field = arg.args[1]
                val = arg.args[2]
                push!(assignments, :($(esc(soa)).$(field)[$(esc(idx))] = $(esc(val))))
            else
                error("@set expects named parameters: @set soa[idx] = (field=value, ...)")
            end
        end
    else
        error("@set expects a tuple with named fields: @set soa[idx] = (field=value, ...)")
    end

    return Expr(:block, assignments...)
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
    ray_queue,
    ::Val{NSamples}
) where {NSamples}
    i = @index(Global, Cartesian)
    y = gpu_int(i[1])
    x = gpu_int(i[2])

    @inbounds if y <= height && x <= width
        # Convert to linear pixel index
        pixel_idx = (y - gpu_int(1)) * width + x
        # Unroll sampling loop with ntuple
        ntuple(Val(NSamples)) do s
            s_idx = gpu_int(s)
            ray_idx = (pixel_idx - gpu_int(1)) * gpu_int(NSamples) + s_idx
            jitter = rand(Vec2f)
            ndc_x = (2.0f0 * (Float32(x) - 0.5f0 + jitter[1]) / Float32(width) - 1.0f0) * aspect
            ndc_y = 1.0f0 - 2.0f0 * (Float32(y) - 0.5f0 + jitter[2]) / Float32(height)
            direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
            ray = Raycore.Ray(o=camera_pos, d=direction)

            # Write to SoA using @set
            @set ray_queue[ray_idx] = (ray=ray, pixel_x=x, pixel_y=y, sample_idx=s_idx)
            nothing
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
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(ray_queue.ray)
        # Read from SoA
        @get ray, pixel_x, pixel_y, sample_idx = ray_queue[idx]
        hit_found, tri, dist, bary = Raycore.closest_hit(bvh, ray)
        # Write to SoA using @set
        @set hit_queue[idx] = (hit_found=hit_found, tri=tri, dist=dist, bary=Vec3f(bary),
                               ray=ray, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
    end
end

# ============================================================================
# Stage 3: Generate Shadow Rays (for all hits × all lights)
# ============================================================================

@kernel function generate_shadow_rays!(
    @Const(hit_queue),
    @Const(ctx),
    shadow_ray_queue,
    ::Val{NLights}
) where {NLights}
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        # Read from SoA
        @get hit_found, tri, dist, bary, ray = hit_queue[idx]

        if hit_found
            # Compute hit point and normal
            hit_point = ray.o + ray.d * dist
            v0, v1, v2 = Raycore.normals(tri)
            u, v, w = bary[1], bary[2], bary[3]
            normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

            # Generate shadow ray for each light (unrolled)
            ntuple(Val(NLights)) do light_idx
                light_idx_gpu = gpu_int(light_idx)
                shadow_ray_idx = (idx - gpu_int(1)) * gpu_int(NLights) + light_idx_gpu
                light = ctx.lights[light_idx_gpu]

                # Shadow ray towards light with proper bias to avoid shadow acne
                shadow_bias = 0.01f0
                shadow_origin = hit_point + normal * shadow_bias
                light_vec = light.position - shadow_origin
                shadow_dir = normalize(light_vec)
                light_dist = norm(light_vec)
                shadow_ray = Raycore.Ray(o=shadow_origin, d=shadow_dir, t_max=light_dist)

                # Write to SoA using @set
                @set shadow_ray_queue[shadow_ray_idx] = (ray=shadow_ray, hit_idx=idx, light_idx=light_idx_gpu)
            end
        else
            # Sky hit - no shadow rays needed, but we need to fill the queue
            dummy_ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
            ntuple(Val(NLights)) do light_idx
                light_idx_gpu = gpu_int(light_idx)
                shadow_ray_idx = (idx - gpu_int(1)) * gpu_int(NLights) + light_idx_gpu
                # Write dummy to SoA using @set
                @set shadow_ray_queue[shadow_ray_idx] = (ray=dummy_ray, hit_idx=idx, light_idx=light_idx_gpu)
                nothing
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
    i = @index(Global, Linear)
    idx = gpu_int(i)
    @inbounds if idx <= length(shadow_ray_queue.ray)
        # Read from SoA
        @get ray, hit_idx, light_idx = shadow_ray_queue[idx]

        # Test for occlusion
        # any_hit respects ray.t_max and only returns hits before the light
        # So if we get a hit, something is blocking the light
        visible = if ray.t_max > 0.0f0
            hit_found, _, _, _ = Raycore.any_hit(bvh, ray)
            !hit_found  # Visible only if no obstruction
        else
            false  # Dummy ray (sky hits)
        end
        # Write to SoA using @set
        @set shadow_result_queue[idx] = (visible=visible, hit_idx=hit_idx, light_idx=light_idx)
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
    shading_queue,
    ::Val{NLights}
) where {NLights}
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        # Read from SoA
        @get hit_found, tri, dist, bary, ray, pixel_x, pixel_y, sample_idx = hit_queue[idx]

        if hit_found
            # Compute hit point and normal
            hit_point = ray.o + ray.d * dist
            v0, v1, v2 = Raycore.normals(tri)
            u, v, w = bary[1], bary[2], bary[3]
            normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

            # Get material
            mat = ctx.materials[tri.material_idx]
            base_color = Vec3f(mat.base_color.r, mat.base_color.g, mat.base_color.b)
            # Start with ambient
            total_color = base_color * ctx.ambient
            # Add contribution from each light using shadow test results (unrolled)
            light_contributions = ntuple(Val(NLights)) do light_idx
                light_idx_gpu = gpu_int(light_idx)
                shadow_idx = (idx - gpu_int(1)) * gpu_int(NLights) + light_idx_gpu
                visible = shadow_results.visible[shadow_idx]

                if visible
                    light = ctx.lights[light_idx_gpu]
                    light_vec = light.position - hit_point
                    light_dist = norm(light_vec)
                    light_dir = light_vec / light_dist

                    diffuse = max(0.0f0, dot(normal, light_dir))
                    attenuation = light.intensity / (light_dist * light_dist)

                    light_color = Vec3f(light.color.r, light.color.g, light.color.b)
                    base_color .* (light_color * (diffuse * attenuation))
                else
                    Vec3f(0, 0, 0)
                end
            end

            # Sum all light contributions
            total_color += sum(light_contributions)
            # Write to SoA using @set
            @set shading_queue[idx] = (color=total_color, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
        else
            # Sky color
            sky_vec = Vec3f(sky_color.r, sky_color.g, sky_color.b)
            # Write to SoA using @set
            @set shading_queue[idx] = (color=sky_vec, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
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
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        # Read from SoA
        @get hit_found, tri, dist, bary, ray = hit_queue[idx]

        if hit_found
            mat = ctx.materials[tri.material_idx]
            if mat.metallic > 0.0f0
                # Compute reflection
                hit_point = ray.o + ray.d * dist
                v0, v1, v2 = Raycore.normals(tri)
                u, v, w = bary[1], bary[2], bary[3]
                normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

                wo = -ray.d
                reflect_dir = Raycore.reflect(wo, normal)

                # Add roughness
                if mat.roughness > 0.0f0
                    offset = (rand(Vec3f) .* 2.0f0 .- 1.0f0) * mat.roughness
                    reflect_dir = normalize(reflect_dir + offset)
                end
                # Use proper bias to avoid self-intersection
                reflect_ray = Raycore.Ray(o=hit_point + normal * 0.01f0, d=reflect_dir)

                # Write to SoA using @set
                @set reflection_ray_soa[idx] = (ray=reflect_ray, hit_idx=idx)
            else
                # No reflection - dummy ray
                dummy_ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
                @set reflection_ray_soa[idx] = (ray=dummy_ray, hit_idx=idx)
            end
        else
            # Sky hit - no reflection
            dummy_ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
            @set reflection_ray_soa[idx] = (ray=dummy_ray, hit_idx=idx)
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
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(reflection_ray_soa.ray)
        # Read from SoA
        @get ray, hit_idx = reflection_ray_soa[idx]

        if ray.t_max > 0.0f0
            hit_found, tri, dist, bary = Raycore.closest_hit(bvh, ray)
            # Write to SoA using @set
            @set reflection_hit_soa[idx] = (hit_found=hit_found, tri_idx=tri.primitive_idx,
                dist=dist, bary=Vec3f(bary),
                ray=ray, primary_hit_idx=hit_idx)
        else
            # Write only hit_found as false for dummy rays
            reflection_hit_soa.hit_found[idx] = false
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
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        # Read from SoA
        @get hit_found, tri, pixel_x, pixel_y, sample_idx = hit_queue[idx]

        if hit_found
            mat = ctx.materials[tri.material_idx]

            if mat.metallic > 0.0f0
                # Access from SoA directly (using different variable names for clarity)
                refl_hit_found = reflection_hit_soa.hit_found[idx]
                refl_tri_idx = reflection_hit_soa.tri_idx[idx]
                refl_dist = reflection_hit_soa.dist[idx]
                refl_bary = reflection_hit_soa.bary[idx]
                refl_ray = reflection_hit_soa.ray[idx]

                # Compute reflection color (simplified - no recursive shadows for performance)
                reflection_color = if refl_hit_found
                    refl_point = refl_ray.o + refl_ray.d * refl_dist

                    # Get triangle from BVH using stored index
                    refl_tri = bvh_triangles[refl_tri_idx]
                    v0, v1, v2 = Raycore.normals(refl_tri)
                    u, v, w = refl_bary[1], refl_bary[2], refl_bary[3]
                    refl_normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

                    refl_mat = ctx.materials[refl_tri.material_idx]
                    refl_base_color = Vec3f(refl_mat.base_color.r, refl_mat.base_color.g, refl_mat.base_color.b)

                    # Simplified lighting (ambient + first light only, no shadows)
                    refl_color = refl_base_color * ctx.ambient

                    if length(ctx.lights) > 0
                        light = ctx.lights[gpu_int(1)]
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
                primary_color = shading_queue.color[idx]
                blended_color = primary_color * (1.0f0 - mat.metallic) + reflection_color * mat.metallic

                # Update shading result in SoA using @set
                @set shading_queue[idx] = (color=blended_color, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
            end
        end
    end
end

# ============================================================================
# Stage 9: Accumulate Final Image
# ============================================================================

@kernel function accumulate_final!(
    @Const(shading_queue),
    img,
    sample_accumulator
)
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(shading_queue.color)
        # Read from SoA
        color = shading_queue.color[idx]
        sample_accumulator[idx] = color
    end
end

@kernel function finalize_image!(
    @Const(sample_accumulator),
    img,
    ::Val{NSamples}
) where {NSamples}
    i = @index(Global, Cartesian)
    y = gpu_int(i[1])
    x = gpu_int(i[2])
    height, width = size(img)

    @inbounds if y <= height && x <= width
        # Convert to linear index
        pixel_idx = (y - gpu_int(1)) * width + x
        # Unroll sample accumulation loop
        colors = ntuple(Val(NSamples)) do s
            s_idx = gpu_int(s)
            sample_idx = (pixel_idx - gpu_int(1)) * gpu_int(NSamples) + s_idx
            sample_accumulator[sample_idx]
        end

        # Sum all colors
        img[y, x] = RGB{Float32}(mean(colors)...)
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

    # Work queues (all SoA)
    primary_ray_queue::NamedTuple
    primary_hit_queue::NamedTuple
    shadow_ray_queue::NamedTuple
    shadow_result_queue::NamedTuple
    reflection_ray_soa::NamedTuple
    reflection_hit_soa::NamedTuple
    shading_queue::NamedTuple
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

    # Allocate work queues as SoA
    primary_ray_queue = similar_soa(img, PrimaryRayWork, num_rays)
    primary_hit_queue = similar_soa(img, PrimaryHitWork{eltype(bvh.original_triangles)}, num_rays)
    shadow_ray_queue = similar_soa(img, ShadowRayWork, num_shadow_rays)
    shadow_result_queue = similar_soa(img, ShadowResult, num_shadow_rays)
    reflection_ray_soa = similar_soa(img, ReflectionRayWork, num_rays)
    reflection_hit_soa = similar_soa(img, ReflectionHitWork{Int32}, num_rays)
    shading_queue = similar_soa(img, ShadedResult, num_rays)
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
    num_lights = Int(length(renderer.ctx.lights))
    num_shadow_rays = num_rays * num_lights

    # Stage 1: Generate primary rays
    gen_kernel! = generate_primary_rays!(backend)
    gen_kernel!(
        renderer.width, renderer.height,
        renderer.camera_pos, focal_length, aspect,
        renderer.primary_ray_queue,
        Val(samples_per_pixel),
        ndrange=(height, width)
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
        Val(num_lights),
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
        renderer.shading_queue,
        Val(num_lights),
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
        renderer.framebuffer,
        renderer.sample_accumulator,
        ndrange=num_rays
    )

    final_kernel! = finalize_image!(backend)
    final_kernel!(
        renderer.sample_accumulator,
        renderer.framebuffer,
        Val(samples_per_pixel),
        ndrange=(height, width)
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
    num_lights = Int(length(ctx.lights))
    num_shadow_rays = num_rays * num_lights

    # Allocate work queues as SoA
    primary_ray_queue = similar_soa(img, PrimaryRayWork, num_rays)
    primary_hit_queue = similar_soa(img, PrimaryHitWork{eltype(bvh.original_triangles)}, num_rays)
    shadow_ray_queue = similar_soa(img, ShadowRayWork, num_shadow_rays)
    shadow_result_queue = similar_soa(img, ShadowResult, num_shadow_rays)
    reflection_ray_soa = similar_soa(img, ReflectionRayWork, num_rays)
    reflection_hit_soa = similar_soa(img, ReflectionHitWork{Int32}, num_rays)
    shading_queue = similar_soa(img, ShadedResult, num_rays)
    sample_accumulator = similar(img, Vec3f, num_rays)

    # Stage 1: Generate primary rays
    gen_kernel! = generate_primary_rays!(backend)
    gen_kernel!(Int32(width), Int32(height), camera_pos, focal_length, aspect, primary_ray_queue, Val(samples_per_pixel), ndrange=(height, width))
    KA.synchronize(backend)

    # Stage 2: Intersect primary rays
    intersect_kernel! = intersect_primary_rays!(backend)
    intersect_kernel!(bvh, primary_ray_queue, primary_hit_queue, ndrange=num_rays)
    KA.synchronize(backend)

    # Stage 3: Generate shadow rays
    shadow_gen_kernel! = generate_shadow_rays!(backend)
    shadow_gen_kernel!(primary_hit_queue, ctx, shadow_ray_queue, Val(num_lights), ndrange=num_rays)

    # Stage 4: Test shadow rays
    shadow_test_kernel! = test_shadow_rays!(backend)
    shadow_test_kernel!(bvh, shadow_ray_queue, shadow_result_queue, ndrange=num_shadow_rays)

    # Stage 5: Shade primary hits with shadows
    shade_kernel! = shade_primary_hits!(backend)
    shade_kernel!(primary_hit_queue, ctx, shadow_result_queue, sky_color, shading_queue, Val(num_lights), ndrange=num_rays)

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
    accum_kernel!(shading_queue, img, sample_accumulator, ndrange=num_rays)

    final_kernel! = finalize_image!(backend)
    final_kernel!(sample_accumulator, img, Val(samples_per_pixel), ndrange=(height, width))
    KA.synchronize(backend)
    return img
end
