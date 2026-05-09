"""
Wavefront Path Tracer integrated with Hikari MaterialScene.

This renderer uses Hikari's MaterialScene and Light types while keeping
the original wavefront-renderer.jl shading model. Materials are treated
as data containers - we extract base_color, metallic, roughness from
Hikari materials but use the wavefront shading equations.
"""

using Raycore, GeometryBasics, LinearAlgebra
using Raycore: gpu_int
using Colors
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA
using Statistics
import Makie

import Hikari
using Hikari: MaterialScene, MaterialIndex, RGBSpectrum, PointLight, Light, LightδPosition

# ============================================================================
# SoA Access Macros (same as wavefront-renderer.jl)
# ============================================================================

macro get(expr)
    if expr.head != :(=)
        error("@get expects assignment syntax: @get field1, field2 = soa[idx]")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    if lhs isa Symbol
        fields = [lhs]
    elseif lhs.head == :tuple
        fields = lhs.args
    else
        error("@get left side must be field names or tuple of field names")
    end

    if rhs.head != :ref
        error("@get right side must be array indexing: soa[idx]")
    end
    soa = rhs.args[1]
    idx = rhs.args[2]

    assignments = [:($(esc(field)) = $(esc(soa)).$(field)[$(esc(idx))]) for field in fields]
    return Expr(:block, assignments...)
end

macro set(expr)
    if expr.head != :(=)
        error("@set expects assignment syntax: @set soa[idx] = (field1=val1, ...)")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    if lhs.head != :ref
        error("@set left side must be array indexing: soa[idx]")
    end
    soa = lhs.args[1]
    idx = lhs.args[2]

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
# Material Property Extraction from Hikari Materials
# ============================================================================

"""
    WavefrontMaterialProps

Simple material properties for wavefront shading.
Extracted from Hikari materials at shading time.
"""
struct WavefrontMaterialProps
    base_color::Vec3f
    metallic::Float32
    roughness::Float32
end

"""
Helper to get texture const_value (for constant textures used in our scene).
"""
@inline function get_texture_value(tex::Hikari.Texture{RGBSpectrum})
    return tex.const_value
end

@inline function get_texture_value(tex::Hikari.Texture{Float32})
    return tex.const_value
end

"""
Extract wavefront-compatible material properties from a Hikari MatteMaterial.
Matte materials are purely diffuse (metallic=0, roughness from sigma).
"""
@inline function extract_material_props(mat::Hikari.MatteMaterial, ::Point2f)
    # Get diffuse color from Kd texture
    kd = get_texture_value(mat.Kd)
    base_color = Vec3f(kd.c[1], kd.c[2], kd.c[3])
    # σ is roughness in degrees for Oren-Nayar; convert to 0-1 range
    σ = get_texture_value(mat.σ)
    roughness = clamp(σ / 90f0, 0f0, 1f0)
    return WavefrontMaterialProps(base_color, 0f0, roughness)
end

"""
Extract wavefront-compatible material properties from a Hikari MirrorMaterial.
Mirrors are fully metallic (metallic=1, roughness=0).
"""
@inline function extract_material_props(mat::Hikari.MirrorMaterial, ::Point2f)
    kr = get_texture_value(mat.Kr)
    base_color = Vec3f(kr.c[1], kr.c[2], kr.c[3])
    return WavefrontMaterialProps(base_color, 1f0, 0f0)
end

"""
Extract wavefront-compatible material properties from a Hikari PlasticMaterial.
Plastic has both diffuse and specular; map to metallic based on Ks intensity.
"""
@inline function extract_material_props(mat::Hikari.PlasticMaterial, ::Point2f)
    kd = get_texture_value(mat.Kd)
    ks = get_texture_value(mat.Ks)
    base_color = Vec3f(kd.c[1], kd.c[2], kd.c[3])
    # Map specular intensity to metallic factor
    metallic = (ks.c[1] + ks.c[2] + ks.c[3]) / 3f0
    roughness = get_texture_value(mat.roughness)
    return WavefrontMaterialProps(base_color, metallic, roughness)
end

"""
Extract wavefront-compatible material properties from a Hikari GlassMaterial.
Glass is mapped to a transparent-ish material; use reflection as base color.
"""
@inline function extract_material_props(mat::Hikari.GlassMaterial, ::Point2f)
    kr = get_texture_value(mat.Kr)
    base_color = Vec3f(kr.c[1], kr.c[2], kr.c[3])
    roughness = get_texture_value(mat.u_roughness)
    # Glass behaves more like a mirror in our simplified model
    return WavefrontMaterialProps(base_color, 0.8f0, roughness)
end

"""
Extract wavefront-compatible material properties from a Hikari MetalMaterial.
Metals are fully metallic (metallic=1) with roughness and reflectance for color.
"""
@inline function extract_material_props(mat::Hikari.MetalMaterial, ::Point2f)
    # Use reflectance as the base color (tinting)
    refl = get_texture_value(mat.reflectance)
    base_color = Vec3f(refl.c[1], refl.c[2], refl.c[3])
    roughness = get_texture_value(mat.roughness)
    # Metals are fully metallic
    return WavefrontMaterialProps(base_color, 1f0, roughness)
end

# Fallback for any other material type
@inline function extract_material_props(mat, ::Point2f)
    return WavefrontMaterialProps(Vec3f(0.5f0, 0.5f0, 0.5f0), 0f0, 0.5f0)
end

"""
Generated function for type-stable material property extraction.
Dispatches to the appropriate extract_material_props based on material_type.
"""
@generated function extract_material_from_scene(
    materials::NTuple{N,Any}, idx::MaterialIndex, uv::Point2f
) where N
    branches = [quote
        if idx.material_type === UInt8($i)
            return @inline extract_material_props(@inbounds(materials[$i][idx.material_idx]), uv)
        end
    end for i in 1:N]
    quote
        $(branches...)
        return WavefrontMaterialProps(Vec3f(0.5f0), 0f0, 0.5f0)
    end
end

# ============================================================================
# Light Representation for Wavefront Renderer
# ============================================================================

"""
    WavefrontLight

Simple point light for wavefront shading, compatible with GPU.
"""
struct WavefrontLight
    position::Point3f
    intensity::Float32
    color::RGB{Float32}
end

"""
Convert a Hikari PointLight to WavefrontLight format.
"""
function WavefrontLight(light::Hikari.PointLight)
    # Extract intensity as luminance of spectrum
    intensity = Hikari.to_Y(light.i)
    # Extract color (normalized RGB)
    rgb = Hikari.rgb(light.i)
    max_val = max(rgb[1], rgb[2], rgb[3], 1f-6)
    color = RGB{Float32}(rgb[1]/max_val, rgb[2]/max_val, rgb[3]/max_val)
    return WavefrontLight(light.position, intensity, color)
end

# ============================================================================
# Work Queue Structures (same as wavefront-renderer.jl)
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
    hit_idx::Int32
    light_idx::Int32
end

struct ShadowResult
    visible::Bool
    hit_idx::Int32
    light_idx::Int32
end

struct ReflectionRayWork
    ray::Raycore.Ray
    hit_idx::Int32
end

struct ReflectionHitWork
    hit_found::Bool
    material_idx::MaterialIndex
    dist::Float32
    bary::Vec3f
    normal::Vec3f
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
# Render Context for Hikari Integration
# ============================================================================

"""
    HikariRenderContext

Holds the lights and materials for GPU rendering.
Uses WavefrontLight for lights (converted from Hikari lights).
Materials are stored as the original Hikari MaterialScene.materials tuple.
"""
struct HikariRenderContext{L<:AbstractVector{WavefrontLight}, M<:Tuple}
    lights::L
    materials::M
    ambient::Float32
end

function Raycore.to_gpu(Arr, ctx::HikariRenderContext)
    lights_gpu = Raycore.to_gpu(Arr, ctx.lights)
    # Materials tuple needs per-type GPU conversion
    materials_gpu = map(ctx.materials) do mats
        Raycore.to_gpu(Arr, map(m -> Hikari.to_gpu(Arr, m), mats))
    end
    return HikariRenderContext(lights_gpu, materials_gpu, ctx.ambient)
end

# ============================================================================
# Helper function
# ============================================================================

function similar_soa(img, T, num_elements)
    fields = [f => similar(img, fieldtype(T, f), num_elements) for f in fieldnames(T)]
    return (; fields...)
end

@generated function for_unrolled(f::F, ::Val{N}) where {F, N}
    return Expr(:block, [:(f($(Raycore.gpu_int(i)))) for i in 1:N]...)
end

# ============================================================================
# Stage 1: Generate Primary Camera Rays
# ============================================================================

@kernel function generate_primary_rays_lookat!(
    @Const(width), @Const(height),
    @Const(camera_pos),
    @Const(camera_right), @Const(camera_up), @Const(camera_forward),
    @Const(half_width), @Const(half_height),
    ray_queue,
    ::Val{NSamples}
) where {NSamples}
    i = @index(Global, Cartesian)
    y = gpu_int(i[1])
    x = gpu_int(i[2])

    @inbounds if y <= height && x <= width
        pixel_idx = (y - gpu_int(1)) * width + x
        ntuple(Val(NSamples)) do s
            s_idx = gpu_int(s)
            ray_idx = (pixel_idx - gpu_int(1)) * gpu_int(NSamples) + s_idx
            jitter = rand(Vec2f)

            u = (2.0f0 * (Float32(x) - 0.5f0 + jitter[1]) / Float32(width) - 1.0f0)
            v = (1.0f0 - 2.0f0 * (Float32(y) - 0.5f0 + jitter[2]) / Float32(height))

            direction = normalize(
                camera_forward +
                camera_right * (u * half_width) +
                camera_up * (v * half_height)
            )
            ray = Raycore.Ray(o=camera_pos, d=direction)

            @set ray_queue[ray_idx] = (ray=ray, pixel_x=x, pixel_y=y, sample_idx=s_idx)
            nothing
        end
    end
end

# ============================================================================
# Stage 2: Intersect Primary Rays (adapted for MaterialScene)
# ============================================================================

@kernel function intersect_primary_rays_hikari!(
    @Const(accel),  # BVH or TLAS from MaterialScene
    @Const(ray_queue),
    hit_queue
)
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(ray_queue.ray)
        @get ray, pixel_x, pixel_y, sample_idx = ray_queue[idx]
        hit_found, tri, dist, bary = Raycore.closest_hit(accel, ray)
        @set hit_queue[idx] = (hit_found=hit_found, tri=tri, dist=dist, bary=Vec3f(bary),
                               ray=ray, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
    end
end

# ============================================================================
# Stage 3: Generate Shadow Rays
# ============================================================================

@kernel function generate_shadow_rays_hikari!(
    @Const(hit_queue),
    @Const(ctx),
    shadow_ray_queue,
    nlights::Val{NLights}
) where {NLights}
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        @get hit_found, tri, dist, bary, ray = hit_queue[idx]

        if hit_found
            hit_point = ray.o + ray.d * dist
            v0, v1, v2 = Raycore.normals(tri)
            u, v, w = bary[1], bary[2], bary[3]
            normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

            for_unrolled(nlights) do light_idx
                light_idx_gpu = gpu_int(light_idx)
                shadow_ray_idx = (idx - gpu_int(1)) * gpu_int(NLights) + light_idx_gpu
                light = ctx.lights[light_idx_gpu]

                shadow_bias = 0.01f0
                shadow_origin = hit_point + normal * shadow_bias
                light_vec = light.position - shadow_origin
                shadow_dir = normalize(light_vec)
                light_dist = norm(light_vec)
                shadow_ray = Raycore.Ray(o=shadow_origin, d=shadow_dir, t_max=light_dist)

                @set shadow_ray_queue[shadow_ray_idx] = (ray=shadow_ray, hit_idx=idx, light_idx=light_idx_gpu)
            end
        else
            dummy_ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
            for_unrolled(nlights) do light_idx
                light_idx_gpu = gpu_int(light_idx)
                shadow_ray_idx = (idx - gpu_int(1)) * gpu_int(NLights) + light_idx_gpu
                shadow_ray_queue.ray[shadow_ray_idx] = dummy_ray
            end
        end
    end
end

# ============================================================================
# Stage 4: Test Shadow Rays
# ============================================================================

@kernel function test_shadow_rays_hikari!(
    @Const(accel),
    @Const(shadow_ray_queue),
    shadow_result_queue
)
    i = @index(Global, Linear)
    idx = gpu_int(i)
    @inbounds if idx <= length(shadow_ray_queue.ray)
        @get ray, hit_idx, light_idx = shadow_ray_queue[idx]

        visible = if ray.t_max > 0.0f0
            hit_found, _, _, _ = Raycore.any_hit(accel, ray)
            !hit_found
        else
            false
        end

        @set shadow_result_queue[idx] = (visible=visible, hit_idx=hit_idx, light_idx=light_idx)
    end
end

# ============================================================================
# Stage 5: Shade Primary Hits with Shadow Information (Hikari materials)
# ============================================================================

@kernel function shade_primary_hits_hikari!(
    @Const(hit_queue),
    @Const(ctx),
    @Const(shadow_results),
    @Const(sky_color),
    shading_queue,
    nlights::Val{NLights}
) where {NLights}
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        @get hit_found, tri, dist, bary, ray, pixel_x, pixel_y, sample_idx = hit_queue[idx]

        if hit_found
            hit_point = ray.o + ray.d * dist
            v0, v1, v2 = Raycore.normals(tri)
            u, v, w = bary[1], bary[2], bary[3]
            normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

            # Get material properties from Hikari material via MaterialIndex
            mat_idx = tri.metadata::MaterialIndex
            mat_props = extract_material_from_scene(ctx.materials, mat_idx, Point2f(0))
            base_color = mat_props.base_color

            # Start with ambient
            total_color = base_color * ctx.ambient

            # Add contribution from each light
            light_samples = ntuple(nlights) do light_idx
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
                    difa = (diffuse * attenuation)
                    Vec3f(base_color * (light_color * difa))
                else
                    Vec3f(0)
                end
            end

            final_color = total_color + sum(light_samples)
            @set shading_queue[idx] = (color=final_color, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
        else
            sky_vec = Vec3f(sky_color.r, sky_color.g, sky_color.b)
            @set shading_queue[idx] = (color=sky_vec, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
        end
    end
end

# ============================================================================
# Stage 6: Generate Reflection Rays (Hikari materials)
# ============================================================================

@kernel function generate_reflection_rays_hikari!(
    @Const(hit_queue),
    @Const(ctx),
    reflection_ray_soa,
    active_count
)
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        @get hit_found, tri, dist, bary, ray = hit_queue[idx]
        dummy_ray = Raycore.Ray(o=Point3f(0, 0, 0), d=Vec3f(0, 0, 1), t_max=0.0f0)

        if hit_found
            mat_idx = tri.metadata::MaterialIndex
            mat_props = extract_material_from_scene(ctx.materials, mat_idx, Point2f(0))

            if mat_props.metallic > 0.0f0
                hit_point = ray.o + ray.d * dist
                v0, v1, v2 = Raycore.normals(tri)
                u, v, w = bary[1], bary[2], bary[3]
                normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

                wo = -ray.d
                reflect_dir = Raycore.reflect(wo, normal)

                if mat_props.roughness > 0.0f0
                    offset = (rand(Vec3f) .* 2.0f0 .- 1.0f0) * mat_props.roughness
                    reflect_dir = normalize(reflect_dir + offset)
                end

                reflect_ray = Raycore.Ray(o=hit_point + normal * 0.01f0, d=reflect_dir)
                @set reflection_ray_soa[idx] = (ray=reflect_ray, hit_idx=idx)
            else
                reflection_ray_soa.ray[idx] = dummy_ray
            end
        else
            reflection_ray_soa.ray[idx] = dummy_ray
        end
    end
end

# ============================================================================
# Stage 7: Intersect Reflection Rays (Hikari materials)
# ============================================================================

@kernel function intersect_reflection_rays_hikari!(
    @Const(accel),
    @Const(reflection_ray_soa),
    reflection_hit_soa
)
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(reflection_ray_soa.ray)
        @get ray, hit_idx = reflection_ray_soa[idx]

        if ray.t_max > 0.0f0
            hit_found, tri, dist, bary = Raycore.closest_hit(accel, ray)
            if hit_found
                v0, v1, v2 = Raycore.normals(tri)
                u, v, w = bary[1], bary[2], bary[3]
                normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

                # Store MaterialIndex instead of Int32 material_idx
                mat_idx = tri.metadata::MaterialIndex
                @set reflection_hit_soa[idx] = (hit_found=true, material_idx=mat_idx,
                    dist=dist, bary=Vec3f(bary), normal=normal,
                    ray=ray, primary_hit_idx=hit_idx)
            else
                reflection_hit_soa.hit_found[idx] = false
            end
        else
            reflection_hit_soa.hit_found[idx] = false
        end
    end
end

# ============================================================================
# Stage 8: Shade Reflection Hits and Blend (Hikari materials)
# ============================================================================

@kernel function shade_reflections_and_blend_hikari!(
    @Const(hit_queue),
    @Const(reflection_hit_soa),
    @Const(ctx),
    @Const(sky_color),
    shading_queue
)
    i = @index(Global, Linear)
    idx = gpu_int(i)

    @inbounds if idx <= length(hit_queue.hit_found)
        @get hit_found, tri, pixel_x, pixel_y, sample_idx = hit_queue[idx]

        if hit_found
            mat_idx = tri.metadata::MaterialIndex
            mat_props = extract_material_from_scene(ctx.materials, mat_idx, Point2f(0))

            if mat_props.metallic > 0.0f0
                @get hit_found, material_idx, dist, bary, normal, ray = reflection_hit_soa[idx]

                reflection_color = if hit_found
                    refl_point = ray.o + ray.d * dist
                    refl_normal = normal

                    refl_mat_props = extract_material_from_scene(ctx.materials, material_idx, Point2f(0))
                    refl_base_color = refl_mat_props.base_color

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

                primary_color = shading_queue.color[idx]
                blended_color = primary_color * (1.0f0 - mat_props.metallic) + reflection_color * mat_props.metallic

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
        color = shading_queue.color[idx]
        sample_accumulator[idx] = color
    end
end

@kernel function finalize_image!(
    @Const(sample_accumulator),
    img,
    nsamples::Val{NSamples}
) where {NSamples}
    i = @index(Global, Cartesian)
    y = gpu_int(i[1])
    x = gpu_int(i[2])
    height, width = size(img)

    @inbounds if y <= height && x <= width
        pixel_idx = (y - gpu_int(1)) * width + x
        samples = ntuple(nsamples) do idx
            s_idx = gpu_int(idx)
            sample_idx = (pixel_idx - gpu_int(1)) * gpu_int(NSamples) + s_idx
            sample_accumulator[sample_idx]
        end
        img[y, x] = RGB{Float32}(mean(samples)...)
    end
end

# ============================================================================
# HikariWavefrontRenderer
# ============================================================================

"""
    HikariWavefrontRenderer

A wavefront renderer that uses Hikari's MaterialScene for materials and lights.
Uses the same shading model as the original wavefront-renderer.jl but with
Hikari data structures.

Materials are stored in MaterialScene (tuple of typed vectors) and lights
are converted to WavefrontLight format for GPU compatibility.
"""
struct HikariWavefrontRenderer{ImgArr <: AbstractMatrix, Accel, Ctx}
    width::Int32
    height::Int32

    framebuffer::ImgArr
    accel::Accel  # BVH or TLAS from MaterialScene
    ctx::Ctx      # HikariRenderContext

    camera_pos::Point3f
    camera_lookat::Point3f
    camera_up::Vec3f
    fov::Float32
    sky_color::RGB{Float32}
    samples_per_pixel::Int32

    # Work queues
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
    HikariWavefrontRenderer(img, material_scene, lights; kwargs...)

Create a HikariWavefrontRenderer from a Hikari MaterialScene and lights.

# Arguments
- `img`: Output image buffer
- `material_scene`: Hikari.MaterialScene containing geometry and materials
- `lights`: Vector of Hikari lights (PointLight, etc.)

# Keyword Arguments
- `camera_pos`: Camera position (default: Point3f(0, -0.9, -2.5))
- `camera_lookat`: Look-at target (default: Point3f(0, 0, 0))
- `camera_up`: Up vector (default: Vec3f(0, 0, 1))
- `fov`: Field of view in degrees (default: 45)
- `sky_color`: Background color (default: light blue)
- `samples_per_pixel`: Anti-aliasing samples (default: 4)
- `ambient`: Ambient light factor (default: 0.1)
"""
function HikariWavefrontRenderer(
        img,
        material_scene::MaterialScene,
        lights::AbstractVector{<:Light};
        camera_pos=Point3f(0, -0.9, -2.5),
        camera_lookat=Point3f(0, 0, 0),
        camera_up=Vec3f(0, 0, 1),
        fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
        samples_per_pixel=4,
        ambient=0.1f0
    )
    height, width = size(img)

    # Convert Hikari lights to WavefrontLight
    wavefront_lights = [WavefrontLight(l) for l in lights]

    # Create render context
    ctx = HikariRenderContext(wavefront_lights, material_scene.materials, ambient)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_lights = Int32(length(wavefront_lights))
    num_shadow_rays = num_rays * num_lights

    # Get triangle type from accelerator
    accel = material_scene.accel
    tri_type = eltype(accel)

    # Allocate work queues
    primary_ray_queue = similar_soa(img, PrimaryRayWork, num_rays)
    primary_hit_queue = similar_soa(img, PrimaryHitWork{tri_type}, num_rays)
    shadow_ray_queue = similar_soa(img, ShadowRayWork, num_shadow_rays)
    shadow_result_queue = similar_soa(img, ShadowResult, num_shadow_rays)
    reflection_ray_soa = similar_soa(img, ReflectionRayWork, num_rays)
    reflection_hit_soa = similar_soa(img, ReflectionHitWork, num_rays)
    shading_queue = similar_soa(img, ShadedResult, num_rays)
    sample_accumulator = similar(img, Vec3f, num_rays)
    active_count = similar(img, Int32, 1)

    return HikariWavefrontRenderer(
        Int32(width), Int32(height),
        img, accel, ctx,
        camera_pos, camera_lookat, camera_up,
        fov, sky_color, Int32(samples_per_pixel),
        primary_ray_queue, primary_hit_queue,
        shadow_ray_queue, shadow_result_queue,
        reflection_ray_soa, reflection_hit_soa,
        shading_queue, sample_accumulator, active_count
    )
end

"""
    to_gpu(ArrayType, renderer::HikariWavefrontRenderer)

Convert a HikariWavefrontRenderer to GPU arrays.
"""
function Raycore.to_gpu(Arr, renderer::HikariWavefrontRenderer)
    img = Arr(renderer.framebuffer)
    accel_gpu = Raycore.to_gpu(Arr, renderer.accel)
    ctx_gpu = Raycore.to_gpu(Arr, renderer.ctx)

    return HikariWavefrontRenderer(
        img, accel_gpu, ctx_gpu;
        camera_pos=renderer.camera_pos,
        camera_lookat=renderer.camera_lookat,
        camera_up=renderer.camera_up,
        fov=renderer.fov,
        sky_color=renderer.sky_color,
        samples_per_pixel=Int(renderer.samples_per_pixel),
        ambient=renderer.ctx.ambient
    )
end

# Inner constructor for to_gpu
function HikariWavefrontRenderer(
        img, accel, ctx;
        camera_pos, camera_lookat, camera_up, fov, sky_color, samples_per_pixel, ambient
    )
    height, width = size(img)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_lights = Int32(length(ctx.lights))
    num_shadow_rays = num_rays * num_lights

    tri_type = eltype(accel)

    primary_ray_queue = similar_soa(img, PrimaryRayWork, num_rays)
    primary_hit_queue = similar_soa(img, PrimaryHitWork{tri_type}, num_rays)
    shadow_ray_queue = similar_soa(img, ShadowRayWork, num_shadow_rays)
    shadow_result_queue = similar_soa(img, ShadowResult, num_shadow_rays)
    reflection_ray_soa = similar_soa(img, ReflectionRayWork, num_rays)
    reflection_hit_soa = similar_soa(img, ReflectionHitWork, num_rays)
    shading_queue = similar_soa(img, ShadedResult, num_rays)
    sample_accumulator = similar(img, Vec3f, num_rays)
    active_count = similar(img, Int32, 1)

    return HikariWavefrontRenderer(
        Int32(width), Int32(height),
        img, accel, ctx,
        camera_pos, camera_lookat, camera_up,
        fov, sky_color, Int32(samples_per_pixel),
        primary_ray_queue, primary_hit_queue,
        shadow_ray_queue, shadow_result_queue,
        reflection_ray_soa, reflection_hit_soa,
        shading_queue, sample_accumulator, active_count
    )
end

"""
    render!(renderer::HikariWavefrontRenderer)

Execute the wavefront path tracing pipeline.
"""
function render!(renderer::HikariWavefrontRenderer)
    width = Int(renderer.width)
    height = Int(renderer.height)
    samples_per_pixel = Int(renderer.samples_per_pixel)

    aspect = Float32(width / height)

    backend = KA.get_backend(renderer.framebuffer)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_lights = Int(length(renderer.ctx.lights))
    num_shadow_rays = num_rays * num_lights

    # Camera basis vectors
    camera_forward = Vec3f(normalize(renderer.camera_lookat - renderer.camera_pos))
    camera_right = Vec3f(normalize(cross(renderer.camera_up, camera_forward)))
    camera_up_ortho = Vec3f(cross(camera_forward, camera_right))

    half_height = tan(deg2rad(renderer.fov / 2))
    half_width = half_height * aspect

    # Stage 1: Generate primary rays
    gen_kernel! = generate_primary_rays_lookat!(backend)
    gen_kernel!(
        renderer.width, renderer.height,
        renderer.camera_pos,
        camera_right, camera_up_ortho, camera_forward,
        half_width, half_height,
        renderer.primary_ray_queue,
        Val(samples_per_pixel),
        ndrange=(height, width)
    )

    # Stage 2: Intersect primary rays
    intersect_kernel! = intersect_primary_rays_hikari!(backend)
    intersect_kernel!(
        renderer.accel,
        renderer.primary_ray_queue,
        renderer.primary_hit_queue,
        ndrange=num_rays
    )

    # Stage 3: Generate shadow rays
    shadow_gen_kernel! = generate_shadow_rays_hikari!(backend)
    shadow_gen_kernel!(
        renderer.primary_hit_queue,
        renderer.ctx,
        renderer.shadow_ray_queue,
        Val(num_lights),
        ndrange=num_rays
    )

    # Stage 4: Test shadow rays
    shadow_test_kernel! = test_shadow_rays_hikari!(backend)
    shadow_test_kernel!(
        renderer.accel,
        renderer.shadow_ray_queue,
        renderer.shadow_result_queue,
        ndrange=num_shadow_rays
    )

    # Stage 5: Shade primary hits
    shade_kernel! = shade_primary_hits_hikari!(backend)
    shade_kernel!(
        renderer.primary_hit_queue,
        renderer.ctx,
        renderer.shadow_result_queue,
        renderer.sky_color,
        renderer.shading_queue,
        Val(num_lights),
        ndrange=num_rays
    )

    # Stage 6: Generate reflection rays
    refl_gen_kernel! = generate_reflection_rays_hikari!(backend)
    refl_gen_kernel!(
        renderer.primary_hit_queue,
        renderer.ctx,
        renderer.reflection_ray_soa,
        renderer.active_count,
        ndrange=num_rays
    )

    # Stage 7: Intersect reflection rays
    refl_intersect_kernel! = intersect_reflection_rays_hikari!(backend)
    refl_intersect_kernel!(
        renderer.accel,
        renderer.reflection_ray_soa,
        renderer.reflection_hit_soa,
        ndrange=num_rays
    )

    # Stage 8: Shade reflections
    refl_shade_kernel! = shade_reflections_and_blend_hikari!(backend)
    refl_shade_kernel!(
        renderer.primary_hit_queue,
        renderer.reflection_hit_soa,
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

# ============================================================================
# Convenience function for creating example scene with Hikari materials
# ============================================================================

"""
    hikari_example_scene()

Create an example scene using Hikari materials that matches the original
wavefront renderer's example_scene.

Returns (material_scene, lights) tuple.
"""
function hikari_example_scene(; glass_cat=false)
    cat_mesh = Makie.loadasset("cat.obj")
    angle = deg2rad(150f0)
    rotation = Makie.Quaternionf(0, sin(angle/2), 0, cos(angle/2))
    rotated_coords = [rotation * Point3f(v) for v in coordinates(cat_mesh)]

    cat_bbox = Rect3f(rotated_coords)
    floor_y = -1.5f0
    cat_offset = Vec3f(0, floor_y - cat_bbox.origin[2], 0)

    cat_mesh = GeometryBasics.normal_mesh(
        [v + cat_offset for v in rotated_coords],
        faces(cat_mesh)
    )

    floor = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)))
    back_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)))
    left_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)))

    sphere1 = Tesselation(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), 64)
    sphere2 = Tesselation(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), 64)

    # Create Hikari materials matching the original scene
    # Original: Material(base_color, metallic, roughness, ior, transmission)
    cat_material = if glass_cat
        # Glass cat: high IOR, transmission
        Hikari.GlassMaterial(
            Kr=RGBSpectrum(0.95f0, 1.0f0, 0.95f0),
            Kt=RGBSpectrum(0.95f0, 1.0f0, 0.95f0),
            u_roughness=0f0,
            v_roughness=0f0,
            index=1.5f0,
            remap_roughness=false
        )
    else
        # Diffuse cat
        Hikari.MatteMaterial(Kd=RGBSpectrum(0.8f0, 0.6f0, 0.4f0), σ=0f0)
    end

    # Floor: diffuse green
    floor_material = Hikari.MatteMaterial(Kd=RGBSpectrum(0.3f0, 0.5f0, 0.3f0), σ=0f0)

    # Back wall: metallic with roughness (original: metallic=0.8, roughness=0.05)
    # Using MetalMaterial with reflectance for color tinting
    back_wall_material = Hikari.MetalMaterial(
        reflectance=(0.8f0, 0.6f0, 0.5f0),
        roughness=0.05f0,
        remap_roughness=false
    )

    # Left wall: diffuse
    left_wall_material = Hikari.MatteMaterial(Kd=RGBSpectrum(0.7f0, 0.7f0, 0.8f0), σ=0f0)

    # Sphere 1: metallic silver with slight roughness (original: metallic=0.8, roughness=0.02)
    sphere1_material = Hikari.MetalMaterial(
        reflectance=(0.9f0, 0.9f0, 0.9f0),
        roughness=0.02f0,
        remap_roughness=false
    )

    # Sphere 2: partially metallic blue (using PlasticMaterial for mixed behavior)
    # PlasticMaterial is appropriate here since it's only partially metallic
    sphere2_material = Hikari.PlasticMaterial(
        Kd=RGBSpectrum(0.3f0, 0.6f0, 0.9f0),
        Ks=RGBSpectrum(0.5f0, 0.5f0, 0.5f0),
        roughness=0.3f0,
        remap_roughness=false
    )

    scene_pairs = [
        (cat_mesh, cat_material),
        (floor, floor_material),
        (back_wall, back_wall_material),
        (left_wall, left_wall_material),
        (normal_mesh(sphere1), sphere1_material),
        (normal_mesh(sphere2), sphere2_material),
    ]

    material_scene = Hikari.MaterialScene(scene_pairs)

    # Create Hikari lights matching original
    lights = [
        Hikari.PointLight(Point3f(3, 4, -2), RGBSpectrum(50.0f0 * 1.0f0, 50.0f0 * 0.9f0, 50.0f0 * 0.8f0)),
        Hikari.PointLight(Point3f(-3, 2, 0), RGBSpectrum(20.0f0 * 0.7f0, 20.0f0 * 0.8f0, 20.0f0 * 1.0f0)),
        Hikari.PointLight(Point3f(0, 5, 5), RGBSpectrum(15.0f0, 15.0f0, 15.0f0)),
    ]

    return material_scene, lights
end
