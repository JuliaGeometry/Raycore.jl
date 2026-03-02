module RaycoreMakieExt

using Raycore
using Makie
using GeometryBasics
using Adapt
import Makie: plot, plot!

# ============================================================================
# TLAS → Mesh conversion (plot geometry directly)
# ============================================================================

Makie.plottype(::Raycore.TLAS) = Makie.Mesh
Makie.plottype(::Raycore.TLAS4) = Makie.Mesh

function Makie.convert_arguments(::Type{Makie.Mesh}, tlas::Union{Raycore.TLAS, Raycore.TLAS4})
    vertices = Point3f[]
    faces = GeometryBasics.TriangleFace{Int}[]
    colors = Float32[]
    normals = Vec3f[]

    metadata_to_color = Dict{Any, Float32}()
    next_color_idx = Ref(0f0)

    function get_color_for_metadata(meta)
        get!(metadata_to_color, meta) do
            next_color_idx[] += 1f0
            next_color_idx[]
        end
    end

    for blas in tlas.blas_array
        for prim in blas.primitives
            start_idx = length(vertices)
            color_val = get_color_for_metadata(prim.metadata)
            for (v, n) in zip(prim.vertices, prim.normals)
                push!(vertices, v)
                push!(colors, color_val)
                push!(normals, Vec3f(n))
            end
            push!(faces, GeometryBasics.TriangleFace(start_idx + 1, start_idx + 2, start_idx + 3))
        end
    end
    return (GeometryBasics.Mesh(vertices, faces; normal=normals, color=colors), )
end

# ============================================================================
# RayPlot recipe — visualize rays traced through a TLAS
# ============================================================================

"""
    RayIntersectionResult

Stores rays and their intersection results for visualization.
Created by [`trace_rays`](@ref).
"""
struct RayIntersectionResult
    rays::Vector{Raycore.Ray}
    hits::Vector{Tuple{Bool, Raycore.Triangle, Float32, GeometryBasics.Vec{3,Float32}, UInt32}}
    tlas::Raycore.TLAS
end

"""
    trace_rays(tlas::Raycore.TLAS, rays::AbstractVector{Raycore.Ray})

Trace rays against a TLAS and return a `RayIntersectionResult` for visualization.

# Example
```julia
using Raycore, RayMakie

tlas = TLAS(KA.CPU())
push!(tlas, mesh)
sync!(tlas)

rays = [Raycore.Ray(o=Point3f(0,0,-5), d=Vec3f(0,0,1))]
result = trace_rays(tlas, rays)
plot(result)
```
"""
function Raycore.trace_rays(tlas::Raycore.TLAS, rays::AbstractVector{<:Raycore.AbstractRay})
    static_tlas = Adapt.adapt(tlas.backend, tlas)
    hits = map(rays) do ray
        Raycore.closest_hit(static_tlas, ray)
    end
    RayIntersectionResult(collect(rays), collect(hits), tlas)
end

"""
    plot(result::RayIntersectionResult; kwargs...)

Makie recipe for visualizing ray intersection results.

# Keyword Arguments
- `show_geometry::Bool = true`: Whether to show the TLAS geometry
- `geometry_alpha::Float64 = 0.4`: Transparency for geometry meshes
- `ray_color::Symbol = :green`: Default color for hit rays
- `hit_color::Symbol = :green`: Color for hit point markers
- `miss_color = (:gray, 0.5)`: Color for rays that missed
- `ray_length::Float32 = 15.0f0`: Length to draw rays that miss
- `show_hit_points::Bool = true`: Whether to show markers at hit points
- `hit_markersize::Float64 = 0.1`: Size of hit point markers
- `show_labels::Bool = false`: Whether to show text labels at hit points
"""
@recipe(RayPlot, result) do scene
    Attributes(
        show_geometry = true,
        geometry_alpha = 0.4,
        geometry_colors = Makie.wong_colors(),
        ray_color = :green,
        hit_color = :green,
        miss_color = (:gray, 0.5),
        ray_length = 15.0f0,
        show_hit_points = true,
        hit_markersize = 0.1,
        show_labels = false,
    )
end

Makie.plottype(::RayIntersectionResult) = RayPlot
Makie.preferred_axis_type(::RayPlot) = LScene

function Makie.plot!(plot::RayPlot)
    result = plot[:result][]

    show_geometry = plot[:show_geometry][]
    geometry_alpha = plot[:geometry_alpha][]
    ray_color = plot[:ray_color][]
    hit_color = plot[:hit_color][]
    miss_color = plot[:miss_color][]
    ray_length = plot[:ray_length][]
    show_hit_points = plot[:show_hit_points][]
    hit_markersize = plot[:hit_markersize][]
    show_labels = plot[:show_labels][]

    # Draw geometry
    if show_geometry
        geo_mesh = Makie.convert_arguments(Makie.Mesh, result.tlas)[1]
        mesh!(plot, geo_mesh; alpha=geometry_alpha)
    end

    # Classify rays into hits and misses
    hit_ray_starts = Point3f[]
    hit_ray_directions = Vec3f[]
    hit_ray_colors = []

    miss_ray_starts = Point3f[]
    miss_ray_directions = Vec3f[]

    hit_points_pos = Point3f[]
    hit_labels_pos = Point3f[]
    hit_labels_text = String[]

    for (i, (ray, hit)) in enumerate(zip(result.rays, result.hits))
        hit_found, hit_triangle, distance, bary_coords, instance_id = hit

        if hit_found
            hit_point = sum(bary_coords .* hit_triangle.vertices)

            push!(hit_ray_starts, ray.o)
            push!(hit_ray_directions, hit_point - ray.o)
            push!(hit_ray_colors, ray_color)

            if show_hit_points
                push!(hit_points_pos, hit_point)

                if show_labels
                    push!(hit_labels_pos, hit_point .+ Vec3f(0.2, 0.2, 0.2))
                    push!(hit_labels_text, "Hit $i\nd=$(round(distance, digits=2))")
                end
            end
        else
            push!(miss_ray_starts, ray.o)
            push!(miss_ray_directions, ray.d * ray_length)
        end
    end

    # Draw hit rays
    if !isempty(hit_ray_starts)
        arrows!(plot, hit_ray_starts, hit_ray_directions,
            color=hit_ray_colors, arrowsize=Vec3f(0.1, 0.1, 0.15))
    end

    # Draw miss rays
    if !isempty(miss_ray_starts)
        arrows!(plot, miss_ray_starts, miss_ray_directions,
            color=miss_color, arrowsize=Vec3f(0.1, 0.1, 0.15))
    end

    # Draw hit points
    if show_hit_points && !isempty(hit_points_pos)
        meshscatter!(plot, hit_points_pos, color=hit_color, markersize=hit_markersize)
    end

    # Draw labels
    if show_labels && !isempty(hit_labels_pos)
        text!(plot, hit_labels_pos, text=hit_labels_text, color=hit_color, fontsize=12)
    end

    return plot
end

end # module
