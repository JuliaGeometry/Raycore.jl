module RaycoreMakieExt

using Raycore
using Makie
using GeometryBasics
import Makie: plot, plot!

"""
    plot(session::RayIntersectionSession; kwargs...)

Makie recipe for visualizing a RayIntersectionSession.

# Keyword Arguments
- `show_bvh::Bool = true`: Whether to show the BVH geometry
- `bvh_alpha::Float64 = 0.4`: Transparency for BVH meshes
- `bvh_colors = [:red, :yellow, :blue]`: Colors to cycle through for different meshes
- `ray_colors = nothing`: Colors for rays. If `nothing`, uses a gradient based on hit distance
- `ray_color::Symbol = :black`: Default color for all rays if `ray_colors` is `nothing`
- `hit_color::Symbol = :green`: Color for hit point markers
- `miss_color::Symbol = :gray`: Color for rays that missed
- `ray_length::Float32 = 15.0f0`: Length to draw rays that miss
- `show_hit_points::Bool = true`: Whether to show markers at hit points
- `hit_markersize::Float64 = 0.2`: Size of hit point markers
- `show_labels::Bool = false`: Whether to show text labels at hit points
- `axis = nothing`: Optional axis to draw on (if not provided, creates new figure)

# Example
```julia
using Raycore, GeometryBasics, GLMakie

# Create geometry
sphere1 = Tesselation(Sphere(Point3f(0, 0, 1), 1.0f0), 20)
sphere2 = Tesselation(Sphere(Point3f(0, 0, 3), 1.0f0), 20)
bvh = Raycore.BVH([sphere1, sphere2])

# Create rays
rays = [
    Raycore.Ray(Point3f(0, 0, -5), Vec3f(0, 0, 1)),
    Raycore.Ray(Point3f(1, 0, -5), Vec3f(0, 0, 1)),
]

# Create and visualize session
session = RayIntersectionSession(rays, bvh, Raycore.closest_hit)
plot(session)
```
"""
@recipe(RayPlot, session) do scene
    Attributes(
        show_bvh = true,
        bvh_alpha = 1.0,
        bvh_colors = Makie.wong_colors(),
        ray_colors = nothing,
        ray_color = :green,
        hit_color = :green,
        miss_color = (:gray, 0.5),
        ray_length = 15.0f0,
        show_hit_points = true,
        hit_markersize = 0.1,
        show_labels = false,
    )
end

Makie.plottype(::Raycore.RayIntersectionSession) = RayPlot
Makie.preferred_axis_type(::RayPlot) = LScene

function Makie.plot!(plot::RayPlot)
    session = plot[:session][]

    # Extract attributes
    show_bvh = plot[:show_bvh][]
    bvh_alpha = plot[:bvh_alpha][]
    bvh_colors = plot[:bvh_colors][]
    ray_colors = plot[:ray_colors][]
    ray_color = plot[:ray_color][]
    hit_color = plot[:hit_color][]
    miss_color = plot[:miss_color][]
    ray_length = plot[:ray_length][]
    show_hit_points = plot[:show_hit_points][]
    hit_markersize = plot[:hit_markersize][]
    show_labels = plot[:show_labels][]

    # Draw BVH if requested
    if show_bvh
        draw_bvh!(plot, session.bvh, bvh_colors, bvh_alpha)
    end

    # Determine ray colors if not provided
    if isnothing(ray_colors)
        # Use single color for all rays
        ray_colors = fill(ray_color, length(session.rays))
    end

    # Collect all data for batch rendering
    hit_ray_starts = Point3f[]
    hit_ray_directions = Vec3f[]
    hit_ray_colors = []

    miss_ray_starts = Point3f[]
    miss_ray_directions = Vec3f[]

    hit_points_pos = Point3f[]
    hit_labels_pos = Point3f[]
    hit_labels_text = String[]

    for (i, (ray, hit)) in enumerate(zip(session.rays, session.hits))
        hit_found, hit_primitive, distance, bary_coords = hit

        # Get color for this ray
        color = i <= length(ray_colors) ? ray_colors[i] : ray_color

        if hit_found
            # Calculate hit point
            hit_point = Raycore.sum_mul(bary_coords, hit_primitive.vertices)

            # Collect ray data
            push!(hit_ray_starts, ray.o)
            push!(hit_ray_directions, hit_point - ray.o)
            push!(hit_ray_colors, color)

            # Collect hit point data
            if show_hit_points
                push!(hit_points_pos, hit_point)

                # Collect label data
                if show_labels
                    push!(hit_labels_pos, hit_point .+ Vec3f(0.2, 0.2, 0.2))
                    push!(hit_labels_text, "Hit $i\nd=$(round(distance, digits=2))")
                end
            end
        else
            # Ray missed - collect miss ray data
            push!(miss_ray_starts, ray.o)
            push!(miss_ray_directions, ray.d * ray_length)
        end
    end

    # Draw all hit rays in one call
    if !isempty(hit_ray_starts)
        arrows3d!(
            plot,
            hit_ray_starts,
            hit_ray_directions,
            color = hit_ray_colors,
            markerscale = 0.3
        )
    end

    # Draw all miss rays in one call
    if !isempty(miss_ray_starts)
        arrows3d!(
            plot,
            miss_ray_starts,
            miss_ray_directions,
            color = miss_color,
            markerscale = 0.3
        )
    end

    # Draw all hit points in one call
    if show_hit_points && !isempty(hit_points_pos)
        meshscatter!(
            plot,
            hit_points_pos,
            color = hit_color,
            markersize = hit_markersize
        )
    end

    # Draw all labels in one call
    if show_labels && !isempty(hit_labels_pos)
        text!(
            plot,
            hit_labels_pos,
            text = hit_labels_text,
            color = hit_color,
            fontsize = 12
        )
    end

    return plot
end

"""
Helper function to draw BVH geometry
"""
function draw_bvh!(plot, bvh::Raycore.BVH, colors, alpha)
    # Group primitives by their material_idx
    primitive_groups = Dict{UInt32, Vector{Raycore.Triangle}}()
    for prim in bvh.primitives
        mat_idx = prim.material_idx
        if !haskey(primitive_groups, mat_idx)
            primitive_groups[mat_idx] = Raycore.Triangle[]
        end
        push!(primitive_groups[mat_idx], prim)
    end

    # Draw each group with a different color
    color_idx = 1
    for (mat_idx, prims) in primitive_groups
        # Get all triangles for this mesh
        vertices = Point3f[]
        faces = GeometryBasics.TriangleFace{Int}[]

        for (i, prim) in enumerate(prims)
            # Add vertices
            start_idx = length(vertices)
            for v in prim.vertices
                push!(vertices, v)
            end
            # Add face (using 1-based indexing)
            push!(faces, GeometryBasics.TriangleFace(start_idx + 1, start_idx + 2, start_idx + 3))
        end

        # Create mesh from vertices and faces
        mesh_obj = GeometryBasics.normal_mesh(vertices, faces)

        # Pick color
        color = colors[mod1(color_idx, length(colors))]
        color_idx += 1

        # Draw mesh
        mesh!(
            plot,
            mesh_obj,
            color = (color, alpha),
            transparency=alpha < 1.0
        )
    end
end

Makie.plottype(::Raycore.BVH) = Makie.Mesh

function Makie.convert_arguments(::Type{Makie.Mesh}, bvh::Raycore.BVH)
    # Convert BVH to a Mesh for plotting
    vertices = Point3f[]
    faces = GeometryBasics.TriangleFace{Int}[]
    colors = Float32[]
    normals = Vec3f[]
    for (i, prim) in enumerate(bvh.primitives)
        start_idx = length(vertices)
        for (v, n) in zip(prim.vertices, prim.normals)
            push!(vertices, v)
            push!(colors, prim.material_idx)
            push!(normals, Vec3f(n))
        end
        push!(faces, GeometryBasics.TriangleFace(start_idx + 1, start_idx + 2, start_idx + 3))
    end
    return (GeometryBasics.Mesh(vertices, faces; normal=normals, color=colors), )
end

end # module
