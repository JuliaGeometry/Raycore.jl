struct RayHit{TMetadata}
    hit::Bool
    point::Point3f
    metadata::TMetadata
end

# Access all triangles from a StaticTLAS
_primitives(tlas::StaticTLAS) = tlas.all_blas_prims

function generate_ray_grid(tlas, ray_direction::Vec3f, grid_size::Int)
    direction = normalize(ray_direction)
    bounds = world_bound(tlas)
    rect = GB.Rect3f(Point3f(bounds.p_min), Point3f(bounds.p_max) - Point3f(bounds.p_min))
    corners = GB.decompose(Point3f, rect)

    # Create perpendicular basis for the grid plane
    if abs(direction[1]) < 0.9f0
        temp = Vec3f(1.0f0, 0.0f0, 0.0f0)
    else
        temp = Vec3f(0.0f0, 1.0f0, 0.0f0)
    end
    basis1 = normalize(cross(direction, temp))
    basis2 = normalize(cross(direction, basis1))

    # Project corners onto basis vectors
    proj1 = [dot(Vec3f(c...), basis1) for c in corners]
    proj2 = [dot(Vec3f(c...), basis2) for c in corners]

    min_proj1, max_proj1 = extrema(proj1)
    min_proj2, max_proj2 = extrema(proj2)

    margin = 0.05f0 * max(max_proj1 - min_proj1, max_proj2 - min_proj2)
    grid_width = max_proj1 - min_proj1 + 2 * margin
    grid_height = max_proj2 - min_proj2 + 2 * margin

    # Place grid origin behind the scene
    depth_proj = [dot(Vec3f(c...), direction) for c in corners]
    min_depth = minimum(depth_proj) - margin

    grid_center = Point3f(0, 0, 0) + min_depth * direction +
                  ((min_proj1 + max_proj1) / 2) * basis1 +
                  ((min_proj2 + max_proj2) / 2) * basis2

    cell_w = grid_width / grid_size
    cell_h = grid_height / grid_size

    ray_origins = Matrix{Point3f}(undef, grid_size, grid_size)
    for i in 1:grid_size
        for j in 1:grid_size
            u = (i - (grid_size + 1) / 2) * cell_w
            v = (j - (grid_size + 1) / 2) * cell_h
            ray_origins[i, j] = grid_center + u * basis1 + v * basis2
        end
    end
    return ray_origins
end

function hits_from_grid(tlas, viewdir; grid_size=32)
    ray_direction = normalize(viewdir)
    ray_origins = generate_ray_grid(tlas, ray_direction, grid_size)
    prims = _primitives(tlas)
    TMetadata = eltype(prims).parameters[1]
    result = similar(ray_origins, RayHit{TMetadata})
    Threads.@threads for idx in CartesianIndices(ray_origins)
        o = ray_origins[idx]
        ray = Ray(; o=o, d=ray_direction)
        hit, prim, dist, bary, _ = closest_hit(tlas, ray)
        hitpoint = sum_mul(bary, prim.vertices)
        @inbounds result[idx] = RayHit{TMetadata}(hit, hitpoint, prim.metadata)
    end
    return result
end

function view_factors(tlas; rays_per_triangle=10000)
    prims = _primitives(tlas)
    result = zeros(UInt32, length(prims), length(prims))
    return view_factors!(result, tlas, rays_per_triangle)
end

function view_factors!(result, tlas, rays_per_triangle=10000)
    prims = _primitives(tlas)
    Threads.@threads for idx in eachindex(prims)
        @inbounds begin
            triangle = prims[idx]
            tri_idx = triangle.metadata
            n = GB.orthogonal_vector(Vec3f, GB.Triangle(triangle.vertices...))
            normal = normalize(n)
            u, v = get_orthogonal_basis(normal)
            for i in 1:rays_per_triangle
                point_on_triangle = random_triangle_point(triangle)
                o = point_on_triangle .+ (normal .* 0.01f0)
                ray = Ray(; o=o, d=random_hemisphere_uniform(normal, u, v))
                hit, hit_prim, dist, _, _ = closest_hit(tlas, ray)
                if hit
                    hit_idx = hit_prim.metadata
                    if hit_idx != tri_idx
                        result[tri_idx, hit_idx] += UInt32(1)
                    end
                end
            end
        end
    end
    return result
end

function get_centroid(tlas, viewdir; grid_size=32)
    hits = hits_from_grid(tlas, viewdir; grid_size=grid_size)
    surface_points = [hit.point for hit in hits if hit.hit]
    return surface_points, mean(surface_points)
end

function get_illumination(tlas, viewdir; grid_size=1000)
    hits = hits_from_grid(tlas, viewdir; grid_size=grid_size)
    prims = _primitives(tlas)
    result = Dict{Int, Float32}()
    for hit in hits
        if hit.hit
            idx = Int(hit.metadata)
            count = get!(result, idx, 0f0)
            result[idx] = count + 1f0
        end
    end
    return [get(result, idx, 0.0f0) for idx in 1:length(prims)]
end
