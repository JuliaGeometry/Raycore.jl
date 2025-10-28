struct RayHit
    hit::Bool
    point::Point3f
    prim_idx::UInt32
end

function hits_from_grid(bvh, viewdir; grid_size=32)
    # Calculate grid bounds
    ray_direction = normalize(viewdir)
    ray_origins = Raycore.generate_ray_grid(bvh, ray_direction, grid_size)
    result = similar(ray_origins, RayHit)
    Threads.@threads for idx in CartesianIndices(ray_origins)
        o = ray_origins[idx]
        ray = Raycore.Ray(; o=o, d=ray_direction)
        hit, prim, dist, bary = Raycore.closest_hit(bvh, ray)
        hitpoint = sum_mul(bary, prim.vertices)
        @inbounds result[idx] = RayHit(hit, hitpoint, prim.material_idx)
    end
    return result
end

function view_factors(bvh; rays_per_triangle=10000)
    result = zeros(UInt32, length(bvh.primitives), length(bvh.primitives))
    return view_factors!(result, bvh, rays_per_triangle)
end

function view_factors!(result, bvh, rays_per_triangle=10000)
    Threads.@threads for idx in eachindex(bvh.primitives)
        @inbounds begin
            triangle = bvh.primitives[idx]
            n = GB.orthogonal_vector(Vec3f, GB.Triangle(triangle.vertices...))
            normal = normalize(Vec3f(n))
            u, v = get_orthogonal_basis(normal)
            for i in 1:rays_per_triangle
                point_on_triangle = random_triangle_point(triangle)
                o = point_on_triangle .+ (normal .* 0.01f0) # Offset so it doesn't self intersect
                ray = Ray(; o=o, d=random_hemisphere_uniform(normal, u, v))
                hit, prim, dist, _ = closest_hit(bvh, ray)
                if hit && prim.material_idx != triangle.material_idx
                    # weigh by angle?
                    result[triangle.material_idx, prim.material_idx] += 1
                end
            end
        end
    end
    return result
end


function get_centroid(bvh, viewdir; grid_size=32)
    # Calculate grid bounds
    hits = hits_from_grid(bvh, viewdir; grid_size=grid_size)
    surface_points = [hit.point for hit in hits if hit.hit]
    return surface_points, mean(surface_points)
end


function get_illumination(bvh, viewdir; grid_size=1000)
    # Calculate grid bounds
    hits = hits_from_grid(bvh, viewdir; grid_size=grid_size)
    result = Dict{UInt32, Float32}()
    for hit in hits
        if hit.hit
            if haskey(result, hit.prim_idx)
                result[hit.prim_idx] += 1f0
            else
                result[hit.prim_idx] = 1f0
            end
        end
    end
    return [get(result, UInt32(idx), 0.0f0) for idx in 1:length(bvh.primitives)]
end
