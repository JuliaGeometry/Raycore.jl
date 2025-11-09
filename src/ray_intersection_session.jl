"""
    RayIntersectionSession{F}

Represents a ray tracing session containing rays, a BVH structure, a hit function,
and the computed intersection results.

# Fields
- `rays::Vector{<:AbstractRay}`: Array of rays to trace
- `bvh::BVH`: BVH acceleration structure to intersect against
- `hit_function::F`: Function to use for intersection testing (e.g., `closest_hit` or `any_hit`)
- `hits::Vector{Tuple{Bool, Triangle, Float32, Point3f}}`: Results of hit_function applied to each ray

# Example
```julia
using Raycore, GeometryBasics

# Create BVH from geometry
sphere = Tesselation(Sphere(Point3f(0, 0, 1), 1.0f0), 20)
bvh = Raycore.BVH([sphere])

# Create rays
rays = [
    Raycore.Ray(Point3f(0, 0, -5), Vec3f(0, 0, 1)),
    Raycore.Ray(Point3f(1, 0, -5), Vec3f(0, 0, 1)),
]

# Create session
session = RayIntersectionSession(rays, bvh, Raycore.closest_hit)

# Access results
for (i, hit) in enumerate(session.hits)
    hit_found, primitive, distance, bary_coords = hit
    if hit_found
        println("Ray \$i hit at distance \$distance")
    end
end
```
"""
struct RayIntersectionSession{Rays, F}
    hit_function::F
    rays::Rays
    bvh::BVH
    hits::Vector{Tuple{Bool, Triangle, Float32, Point3f}}

    function RayIntersectionSession(hit_function::F, rays::Rays, bvh::BVH) where {Rays,F}
        # Compute all hits
        hits = [hit_function(bvh, ray) for ray in rays]
        new{Rays, F}(hit_function, rays, bvh, hits)
    end
end

"""
    hit_points(session::RayIntersectionSession)

Extract all valid hit points from a RayIntersectionSession.

Returns a vector of `Point3f` containing the world-space hit points for all rays that intersected geometry.
"""
function hit_points(session::RayIntersectionSession)
    return map(filter(first, session.hits)) do hit
        _, hit_primitive, _, bary_coords = hit
        return sum_mul(bary_coords, hit_primitive.vertices)
    end
end

"""
    hit_distances(session::RayIntersectionSession)

Extract all hit distances from a RayIntersectionSession.

Returns a vector of `Float32` containing distances for all rays that intersected geometry.
"""
function hit_distances(session::RayIntersectionSession)
    return map(filter(first, session.hits)) do hit
        return hit[3]
    end
end

"""
    hit_count(session::RayIntersectionSession)

Count the number of rays that hit geometry in the session.
"""
function hit_count(session::RayIntersectionSession)
    count(hit -> hit[1], session.hits)
end

"""
    miss_count(session::RayIntersectionSession)

Count the number of rays that missed all geometry in the session.
"""
function miss_count(session::RayIntersectionSession)
    count(hit -> !hit[1], session.hits)
end
