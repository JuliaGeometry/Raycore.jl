"""
    RayIntersectionSession{F}

Represents a ray tracing session containing rays, a BVH structure, a hit function,
and the computed intersection results.

# Fields
- `rays::Vector{<:AbstractRay}`: Array of rays to trace
- `bvh::BVHAccel`: BVH acceleration structure to intersect against
- `hit_function::F`: Function to use for intersection testing (e.g., `closest_hit` or `any_hit`)
- `hits::Vector{Tuple{Bool, Triangle, Float32, Point3f}}`: Results of hit_function applied to each ray

# Example
```julia
using RayCaster, GeometryBasics

# Create BVH from geometry
sphere = Tesselation(Sphere(Point3f(0, 0, 1), 1.0f0), 20)
bvh = RayCaster.BVHAccel([sphere])

# Create rays
rays = [
    RayCaster.Ray(Point3f(0, 0, -5), Vec3f(0, 0, 1)),
    RayCaster.Ray(Point3f(1, 0, -5), Vec3f(0, 0, 1)),
]

# Create session
session = RayIntersectionSession(rays, bvh, RayCaster.closest_hit)

# Access results
for (i, hit) in enumerate(session.hits)
    hit_found, primitive, distance, bary_coords = hit
    if hit_found
        println("Ray \$i hit at distance \$distance")
    end
end
```
"""
struct RayIntersectionSession{F}
    rays::Vector{<:AbstractRay}
    bvh::BVHAccel
    hit_function::F
    hits::Vector{Tuple{Bool, Triangle, Float32, Point3f}}

    function RayIntersectionSession(rays::Vector{<:AbstractRay}, bvh::BVHAccel, hit_function::F) where {F}
        # Compute all hits
        hits = [hit_function(bvh, ray) for ray in rays]
        new{F}(rays, bvh, hit_function, hits)
    end
end

"""
    hit_points(session::RayIntersectionSession)

Extract all valid hit points from a RayIntersectionSession.

Returns a vector of `Point3f` containing the world-space hit points for all rays that intersected geometry.
"""
function hit_points(session::RayIntersectionSession)
    points = Point3f[]
    for (ray, hit) in zip(session.rays, session.hits)
        hit_found, hit_primitive, distance, bary_coords = hit
        if hit_found
            hit_point = sum_mul(bary_coords, hit_primitive.vertices)
            push!(points, hit_point)
        end
    end
    return points
end

"""
    hit_distances(session::RayIntersectionSession)

Extract all hit distances from a RayIntersectionSession.

Returns a vector of `Float32` containing distances for all rays that intersected geometry.
"""
function hit_distances(session::RayIntersectionSession)
    distances = Float32[]
    for hit in session.hits
        hit_found, _, distance, _ = hit
        if hit_found
            push!(distances, distance)
        end
    end
    return distances
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
