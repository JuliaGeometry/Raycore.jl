@inline function concentric_sample_disk(u::Point2f)::Point2f
    # Map uniform random numbers to [-1, 1].
    offset_x = 2f0 * u[1] - 1f0
    offset_y = 2f0 * u[2] - 1f0
    # No branch for the degenerate origin: a tiny epsilon keeps the divisions finite, and the
    # result there is ~0 anyway.
    safe_offset_x = offset_x + 1.0f-10
    safe_offset_y = offset_y + 1.0f-10
    is_x_larger = abs(offset_x) > abs(offset_y)
    r = ifelse(is_x_larger, offset_x, offset_y)
    θ = ifelse(is_x_larger,
               (offset_y / safe_offset_x) * π / 4f0,
               π / 2f0 - (offset_x / safe_offset_y) * π / 4f0)
    return Point2f(r * cos(θ), r * sin(θ))
end

@inline function cosine_sample_hemisphere(u::Point2f)::Vec3f
    d = concentric_sample_disk(u)
    z = √max(0f0, 1f0 - d[1]^2 - d[2]^2)
    Vec3f(d[1], d[2], z)
end

sum_mul(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

"""
    reflect(wo, n) -> wi

`wo` reflected about `n`: `-wo + 2 (wo ⋅ n) n`.
"""
@inline reflect(wo::Vec3f, n::Vec3f)::Vec3f = -wo + 2f0 * dot(wo, n) * n

"""
    coordinate_system(n::Vec3f) -> (tangent, bitangent)

An orthonormal basis around the unit vector `n`.
"""
@inline function coordinate_system(n::Vec3f)
    if abs(n[1]) > abs(n[2])
        inv_len = 1f0 / sqrt(n[1] * n[1] + n[3] * n[3])
        tangent = Vec3f(n[3] * inv_len, 0f0, -n[1] * inv_len)
    else
        inv_len = 1f0 / sqrt(n[2] * n[2] + n[3] * n[3])
        tangent = Vec3f(0f0, n[3] * inv_len, -n[2] * inv_len)
    end
    return (tangent, cross(n, tangent))
end

"""
Flip normal `n` so that it lies in the same hemisphere as `v`.
"""
@inline face_forward(n, v) = (n ⋅ v) < 0 ? -n : n

function random_hemisphere_uniform(n::Vec3{T}, u::Vec3{T}, v::Vec3{T}) where {T}
    # Generate random numbers
    xi1 = rand(T)
    xi2 = rand(T)

    # Use cos(theta) uniformly distributed in [0,1]
    theta = acos(xi1)
    phi = T(2) * T(pi) * xi2

    # Calculate coordinates in the local frame (hemisphere sampling)
    x_local = sin(theta) * cos(phi)
    y_local = sin(theta) * sin(phi)
    z_local = cos(theta)

    # Transform back to the original coordinate system
    return u .* x_local .+ v .* y_local .+ n .* z_local
end

function random_triangle_point(triangle)
    # Assume triangle.vertices is an array of 3 Point3f
    vertices = triangle.vertices
    p1, p2, p3 = vertices[1], vertices[2], vertices[3]

    # Generate two random numbers in [0,1]
    r1 = rand(Float32)
    r2 = rand(Float32)

    # Map to barycentric coordinates
    sqrt_r1 = sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2
    # Compute the point using barycentrics
    return Point3f(u * p1 + v * p2 + w * p3)
end
