struct Ray{T} <: AbstractRay
    o::Point{3,T}
    d::Vec{3,T}
    t_max::T
    time::T
end

# Constructor that infers T from arguments
function Ray(; o::Point{3,T}, d::Vec{3,T}, t_max::T = T(Inf), time::T = zero(T)) where {T}
    Ray{T}(o, d, t_max, time)
end

@inline function Ray(ray::Ray{T}; o::Point{3,T} = ray.o, d::Vec{3,T} = ray.d, t_max::T = ray.t_max, time::T = ray.time) where {T}
    Ray{T}(o, d, t_max, time)
end


struct RayDifferentials{T} <: AbstractRay
    o::Point{3,T}
    d::Vec{3,T}
    t_max::T
    time::T

    has_differentials::Bool
    rx_origin::Point{3,T}
    ry_origin::Point{3,T}
    rx_direction::Vec{3,T}
    ry_direction::Vec{3,T}
end

# Constructor that infers T from arguments
function RayDifferentials(; o::Point{3,T}, d::Vec{3,T}, t_max = T(Inf), time = zero(T),
        has_differentials::Bool = false, rx_origin = zeros(Point{3,T}),
        ry_origin = zeros(Point{3,T}), rx_direction = zeros(Vec{3,T}),
        ry_direction = zeros(Vec{3,T})) where {T}
    RayDifferentials{T}(o, d, t_max, time, has_differentials, rx_origin, ry_origin, rx_direction, ry_direction)
end

@inline function RayDifferentials(ray::RayDifferentials{T};
        o::Point{3,T} = ray.o, d::Vec{3,T} = ray.d, t_max::T = ray.t_max, time::T = ray.time,
        has_differentials::Bool = ray.has_differentials, rx_origin::Point{3,T} = ray.rx_origin, ry_origin::Point{3,T} = ray.ry_origin,
        rx_direction::Vec{3,T} = ray.rx_direction, ry_direction::Vec{3,T} = ray.ry_direction
    ) where {T}
    RayDifferentials{T}(o, d, t_max, time, has_differentials, rx_origin, ry_origin, rx_direction, ry_direction)
end

@inline function RayDifferentials(r::Ray{T})::RayDifferentials{T} where {T}
    RayDifferentials(o = r.o, d = r.d, t_max = r.t_max, time = r.time)
end

@inline function set_direction(r::Ray{T}, d::Vec{3,T}) where {T}
    d = map(i-> i ≈ zero(T) ? zero(T) : i, d)
    return Ray(r, d=d)
end

@inline function set_direction(r::RayDifferentials{T}, d::Vec{3,T}) where {T}
    d = map(i -> i ≈ zero(T) ? zero(T) : i, d)
    return RayDifferentials(r, d=d)
end

@inline check_direction(r::AbstractRay) = set_direction(r, r.d)

apply(r::AbstractRay, t::Number) = r.o + r.d * t

@inline function scale_differentials(rd::RayDifferentials{T}, s::T) where {T}
    return RayDifferentials(rd;
        rx_origin = rd.o + (rd.rx_origin - rd.o) * s,
        ry_origin = rd.o + (rd.ry_origin - rd.o) * s,
        rx_direction = rd.d + (rd.rx_direction - rd.d) * s,
        ry_direction = rd.d + (rd.ry_direction - rd.d) * s
    )
end

increase_hit(ray::Ray{T}, t_hit) where {T} = Ray(ray; t_max=t_hit)
increase_hit(ray::RayDifferentials{T}, t_hit) where {T} = RayDifferentials(ray; t_max=t_hit)

@inline function intersect_p!(shape::AbstractShape, ray::R) where {R<:AbstractRay}
    intersects, t_hit, barycentric = intersect(shape, ray)
    !intersects && return false, ray, barycentric
    ray = increase_hit(ray, t_hit)
    return true, ray, barycentric
end
