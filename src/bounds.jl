struct Bounds2{T}
    p_min::Point{2,T}
    p_max::Point{2,T}
end

struct Bounds3{T}
    p_min::Point{3,T}
    p_max::Point{3,T}
end

# By default -- create bounds in invalid configuraiton.
Bounds2{T}() where {T} = Bounds2(Point{2,T}(T(Inf)), Point{2,T}(T(-Inf)))
Bounds3{T}() where {T} = Bounds3(Point{3,T}(T(Inf)), Point{3,T}(T(-Inf)))
Bounds2(p::Point{2,T}) where {T} = Bounds2{T}(p, p)
Bounds3(p::Point{3,T}) where {T} = Bounds3{T}(p, p)
Bounds2c(p1::Point{2,T}, p2::Point{2,T}) where {T} = Bounds2{T}(min.(p1, p2), max.(p1, p2))
Bounds3c(p1::Point{3,T}, p2::Point{3,T}) where {T} = Bounds3{T}(min.(p1, p2), max.(p1, p2))

function Base.:(==)(b1::Union{Bounds2,Bounds3}, b2::Union{Bounds2,Bounds3})
    b1.p_min == b2.p_min && b1.p_max == b2.p_max
end
function Base.:≈(b1::Union{Bounds2,Bounds3}, b2::Union{Bounds2,Bounds3})
    b1.p_min ≈ b2.p_min && b1.p_max ≈ b2.p_max
end

function Base.getindex(b::Union{Bounds2{T}, Bounds3{T}}, i::I) where {T,I<:Integer}
    i === I(1) && return b.p_min
    i === I(2) && return b.p_max
    N = b isa Bounds2 ? 2 : 3
    return Point{N, T}(T(NaN))
end

function is_valid(b::Bounds3{T})::Bool where {T}
    all(b.p_min .!= T(Inf)) && all(b.p_max .!= T(-Inf))
end

function Base.length(b::Bounds2{T})::Int64 where {T}
    δ = ceil.(b.p_max .- b.p_min .+ one(T))
    UInt32(δ[1] * δ[2])
end

function Base.iterate(
        b::Bounds2{T}, i::Integer = Int32(1),
    )::Union{Nothing,Tuple{Point{2,T},Int32}} where {T}
    i > length(b) && return nothing

    j = i - Int32(1)
    δ = b.p_max .- b.p_min .+ one(T)
    b.p_min .+ Point{2,T}(j % δ[1], j ÷ δ[1]), i + Int32(1)
end

# Index through 8 corners.
function corner(b::Bounds3{T}, c::Integer) where {T}
    c -= Int32(1)
    x = (c & Int32(1)) == Int32(0) ? b.p_min[1] : b.p_max[1]
    y = (c & Int32(2)) == Int32(0) ? b.p_min[2] : b.p_max[2]
    z = (c & Int32(4)) == Int32(0) ? b.p_min[3] : b.p_max[3]
    Point{3,T}(x, y, z)
end

function Base.union(b1::B, b2::B) where B<:Union{Bounds2,Bounds3}
    B(min.(b1.p_min, b2.p_min), max.(b1.p_max, b2.p_max))
end

function Base.intersect(b1::B, b2::B) where B<:Union{Bounds2,Bounds3}
    B(max.(b1.p_min, b2.p_min), min.(b1.p_max, b2.p_max))
end

function overlaps(b1::Bounds3{T}, b2::Bounds3{T}) where {T}
    all(b1.p_max .>= b2.p_min) && all(b1.p_min .<= b2.p_max)
end

function inside(b::Bounds3{T}, p::Point{3,T}) where {T}
    all(p .>= b.p_min) && all(p .<= b.p_max)
end

function inside_exclusive(b::Bounds3{T}, p::Point{3,T}) where {T}
    all(p .>= b.p_min) && all(p .< b.p_max)
end

expand(b::Bounds3{T}, δ::T) where {T} = Bounds3{T}(b.p_min .- δ, b.p_max .+ δ)
diagonal(b::Union{Bounds2,Bounds3}) = b.p_max - b.p_min

function surface_area(b::Bounds3)
    d = diagonal(b)
    2 * (d[1] * d[2] + d[1] * d[3] + d[2] * d[3])
end

function area(b::Bounds2{T}) where {T}
    δ = b.p_max .- b.p_min
    δ[1] * δ[2]
end

@inline function sides(b::Union{Bounds2,Bounds3})
    return map(b.p_max, b.p_min) do b1, b0
        return abs(b1 - b0)
    end
end

@inline function inclusive_sides(b::Union{Bounds2{T},Bounds3{T}}) where {T}
    return map(b.p_max, b.p_min) do b1, b0
        abs(b1 - (b0 - one(T)))
    end
end

function volume(b::Bounds3)
    d = diagonal(b)
    d[1] * d[2] * d[3]
end

"""
Return index of the longest axis.
Useful for deciding which axis to subdivide,
when building ray-tracing acceleration structures.

1 - x, 2 - y, 3 - z.
"""
function maximum_extent(b::Bounds3)
    d = diagonal(b)
    if d[1] > d[2] && d[1] > d[3]
        return 1
    elseif d[2] > d[3]
        return 2
    end
    return 3
end

lerp(v1::T, v2::T, t::T) where {T} = (one(T) - t) * v1 + t * v2
lerp(p0::Point{3,T}, p1::Point{3,T}, t::T) where {T} = (one(T) - t) .* p0 .+ t .* p1
# Linearly interpolate point between the corners of the bounds.
lerp(b::Bounds3{T}, p::Point{3,T}) where {T} = lerp.(p, b.p_min, b.p_max)

distance(p1::Point{3,T}, p2::Point{3,T}) where {T} = norm(p1 - p2)
function distance_squared(p1::Point{3,T}, p2::Point{3,T}) where {T}
    p = p1 - p2
    p ⋅ p
end

"""Get offset of a point from the minimum point of the bounds."""
function offset(b::Bounds3{T}, p::Point{3,T}) where {T}
    o = p - b.p_min
    g = b.p_max .> b.p_min
    !any(g) && return o
    Point{3,T}(
        o[1] / (g[1] ? b.p_max[1] - b.p_min[1] : one(T)),
        o[2] / (g[2] ? b.p_max[2] - b.p_min[2] : one(T)),
        o[3] / (g[3] ? b.p_max[3] - b.p_min[3] : one(T)),
    )
end

function bounding_sphere(b::Bounds3{T})::Tuple{Point{3,T},T} where {T}
    center = (b.p_min + b.p_max) / T(2)
    radius = inside(b, center) ? distance(center, b.p_max) : zero(T)
    center, radius
end

function intersect(b::Bounds3{T}, ray::AbstractRay)::Tuple{Bool,T,T} where {T}
    t0, t1 = zero(T), ray.t_max
    @_inbounds for i in 1:3
        # Update interval for i-th bbox slab.
        inv_ray_dir = one(T) / ray.d[i]
        t_near = (b.p_min[i] - ray.o[i]) * inv_ray_dir
        t_far = (b.p_max[i] - ray.o[i]) * inv_ray_dir
        if t_near > t_far
            t_near, t_far = t_far, t_near
        end

        t0 = t_near > t0 ? t_near : t0
        t1 = t_far < t1 ? t_far : t1
        t0 > t1 && return false, zero(T), zero(T)
    end
    true, t0, t1
end

@inline function is_dir_negative(dir::Vec{3,T}) where {T}
    @_inbounds Point3{UInt8}(
        dir[1] < zero(T) ? 2 : 1,
        dir[2] < zero(T) ? 2 : 1,
        dir[3] < zero(T) ? 2 : 1,
    )
end

"""
dir_is_negative: 1 -- false, 2 -- true
"""
@inline function intersect_p(
        b::Bounds3{T}, ray::AbstractRay,
        inv_dir::Vec{3,T}, dir_is_negative::Point3{UInt8},
    )::Bool where {T}
    @_inbounds begin
        tx_min = (b[dir_is_negative[1]][1] - ray.o[1]) * inv_dir[1]
        tx_max = (b[3-dir_is_negative[1]][1] - ray.o[1]) * inv_dir[1]
        ty_min = (b[dir_is_negative[2]][2] - ray.o[2]) * inv_dir[2]
        ty_max = (b[3-dir_is_negative[2]][2] - ray.o[2]) * inv_dir[2]

        (tx_min > ty_max || ty_min > tx_max) && return false
        ty_min > tx_min && (tx_min = ty_min)
        ty_max > tx_max && (tx_max = ty_max)

        tz_min = (b[dir_is_negative[3]][3] - ray.o[3]) * inv_dir[3]
        tz_max = (b[3-dir_is_negative[3]][3] - ray.o[3]) * inv_dir[3]
        (tx_min > tz_max || tz_min > tx_max) && return false

        (tz_min > tx_min) && (tx_min = tz_min)
        (tz_max < tx_max) && (tx_max = tz_max)
        return tx_min < ray.t_max && tx_max > zero(T)
    end
end
