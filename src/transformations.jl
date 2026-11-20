struct Transformation
    m::Mat4f
    inv_m::Mat4f
end

Transformation() = Transformation(Mat4f(I), Mat4f(I))
Transformation(m::Mat4f) = Transformation(m, inv(m))
is_identity(t::Transformation) = t.m == I && t.inv_m == I
function Base.transpose(t::Transformation)
    Transformation(transpose(t.m), transpose(t.inv_m))
end
Base.inv(t::Transformation) = Transformation(t.inv_m, t.m)

function Base.:(==)(t1::Transformation, t2::Transformation)
    all(t1.m .== t2.m) && all(t1.inv_m .== t2.inv_m)
end
function Base.:≈(t1::Transformation, t2::Transformation)
    all(t1.m .≈ t2.m) && all(t1.inv_m .≈ t2.inv_m)
end
function Base.:*(t1::Transformation, t2::Transformation)
    Transformation(t1.m * t2.m, t2.inv_m * t1.inv_m)
end

function translate(δ::Vec3f)
    m = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        δ[1], δ[2], δ[3], 1,
    )
    m_inv = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -δ[1], -δ[2], -δ[3], 1,
    )
    Transformation(m, m_inv)
end

function scale(x::Number, y::Number, z::Number)

    m = Mat4f(
        x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1,
    )
    m_inv = Mat4f(
        1 / x, 0, 0, 0,
        0, 1 / y, 0, 0,
        0, 0, 1 / z, 0,
        0, 0, 0, 1,
    )
    Transformation(m, m_inv)
end

"""
    rotation_matrix(axis::Vec3f, s::Float32, c::Float32) -> Mat3f

The rotation about the unit vector `axis` by the angle whose sine is `s` and cosine `c`
(Rodrigues' formula), written column by column: `Mat3f(...)` fills column-major.
"""
@inline function rotation_matrix(axis::Vec3f, s::Float32, c::Float32)::Mat3f
    a = axis
    t = 1f0 - c
    Mat3f(
        t * a[1] * a[1] + c,        t * a[1] * a[2] + s * a[3], t * a[1] * a[3] - s * a[2],  # column 1
        t * a[1] * a[2] - s * a[3], t * a[2] * a[2] + c,        t * a[2] * a[3] + s * a[1],  # column 2
        t * a[1] * a[3] + s * a[2], t * a[2] * a[3] - s * a[1], t * a[3] * a[3] + c,         # column 3
    )
end

"""
    rotation_matrix(θ, axis) -> Mat3f

The rotation by `θ` degrees about `axis`, counter-clockwise looking down the axis, like
pbrt's `Rotate`.
"""
rotation_matrix(θ::Real, axis::Vec3f) = rotation_matrix(normalize(axis), sincosd(Float32(θ))...)

"""
    rotate(θ, axis) -> Transformation

The rotation by `θ` degrees about `axis`, see [`rotation_matrix`](@ref).
"""
function rotate(θ::Real, axis::Vec3f)
    r = rotation_matrix(θ, axis)
    m = Mat4f(vcat(hcat(r, zero(Vec3f)), SMatrix{1,4,Float32}(0, 0, 0, 1)))
    Transformation(m, transpose(m))
end

function look_at(position::Point3f, target::Point3f, up::Vec3f)
    zaxis = normalize(position - target)
    xaxis = normalize(cross(up, zaxis))
    yaxis = normalize(cross(zaxis, xaxis))
    T0, T1 = 0.0f0, 1.0f0
    m = Mat4f(
        xaxis[1], yaxis[1], zaxis[1], T0,
        xaxis[2], yaxis[2], zaxis[2], T0,
        xaxis[3], yaxis[3], zaxis[3], T0,
        T0, T0, T0, T1
    )
    Transformation(m, transpose(m)) * translate(Vec3f(-position))
end


function perspective(fov::Float32, near::Float32, far::Float32)
    # Projective divide for perspective projection.
    p = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -(far + near) / (far - near), -1f0,
        0, 0, (-2f0 * near * far) / (far - near), 0,
    )
    # Scale canonical perspective view to specified field of view.

    inv_tan = 1f0 / tan(deg2rad(fov) / 2f0)
    scale(inv_tan, inv_tan, 1f0) * Transformation(p)
end

function (t::Transformation)(p::Point3f)::Point3f
    ph = Point4f(p..., 1f0)
    pt = t.m * ph
    # Always divide by w - avoids branch that causes SPIR-V structured control flow errors
    # Division by 1.0 is essentially free on GPU
    Point3f(pt[Vec(1, 2, 3)]) ./ pt[4]
end

(t::Transformation)(v::Vec3f)::Vec3f = t.m[Vec(1, 2, 3), Vec(1, 2, 3)] * v
(t::Transformation)(n::Normal3f)::Normal3f = transpose(t.inv_m[Vec(1, 2, 3), Vec(1, 2, 3)]) * n

function (t::Transformation)(b::Bounds3)
    mapreduce(i -> Bounds3(t(corner(b, i))), ∪, 1:8)
end

function apply(t::Transformation, r::Ray)
    return Ray(r, o=t(r.o), d=t(r.d))
end

function apply(t::Transformation, r::RayDifferentials)
    return RayDifferentials(r;
        o=t(r.o), d=t(r.d),
        rx_origin=t(r.rx_origin),
        ry_origin=t(r.ry_origin),
        rx_direction=t(r.rx_direction),
        ry_direction=t(r.ry_direction)
    )
end

function has_scale(t::Transformation)

    a = norm(t(Vec3f(1, 0, 0)))
    b = norm(t(Vec3f(0, 1, 0)))
    c = norm(t(Vec3f(0, 0, 1)))
    a ≉ 1 || b ≉ 1 || c ≉ 1
end

function swaps_handedness(t::Transformation)
    det(t.m[Vec(1, 2, 3), Vec(1, 2, 3)]) < 0
end
