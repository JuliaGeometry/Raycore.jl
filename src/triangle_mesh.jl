struct TriangleMesh{T, VT<:AbstractVector{Point{3,T}}, IT<:AbstractVector{UInt32}, NT<:AbstractVector{Normal{3,T}}, TT<:AbstractVector{Vec{3,T}}, UT<:AbstractVector{Point{2,T}}} <: AbstractShape
    vertices::VT
    # For the i-th triangle, its 3 vertex positions are:
    # [vertices[indices[3 * i + j]] for j in 0:2].
    indices::IT
    # Optional normal vectors, one per vertex.
    normals::NT
    # Optional tangent vectors, one per vertex.
    tangents::TT
    # Optional parametric (u, v) values, one for each vertex.
    uv::UT

    function TriangleMesh(
            vertices::VT,
            indices::IT,
            normals::NT,
            tangents::TT,
            uv::UT,
        ) where {T, VT<:AbstractVector{Point{3,T}}, IT, NT, TT, UT}

        return new{T, VT, IT, NT, TT, UT}(
            vertices,
            copy(indices), copy(normals),
            copy(tangents), copy(uv),
        )
    end
end

# Convenience constructors with defaults
function TriangleMesh(vertices::AbstractVector{Point{3,T}}, indices::AbstractVector{UInt32},
        normals::AbstractVector{Normal{3,T}} = Normal{3,T}[],
        tangents::AbstractVector{Vec{3,T}} = Vec{3,T}[],
        uv::AbstractVector{Point{2,T}} = Point{2,T}[]) where {T}
    TriangleMesh(vertices, indices, normals, tangents, uv)
end

struct Triangle{T} <: AbstractShape
    vertices::SVector{3,Point{3,T}}
    normals::SVector{3,Normal{3,T}}
    tangents::SVector{3,Vec{3,T}}
    uv::SVector{3,Point{2,T}}
    material_idx::UInt32
    primitive_idx::UInt32
end

Triangle(tri::Triangle; material_idx=tri.material_idx, primitive_idx=tri.primitive_idx) =
    Triangle(tri.vertices, tri.normals, tri.tangents, tri.uv, material_idx, primitive_idx)

function Triangle(m::TriangleMesh{T}, face_indx, material_idx=0, primidx=0) where {T}
    f_idx = 1 + (3 * (face_indx - 1))
    vs = @SVector [m.vertices[m.indices[f_idx + i]] for i in 0:2]
    ns = @SVector [m.normals[m.indices[f_idx + i]] for i in 0:2] # Every mesh should have normals!?
    if !isempty(m.tangents)
        ts = @SVector [m.tangents[m.indices[f_idx + i]] for i in 0:2]
    else
        ts = @SVector [Vec{3,T}(T(NaN)) for _ in 1:3]
    end
    if !isempty(m.uv)
        uv = @SVector [m.uv[m.indices[f_idx + i]] for i in 0:2]
    else
        uv = SVector(Point{2,T}(0), Point{2,T}(1, 0), Point{2,T}(1, 1))
    end
    return Triangle{T}(vs, ns, ts, uv, material_idx, primidx)
end

function TriangleMesh(mesh::GeometryBasics.Mesh)
    nmesh = GeometryBasics.expand_faceviews(mesh)
    fs = decompose(TriangleFace{UInt32}, nmesh)
    vertices = decompose(Point3f, nmesh)
    normals = Normal3f.(decompose_normals(nmesh))
    uvs = GeometryBasics.decompose_uv(nmesh)
    if isnothing(uvs)
        uvs = Point2f[]
    end
    indices = collect(reinterpret(UInt32, fs))
    return TriangleMesh(
        vertices, indices,
        normals, Vec3f[], Point2f.(uvs),
    )
end

function area(t::Triangle{T}) where {T}
    vs = vertices(t)
    T(0.5) * norm((vs[2] - vs[1]) × (vs[3] - vs[1]))
end

function is_degenerate(vs::AbstractVector{Point{3,T}})::Bool where {T}
    v = (vs[3] - vs[1]) × (vs[2] - vs[1])
    (v ⋅ v) ≈ zero(T)
end

vertices(t::Triangle) = t.vertices
normals(t::Triangle) = t.normals
tangents(t::Triangle) = t.tangents
uvs(t::Triangle) = t.uv

@inline function _edge_function(vs::SVector{3,Point{3,T}}) where {T}
    @_inbounds Point{3,T}(
        vs[2][1] * vs[3][2] - vs[2][2] * vs[3][1],
        vs[3][1] * vs[1][2] - vs[3][2] * vs[1][1],
        vs[1][1] * vs[2][2] - vs[1][2] * vs[2][1],
    )
end

object_bound(t::Triangle) = mapreduce(
    v -> Bounds3((v)),
    ∪, vertices(t)
)

world_bound(t::Triangle) = reduce(∪, Bounds3.(vertices(t)))

function _argmax(vec::Vec{3,T}) where {T}
    max_val = vec[1]
    max_idx = Int32(1)
    Base.Cartesian.@nexprs 3 i -> begin
        if vec[i] > max_val
            max_val = vec[i]
            max_idx = Int32(i)
        end
    end
    return max_idx
end

@inline function _to_ray_coordinate_space(
        vertices::AbstractVector{Point{3,T}}, ray::AbstractRay,
    ) where {T}
    # Compute permutation.
    kz = _argmax(map(abs, ray.d))
    kx = kz + Int32(1)
    kx == Int32(4) && (kx = Int32(1))
    ky = kx + Int32(1)
    ky == Int32(4) && (ky = Int32(1))
    permutation = Vec3(kx, ky, kz)
    # Permute ray direction.
    d = map(x-> ray.d[x], permutation)
    # Compute shear.
    denom = one(T) / d[3]
    shear = Point{3,T}(-d[1] * denom, -d[2] * denom, denom)
    # Translate, apply permutation and shear to vertices.
    rkz = ray.o[kz]
    tvs = ntuple(3) do i
        v = vertices[i]
        vo = map(x-> (v-ray.o)[x], permutation)
        return vo + Point{3,T}(
            shear[1] * (v[kz] - rkz),
            shear[2] * (v[kz] - rkz),
            zero(T),
        )
    end
    return SVector{3, Point{3,T}}(tvs), shear
end

@inline function partial_derivatives(
        ::Triangle{T}, vs::AbstractVector{Point{3,T}}, uv::AbstractVector{Point{2,T}},
    )::Tuple{Vec{3,T},Vec{3,T},Vec{3,T},Vec{3,T}} where {T}

    # Compute deltas for partial derivative matrix.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec{3,T}(vs[1] - vs[3]), Vec{3,T}(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ zero(T)
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        _, ∂p∂u, ∂p∂v = coordinate_system(Vec{3,T}(v))
        return ∂p∂u, ∂p∂v, δp_13, δp_23
    end
    inv_det = one(T) / det
    ∂p∂u = Vec{3,T}(δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec{3,T}(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    ∂p∂u, ∂p∂v, δp_13, δp_23
end

@inline function _all(f, x::StaticVector{3})
    f(x[1]) && f(x[2]) && f(x[3])
end

@inline function normal_derivatives(
        t::Triangle{T}, uv::AbstractVector{Point{2,T}},
    )::Tuple{Normal{3,T},Normal{3,T}} where {T}
    t_normals = normals(t)
    _all(x -> _all(isnan, x), t_normals) && return Normal{3,T}(0), Normal{3,T}(0)
    # Compute deltas for partial detivatives of normal.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δn_13, δn_23 = t_normals[1] - t_normals[3], t_normals[2] - t_normals[3]
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    det ≈ zero(T) && return Normal{3,T}(0), Normal{3,T}(0)

    inv_det = one(T) / det
    ∂n∂u = (δuv_23[2] * δn_13 - δuv_13[2] * δn_23) * inv_det
    ∂n∂v = (-δuv_23[1] * δn_13 + δuv_13[1] * δn_23) * inv_det
    ∂n∂u, ∂n∂v
end

# Note: surface_interaction and init_triangle_shading_geometry have been removed
# These functions are now handled by Trace.jl's triangle_to_surface_interaction
# Raycore only provides low-level ray-triangle intersection via intersect_triangle

@inline function intersect(triangle::Triangle{T}, ray::AbstractRay)::Tuple{Bool,T,Point{3,T}} where {T}
    verts = vertices(triangle)  # Get triangle vertices
    return intersect_triangle(verts, ray)  # Check if ray hits triangle
end

@inline function intersect_p(t::Triangle{T}, ray::Union{Ray{T},RayDifferentials{T}}, ::Bool=false) where {T}
    intersect_triangle(t.vertices, ray)[1]
end

@inline function intersect_triangle(
        vs::SVector{3, Point{3,T}}, ray::Union{Ray{T},RayDifferentials{T}}
    ) where {T}
    barycentric = Point{3,T}(0)
    t_hit = zero(T)
    is_degenerate(vs) && return false, t_hit, barycentric
    t_vs, shear = _to_ray_coordinate_space(vs, ray)
    # Compute edge function coefficients.
    edges = _edge_function(t_vs)
    if iszero(edges)
        return false, t_hit, barycentric
    end
    # Perform triangle edge & determinant tests.
    # Point is inside a triangle if all edges have the same sign.
    any(edges .< zero(T)) && any(edges .> zero(T)) && return false, t_hit, barycentric

    det = sum(edges)
    det ≈ zero(T) && return false, t_hit, barycentric
    # Compute scaled hit distance to triangle.
    shear_z = shear[3]
    t_scaled = (
        edges[1] * t_vs[1][3] * shear_z
        + edges[2] * t_vs[2][3] * shear_z
        + edges[3] * t_vs[3][3] * shear_z
    )
    # Test against t_max range.
    det < zero(T) && (t_scaled >= zero(T) || t_scaled < ray.t_max * det) && return false, t_hit, barycentric
    det > zero(T) && (t_scaled <= zero(T) || t_scaled > ray.t_max * det) && return false, t_hit, barycentric
    # TODO test against alpha texture if present.
    # Compute barycentric coordinates and t value for triangle intersection.
    inv_det = one(T) / det
    barycentric = edges .* inv_det
    t_hit = t_scaled * inv_det
    return true, t_hit, barycentric
end
