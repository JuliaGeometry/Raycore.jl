struct TriangleMesh{VT<:AbstractVector{Point3f}, IT<:AbstractVector{UInt32}, NT<:AbstractVector{Normal3f}, TT<:AbstractVector{Vec3f}, UT<:AbstractVector{Point2f}} <: AbstractShape
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
            normals::NT = Normal3f[],
            tangents::TT = Vec3f[],
            uv::UT = Point2f[],
        ) where {VT, IT, NT, TT, UT}

        return new{VT, IT, NT, TT, UT}(
            vertices,
            copy(indices), copy(normals),
            copy(tangents), copy(uv),
        )
    end
end

function TriangleMesh(
        object_to_world::Transformation,
        indices::Vector{UInt32},
        vertices::Vector{Point3f},
        normals::Vector{Normal3f} = Normal3f[],
        tangents::Vector{Vec3f} = Vec3f[],
        uv::Vector{Point2f} = Point2f[],
    )
    vertices = object_to_world.(vertices)
    return TriangleMesh(
        vertices,
        copy(indices), copy(normals),
        copy(tangents), copy(uv),
    )
end

function TriangleMesh(ArrType, mesh::TriangleMesh)
    TriangleMesh(
        ArrType(mesh.vertices),
        ArrType(mesh.indices),
        ArrType(mesh.normals),
        ArrType(mesh.tangents),
        ArrType(mesh.uv),
    )
end

struct Triangle <: AbstractShape
    vertices::SVector{3,Point3f}
    normals::SVector{3,Normal3f}
    tangents::SVector{3,Vec3f}
    uv::SVector{3,Point2f}
    material_idx::UInt32
end

function Triangle(m::TriangleMesh, face_indx, material_idx=0)
    f_idx = 1 + (3 * (face_indx - 1))
    vs = @SVector [m.vertices[m.indices[f_idx + i]] for i in 0:2]
    ns = @SVector [m.normals[m.indices[f_idx + i]] for i in 0:2] # Every mesh should have normals!?
    if !isempty(m.tangents)
        ts = @SVector [m.tangents[m.indices[f_idx + i]] for i in 0:2]
    else
        ts = @SVector [Vec3f(NaN) for _ in 1:3]
    end
    if !isempty(m.uv)
        uv = @SVector [m.uv[m.indices[f_idx + i]] for i in 0:2]
    else
        uv = SVector(Point2f(0), Point2f(1, 0), Point2f(1, 1))
    end
    return Triangle(vs, ns, ts, uv, material_idx)
end

function create_triangle_mesh(
        core::ShapeCore,
        indices::Vector{UInt32},
        vertices::Vector{Point3f},
        normals::Vector{Normal3f} = Normal3f[],
        tangents::Vector{Vec3f} = Vec3f[],
        uv::Vector{Point2f} = Point2f[],
    )
    return TriangleMesh(
        core.object_to_world, indices, vertices,
        normals, tangents, uv,
    )
end

function create_triangle_mesh(mesh::GeometryBasics.Mesh, core::ShapeCore=ShapeCore())
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
        core.object_to_world, indices, vertices,
        normals, Vec3f[], Point2f.(uvs),
    )
end

function area(t::Triangle)
    vs = vertices(t)
    0.5f0 * norm((vs[2] - vs[1]) × (vs[3] - vs[1]))
end

function is_degenerate(vs::AbstractVector{Point3f})::Bool
    v = (vs[3] - vs[1]) × (vs[2] - vs[1])
    (v ⋅ v) ≈ 0f0
end

vertices(t::Triangle) = t.vertices
normals(t::Triangle) = t.normals
tangents(t::Triangle) = t.tangents
uvs(t::Triangle) = t.uv

@inline function _edge_function(vs)
    @_inbounds Point3f(
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

function _argmax(vec::Vec3)
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
        vertices::AbstractVector{Point3f}, ray::AbstractRay,
    )
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
    denom = 1f0 / d[3]
    shear = Point3f(-d[1] * denom, -d[2] * denom, denom)
    # Translate, apply permutation and shear to vertices.
    rkz = ray.o[kz]
    tvs = ntuple(3) do i
        v = vertices[i]
        vo = map(x-> (v-ray.o)[x], permutation)
        return vo + Point3f(
            shear[1] * (v[kz] - rkz),
            shear[2] * (v[kz] - rkz),
            0.0f0,
        )
    end
    return SVector{3, Point3f}(tvs), shear
end

@inline function partial_derivatives(
        ::Triangle, vs::AbstractVector{Point3f}, uv::AbstractVector{Point2f},
    )::Tuple{Vec3f,Vec3f,Vec3f,Vec3f}

    # Compute deltas for partial derivative matrix.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec3f(vs[1] - vs[3]), Vec3f(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ 0f0
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        _, ∂p∂u, ∂p∂v = coordinate_system(Vec3f(v))
        return ∂p∂u, ∂p∂v, δp_13, δp_23
    end
    inv_det = 1f0 / det
    ∂p∂u = Vec3f(δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec3f(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    ∂p∂u, ∂p∂v, δp_13, δp_23
end

@inline function _all(f, x::StaticVector{3})
    f(x[1]) && f(x[2]) && f(x[3])
end


@inline function normal_derivatives(
        t::Triangle, uv::AbstractVector{Point2f},
    )::Tuple{Normal3f,Normal3f}
    t_normals = normals(t)
    _all(x -> _all(isnan, x), t_normals) && return Normal3f(0), Normal3f(0)
    # Compute deltas for partial detivatives of normal.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δn_13, δn_23 = t_normals[1] - t_normals[3], t_normals[2] - t_normals[3]
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    det ≈ 0f0 && return Normal3f(0), Normal3f(0)

    inv_det = 1f0 / det
    ∂n∂u = (δuv_23[2] * δn_13 - δuv_13[2] * δn_23) * inv_det
    ∂n∂v = (-δuv_23[1] * δn_13 + δuv_13[1] * δn_23) * inv_det
    ∂n∂u, ∂n∂v
end


# Note: surface_interaction and init_triangle_shading_geometry have been removed
# These functions are now handled by Trace.jl's triangle_to_surface_interaction
# RayCaster only provides low-level ray-triangle intersection via intersect_triangle

@inline function intersect(triangle::Triangle, ray::AbstractRay)::Tuple{Bool,Float32,Point3f}
    verts = vertices(triangle)  # Get triangle vertices
    return intersect_triangle(verts, ray)  # Check if ray hits triangle
end

@inline function intersect_p(t::Triangle, ray::Union{Ray,RayDifferentials}, ::Bool=false)
    intersect_triangle(t.vertices, ray)[1]
end

@inline function intersect_triangle(
        vs::SVector{3, Point3f}, ray::Union{Ray,RayDifferentials}
    )
    barycentric = Point3f(0)
    t_hit = 0f0
    is_degenerate(vs) && return false, t_hit, barycentric
    t_vs, shear = _to_ray_coordinate_space(vs, ray)
    # Compute edge function coefficients.
    edges = _edge_function(t_vs)
    if iszero(edges)
        return false, t_hit, barycentric
    end
    # Perform triangle edge & determinant tests.
    # Point is inside a triangle if all edges have the same sign.
    any(edges .< 0.0f0) && any(edges .> 0.0f0) && return false, t_hit, barycentric

    det = sum(edges)
    det ≈ 0f0 && return false, t_hit, barycentric
    # Compute scaled hit distance to triangle.
    shear_z = shear[3]
    t_scaled = (
        edges[1] * t_vs[1][3] * shear_z
        + edges[2] * t_vs[2][3] * shear_z
        + edges[3] * t_vs[3][3] * shear_z
    )
    # Test against t_max range.
    det < 0f0 && (t_scaled >= 0f0 || t_scaled < ray.t_max * det) && return false, t_hit, barycentric
    det > 0f0 && (t_scaled <= 0f0 || t_scaled > ray.t_max * det) && return false, t_hit, barycentric
    # TODO test against alpha texture if present.
    # Compute barycentric coordinates and t value for triangle intersection.
    inv_det = 1.0f0 / det
    barycentric = edges .* inv_det
    t_hit = t_scaled * inv_det
    return true, t_hit, barycentric
end
