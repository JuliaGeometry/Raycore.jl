# ============================================================================
# Hardware-Accelerated Ray Tracing Interface
# ============================================================================
#
# Backend-agnostic types and stubs for hardware RT (Vulkan, Metal, OptiX).
# Concrete backends implement the stubs via package extensions.
#
# Design:
# - HWTLAS{Backend,...} is a concrete type with parametric fields
# - Backend extensions fill in BLASType, AccelHandle, etc.
# - Hikari dispatches on AbstractAccel/AbstractAdaptedAccel without backend imports

# ============================================================================
# Transport types (same memory layout for all backends)
# ============================================================================

"""
    RTRay

Ray input for hardware RT dispatch. 32 bytes, matches Vulkan/Metal layout.
"""
struct RTRay
    origin_x::Float32
    origin_y::Float32
    origin_z::Float32
    tmin::Float32
    dir_x::Float32
    dir_y::Float32
    dir_z::Float32
    tmax::Float32
end

"""
    RTHitResult

Ray hit output from hardware RT dispatch. 32 bytes.
"""
struct RTHitResult
    hit::UInt32
    t::Float32
    primitive_id::UInt32
    instance_custom_index::UInt32
    bary_u::Float32
    bary_v::Float32
    _pad1::UInt32
    _pad2::UInt32
end

# ============================================================================
# HWTLAS - Hardware Top-Level Acceleration Structure
# ============================================================================

"""
    HWTLAS{Backend, BLASType, TriGPU, OffGPU, AccelHandle} <: AbstractAccel

Hardware-accelerated TLAS. Drop-in replacement for `TLAS` using hardware RT.
All fields are parametric - backend extensions (Lava, Metal) fill in concrete types.

On `push!`, calls `build_hw_blas` (extension stub) to build a backend BLAS.
On `sync!`, calls `build_hw_tlas` (extension stub) to build the backend TLAS.
"""
mutable struct HWTLAS{Backend, BLASType, TriGPU, OffGPU, AccelHandle} <: AbstractAccel
    backend::Backend

    # Geometry (accumulated on push!)
    blas_list::Vector{BLASType}
    blas_triangles::Vector{Vector{Any}}
    blas_offsets::Vector{UInt32}

    # Instances
    instance_blas_indices::Vector{Int}
    instance_transforms::Vector{NTuple{12,Float32}}
    instance_custom_indices::Vector{UInt32}

    # Handle management
    handle_to_range::Dict{TLASHandle, UnitRange{Int}}
    deleted_handles::Set{TLASHandle}
    next_handle_id::UInt32

    # Bounding box
    root_aabb::Bounds3

    # Built on sync! (backend-specific)
    hw_tlas::Any
    hw_accel::AccelHandle
    tri_gpu::TriGPU
    off_gpu::OffGPU

    dirty::Bool
end

function HWTLAS(backend)
    HWTLAS{typeof(backend), Any, Any, Any, Any}(
        backend,
        Any[], Vector{Any}[], UInt32[],
        Int[], NTuple{12,Float32}[], UInt32[],
        Dict{TLASHandle, UnitRange{Int}}(), Set{TLASHandle}(), UInt32(1),
        Bounds3(),
        nothing, nothing, nothing, nothing,
        true,
    )
end

# ============================================================================
# HWAdaptedAccel - GPU-adapted form
# ============================================================================

"""
    HWAdaptedAccel{H<:HWTLAS} <: AbstractAdaptedAccel

GPU-adapted form of HWTLAS. Returned by `Adapt.adapt_structure`.
Dispatches ray tracing to hardware implementations.
"""
struct HWAdaptedAccel{H<:HWTLAS} <: AbstractAdaptedAccel
    hwtlas::H
end

Adapt.adapt_structure(to, h::HWTLAS) = HWAdaptedAccel(h)

# ============================================================================
# HWTLAS management (backend-agnostic, calls stubs for backend-specific parts)
# ============================================================================

function _hwtlas_add_geometry!(hwtlas::HWTLAS, mesh::GeometryBasics.Mesh)
    nmesh = GeometryBasics.expand_faceviews(mesh)
    fs = decompose(TriangleFace{UInt32}, nmesh)
    verts = decompose(Point3f, nmesh)
    norms = Normal3f.(decompose_normals(nmesh))
    uvs_raw = GeometryBasics.decompose_uv(nmesh)
    uvs = isnothing(uvs_raw) ? Point2f[] : Point2f.(uvs_raw)
    indices = collect(reinterpret(UInt32, fs))

    has_meta = hasproperty(nmesh, :face_meta)
    n_faces = length(fs)

    cpu_triangles = [begin
            meta = has_meta ? nmesh.face_meta[indices[3*(i-1)+1]] : UInt32(i)
            build_triangle(verts, norms, uvs, indices, i, meta)
        end
        for i in 1:n_faces
        if !is_degenerate_face(verts, indices, i)
    ]
    isempty(cpu_triangles) && error("Geometry has no valid triangles")

    # Extract vertex positions for backend BLAS build
    n_tris = length(cpu_triangles)
    blas_vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        vs = cpu_triangles[i].vertices
        for j in 1:3
            v = vs[j]
            blas_vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end
    blas_indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        blas_indices[i+1] = UInt32(i)
    end

    hw_blas = build_hw_blas(hwtlas.backend, blas_vertices, blas_indices)
    push!(hwtlas.blas_list, hw_blas)
    push!(hwtlas.blas_triangles, cpu_triangles)
    blas_idx = length(hwtlas.blas_list)

    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    for tri in cpu_triangles
        for v in tri.vertices
            hwtlas.root_aabb = union(hwtlas.root_aabb, Bounds3(Point3f(v)))
        end
    end

    hwtlas.dirty = true
    return blas_idx
end

"""
Internal: add N instances of `blas_idx` to the HWTLAS.

`instance_ids` (if given) supplies the per-instance interface override that
the HW closest-hit shader reads via `gl_InstanceCustomIndexEXT`.  When
`nothing`, every instance gets `0` (inherit from triangle metadata).
"""
function _hwtlas_add_instances!(hwtlas::HWTLAS, blas_idx::Int, transforms;
                                instance_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing)
    if instance_ids !== nothing && length(instance_ids) != length(transforms)
        throw(ArgumentError("instance_ids length $(length(instance_ids)) != transforms length $(length(transforms))"))
    end
    start_idx = length(hwtlas.instance_blas_indices) + 1
    for (i, transform) in enumerate(transforms)
        iid = instance_ids === nothing ? UInt32(0) : UInt32(instance_ids[i])
        push!(hwtlas.instance_blas_indices, blas_idx)
        push!(hwtlas.instance_transforms, mat4_to_transform_matrix(transform))
        push!(hwtlas.instance_custom_indices, iid)
    end
    end_idx = length(hwtlas.instance_blas_indices)

    handle = TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    hwtlas.handle_to_range[handle] = start_idx:end_idx
    return handle
end

const Mat4f = SMatrix{4, 4, Float32, 16}

function Base.push!(hwtlas::HWTLAS, mesh::GeometryBasics.Mesh, transform::Mat4f=Mat4f(I);
                    instance_id::UInt32=UInt32(0))
    blas_idx = _hwtlas_add_geometry!(hwtlas, mesh)
    return _hwtlas_add_instances!(hwtlas, blas_idx, (transform,);
                                  instance_ids=UInt32[instance_id])
end

function Base.push!(hwtlas::HWTLAS, mesh::GeometryBasics.Mesh, transforms::AbstractVector{Mat4f};
                    instance_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing)
    blas_idx = _hwtlas_add_geometry!(hwtlas, mesh)
    return _hwtlas_add_instances!(hwtlas, blas_idx, transforms; instance_ids)
end

function Base.delete!(hwtlas::HWTLAS, handle::TLASHandle)::Bool
    haskey(hwtlas.handle_to_range, handle) || return false
    handle in hwtlas.deleted_handles && return false
    push!(hwtlas.deleted_handles, handle)
    hwtlas.dirty = true
    return true
end

function sync!(hwtlas::HWTLAS)
    hwtlas.dirty || return hwtlas

    # Compact: deleted handles leave orphan entries in instance_blas_indices
    # and instance_transforms. Without filtering, each re-push! grows the TLAS
    # forever, wastes VRAM, and eventually triggers GPU faults. Rebuild the
    # instance arrays without the deleted ranges, then rebuild blas_list to
    # drop BLAS that no instance still points to.
    if !isempty(hwtlas.deleted_handles)
        deleted_inst_idx = BitSet()
        for h in hwtlas.deleted_handles
            r = get(hwtlas.handle_to_range, h, nothing)
            r === nothing && continue
            for i in r
                push!(deleted_inst_idx, i)
            end
            delete!(hwtlas.handle_to_range, h)
        end

        keep_inst = [i for i in eachindex(hwtlas.instance_blas_indices) if !(i in deleted_inst_idx)]
        hwtlas.instance_blas_indices = [hwtlas.instance_blas_indices[i] for i in keep_inst]
        hwtlas.instance_transforms = [hwtlas.instance_transforms[i] for i in keep_inst]
        hwtlas.instance_custom_indices = [hwtlas.instance_custom_indices[i] for i in keep_inst]

        # After removing instances, some BLAS may be unreferenced.
        # Rebuild blas_list/blas_triangles/blas_offsets, remapping instance indices.
        still_used = Set(hwtlas.instance_blas_indices)
        if length(still_used) < length(hwtlas.blas_list)
            old_to_new = Dict{Int,Int}()
            new_blas = similar(hwtlas.blas_list, 0)
            new_tris = Vector{Any}[]
            new_offsets = UInt32[]
            running_offset = UInt32(0)
            for (old_idx, blas) in enumerate(hwtlas.blas_list)
                old_idx in still_used || continue
                push!(new_blas, blas)
                push!(new_tris, hwtlas.blas_triangles[old_idx])
                push!(new_offsets, running_offset)
                running_offset += UInt32(length(hwtlas.blas_triangles[old_idx]))
                old_to_new[old_idx] = length(new_blas)
            end
            hwtlas.blas_list = new_blas
            hwtlas.blas_triangles = new_tris
            hwtlas.blas_offsets = new_offsets
            hwtlas.instance_blas_indices = [old_to_new[i] for i in hwtlas.instance_blas_indices]
            # `instance_custom_indices` is the per-instance interface override
            # (scene-level, independent of BLAS index).  No remapping needed
            # when we compact BLASes away.
        end

        # Rebuild handle_to_range against the compacted indices.
        idx_shift = Dict{Int,Int}()
        shift = 0
        sorted_deleted = sort(collect(deleted_inst_idx))
        for d in sorted_deleted
            shift += 1
            idx_shift[d] = shift
        end
        shift_of(i) = begin
            s = 0
            for d in sorted_deleted
                d < i && (s += 1)
            end
            s
        end
        new_range = Dict{TLASHandle,UnitRange{Int}}()
        for (h, r) in hwtlas.handle_to_range
            lo = first(r) - shift_of(first(r))
            hi = last(r) - shift_of(last(r))
            new_range[h] = lo:hi
        end
        hwtlas.handle_to_range = new_range
        empty!(hwtlas.deleted_handles)
    end

    n_inst = length(hwtlas.instance_blas_indices)
    if n_inst == 0
        hwtlas.hw_tlas = nothing
        hwtlas.hw_accel = nothing
        hwtlas.dirty = false
        return hwtlas
    end

    blas_refs = [hwtlas.blas_list[hwtlas.instance_blas_indices[i]] for i in 1:n_inst]

    # Backend builds the TLAS + accel handle
    hw_tlas, hw_accel, tri_gpu, off_gpu = build_hw_tlas(
        hwtlas.backend, blas_refs, hwtlas.blas_triangles, hwtlas.blas_offsets;
        transforms=hwtlas.instance_transforms,
        custom_indices=hwtlas.instance_custom_indices)

    hwtlas.hw_tlas = hw_tlas
    hwtlas.hw_accel = hw_accel
    hwtlas.tri_gpu = tri_gpu
    hwtlas.off_gpu = off_gpu
    hwtlas.dirty = false
    return hwtlas
end

# Accessors
world_bound(hwtlas::HWTLAS) = hwtlas.root_aabb
n_geometries(hwtlas::HWTLAS) = length(hwtlas.blas_list)
n_instances(hwtlas::HWTLAS) = length(hwtlas.instance_blas_indices)
refit_tlas!(hwtlas::HWTLAS) = nothing

# RayMakie compat: tlas.instances -> lightweight view
struct _HWTLASInstances
    n::Int
end
Base.isempty(x::_HWTLASInstances) = x.n == 0
Base.length(x::_HWTLASInstances) = x.n

function Base.getproperty(hwtlas::HWTLAS, s::Symbol)
    s === :instances ? _HWTLASInstances(length(getfield(hwtlas, :instance_blas_indices))) :
                       getfield(hwtlas, s)
end

# ============================================================================
# Extension stubs - backends implement these
# ============================================================================

# Indirect dispatch: backends that can read ndrange from GPU memory
"""    supports_indirect_dispatch(backend) -> Bool"""
supports_indirect_dispatch(backend) = false

"""    indirect_ndrange(size_buf) -> size_buf"""
function indirect_ndrange end

# BLAS/TLAS construction
"""    build_hw_blas(backend, vertices, indices) -> BLASType"""
function build_hw_blas end

"""    build_hw_tlas(backend, blas_refs, blas_triangles, blas_offsets; transforms, custom_indices) -> (hw_tlas, hw_accel, tri_gpu, off_gpu)"""
function build_hw_tlas end

# Batch ray tracing
"""    trace_closest_hits!(results, rays, accel, n)"""
function trace_closest_hits! end

"""    trace_closest_hits_indirect!(results, rays, accel, n_buf)"""
function trace_closest_hits_indirect! end

"""    batch_trace_indirect(results, rays, accel, n_buf) -> AbstractAdaptedAccel"""
function batch_trace_indirect end

# Transform matrix conversion (row-major 3x4 for Vulkan/Metal)
"""    mat4_to_transform_matrix(m::Mat4f) -> NTuple{12,Float32}"""
function mat4_to_transform_matrix end

# Custom shader pipeline
"""    set_custom_anyhit!(accel, anyhit_fn, raygen_fn)"""
function set_custom_anyhit! end

# RT shader intrinsics (called inside GPU shaders, dispatched per backend)
function rt_primitive_id end
function rt_instance_custom_index end
function rt_launch_id_x end
function rt_global_invocation_id_x end
function rt_ignore_intersection end
function rt_terminate_ray end
function rt_payload_store! end
function rt_payload_load end
function rt_trace_ray! end
