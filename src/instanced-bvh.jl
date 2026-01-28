# ==============================================================================
# Instanced BVH - Two-Level Acceleration Structure
# ==============================================================================
#
# Based on AMD RadeonRays SDK architecture:
# - BLAS (Bottom-Level Acceleration Structure): BVH over triangle geometry
# - TLAS (Top-Level Acceleration Structure): BVH over instances with transforms
# - Two-level traversal with transform handling
#
# Key optimizations:
# - LBVH (Linear BVH) construction using 30-bit Morton codes
# - Parametrized array types for CPU/GPU compatibility
# - Fully type-stable traversal kernels
# - Compact memory layout for cache efficiency
#
# Architecture (GPU-First):
# - BLASes are built directly on the backend (GPU arrays from the start)
# - TLAS stores a Vector of backend BLASes (CPU vector for management, but arrays inside are on GPU)
# - During sync!/adapt, isbits device pointers are extracted for kernel traversal
# - CPU-side dictionaries provide O(1) instance lookup
# - No CPU→GPU copy needed for BLAS data during sync!

using StaticArrays
using LinearAlgebra: I
import KernelAbstractions as KA
import AcceleratedKernels as AK

# ==============================================================================
# Core Data Structures
# ==============================================================================

"""
    BVHNode2

Compact BVH node for two-child binary trees.
Stores AABBs for both children inline (BVH2IL layout from RadeonRays).

Fields:
- `aabb0_min`, `aabb0_max`: Child 0 bounding box (or vertex data for leaves)
- `aabb1_min`, `aabb1_max`: Child 1 bounding box (or unused for leaves)
- `child0`: Child 0 index (INVALID_NODE for leaves)
- `child1`: Child 1 index (or primitive index for leaves)
- `parent`: Parent node index
"""
struct BVHNode2
    # Child 0 AABB (or triangle vertex 0 for leaves)
    aabb0_min::Point3f
    aabb0_max::Point3f

    # Child 1 AABB (or triangle vertices 1,2 for leaves)
    aabb1_min::Point3f
    aabb1_max::Point3f

    # Topology
    child0::UInt32  # INVALID_NODE (0xFFFFFFFF) indicates leaf
    child1::UInt32  # Primitive index for leaves, child index for interior
    parent::UInt32
end

const INVALID_NODE = 0xffffffff

"""Check if a node is a leaf node."""
@inline is_leaf(node::BVHNode2) = node.child0 == INVALID_NODE

"""Check if a node is an interior node."""
@inline is_interior(node::BVHNode2) = node.child0 != INVALID_NODE

"""
    InstanceDescriptor

Describes an instance of a bottom-level BVH in world space.

Fields:
- `blas_index`: Index into BLAS array
- `instance_id`: User-defined instance ID
- `transform`: World-to-local transformation matrix (4x4)
- `inv_transform`: Local-to-world transformation matrix (4x4)
- `flags`: Instance flags (reserved for future use)
"""
struct InstanceDescriptor
    blas_index::UInt32          # Which BLAS to instance
    instance_id::UInt32         # User-provided ID
    transform::Mat4f            # Local-to-world
    inv_transform::Mat4f        # World-to-local
    flags::UInt32               # Reserved
end

"""
    BLAS{NodeArray, TriArray}

Bottom-Level Acceleration Structure - BVH over triangle geometry.

Type parameters allow CPU (Vector) or GPU (CuArray, ROCArray) storage.
"""
struct BLAS{
    NodeArray <: AbstractVector{BVHNode2},
    TriArray <: AbstractVector{<:Triangle}
}
    nodes::NodeArray
    primitives::TriArray
    root_aabb::Bounds3
end

# ==============================================================================
# StaticTLAS - Immutable structure for kernel traversal
# ==============================================================================

"""
    StaticTLAS{NodeArray, InstArray, BLASArray}

Immutable Top-Level Acceleration Structure for GPU kernel traversal.
This is what `Adapt.adapt_structure` returns from a TLAS.

The struct is immutable and contains only the arrays needed for ray traversal.
No management state (dictionaries, free lists, etc.) - those stay on CPU in TLAS.
"""
struct StaticTLAS{
    NodeArray <: AbstractVector{BVHNode2},
    InstArray <: AbstractVector{InstanceDescriptor},
    BLASArray <: AbstractVector{<:BLAS}
}
    nodes::NodeArray
    instances::InstArray
    blas_array::BLASArray
    root_aabb::Bounds3
end

# ==============================================================================
# TLAS - Mutable structure with backend arrays + CPU management
# ==============================================================================

"""
    TLASHandle

Stable handle for referencing instances in a TLAS.
Simple unique ID for O(1) lookup in handle_to_range dictionary.
"""
struct TLASHandle
    id::UInt32
end

# Sentinel for invalid handle
const INVALID_HANDLE = TLASHandle(UInt32(0))

"""
    TLAS{Backend}

Mutable Top-Level Acceleration Structure with direct GPU arrays.

GPU-first design: instances are appended directly to GPU array using efficient
GPU append. CPU-side dictionary provides O(1) handle lookups.

`Adapt.adapt_structure` returns a `StaticTLAS` wrapping the backend arrays for
kernel traversal.

# Type Parameters
- `Backend`: KernelAbstractions backend (CPU(), OpenCLBackend(), CUDABackend(), etc.)

# Fields
- `backend`: KernelAbstractions backend for kernels
- `nodes`: BVH nodes array (on backend, rebuilt on sync!)
- `instances`: Instance descriptors array (GPU array, direct append)
- `blas_array`: BLAS objects array (GPU array with isbits pointers)
- `root_aabb`: World-space bounding box
- `handle_to_range`: Handle -> range in instances array (CPU-side)
- `deleted_handles`: Handles deleted but not yet compacted (CPU-side)
- `gpu_blas_arrays`: Keeps GPU arrays alive for isbits pointers
- `dirty`: Whether BVH topology needs rebuild

# Usage
```julia
tlas = TLAS(CPU())
h1 = push!(tlas, geometry)
h2 = push!(tlas, Instance(geometry, transforms))
update_transform!(tlas, h2, new_transform)
delete!(tlas, h1)
sync!(tlas)  # Rebuild BVH structure
static = adapt(backend, tlas)  # Get StaticTLAS for kernels
```
"""
mutable struct TLAS{Backend}
    backend::Backend

    # Backend arrays for kernel traversal (GPU from start)
    nodes::Any       # AbstractVector{BVHNode2} - rebuilt on sync!
    instances::Any   # AbstractVector{InstanceDescriptor} - direct GPU append
    blas_array::Any  # Backend array of isbits BLASes

    root_aabb::Bounds3

    # CPU-side management (dictionaries must stay on CPU for O(1) lookup)
    handle_to_range::Dict{TLASHandle, UnitRange{Int}}
    deleted_handles::Set{TLASHandle}

    # Backend arrays kept alive for GC (isbits pointers in blas_array reference these)
    gpu_blas_arrays::Vector{Any}

    # Whether BVH topology needs rebuild
    dirty::Bool

    # Counters
    next_handle_id::UInt32
    next_instance_id::UInt32
end

# Legacy alias for compatibility
const TLASBuilder = TLAS

# Note: _get_isbits_ptr is defined in heterovec.jl and reused here

# ==============================================================================
# Instance - High-level wrapper for instanced geometry
# ==============================================================================

"""
    Instance{G, T, M}

User-friendly wrapper for instanced geometry.

Each Instance represents one or more copies of a geometry with different transforms.
Used with TLAS construction and push! operations.

# Type Parameters
- `G`: Geometry type (TriangleMesh, or anything convertible via `to_triangle_mesh`)
- `T`: Transform array type (AbstractVector{Mat4f})
- `M`: Metadata array type (typically AbstractVector{UInt32})

# Fields
- `geometry`: The mesh/geometry to instance
- `transforms`: Array of local-to-world transforms (one per instance)
- `metadata`: Array of per-instance user data (typically instance IDs)
"""
struct Instance{G, T<:AbstractVector{Mat4f}, M<:AbstractVector}
    geometry::G
    transforms::T
    metadata::M
end

# Convenience constructors
"""Single instance with transform and metadata."""
Instance(geom, transform::Mat4f, metadata) =
    Instance(geom, [transform], [metadata])

"""Single instance at identity transform."""
Instance(geom; metadata=UInt32(1)) =
    Instance(geom, [Mat4f(I)], [metadata])

"""Multiple instances with same metadata."""
function Instance(geom, transforms::AbstractVector{Mat4f}; metadata=UInt32(1))
    Instance(geom, transforms, fill(metadata, length(transforms)))
end

# ------------------------------------------------------------------------------
# TLAS Constructor and Core Operations
# ------------------------------------------------------------------------------

"""
    TLAS(backend) -> TLAS

Create an empty TLAS for the given backend.
Use `push!` to add geometries/instances, then `sync!` to rebuild the BVH.
`Adapt.adapt_structure` returns a StaticTLAS for kernel traversal.

# Example
```julia
tlas = TLAS(OpenCLBackend())
h1 = push!(tlas, geometry)
h2 = push!(tlas, Instance(geometry, transforms))
sync!(tlas)  # Rebuild BVH on backend
static = adapt(backend, tlas)  # StaticTLAS with isbits pointers for kernels
```
"""
function TLAS(backend)
    # GPU-first design: all arrays on backend from the start
    tlas = TLAS(
        backend,
        KA.allocate(backend, BVHNode2, 0),           # nodes (empty, rebuilt on sync!)
        KA.allocate(backend, InstanceDescriptor, 0), # instances (direct GPU append)
        _allocate_empty_blas_array(backend),         # blas_array (GPU array of isbits BLASes)
        Bounds3(),                                   # root_aabb
        Dict{TLASHandle, UnitRange{Int}}(),         # handle_to_range
        Set{TLASHandle}(),                          # deleted_handles
        Any[],                                       # gpu_blas_arrays (GC roots)
        true,                                        # dirty
        UInt32(1),                                   # next_handle_id
        UInt32(1)                                    # next_instance_id
    )

    # Register finalizer to free GPU memory when TLAS is garbage collected
    finalizer(_free_tlas_gpu_memory!, tlas)

    return tlas
end

"""Free all GPU memory held by a TLAS. Called by finalizer."""
function _free_tlas_gpu_memory!(tlas::TLAS)
    # Free main backend arrays
    _try_unsafe_free!(tlas.nodes)
    _try_unsafe_free!(tlas.instances)
    _try_unsafe_free!(tlas.blas_array)

    # Free all backing arrays for isbits BLASes
    for arr in tlas.gpu_blas_arrays
        _try_unsafe_free!(arr)
    end
    empty!(tlas.gpu_blas_arrays)

    return nothing
end

"""Try to free a GPU array if it supports unsafe_free!."""
function _try_unsafe_free!(arr)
    # GPU arrays (CLArray, CuArray) typically have unsafe_free! from GPUArrays
    # Use duck typing to check if the method exists
    try
        # Try to call the method if it exists on the array's module
        M = parentmodule(typeof(arr))
        if isdefined(M, :unsafe_free!) && applicable(getfield(M, :unsafe_free!), arr)
            getfield(M, :unsafe_free!)(arr)
        end
    catch
        # Silently ignore - GC will handle cleanup
    end
    return nothing
end

"""Helper to create initial empty BLAS array placeholder."""
function _allocate_empty_blas_array(_backend)
    # Return nothing - the array will be created on first push with the correct type
    return nothing
end

"""Get the isbits pointer type for a given element type and backend."""
function _get_isbits_ptr_type(backend::KA.CPU, ::Type{T}) where T
    return Vector{T}  # On CPU, Vector is already isbits-compatible for our purposes
end

function _get_isbits_ptr_type(backend, ::Type{T}) where T
    # For GPU backends, use argconvert to get the isbits device pointer type
    arr = KA.allocate(backend, T, 1)
    isbits_ptr = _get_isbits_ptr(backend, arr)
    return typeof(isbits_ptr)
end

"""
Convert a BLAS with backend arrays to isbits BLAS with device pointers.
Stores the backend arrays in `keep_alive` vector to prevent GC.
"""
function _to_isbits_blas(backend, blas::BLAS, keep_alive::Vector{Any})
    # Store the backend arrays to keep them alive
    push!(keep_alive, blas.nodes)
    push!(keep_alive, blas.primitives)

    # Get isbits device pointers
    isbits_nodes = _get_isbits_ptr(backend, blas.nodes)
    isbits_prims = _get_isbits_ptr(backend, blas.primitives)

    return BLAS(isbits_nodes, isbits_prims, blas.root_aabb)
end

"""
Append a single isbits BLAS to blas_array using GPU-friendly append!.
Returns the (possibly new) blas_array.
"""
function _append_blas!(backend, blas_array, isbits_blas)
    # Create a single-element array on CPU with the isbits BLAS, then adapt to backend
    single_arr = [isbits_blas]
    backend_arr = Adapt.adapt(backend, single_arr)

    if blas_array === nothing
        # First BLAS - create the array with correct type
        return backend_arr
    else
        # Append to existing array
        append!(blas_array, backend_arr)
        return blas_array
    end
end

"""
    is_valid(tlas::TLAS, handle::TLASHandle) -> Bool

Check if a handle is still valid (not deleted). O(1) operation.
"""
function is_valid(tlas::TLAS, handle::TLASHandle)::Bool
    haskey(tlas.handle_to_range, handle) && !(handle in tlas.deleted_handles)
end

"""
    n_instances(tlas::TLAS, handle::TLASHandle) -> Int

Get the number of instances referenced by a handle.
"""
function n_instances(tlas::TLAS, handle::TLASHandle)::Int
    haskey(tlas.handle_to_range, handle) || return 0
    handle in tlas.deleted_handles && return 0
    return length(tlas.handle_to_range[handle])
end

"""
    n_geometries(tlas::TLAS) -> Int

Get the total number of distinct geometries (BLASes) in the TLAS.
"""
n_geometries(tlas::TLAS) = tlas.blas_array === nothing ? 0 : length(tlas.blas_array)

"""
    n_total_instances(tlas::TLAS) -> Int

Get the total number of active instances in the TLAS.
"""
n_total_instances(tlas::TLAS) = length(tlas.instances)

# ------------------------------------------------------------------------------
# TLAS: push! operations - Direct GPU append
# ------------------------------------------------------------------------------

"""
    push!(tlas::TLAS, geometry) -> TLASHandle

Add geometry as a single instance at identity transform.
Returns a stable handle for later reference.
"""
function Base.push!(tlas::TLAS, geometry)
    inst = Instance(geometry)
    return push!(tlas, inst)
end

"""
    push!(tlas::TLAS, geometry, material_idx) -> TLASHandle

Add geometry as a single instance at identity transform with material index.
The material_idx (typically a HeteroVecIndex/MaterialIndex) is stored as metadata
on the triangles for material lookup during shading.
"""
function Base.push!(tlas::TLAS, geometry, material_idx)
    inst = Instance(geometry; metadata=material_idx)
    return push!(tlas, inst)
end

"""
    push!(tlas::TLAS, geometry, material_idx, transform::Mat4f) -> TLASHandle

Add geometry as a single instance with transform and material index.
"""
function Base.push!(tlas::TLAS, geometry, material_idx, transform::Mat4f)
    inst = Instance(geometry, transform, material_idx)
    return push!(tlas, inst)
end

"""
    push!(tlas::TLAS, geometry, material_idx, transforms::AbstractVector{Mat4f}) -> TLASHandle

Add geometry as multiple instances with transforms sharing the same material index.
"""
function Base.push!(tlas::TLAS, geometry, material_idx, transforms::AbstractVector{Mat4f})
    inst = Instance(geometry, transforms; metadata=material_idx)
    return push!(tlas, inst)
end

"""
    push!(tlas::TLAS, inst::Instance) -> TLASHandle

Add an Instance with transforms and metadata.
Returns a single handle referencing all instances in the group.

Uses direct GPU append - instances are appended to the GPU array immediately.
"""
function Base.push!(tlas::TLAS, inst::Instance)
    # Build triangles on CPU first, then convert to backend
    mesh = to_triangle_mesh(inst.geometry)
    cpu_triangles = [Triangle(mesh, i, inst.metadata[1])
                     for i in 1:div(length(mesh.indices), 3)
                     if !is_degenerate(get_vertices(mesh, i))]
    isempty(cpu_triangles) && error("Geometry has no valid triangles")

    # Convert triangles to backend and build BLAS directly on backend
    backend_triangles = Adapt.adapt(tlas.backend, cpu_triangles)
    blas = build_blas(backend_triangles)

    # Convert to isbits BLAS and append to blas_array (GPU append)
    isbits_blas = _to_isbits_blas(tlas.backend, blas, tlas.gpu_blas_arrays)
    tlas.blas_array = _append_blas!(tlas.backend, tlas.blas_array, isbits_blas)
    blas_idx = UInt32(length(tlas.blas_array))

    # Create InstanceDescriptors on CPU
    cpu_descriptors = map(inst.transforms) do transform
        inv_transform = Mat4f(inv(transform))
        instance_id = tlas.next_instance_id
        tlas.next_instance_id += UInt32(1)
        InstanceDescriptor(blas_idx, instance_id, transform, inv_transform, UInt32(0))
    end

    # Record range before append
    start_idx = length(tlas.instances) + 1

    # Adapt to backend and append (GPU append)
    backend_descriptors = Adapt.adapt(tlas.backend, cpu_descriptors)
    append!(tlas.instances, backend_descriptors)

    end_idx = length(tlas.instances)

    # Create handle and register range
    handle = TLASHandle(tlas.next_handle_id)
    tlas.next_handle_id += UInt32(1)
    tlas.handle_to_range[handle] = start_idx:end_idx

    tlas.dirty = true
    return handle
end

# ------------------------------------------------------------------------------
# TLAS: delete! operation
# ------------------------------------------------------------------------------

"""
    delete!(tlas::TLAS, handle::TLASHandle) -> Bool

Remove all instances referenced by the handle. Returns true if successful.
The handle becomes invalid after deletion.

Note: The instances array is compacted during sync!, not immediately.
"""
function Base.delete!(tlas::TLAS, handle::TLASHandle)::Bool
    haskey(tlas.handle_to_range, handle) || return false
    handle in tlas.deleted_handles && return false

    # Mark as deleted (will be compacted on sync!)
    push!(tlas.deleted_handles, handle)
    tlas.dirty = true

    return true
end

# ------------------------------------------------------------------------------
# TLAS: get_instance - retrieve instance data
# ------------------------------------------------------------------------------

"""
    get_instance(tlas::TLAS, handle::TLASHandle) -> InstanceDescriptor
    get_instance(tlas::TLAS, handle::TLASHandle, instance_idx::Integer) -> InstanceDescriptor

Retrieve the InstanceDescriptor for a handle. If the handle has multiple instances
(created with multiple transforms), use `instance_idx` to specify which one (1-based).

Note: Reads from GPU array, may involve a device-to-host copy.
"""
function get_instance(tlas::TLAS, handle::TLASHandle, instance_idx::Integer=1)
    haskey(tlas.handle_to_range, handle) || error("Invalid handle")
    handle in tlas.deleted_handles && error("Handle has been deleted")
    range = tlas.handle_to_range[handle]
    1 <= instance_idx <= length(range) || error("Instance index $instance_idx out of range 1:$(length(range))")

    idx = first(range) + instance_idx - 1
    # Copy single element from GPU to CPU
    return Array(tlas.instances[idx:idx])[1]
end

"""
    get_instances(tlas::TLAS, handle::TLASHandle) -> Vector{InstanceDescriptor}

Retrieve all InstanceDescriptors for a handle (for handles with multiple transforms).

Note: Reads from GPU array, involves a device-to-host copy.
"""
function get_instances(tlas::TLAS, handle::TLASHandle)
    haskey(tlas.handle_to_range, handle) || error("Invalid handle")
    handle in tlas.deleted_handles && error("Handle has been deleted")
    range = tlas.handle_to_range[handle]
    # Copy range from GPU to CPU
    return Array(tlas.instances[range])
end

# ------------------------------------------------------------------------------
# TLAS: update_transform! operations - Direct GPU updates
# ------------------------------------------------------------------------------

"""
    update_transform!(tlas::TLAS, handle::TLASHandle, transform::Mat4f)

Update the transform of a single-instance handle directly on GPU.
For handles with multiple instances, use `update_transforms!`.

After calling this, use `refit_tlas!` to update the BVH AABBs.
"""
function update_transform!(tlas::TLAS, handle::TLASHandle, transform::Mat4f)
    haskey(tlas.handle_to_range, handle) || error("Invalid handle")
    handle in tlas.deleted_handles && error("Handle has been deleted")
    range = tlas.handle_to_range[handle]
    length(range) == 1 || error("Handle has $(length(range)) instances, use update_transforms! for multiple")

    # Update single instance using kernel
    transforms = Adapt.adapt(tlas.backend, [transform])
    update_instance_transforms!(tlas, transforms, 1, first(range))

    return nothing
end

"""
    update_transforms!(tlas::TLAS, handle::TLASHandle, transforms::AbstractVector{Mat4f})

Update all instances' transforms in a group directly on GPU.
Length must match the number of instances in the handle.

The transforms array can be CPU or GPU - will be adapted to backend.
After calling this, use `refit_tlas!` to update the BVH AABBs.
"""
function update_transforms!(tlas::TLAS, handle::TLASHandle, transforms::AbstractVector{Mat4f})
    haskey(tlas.handle_to_range, handle) || error("Invalid handle")
    handle in tlas.deleted_handles && error("Handle has been deleted")
    range = tlas.handle_to_range[handle]
    length(transforms) == length(range) || error("Transform count ($(length(transforms))) != instance count ($(length(range)))")

    # Adapt transforms to backend if needed
    backend_transforms = Adapt.adapt(tlas.backend, transforms)
    update_instance_transforms!(tlas, backend_transforms, length(range), first(range))

    return nothing
end

# ------------------------------------------------------------------------------
# TLAS: update! for geometry replacement
# ------------------------------------------------------------------------------

"""
    update!(tlas::TLAS, handle::TLASHandle, new_geometry)

Replace the geometry (BLAS) for a handle. All instances sharing this BLAS get updated.
"""
function update!(tlas::TLAS, handle::TLASHandle, new_geometry)
    haskey(tlas.handle_to_range, handle) || error("Invalid handle")
    handle in tlas.deleted_handles && error("Handle has been deleted")
    range = tlas.handle_to_range[handle]
    isempty(range) && error("Handle has no instances")

    # Get blas_index from first instance (read from GPU)
    first_desc = Array(tlas.instances[first(range):first(range)])[1]
    blas_idx = Int(first_desc.blas_index)

    # Build new BLAS on backend (GPU-first)
    mesh = to_triangle_mesh(new_geometry)
    cpu_triangles = [Triangle(mesh, i, first_desc.instance_id)
                     for i in 1:div(length(mesh.indices), 3)
                     if !is_degenerate(get_vertices(mesh, i))]
    isempty(cpu_triangles) && error("New geometry has no valid triangles")

    backend_triangles = Adapt.adapt(tlas.backend, cpu_triangles)
    new_blas = build_blas(backend_triangles)

    # Convert to isbits and replace in blas_array
    # First, update gpu_blas_arrays (replace the old backing arrays)
    old_nodes_idx = 2*(blas_idx-1) + 1
    old_prims_idx = 2*(blas_idx-1) + 2
    tlas.gpu_blas_arrays[old_nodes_idx] = new_blas.nodes
    tlas.gpu_blas_arrays[old_prims_idx] = new_blas.primitives

    # Create isbits version and replace in blas_array
    isbits_blas = BLAS(
        _get_isbits_ptr(tlas.backend, new_blas.nodes),
        _get_isbits_ptr(tlas.backend, new_blas.primitives),
        new_blas.root_aabb
    )
    tlas.blas_array[blas_idx] = isbits_blas

    tlas.dirty = true
    return nothing
end

# ------------------------------------------------------------------------------
# TLAS: sync! - Rebuild BVH from GPU instances array
# ------------------------------------------------------------------------------

"""
    sync!(tlas::TLAS) -> TLAS

Rebuild the BVH structure if dirty. No-op if already up-to-date.

If there are deleted handles, compacts the instances array first.
Then rebuilds the BVH topology from the (compacted) instances.
"""
function sync!(tlas::TLAS)
    tlas.dirty || return tlas
    _rebuild_bvh!(tlas)
    return tlas
end

"""Internal: Compact deleted instances and rebuild TLAS BVH."""
function _rebuild_bvh!(tlas::TLAS)
    # If there are deletions, compact the instances array
    if !isempty(tlas.deleted_handles)
        _compact_instances!(tlas)
    end

    n = length(tlas.instances)
    if n == 0
        tlas.nodes = KA.allocate(tlas.backend, BVHNode2, 0)
        tlas.root_aabb = Bounds3()
        tlas.dirty = false
        return
    end

    # Build TLAS BVH structure from existing GPU arrays
    # instances are already on backend, blas_array has isbits BLASes
    built_tlas = build_tlas(tlas.blas_array, tlas.instances)

    # Update nodes with rebuilt structure
    tlas.nodes = built_tlas.nodes
    tlas.root_aabb = built_tlas.root_aabb

    tlas.dirty = false
    return
end

"""Internal: Compact instances array by removing deleted handles."""
function _compact_instances!(tlas::TLAS)
    # Collect ranges to keep (non-deleted handles)
    ranges_to_keep = UnitRange{Int}[]
    for (handle, range) in tlas.handle_to_range
        handle in tlas.deleted_handles && continue
        push!(ranges_to_keep, range)
    end
    sort!(ranges_to_keep, by=first)

    # If nothing to compact, just clear deleted handles
    if length(ranges_to_keep) == length(tlas.handle_to_range) - length(tlas.deleted_handles)
        # Copy valid ranges to new array and update handle mappings
        cpu_instances = Array(tlas.instances)
        new_instances = InstanceDescriptor[]
        new_handle_to_range = Dict{TLASHandle, UnitRange{Int}}()

        for (handle, range) in tlas.handle_to_range
            handle in tlas.deleted_handles && continue
            new_start = length(new_instances) + 1
            append!(new_instances, cpu_instances[range])
            new_end = length(new_instances)
            new_handle_to_range[handle] = new_start:new_end
        end

        # Update TLAS with compacted instances
        tlas.instances = Adapt.adapt(tlas.backend, new_instances)
        tlas.handle_to_range = new_handle_to_range
    end

    # Remove deleted handles from tracking
    for handle in tlas.deleted_handles
        delete!(tlas.handle_to_range, handle)
    end
    empty!(tlas.deleted_handles)
end

# ------------------------------------------------------------------------------
# TLAS: Adapt integration - Returns StaticTLAS for kernel traversal
# ------------------------------------------------------------------------------

"""
    Adapt.adapt_structure(to, tlas::TLAS) -> StaticTLAS

Convert TLAS to StaticTLAS for GPU kernel usage.
Syncs if dirty, then extracts isbits device pointers for kernels.

GPU-first: All arrays are already on the TLAS's backend.
The target backend must match the TLAS backend.

Note: The returned StaticTLAS references arrays owned by the TLAS.
The TLAS must stay alive while the StaticTLAS is in use.
"""
function Adapt.adapt_structure(to, tlas::TLAS)
    sync!(tlas)
    return StaticTLAS(
        adapt(to, tlas.nodes),
        adapt(to, tlas.instances),
        adapt(to, tlas.blas_array),
        tlas.root_aabb
    )
end

"""
    Adapt.adapt_structure(to, tlas::StaticTLAS) -> StaticTLAS

Adapt StaticTLAS arrays. If already isbits, returns as-is.
Otherwise adapts each array (CLArray → CLDeviceVector).

Note: StaticTLAS should come from adapting a mutable TLAS, where BLASes
already have isbits device pointers. Use TLAS(items) to create a mutable
TLAS that properly manages GPU array lifetimes.
"""
function Adapt.adapt_structure(to, tlas::StaticTLAS)
    # If already isbits, return as-is (already kernel-ready)
    isbitstype(typeof(tlas)) && return tlas

    # BLASes should already have isbits pointers from mutable TLAS path
    # Just adapt the outer arrays (CLArray → CLDeviceVector)
    return StaticTLAS(
        Adapt.adapt(to, tlas.nodes),
        Adapt.adapt(to, tlas.instances),
        Adapt.adapt(to, tlas.blas_array),
        tlas.root_aabb
    )
end

# Adapt BLAS when adapting to kernel arguments
function Adapt.adapt_structure(to, blas::BLAS)
    BLAS(
        Adapt.adapt(to, blas.nodes),
        Adapt.adapt(to, blas.primitives),
        blas.root_aabb
    )
end

# ==============================================================================
# AABB and Morton Code Utilities
# ==============================================================================

"""Compute AABB from BVHNode2 for BLAS (BVH2IL format with triangle vertices in leaves)."""
@inline function get_node_aabb(node::BVHNode2, is_interior::Bool)::Bounds3
    if is_interior
        # Interior: union of children AABBs
        Bounds3(
            min.(node.aabb0_min, node.aabb1_min),
            max.(node.aabb0_max, node.aabb1_max)
        )
    else
        # Leaf: vertices stored in aabb slots (BVH2IL format)
        # Compute AABB from triangle vertices v0, v1, v2
        v0 = Point3f(node.aabb0_min...)
        v1 = Point3f(node.aabb0_max...)
        v2 = Point3f(node.aabb1_min...)

        p_min = min.(min.(v0, v1), v2)
        p_max = max.(max.(v0, v1), v2)

        Bounds3(p_min, p_max)
    end
end

"""Compute AABB from BVHNode2 for TLAS (AABBs stored directly in leaves)."""
@inline function get_tlas_node_aabb(node::BVHNode2, is_interior::Bool)::Bounds3
    if is_interior
        # Interior: union of children AABBs
        Bounds3(
            min.(node.aabb0_min, node.aabb1_min),
            max.(node.aabb0_max, node.aabb1_max)
        )
    else
        # Leaf: instance AABB stored directly in aabb0 fields
        Bounds3(node.aabb0_min, node.aabb0_max)
    end
end

"""3-dilate bits for Morton code (spreads bits by factor of 3)."""
@inline function expand_bits(x::UInt32)::UInt32
    x = (x * 0x00010001) & 0xFF0000FF
    x = (x * 0x00000101) & 0x0F00F00F
    x = (x * 0x00000011) & 0xC30C30C3
    x = (x * 0x00000005) & 0x49249249
    return x
end

"""
Calculate 30-bit Morton code from normalized 3D point [0,1]³.
Interleaves x,y,z bits to create space-filling Z-curve ordering.
"""
@inline function morton_code_30bit(p::Point3f)::UInt32
    # Clamp to [0, 1023] for 10-bit precision per axis
    unit_side = 1024.0f0
    x = clamp(p[1] * unit_side, 0.0f0, unit_side - 1.0f0)
    y = clamp(p[2] * unit_side, 0.0f0, unit_side - 1.0f0)
    z = clamp(p[3] * unit_side, 0.0f0, unit_side - 1.0f0)

    # Interleave bits: xxyyzzxxyyzzxxyyzz...
    return (expand_bits(unsafe_trunc(UInt32, x)) << 2) |
           (expand_bits(unsafe_trunc(UInt32, y)) << 1) |
            expand_bits(unsafe_trunc(UInt32, z))
end

"""Count leading zeros (clz) for 32-bit integer."""
@inline function clz32(x::UInt32)::Int32
    x == 0 && return Int32(32)
    return Int32(31 - (sizeof(UInt32)*8 - 1 - leading_zeros(x)))
end

"""
Compute longest common prefix (LCP) of Morton codes.
Uses index fallback when codes are identical.
"""
@inline function delta(i1::Int32, i2::Int32, morton_codes::AbstractVector{UInt32}, num_prims::Int32)::Int32
    # Bounds check
    left = min(i1, i2)
    right = max(i1, i2)

    (left < 1 || right > num_prims) && return Int32(-1)

    left_code = morton_codes[left]
    right_code = morton_codes[right]

    # If codes differ, return common prefix length
    # If codes are same, use indices as tiebreaker
    if left_code != right_code
        return Int32(clz32(left_code ⊻ right_code))
    else
        return Int32(32 + clz32(UInt32(left) ⊻ UInt32(right)))
    end
end

"""Find the span of primitives covered by this internal node (Karras 2012)."""
@inline function find_span_for_node(
    idx::Int32,
    morton_codes::AbstractVector{UInt32},
    n_prims::Int32
)::Tuple{Int32, Int32}
    # Determine direction
    d_left = delta(idx, idx - Int32(1), morton_codes, n_prims)
    d_right = delta(idx, idx + Int32(1), morton_codes, n_prims)
    d = d_right > d_left ? Int32(1) : Int32(-1)

    # Compute upper bound for length
    delta_min = delta(idx, idx - d, morton_codes, n_prims)
    l_max = Int32(2)
    while delta(idx, idx + l_max * d, morton_codes, n_prims) > delta_min
        l_max *= Int32(2)
    end

    # Binary search for the other end
    l = Int32(0)
    t = l_max
    while t > Int32(1)
        t = t ÷ Int32(2)
        if delta(idx, idx + (l + t) * d, morton_codes, n_prims) > delta_min
            l = l + t
        end
    end
    j = idx + l * d

    # Return sorted span
    return d > Int32(0) ? (idx, j) : (j, idx)
end

"""Find the split position within a span (Karras 2012)."""
@inline function find_split_in_span(
    span_left::Int32,
    span_right::Int32,
    morton_codes::AbstractVector{UInt32},
    n_prims::Int32
)::Int32
    # Calculate the number of identical bits from higher end
    numidentical = delta(span_left, span_right, morton_codes, n_prims)

    # Binary search for split position using midpoint
    left = span_left
    right = span_right
    while right > left + Int32(1)
        # Proposed split at midpoint
        newsplit = (right + left) ÷ Int32(2)

        # If it has more equal leading bits than left and right, accept it
        if delta(left, newsplit, morton_codes, n_prims) > numidentical
            left = newsplit
        else
            right = newsplit
        end
    end

    return left
end

"""Compute leaf node index from primitive index."""
@inline function leaf_index(prim_idx::Integer, n_prims::Int32)::Int
    return Int(n_prims) - 1 + prim_idx
end

"""Refit AABB for one internal node from its children."""
@inline function refit_node_aabb(
    node_idx::Int32,
    nodes::AbstractVector{BVHNode2},
    n_prims::Int32
)::BVHNode2
    @inbounds node = nodes[node_idx]
    child0 = node.child0
    child1 = node.child1

    is_child0_internal = child0 < n_prims
    is_child1_internal = child1 < n_prims

    aabb0 = get_node_aabb(nodes[child0], is_child0_internal)
    aabb1 = get_node_aabb(nodes[child1], is_child1_internal)

    return BVHNode2(
        aabb0.p_min, aabb0.p_max,
        aabb1.p_min, aabb1.p_max,
        node.child0, node.child1, node.parent
    )
end

@inline function refit_tlas_node_aabb(
    node_idx::Int32,
    nodes::AbstractVector{BVHNode2},
    n_instances::Int32
)::BVHNode2
    @inbounds node = nodes[node_idx]
    child0 = node.child0
    child1 = node.child1

    is_child0_internal = child0 < n_instances
    is_child1_internal = child1 < n_instances

    aabb0 = get_tlas_node_aabb(nodes[child0], is_child0_internal)
    aabb1 = get_tlas_node_aabb(nodes[child1], is_child1_internal)

    return BVHNode2(
        aabb0.p_min, aabb0.p_max,
        aabb1.p_min, aabb1.p_max,
        node.child0, node.child1, node.parent
    )
end

# ==============================================================================
# BLAS Construction (LBVH Algorithm)
# ==============================================================================

"""
    build_blas(primitives) -> BLAS

Build a Bottom-Level Acceleration Structure using Linear BVH (LBVH).

Uses KernelAbstractions for automatic CPU/GPU execution based on input array type.

Algorithm:
1. Compute scene AABB
2. Calculate Morton codes in parallel (GPU kernel)
3. Sort primitives by Morton code
4. Build binary radix tree in parallel (GPU kernel)
5. Compute AABBs bottom-up

Based on Karras 2012 "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"

# Arguments
- `primitives`: Vector or GPU array of Triangle objects (array type determines backend)

# Example
```julia
# CPU execution
blas_cpu = build_blas(triangles)  # Vector{Triangle}

# GPU execution (CUDA)
using CUDA
gpu_triangles = CuArray(triangles)
blas_gpu = build_blas(gpu_triangles)  # CuArray{Triangle}
```
"""
function build_blas(
    primitives::AbstractVector{T}
) where {T <: Triangle}
    n = length(primitives)
    n == 0 && error("Cannot build BLAS from empty primitive list")

    # Infer backend from input array type
    backend = KA.get_backend(primitives)

    # Compute scene AABB (works on both CPU and GPU arrays)
    scene_aabb = mapreduce(world_bound, ∪, primitives, init=Bounds3())
    scene_min = scene_aabb.p_min
    scene_extent = Vec3f(scene_aabb.p_max - scene_aabb.p_min)

    # Allocate arrays on same backend as input
    morton_codes = KA.allocate(backend, UInt32, n)

    # Launch kernel: Calculate Morton codes
    calc_kernel! = calculate_morton_codes_kernel!(backend)
    calc_kernel!(morton_codes, primitives, scene_min, scene_extent, ndrange=n)

    # Sort primitives by Morton codes
    # AcceleratedKernels only supports GPU backends, use Julia's sortperm for CPU
    if backend isa KA.CPU
        perm = sortperm(morton_codes)
        morton_codes .= morton_codes[perm]
        primitives .= primitives[perm]
    else
        AK.merge_sort_by_key!(morton_codes, primitives)
    end

    # Allocate nodes and initialize with empty values
    # Use kernel-based fill (OpenCL's fill! doesn't support struct types)
    nodes = KA.allocate(backend, BVHNode2, 2*n - 1)
    empty_node = BVHNode2(
        Point3f(0), Point3f(0), Point3f(0), Point3f(0),
        INVALID_NODE, INVALID_NODE, INVALID_NODE
    )
    fill_kernel! = fill_bvhnode2_kernel!(backend)
    fill_kernel!(nodes, empty_node, ndrange=length(nodes))

    # Launch kernel: Emit topology (only if n > 1, i.e., there are internal nodes)
    if n > 1
        topo_kernel! = emit_topology_kernel!(backend)
        topo_kernel!(nodes, morton_codes, Int32(n), ndrange=n-1)

        # Launch kernel: Set parent pointers
        parent_kernel! = set_parent_pointers_kernel!(backend)
        parent_kernel!(nodes, Int32(n), ndrange=n-1)
    end

    # Launch kernel: Create leaf nodes
    leaf_kernel! = create_leaf_nodes_kernel!(backend)
    leaf_kernel!(nodes, primitives, Int32(n), ndrange=n)

    # Refit AABBs bottom-up (parallel using atomic counters)
    update_flags = KA.zeros(backend, UInt32, n - 1)  # One flag per internal node
    refit_kernel! = refit_aabbs_kernel!(backend)
    refit_kernel!(nodes, update_flags, Int32(n), ndrange=n)

    # Compute root AABB - check if root is interior or leaf
    # Use explicit copy to CPU to avoid scalar indexing issues on GPU
    KA.synchronize(backend)
    root_node = Array(nodes[1:1])[1]
    root_is_interior = is_interior(root_node)
    root_aabb = get_node_aabb(root_node, root_is_interior)

    return BLAS(nodes, primitives, root_aabb)
end

# ==============================================================================
# TLAS Construction
# ==============================================================================

"""Build topology for one TLAS internal node (same algorithm as BLAS)."""
@inline function build_tlas_topology_for_node(
    idx::Int32,
    morton_codes::AbstractVector{UInt32},
    n_instances::Int32
)::BVHNode2
    # Helper function
    @inline leaf_idx(j::Int32) = n_instances - Int32(1) + j

    # Find span
    span_left, span_right = find_span_for_node(idx, morton_codes, n_instances)

    # Find split
    split = find_split_in_span(span_left, span_right, morton_codes, n_instances)

    # Determine children (matches HLSL reference exactly)
    # child0 is leaf only if split == span_left
    # child1 is leaf only if split + 1 == span_right
    child0 = (split == span_left) ? leaf_idx(split) : split
    child1_idx = split + Int32(1)
    child1 = (child1_idx == span_right) ? leaf_idx(child1_idx) : child1_idx

    return BVHNode2(
        Point3f(0), Point3f(0), Point3f(0), Point3f(0),
        UInt32(child0), UInt32(child1), INVALID_NODE
    )
end

"""
    build_tlas(blas_array::AbstractVector{BLAS}, instances::AbstractVector{InstanceDescriptor}) -> StaticTLAS

Build a Top-Level Acceleration Structure over instances.
Uses LBVH over transformed instance AABBs.

Returns a StaticTLAS suitable for ray traversal.
Uses KernelAbstractions for automatic CPU/GPU execution based on input array type.
"""
function build_tlas(
    blas_array::AbstractVector{B},
    instances::AbstractVector{InstanceDescriptor}
) where {B <: BLAS}
    n = length(instances)
    n == 0 && return StaticTLAS(BVHNode2[], instances, blas_array, Bounds3())

    # Infer backend from blas_array type (instances may be CPU Vector)
    backend = KA.get_backend(blas_array)

    # Compute scene AABB from transformed instance bounds using GPU kernel
    # Allocate arrays for per-instance world AABBs
    aabb_mins = KA.allocate(backend, Point3f, n)
    aabb_maxs = KA.allocate(backend, Point3f, n)

    # Adapt instances to backend for kernel execution (instances may be CPU Vector)
    backend_instances = Adapt.adapt(backend, instances)

    # Launch kernel to compute world AABBs in parallel
    aabb_kernel! = compute_instance_aabbs_kernel!(backend)
    aabb_kernel!(aabb_mins, aabb_maxs, backend_instances, blas_array, ndrange=n)
    KA.synchronize(backend)

    # Copy results to CPU and compute scene AABB via reduction
    cpu_mins = Array(aabb_mins)
    cpu_maxs = Array(aabb_maxs)

    scene_min_p = cpu_mins[1]
    scene_max_p = cpu_maxs[1]
    for i in 2:n
        scene_min_p = Point3f(min(scene_min_p[1], cpu_mins[i][1]),
                              min(scene_min_p[2], cpu_mins[i][2]),
                              min(scene_min_p[3], cpu_mins[i][3]))
        scene_max_p = Point3f(max(scene_max_p[1], cpu_maxs[i][1]),
                              max(scene_max_p[2], cpu_maxs[i][2]),
                              max(scene_max_p[3], cpu_maxs[i][3]))
    end
    scene_aabb = Bounds3(scene_min_p, scene_max_p)

    scene_min = scene_aabb.p_min
    aabb_extent = scene_aabb.p_max - scene_aabb.p_min
    # Handle degenerate cases (avoid division by zero)
    scene_extent = Vec3f(
        max(aabb_extent[1], 1f-6),
        max(aabb_extent[2], 1f-6),
        max(aabb_extent[3], 1f-6)
    )

    # Allocate arrays on same backend as input
    morton_codes = KA.allocate(backend, UInt32, n)

    # Launch kernel: Calculate Morton codes for instances
    calc_kernel! = calculate_tlas_morton_codes_kernel!(backend)
    calc_kernel!(morton_codes, backend_instances, blas_array, scene_min, scene_extent, ndrange=n)
    KA.synchronize(backend)

    # Sort indices by Morton codes
    # AcceleratedKernels only supports GPU backends, use Julia's sortperm for CPU
    if backend isa KA.CPU
        sorted_indices = sortperm(morton_codes)
        morton_codes .= morton_codes[sorted_indices]
    else
        sorted_indices = KA.allocate(backend, Int, n)
        iota_k! = iota_kernel!(backend)
        iota_k!(sorted_indices, ndrange=n)
        KA.synchronize(backend)
        AK.merge_sort_by_key!(morton_codes, sorted_indices)
    end

    # Allocate nodes and initialize with empty values
    # Use kernel-based fill (OpenCL's fill! doesn't support struct types)
    nodes = KA.allocate(backend, BVHNode2, max(1, 2*n - 1))
    empty_node = BVHNode2(
        Point3f(0), Point3f(0), Point3f(0), Point3f(0),
        INVALID_NODE, INVALID_NODE, INVALID_NODE
    )
    fill_kernel! = fill_bvhnode2_kernel!(backend)
    fill_kernel!(nodes, empty_node, ndrange=length(nodes))

    # Single-instance case: trivial TLAS
    if n == 1
        # For CPU, sorted_indices is already a CPU array; for GPU, copy to avoid scalar indexing
        original_idx = backend isa KA.CPU ? sorted_indices[1] : Array(sorted_indices[1:1])[1]

        # Use scene_aabb computed from kernel (same as the single instance's world AABB)
        world_aabb = scene_aabb

        # Create leaf node on CPU and copy to backend
        leaf_node = BVHNode2(
            world_aabb.p_min, world_aabb.p_max,
            Point3f(0), Point3f(0),
            INVALID_NODE, UInt32(original_idx - 1),
            INVALID_NODE
        )
        cpu_nodes = [leaf_node]
        copyto!(nodes, Adapt.adapt(backend, cpu_nodes))

        return StaticTLAS(nodes, backend_instances, blas_array, world_aabb)
    end

    # Multi-instance case: build proper LBVH

    # Launch kernel: Emit topology (reuse BLAS topology kernel - same algorithm)
    topo_kernel! = emit_topology_kernel!(backend)
    topo_kernel!(nodes, morton_codes, Int32(n), ndrange=n-1)
    # Launch kernel: Set parent pointers
    parent_kernel! = set_parent_pointers_kernel!(backend)
    parent_kernel!(nodes, Int32(n), ndrange=n-1)
    # Launch kernel: Create TLAS leaf nodes (different from BLAS - stores AABBs, not vertices)
    leaf_kernel! = create_tlas_leaf_nodes_kernel!(backend)
    leaf_kernel!(nodes, sorted_indices, backend_instances, blas_array, Int32(n), ndrange=n)
    # Refit AABBs bottom-up (parallel using atomic counters)
    update_flags = KA.zeros(backend, UInt32, n - 1)
    refit_kernel! = refit_tlas_aabbs_kernel!(backend)
    refit_kernel!(nodes, update_flags, Int32(n), ndrange=n)

    # Get root AABB (copy to CPU to avoid scalar indexing)
    root_node = Array(nodes[1:1])[1]
    root_aabb = get_tlas_node_aabb(root_node, true)

    return StaticTLAS(nodes, backend_instances, blas_array, root_aabb)
end


# Type union for traversal - both TLAS and StaticTLAS have the same traversal-relevant fields
const TraversableTLAS = Union{TLAS, StaticTLAS}

# ==============================================================================
# Transform Utilities
# ==============================================================================

"""Transform point by 4x4 matrix."""
@inline function transform_point(m::Mat4f, p::Point3f)::Point3f
    ph = SVector{4, Float32}(p[1], p[2], p[3], 1.0f0)
    pt = m * ph
    w = pt[4]
    Point3f(pt[1] / w, pt[2] / w, pt[3] / w)
end

"""Transform direction by 4x4 matrix (ignoring translation)."""
@inline function transform_direction(m::Mat4f, v::Vec3f)::Vec3f
    Vec3f(
        m[1,1] * v[1] + m[1,2] * v[2] + m[1,3] * v[3],
        m[2,1] * v[1] + m[2,2] * v[2] + m[2,3] * v[3],
        m[3,1] * v[1] + m[3,2] * v[2] + m[3,3] * v[3]
    )
end

# ==============================================================================
# Two-Level Traversal
# ==============================================================================

"""Sentinel value to mark top-level to bottom-level transitions."""
const TOP_LEVEL_SENTINEL = 0xFFFFFFFE

"""
    safe_invdir(d::Vec3f) -> Vec3f

Safe ray direction inversion that avoids division by zero.
Clamps near-zero components to ±1e-5.
Matches HLSL reference implementation.
"""
@inline function safe_invdir(d::Vec3f)::Vec3f
    ooeps = 1.0f-5
    inv_x = 1.0f0 / (abs(d[1]) > ooeps ? d[1] : copysign(ooeps, d[1]))
    inv_y = 1.0f0 / (abs(d[2]) > ooeps ? d[2] : copysign(ooeps, d[2]))
    inv_z = 1.0f0 / (abs(d[3]) > ooeps ? d[3] : copysign(ooeps, d[3]))
    return Vec3f(inv_x, inv_y, inv_z)
end

"""
    fast_intersect_triangle(ray_o, ray_d, v0, v1, v2, t_min, closest_t) -> (hit, t, u, v)

Möller-Trumbore ray-triangle intersection test.
Matches HLSL reference implementation.
"""
@inline function fast_intersect_triangle(
    ray_o::Point3f, ray_d::Vec3f,
    v0::Point3f, v1::Point3f, v2::Point3f,
    t_min::Float32, closest_t::Float32
)
    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    s1 = cross(ray_d, e2)
    determinant = dot(s1, e1)
    invd = 1.0f0 / determinant

    # Calculate distance from v0 to ray origin
    d = ray_o - v0
    u = dot(d, s1) * invd

    # Test u parameter
    if u < 0.0f0 || u > 1.0f0
        return (false, 0.0f0, 0.0f0, 0.0f0)
    end

    # Prepare to test v parameter
    s2 = cross(d, e1)
    v = dot(ray_d, s2) * invd

    # Test v parameter
    if v < 0.0f0 || (u + v) > 1.0f0
        return (false, 0.0f0, 0.0f0, 0.0f0)
    end

    # Calculate t
    t = dot(e2, s2) * invd

    # Test t against range
    if t < t_min || t > closest_t
        return (false, 0.0f0, 0.0f0, 0.0f0)
    end

    return (true, t, u, v)
end

"""
    intersect_internal_node(node, ray_inv_d, ray_o, t_min, t_max) -> (near_child, far_child)

Test ray against internal node's two children AABBs.
Returns ordered children indices (near first, far second).
INVALID_NODE if child is not intersected.
Matches HLSL IntersectInternalNode.
"""
@inline function intersect_internal_node(
    node::BVHNode2,
    ray_inv_d::Vec3f,
    ray_o::Point3f,
    t_min::Float32,
    t_max::Float32
)
    # Get child AABBs
    aabb0 = Bounds3(Point3f(node.aabb0_min...), Point3f(node.aabb0_max...))
    aabb1 = Bounds3(Point3f(node.aabb1_min...), Point3f(node.aabb1_max...))

    # Test both children
    t0_min, t0_max = fast_intersect_bbox(ray_o, ray_inv_d, aabb0, t_min, t_max)
    t1_min, t1_max = fast_intersect_bbox(ray_o, ray_inv_d, aabb1, t_min, t_max)

    # Determine which children to traverse
    traverse0 = (t0_min <= t0_max) ? node.child0 : INVALID_NODE
    traverse1 = (t1_min <= t1_max) ? node.child1 : INVALID_NODE

    # Order by distance (near first)
    if t0_min < t1_min && traverse0 != INVALID_NODE
        return (traverse0, traverse1)
    else
        return (traverse1, traverse0)
    end
end

"""
    fast_intersect_bbox(ray_o, ray_inv_d, bbox, t_min, t_max) -> (entry_t, exit_t)

Fast ray-AABB intersection using slab method.
Returns parametric distances to entry and exit points.
Matches HLSL fast_intersect_bbox.
"""
@inline function fast_intersect_bbox(
    ray_o::Point3f,
    ray_inv_d::Vec3f,
    bbox::Bounds3,
    t_min::Float32,
    t_max::Float32
)
    oxinvdir = -ray_o .* ray_inv_d
    f = bbox.p_max .* ray_inv_d .+ oxinvdir
    n = bbox.p_min .* ray_inv_d .+ oxinvdir

    tmax_vec = max.(f, n)
    tmin_vec = min.(f, n)

    max_t = min(minimum(tmax_vec), t_max)
    min_t = max(maximum(tmin_vec), t_min)

    return (min_t, max_t)
end

"""
    intersect_leaf_node(node, ray_d, ray_o, t_min, closest_t) -> (hit, t, u, v)

Test ray against triangle stored in leaf node.
Returns hit status and intersection parameters.
Matches HLSL IntersectLeafNode.
"""
@inline function intersect_leaf_node(
    node::BVHNode2,
    ray_d::Vec3f,
    ray_o::Point3f,
    t_min::Float32,
    closest_t::Float32
)
    # In BVH2IL format, leaf nodes store triangle vertices in AABB slots
    v0 = Point3f(node.aabb0_min...)
    v1 = Point3f(node.aabb0_max...)
    v2 = Point3f(node.aabb1_min...)

    return fast_intersect_triangle(ray_o, ray_d, v0, v1, v2, t_min, closest_t)
end

"""
    closest_hit(tlas::TLAS, ray::AbstractRay) -> (hit, primitive, distance, barycentric, instance_id)

Traverse two-level BVH to find closest ray intersection.

Algorithm:
1. Traverse TLAS to find candidate instances
2. Transform ray to local space
3. Traverse BLAS for geometry intersection
4. Transform back to world space
5. Return closest hit across all instances
"""
@inline function closest_hit(tlas::TraversableTLAS, ray::R) where {R <: AbstractRay}
    # Initialize traversal state - matches HLSL TraceRays
    ray = check_direction(ray)
    ray_o::Point3f = ray.o
    ray_d::Vec3f = ray.d
    ray_mint::Float32 = 0.0f0  # Minimum t for intersection
    ray_maxt::Float32 = ray.t_max
    ray_inv_d::Vec3f = safe_invdir(ray_d)  # Use safe inversion to avoid division by zero

    # Stack for traversal
    stack = MVector{64, UInt32}(undef)
    stack_ptr::Int32 = Int32(1)
    @inbounds stack[stack_ptr] = INVALID_NODE

    # Traversal state - use Int32 for indices to avoid UInt32 arithmetic issues
    current_instance::Int32 = Int32(-1)  # -1 means no instance (top level)
    closest_instance::Int32 = Int32(-1)
    closest_prim::UInt32 = INVALID_NODE
    hit_u::Float32 = 0.0f0
    hit_v::Float32 = 0.0f0

    # Entry point is node 1 (1-indexed in Julia)
    node_index::UInt32 = UInt32(1)

    # Get typed references to avoid repeated field access
    tlas_nodes = tlas.nodes
    tlas_instances = tlas.instances
    tlas_blas_array = tlas.blas_array

    @inbounds while node_index != INVALID_NODE
        # Fetch node based on current level
        node::BVHNode2 = if current_instance < Int32(0)
            tlas_nodes[node_index]
        else
            # current_instance is a 0-indexed instance index
            inst = tlas_instances[current_instance + Int32(1)]
            blas = tlas_blas_array[inst.blas_index]
            blas.nodes[node_index]
        end

        is_leaf::Bool = (node.child0 == INVALID_NODE)

        if !is_leaf
            # Interior node - test both children and get ordered traversal
            near_child, far_child = intersect_internal_node(node, ray_inv_d, ray_o, ray_mint, ray_maxt)

            # Push far child if valid
            if far_child != INVALID_NODE
                stack_ptr += Int32(1)
                stack[stack_ptr] = far_child
            end

            # Visit near child if valid
            if near_child != INVALID_NODE
                node_index = near_child
                continue
            end
        elseif current_instance < Int32(0)
            # Top-level leaf - transition to instance
            current_instance = Int32(node.child1)  # 0-indexed instance index

            # Push sentinel
            stack_ptr += Int32(1)
            stack[stack_ptr] = TOP_LEVEL_SENTINEL

            # Get instance and transform ray
            node_index = UInt32(1)  # Start at root of BLAS
            inst = tlas_instances[current_instance + Int32(1)]
            ray_o = transform_point(inst.inv_transform, ray.o)
            ray_d = transform_direction(inst.inv_transform, ray.d)
            ray_inv_d = safe_invdir(ray_d)
            continue
        else
            # Bottom-level leaf - test triangle
            hit, t, u, v = intersect_leaf_node(node, ray_d, ray_o, ray_mint, ray_maxt)
            if hit
                # Update closest hit
                ray_maxt = t
                closest_instance = current_instance
                closest_prim = node.child1
                hit_u = u
                hit_v = v
            end
        end

        # Pop from stack
        node_index = stack[stack_ptr]
        stack_ptr -= Int32(1)

        # Check for level transition
        if node_index == TOP_LEVEL_SENTINEL
            # Return to top level
            node_index = stack[stack_ptr]
            stack_ptr -= Int32(1)
            current_instance = Int32(-1)

            # Restore original ray
            ray_o = ray.o
            ray_d = ray.d
            ray_inv_d = safe_invdir(ray_d)
        end
    end

    # Fill in hit output - matches HLSL
    @inbounds if closest_instance >= Int32(0)
        inst = tlas_instances[closest_instance + Int32(1)]
        blas = tlas_blas_array[inst.blas_index]
        tri = blas.primitives[closest_prim]
        w = 1.0f0 - hit_u - hit_v
        bary = SVector{3, Float32}(w, hit_u, hit_v)
        return (true, tri, ray_maxt, bary, inst.instance_id)
    else
        # No hit - return dummy values
        dummy_tri = tlas_blas_array[1].primitives[1]
        bary = SVector{3, Float32}(0.0f0, 0.0f0, 0.0f0)
        return (false, dummy_tri, 0.0f0, bary, INVALID_NODE)
    end
end

"""
    any_hit(tlas::TLAS, ray::AbstractRay) -> (hit, primitive, distance, barycentric, instance_id)

Traverse two-level BVH to find ANY ray intersection (returns on first hit).
Faster than closest_hit when only occlusion testing is needed.

Matches HLSL TraceRays with ANY_HIT defined.
"""
@inline function any_hit(tlas::TraversableTLAS, ray::R) where {R <: AbstractRay}
    # Initialize traversal state - matches HLSL TraceRays
    ray = check_direction(ray)
    ray_o::Point3f = ray.o
    ray_d::Vec3f = ray.d
    ray_mint::Float32 = 0.0f0
    ray_maxt::Float32 = ray.t_max
    ray_inv_d::Vec3f = safe_invdir(ray_d)

    # Stack for traversal
    stack = MVector{64, UInt32}(undef)
    stack_ptr::Int32 = Int32(1)
    @inbounds stack[stack_ptr] = INVALID_NODE

    # Traversal state - use Int32 for indices to avoid UInt32 arithmetic issues
    current_instance::Int32 = Int32(-1)  # -1 means no instance (top level)

    # Entry point is node 1 (1-indexed in Julia)
    node_index::UInt32 = UInt32(1)

    # Get typed references to avoid repeated field access
    tlas_nodes = tlas.nodes
    tlas_instances = tlas.instances
    tlas_blas_array = tlas.blas_array

    @inbounds while node_index != INVALID_NODE
        # Fetch node based on current level
        node::BVHNode2 = if current_instance < Int32(0)
            tlas_nodes[node_index]
        else
            # current_instance is a 0-indexed instance index
            inst = tlas_instances[current_instance + Int32(1)]
            blas = tlas_blas_array[inst.blas_index]
            blas.nodes[node_index]
        end

        is_leaf::Bool = (node.child0 == INVALID_NODE)

        if !is_leaf
            # Interior node - test both children and get ordered traversal
            near_child, far_child = intersect_internal_node(node, ray_inv_d, ray_o, ray_mint, ray_maxt)

            # Push far child if valid
            if far_child != INVALID_NODE
                stack_ptr += Int32(1)
                stack[stack_ptr] = far_child
            end

            # Visit near child if valid
            if near_child != INVALID_NODE
                node_index = near_child
                continue
            end
        elseif current_instance < Int32(0)
            # Top-level leaf - transition to instance
            current_instance = Int32(node.child1)  # 0-indexed instance index

            # Push sentinel
            stack_ptr += Int32(1)
            stack[stack_ptr] = TOP_LEVEL_SENTINEL

            # Get instance and transform ray
            node_index = UInt32(1)  # Start at root of BLAS
            inst = tlas_instances[current_instance + Int32(1)]
            ray_o = transform_point(inst.inv_transform, ray.o)
            ray_d = transform_direction(inst.inv_transform, ray.d)
            ray_inv_d = safe_invdir(ray_d)
            continue
        else
            # Bottom-level leaf - test triangle
            hit, t, u, v = intersect_leaf_node(node, ray_d, ray_o, ray_mint, ray_maxt)
            if hit
                # ANY_HIT: return immediately on first hit
                inst = tlas_instances[current_instance + Int32(1)]
                blas = tlas_blas_array[inst.blas_index]
                tri = blas.primitives[node.child1]
                w = 1.0f0 - u - v
                bary = SVector{3, Float32}(w, u, v)
                return (true, tri, t, bary, inst.instance_id)
            end
        end

        # Pop from stack
        node_index = stack[stack_ptr]
        stack_ptr -= Int32(1)

        # Check for level transition
        if node_index == TOP_LEVEL_SENTINEL
            # Return to top level
            node_index = stack[stack_ptr]
            stack_ptr -= Int32(1)
            current_instance = Int32(-1)

            # Restore original ray
            ray_o = ray.o
            ray_d = ray.d
            ray_inv_d = safe_invdir(ray_d)
        end
    end

    # No hit found
    @inbounds dummy_tri = tlas_blas_array[1].primitives[1]
    bary = SVector{3, Float32}(0.0f0, 0.0f0, 0.0f0)
    return (false, dummy_tri, 0.0f0, bary, INVALID_NODE)
end

# ==============================================================================
# Helper Functions
# ==============================================================================

"""Get world-space AABB of a TLAS."""
function world_bound(tlas::TraversableTLAS)::Bounds3
    return tlas.root_aabb
end

"""Get world-space AABB of a BLAS."""
function world_bound(blas::BLAS)::Bounds3
    return blas.root_aabb
end

# ==============================================================================
# Dynamic TLAS Updates
# ==============================================================================

"""
    update_instance_transform!(tlas::TLAS, instance_idx::Integer, transform::Mat4f)

Update the transform of a single instance. Call `refit_tlas!` after updating transforms.

# Arguments
- `tlas`: The TLAS to update
- `instance_idx`: 1-based index of the instance to update
- `transform`: New local-to-world transformation matrix
"""
function update_instance_transform!(tlas::TLAS, instance_idx::Integer, transform::Mat4f)
    inv_transform = Mat4f(inv(transform))

    # Use scalar indexing to update single element directly on GPU
    @allowscalar begin
        old_inst = tlas.instances[instance_idx]
        tlas.instances[instance_idx] = InstanceDescriptor(
            old_inst.blas_index,
            old_inst.instance_id,
            transform,
            inv_transform,
            old_inst.flags
        )
    end
    return nothing
end

"""
    refit_tlas!(tlas::TLAS)

Refit the TLAS after instance transforms have been updated.
Updates leaf AABBs from instance transforms and propagates changes up the tree.

This is much faster than rebuilding the TLAS from scratch when only transforms change.
Operates directly on the backend arrays stored in the TLAS.
"""
function refit_tlas!(tlas::TLAS)
    n = length(tlas.instances)
    n == 0 && return tlas
    backend = tlas.backend

    # Update leaf node AABBs from new transforms (kernel)
    leaf_kernel! = update_tlas_leaf_aabbs_kernel!(backend)
    leaf_kernel!(tlas.nodes, tlas.instances, tlas.blas_array, Int32(n), ndrange=n)
    # Refit internal nodes bottom-up using atomic counters
    if n > 1
        update_flags = KA.zeros(backend, UInt32, n - 1)
        refit_kernel! = refit_tlas_aabbs_kernel!(backend)
        refit_kernel!(tlas.nodes, update_flags, Int32(n), ndrange=n)
    end

    return tlas
end

"""
    update_instance_transforms!(tlas::TLAS, transforms::AbstractVector{Mat4f}, n_to_update::Integer)

Batch update instance transforms. Updates the first `n_to_update` instances
with the provided transforms array.

This is more efficient than calling `update_instance_transform!` in a loop,
especially on GPU where we can parallelize the updates.

The backend is inferred from the `transforms` array type, so for GPU updates
pass a GPU array (e.g., `ROCArray{Mat4f}`).

Call `refit_tlas!(tlas)` after this to update the BVH AABBs.
"""
function update_instance_transforms!(tlas::TLAS, transforms::AbstractVector{Mat4f}, n_to_update::Integer)
    backend = KA.get_backend(transforms)
    kernel! = update_instance_transforms_kernel!(backend)
    kernel!(tlas.instances, transforms, Int32(n_to_update), ndrange=n_to_update)
    KA.synchronize(backend)
    return nothing
end

"""
    update_instance_transforms!(tlas::TLAS, transforms::AbstractVector{Mat4f}, n_to_update::Integer, first_idx::Integer)

Batch update instance transforms starting at a specific index.

This variant updates instances starting at `first_idx` instead of 1, which is
needed for scenes with multiple meshscatter plots where each plot's instances
are at different offsets in the TLAS.

The backend is inferred from the `transforms` array type.
Call `refit_tlas!(tlas)` after this to update the BVH AABBs.
"""
function update_instance_transforms!(tlas::TLAS, transforms::AbstractVector{Mat4f}, n_to_update::Integer, first_idx::Integer)
    backend = KA.get_backend(transforms)
    kernel! = update_instance_transforms_offset_kernel!(backend)
    kernel!(tlas.instances, transforms, Int32(n_to_update), Int32(first_idx), ndrange=n_to_update)
    KA.synchronize(backend)
    return nothing
end


# ==============================================================================
# BVH-Compatible API
# ==============================================================================

"""
    TLAS(primitives::AbstractVector, metadata_fn::Function; backend=KA.CPU())

Universal BVH constructor. Each primitive becomes a BLAS with a single instance.
Drop-in replacement for `BVH(primitives, metadata_fn)`.

Each mesh is automatically treated as an instance at identity transform.
Perfect for scenes where you just have different meshes and want automatic instancing.

GPU-first: Specify backend to build all BLASes directly on GPU.

Example:
```julia
geometries = [cat_mesh, floor, sphere]
tlas = TLAS(geometries, (mesh_idx, tri_idx) -> UInt32(mesh_idx))
# Same API as BVH, but uses instancing internally!

# GPU-first:
tlas = TLAS(geometries, metadata_fn; backend=OpenCLBackend())
```
"""
function TLAS(
    primitives::AbstractVector{P},
    metadata_fn::Function;
    backend = KA.CPU()
) where {P}
    # Each primitive becomes its own BLAS
    first_mesh = Raycore.to_triangle_mesh(first(primitives))
    first_metadata = metadata_fn(1, 1)
    TMetadata = typeof(first_metadata)

    # Build first BLAS on backend
    first_triangles = Triangle{TMetadata}[]
    for i in 1:div(length(first_mesh.indices), 3)
        if !is_degenerate(get_vertices(first_mesh, i))
            metadata = metadata_fn(1, i)
            push!(first_triangles, Triangle(first_mesh, i, metadata))
        end
    end
    backend_triangles = Adapt.adapt(backend, first_triangles)
    first_blas = build_blas(backend_triangles)

    # Create array of BLASes (with backend arrays)
    blas_array = typeof(first_blas)[first_blas]
    instances = InstanceDescriptor[]

    # Add first instance
    identity = Mat4f(I)
    push!(instances, InstanceDescriptor(
        UInt32(1),  # BLAS index (1-based)
        UInt32(1),  # Instance ID = mesh index for metadata
        identity,
        identity,
        UInt32(0)
    ))

    # Process remaining meshes
    for mi in 2:length(primitives)
        prim = primitives[mi]
        triangle_mesh = Raycore.to_triangle_mesh(prim)
        triangles = Triangle{TMetadata}[]

        for i in 1:div(length(triangle_mesh.indices), 3)
            if !is_degenerate(get_vertices(triangle_mesh, i))
                metadata = metadata_fn(mi, i)
                push!(triangles, Triangle(triangle_mesh, i, metadata))
            end
        end

        # Build BLAS on backend
        backend_tris = Adapt.adapt(backend, triangles)
        blas = build_blas(backend_tris)
        push!(blas_array, blas)

        # Create instance at identity
        push!(instances, InstanceDescriptor(
            UInt32(length(blas_array)),
            UInt32(mi),
            identity,
            identity,
            UInt32(0)
        ))
    end

    return build_tlas(blas_array, instances)
end

# Note: TLAS(items::AbstractVector) is defined in the High-Level Instance API section below.
# Use TLAS(meshes, metadata_fn) for per-triangle metadata, or
# TLAS([Instance(mesh1), Instance(mesh2), ...]) for the new Instance API.

# Make closest_hit and any_hit work with both argument orders for compatibility
"""
    closest_hit(ray::AbstractRay, tlas::TLAS)

BVH-compatible argument order for closest_hit.
Returns (hit_found, triangle, distance, barycentric) - same as BVH.
"""
function closest_hit(ray::AbstractRay, tlas::TraversableTLAS)
    hit, tri, t, bary, inst_id = closest_hit(tlas, ray)
    return (hit, tri, t, bary)
end

"""
    any_hit(ray::AbstractRay, tlas::TLAS)

BVH-compatible argument order for any_hit.
Returns (hit_found, triangle, distance, barycentric) - same as BVH.
"""
function any_hit(ray::AbstractRay, tlas::TraversableTLAS)
    hit, tri, t, bary, inst_id = any_hit(tlas, ray)
    return (hit, tri, t, bary)
end

"""
    Base.eltype(tlas::TraversableTLAS)

Get the element type of primitives stored in the TLAS.
Returns the element type of the first BLAS's primitives.
This is needed for compatibility with code that expects `eltype(bvh.primitives)`.

GPU-first: Uses gpu_blas_arrays which contains the backing arrays.
"""
function Base.eltype(tlas::TLAS)
    # gpu_blas_arrays contains pairs of (nodes, prims) for each BLAS
    # The second entry is the primitives array of the first BLAS
    length(tlas.gpu_blas_arrays) < 2 && return Triangle{UInt32}
    prims_array = tlas.gpu_blas_arrays[2]
    return eltype(prims_array)
end

function Base.eltype(::StaticTLAS{NA, IA, BA}) where {NA, IA, BA}
    BLASType = eltype(BA)
    return eltype(fieldtype(BLASType, :primitives))
end

# ==============================================================================
# Convenience TLAS Constructor
# ==============================================================================

"""
    TLAS(items::AbstractVector; backend=KA.CPU()) -> (TLAS, Vector{TLASHandle})

Create a mutable TLAS from a vector of geometries or Instance objects.

Plain geometries are automatically wrapped as single instances at identity transform.
Returns the mutable TLAS and a vector of TLASHandles for later reference.

The TLAS stores backend arrays directly in its fields (nodes, instances, blas_array).
The TLAS must stay alive while any adapted StaticTLAS is in use.

# Examples
```julia
# Mix of plain geometry and instances
tlas, handles = TLAS([
    floor_mesh,                              # Auto-wrapped as Instance
    Instance(sphere, particle_transforms),   # 10k instances sharing sphere BLAS
    Instance(wall, wall_transform, UInt32(2))
])

# Later: update transforms via TLASHandle (uses update! function)
# update!(tlas, handles[2], new_transforms)
refit_tlas!(tlas)
```
"""
function TLAS(items::AbstractVector; backend=KA.CPU())
    isempty(items) && error("Cannot create TLAS from empty item list")

    # Create mutable TLAS with the specified backend
    tlas = TLAS(backend)
    handles = TLASHandle[]

    # Push each item
    for item in items
        inst = item isa Instance ? item : Instance(item)
        h = push!(tlas, inst)
        push!(handles, h)
    end

    # Sync to build the BVH structure
    sync!(tlas)

    return tlas, handles
end

"""
    n_instances(tlas::TraversableTLAS)

Return total number of instance descriptors in the TLAS.
"""
n_instances(tlas::TraversableTLAS) = length(tlas.instances)

"""
    n_geometries(tlas::TraversableTLAS)

Return number of unique BLAS geometries in the TLAS.
"""
n_geometries(tlas::TraversableTLAS) = length(tlas.blas_array)

# Export public API
export BLAS, TLAS, StaticTLAS, TraversableTLAS, InstanceDescriptor, BVHNode2
export build_blas, build_tlas, closest_hit, any_hit, world_bound
export update_instance_transform!, update_instance_transforms!, refit_tlas!
export INVALID_NODE

# Instance API
export Instance, TLASHandle
export n_instances, n_geometries, get_instance, get_instances
export update_transform!, update_transforms!, is_valid
