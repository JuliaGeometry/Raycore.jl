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

using StaticArrays
using LinearAlgebra: I
import KernelAbstractions as KA

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

"""
    TLAS{NodeArray, InstArray, BLASArray}

Top-Level Acceleration Structure - BVH over instances.
"""
struct TLAS{
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
    KA.synchronize(backend)

    # Sort by Morton codes (works on GPU arrays directly)
    sorted_indices = sortperm(morton_codes)
    morton_codes_sorted = morton_codes[sorted_indices]
    sorted_prims = primitives[sorted_indices]

    copyto!(morton_codes, morton_codes_sorted)

    # Allocate nodes
    nodes = KA.allocate(backend, BVHNode2, 2*n - 1)
    empty_node = BVHNode2(
        Point3f(0), Point3f(0), Point3f(0), Point3f(0),
        INVALID_NODE, INVALID_NODE, INVALID_NODE
    )
    fill!(nodes, empty_node)

    # Launch kernel: Emit topology
    topo_kernel! = emit_topology_kernel!(backend)
    topo_kernel!(nodes, morton_codes, Int32(n), ndrange=n-1)
    KA.synchronize(backend)

    # Launch kernel: Set parent pointers
    parent_kernel! = set_parent_pointers_kernel!(backend)
    parent_kernel!(nodes, Int32(n), ndrange=n-1)
    KA.synchronize(backend)

    # Launch kernel: Create leaf nodes
    leaf_kernel! = create_leaf_nodes_kernel!(backend)
    leaf_kernel!(nodes, sorted_prims, Int32(n), ndrange=n)
    KA.synchronize(backend)

    # Refit AABBs bottom-up (parallel using atomic counters)
    update_flags = KA.zeros(backend, UInt32, n - 1)  # One flag per internal node
    refit_kernel! = refit_aabbs_kernel!(backend)
    refit_kernel!(nodes, update_flags, Int32(n), ndrange=n)
    KA.synchronize(backend)

    # Compute root AABB - check if root is interior or leaf
    root_is_interior = is_interior(nodes[1])
    root_aabb = get_node_aabb(nodes[1], root_is_interior)

    return BLAS(nodes, sorted_prims, root_aabb)
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
    build_tlas(blas_array::AbstractVector{BLAS}, instances::AbstractVector{InstanceDescriptor}) -> TLAS

Build a Top-Level Acceleration Structure over instances.
Uses LBVH over transformed instance AABBs.

Uses KernelAbstractions for automatic CPU/GPU execution based on input array type.
"""
function build_tlas(
    blas_array::AbstractVector{B},
    instances::AbstractVector{InstanceDescriptor}
) where {B <: BLAS}
    n = length(instances)
    n == 0 && return TLAS(BVHNode2[], instances, blas_array, Bounds3())

    # Infer backend from instances array type
    backend = KA.get_backend(instances)

    # Compute scene AABB from transformed instance bounds
    scene_aabb = Bounds3()
    for inst in instances
        blas = blas_array[inst.blas_index]
        local_aabb = blas.root_aabb

        # Transform all 8 corners and compute world AABB
        world_aabb = Bounds3()
        for c in 1:8
            world_corner = transform_point(inst.transform, corner(local_aabb, c))
            world_aabb = world_aabb ∪ Bounds3(world_corner)
        end

        scene_aabb = scene_aabb ∪ world_aabb
    end

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
    calc_kernel!(morton_codes, instances, blas_array, scene_min, scene_extent, ndrange=n)
    KA.synchronize(backend)

    # Sort by Morton codes
    sorted_indices = sortperm(morton_codes)
    morton_codes_sorted = morton_codes[sorted_indices]
    copyto!(morton_codes, morton_codes_sorted)

    # Allocate nodes
    nodes = KA.allocate(backend, BVHNode2, max(1, 2*n - 1))
    empty_node = BVHNode2(
        Point3f(0), Point3f(0), Point3f(0), Point3f(0),
        INVALID_NODE, INVALID_NODE, INVALID_NODE
    )
    fill!(nodes, empty_node)

    # Single-instance case: trivial TLAS
    if n == 1
        original_idx = sorted_indices[1]
        inst = instances[original_idx]
        blas = blas_array[inst.blas_index]
        local_aabb = blas.root_aabb

        world_aabb = Bounds3()
        for c in 1:8
            world_corner = transform_point(inst.transform, corner(local_aabb, c))
            world_aabb = world_aabb ∪ Bounds3(world_corner)
        end

        nodes[1] = BVHNode2(
            world_aabb.p_min, world_aabb.p_max,
            Point3f(0), Point3f(0),
            INVALID_NODE, UInt32(original_idx - 1),
            INVALID_NODE
        )

        return TLAS(nodes, instances, blas_array, world_aabb)
    end

    # Multi-instance case: build proper LBVH

    # Launch kernel: Emit topology (reuse BLAS topology kernel - same algorithm)
    topo_kernel! = emit_topology_kernel!(backend)
    topo_kernel!(nodes, morton_codes, Int32(n), ndrange=n-1)
    KA.synchronize(backend)

    # Launch kernel: Set parent pointers
    parent_kernel! = set_parent_pointers_kernel!(backend)
    parent_kernel!(nodes, Int32(n), ndrange=n-1)
    KA.synchronize(backend)

    # Launch kernel: Create TLAS leaf nodes (different from BLAS - stores AABBs, not vertices)
    leaf_kernel! = create_tlas_leaf_nodes_kernel!(backend)
    leaf_kernel!(nodes, sorted_indices, instances, blas_array, Int32(n), ndrange=n)
    KA.synchronize(backend)

    # Refit AABBs bottom-up (parallel using atomic counters)
    update_flags = KA.zeros(backend, UInt32, n - 1)
    refit_kernel! = refit_tlas_aabbs_kernel!(backend)
    refit_kernel!(nodes, update_flags, Int32(n), ndrange=n)
    KA.synchronize(backend)

    root_aabb = get_tlas_node_aabb(nodes[1], true)

    return TLAS(nodes, instances, blas_array, root_aabb)
end

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
@inline function closest_hit(tlas::TLAS, ray::R) where {R <: AbstractRay}
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
@inline function any_hit(tlas::TLAS{NA, IA, BA}, ray::R) where {NA, IA, BA, R <: AbstractRay}
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
function world_bound(tlas::TLAS)::Bounds3
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
    old_inst = tlas.instances[instance_idx]
    tlas.instances[instance_idx] = InstanceDescriptor(
        old_inst.blas_index,
        old_inst.instance_id,
        transform,
        inv_transform,
        old_inst.flags
    )
    return nothing
end

"""
    refit_tlas!(tlas::TLAS; backend=nothing)

Refit the TLAS after instance transforms have been updated.
Updates leaf AABBs from instance transforms and propagates changes up the tree.

This is much faster than rebuilding the TLAS from scratch when only transforms change.
Works on both CPU and GPU TLAS.

For GPU TLAS (created via `to_gpu`), you must provide the backend explicitly:
    refit_tlas!(tlas_gpu; backend=ROCBackend())
"""
function refit_tlas!(tlas::TLAS; backend=nothing)
    n = length(tlas.instances)
    n == 0 && return tlas

    # Auto-detect backend for CPU arrays, require explicit for GPU device vectors
    if backend === nothing
        backend = KA.get_backend(tlas.nodes)
    end

    nodes = tlas.nodes
    instances = tlas.instances
    blas_array = tlas.blas_array

    # Update leaf node AABBs from new transforms (GPU kernel)
    leaf_kernel! = update_tlas_leaf_aabbs_kernel!(backend)
    leaf_kernel!(nodes, instances, blas_array, Int32(n), ndrange=n)
    KA.synchronize(backend)

    # Refit internal nodes bottom-up using atomic counters
    if n > 1
        update_flags = KA.zeros(backend, UInt32, n - 1)
        refit_kernel! = refit_tlas_aabbs_kernel!(backend)
        refit_kernel!(nodes, update_flags, Int32(n), ndrange=n)
        KA.synchronize(backend)
    end

    # Update root AABB
    # Note: Can't mutate tlas.root_aabb directly since TLAS is immutable
    # The root_aabb in the struct may be stale, but traversal uses node AABBs

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
    TLAS(primitives::AbstractVector, metadata_fn::Function)

Universal BVH constructor. Each primitive becomes a BLAS with a single instance.
Drop-in replacement for `BVH(primitives, metadata_fn)`.

Each mesh is automatically treated as an instance at identity transform.
Perfect for scenes where you just have different meshes and want automatic instancing.

Example:
```julia
geometries = [cat_mesh, floor, sphere]
tlas = TLAS(geometries, (mesh_idx, tri_idx) -> UInt32(mesh_idx))
# Same API as BVH, but uses instancing internally!
```
"""
function TLAS(
    primitives::AbstractVector{P},
    metadata_fn::Function
) where {P}
    # Each primitive becomes its own BLAS
    first_mesh = Raycore.to_triangle_mesh(first(primitives))
    first_metadata = metadata_fn(1, 1)
    TMetadata = typeof(first_metadata)

    # Build first BLAS to get concrete type
    # Filter out degenerate triangles (e.g., at poles of UV-sphere tessellations)
    first_triangles = Triangle{TMetadata}[]
    for i in 1:div(length(first_mesh.indices), 3)
        if !is_degenerate(get_vertices(first_mesh, i))
            metadata = metadata_fn(1, i)
            push!(first_triangles, Triangle(first_mesh, i, metadata))
        end
    end
    first_blas = build_blas(first_triangles)

    # Create concretely-typed arrays using first element's type
    BLASType = typeof(first_blas)
    blas_array = BLASType[first_blas]
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

        # Filter out degenerate triangles
        for i in 1:div(length(triangle_mesh.indices), 3)
            if !is_degenerate(get_vertices(triangle_mesh, i))
                metadata = metadata_fn(mi, i)
                push!(triangles, Triangle(triangle_mesh, i, metadata))
            end
        end

        # Build BLAS for this mesh
        blas = build_blas(triangles)
        push!(blas_array, blas)

        # Create instance at identity (mesh already positioned)
        push!(instances, InstanceDescriptor(
            UInt32(length(blas_array)),  # BLAS index
            UInt32(mi),                   # Instance ID = mesh index for metadata
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
function closest_hit(ray::AbstractRay, tlas::TLAS)
    hit, tri, t, bary, inst_id = closest_hit(tlas, ray)
    return (hit, tri, t, bary)
end

"""
    any_hit(ray::AbstractRay, tlas::TLAS)

BVH-compatible argument order for any_hit.
Returns (hit_found, triangle, distance, barycentric) - same as BVH.
"""
function any_hit(ray::AbstractRay, tlas::TLAS)
    hit, tri, t, bary, inst_id = any_hit(tlas, ray)
    return (hit, tri, t, bary)
end

"""
    Base.eltype(tlas::TLAS)

Get the element type of primitives stored in the TLAS.
Returns the element type of the first BLAS's primitives.
This is needed for compatibility with code that expects `eltype(bvh.primitives)`.

Uses type parameters to avoid indexing into GPU arrays from CPU.
"""
function Base.eltype(::TLAS{NA, IA, BA}) where {NA, IA, BA}
    # Extract BLAS type from the BLASArray type parameter
    # BA <: AbstractVector{<:BLAS{NodeArray, TriArray}}
    BLASType = eltype(BA)
    # BLAS{NodeArray, TriArray} -> TriArray -> eltype(TriArray)
    return eltype(fieldtype(BLASType, :primitives))
end

# ==============================================================================
# High-Level Instance API
# ==============================================================================

"""
    Instance{G, T, M}

User-friendly wrapper for instanced geometry.

Each Instance represents one or more copies of a geometry with different transforms.
Multiple transforms = multiple instances sharing the same BLAS.

# Fields
- `geometry::G`: The source geometry (mesh, etc.)
- `transforms::T`: Per-instance local-to-world transforms (Vector or GPU array)
- `metadata::M`: Per-instance metadata (Vector or GPU array, e.g., material indices)

# Examples
```julia
# Single instance at identity
inst = Instance(mesh)

# Single instance with transform and metadata
inst = Instance(mesh, my_transform, UInt32(1))

# Multiple instances (10k particles sharing one sphere mesh)
transforms = [translation(pos) * scale(r) for (pos, r) in particles]
metadata = fill(UInt32(1), length(transforms))  # All use material 1
inst = Instance(sphere_mesh, transforms, metadata)

# GPU arrays work too
transforms_gpu = ROCArray(transforms)
metadata_gpu = ROCArray(metadata)
inst_gpu = Instance(sphere_mesh, transforms_gpu, metadata_gpu)
```
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
    Instance(geom, [Mat4f(LinearAlgebra.I)], [metadata])

"""Multiple instances with same metadata."""
function Instance(geom, transforms::AbstractVector{Mat4f}; metadata=UInt32(1))
    Instance(geom, transforms, fill(metadata, length(transforms)))
end

"""
    InstanceHandle

Stable handle for referencing instances in a TLAS.

The handle stores the BLAS index which is stable across add/remove operations.
Use `find_instances(tlas, handle)` to get the current instance range.

# Fields
- `blas_index`: Which BLAS geometry this instance uses (stable identifier)
"""
struct InstanceHandle
    blas_index::UInt32
end

"""
    find_instances(tlas::TLAS, handle::InstanceHandle) -> UnitRange{Int}

Find the current instance range for a handle by searching for matching blas_index.
Returns the range of indices in `tlas.instances` that use this BLAS.

This is O(n) but handles are stable across add/remove operations.
"""
function find_instances(tlas::TLAS, handle::InstanceHandle)::UnitRange{Int}
    start_idx = 0
    end_idx = 0
    for (i, inst) in enumerate(tlas.instances)
        if inst.blas_index == handle.blas_index
            if start_idx == 0
                start_idx = i
            end
            end_idx = i
        end
    end
    start_idx == 0 && error("Handle not found in TLAS (blas_index=$(handle.blas_index))")
    return start_idx:end_idx
end

Base.length(tlas::TLAS, h::InstanceHandle) = length(find_instances(tlas, h))

"""
    TLAS(items::AbstractVector) -> TLAS

Create a TLAS from a vector of geometries or Instance objects.

Plain geometries are automatically wrapped as single instances at identity transform.
Returns the TLAS and a vector of InstanceHandles for later reference.

# Examples
```julia
# Mix of plain geometry and instances
tlas, handles = TLAS([
    floor_mesh,                              # Auto-wrapped as Instance
    Instance(sphere, particle_transforms),   # 10k instances sharing sphere BLAS
    Instance(wall, wall_transform, UInt32(2))
])

# Later: update transforms (fast, uses GPU kernel)
update_transforms!(tlas, handles[2], new_particle_transforms)
refit_tlas!(tlas)

# Add/remove instances (requires rebuild)
new_handle = add_instance!(tlas, Instance(new_mesh, transforms))
remove_instance!(tlas, handles[1])
rebuild_tlas!(tlas)
```
"""
function TLAS(items::AbstractVector)
    # Convert items to Instances
    instances = Instance[]
    for item in items
        if item isa Instance
            push!(instances, item)
        else
            # Wrap plain geometry as single instance at identity
            push!(instances, Instance(item))
        end
    end

    return _build_tlas_from_instances(instances)
end

"""
    _build_tlas_from_instances(instances::Vector{<:Instance}) -> (TLAS, Vector{InstanceHandle})

Internal: Build TLAS from Instance objects.
"""
function _build_tlas_from_instances(instances::Vector{<:Instance})
    isempty(instances) && error("Cannot create TLAS from empty instance list")

    # Build first BLAS to get concrete type for the array
    first_inst = instances[1]
    first_mesh = to_triangle_mesh(first_inst.geometry)
    # Filter out degenerate triangles (e.g., at poles of UV-sphere tessellations)
    first_triangles = [Triangle(first_mesh, i, first_inst.metadata[1])
                       for i in 1:div(length(first_mesh.indices), 3)
                       if !is_degenerate(get_vertices(first_mesh, i))]
    first_blas = build_blas(first_triangles)

    # Create concretely-typed arrays
    blas_array = typeof(first_blas)[first_blas]
    instance_descriptors = InstanceDescriptor[]
    handles = InstanceHandle[]

    # Add first instance's descriptors
    blas_idx = UInt32(1)
    for (i, transform) in enumerate(first_inst.transforms)
        inv_transform = Mat4f(inv(transform))
        instance_id = UInt32(length(instance_descriptors) + 1)
        desc = InstanceDescriptor(blas_idx, instance_id, transform, inv_transform, UInt32(0))
        push!(instance_descriptors, desc)
    end
    push!(handles, InstanceHandle(blas_idx))

    # Process remaining instances
    for inst in instances[2:end]
        mesh = to_triangle_mesh(inst.geometry)
        # Filter out degenerate triangles
        triangles = [Triangle(mesh, i, inst.metadata[1])
                     for i in 1:div(length(mesh.indices), 3)
                     if !is_degenerate(get_vertices(mesh, i))]
        blas = build_blas(triangles)
        push!(blas_array, blas)
        blas_idx = UInt32(length(blas_array))

        # Create InstanceDescriptors for each transform
        for (i, transform) in enumerate(inst.transforms)
            inv_transform = Mat4f(inv(transform))
            instance_id = UInt32(length(instance_descriptors) + 1)
            desc = InstanceDescriptor(blas_idx, instance_id, transform, inv_transform, UInt32(0))
            push!(instance_descriptors, desc)
        end

        push!(handles, InstanceHandle(blas_idx))
    end

    # Build TLAS
    tlas = build_tlas(blas_array, instance_descriptors)

    return tlas, handles
end

"""
    add_instance!(tlas::TLAS, inst::Instance) -> InstanceHandle

Add a new instance to the TLAS. Returns a stable handle for later reference.

**Note:** Call `rebuild_tlas!(tlas)` after adding instances to update the BVH structure.
Uses `append!` internally which works on both CPU and GPU arrays.
"""
function add_instance!(tlas::TLAS, inst::Instance)
    # Build BLAS from geometry
    mesh = to_triangle_mesh(inst.geometry)
    triangles = [Triangle(mesh, i, inst.metadata[1]) for i in 1:div(length(mesh.indices), 3)]
    blas = build_blas(triangles)

    # Append to blas_array
    push!(tlas.blas_array, blas)
    blas_idx = UInt32(length(tlas.blas_array))

    # Create InstanceDescriptors and append
    # Note: instance_id is sequential; material info is stored in Triangle metadata
    new_descriptors = InstanceDescriptor[]
    base_id = UInt32(length(tlas.instances))
    for (i, transform) in enumerate(inst.transforms)
        inv_transform = Mat4f(inv(transform))
        instance_id = base_id + UInt32(i)
        desc = InstanceDescriptor(
            blas_idx,
            instance_id,
            transform,
            inv_transform,
            UInt32(0)
        )
        push!(new_descriptors, desc)
    end
    append!(tlas.instances, new_descriptors)

    return InstanceHandle(blas_idx)
end

"""
    remove_instance!(tlas::TLAS, handle::InstanceHandle)

Mark an instance as inactive (sets flags to indicate removal).

**Note:** Call `rebuild_tlas!(tlas)` after removing instances to actually compact
the arrays and update the BVH structure.

For performance, consider batching removals before calling rebuild.
"""
function remove_instance!(tlas::TLAS, handle::InstanceHandle)
    # Find current instance range and mark as inactive
    instance_range = find_instances(tlas, handle)
    for idx in instance_range
        old_inst = tlas.instances[idx]
        # Use flags field to mark as removed (0xFFFFFFFF = removed)
        tlas.instances[idx] = InstanceDescriptor(
            old_inst.blas_index,
            old_inst.instance_id,
            old_inst.transform,
            old_inst.inv_transform,
            UInt32(0xFFFFFFFF)  # Mark as removed
        )
    end
    return nothing
end

"""
    rebuild_tlas!(tlas::TLAS)

Rebuild the TLAS structure after add/remove operations.

This compacts the instance array (removing marked-as-deleted instances)
and rebuilds the BVH topology. Relatively expensive (~1ms for 10k instances).

For transform-only updates, use `refit_tlas!` instead which is much faster.
"""
function rebuild_tlas!(tlas::TLAS)
    # Filter out removed instances (flags == 0xFFFFFFFF)
    active_instances = filter(inst -> inst.flags != UInt32(0xFFFFFFFF), tlas.instances)

    if isempty(active_instances)
        # Empty TLAS - just clear
        resize!(tlas.instances, 0)
        resize!(tlas.nodes, 0)
        return tlas
    end

    # Rebuild from active instances
    new_tlas = build_tlas(tlas.blas_array, active_instances)

    # Update in place using resize! and copyto!
    resize!(tlas.instances, length(new_tlas.instances))
    copyto!(tlas.instances, new_tlas.instances)

    resize!(tlas.nodes, length(new_tlas.nodes))
    copyto!(tlas.nodes, new_tlas.nodes)

    return tlas
end

"""
    n_instances(tlas::TLAS)

Return total number of instance descriptors in the TLAS.
"""
n_instances(tlas::TLAS) = length(tlas.instances)

"""
    n_geometries(tlas::TLAS)

Return number of unique BLAS geometries in the TLAS.
"""
n_geometries(tlas::TLAS) = length(tlas.blas_array)

"""
    update_transform!(tlas::TLAS, handle::InstanceHandle, transform::Mat4f)

Update the transform for a single-instance handle (e.g., a Makie plot's model matrix).

This is a convenience function for the common case where a handle has exactly one instance.
For multi-instance handles (like MeshScatter), use `update_transforms!` instead.

After updating, call `refit_tlas!(tlas)` to update the BVH AABBs.
"""
function update_transform!(tlas::TLAS, handle::InstanceHandle, transform::Mat4f)
    instance_range = find_instances(tlas, handle)
    @assert length(instance_range) == 1 "update_transform! is for single-instance handles; use update_transforms! for multi-instance"

    idx = first(instance_range)
    old_inst = tlas.instances[idx]
    inv_transform = Mat4f(inv(transform))
    tlas.instances[idx] = InstanceDescriptor(
        old_inst.blas_index,
        old_inst.instance_id,
        transform,
        inv_transform,
        old_inst.flags
    )
    return nothing
end

"""
    update_transforms!(tlas::TLAS, handle::InstanceHandle, transforms::AbstractVector{Mat4f})

Update transforms for all instances in a handle (e.g., MeshScatter positions).

The number of transforms must match the number of instances in the handle.
After updating, call `refit_tlas!(tlas)` to update the BVH AABBs.
"""
function update_transforms!(tlas::TLAS, handle::InstanceHandle, transforms::AbstractVector{Mat4f})
    instance_range = find_instances(tlas, handle)
    @assert length(transforms) == length(instance_range) "Transform count must match instance count"

    for (i, idx) in enumerate(instance_range)
        old_inst = tlas.instances[idx]
        transform = transforms[i]
        inv_transform = Mat4f(inv(transform))
        tlas.instances[idx] = InstanceDescriptor(
            old_inst.blas_index,
            old_inst.instance_id,
            transform,
            inv_transform,
            old_inst.flags
        )
    end
    return nothing
end

# Export public API
export BLAS, TLAS, InstanceDescriptor, BVHNode2
export build_blas, build_tlas, closest_hit, any_hit, world_bound
export update_instance_transform!, update_instance_transforms!, refit_tlas!
export INVALID_NODE

# New Instance API
export Instance, InstanceHandle
export add_instance!, remove_instance!, rebuild_tlas!
export update_transform!, update_transforms!, find_instances
export n_instances, n_geometries
