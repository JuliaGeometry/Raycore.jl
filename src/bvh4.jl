# ==============================================================================
# BVH4 - 4-Wide Bounding Volume Hierarchy (HIPRT-style optimization)
# ==============================================================================
#
# Key optimizations from AMD HIPRT:
# - 4-wide nodes reduce tree depth by ~50% vs binary BVH
# - Better cache utilization (128-byte aligned nodes)
# - Fewer memory fetches during traversal
# - Built via collapse pass from LBVH binary tree
#
# Architecture:
# 1. Build binary LBVH as before (fast parallel construction)
# 2. Collapse pass converts BVH2 -> BVH4 (parallel GPU kernel)
# 3. Traversal uses 4-wide AABB tests

using StaticArrays
import KernelAbstractions as KA
using KernelAbstractions: @index, @atomicswap
using Atomix

# ==============================================================================
# BVH4 Node Structure
# ==============================================================================

const INVALID_NODE4 = 0xffffffff

"""
    BVHNode4

4-wide BVH node optimized for GPU traversal.
Stores up to 4 children with their AABBs inline.

Memory layout: 128 bytes (one cache line on most GPUs)
- 4 child indices: 16 bytes
- 4 AABBs: 96 bytes (24 bytes each)
- Metadata: 16 bytes

Leaf nodes indicated by child_count == 0 and first primitive index in child0.
"""
struct BVHNode4
    # Child indices (INVALID_NODE4 for unused slots)
    child0::UInt32
    child1::UInt32
    child2::UInt32
    child3::UInt32

    # Child 0 AABB
    aabb0_min::Point3f
    aabb0_max::Point3f

    # Child 1 AABB
    aabb1_min::Point3f
    aabb1_max::Point3f

    # Child 2 AABB
    aabb2_min::Point3f
    aabb2_max::Point3f

    # Child 3 AABB
    aabb3_min::Point3f
    aabb3_max::Point3f

    # Metadata
    parent::UInt32
    child_count::UInt8       # 0 = leaf, 1-4 = interior
    primitive_count::UInt8   # For leaves: number of primitives
    _pad1::UInt8
    _pad2::UInt8
end

# Verify size (should be 128 bytes for cache alignment)
# 4*4 + 4*24 + 4 + 4 = 16 + 96 + 8 = 120 bytes + padding

"""Create an empty/invalid BVH4 node."""
@inline function empty_bvh4_node()
    BVHNode4(
        INVALID_NODE4, INVALID_NODE4, INVALID_NODE4, INVALID_NODE4,
        Point3f(0), Point3f(0),
        Point3f(0), Point3f(0),
        Point3f(0), Point3f(0),
        Point3f(0), Point3f(0),
        INVALID_NODE4, UInt8(0), UInt8(0), UInt8(0), UInt8(0)
    )
end

"""Check if node is a leaf."""
@inline is_leaf4(node::BVHNode4) = node.child_count == 0

"""Check if node is interior."""
@inline is_interior4(node::BVHNode4) = node.child_count > 0

"""Get child index by position (0-3)."""
@inline function get_child4(node::BVHNode4, i::Int)::UInt32
    i == 1 && return node.child0
    i == 2 && return node.child1
    i == 3 && return node.child2
    return node.child3
end

"""Get child AABB by position (1-4)."""
@inline function get_child_aabb4(node::BVHNode4, i::Int)::Bounds3
    if i == 1
        return Bounds3(node.aabb0_min, node.aabb0_max)
    elseif i == 2
        return Bounds3(node.aabb1_min, node.aabb1_max)
    elseif i == 3
        return Bounds3(node.aabb2_min, node.aabb2_max)
    else
        return Bounds3(node.aabb3_min, node.aabb3_max)
    end
end

"""Get the node's total AABB (union of all valid children)."""
@inline function get_node_aabb4(node::BVHNode4)::Bounds3
    aabb = Bounds3(node.aabb0_min, node.aabb0_max)
    if node.child_count >= 2
        aabb = aabb ∪ Bounds3(node.aabb1_min, node.aabb1_max)
    end
    if node.child_count >= 3
        aabb = aabb ∪ Bounds3(node.aabb2_min, node.aabb2_max)
    end
    if node.child_count >= 4
        aabb = aabb ∪ Bounds3(node.aabb3_min, node.aabb3_max)
    end
    return aabb
end

# ==============================================================================
# BVH4 Leaf Node (stores triangle data)
# ==============================================================================

"""
    BVH4Leaf

Leaf node that stores triangle vertices directly (like BVH2IL format).
For BVH4, we keep triangles separate and reference by index.
"""
struct BVH4Leaf
    prim_start::UInt32    # First primitive index
    prim_count::UInt32    # Number of primitives (usually 1-2)
    _pad1::UInt32
    _pad2::UInt32
end

# ==============================================================================
# BLAS4 / TLAS4 Structures
# ==============================================================================

"""
    BLAS4{NodeArray, TriArray}

Bottom-Level Acceleration Structure using BVH4 nodes.
"""
struct BLAS4{
    NodeArray <: AbstractVector{BVHNode4},
    TriArray <: AbstractVector{<:Triangle}
}
    nodes::NodeArray
    primitives::TriArray
    root_aabb::Bounds3
    num_interior::Int32    # Number of interior nodes
end

"""
    TLAS4{NodeArray, InstArray, BLASArray}

Top-Level Acceleration Structure using BVH4 nodes.
"""
struct TLAS4{
    NodeArray <: AbstractVector{BVHNode4},
    InstArray <: AbstractVector{InstanceDescriptor},
    BLASArray <: AbstractVector{<:BLAS4}
}
    nodes::NodeArray
    instances::InstArray
    blas_array::BLASArray
    root_aabb::Bounds3
end

# ==============================================================================
# Collapse Pass: BVH2 -> BVH4
# ==============================================================================

"""
    CollapseTask

Work item for the BVH2 -> BVH4 collapse pass.
"""
struct CollapseTask
    bvh2_node_idx::UInt32    # Source node in BVH2
    bvh4_node_idx::UInt32    # Destination node in BVH4
    depth::UInt32            # Current depth (for load balancing)
end

"""
Gather up to 4 children from a subtree of the binary BVH.
Performs a BFS-like traversal to collect the 4 best children.

Returns: (child_indices, child_aabbs, child_count, is_leaf_flags)
"""
@inline function gather_children_bvh2(
    root_idx::UInt32,
    nodes2::AbstractVector{BVHNode2},
    n_prims::Int32
)
    # Use a small fixed-size work queue
    # Start with root's two children
    queue = MVector{8, UInt32}(undef)
    queue_size = 0

    # Output arrays
    children = MVector{4, UInt32}(INVALID_NODE4, INVALID_NODE4, INVALID_NODE4, INVALID_NODE4)
    aabbs = MVector{4, Bounds3}(Bounds3(), Bounds3(), Bounds3(), Bounds3())
    child_is_leaf = MVector{4, Bool}(false, false, false, false)
    child_count = 0

    # Start with root node
    @inbounds root = nodes2[root_idx]

    if is_leaf(root)
        # Root is already a leaf - single child
        children[1] = root_idx
        aabbs[1] = get_node_aabb(root, false)
        child_is_leaf[1] = true
        return (children, aabbs, Int32(1), child_is_leaf)
    end

    # Add root's children to queue
    queue[1] = root.child0
    queue[2] = root.child1
    queue_size = 2

    # Expand until we have 4 children or can't expand more
    while child_count < 4 && queue_size > 0
        # Find the best node to add (prefer interior nodes to expand)
        best_idx = 1
        for i in 1:queue_size
            @inbounds node_idx = queue[i]
            @inbounds node = nodes2[node_idx]
            # Prefer non-leaves that can be expanded
            if is_interior(node) && child_count + queue_size - 1 + 2 <= 4
                best_idx = i
                break
            end
        end

        # Pop from queue
        @inbounds node_idx = queue[best_idx]
        @inbounds queue[best_idx] = queue[queue_size]
        queue_size -= 1

        @inbounds node = nodes2[node_idx]
        is_node_interior = is_interior(node)

        if is_node_interior && child_count + queue_size + 2 <= 4
            # Expand this node - add its children to queue
            queue_size += 1
            @inbounds queue[queue_size] = node.child0
            queue_size += 1
            @inbounds queue[queue_size] = node.child1
        else
            # Add this node as a child
            child_count += 1
            @inbounds children[child_count] = node_idx
            @inbounds aabbs[child_count] = if is_node_interior
                # Interior node - union of child AABBs
                Bounds3(
                    min.(node.aabb0_min, node.aabb1_min),
                    max.(node.aabb0_max, node.aabb1_max)
                )
            else
                # Leaf node - compute from triangle vertices
                get_node_aabb(node, false)
            end
            @inbounds child_is_leaf[child_count] = !is_node_interior
        end
    end

    # Any remaining queue items become children directly
    while queue_size > 0 && child_count < 4
        child_count += 1
        @inbounds node_idx = queue[queue_size]
        queue_size -= 1

        @inbounds children[child_count] = node_idx
        @inbounds node = nodes2[node_idx]
        is_node_interior = is_interior(node)
        @inbounds aabbs[child_count] = if is_node_interior
            Bounds3(
                min.(node.aabb0_min, node.aabb1_min),
                max.(node.aabb0_max, node.aabb1_max)
            )
        else
            get_node_aabb(node, false)
        end
        @inbounds child_is_leaf[child_count] = !is_node_interior
    end

    return (children, aabbs, Int32(child_count), child_is_leaf)
end

"""
Collapse a BVH2 into a BVH4.

This is a sequential CPU implementation for simplicity.
A GPU version would use work queues similar to HIPRT.

The key insight is that we need to:
1. Collect up to 4 subtrees at each BVH4 interior node
2. For leaf subtrees, create BVH4 leaf nodes pointing to primitives
3. For interior subtrees, recursively process them
4. Fix up child pointers to point to BVH4 indices (not BVH2 indices)
"""
function collapse_bvh2_to_bvh4(
    nodes2::AbstractVector{BVHNode2},
    primitives::AbstractVector{<:Triangle},
    n_prims::Int32
)
    # Estimate BVH4 node count
    max_nodes4 = length(nodes2) + 1
    nodes4 = Vector{BVHNode4}(undef, max_nodes4)
    node4_count = 0

    # Map from BVH2 index to BVH4 index (for subtrees that become BVH4 nodes)
    bvh2_to_bvh4 = Dict{UInt32, UInt32}()

    # Work queue: (bvh2_idx, slot to update in parent, parent bvh4 idx)
    # slot = which child slot (1-4) in the parent to update
    queue = Vector{Tuple{UInt32, Int, UInt32}}()

    # Process root
    @inbounds root = nodes2[1]

    if is_leaf(root)
        # Single triangle - create one leaf node
        node4_count += 1
        prim_idx = root.child1
        v0 = Point3f(root.aabb0_min...)
        v1 = Point3f(root.aabb0_max...)
        v2 = Point3f(root.aabb1_min...)
        p_min = min.(min.(v0, v1), v2)
        p_max = max.(max.(v0, v1), v2)

        nodes4[node4_count] = BVHNode4(
            prim_idx, INVALID_NODE4, INVALID_NODE4, INVALID_NODE4,
            p_min, p_max,
            Point3f(0), Point3f(0),
            Point3f(0), Point3f(0),
            Point3f(0), Point3f(0),
            INVALID_NODE4, UInt8(0), UInt8(1), UInt8(0), UInt8(0)
        )
    else
        # Gather children from root
        children_bvh2, aabbs, child_count, child_is_leaf_flags = gather_children_bvh2(UInt32(1), nodes2, n_prims)

        node4_count += 1
        root4_idx = UInt32(node4_count)

        # Create placeholders for child indices (will be filled in)
        child_indices = MVector{4, UInt32}(INVALID_NODE4, INVALID_NODE4, INVALID_NODE4, INVALID_NODE4)

        # Process each child
        for i in 1:child_count
            bvh2_child_idx = children_bvh2[i]
            @inbounds node2_child = nodes2[bvh2_child_idx]

            if child_is_leaf_flags[i]
                # This is a leaf - create BVH4 leaf node
                node4_count += 1
                leaf4_idx = UInt32(node4_count)
                child_indices[i] = leaf4_idx

                prim_idx = node2_child.child1
                v0 = Point3f(node2_child.aabb0_min...)
                v1 = Point3f(node2_child.aabb0_max...)
                v2 = Point3f(node2_child.aabb1_min...)
                p_min = min.(min.(v0, v1), v2)
                p_max = max.(max.(v0, v1), v2)

                nodes4[node4_count] = BVHNode4(
                    prim_idx, INVALID_NODE4, INVALID_NODE4, INVALID_NODE4,
                    p_min, p_max,
                    Point3f(0), Point3f(0),
                    Point3f(0), Point3f(0),
                    Point3f(0), Point3f(0),
                    root4_idx, UInt8(0), UInt8(1), UInt8(0), UInt8(0)
                )
            else
                # Interior subtree - queue for processing
                push!(queue, (bvh2_child_idx, i, root4_idx))
            end
        end

        # Create root interior node (will update child pointers later for queued children)
        nodes4[1] = BVHNode4(
            child_indices[1], child_indices[2], child_indices[3], child_indices[4],
            aabbs[1].p_min, aabbs[1].p_max,
            aabbs[2].p_min, aabbs[2].p_max,
            aabbs[3].p_min, aabbs[3].p_max,
            aabbs[4].p_min, aabbs[4].p_max,
            INVALID_NODE4, UInt8(child_count), UInt8(0), UInt8(0), UInt8(0)
        )

        # Process queued interior subtrees
        while !isempty(queue)
            bvh2_idx, parent_slot, parent4_idx = popfirst!(queue)

            # Gather children from this subtree
            sub_children, sub_aabbs, sub_count, sub_is_leaf = gather_children_bvh2(bvh2_idx, nodes2, n_prims)

            node4_count += 1
            current4_idx = UInt32(node4_count)

            # Update parent's child pointer
            @inbounds parent = nodes4[parent4_idx]
            if parent_slot == 1
                nodes4[parent4_idx] = BVHNode4(
                    current4_idx, parent.child1, parent.child2, parent.child3,
                    parent.aabb0_min, parent.aabb0_max, parent.aabb1_min, parent.aabb1_max,
                    parent.aabb2_min, parent.aabb2_max, parent.aabb3_min, parent.aabb3_max,
                    parent.parent, parent.child_count, parent.primitive_count, parent._pad1, parent._pad2
                )
            elseif parent_slot == 2
                nodes4[parent4_idx] = BVHNode4(
                    parent.child0, current4_idx, parent.child2, parent.child3,
                    parent.aabb0_min, parent.aabb0_max, parent.aabb1_min, parent.aabb1_max,
                    parent.aabb2_min, parent.aabb2_max, parent.aabb3_min, parent.aabb3_max,
                    parent.parent, parent.child_count, parent.primitive_count, parent._pad1, parent._pad2
                )
            elseif parent_slot == 3
                nodes4[parent4_idx] = BVHNode4(
                    parent.child0, parent.child1, current4_idx, parent.child3,
                    parent.aabb0_min, parent.aabb0_max, parent.aabb1_min, parent.aabb1_max,
                    parent.aabb2_min, parent.aabb2_max, parent.aabb3_min, parent.aabb3_max,
                    parent.parent, parent.child_count, parent.primitive_count, parent._pad1, parent._pad2
                )
            else
                nodes4[parent4_idx] = BVHNode4(
                    parent.child0, parent.child1, parent.child2, current4_idx,
                    parent.aabb0_min, parent.aabb0_max, parent.aabb1_min, parent.aabb1_max,
                    parent.aabb2_min, parent.aabb2_max, parent.aabb3_min, parent.aabb3_max,
                    parent.parent, parent.child_count, parent.primitive_count, parent._pad1, parent._pad2
                )
            end

            # Create child index array
            sub_child_indices = MVector{4, UInt32}(INVALID_NODE4, INVALID_NODE4, INVALID_NODE4, INVALID_NODE4)

            # Process children
            for i in 1:sub_count
                bvh2_child = sub_children[i]
                @inbounds node2_child = nodes2[bvh2_child]

                if sub_is_leaf[i]
                    # Leaf
                    node4_count += 1
                    leaf4_idx = UInt32(node4_count)
                    sub_child_indices[i] = leaf4_idx

                    prim_idx = node2_child.child1
                    v0 = Point3f(node2_child.aabb0_min...)
                    v1 = Point3f(node2_child.aabb0_max...)
                    v2 = Point3f(node2_child.aabb1_min...)
                    p_min = min.(min.(v0, v1), v2)
                    p_max = max.(max.(v0, v1), v2)

                    nodes4[node4_count] = BVHNode4(
                        prim_idx, INVALID_NODE4, INVALID_NODE4, INVALID_NODE4,
                        p_min, p_max,
                        Point3f(0), Point3f(0),
                        Point3f(0), Point3f(0),
                        Point3f(0), Point3f(0),
                        current4_idx, UInt8(0), UInt8(1), UInt8(0), UInt8(0)
                    )
                else
                    # Interior - queue for later
                    push!(queue, (bvh2_child, i, current4_idx))
                end
            end

            # Create this interior node
            nodes4[current4_idx] = BVHNode4(
                sub_child_indices[1], sub_child_indices[2], sub_child_indices[3], sub_child_indices[4],
                sub_aabbs[1].p_min, sub_aabbs[1].p_max,
                sub_aabbs[2].p_min, sub_aabbs[2].p_max,
                sub_aabbs[3].p_min, sub_aabbs[3].p_max,
                sub_aabbs[4].p_min, sub_aabbs[4].p_max,
                parent4_idx, UInt8(sub_count), UInt8(0), UInt8(0), UInt8(0)
            )
        end
    end

    # Resize to actual count
    resize!(nodes4, node4_count)

    return nodes4
end

# ==============================================================================
# BVH4 Build Function
# ==============================================================================

"""
    build_blas4(primitives) -> BLAS4

Build a BLAS using BVH4 nodes for faster traversal.

1. Build standard LBVH (BVH2) using existing kernels
2. Collapse BVH2 -> BVH4
"""
function build_blas4(primitives::AbstractVector{T}) where {T <: Triangle}
    n = length(primitives)
    n == 0 && error("Cannot build BLAS4 from empty primitive list")

    # First build BVH2 using existing infrastructure
    blas2 = build_blas(primitives)

    # Collapse to BVH4
    nodes4 = collapse_bvh2_to_bvh4(blas2.nodes, blas2.primitives, Int32(n))

    return BLAS4(nodes4, blas2.primitives, blas2.root_aabb, Int32(length(nodes4)))
end

# ==============================================================================
# BVH4 Traversal
# ==============================================================================

"""
    fast_intersect_bbox4(ray_o, ray_inv_d, node, child_idx, t_min, t_max) -> (hit, t_entry)

Test ray against one child AABB of a BVH4 node.
"""
@inline function fast_intersect_bbox4(
    ray_o::Point3f,
    ray_inv_d::Vec3f,
    node::BVHNode4,
    child_idx::Int,
    t_min::Float32,
    t_max::Float32
)::Tuple{Bool, Float32}
    aabb = get_child_aabb4(node, child_idx)

    oxinvdir = -ray_o .* ray_inv_d
    f = aabb.p_max .* ray_inv_d .+ oxinvdir
    n = aabb.p_min .* ray_inv_d .+ oxinvdir

    tmax_vec = max.(f, n)
    tmin_vec = min.(f, n)

    max_t = min(minimum(tmax_vec), t_max)
    min_t = max(maximum(tmin_vec), t_min)

    return (min_t <= max_t, min_t)
end

"""
    intersect_all_children4(node, ray_inv_d, ray_o, t_min, t_max) -> sorted hits

Test ray against all 4 children AABBs and return sorted by distance.
Returns up to 4 (child_idx, t_entry) pairs, sorted near-to-far.
"""
@inline function intersect_all_children4(
    node::BVHNode4,
    ray_inv_d::Vec3f,
    ray_o::Point3f,
    t_min::Float32,
    t_max::Float32
)
    # Test all children
    hits = MVector{4, Tuple{UInt32, Float32}}(
        (INVALID_NODE4, Inf32),
        (INVALID_NODE4, Inf32),
        (INVALID_NODE4, Inf32),
        (INVALID_NODE4, Inf32)
    )
    hit_count = 0

    for i in 1:Int(node.child_count)
        child_idx = get_child4(node, i)
        if child_idx != INVALID_NODE4
            hit, t_entry = fast_intersect_bbox4(ray_o, ray_inv_d, node, i, t_min, t_max)
            if hit
                hit_count += 1
                hits[hit_count] = (child_idx, t_entry)
            end
        end
    end

    # Simple insertion sort for up to 4 elements
    for i in 2:hit_count
        j = i
        while j > 1 && hits[j][2] < hits[j-1][2]
            hits[j], hits[j-1] = hits[j-1], hits[j]
            j -= 1
        end
    end

    return hits, hit_count
end

"""
    closest_hit4(blas::BLAS4, ray::AbstractRay) -> (hit, primitive, distance, barycentric)

Traverse BVH4 to find closest intersection.
"""
@inline function closest_hit4(blas::BLAS4, ray::R) where {R <: AbstractRay}
    ray = check_direction(ray)
    ray_o::Point3f = ray.o
    ray_d::Vec3f = ray.d
    ray_mint::Float32 = 0.0f0
    ray_maxt::Float32 = ray.t_max
    ray_inv_d::Vec3f = safe_invdir(ray_d)

    # Stack for traversal (BVH4 needs smaller stack than BVH2)
    stack = MVector{32, UInt32}(undef)
    stack_ptr::Int32 = Int32(0)

    # Track closest hit
    closest_prim::UInt32 = INVALID_NODE4
    hit_u::Float32 = 0.0f0
    hit_v::Float32 = 0.0f0

    nodes = blas.nodes
    prims = blas.primitives

    # Start at root
    node_idx::UInt32 = UInt32(1)

    @inbounds while true
        node = nodes[node_idx]

        if is_interior4(node)
            # Interior node - test all children
            hits, hit_count = intersect_all_children4(node, ray_inv_d, ray_o, ray_mint, ray_maxt)

            # Push far children to stack (in reverse order so nearest is popped first)
            for i in hit_count:-1:2
                if hits[i][1] != INVALID_NODE4
                    stack_ptr += Int32(1)
                    stack[stack_ptr] = hits[i][1]
                end
            end

            # Visit nearest child
            if hit_count > 0 && hits[1][1] != INVALID_NODE4
                node_idx = hits[1][1]
                continue
            end
        else
            # Leaf node - test triangle
            prim_idx = node.child0
            if prim_idx != INVALID_NODE4 && prim_idx <= length(prims)
                tri = prims[prim_idx]
                verts = tri.vertices
                hit, t, u, v = fast_intersect_triangle(
                    ray_o, ray_d,
                    verts[1], verts[2], verts[3],
                    ray_mint, ray_maxt
                )
                if hit
                    ray_maxt = t
                    closest_prim = prim_idx
                    hit_u = u
                    hit_v = v
                end
            end
        end

        # Pop from stack
        if stack_ptr > Int32(0)
            node_idx = stack[stack_ptr]
            stack_ptr -= Int32(1)
        else
            break
        end
    end

    # Return result
    @inbounds if closest_prim != INVALID_NODE4
        tri = prims[closest_prim]
        w = 1.0f0 - hit_u - hit_v
        bary = SVector{3, Float32}(w, hit_u, hit_v)
        return (true, tri, ray_maxt, bary)
    else
        dummy_tri = prims[1]
        bary = SVector{3, Float32}(0.0f0, 0.0f0, 0.0f0)
        return (false, dummy_tri, 0.0f0, bary)
    end
end

"""
    any_hit4(blas::BLAS4, ray::AbstractRay) -> (hit, primitive, distance, barycentric)

Traverse BVH4 to find any intersection (early exit).
"""
@inline function any_hit4(blas::BLAS4, ray::R) where {R <: AbstractRay}
    ray = check_direction(ray)
    ray_o::Point3f = ray.o
    ray_d::Vec3f = ray.d
    ray_mint::Float32 = 0.0f0
    ray_maxt::Float32 = ray.t_max
    ray_inv_d::Vec3f = safe_invdir(ray_d)

    # Stack for traversal
    stack = MVector{32, UInt32}(undef)
    stack_ptr::Int32 = Int32(0)

    nodes = blas.nodes
    prims = blas.primitives

    # Start at root
    node_idx::UInt32 = UInt32(1)

    @inbounds while true
        node = nodes[node_idx]

        if is_interior4(node)
            # Interior node - test all children
            hits, hit_count = intersect_all_children4(node, ray_inv_d, ray_o, ray_mint, ray_maxt)

            # Push all hit children (order doesn't matter for any_hit)
            for i in hit_count:-1:2
                if hits[i][1] != INVALID_NODE4
                    stack_ptr += Int32(1)
                    stack[stack_ptr] = hits[i][1]
                end
            end

            if hit_count > 0 && hits[1][1] != INVALID_NODE4
                node_idx = hits[1][1]
                continue
            end
        else
            # Leaf node - test triangle
            prim_idx = node.child0
            if prim_idx != INVALID_NODE4 && prim_idx <= length(prims)
                tri = prims[prim_idx]
                verts = tri.vertices
                hit, t, u, v = fast_intersect_triangle(
                    ray_o, ray_d,
                    verts[1], verts[2], verts[3],
                    ray_mint, ray_maxt
                )
                if hit
                    # Early exit on first hit
                    w = 1.0f0 - u - v
                    bary = SVector{3, Float32}(w, u, v)
                    return (true, tri, t, bary)
                end
            end
        end

        # Pop from stack
        if stack_ptr > Int32(0)
            node_idx = stack[stack_ptr]
            stack_ptr -= Int32(1)
        else
            break
        end
    end

    # No hit
    dummy_tri = prims[1]
    bary = SVector{3, Float32}(0.0f0, 0.0f0, 0.0f0)
    return (false, dummy_tri, 0.0f0, bary)
end

# ==============================================================================
# Exports
# ==============================================================================

export BVHNode4, BLAS4, TLAS4
export build_blas4, closest_hit4, any_hit4
export is_leaf4, is_interior4, get_child4, get_child_aabb4
