# ==============================================================================
# Collision Detection for TLAS
# ==============================================================================
#
# Two-pass GPU collision detection using the existing BVH2 structure.
# Follows ImplicitBVH's leaf-vs-tree (LVT) pattern adapted for Raycore's
# two-level TLAS/BLAS with instance transforms.
#
# Architecture:
#   Pass 1: Count contacts per instance (write nothing)
#   Prefix sum: Compute write offsets
#   Pass 2: Write contact pairs at pre-computed offsets
#
# Works on any KernelAbstractions backend (CPU, CUDA, AMDGPU, Lava, etc.)

# ==============================================================================
# Contact types
# ==============================================================================

"""
    ContactPair

A pair of contacting instances in a TLAS, identified by their 1-based instance indices.
"""
struct ContactPair
    instance_a::UInt32
    instance_b::UInt32
end

"""
    CollisionResult{C, B}

Result of collision detection, containing contact pairs and a reusable cache buffer.

Fields:
- `contacts`: Vector of `ContactPair` (or similar) on the compute backend
- `num_contacts`: Number of valid contacts
- `cache`: Internal buffer for reuse across frames via `cache` keyword
"""
struct CollisionResult{C, B}
    contacts::C
    num_contacts::Int
    cache::B
end

# ==============================================================================
# AABB overlap test for BVHNode2
# ==============================================================================

"""Test if two AABBs overlap."""
@inline function aabb_overlaps(a_min::Point3f, a_max::Point3f, b_min::Point3f, b_max::Point3f)
    all(a_max .>= b_min) && all(a_min .<= b_max)
end

"""Get the AABB of a TLAS node (internal: union of children, leaf: stored AABB)."""
@inline function tlas_node_aabb(node::BVHNode2)
    if is_interior(node)
        # Interior: union of children AABBs
        p_min = min.(node.aabb0_min, node.aabb1_min)
        p_max = max.(node.aabb0_max, node.aabb1_max)
        return (p_min, p_max)
    else
        # Leaf: instance AABB stored in aabb0
        return (node.aabb0_min, node.aabb0_max)
    end
end

# ==============================================================================
# Instance-level collision kernel (TLAS broad-phase)
# ==============================================================================

"""
    collide_instances_kernel!

For each TLAS leaf (instance), traverse the TLAS tree to find overlapping instances.
Two-pass: when `contacts` is nothing, only counts. When not nothing, writes pairs.

Uses stack-based depth-first traversal, same pattern as closest_hit but testing
AABB-AABB overlap instead of ray-AABB intersection.
"""
@kernel function collide_instances_kernel!(
    contact_counts,
    contacts,      # Nothing on counting pass, AbstractVector{ContactPair} on writing pass
    @Const(nodes),
    n_instances::Int32
)
    i = @index(Global, Linear)

    @inbounds if i <= n_instances
        # Get this instance's leaf node and AABB
        leaf_idx = Int(n_instances) - 1 + i
        leaf_node = nodes[leaf_idx]
        a_min, a_max = tlas_node_aabb(leaf_node)
        instance_a = leaf_node.child1  # 0-indexed instance index stored in child1

        # Stack for traversal (16 levels is plenty for TLAS)
        stack = MVector{16, UInt32}(undef)
        stack_ptr = Int32(0)

        # Start at root
        node_index = UInt32(1)
        count = UInt32(0)

        while true
            node = nodes[node_index]

            if is_interior(node)
                # Test children AABBs
                overlap0 = aabb_overlaps(a_min, a_max, node.aabb0_min, node.aabb0_max)
                overlap1 = aabb_overlaps(a_min, a_max, node.aabb1_min, node.aabb1_max)

                if overlap0 && overlap1
                    # Both overlap — push far, visit near
                    stack_ptr += Int32(1)
                    stack[stack_ptr] = node.child1
                    node_index = node.child0
                    continue
                elseif overlap0
                    node_index = node.child0
                    continue
                elseif overlap1
                    node_index = node.child1
                    continue
                end
                # Neither overlaps — fall through to pop
            else
                # Leaf node — check if it's a different instance
                instance_b = node.child1
                if instance_b > instance_a  # Only count each pair once (a < b)
                    b_min, b_max = tlas_node_aabb(node)
                    if aabb_overlaps(a_min, a_max, b_min, b_max)
                        count += UInt32(1)
                        if contacts !== nothing
                            # Writing pass: write at pre-computed offset
                            write_idx = contact_counts[i] - count + UInt32(1)
                            contacts[write_idx] = ContactPair(instance_a + UInt32(1), instance_b + UInt32(1))
                        end
                    end
                end
            end

            # Pop from stack
            if stack_ptr > Int32(0)
                node_index = stack[stack_ptr]
                stack_ptr -= Int32(1)
            else
                break
            end
        end

        # Store count (on counting pass, this is the total; on writing pass, it's for offset calc)
        if contacts === nothing
            contact_counts[i] = count
        end
    end
end

# ==============================================================================
# Public API
# ==============================================================================

"""
    collide_instances(tlas::TLAS; cache=nothing) -> CollisionResult

Find all pairs of instances whose world-space AABBs overlap.

This is a broad-phase test — it identifies which instances *might* be in contact
based on their bounding boxes. For exact triangle-triangle contact, use `collide`.

Returns a `CollisionResult` with `ContactPair`s (1-indexed instance IDs).

# Example
```julia
tlas = TLAS(backend)
push!(tlas, mesh_a, transform_a)
push!(tlas, mesh_b, transform_b)
sync!(tlas)

result = collide_instances(tlas)
for i in 1:result.num_contacts
    pair = result.contacts[i]
    println("Instance \$(pair.instance_a) overlaps instance \$(pair.instance_b)")
end

# Reuse buffers for next frame:
result2 = collide_instances(tlas; cache=result.cache)
```
"""
function collide_instances(tlas::TLAS; cache=nothing)
    sync!(tlas)
    n = Int32(length(tlas.instances))
    n == 0 && return CollisionResult(
        KA.allocate(tlas.backend, ContactPair, 0), 0,
        KA.allocate(tlas.backend, UInt32, 0)
    )

    backend = tlas.backend
    nodes = tlas.nodes

    # Allocate or reuse count buffer
    if cache !== nothing && length(cache) >= n
        contact_counts = cache
    else
        contact_counts = KA.allocate(backend, UInt32, Int(n))
    end
    contact_counts .= UInt32(0)

    # Pass 1: Count contacts per instance
    kern! = collide_instances_kernel!(backend)
    kern!(contact_counts, nothing, nodes, n, ndrange=Int(n))
    KA.synchronize(backend)

    # Prefix sum to get write offsets
    # After accumulate, contact_counts[i] = total contacts for instances 1..i
    AK.accumulate!(+, contact_counts, init=UInt32(0))

    # Total contacts
    total = Int(@allowscalar contact_counts[end])
    if total == 0
        return CollisionResult(
            KA.allocate(backend, ContactPair, 0), 0, contact_counts
        )
    end

    # Allocate contacts
    contacts = KA.allocate(backend, ContactPair, total)

    # Pass 2: Write contact pairs
    kern!(contact_counts, contacts, nodes, n, ndrange=Int(n))
    KA.synchronize(backend)

    return CollisionResult(contacts, total, contact_counts)
end

"""
    collide_instances_any(tlas::TLAS, handle_a::TLASHandle, handle_b::TLASHandle) -> Bool

Test whether two specific instance groups overlap (broad-phase AABB test).
Fast early-exit — returns true on first overlap found.
"""
function collide_instances_any(tlas::TLAS, handle_a::TLASHandle, handle_b::TLASHandle)
    sync!(tlas)
    range_a = tlas.handle_to_range[handle_a]
    range_b = tlas.handle_to_range[handle_b]

    # CPU check — for a quick boolean we just test instance AABBs directly
    nodes = Array(tlas.nodes)  # Small download for TLAS nodes
    instances = Array(tlas.instances)
    n = Int32(length(instances))

    for ia in range_a, ib in range_b
        leaf_a = nodes[Int(n) - 1 + ia]
        leaf_b = nodes[Int(n) - 1 + ib]
        a_min, a_max = tlas_node_aabb(leaf_a)
        b_min, b_max = tlas_node_aabb(leaf_b)
        if aabb_overlaps(a_min, a_max, b_min, b_max)
            return true
        end
    end
    return false
end
