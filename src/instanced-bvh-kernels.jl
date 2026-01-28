# ==============================================================================
# Instanced BVH - GPU Kernels with KernelAbstractions
# ==============================================================================
#
# Follows KA best practices:
# - @kernel only for parallel dispatch
# - All logic in regular type-stable Julia functions
# - Minimal code inside @kernel functions

import KernelAbstractions as KA
using KernelAbstractions: @index
using Atomix: @atomicswap

# ==============================================================================
# GPU Kernel 0: Fill arrays (workaround for OpenCL fill! struct issue)
# ==============================================================================

"""GPU kernel: Fill array with a value (workaround for OpenCL's fill! not supporting structs)."""
KA.@kernel function fill_bvhnode2_kernel!(arr, val)
    i = @index(Global, Linear)
    @inbounds arr[i] = val
end

"""GPU kernel: Fill array with sequential indices [1, 2, 3, ..., n]."""
KA.@kernel function iota_kernel!(arr)
    i = @index(Global, Linear)
    @inbounds arr[i] = i
end

# ==============================================================================
# GPU Kernel 0b: Compute Instance World AABBs
# ==============================================================================

"""
Compute world AABB for a single instance by transforming local AABB corners.
Returns (min_point, max_point) as two Point3f values.
"""
@inline function compute_instance_world_aabb(
    inst::InstanceDescriptor,
    blas_array::AbstractVector{<:BLAS}
)
    blas = blas_array[inst.blas_index]
    local_aabb = blas.root_aabb

    # Initialize with first corner
    corner1 = transform_point(inst.transform, corner(local_aabb, 1))
    min_p = corner1
    max_p = corner1

    # Expand to include all 8 corners
    for c in 2:8
        world_corner = transform_point(inst.transform, corner(local_aabb, c))
        min_p = Point3f(min(min_p[1], world_corner[1]),
                        min(min_p[2], world_corner[2]),
                        min(min_p[3], world_corner[3]))
        max_p = Point3f(max(max_p[1], world_corner[1]),
                        max(max_p[2], world_corner[2]),
                        max(max_p[3], world_corner[3]))
    end

    return (min_p, max_p)
end

"""GPU kernel: Compute world AABBs for all instances, storing min/max points separately."""
KA.@kernel function compute_instance_aabbs_kernel!(
    aabb_mins::AbstractVector{Point3f},
    aabb_maxs::AbstractVector{Point3f},
    @Const(instances),
    @Const(blas_array)
)
    i = @index(Global, Linear)
    @inbounds begin
        inst = instances[i]
        min_p, max_p = compute_instance_world_aabb(inst, blas_array)
        aabb_mins[i] = min_p
        aabb_maxs[i] = max_p
    end
end

# ==============================================================================
# GPU Kernel 1: Calculate Morton Codes
# ==============================================================================

"""
Calculate Morton code for a single primitive.
This is a regular Julia function, callable from CPU or GPU.
"""
@inline function calculate_morton_code_for_prim(
    prim_idx::Int,
    primitives::AbstractVector{<:Triangle},
    scene_min::Point3f,
    scene_extent::Vec3f
)::UInt32
    tri_aabb = world_bound(primitives[prim_idx])
    centroid = 0.5f0 * (tri_aabb.p_min + tri_aabb.p_max)
    normalized = (centroid - scene_min) ./ scene_extent
    return morton_code_30bit(normalized)
end

"""GPU kernel: Parallel dispatch for Morton code calculation."""
KA.@kernel function calculate_morton_codes_kernel!(
    morton_codes,
    primitives,
    scene_min,
    scene_extent
)
    i = @index(Global, Linear)
    @inbounds morton_codes[i] = calculate_morton_code_for_prim(i, primitives, scene_min, scene_extent)
end

# ==============================================================================
# GPU Kernel 2: Emit Topology
# ==============================================================================

"""
Build topology for one internal node.
Regular Julia function for testability.
"""
@inline function build_topology_for_node(
    idx::Int32,
    morton_codes::AbstractVector{UInt32},
    n_prims::Int32
)::BVHNode2
    # Helper functions
    @inline leaf_idx(j::Int32) = n_prims - Int32(1) + j

    # Find span
    span_left, span_right = find_span_for_node(idx, morton_codes, n_prims)

    # Find split
    split = find_split_in_span(span_left, span_right, morton_codes, n_prims)

    # Determine children
    # If split is at boundary, it's a leaf. Otherwise it's a valid internal node.
    child0 = (split == span_left) ? leaf_idx(split) : split
    child1_idx = split + Int32(1)
    child1 = (child1_idx == span_right) ? leaf_idx(child1_idx) : child1_idx

    return BVHNode2(
        Point3f(0), Point3f(0), Point3f(0), Point3f(0),
        UInt32(child0), UInt32(child1), INVALID_NODE
    )
end

"""GPU kernel: Parallel topology emission."""
KA.@kernel function emit_topology_kernel!(nodes, morton_codes, n_prims::Int32)
    i = @index(Global, Linear)
    idx = Int32(i)
    if idx < n_prims
        @inbounds nodes[idx] = build_topology_for_node(idx, morton_codes, n_prims)
    end
end

# ==============================================================================
# GPU Kernel 3: Set Parent Pointers
# ==============================================================================

"""Set parent pointers for one node's children. Regular Julia function."""
@inline function set_parents_for_node(
    node_idx::Int32,
    nodes::AbstractVector{BVHNode2}
)::Tuple{Int32, BVHNode2, Int32, BVHNode2}
    @inbounds node = nodes[node_idx]
    child0_idx = Int32(node.child0)
    child1_idx = Int32(node.child1)

    @inbounds old0 = nodes[child0_idx]
    @inbounds old1 = nodes[child1_idx]

    new0 = BVHNode2(
        old0.aabb0_min, old0.aabb0_max, old0.aabb1_min, old0.aabb1_max,
        old0.child0, old0.child1, UInt32(node_idx)
    )
    new1 = BVHNode2(
        old1.aabb0_min, old1.aabb0_max, old1.aabb1_min, old1.aabb1_max,
        old1.child0, old1.child1, UInt32(node_idx)
    )

    return (child0_idx, new0, child1_idx, new1)
end

"""GPU kernel: Parallel parent pointer assignment."""
KA.@kernel function set_parent_pointers_kernel!(nodes, n_prims::Int32)
    i = @index(Global, Linear)
    idx = Int32(i)
    if idx < n_prims
        child0_idx, new0, child1_idx, new1 = set_parents_for_node(idx, nodes)
        @inbounds nodes[child0_idx] = new0
        @inbounds nodes[child1_idx] = new1
    end
end

# ==============================================================================
# GPU Kernel 4: Create Leaf Nodes
# ==============================================================================

"""Create leaf node for one primitive. Regular Julia function."""
@inline function create_leaf_for_prim(
    prim_idx::Int,
    primitives::AbstractVector{<:Triangle},
    parent_node::BVHNode2,
    n_prims::Int32
)::BVHNode2
    # Store triangle vertices directly in leaf node (BVH2IL format)
    tri = primitives[prim_idx]
    verts = tri.vertices
    v0 = verts[1]
    v1 = verts[2]
    v2 = verts[3]

    return BVHNode2(
        v0, v1, v2, Point3f(0),  # Store v0, v1, v2 in aabb slots
        INVALID_NODE, UInt32(prim_idx), parent_node.parent
    )
end

"""GPU kernel: Parallel leaf node creation."""
KA.@kernel function create_leaf_nodes_kernel!(nodes, primitives, n_prims::Int32)
    i = @index(Global, Linear)
    if i <= n_prims
        @inline leaf_idx(j::Int) = Int(n_prims) - 1 + j
        leaf_node_idx = leaf_idx(i)
        @inbounds parent_node = nodes[leaf_node_idx]
        @inbounds nodes[leaf_node_idx] = create_leaf_for_prim(i, primitives, parent_node, n_prims)
    end
end

# ==============================================================================
# Refit AABBs Kernel (Parallel Bottom-Up)
# ==============================================================================

"""
Parallel bottom-up AABB refit using atomic counters.

Each thread starts at a leaf and walks up the tree. Uses atomic operations
to ensure each internal node is updated exactly once after both children are ready.
Based on RadeonRays Refit kernel.
"""
KA.@kernel function refit_aabbs_kernel!(
    nodes,
    update_flags,
    n_prims::Int32
)
    prim_idx = @index(Global, Linear)

    # Start at parent of this leaf
    leaf_idx = leaf_index(prim_idx, n_prims)
    @inbounds parent_idx = nodes[leaf_idx].parent

    # Walk up the tree
    while parent_idx != INVALID_NODE
        # Atomic exchange: mark this node as visited
        # If old_value == 0: we're first thread, bail out
        # If old_value == 1: we're second thread, update AABB and continue
        old_value = @inbounds @atomicswap update_flags[parent_idx] = UInt32(1)

        if old_value == UInt32(1)
            # Second thread arrived - compute AABB from both children
            @inbounds begin
                node = nodes[parent_idx]
                child0 = node.child0
                child1 = node.child1

                is_child0_internal = child0 < n_prims
                is_child1_internal = child1 < n_prims

                aabb0 = get_node_aabb(nodes[child0], is_child0_internal)
                aabb1 = get_node_aabb(nodes[child1], is_child1_internal)

                # Update this node's AABBs
                updated_node = BVHNode2(
                    aabb0.p_min, aabb0.p_max,
                    aabb1.p_min, aabb1.p_max,
                    node.child0, node.child1, node.parent
                )
                nodes[parent_idx] = updated_node

                # Move to parent
                parent_idx = node.parent
            end
        else
            # First thread - bail out
            break
        end
    end
end

# ==============================================================================
# TLAS-Specific Kernels
# ==============================================================================

"""
Calculate Morton code for a single instance centroid.
"""
@inline function calculate_tlas_morton_code(
    inst_idx::Int,
    instances::AbstractVector{InstanceDescriptor},
    blas_array::AbstractVector{<:BLAS},
    scene_min::Point3f,
    scene_extent::Vec3f
)::UInt32
    inst = instances[inst_idx]
    blas = blas_array[inst.blas_index]
    local_aabb = blas.root_aabb

    # Transform centroid to world space
    local_center = 0.5f0 * (local_aabb.p_min + local_aabb.p_max)
    world_center = transform_point(inst.transform, local_center)

    # Normalize and compute Morton code
    normalized = (world_center - scene_min) ./ scene_extent
    return morton_code_30bit(normalized)
end

"""GPU kernel: Calculate Morton codes for TLAS instances."""
KA.@kernel function calculate_tlas_morton_codes_kernel!(
    morton_codes,
    @Const(instances),
    @Const(blas_array),
    scene_min,
    scene_extent
)
    i = @index(Global, Linear)
    @inbounds morton_codes[i] = calculate_tlas_morton_code(
        i, instances, blas_array, scene_min, scene_extent
    )
end

"""
Create TLAS leaf node for one instance (stores world-space AABB, not triangle vertices).
"""
@inline function create_tlas_leaf_for_instance(
    sorted_leaf_idx::Int,
    sorted_indices::AbstractVector{<:Integer},
    instances::AbstractVector{InstanceDescriptor},
    blas_array::AbstractVector{<:BLAS},
    parent::UInt32
)::BVHNode2
    # Get original instance index (sorted_indices maps sorted position -> original position)
    original_idx = sorted_indices[sorted_leaf_idx]
    inst = instances[original_idx]
    blas = blas_array[inst.blas_index]
    local_aabb = blas.root_aabb

    # Transform AABB to world space (8 corners)
    world_aabb = Bounds3()
    for c in 1:8
        world_corner = transform_point(inst.transform, corner(local_aabb, c))
        world_aabb = world_aabb ∪ Bounds3(world_corner)
    end

    return BVHNode2(
        world_aabb.p_min, world_aabb.p_max, Point3f(0), Point3f(0),
        INVALID_NODE, UInt32(original_idx - 1),  # 0-indexed instance index
        parent
    )
end

"""GPU kernel: Create TLAS leaf nodes."""
KA.@kernel function create_tlas_leaf_nodes_kernel!(
    nodes,
    @Const(sorted_indices),
    @Const(instances),
    @Const(blas_array),
    n_instances::Int32
)
    i = @index(Global, Linear)
    if i <= n_instances
        leaf_node_idx = Int(n_instances) - 1 + i
        @inbounds parent = nodes[leaf_node_idx].parent
        @inbounds nodes[leaf_node_idx] = create_tlas_leaf_for_instance(
            i, sorted_indices, instances, blas_array, parent
        )
    end
end

"""
Parallel bottom-up AABB refit for TLAS using atomic counters.
Uses get_tlas_node_aabb which treats leaves as storing AABBs directly (not triangle vertices).
"""
KA.@kernel function refit_tlas_aabbs_kernel!(
    nodes,
    update_flags,
    n_instances::Int32
)
    inst_idx = @index(Global, Linear)

    # Start at parent of this leaf
    leaf_idx = Int(n_instances) - 1 + inst_idx
    @inbounds parent_idx = nodes[leaf_idx].parent

    # Walk up the tree
    while parent_idx != INVALID_NODE
        # Atomic exchange: mark this node as visited
        old_value = @inbounds @atomicswap update_flags[parent_idx] = UInt32(1)

        if old_value == UInt32(1)
            # Second thread arrived - compute AABB from both children
            @inbounds begin
                node = nodes[parent_idx]
                child0 = node.child0
                child1 = node.child1

                is_child0_internal = child0 < n_instances
                is_child1_internal = child1 < n_instances

                # Use TLAS-specific AABB computation (leaves store AABBs, not vertices)
                aabb0 = get_tlas_node_aabb(nodes[child0], is_child0_internal)
                aabb1 = get_tlas_node_aabb(nodes[child1], is_child1_internal)

                # Update this node's AABBs
                updated_node = BVHNode2(
                    aabb0.p_min, aabb0.p_max,
                    aabb1.p_min, aabb1.p_max,
                    node.child0, node.child1, node.parent
                )
                nodes[parent_idx] = updated_node

                # Move to parent
                parent_idx = node.parent
            end
        else
            # First thread - bail out
            break
        end
    end
end

# ==============================================================================
# GPU Dynamic Update Kernels
# ==============================================================================

"""
GPU kernel: Batch update instance transforms.

Updates both transform and inv_transform for each instance from the provided
transform array. The transforms array should contain the new local-to-world
transforms for instances 1:n (in particle order, NOT Morton order).
"""
KA.@kernel function update_instance_transforms_kernel!(
    instances,
    @Const(transforms),
    n_particles::Int32
)
    i = @index(Global, Linear)
    if i <= n_particles
        @inbounds begin
            old_inst = instances[i]
            transform = transforms[i]
            # Compute inverse transform
            inv_transform = Mat4f(inv(transform))
            # Update instance (preserve blas_index, instance_id, flags)
            instances[i] = InstanceDescriptor(
                old_inst.blas_index,
                old_inst.instance_id,
                transform,
                inv_transform,
                old_inst.flags
            )
        end
    end
end

"""
GPU kernel: Update instance transforms with offset.

Same as update_instance_transforms_kernel! but updates instances starting at
`first_idx` instead of 1. This is needed for TLAS with multiple meshscatter
plots where each plot's instances are at different offsets.
"""
KA.@kernel function update_instance_transforms_offset_kernel!(
    instances,
    @Const(transforms),
    n_particles::Int32,
    first_idx::Int32
)
    i = @index(Global, Linear)
    if i <= n_particles
        @inbounds begin
            inst_idx = first_idx + i - Int32(1)
            old_inst = instances[inst_idx]
            transform = transforms[i]
            # Compute inverse transform
            inv_transform = Mat4f(inv(transform))
            # Update instance (preserve blas_index, instance_id, flags)
            instances[inst_idx] = InstanceDescriptor(
                old_inst.blas_index,
                old_inst.instance_id,
                transform,
                inv_transform,
                old_inst.flags
            )
        end
    end
end

"""
GPU kernel: Update TLAS leaf node AABBs from instance transforms.

After transforms are updated, this kernel recomputes world-space AABBs for
all leaf nodes. Must be called before refit_tlas_aabbs_kernel!.

NOTE: Leaf nodes are Morton-sorted, so we must use the stored instance index
(child1) to look up the correct instance.
"""
KA.@kernel function update_tlas_leaf_aabbs_kernel!(
    nodes,
    @Const(instances),
    @Const(blas_array),
    n_instances::Int32
)
    i = @index(Global, Linear)
    if i <= n_instances
        @inbounds begin
            leaf_node_idx = Int(n_instances) - 1 + i
            old_node = nodes[leaf_node_idx]

            # Get the actual instance index from the leaf node (stored as 0-indexed in child1)
            inst_idx = Int(old_node.child1) + 1
            inst = instances[inst_idx]
            blas = blas_array[inst.blas_index]
            local_aabb = blas.root_aabb

            # Transform AABB to world space (8 corners)
            world_aabb = Bounds3()
            for c in 1:8
                world_corner = transform_point(inst.transform, corner(local_aabb, c))
                world_aabb = world_aabb ∪ Bounds3(world_corner)
            end

            # Update leaf node with new AABB (preserve topology)
            nodes[leaf_node_idx] = BVHNode2(
                world_aabb.p_min, world_aabb.p_max, Point3f(0), Point3f(0),
                old_node.child0, old_node.child1, old_node.parent
            )
        end
    end
end

"""
GPU kernel: Batch update materials based on particle velocities.

Updates material colors using a heat-map based on velocity magnitude.
Also updates metallic/roughness with per-particle noise variation.
"""
KA.@kernel function update_particle_materials_kernel!(
    materials,
    @Const(positions),
    @Const(velocities),
    @Const(radii),
    n_particles::Int32,
    max_speed::Float32
)
    i = @index(Global, Linear)
    if i <= n_particles
        @inbounds begin
            vel = velocities[i]
            speed = sqrt(vel[1]^2 + vel[2]^2 + vel[3]^2)

            # Velocity to heat color (same as CPU version)
            t = clamp(speed / max_speed, 0.0f0, 1.0f0)
            color = if t < 0.25f0
                s = t / 0.25f0
                RGB{Float32}(0.1f0, 0.2f0 + 0.5f0 * s, 0.8f0)
            elseif t < 0.5f0
                s = (t - 0.25f0) / 0.25f0
                RGB{Float32}(0.1f0 + 0.6f0 * s, 0.7f0, 0.8f0 - 0.6f0 * s)
            elseif t < 0.75f0
                s = (t - 0.5f0) / 0.25f0
                RGB{Float32}(0.7f0 + 0.3f0 * s, 0.7f0 - 0.4f0 * s, 0.2f0 - 0.1f0 * s)
            else
                s = (t - 0.75f0) / 0.25f0
                RGB{Float32}(1.0f0, 0.3f0 + 0.7f0 * s, 0.1f0 + 0.9f0 * s)
            end

            # Noise for metallic/roughness variety
            noise = sin(Float32(i) * 0.1f0) * 0.5f0 + 0.5f0
            metallic = 0.3f0 + noise * 0.6f0
            roughness = 0.2f0 + (1.0f0 - noise) * 0.4f0

            materials[i] = Material(color, metallic, roughness, 1.0f0, 0.0f0)
        end
    end
end

# Export kernel helper functions for testing
export calculate_morton_code_for_prim, build_topology_for_node
export set_parents_for_node, create_leaf_for_prim, refit_node_aabb
export calculate_tlas_morton_code, create_tlas_leaf_for_instance
export update_instance_transforms_kernel!, update_instance_transforms_offset_kernel!, update_tlas_leaf_aabbs_kernel!
export update_particle_materials_kernel!
