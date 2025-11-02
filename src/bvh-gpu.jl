# GPU-optimized BVH structures and traversal for ray tracing
# Based on state-of-the-art GPU BVH techniques from NVIDIA and AMD research

using StaticArrays

# ============================================================================
# GPU-Optimized Data Structures
# ============================================================================
# Note: We reuse the existing LinearBVH structure from bvh.jl as it's already
# GPU-friendly (flat array, depth-first layout). The main optimization is in
# triangle pre-processing and kernel implementation.

"""
    CompactTriangle

Pre-transformed triangle data for efficient GPU ray-triangle intersection.
Uses edge vectors for Möller-Trumbore intersection algorithm.

Fields:
- v0: First vertex position (4th component for alignment)
- edge1: v1 - v0 edge vector
- edge2: v2 - v0 edge vector
- normal: Face normal (4th component for alignment)
- indices: [material_idx, primitive_idx] for shading
"""
struct CompactTriangle
    v0::SVector{4, Float32}
    edge1::SVector{4, Float32}
    edge2::SVector{4, Float32}
    normal::SVector{4, Float32}
    indices::SVector{2, UInt32}
end

# ============================================================================
# Conversion Functions: CPU BVH → GPU-Optimized Format
# ============================================================================

"""
    to_compact_triangle(tri::Triangle) -> CompactTriangle

Convert a Triangle to CompactTriangle format with pre-computed edge vectors.
"""
@inline function to_compact_triangle(tri::Triangle)::CompactTriangle
    vs = vertices(tri)
    v0, v1, v2 = vs[1], vs[2], vs[3]
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute face normal
    face_normal = normalize(cross(Vec3f(edge1), Vec3f(edge2)))

    return CompactTriangle(
        SVector{4, Float32}(v0[1], v0[2], v0[3], 0f0),
        SVector{4, Float32}(edge1[1], edge1[2], edge1[3], 0f0),
        SVector{4, Float32}(edge2[1], edge2[2], edge2[3], 0f0),
        SVector{4, Float32}(face_normal[1], face_normal[2], face_normal[3], 0f0),
        SVector{2, UInt32}(tri.material_idx, tri.primitive_idx)
    )
end

# ============================================================================
# GPU-Optimized Ray-Triangle Intersection
# ============================================================================

"""
    intersect_compact_triangle(tri::CompactTriangle, ray_o, ray_d, t_max)

Watertight Möller-Trumbore ray-triangle intersection for GPU.
Uses pre-computed edge vectors for efficiency.

Returns: (hit, t, u, v) where u,v are barycentric coordinates
"""
@inline function intersect_compact_triangle(
    tri::CompactTriangle,
    ray_o::Point3f,
    ray_d::Vec3f,
    t_max::Float32
)
    EPSILON = 1.0f-6

    # Möller-Trumbore algorithm - work directly with SVector components to avoid allocations
    # h = cross(ray_d, edge2)
    h_x = ray_d[2] * tri.edge2[3] - ray_d[3] * tri.edge2[2]
    h_y = ray_d[3] * tri.edge2[1] - ray_d[1] * tri.edge2[3]
    h_z = ray_d[1] * tri.edge2[2] - ray_d[2] * tri.edge2[1]

    # a = dot(edge1, h)
    a = tri.edge1[1] * h_x + tri.edge1[2] * h_y + tri.edge1[3] * h_z

    # Check if ray is parallel to triangle
    if abs(a) < EPSILON
        return (false, t_max, 0f0, 0f0)
    end

    f = 1f0 / a

    # s = ray_o - v0
    s_x = ray_o[1] - tri.v0[1]
    s_y = ray_o[2] - tri.v0[2]
    s_z = ray_o[3] - tri.v0[3]

    # u = f * dot(s, h)
    u = f * (s_x * h_x + s_y * h_y + s_z * h_z)

    if u < 0f0 || u > 1f0
        return (false, t_max, 0f0, 0f0)
    end

    # q = cross(s, edge1)
    q_x = s_y * tri.edge1[3] - s_z * tri.edge1[2]
    q_y = s_z * tri.edge1[1] - s_x * tri.edge1[3]
    q_z = s_x * tri.edge1[2] - s_y * tri.edge1[1]

    # v = f * dot(ray_d, q)
    v = f * (ray_d[1] * q_x + ray_d[2] * q_y + ray_d[3] * q_z)

    if v < 0f0 || u + v > 1f0
        return (false, t_max, 0f0, 0f0)
    end

    # t = f * dot(edge2, q)
    t = f * (tri.edge2[1] * q_x + tri.edge2[2] * q_y + tri.edge2[3] * q_z)

    if t > EPSILON && t < t_max
        return (true, t, u, v)
    end

    return (false, t_max, 0f0, 0f0)
end

# ============================================================================
# GPUBVH Main Structure
# ============================================================================

"""
    GPUBVH{NodeVec, TriVec, OrigTriVec}

GPU-optimized BVH acceleration structure.

Key optimizations:
- Reuses existing LinearBVH node structure (already GPU-friendly)
- Pre-transforms triangles with edge vectors for Möller-Trumbore
- Designed for efficient GPU kernel traversal

Fields:
- nodes: LinearBVH nodes (flat array, depth-first layout)
- triangles: Pre-transformed compact triangles
- original_triangles: Original triangles (for normals, UVs, materials)
- max_node_primitives: Maximum primitives per leaf node
"""
struct GPUBVH{
    NodeVec <: AbstractVector{LinearBVH},
    TriVec <: AbstractVector{CompactTriangle},
    OrigTriVec <: AbstractVector{Triangle}
} <: AccelPrimitive
    nodes::NodeVec
    triangles::TriVec
    original_triangles::OrigTriVec
    max_node_primitives::UInt8
end

"""
    GPUBVH(bvh::BVHAccel)

Convert a standard BVHAccel to GPU-optimized GPUBVH format.
Pre-processes triangles but keeps the existing BVH node structure.
"""
function GPUBVH(bvh::BVHAccel)
    # Convert triangles to compact format with pre-computed edges
    compact_tris = [to_compact_triangle(tri) for tri in bvh.primitives]

    return GPUBVH(
        bvh.nodes,
        compact_tris,
        bvh.primitives,
        bvh.max_node_primitives
    )
end

# ============================================================================
# World Bound
# ============================================================================

@inline function world_bound(gpubvh::GPUBVH)::Bounds3
    length(gpubvh.nodes) > 0 ? gpubvh.nodes[1].bounds : Bounds3()
end

# ============================================================================
# GPU BVH Traversal - Manual implementation
# ============================================================================

"""
    closest_hit(gpubvh::GPUBVH, ray::AbstractRay)

Find the closest intersection between a ray and the GPU BVH.
Uses manual traversal with compact triangle intersection for best performance.
"""
@inline function closest_hit(gpubvh::GPUBVH, ray::AbstractRay, allocator=MemAllocator())
    # Use the standard traverse_bvh with a custom callback
    # This reuses all the optimized traversal logic
    ray_initial = ray
    ray = check_direction(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    # Initialize traversal
    local to_visit_offset::Int32 = Int32(1)
    current_node_idx = Int32(1)
    # Direct MVector construction for type stability (critical for GPU, especially OpenCL/SPIR-V)
    nodes_to_visit = MVector{64, Int32}(undef)
    nodes = gpubvh.nodes
    triangles = gpubvh.triangles
    original_tris = gpubvh.original_triangles

    # Track closest hit
    hit_found = false
    hit_tri_idx = Int32(0)
    closest_t = ray.t_max
    hit_u = 0f0
    hit_v = 0f0

    # Traverse BVH
    @_inbounds while true
        current_node = nodes[current_node_idx]

        # Test ray against current node's bounding box
        if intersect_p(current_node.bounds, ray, inv_dir, dir_is_neg)
            local cnprim::Int32 = current_node.n_primitives % Int32
            if !current_node.is_interior && cnprim > Int32(0)
                # Leaf node - test all triangles
                offset = current_node.offset % Int32

                for i in Int32(0):(cnprim - Int32(1))
                    tri_idx = offset + i
                    compact_tri = triangles[tri_idx]

                    # Use compact intersection
                    tmp_hit, t, u, v = intersect_compact_triangle(compact_tri, ray.o, ray.d, closest_t)
                    if tmp_hit && t < closest_t
                        closest_t = t
                        hit_found = true
                        hit_tri_idx = tri_idx
                        hit_u = u
                        hit_v = v
                    end
                end

                # Done with leaf, pop next node from stack
                if to_visit_offset === Int32(1)
                    break
                end
                to_visit_offset -= Int32(1)
                current_node_idx = nodes_to_visit[to_visit_offset]
            else
                # Interior node - push children to stack
                # Explicitly unroll axis cases to avoid LLVM select chains in SPIR-V
                local is_neg = if current_node.split_axis == Int32(1)
                    dir_is_neg[1] == Int32(2)
                elseif current_node.split_axis == Int32(2)
                    dir_is_neg[2] == Int32(2)
                else  # split_axis == 3
                    dir_is_neg[3] == Int32(2)
                end

                if is_neg
                    nodes_to_visit[to_visit_offset] = current_node_idx + Int32(1)
                    current_node_idx = current_node.offset % Int32
                else
                    nodes_to_visit[to_visit_offset] = current_node.offset % Int32
                    current_node_idx += Int32(1)
                end
                to_visit_offset += Int32(1)
            end
        else
            # Miss - pop next node from stack
            if to_visit_offset === Int32(1)
                break
            end
            to_visit_offset -= Int32(1)
            current_node_idx = nodes_to_visit[to_visit_offset]
        end
    end

    # Return result
    if hit_found
        orig_tri = original_tris[hit_tri_idx]
        w = 1f0 - hit_u - hit_v
        return (true, orig_tri, closest_t, Point3f(w, hit_u, hit_v))
    else
        # Return dummy result matching standard BVH behavior
        dummy_tri = original_tris[1]
        return (false, dummy_tri, 0f0, Point3f(0f0))
    end
end

# ============================================================================
# Any Hit (occlusion testing)
# ============================================================================

"""
    any_hit(gpubvh::GPUBVH, ray::AbstractRay)

Test if a ray intersects any primitive in the GPU BVH (for occlusion testing).
Stops at the first intersection found.
"""
@inline function any_hit(gpubvh::GPUBVH, ray::AbstractRay, allocator=MemAllocator())
    ray = check_direction(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    # Initialize traversal
    local to_visit_offset::Int32 = Int32(1)
    current_node_idx = Int32(1)
    # Direct MVector construction for type stability (critical for GPU, especially OpenCL/SPIR-V)
    nodes_to_visit = MVector{64, Int32}(undef)
    nodes = gpubvh.nodes
    triangles = gpubvh.triangles
    original_tris = gpubvh.original_triangles

    # Traverse BVH
    @_inbounds while true
        current_node = nodes[current_node_idx]

        # Test ray against current node's bounding box
        if intersect_p(current_node.bounds, ray, inv_dir, dir_is_neg)
            local cnprim::Int32 = current_node.n_primitives % Int32
            if !current_node.is_interior && cnprim > Int32(0)
                # Leaf node - test triangles
                offset = current_node.offset % Int32

                for i in Int32(0):(cnprim - Int32(1))
                    tri_idx = offset + i
                    compact_tri = triangles[tri_idx]

                    # Test for any hit
                    tmp_hit, t, u, v = intersect_compact_triangle(compact_tri, ray.o, ray.d, ray.t_max)
                    if tmp_hit
                        # Return immediately on first hit
                        orig_tri = original_tris[tri_idx]
                        w = 1f0 - u - v
                        return (true, orig_tri, t, Point3f(w, u, v))
                    end
                end

                # Done with leaf, pop next node from stack
                if to_visit_offset === Int32(1)
                    break
                end
                to_visit_offset -= Int32(1)
                current_node_idx = nodes_to_visit[to_visit_offset]
            else
                # Interior node - push children to stack
                # Explicitly unroll axis cases to avoid LLVM select chains in SPIR-V
                local is_neg = if current_node.split_axis == Int32(1)
                    dir_is_neg[1] == Int32(2)
                elseif current_node.split_axis == Int32(2)
                    dir_is_neg[2] == Int32(2)
                else  # split_axis == 3
                    dir_is_neg[3] == Int32(2)
                end

                if is_neg
                    nodes_to_visit[to_visit_offset] = current_node_idx + Int32(1)
                    current_node_idx = current_node.offset % Int32
                else
                    nodes_to_visit[to_visit_offset] = current_node.offset % Int32
                    current_node_idx += Int32(1)
                end
                to_visit_offset += Int32(1)
            end
        else
            # Miss - pop next node from stack
            if to_visit_offset === Int32(1)
                break
            end
            to_visit_offset -= Int32(1)
            current_node_idx = nodes_to_visit[to_visit_offset]
        end
    end

    # No hit found
    dummy_tri = original_tris[1]
    return (false, dummy_tri, 0f0, Point3f(0f0))
end

# ============================================================================
# Pretty Printing
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", gpubvh::GPUBVH)
    n_nodes = length(gpubvh.nodes)
    n_triangles = length(gpubvh.triangles)

    # Count leaf vs interior nodes
    n_leaves = count(node -> !node.is_interior, gpubvh.nodes)
    n_interior = n_nodes - n_leaves

    println(io, "GPUBVH:")
    println(io, "  Triangles:     ", n_triangles, " (pre-transformed)")
    println(io, "  BVH nodes:     ", n_nodes, " (", n_interior, " interior, ", n_leaves, " leaves)")
    print(io,   "  Max prims:     ", Int(gpubvh.max_node_primitives), " per leaf")
end

function Base.show(io::IO, gpubvh::GPUBVH)
    if get(io, :compact, false)
        n_triangles = length(gpubvh.triangles)
        n_nodes = length(gpubvh.nodes)
        print(io, "GPUBVH(triangles=", n_triangles, ", nodes=", n_nodes, ")")
    else
        show(io, MIME("text/plain"), gpubvh)
    end
end
