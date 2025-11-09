abstract type AccelPrimitive <: Primitive end

struct BVHPrimitiveInfo
    primitive_number::UInt32
    bounds::Bounds3
    centroid::Point3f

    function BVHPrimitiveInfo(primitive_number::Integer, bounds::Bounds3)
        new(
            primitive_number, bounds,
            0.5f0 * bounds.p_min + 0.5f0 * bounds.p_max,
        )
    end
end

struct BVHNode
    bounds::Bounds3
    children::Tuple{Maybe{BVHNode},Maybe{BVHNode}}
    split_axis::UInt8
    offset::UInt32
    n_primitives::UInt32

    """
    Construct leaf node.
    """
    function BVHNode(offset::Integer, n_primitives::Integer, bounds::Bounds3)
        new(bounds, (nothing, nothing), 0, offset, n_primitives)
    end
    """
    Construct intermediary node.
    """
    function BVHNode(axis::Integer, left::BVHNode, right::BVHNode)
        new(left.bounds ∪ right.bounds, (left, right), axis, 0, 0)
    end
end

abstract type LinearNode end

struct LinearBVH <: LinearNode
    bounds::Bounds3
    offset::UInt32
    n_primitives::UInt32
    split_axis::UInt8
    is_interior::Bool
end

function LinearBVHLeaf(bounds::Bounds3, primitives_offset::Integer, n_primitives::Integer)
    LinearBVH(bounds, primitives_offset, n_primitives, 0, false)
end

function LinearBVHInterior(bounds::Bounds3, second_child_offset::Integer, split_axis::Integer)
    LinearBVH(bounds, second_child_offset, 0, split_axis, true)
end

function primitives_to_bvh(primitives, max_node_primitives=1)
    max_node_primitives = min(255, max_node_primitives)
    isempty(primitives) && return (primitives, max_node_primitives, LinearBVH[])
    primitives_info = [
        BVHPrimitiveInfo(i, world_bound(p))
        for (i, p) in enumerate(primitives)
    ]
    total_nodes = Ref(0)
    ordered_primitives = similar(primitives, 0)
    root = _init(
        primitives, primitives_info, 1, length(primitives),
        total_nodes, ordered_primitives, max_node_primitives,
    )

    offset = Ref{UInt32}(1)
    flattened = Vector{LinearBVH}(undef, total_nodes[])
    _unroll(flattened, root, offset)
    @real_assert total_nodes[] + 1 == offset[]
    return (ordered_primitives, max_node_primitives, flattened)
end

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

"""
    BVH{NodeVec, TriVec, OrigTriVec}

GPU-optimized BVH acceleration structure.

Key optimizations:
- Uses LinearBVH node structure (flat array, depth-first layout)
- Pre-transforms triangles with edge vectors for Möller-Trumbore
- Designed for efficient GPU kernel traversal

Fields:
- nodes: LinearBVH nodes (flat array, depth-first layout)
- triangles: Pre-transformed compact triangles
- primitives: Original triangles (for normals, UVs, materials)
- max_node_primitives: Maximum primitives per leaf node
"""
struct BVH{
    NodeVec <: AbstractVector{LinearBVH},
    TriVec <: AbstractVector{CompactTriangle},
    OrigTriVec <: AbstractVector{Triangle}
} <: AccelPrimitive
    nodes::NodeVec
    triangles::TriVec
    primitives::OrigTriVec
    max_node_primitives::UInt8
end


to_triangle_mesh(x::TriangleMesh) = x

function to_triangle_mesh(x::GeometryBasics.AbstractGeometry)
    m = GeometryBasics.uv_normal_mesh(x)
    return TriangleMesh(m)
end

"""
    BVH(primitives::AbstractVector, max_node_primitives::Integer=1)

Construct a BVH acceleration structure from a list of primitives (meshes or geometries).

Arguments:
- `primitives`: Vector of triangle meshes or GeometryBasics geometries
- `max_node_primitives`: Maximum number of primitives per leaf node (default: 1)

Returns a GPU-optimized BVH with pre-transformed triangles for efficient ray tracing.
"""
function BVH(
        primitives::AbstractVector{P}, max_node_primitives::Integer=1,
    ) where {P}
    triangles = Triangle[]
    for (mi, prim) in enumerate(primitives)
        triangle_mesh = to_triangle_mesh(prim)
        vertices = triangle_mesh.vertices
        for i in 1:div(length(triangle_mesh.indices), 3)
            push!(triangles, Triangle(triangle_mesh, i, mi, length(triangles) + 1))
        end
    end
    ordered_primitives, max_prim, nodes = primitives_to_bvh(triangles, max_node_primitives)
    ordered_primitives = map(enumerate(ordered_primitives)) do (i, tri)
        Triangle(tri, primitive_idx=UInt32(i))
    end

    # Convert triangles to compact format with pre-computed edges
    compact_tris = [to_compact_triangle(tri) for tri in ordered_primitives]

    return BVH(
        nodes,
        compact_tris,
        ordered_primitives,
        UInt8(max_prim)
    )
end

mutable struct BucketInfo
    count::UInt32
    bounds::Bounds3
end

function _init(
        primitives::AbstractVector, primitives_info::Vector{BVHPrimitiveInfo},
        from::Integer, to::Integer, total_nodes::Ref{Int64},
        ordered_primitives::AbstractVector, max_node_primitives::Integer,
    )

    total_nodes[] += 1
    n_primitives = to - from + 1
    # Compute bounds for all primitives in BVH node.
    bounds = mapreduce(
        i -> primitives_info[i].bounds, ∪, from:to, init = Bounds3(),
    )
    @inline function _create_leaf()::BVHNode
        first_offset = length(ordered_primitives) + 1
        for i in from:to
            push!(
                ordered_primitives,
                primitives[primitives_info[i].primitive_number],
            )
        end
        return BVHNode(first_offset, n_primitives, bounds)
    end

    n_primitives == 1 && return _create_leaf()
    # Compute bound of primitive centroids, choose split dimension.
    centroid_bounds = mapreduce(
        i -> Bounds3(primitives_info[i].centroid), ∪, from:to,
        init = Bounds3(),
    )
    dim = maximum_extent(centroid_bounds)
    ( # Create leaf node.
        !is_valid(centroid_bounds)
        ||
        centroid_bounds.p_min[dim] == centroid_bounds.p_max[dim]
    ) && return _create_leaf()
    # Partition primitives into sets and build children.
    if n_primitives <= 2 # Equally-sized subsets.
        mid = (from + to) ÷ 2
        pmid = mid > from ? mid - from + 1 : 1
        partialsort!(
            @view(primitives_info[from:to]), pmid, by = i -> i.centroid[dim],
        )
    else # Perform Surface-Area-Heuristic partitioning.
        n_buckets = 12
        buckets = [BucketInfo(0, Bounds3(Point3f(0f0))) for _ in 1:n_buckets]
        # Initialize buckets.
        for i in from:to
            b = Int32(floor(n_buckets * offset(
                centroid_bounds, primitives_info[i].centroid,
            )[dim])) + 1
            (b == n_buckets + 1) && (b -= 1)
            buckets[b].count += 1
            buckets[b].bounds = buckets[b].bounds ∪ primitives_info[i].bounds
        end
        # Compute costs for splitting after each bucket.
        costs = Vector{Float32}(undef, n_buckets - 1)
        for i in 1:(n_buckets-1)
            it1, it2 = 1:i, (i+1):(n_buckets-1)
            s1, s2 = 0, 0
            if length(it1) > 0
                s1 = length(it1) * surface_area(
                    mapreduce(b -> buckets[b].bounds, ∪, it1),
                )
            end
            if length(it2) > 0
                s2 = length(it2) * surface_area(
                    mapreduce(b -> buckets[b].bounds, ∪, it2),
                )
            end
            costs[i] = 1f0 + (s1 + s2) / surface_area(bounds)
        end
        # Find bucket to split that minimizes SAH metric.
        min_cost_id = argmin(costs)
        leaf_cost = n_primitives
        # Either create leaf or split primitives at selected SAH bucket.
        !(
            n_primitives > max_node_primitives
            ||
            costs[min_cost_id] < leaf_cost
        ) && return _create_leaf()
        mid = partition!(primitives_info, from:to, i -> begin
            b = Int32(floor(
                n_buckets * offset(centroid_bounds, i.centroid)[dim],
            )) + 1
            (b == n_buckets + 1) && (b -= 1)
            b <= min_cost_id
        end)
    end
    BVHNode(
        dim,
        _init(
            primitives, primitives_info, from, mid,
            total_nodes, ordered_primitives, max_node_primitives,
        ),
        _init(
            primitives, primitives_info, mid + 1, to,
            total_nodes, ordered_primitives, max_node_primitives,
        ),
    )
end

function _unroll(
        linear_nodes::Vector{LinearBVH}, node::BVHNode, offset::Ref{UInt32},
    )

    l_offset = offset[]
    offset[] += 1

    if node.children[1] isa Nothing
        linear_nodes[l_offset] = LinearBVHLeaf(
            node.bounds, node.offset, node.n_primitives,
        )
        return l_offset + 1
    end

    _unroll(linear_nodes, node.children[1], offset)
    second_child_offset = _unroll(linear_nodes, node.children[2], offset) - 1
    linear_nodes[l_offset] = LinearBVHInterior(
        node.bounds, second_child_offset, node.split_axis,
    )
    l_offset + 1
end

@inline function world_bound(bvh::BVH)::Bounds3
    length(bvh.nodes) > 0 ? bvh.nodes[1].bounds : Bounds3()
end

struct MemAllocator
end
@inline _allocate(::MemAllocator, T::Type, n::Val{N}) where {N} = MVector{N,T}(undef)
Base.@propagate_inbounds function _setindex(arr::MVector{N, T}, idx::Integer, value::T) where {N, T}
    arr[idx] = value
    return arr
end


"""
    closest_hit(bvh::BVH, ray::AbstractRay)

Find the closest intersection between a ray and the GPU BVH.
Uses manual traversal with compact triangle intersection for best performance.

Returns:
- `hit_found`: Boolean indicating if an intersection was found
- `hit_primitive`: The primitive that was hit (if any)
- `distance`: Distance along the ray to the hit point (hit_point = ray.o + ray.d * distance)
- `barycentric_coords`: Barycentric coordinates of the hit point
"""
@inline function closest_hit(bvh::BVH, ray::AbstractRay, allocator=MemAllocator())
    ray = check_direction(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    # Initialize traversal
    local to_visit_offset::Int32 = Int32(1)
    current_node_idx = Int32(1)
    # Direct MVector construction for type stability (critical for GPU, especially OpenCL/SPIR-V)
    nodes_to_visit = MVector{64, Int32}(undef)
    nodes = bvh.nodes
    triangles = bvh.triangles
    original_tris = bvh.primitives

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
        # Use SVector for GPU compatibility - Point3f is just an alias for SVector
        bary_point = SVector{3, Float32}(w, hit_u, hit_v)
        return (true, orig_tri, closest_t, bary_point)
    else
        # Return dummy result matching standard BVH behavior
        dummy_tri = original_tris[1]
        bary_point = SVector{3, Float32}(0f0, 0f0, 0f0)
        return (false, dummy_tri, 0f0, bary_point)
    end
end

"""
    any_hit(bvh::BVH, ray::AbstractRay)

Test if a ray intersects any primitive in the GPU BVH (for occlusion testing).
Stops at the first intersection found.

Returns:
- `hit_found`: Boolean indicating if any intersection was found
- `hit_primitive`: The primitive that was hit (if any)
- `distance`: Distance along the ray to the hit point (hit_point = ray.o + ray.d * distance)
- `barycentric_coords`: Barycentric coordinates of the hit point
"""
@inline function any_hit(bvh::BVH, ray::AbstractRay, allocator=MemAllocator())
    ray = check_direction(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    # Initialize traversal
    local to_visit_offset::Int32 = Int32(1)
    current_node_idx = Int32(1)
    # Direct MVector construction for type stability (critical for GPU, especially OpenCL/SPIR-V)
    nodes_to_visit = MVector{64, Int32}(undef)
    nodes = bvh.nodes
    triangles = bvh.triangles
    original_tris = bvh.primitives

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
                        bary_point = SVector{3, Float32}(w, u, v)
                        return (true, orig_tri, t, bary_point)
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
    bary_point = SVector{3, Float32}(0f0, 0f0, 0f0)
    return (false, dummy_tri, 0f0, bary_point)
end

function calculate_ray_grid_bounds(bounds::GeometryBasics.Rect, ray_direction::Vec3f)
    # Normalize the direction vector (in case it's not already a unit vector)
    direction = normalize(ray_direction)
    # 1. Find a plane perpendicular to the ray direction
    # We need two basis vectors that are perpendicular to the ray direction
    # First, find a non-parallel vector to create our first basis vector
    if abs(direction[1]) < 0.9f0
        temp = Vec3f(1.0, 0.0, 0.0)
    else
        temp = Vec3f(0.0, 1.0, 0.0)
    end

    # Create two perpendicular basis vectors for the grid
    basis1 = normalize(cross(direction, temp))
    basis2 = normalize(cross(direction, basis1))

    corners = decompose(Point3f, bounds)

    # 3. Project corners onto our basis vectors
    proj1 = [dot(corner, basis1) for corner in corners]
    proj2 = [dot(corner, basis2) for corner in corners]

    # 4. Find the min and max projections to determine grid bounds
    min_proj1, max_proj1 = extrema(proj1)
    min_proj2, max_proj2 = extrema(proj2)

    # 5. Add a small margin to ensure coverage
    margin = 0.05f0 * max(max_proj1 - min_proj1, max_proj2 - min_proj2)
    grid_width = max_proj1 - min_proj1 + 2 * margin
    grid_height = max_proj2 - min_proj2 + 2 * margin

    # 6. Calculate the origin point of the grid
    # Choose a point that's sufficiently far back from the bounding box
    # Project all corners onto the ray direction
    depth_proj = [dot(corner, direction) for corner in corners]
    min_depth = minimum(depth_proj) - margin

    # Grid center in world space
    grid_center = Point3f(0, 0, 0) + min_depth * direction +
                  ((min_proj1 + max_proj1) / 2f0) * basis1 +
                  ((min_proj2 + max_proj2) / 2f0) * basis2

    # 7. Return the grid information
    return (
        center=grid_center,
        width=grid_width,
        height=grid_height,
        basis1=basis1,
        basis2=basis2,
    )
end

# Function to generate ray origins for the grid
function generate_ray_grid(grid_info, grid_size::Int)
    ray_origins = Matrix{Point3f}(undef, grid_size, grid_size)
    cell_size_width = grid_info.width / grid_size
    cell_size_height = grid_info.height / grid_size
    for i in 1:grid_size
        for j in 1:grid_size
            # Calculate the offset from the center
            u = (i - (grid_size + 1) / 2) * cell_size_width
            v = (j - (grid_size + 1) / 2) * cell_size_height

            # Calculate the ray origin
            ray_origins[i, j] = grid_info.center + u * grid_info.basis1 + v * grid_info.basis2
        end
    end
    return ray_origins
end

"""
    generate_ray_grid(bvh::BVH, ray_direction::Vec3f, grid_size::Int)

Generate a grid of ray origins based on the BVH bounding box and a given ray direction.
"""
function generate_ray_grid(bvh::BVH, ray_direction::Vec3f, grid_size::Int)
    bounds = world_bound(bvh)
    bb = Rect3f(bounds.p_min, bounds.p_max .- bounds.p_min)
    grid_info = calculate_ray_grid_bounds(bb, ray_direction)
    return generate_ray_grid(grid_info, grid_size)
end


function GeometryBasics.Mesh(bvh::BVH)
    points = Point3f[]
    faces = GLTriangleFace[]
    prims = bvh.primitives # Use original triangles, not compact ones
    for (ti, tringle) in enumerate(prims)
        push!(points, tringle.vertices...)
        tt = ((ti - 1) * 3) + 1
        face = GLTriangleFace(tt, tt + 1, tt + 2)
        push!(faces, face)
    end
    return GeometryBasics.Mesh(points, faces)
end

# Pretty printing for BVH
function Base.show(io::IO, ::MIME"text/plain", bvh::BVH)
    n_triangles = length(bvh.triangles)
    n_nodes = length(bvh.nodes)

    # Count leaf vs interior nodes
    n_leaves = count(node -> !node.is_interior, bvh.nodes)
    n_interior = n_nodes - n_leaves

    println(io, "BVH:")
    println(io, "  Triangles:     ", n_triangles, " (pre-transformed)")
    println(io, "  BVH nodes:     ", n_nodes, " (", n_interior, " interior, ", n_leaves, " leaves)")
    print(io,   "  Max prims:     ", Int(bvh.max_node_primitives), " per leaf")
end

function Base.show(io::IO, bvh::BVH)
    if get(io, :compact, false)
        n_triangles = length(bvh.triangles)
        n_nodes = length(bvh.nodes)
        print(io, "BVH(triangles=", n_triangles, ", nodes=", n_nodes, ")")
    else
        show(io, MIME("text/plain"), bvh)
    end
end
