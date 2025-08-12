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

struct BVHAccel{
            PVec <:AbstractVector,
            NodeVec <: AbstractVector{LinearBVH}
        } <: AccelPrimitive
    primitives::PVec
    max_node_primitives::UInt8
    nodes::NodeVec
end


to_triangle_mesh(x::TriangleMesh) = x

function to_triangle_mesh(x::GeometryBasics.AbstractGeometry)
    m = GeometryBasics.uv_normal_mesh(x)
    return create_triangle_mesh(m)
end


function BVHAccel(
        primitives::AbstractVector{P}, max_node_primitives::Integer=1,
    ) where {P}
    triangles = Triangle[]
    prim_idx = 1
    for (mi, prim) in enumerate(primitives)
        triangle_mesh = to_triangle_mesh(prim)
        vertices = triangle_mesh.vertices
        for i in 1:div(length(triangle_mesh.indices), 3)
            push!(triangles, Triangle(triangle_mesh, i, prim_idx))
            prim_idx += 1
        end
    end
    ordered_primitives, max_prim, nodes = primitives_to_bvh(triangles, max_node_primitives)
    return BVHAccel(ordered_primitives, UInt8(max_prim), nodes)
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

@inline function world_bound(bvh::BVHAccel)::Bounds3
    length(bvh.nodes) > Int32(0) ? bvh.nodes[1].bounds : Bounds3()
end

@inline function intersect!(bvh::BVHAccel{P}, ray::AbstractRay) where {P}
    hit = false
    interaction = SurfaceInteraction()
    ray = check_direction(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    to_visit_offset, current_node_i = Int32(1), Int32(1)
    nodes_to_visit = zeros(MVector{64,Int32})
    primitives = bvh.primitives
    @_inbounds primitive = primitives[1]
    nodes = bvh.nodes
    @_inbounds while true
        ln = nodes[current_node_i]
        if intersect_p(ln.bounds, ray, inv_dir, dir_is_neg)
            if !ln.is_interior && ln.n_primitives > Int32(0)
                # Intersect ray with primitives in node.
                for i in Int32(0):ln.n_primitives - Int32(1)
                    offset = ln.offset % Int32
                    tmp_primitive = primitives[offset+i]
                    tmp_hit, ray, tmp_interaction = intersect_p!(
                        tmp_primitive, ray,
                    )
                    if tmp_hit
                        hit = tmp_hit
                        interaction = tmp_interaction
                        primitive = tmp_primitive
                    end
                end
                to_visit_offset == Int32(1) && break
                to_visit_offset -= Int32(1)
                current_node_i = nodes_to_visit[to_visit_offset]
            else
                if dir_is_neg[ln.split_axis] == Int32(2)
                    nodes_to_visit[to_visit_offset] = current_node_i + Int32(1)
                    current_node_i = ln.offset % Int32
                else
                    nodes_to_visit[to_visit_offset] = ln.offset % Int32
                    current_node_i += Int32(1)
                end
                to_visit_offset += Int32(1)
            end
        else
            to_visit_offset == 1 && break
            to_visit_offset -= Int32(1)
            current_node_i = nodes_to_visit[to_visit_offset]
        end
    end
    return hit, primitive, interaction
end

@inline function intersect_p(bvh::BVHAccel, ray::AbstractRay)

    length(bvh.nodes) == Int32(0) && return false

    ray = check_direction(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    to_visit_offset, current_node_i = Int32(1), Int32(1)
    nodes_to_visit = zeros(MVector{64,Int32})
    primitives = bvh.primitives
    @_inbounds while true
        ln = bvh.nodes[current_node_i]
        if intersect_p(ln.bounds, ray, inv_dir, dir_is_neg)
            if !ln.is_interior && ln.n_primitives > Int32(0)
                for i in Int32(0):ln.n_primitives-Int32(1)
                    offset = ln.offset % Int32
                    intersect_p(
                        primitives[offset + i], ray,
                    ) && return true
                end
                to_visit_offset == 1 && break
                to_visit_offset -= Int32(1)
                current_node_i = nodes_to_visit[to_visit_offset]
            else
                if dir_is_neg[ln.split_axis] == Int32(2)
                    # @setindex 64 nodes_to_visit[to_visit_offset] = Int32(current_node_i + 1)
                    nodes_to_visit[to_visit_offset] = current_node_i + Int32(1)
                    current_node_i = ln.offset % Int32
                else
                    # @setindex 64 nodes_to_visit[to_visit_offset] = Int32(ln.offset)
                    nodes_to_visit[to_visit_offset] = ln.offset % Int32
                    current_node_i += Int32(1)
                end
                to_visit_offset += Int32(1)
            end
        else
            to_visit_offset == Int32(1) && break
            to_visit_offset -= Int32(1)
            current_node_i = Int32(nodes_to_visit[to_visit_offset])
        end
    end
    false
end

function calculate_ray_grid_bounds(bounds::GeometryBasics.Rect, ray_direction::Vec3f)
    # Normalize the direction vector (in case it's not already a unit vector)
    direction = normalize(ray_direction)
    # 1. Find a plane perpendicular to the ray direction
    # We need two basis vectors that are perpendicular to the ray direction
    # First, find a non-parallel vector to create our first basis vector
    if abs(direction[1]) < 0.9
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
    generate_ray_grid(bvh::BVHAccel, ray_direction::Vec3f, grid_size::Int)

Generate a grid of ray origins based on the BVH bounding box and a given ray direction.
"""
function generate_ray_grid(bvh::BVHAccel, ray_direction::Vec3f, grid_size::Int)
    bounds = world_bound(bvh)
    bb = Rect3f(bounds.p_min, bounds.p_max .- bounds.p_min)
    grid_info = calculate_ray_grid_bounds(bb, ray_direction)
    return generate_ray_grid(grid_info, grid_size)
end


function GeometryBasics.Mesh(bvh::BVHAccel)
    points = Point3f[]
    faces = GLTriangleFace[]
    prims = sort(bvh.primitives; by=x -> x.material_idx)
    for (ti, tringle) in enumerate(prims)
        push!(points, tringle.vertices...)
        tt = ((ti - 1) * 3) + 1
        face = GLTriangleFace(tt, tt + 1, tt + 2)
        push!(faces, face)
    end
    return GeometryBasics.Mesh(points, faces)
end
