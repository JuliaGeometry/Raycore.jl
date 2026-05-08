# Instanced BVH Architecture

## Overview

This document describes the two-level instanced BVH implementation in Raycore, inspired by AMD's RadeonRays SDK architecture. The design enables efficient ray tracing of scenes with repeated geometry (instances) while maintaining GPU/CPU portability through parametrized array types.

## Architecture

### Two-Level Hierarchy

The instanced BVH uses a **two-level acceleration structure**:

1. **BLAS (Bottom-Level Acceleration Structure)**: BVH over triangle geometry
   - One BLAS per unique mesh
   - Built once, reused for all instances
   - Stores geometry in local/object space

2. **TLAS (Top-Level Acceleration Structure)**: BVH over instances
   - Contains transformed instances of BLAS objects
   - Each instance has a transformation matrix
   - Built per-frame or when instances change

### Memory Layout

```
TLAS (Scene-level)
├── Node[0]: Root (Interior)
│   ├── Child0 → Node[1]
│   └── Child1 → Node[2]
├── Node[1]: Interior
│   ├── Child0 → Instance 0 (Leaf)
│   └── Child1 → Instance 1 (Leaf)
└── Node[2]: Instance 2 (Leaf)

BLAS (Mesh-level)
├── Node[0]: Root (Interior)
│   ├── Child0 → Node[1]
│   └── Child1 → Triangle 0 (Leaf)
└── Node[1]: Triangle 1 (Leaf)
```

## Core Data Structures

### BVHNode2

Compact BVH node for binary trees (BVH2IL layout from RadeonRays):

```julia
struct BVHNode2
    aabb0_min::Point3f      # Child 0 AABB min
    aabb0_max::Point3f      # Child 0 AABB max
    aabb1_min::Point3f      # Child 1 AABB min
    aabb1_max::Point3f      # Child 1 AABB max
    child0::UInt32          # Child 0 index (INVALID_NODE for leaves)
    child1::UInt32          # Child 1 index (primitive for leaves)
    parent::UInt32          # Parent node index
end
```

**Design Rationale:**
- **Inline AABBs**: Storing both children's AABBs directly in the parent node enables branchless intersection tests
- **Compact size**: 64 bytes per node for cache efficiency
- **Leaf identification**: `child0 == INVALID_NODE` marks leaf nodes
- **Flexible use**: Same structure for both BLAS and TLAS

### InstanceDescriptor

Describes an instance of a BLAS in world space:

```julia
struct InstanceDescriptor
    blas_index::UInt32          # Which BLAS to instance
    instance_id::UInt32         # User-defined ID
    transform::Mat4f            # Local-to-world transformation
    inv_transform::Mat4f        # World-to-local transformation
    flags::UInt32               # Reserved for future use
end
```

### BLAS

```julia
struct BLAS{NodeArray <: AbstractVector{BVHNode2},
            TriArray <: AbstractVector{<:Triangle}}
    nodes::NodeArray            # BVH nodes
    primitives::TriArray        # Triangles (sorted by Morton code)
    root_aabb::Bounds3          # Bounding box in local space
end
```

**Type Parameters:**
- `NodeArray`: Can be `Vector`, `CuArray`, `ROCArray`, etc.
- `TriArray`: Same - enables CPU/GPU execution

### TLAS

```julia
struct TLAS{NodeArray <: AbstractVector{BVHNode2},
            InstArray <: AbstractVector{InstanceDescriptor},
            BLASArray <: AbstractVector{<:BLAS}}
    nodes::NodeArray            # Top-level BVH nodes
    instances::InstArray        # Instance descriptors
    blas_array::BLASArray       # Array of BLAS objects
    root_aabb::Bounds3          # World-space bounding box
end
```

## Construction Algorithm

### LBVH (Linear BVH)

Both BLAS and TLAS use the **LBVH algorithm** (Karras 2012):

#### Step 1: Compute Scene AABB
```julia
scene_aabb = mapreduce(world_bound, ∪, primitives)
```

#### Step 2: Calculate 30-bit Morton Codes

Morton codes provide a space-filling Z-curve ordering:

```julia
function morton_code_30bit(p::Point3f)::UInt32
    # Normalize to [0, 1023] (10 bits per axis)
    unit_side = 1024.0f0
    x = clamp(p[1] * unit_side, 0.0f0, unit_side - 1.0f0)
    y = clamp(p[2] * unit_side, 0.0f0, unit_side - 1.0f0)
    z = clamp(p[3] * unit_side, 0.0f0, unit_side - 1.0f0)

    # Interleave bits: xxyyzzxxyyzzxxyyzz...
    return (expand_bits(UInt32(x)) << 2) |
           (expand_bits(UInt32(y)) << 1) |
            expand_bits(UInt32(z))
end
```

**Why 30 bits?**
- 10 bits per axis = 1024³ spatial resolution
- Leaves 2 bits for flags if needed
- Fits comfortably in UInt32

#### Step 3: Sort Primitives

```julia
sorted_indices = sortperm(morton_codes)
morton_codes .= morton_codes[sorted_indices]
sorted_prims = primitives[sorted_indices]
```

#### Step 4: Build Binary Radix Tree

Using Karras' algorithm to find node spans:

```julia
# Find span of internal node i
d_left = delta(i, i-1, morton_codes, n)
d_right = delta(i, i+1, morton_codes, n)
direction = sign(d_right - d_left)

# Binary search for exact span
# ... (see implementation)

# Find split point
split = find_split(span_left, span_right, morton_codes, n)

# Determine children
child0 = (split == span_left) ? leaf(split) : internal(split)
child1 = (split+1 == span_right) ? leaf(split+1) : internal(split+1)
```

**Key function: `delta` (Longest Common Prefix)**

```julia
function delta(i1::Int32, i2::Int32, codes::Vector{UInt32}, n::Int32)::Int32
    left = min(i1, i2)
    right = max(i1, i2)

    (left < 1 || right > n) && return Int32(-1)

    left_code = codes[left]
    right_code = codes[right]

    # If codes differ, count common prefix bits
    # If codes identical, use indices as tiebreaker
    if left_code != right_code
        return Int32(clz32(left_code ⊻ right_code))
    else
        return Int32(32 + clz32(UInt32(left) ⊻ UInt32(right)))
    end
end
```

#### Step 5: Compute AABBs Bottom-Up

```julia
# Create leaf nodes with primitive AABBs
for i in 1:n
    leaf_idx = leaf_index(i, n)
    tri_aabb = world_bound(sorted_prims[i])
    nodes[leaf_idx] = create_leaf(tri_aabb, i)
end

# Propagate AABBs upward
for i in (n-1):-1:1
    child0_aabb = get_node_aabb(nodes[nodes[i].child0])
    child1_aabb = get_node_aabb(nodes[nodes[i].child1])
    nodes[i] = create_interior(child0_aabb ∪ child1_aabb, ...)
end
```

## Traversal Algorithm

### Two-Level Traversal with Stack

```julia
function closest_hit(tlas::TLAS, ray::AbstractRay)
    # Initialize state
    stack = MVector{64, UInt32}(undef)
    stack_ptr = 1
    current_node_idx = 1
    current_instance = INVALID_NODE

    while current_node_idx != INVALID_NODE
        # Fetch node (from TLAS or BLAS)
        node = (current_instance == INVALID_NODE) ?
               tlas.nodes[current_node_idx] :
               tlas.blas_array[current_instance].nodes[current_node_idx]

        # Test ray-AABB intersection
        if intersect_aabb(node, ray)
            if is_leaf(node)
                if current_instance == INVALID_NODE
                    # Top-level leaf: transition to BLAS
                    instance_idx = node.child1
                    inst = tlas.instances[instance_idx]

                    # Push sentinel
                    stack[stack_ptr++] = TOP_LEVEL_SENTINEL

                    # Transform ray to local space
                    ray_local = transform_ray(inst.inv_transform, ray)

                    # Switch to BLAS traversal
                    current_instance = inst.blas_index
                    current_node_idx = 1
                else
                    # Bottom-level leaf: test triangle
                    test_triangle_intersection(...)
                end
            else
                # Interior node: push far, visit near
                push_and_traverse(...)
            end
        else
            # Pop from stack
            current_node_idx = stack[--stack_ptr]

            # Check for level transition
            if current_node_idx == TOP_LEVEL_SENTINEL
                current_node_idx = stack[--stack_ptr]
                current_instance = INVALID_NODE
                ray = restore_original_ray()
            end
        end
    end
end
```

### Key Features

1. **Sentinel-based Level Switching**: `TOP_LEVEL_SENTINEL (0xFFFFFFFE)` marks transitions
2. **Ray Transformation**: Transform once when entering BLAS, restore when returning
3. **Hybrid Stack**: LDS (16 entries) + global (64 entries) for GPU efficiency
4. **Ordered Traversal**: Visit near child first for early termination

## Type Stability

### Critical for GPU Performance

All functions must be **type-stable** to generate efficient GPU kernels:

```julia
# ✓ Type-stable
@inline function get_node_aabb(node::BVHNode2, is_interior::Bool)::Bounds3
    if is_interior
        Bounds3(min.(node.aabb0_min, node.aabb1_min),
                max.(node.aabb0_max, node.aabb1_max))
    else
        Bounds3(node.aabb0_min, node.aabb0_max)
    end
end

# ✗ NOT type-stable (return type depends on runtime value)
function bad_example(node::BVHNode2, flag::Bool)
    if flag
        return node.aabb0_min  # Point3f
    else
        return node.child0     # UInt32
    end
end
```

### Testing Type Stability

```julia
using Test

@testset "Type Stability" begin
    node = BVHNode2(...)

    # Should infer return type at compile time
    result = @inferred get_node_aabb(node, true)
    @test result isa Bounds3
end
```

## GPU Portability

### Parametrized Array Types

```julia
# CPU
cpu_blas = BLAS{Vector{BVHNode2}, Vector{Triangle}}(...)

# CUDA
using CUDA
gpu_nodes = CuArray(cpu_blas.nodes)
gpu_tris = CuArray(cpu_blas.primitives)
gpu_blas = BLAS{CuArray{BVHNode2}, CuArray{Triangle}}(
    gpu_nodes, gpu_tris, cpu_blas.root_aabb
)
```

### KernelAbstractions Integration

```julia
using KernelAbstractions

@kernel function traverse_rays_kernel!(hits, @Const(tlas), @Const(rays))
    i = @index(Global)
    ray = rays[i]
    hit, tri, t, bary, inst_id = closest_hit(tlas, ray)
    hits[i] = (hit, t, inst_id)
end

# Execute on different backends
backend = get_backend(rays)  # CPU, CUDA, ROC, oneAPI, etc.
kernel! = traverse_rays_kernel!(backend)
kernel!(hits, tlas, rays, ndrange=length(rays))
```

## Performance Characteristics

### Construction

- **LBVH**: O(n log n) due to sorting
- **Parallel-friendly**: Each internal node computed independently
- **Memory**: ~64 bytes per node × (2n-1) nodes

### Traversal

- **Stack depth**: Typically 20-40 for balanced trees, max 64
- **Memory access**: Sequential node reads (cache-friendly)
- **Transform overhead**: 2 matrix-vector mults per instance transition

### Instancing Benefits

**Memory Savings:**
- Without instancing: n_instances × n_triangles × sizeof(Triangle)
- With instancing: n_triangles × sizeof(Triangle) + n_instances × 128 bytes
- **Example**: 1000 cubes = 98% memory reduction

**Update Performance:**
- Updating instance transform: O(1) + TLAS rebuild
- No BLAS rebuild needed for rigid transformations

## Comparison to Radeon Rays

### Similarities

✓ BVH2IL node layout (inline children AABBs)
✓ LBVH construction with 30-bit Morton codes
✓ Two-level hierarchy (TLAS/BLAS)
✓ Sentinel-based level switching
✓ Transform caching in instances

### Differences

- **No SAH restructuring**: RadeonRays has optional treelet optimization
- **Simplified construction**: No multi-threaded workgroup coordination
- **Julia-specific**: Leverages multiple dispatch and type system
- **No hardware RT**: Pure software traversal (can use RT when available)

## Future Enhancements

### Short Term

1. **Complete triangle intersection in traversal**
2. **Implement any_hit for occlusion queries**
3. **Add multi-instance TLAS builder** (currently only single instance)

### Medium Term

1. **SAH-based restructuring** (treelet optimization)
2. **Compressed-wide BVH (CWBVH)** for higher branching factor
3. **Hardware ray tracing backend** (Vulkan RT, OptiX, DXR)

### Long Term

1. **Dynamic BVH updates** (refit without full rebuild)
2. **Streaming geometry** (out-of-core BVH)
3. **Neural BVH optimization** (learned splitting)

## References

1. Karras, T. (2012). "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
2. AMD RadeonRays SDK: https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK
3. Ylitie, H. et al. (2017). "Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs"
4. Meister, D. et al. (2020). "A Survey on Bounding Volume Hierarchies for Ray Tracing"

## Example Usage

```julia
using Raycore
using GeometryBasics
import KernelAbstractions as KA

# Create geometry and build TLAS using the high-level API
cube_mesh = normal_mesh(Rect3f(Point3f(-0.5), Vec3f(1.0)))

tlas = Raycore.TLAS(KA.CPU())

# Add 10 instances of the cube at different positions
for i in 1:10
    t = Raycore.translate(Vec3f(i * 2, 0, 0))
    push!(tlas, cube_mesh, t.m)
end

Raycore.sync!(tlas)

# Get immutable StaticTLAS for traversal
static = Adapt.adapt(KA.CPU(), tlas)

# Trace ray
ray = Ray(o=Point3f(0, 0, -5), d=Vec3f(0, 0, 1))
hit, tri, t, bary, inst_id = closest_hit(static, ray)

if hit
    println("Hit instance $inst_id at distance $t")
    hit_point = ray.o + ray.d * t
    println("Hit point: $hit_point")
end
```
