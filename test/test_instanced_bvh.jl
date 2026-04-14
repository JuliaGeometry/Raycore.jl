# ==============================================================================
# Instanced BVH Tests
# ==============================================================================

using Test
using Raycore
using GeometryBasics
using StaticArrays
using LinearAlgebra
using KernelAbstractions

# Use qualified names to avoid conflicts with other packages
const RTriangle = Raycore.Triangle   # Conflicts with GeometryBasics.Triangle
const RBLAS = Raycore.BLAS           # Conflicts with LinearAlgebra.BLAS
const is_leaf = Raycore.is_leaf
const is_interior = Raycore.is_interior

@testset "Instanced BVH" begin

@testset "Morton Code Generation" begin
    # Test Morton code for known points
    p1 = Point3f(0.0, 0.0, 0.0)
    p2 = Point3f(1.0, 1.0, 1.0)
    p3 = Point3f(0.5, 0.5, 0.5)

    code1 = Raycore.morton_code_30bit(p1)
    code2 = Raycore.morton_code_30bit(p2)
    code3 = Raycore.morton_code_30bit(p3)

    @test code1 isa UInt32
    @test code2 isa UInt32
    @test code3 isa UInt32

    # Morton codes should order points along Z-curve
    @test code1 < code2
    @test code1 < code3 < code2
end

@testset "BLAS Construction - Single Triangle" begin
    # Create a single triangle
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        nothing
    )

    primitives = [tri]
    blas = build_blas(primitives)

    @test blas isa RBLAS
    @test length(blas.nodes) == 1  # Single triangle = 1 node (leaf)
    @test length(blas.primitives) == 1
    @test is_leaf(blas.nodes[1])
    @test blas.nodes[1].child1 == UInt32(1)  # Points to primitive 1
end

@testset "BLAS Construction - Multiple Triangles" begin
    # Create a simple quad (2 triangles)
    v1 = Point3f(0, 0, 0)
    v2 = Point3f(1, 0, 0)
    v3 = Point3f(1, 1, 0)
    v4 = Point3f(0, 1, 0)

    tri1 = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(1, 1)),
        nothing
    )

    tri2 = RTriangle(
        SVector(v1, v3, v4),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 1), Point2f(0, 1)),
        nothing
    )

    primitives = [tri1, tri2]
    blas = build_blas(primitives)

    @test blas isa RBLAS
    @test length(blas.nodes) == 3  # 2 leaves + 1 interior = 3 nodes
    @test length(blas.primitives) == 2

    # Check root is interior node
    @test is_interior(blas.nodes[1])

    # Check root AABB contains all primitives
    root_aabb = blas.root_aabb
    @test root_aabb.p_min[1] ≈ 0.0f0
    @test root_aabb.p_min[2] ≈ 0.0f0
    @test root_aabb.p_max[1] ≈ 1.0f0
    @test root_aabb.p_max[2] ≈ 1.0f0
end

@testset "BLAS Type Stability" begin
    # Test type stability of build_blas
    v1 = Point3f(0, 0, 0)
    v2 = Point3f(1, 0, 0)
    v3 = Point3f(0, 1, 0)

    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        nothing
    )

    primitives = [tri]

    # build_blas should be type-stable
    result_type = @inferred build_blas(primitives)
    @test result_type isa RBLAS
end

@testset "Transform Utilities" begin
    # Test point transformation
    identity = Mat4f(I)
    p = Point3f(1, 2, 3)
    p_transformed = Raycore.transform_point(identity, p)
    @test p_transformed ≈ p

    # Test translation
    translation = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        5, 10, 15, 1
    )
    p_translated = Raycore.transform_point(translation, p)
    @test p_translated ≈ Point3f(6, 12, 18)

    # Test direction transformation (should ignore translation)
    v = Vec3f(1, 0, 0)
    v_transformed = Raycore.transform_direction(translation, v)
    @test v_transformed ≈ v

    # Test type stability
    @test (@inferred Raycore.transform_point(identity, p)) isa Point3f
    @test (@inferred Raycore.transform_direction(identity, v)) isa Vec3f
end

@testset "BVHNode2 Utilities" begin
    # Test leaf detection
    leaf_node = BVHNode2(
        Point3f(0), Point3f(1),
        Point3f(0), Point3f(0),
        INVALID_NODE, UInt32(5), INVALID_NODE
    )
    @test is_leaf(leaf_node)
    @test !is_interior(leaf_node)

    # Test interior detection
    interior_node = BVHNode2(
        Point3f(0), Point3f(1),
        Point3f(0), Point3f(1),
        UInt32(2), UInt32(3), INVALID_NODE
    )
    @test !is_leaf(interior_node)
    @test is_interior(interior_node)

    # Test AABB extraction
    aabb = Raycore.get_node_aabb(interior_node, true)
    @test aabb isa Bounds3
    @test aabb.p_min == Point3f(0, 0, 0)
    @test aabb.p_max == Point3f(1, 1, 1)
end

@testset "AABB Utilities" begin
    # Test expand_bits
    @test Raycore.expand_bits(UInt32(0)) == UInt32(0)
    @test Raycore.expand_bits(UInt32(1)) isa UInt32

    # Test clz32
    @test Raycore.clz32(UInt32(0)) == Int32(32)
    @test Raycore.clz32(UInt32(1)) == Int32(31)
    @test Raycore.clz32(UInt32(0x80000000)) == Int32(0)
end

@testset "Delta Function (LCP)" begin
    # Test longest common prefix calculation
    codes = UInt32[0x00000001, 0x00000002, 0x00000004, 0x00000008]

    # Adjacent codes with different prefixes
    d1 = Raycore.delta(Int32(1), Int32(2), codes, Int32(4))
    d2 = Raycore.delta(Int32(2), Int32(3), codes, Int32(4))

    @test d1 isa Int32
    @test d2 isa Int32

    # Out of bounds should return -1
    d_oob = Raycore.delta(Int32(1), Int32(10), codes, Int32(4))
    @test d_oob == Int32(-1)
end

# ==============================================================================
# TLAS Construction Tests
# ==============================================================================

@testset "TLAS Construction - Single Instance" begin
    # Create a single triangle as BLAS
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(1)
    )

    blas = build_blas([tri])
    identity = Mat4f(I)
    instances = [InstanceDescriptor(UInt32(1), UInt32(1), identity, identity, UInt32(0))]

    tlas = build_tlas([blas], instances)

    @test tlas isa Raycore.StaticTLAS
    @test length(tlas.instances) == 1
    @test length(tlas.blas_descriptors) == 1
    @test length(tlas.nodes) == 1  # Single instance = 1 node (leaf)
end

@testset "TLAS Construction - Multiple Instances" begin
    # Create a triangle BLAS
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(1)
    )

    blas = build_blas([tri])

    # Create two instances with different transforms
    identity = Mat4f(I)
    translation = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        5, 0, 0, 1
    )
    inv_translation = Mat4f(inv(translation))

    instances = [
        InstanceDescriptor(UInt32(1), UInt32(1), identity, identity, UInt32(0)),
        InstanceDescriptor(UInt32(1), UInt32(2), translation, inv_translation, UInt32(0))
    ]

    tlas = build_tlas([blas], instances)

    @test tlas isa Raycore.StaticTLAS
    @test length(tlas.instances) == 2
    @test length(tlas.blas_descriptors) == 1
    @test length(tlas.nodes) == 3  # 2 leaves + 1 interior = 3 nodes

    # World bound should encompass both instances
    wb = world_bound(tlas)
    @test wb.p_min[1] ≈ 0.0f0
    @test wb.p_max[1] ≈ 6.0f0  # Original + translated
end

# ==============================================================================
# TLAS Ray Intersection Tests
# ==============================================================================

@testset "TLAS closest_hit - Basic" begin
    # Create a unit triangle in XY plane at z=0
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(42)
    )

    blas = build_blas([tri])
    identity = Mat4f(I)
    instances = [InstanceDescriptor(UInt32(1), UInt32(1), identity, identity, UInt32(0))]
    tlas = build_tlas([blas], instances)

    # Ray pointing down at center of triangle
    ray = Ray(o=Point3f(0.25, 0.25, 1.0), d=Vec3f(0, 0, -1))
    hit, prim, dist, bary, inst_id = closest_hit(tlas, ray)

    @test hit == true
    @test dist ≈ 1.0f0
    @test prim.metadata == UInt32(42)

    # Ray missing the triangle
    ray_miss = Ray(o=Point3f(2, 2, 1.0), d=Vec3f(0, 0, -1))
    hit_miss, _, _, _, _ = closest_hit(tlas, ray_miss)
    @test hit_miss == false
end

@testset "TLAS closest_hit - Transformed Instance" begin
    # Create a unit triangle
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(1)
    )

    blas = build_blas([tri])

    # Translate instance by (10, 0, 0)
    translation = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        10, 0, 0, 1
    )
    inv_translation = Mat4f(inv(translation))
    instances = [InstanceDescriptor(UInt32(1), UInt32(1), translation, inv_translation, UInt32(0))]

    tlas = build_tlas([blas], instances)

    # Ray at original position should miss
    ray_miss = Ray(o=Point3f(0.25, 0.25, 1.0), d=Vec3f(0, 0, -1))
    hit_miss, _, _, _, _ = closest_hit(tlas, ray_miss)
    @test hit_miss == false

    # Ray at translated position should hit
    ray_hit = Ray(o=Point3f(10.25, 0.25, 1.0), d=Vec3f(0, 0, -1))
    hit, _, dist, _, _ = closest_hit(tlas, ray_hit)
    @test hit == true
    @test dist ≈ 1.0f0
end

@testset "TLAS closest_hit - Multiple Instances (Closest Selection)" begin
    # Create a unit triangle
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(1)
    )

    blas = build_blas([tri])
    identity = Mat4f(I)

    # Two instances: one at z=0, one at z=-5 (further from camera)
    translate_back = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, -5, 1
    )
    inv_translate_back = Mat4f(inv(translate_back))

    instances = [
        InstanceDescriptor(UInt32(1), UInt32(1), identity, identity, UInt32(0)),
        InstanceDescriptor(UInt32(1), UInt32(2), translate_back, inv_translate_back, UInt32(0))
    ]

    tlas = build_tlas([blas], instances)

    # Ray should hit the closer one (z=0)
    ray = Ray(o=Point3f(0.25, 0.25, 1.0), d=Vec3f(0, 0, -1))
    hit, _, dist, _, inst_id = closest_hit(tlas, ray)

    @test hit == true
    @test dist ≈ 1.0f0  # Distance to z=0 plane
    @test inst_id == UInt32(1)  # First instance
end

@testset "TLAS any_hit - Basic" begin
    # Create a unit triangle
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(1)
    )

    blas = build_blas([tri])
    identity = Mat4f(I)
    instances = [InstanceDescriptor(UInt32(1), UInt32(1), identity, identity, UInt32(0))]
    tlas = build_tlas([blas], instances)

    # Ray hitting triangle
    ray = Ray(o=Point3f(0.25, 0.25, 1.0), d=Vec3f(0, 0, -1))
    hit, _, _, _, _ = any_hit(tlas, ray)
    @test hit == true

    # Ray missing triangle
    ray_miss = Ray(o=Point3f(2, 2, 1.0), d=Vec3f(0, 0, -1))
    hit_miss, _, _, _, _ = any_hit(tlas, ray_miss)
    @test hit_miss == false
end

# ==============================================================================
# GB.Mesh TLAS API Tests
# ==============================================================================

# Helper to create a GB.Mesh with normals
function make_test_mesh(verts, normals)
    faces = [GLTriangleFace(1, 2, 3)]
    GeometryBasics.mesh(verts, faces; normal=normals)
end

@testset "TLASHandle and n_instances" begin
    mesh1 = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = make_test_mesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([mesh1, mesh2])

    @test length(handles) == 2
    @test handles[1] isa TLASHandle
    @test handles[2] isa TLASHandle

    count1 = n_instances(tlas, handles[1])
    count2 = n_instances(tlas, handles[2])

    @test count1 == 1
    @test count2 == 1
    @test is_valid(tlas, handles[1])
    @test is_valid(tlas, handles[2])
end

@testset "TLAS with multi-transform push!" begin
    mesh1 = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = make_test_mesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # Use multi-transform push! for mesh1 (instancing)
    transforms = [
        Mat4f(I),
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1)
    ]

    tlas = Raycore.TLAS(KernelAbstractions.CPU())
    h1 = push!(tlas, mesh1, transforms)
    h2 = push!(tlas, mesh2)
    sync!(tlas)

    @test n_geometries(tlas) == 2  # 2 unique BLAS
    @test n_instances(tlas) == 3   # 2 + 1 = 3 instance descriptors

    @test n_instances(tlas, h1) == 2  # First handle has 2 instances
    @test n_instances(tlas, h2) == 1  # Second handle has 1 instance
end

@testset "TLAS from GB.Mesh Vector" begin
    mesh1 = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = make_test_mesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([mesh1, mesh2])

    @test n_geometries(tlas) == 2
    @test n_instances(tlas) == 2
    @test length(handles) == 2
end

# ==============================================================================
# Dynamic Update Tests
# ==============================================================================

@testset "update_transform! (single instance)" begin
    mesh = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([mesh])

    new_transform = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        10, 0, 0, 1
    )
    update_transform!(tlas, handles[1], new_transform)

    @test get_instance(tlas, handles[1]).transform ≈ new_transform
end

@testset "update_transforms! (multiple instances)" begin
    mesh = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # Create with 3 transforms using multi-transform push!
    initial_transforms = [Mat4f(I), Mat4f(I), Mat4f(I)]
    tlas = Raycore.TLAS(KernelAbstractions.CPU())
    h = push!(tlas, mesh, initial_transforms)
    sync!(tlas)
    handles = [h]

    # Update all transforms
    new_transforms = [
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1),
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1),
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1)
    ]
    update_transforms!(tlas, handles[1], new_transforms)

    instances = get_instances(tlas, handles[1])
    for (i, inst) in enumerate(instances)
        @test inst.transform ≈ new_transforms[i]
    end
end

@testset "push! GB.Mesh" begin
    mesh1 = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = make_test_mesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([mesh1])

    @test n_geometries(tlas) == 1
    @test n_instances(tlas) == 1

    # Add new mesh using push! + sync!
    new_handle = push!(tlas, mesh2)
    sync!(tlas)

    @test n_geometries(tlas) == 2
    @test n_instances(tlas) == 2
    @test new_handle isa TLASHandle
end

@testset "delete! and sync!" begin
    mesh1 = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = make_test_mesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([mesh1, mesh2])

    @test n_instances(tlas) == 2
    @test is_valid(tlas, handles[1])
    @test is_valid(tlas, handles[2])

    deleted = delete!(tlas, handles[1])
    @test deleted == true

    @test !is_valid(tlas, handles[1])

    sync!(tlas)

    @test n_instances(tlas) == 1
    @test is_valid(tlas, handles[2])
end

# ==============================================================================
# Type Stability Tests for TLAS
# ==============================================================================

@testset "TLAS Type Stability" begin
    # Create triangle with concrete metadata type
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(1)
    )

    blas = build_blas([tri])
    identity = Mat4f(I)
    instances = [InstanceDescriptor(UInt32(1), UInt32(1), identity, identity, UInt32(0))]
    tlas = build_tlas([blas], instances)

    ray = Ray(o=Point3f(0.25, 0.25, 1.0), d=Vec3f(0, 0, -1))

    # Test type stability of closest_hit
    result_type = @inferred closest_hit(tlas, ray)
    @test result_type[1] isa Bool
    @test result_type[2] isa RTriangle{UInt32}
    @test result_type[3] isa Float32
    @test result_type[4] isa SVector{3, Float32}
    @test result_type[5] isa UInt32

    # Test type stability of any_hit
    result_type_any = @inferred any_hit(tlas, ray)
    @test result_type_any[1] isa Bool
end

@testset "TLAS eltype" begin
    # Verify eltype returns the correct triangle type without indexing into arrays
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    tri = RTriangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        UInt32(1)
    )

    blas = build_blas([tri])
    identity = Mat4f(I)
    instances = [InstanceDescriptor(UInt32(1), UInt32(1), identity, identity, UInt32(0))]
    tlas = build_tlas([blas], instances)

    @test eltype(tlas) == RTriangle{UInt32}
end

@testset "n_instances and n_geometries" begin
    mesh = make_test_mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # 5 instances of same geometry using multi-transform push!
    transforms = [Mat4f(I) for _ in 1:5]
    tlas = Raycore.TLAS(KernelAbstractions.CPU())
    push!(tlas, mesh, transforms)
    sync!(tlas)

    @test n_geometries(tlas) == 1
    @test n_instances(tlas) == 5
end

end # main testset "Instanced BVH"

# ==============================================================================
# KernelAbstractions Dynamic Scene Tests (OpenCL backend via pocl)
# ==============================================================================

# Load packages at top-level (required for KA.@kernel macro)
using pocl_jll
using OpenCL
import KernelAbstractions as KA
using KernelAbstractions: @index, @Const
import Adapt

# ===========================================================================
# Kernel definitions - at top-level so KA.@kernel macro works
# ===========================================================================

# Kernel 1: Basic closest_hit - returns hit/distance
KA.@kernel function closest_hit_kernel!(hits, distances, tlas, origins, directions)
    i = @index(Global, Linear)
    @inbounds begin
        ray = Ray(o=origins[i], d=directions[i])
        hit, _, dist, _, _ = closest_hit(tlas, ray)
        hits[i] = hit
        distances[i] = dist
    end
end

# Kernel 2: any_hit for shadow/occlusion testing
KA.@kernel function any_hit_kernel!(hits, tlas, origins, directions)
    i = @index(Global, Linear)
    @inbounds begin
        ray = Ray(o=origins[i], d=directions[i])
        hit, _, _, _, _ = any_hit(tlas, ray)
        hits[i] = hit
    end
end

# Kernel 3: closest_hit with instance ID retrieval
KA.@kernel function closest_hit_instance_id_kernel!(hits, distances, instance_ids, tlas, origins, directions)
    i = @index(Global, Linear)
    @inbounds begin
        ray = Ray(o=origins[i], d=directions[i])
        hit, _, dist, _, inst_id = closest_hit(tlas, ray)
        hits[i] = hit
        distances[i] = dist
        instance_ids[i] = inst_id
    end
end

# Kernel 4: closest_hit with primitive metadata retrieval
KA.@kernel function closest_hit_metadata_kernel!(hits, metadata_out, tlas, origins, directions)
    i = @index(Global, Linear)
    @inbounds begin
        ray = Ray(o=origins[i], d=directions[i])
        hit, prim, _, _, _ = closest_hit(tlas, ray)
        hits[i] = hit
        # Only access metadata if hit
        metadata_out[i] = hit ? prim.metadata : UInt32(0)
    end
end

# Kernel 5: closest_hit with barycentric coordinates
KA.@kernel function closest_hit_bary_kernel!(hits, barys, tlas, origins, directions)
    i = @index(Global, Linear)
    @inbounds begin
        ray = Ray(o=origins[i], d=directions[i])
        hit, _, _, bary, _ = closest_hit(tlas, ray)
        hits[i] = hit
        barys[i] = bary
    end
end

# Kernel 6: Batch trace with all outputs (stress test)
KA.@kernel function full_trace_kernel!(hits, distances, instance_ids, metadata_out, barys, tlas, origins, directions)
    i = @index(Global, Linear)
    @inbounds begin
        ray = Ray(o=origins[i], d=directions[i])
        hit, prim, dist, bary, inst_id = closest_hit(tlas, ray)
        hits[i] = hit
        distances[i] = dist
        instance_ids[i] = inst_id
        metadata_out[i] = hit ? prim.metadata : UInt32(0)
        barys[i] = bary
    end
end

# GPU kernel compilation is incompatible with --check-bounds=yes (Pkg.test default)
# because bounds checking injects error-throwing paths that can't compile to SPIR-V.
# Use: Pkg.test("Raycore"; julia_args=`--check-bounds=auto`)
if Base.JLOptions().check_bounds == 1  # 1 = --check-bounds=yes
    @testset "KernelAbstractions Dynamic Scenes (OpenCL/pocl)" begin
        @test_broken false  # skipped: --check-bounds=yes is incompatible with GPU kernel compilation
    end
else
@testset "KernelAbstractions Dynamic Scenes (OpenCL/pocl)" begin
    # Must select pocl platform/device before creating the backend
    pocl_platform = OpenCL.cl.platforms()[1]
    pocl_device = OpenCL.cl.devices(pocl_platform)[1]
    OpenCL.cl.device!(pocl_device)
    cl_backend = OpenCL.OpenCLBackend()

    # Helper to create a simple GB.Mesh
    function make_triangle_mesh(offset::Vec3f=Vec3f(0, 0, 0))
        verts = [
            Point3f(0, 0, 0) + offset,
            Point3f(1, 0, 0) + offset,
            Point3f(0, 1, 0) + offset
        ]
        norms = fill(Normal3f(0, 0, 1), 3)
        faces = [GLTriangleFace(1, 2, 3)]
        return GeometryBasics.mesh(verts, faces; normal=norms)
    end

    @testset "TLAS adapt to CLArray" begin
        mesh = make_triangle_mesh()
        tlas, handles = TLAS([mesh]; backend=cl_backend)

        # Adapt TLAS to OpenCL arrays (GPU-first: backend must match)
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        @test cl_tlas isa Raycore.StaticTLAS
        # GPU arrays (CLArray) are not isbits on the host — KA handles
        # the device pointer conversion during kernel launch.
        # The kernel tests below verify that the TLAS works correctly on GPU.
        @test cl_tlas.nodes isa CLArray
    end

    @testset "TLAS sync with many instances" begin
        mesh = make_triangle_mesh()
        transforms = [Mat4f(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            Float32(mod(i - 1, 9)) * 1.5f0,
                            Float32((i - 1) ÷ 9) * 1.25f0,
                            0,
                            1) for i in 1:81]

        tlas = Raycore.TLAS(cl_backend)
        push!(tlas, mesh, transforms)
        sync!(tlas)

        @test length(tlas.instances) == 81
        @test length(tlas.nodes) == 161
        @test Raycore.world_bound(tlas) isa Bounds3
    end

    @testset "closest_hit_kernel! - basic intersection" begin
        mesh = make_triangle_mesh()
        tlas, _ = TLAS([mesh]; backend=cl_backend)
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        n = 4
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)
        distances = KA.allocate(cl_backend, Float32, n)

        # Test rays: 2 hits, 2 misses
        KA.copyto!(cl_backend, origins, [
            Point3f(0.25, 0.25, 1.0),  # hit
            Point3f(0.5, 0.25, 1.0),   # hit
            Point3f(5.0, 5.0, 1.0),    # miss
            Point3f(-1.0, -1.0, 1.0)   # miss
        ])
        KA.copyto!(cl_backend, directions, fill(Vec3f(0, 0, -1), n))

        kernel = closest_hit_kernel!(cl_backend)
        kernel(hits, distances, cl_tlas, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        distances_cpu = Array(distances)

        @test hits_cpu[1] == true
        @test hits_cpu[2] == true
        @test hits_cpu[3] == false
        @test hits_cpu[4] == false
        @test distances_cpu[1] ≈ 1.0f0
        @test distances_cpu[2] ≈ 1.0f0
    end

    @testset "any_hit_kernel! - shadow/occlusion test" begin
        mesh = make_triangle_mesh()
        tlas, _ = TLAS([mesh]; backend=cl_backend)
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        n = 4
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)

        # Test rays
        KA.copyto!(cl_backend, origins, [
            Point3f(0.25, 0.25, 1.0),  # hit
            Point3f(0.1, 0.1, 1.0),    # hit
            Point3f(5.0, 5.0, 1.0),    # miss
            Point3f(0.9, 0.9, 1.0)     # miss (outside triangle)
        ])
        KA.copyto!(cl_backend, directions, fill(Vec3f(0, 0, -1), n))

        kernel = any_hit_kernel!(cl_backend)
        kernel(hits, cl_tlas, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        @test hits_cpu[1] == true
        @test hits_cpu[2] == true
        @test hits_cpu[3] == false
        @test hits_cpu[4] == false
    end

    @testset "closest_hit_instance_id_kernel! - instance identification" begin
        mesh = make_triangle_mesh()

        # Three instances at different positions. Traversal returns the
        # 1-based instance array index; here we push 3 instances so each
        # ray hits position 1, 2, 3.
        transforms = [
            Mat4f(I),  # Instance 1 at origin
            Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1),   # Instance 2 at x=5
            Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1)    # Instance 3 at y=5
        ]
        tlas_tmp = Raycore.TLAS(cl_backend)
        push!(tlas_tmp, mesh, transforms)
        sync!(tlas_tmp)
        tlas = tlas_tmp
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        n = 3
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)
        distances = KA.allocate(cl_backend, Float32, n)
        instance_ids = KA.allocate(cl_backend, UInt32, n)

        # Each ray targets a different instance
        KA.copyto!(cl_backend, origins, [
            Point3f(0.25, 0.25, 1.0),   # hits instance 1
            Point3f(5.25, 0.25, 1.0),   # hits instance 2
            Point3f(0.25, 5.25, 1.0)    # hits instance 3
        ])
        KA.copyto!(cl_backend, directions, fill(Vec3f(0, 0, -1), n))

        kernel = closest_hit_instance_id_kernel!(cl_backend)
        kernel(hits, distances, instance_ids, cl_tlas, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        instance_ids_cpu = Array(instance_ids)

        @test all(hits_cpu)
        # closest_hit returns the 1-based instance array index
        @test instance_ids_cpu[1] == UInt32(1)
        @test instance_ids_cpu[2] == UInt32(2)
        @test instance_ids_cpu[3] == UInt32(3)
    end

    @testset "closest_hit_metadata_kernel! - primitive metadata" begin
        # Create meshes at different positions (metadata test simplified - mesh default is 0)
        mesh1 = make_triangle_mesh(Vec3f(0, 0, 0))
        mesh2 = make_triangle_mesh(Vec3f(5, 0, 0))
        mesh3 = make_triangle_mesh(Vec3f(0, 5, 0))

        tlas, _ = TLAS([mesh1, mesh2, mesh3]; backend=cl_backend)
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        n = 4
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)
        metadata_out = KA.allocate(cl_backend, UInt32, n)

        KA.copyto!(cl_backend, origins, [
            Point3f(0.25, 0.25, 1.0),   # hits mesh1
            Point3f(5.25, 0.25, 1.0),   # hits mesh2
            Point3f(0.25, 5.25, 1.0),   # hits mesh3
            Point3f(10.0, 10.0, 1.0)    # miss
        ])
        KA.copyto!(cl_backend, directions, fill(Vec3f(0, 0, -1), n))

        kernel = closest_hit_metadata_kernel!(cl_backend)
        kernel(hits, metadata_out, cl_tlas, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)

        @test hits_cpu[1] == true
        @test hits_cpu[2] == true
        @test hits_cpu[3] == true
        @test hits_cpu[4] == false
        # Note: metadata from mesh is 0 by default, so we just test hits work
    end

    @testset "closest_hit_bary_kernel! - barycentric coordinates" begin
        mesh = make_triangle_mesh()
        tlas, _ = TLAS([mesh]; backend=cl_backend)
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        n = 3
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)
        barys = KA.allocate(cl_backend, SVector{3, Float32}, n)

        # Triangle vertices: (0,0,0), (1,0,0), (0,1,0)
        # Hit points chosen to give predictable barycentrics
        KA.copyto!(cl_backend, origins, [
            Point3f(0.25, 0.25, 1.0),  # should give bary ≈ (0.25, 0.25, 0.5)
            Point3f(0.1, 0.1, 1.0),    # should give bary ≈ (0.1, 0.1, 0.8)
            Point3f(0.5, 0.0, 1.0)     # edge hit, bary ≈ (0.5, 0.0, 0.5)
        ])
        KA.copyto!(cl_backend, directions, fill(Vec3f(0, 0, -1), n))

        kernel = closest_hit_bary_kernel!(cl_backend)
        kernel(hits, barys, cl_tlas, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        barys_cpu = Array(barys)

        @test all(hits_cpu)
        # Barycentrics are (w, u, v) where w = 1-u-v
        # For hit at (0.25, 0.25): u=0.25, v=0.25, w=0.5
        @test barys_cpu[1][1] ≈ 0.5f0 atol=0.01   # w
        @test barys_cpu[1][2] ≈ 0.25f0 atol=0.01  # u
        # For hit at (0.1, 0.1): u=0.1, v=0.1, w=0.8
        @test barys_cpu[2][1] ≈ 0.8f0 atol=0.01   # w
        @test barys_cpu[2][2] ≈ 0.1f0 atol=0.01   # u
        # For edge hit at (0.5, 0.0): u=0.5, v=0.0, w=0.5
        @test barys_cpu[3][1] ≈ 0.5f0 atol=0.01   # w
        @test barys_cpu[3][2] ≈ 0.5f0 atol=0.01   # u
    end

    @testset "full_trace_kernel! - comprehensive output" begin
        mesh1 = make_triangle_mesh(Vec3f(0, 0, 0))
        mesh2 = make_triangle_mesh(Vec3f(5, 0, 0))

        # Two default-override (inherit) instances; closest_hit returns
        # their 1-based array positions (1 and 2).
        tlas, _ = TLAS([mesh1, mesh2]; backend=cl_backend)
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        n = 3
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)
        distances = KA.allocate(cl_backend, Float32, n)
        instance_ids = KA.allocate(cl_backend, UInt32, n)
        metadata_out = KA.allocate(cl_backend, UInt32, n)
        barys = KA.allocate(cl_backend, SVector{3, Float32}, n)

        KA.copyto!(cl_backend, origins, [
            Point3f(0.25, 0.25, 2.0),   # hits mesh1 at dist=2
            Point3f(5.25, 0.25, 3.0),   # hits mesh2 at dist=3
            Point3f(10.0, 10.0, 1.0)    # miss
        ])
        KA.copyto!(cl_backend, directions, fill(Vec3f(0, 0, -1), n))

        kernel = full_trace_kernel!(cl_backend)
        kernel(hits, distances, instance_ids, metadata_out, barys, cl_tlas, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        distances_cpu = Array(distances)
        instance_ids_cpu = Array(instance_ids)
        barys_cpu = Array(barys)

        @test hits_cpu[1] == true
        @test hits_cpu[2] == true
        @test hits_cpu[3] == false

        @test distances_cpu[1] ≈ 2.0f0
        @test distances_cpu[2] ≈ 3.0f0

        @test instance_ids_cpu[1] == UInt32(1)
        @test instance_ids_cpu[2] == UInt32(2)

        # Barycentrics are (w, u, v) where w = 1-u-v
        # For hit at (0.25, 0.25): u=0.25, v=0.25, w=0.5
        @test barys_cpu[1][1] ≈ 0.5f0 atol=0.01  # w
        @test barys_cpu[2][1] ≈ 0.5f0 atol=0.01  # w
    end

    @testset "Dynamic transform updates via kernel" begin
        mesh = make_triangle_mesh()

        # Create mutable TLAS with backend for dynamic updates
        tlas = Raycore.TLAS(cl_backend)
        push!(tlas, mesh)
        Raycore.sync!(tlas)

        # Initial position: ray at origin should hit
        cl_tlas1 = Adapt.adapt(cl_backend, tlas)

        n = 1
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)
        distances = KA.allocate(cl_backend, Float32, n)

        KA.copyto!(cl_backend, origins, [Point3f(0.25, 0.25, 1.0)])
        KA.copyto!(cl_backend, directions, [Vec3f(0, 0, -1)])

        kernel = closest_hit_kernel!(cl_backend)
        kernel(hits, distances, cl_tlas1, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)
        @test Array(hits)[1] == true

        # Update transform: move to x=10 (use index-based API)
        new_transform = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1)
        update_instance_transform!(tlas, 1, new_transform)
        refit_tlas!(tlas)

        # Adapt again after update
        cl_tlas2 = Adapt.adapt(cl_backend, tlas)

        # Now ray at origin should miss
        kernel(hits, distances, cl_tlas2, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)
        @test Array(hits)[1] == false

        # Ray at x=10 should hit
        KA.copyto!(cl_backend, origins, [Point3f(10.25, 0.25, 1.0)])
        kernel(hits, distances, cl_tlas2, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)
        @test Array(hits)[1] == true
        @test Array(distances)[1] ≈ 1.0f0
    end

    @testset "Dynamic scene: add instances via kernel" begin
        mesh1 = make_triangle_mesh()
        mesh2 = make_triangle_mesh(Vec3f(5, 0, 0))

        # Create mutable TLAS
        tlas = Raycore.TLAS(cl_backend)
        h1 = push!(tlas, mesh1)
        Raycore.sync!(tlas)

        n = 2
        origins = KA.allocate(cl_backend, Point3f, n)
        directions = KA.allocate(cl_backend, Vec3f, n)
        hits = KA.allocate(cl_backend, Bool, n)
        distances = KA.allocate(cl_backend, Float32, n)

        KA.copyto!(cl_backend, origins, [Point3f(0.25, 0.25, 1.0), Point3f(5.25, 0.25, 1.0)])
        KA.copyto!(cl_backend, directions, fill(Vec3f(0, 0, -1), n))

        kernel = closest_hit_kernel!(cl_backend)

        # Test with just first instance
        cl_tlas1 = Adapt.adapt(cl_backend, tlas)
        kernel(hits, distances, cl_tlas1, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        @test hits_cpu[1] == true   # first mesh
        @test hits_cpu[2] == false  # second mesh not added yet

        # Add second instance
        h2 = push!(tlas, mesh2)
        Raycore.sync!(tlas)

        # Test again with both instances
        cl_tlas2 = Adapt.adapt(cl_backend, tlas)
        kernel(hits, distances, cl_tlas2, origins, directions; ndrange=n)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        @test hits_cpu[1] == true   # first mesh
        @test hits_cpu[2] == true   # second mesh now present
    end

    @testset "Batch ray tracing via kernel (64 rays)" begin
        mesh = make_triangle_mesh()
        tlas, _ = TLAS([mesh]; backend=cl_backend)
        cl_tlas = Adapt.adapt(cl_backend, tlas)

        # Create batch of rays
        n_rays = 64
        origins_vec = [Point3f(0.25 + 0.5*(i % 8)/7, 0.25 + 0.5*((i ÷ 8) % 8)/7, 1.0) for i in 0:n_rays-1]
        directions_vec = fill(Vec3f(0, 0, -1), n_rays)

        origins = KA.allocate(cl_backend, Point3f, n_rays)
        directions = KA.allocate(cl_backend, Vec3f, n_rays)
        hits = KA.allocate(cl_backend, Bool, n_rays)
        distances = KA.allocate(cl_backend, Float32, n_rays)

        KA.copyto!(cl_backend, origins, origins_vec)
        KA.copyto!(cl_backend, directions, directions_vec)

        kernel = closest_hit_kernel!(cl_backend)
        kernel(hits, distances, cl_tlas, origins, directions; ndrange=n_rays)
        KA.synchronize(cl_backend)

        hits_cpu = Array(hits)
        n_hits = count(hits_cpu)
        @test n_hits > 0       # At least some hits
        @test n_hits < n_rays  # Some misses near edges
    end

    @testset "StaticTLAS field types after adapt" begin
        mesh = make_triangle_mesh()
        tlas, _ = TLAS([mesh]; backend=cl_backend)

        cl_tlas = Adapt.adapt(cl_backend, tlas)

        # Verify fields are CLArrays after adapt (not plain Vector)
        @test cl_tlas.nodes isa CLArray
        @test cl_tlas.instances isa CLArray
        @test cl_tlas.all_blas_nodes isa CLArray
        @test cl_tlas.all_blas_prims isa CLArray
        @test cl_tlas.blas_descriptors isa CLArray
        # root_aabb stays isbits (not an array)
        @test isbitstype(typeof(cl_tlas.root_aabb))
    end

    @testset "World bound preserved after adapt" begin
        mesh = make_triangle_mesh()
        transforms = [
            Mat4f(I),
            Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 10, 0, 1)
        ]
        tlas, _ = begin; tlas_tmp = Raycore.TLAS(cl_backend); push!(tlas_tmp, mesh, transforms); sync!(tlas_tmp); (tlas_tmp, [TLASHandle(UInt32(1))]); end

        gpu_bound = tlas.root_aabb
        cl_tlas = Adapt.adapt(cl_backend, tlas)
        cl_bound = cl_tlas.root_aabb

        @test gpu_bound.p_min ≈ cl_bound.p_min
        @test gpu_bound.p_max ≈ cl_bound.p_max
    end

end
end # if check_bounds
