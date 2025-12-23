# ==============================================================================
# Instanced BVH Tests
# ==============================================================================

using Test
using Raycore
using GeometryBasics
using StaticArrays
using LinearAlgebra

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
    tri = Triangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        nothing
    )

    primitives = [tri]
    blas = build_blas(primitives)

    @test blas isa BLAS
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

    tri1 = Triangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(1, 1)),
        nothing
    )

    tri2 = Triangle(
        SVector(v1, v3, v4),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 1), Point2f(0, 1)),
        nothing
    )

    primitives = [tri1, tri2]
    blas = build_blas(primitives)

    @test blas isa BLAS
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

    tri = Triangle(
        SVector(v1, v2, v3),
        SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
        SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
        nothing
    )

    primitives = [tri]

    # build_blas should be type-stable
    result_type = @inferred build_blas(primitives)
    @test result_type isa BLAS
end

@testset "InstanceDescriptor Creation" begin
    # Test instance descriptor construction
    identity_mat = Mat4f(I)
    inst = InstanceDescriptor(
        UInt32(1),      # BLAS index
        UInt32(42),     # Instance ID
        identity_mat,   # Transform
        identity_mat,   # Inverse transform
        UInt32(0)       # Flags
    )

    @test inst.blas_index == UInt32(1)
    @test inst.instance_id == UInt32(42)
    @test inst.transform == identity_mat
    @test inst.flags == UInt32(0)
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

end # main testset
