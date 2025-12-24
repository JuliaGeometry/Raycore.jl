# ==============================================================================
# Instanced BVH Tests
# ==============================================================================

using Test
using Raycore
using GeometryBasics
using StaticArrays
using LinearAlgebra

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

    @test tlas isa TLAS
    @test length(tlas.instances) == 1
    @test length(tlas.blas_array) == 1
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

    @test tlas isa TLAS
    @test length(tlas.instances) == 2
    @test length(tlas.blas_array) == 1
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

@testset "TLAS BVH-compatible API (argument order)" begin
    # Test that closest_hit(ray, tlas) works (BVH-compatible order)
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

    # BVH-compatible order (ray first)
    hit, prim, dist, bary = closest_hit(ray, tlas)
    @test hit == true
    @test dist ≈ 1.0f0

    # any_hit with BVH-compatible order
    hit_any, _, _, _ = any_hit(ray, tlas)
    @test hit_any == true
end

# ==============================================================================
# Instance API Tests
# ==============================================================================

@testset "Instance - Convenience Constructors" begin
    # Create a simple mesh (we'll use triangles as a stand-in)
    v1, v2, v3 = Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)
    mesh = Raycore.TriangleMesh(
        [v1, v2, v3],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # Single instance at identity
    inst1 = Instance(mesh)
    @test length(inst1.transforms) == 1
    @test length(inst1.metadata) == 1
    @test inst1.transforms[1] ≈ Mat4f(I)

    # Single instance with custom metadata
    inst2 = Instance(mesh; metadata=UInt32(42))
    @test inst2.metadata[1] == UInt32(42)

    # Single instance with transform and metadata
    transform = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        5, 0, 0, 1
    )
    inst3 = Instance(mesh, transform, UInt32(99))
    @test inst3.transforms[1] ≈ transform
    @test inst3.metadata[1] == UInt32(99)

    # Multiple instances with same metadata
    transforms = [Mat4f(I), transform]
    inst4 = Instance(mesh, transforms; metadata=UInt32(7))
    @test length(inst4.transforms) == 2
    @test length(inst4.metadata) == 2
    @test all(m == UInt32(7) for m in inst4.metadata)
end

@testset "InstanceHandle and find_instances" begin
    # Create meshes
    mesh1 = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = Raycore.TriangleMesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # Build TLAS with Instance API
    tlas, handles = TLAS([Instance(mesh1), Instance(mesh2)])

    @test length(handles) == 2
    @test handles[1] isa InstanceHandle
    @test handles[2] isa InstanceHandle

    # Find instances by handle
    range1 = find_instances(tlas, handles[1])
    range2 = find_instances(tlas, handles[2])

    @test length(range1) == 1
    @test length(range2) == 1
    @test range1 != range2
end

@testset "TLAS from Instance Vector" begin
    # Create meshes
    mesh1 = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = Raycore.TriangleMesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # Create with multiple transforms for mesh1 (instancing)
    transforms = [
        Mat4f(I),
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1)  # Translated
    ]

    tlas, handles = TLAS([
        Instance(mesh1, transforms),  # 2 instances of mesh1
        Instance(mesh2)               # 1 instance of mesh2
    ])

    @test n_geometries(tlas) == 2  # 2 unique BLAS
    @test n_instances(tlas) == 3   # 2 + 1 = 3 instance descriptors

    # Test handles
    @test length(handles) == 2
    @test length(find_instances(tlas, handles[1])) == 2  # First handle has 2 instances
    @test length(find_instances(tlas, handles[2])) == 1  # Second handle has 1 instance
end

@testset "TLAS with plain geometry (auto-wrap)" begin
    # Plain meshes should be auto-wrapped as Instances
    mesh1 = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = Raycore.TriangleMesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # Mix of plain geometry and Instance
    tlas, handles = TLAS([mesh1, Instance(mesh2)])

    @test n_geometries(tlas) == 2
    @test n_instances(tlas) == 2
    @test length(handles) == 2
end

# ==============================================================================
# Dynamic Update Tests
# ==============================================================================

@testset "update_transform! (single instance)" begin
    mesh = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([Instance(mesh)])

    # Update transform
    new_transform = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        10, 0, 0, 1
    )
    update_transform!(tlas, handles[1], new_transform)

    # Verify transform was updated
    idx = first(find_instances(tlas, handles[1]))
    @test tlas.instances[idx].transform ≈ new_transform
end

@testset "update_transforms! (multiple instances)" begin
    mesh = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # Create instance with 3 transforms
    initial_transforms = [Mat4f(I), Mat4f(I), Mat4f(I)]
    tlas, handles = TLAS([Instance(mesh, initial_transforms)])

    # Update all transforms
    new_transforms = [
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1),
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1),
        Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1)
    ]
    update_transforms!(tlas, handles[1], new_transforms)

    # Verify transforms were updated
    instance_range = find_instances(tlas, handles[1])
    for (i, idx) in enumerate(instance_range)
        @test tlas.instances[idx].transform ≈ new_transforms[i]
    end
end

@testset "add_instance!" begin
    mesh1 = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = Raycore.TriangleMesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([Instance(mesh1)])

    @test n_geometries(tlas) == 1
    @test n_instances(tlas) == 1

    # Add new instance
    new_handle = add_instance!(tlas, Instance(mesh2))

    @test n_geometries(tlas) == 2
    @test n_instances(tlas) == 2
    @test new_handle isa InstanceHandle
end

@testset "remove_instance! and rebuild_tlas!" begin
    mesh1 = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )
    mesh2 = Raycore.TriangleMesh(
        [Point3f(5, 0, 0), Point3f(6, 0, 0), Point3f(5, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    tlas, handles = TLAS([Instance(mesh1), Instance(mesh2)])

    @test n_instances(tlas) == 2

    # Mark first instance for removal
    remove_instance!(tlas, handles[1])

    # Verify it's marked (flags == 0xFFFFFFFF)
    idx = first(find_instances(tlas, handles[1]))
    @test tlas.instances[idx].flags == UInt32(0xFFFFFFFF)

    # Rebuild to compact
    rebuild_tlas!(tlas)

    # Now should only have 1 instance
    @test n_instances(tlas) == 1
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
    mesh = Raycore.TriangleMesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)],
        UInt32[1, 2, 3],
        [Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)]
    )

    # 5 instances of same geometry
    transforms = [Mat4f(I) for _ in 1:5]
    tlas, _ = TLAS([Instance(mesh, transforms)])

    @test n_geometries(tlas) == 1
    @test n_instances(tlas) == 5
end

end # main testset
