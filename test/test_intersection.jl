@testset "Ray-Bounds intersection" begin
    b = RayCaster.Bounds3(Point3f(1), Point3f(2))
    b_neg = RayCaster.Bounds3(Point3f(-2), Point3f(-1))
    r0 = RayCaster.Ray(o = Point3f(0), d = Vec3f(1, 0, 0))
    r1 = RayCaster.Ray(o = Point3f(0), d = Vec3f(1))
    ri = RayCaster.Ray(o = Point3f(1.5), d = Vec3f(1, 1, 0))

    r, t0, t1 = RayCaster.intersect(b, r1)
    @test r && t0 ≈ 1f0 && t1 ≈ 2f0
    r, t0, t1 = RayCaster.intersect(b, r0)
    @test !r && t0 ≈ 0f0 && t1 ≈ 0f0
    r, t0, t1 = RayCaster.intersect(b, ri)
    @test r && t0 ≈ 0f0 && t1 ≈ 0.5f0

    # Test intersection with precomputed direction reciprocal.
    inv_dir = 1f0 ./ r1.d
    dir_is_negative = RayCaster.is_dir_negative(r1.d)
    @test RayCaster.intersect_p(b, r1, inv_dir, dir_is_negative)
    @test !RayCaster.intersect_p(b_neg, r1, inv_dir, dir_is_negative)
end

# Note: Ray-Sphere intersection tests moved to Trace.jl
# RayCaster no longer has Sphere shapes - only low-level triangle intersection

@testset "Test triangle" begin
    triangles = RayCaster.TriangleMesh(
        [Point3f(0, 0, 2), Point3f(1, 0, 2), Point3f(1, 1, 2)],
        UInt32[1, 2, 3],
        [RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1)],
    )

    triangle = RayCaster.Triangle(triangles, 1)
    tv = RayCaster.vertices(triangle)
    a = norm(tv[1] - tv[2])^2 * 0.5f0
    @test RayCaster.area(triangle) ≈ a

    target_wb = RayCaster.Bounds3(Point3f(0, 0, 2), Point3f(1, 1, 2))
    # In the refactored API, object_bound returns world bounds since transformation is applied during creation
    @test RayCaster.object_bound(triangle) ≈ target_wb

    # Test ray intersection - API has changed: intersect now returns (Bool, Float32, Point3f) with barycentric coords
    ray = RayCaster.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    intersects_p = RayCaster.intersect_p(triangle, ray)
    intersects, t_hit, bary_coords = RayCaster.intersect(triangle, ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 4f0
    @test RayCaster.apply(ray, t_hit) ≈ Point3f(0, 0, 2)
    # Barycentric coordinates for vertex 0 (corner hit)
    @test bary_coords ≈ Point3f(1, 0, 0)

    # Test ray intersection (different point).
    ray = RayCaster.Ray(o = Point3f(0.5, 0.25, 0), d = Vec3f(0, 0, 1))
    intersects_p = RayCaster.intersect_p(triangle, ray)
    intersects, t_hit, bary_coords = RayCaster.intersect(triangle, ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 2f0
    @test RayCaster.apply(ray, t_hit) ≈ Point3f(0.5, 0.25, 2)
end

# BVH tests with spheres removed - refactored RayCaster only supports triangle meshes in BVH
@testset "BVH" begin
    # Create triangle meshes instead of spheres
    triangle_meshes = []
    for i in 0:1:3  # Use fewer triangles for simpler test
        core = RayCaster.translate(Vec3f(i*3, i*3, 0))
        mesh = RayCaster.TriangleMesh(
            core.([Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(1, 1, 0)]),
            UInt32[1, 2, 3],
            [RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1)],
        )
        push!(triangle_meshes, mesh)
    end

    bvh = RayCaster.BVHAccel(triangle_meshes)
    # Test basic BVH functionality with triangle meshes
    @test !isnothing(RayCaster.world_bound(bvh))

    # Simple intersection test
    ray = RayCaster.Ray(o = Point3f(0.5, 0.5, -1), d = Vec3f(0, 0, 1))
    intersects, interaction = RayCaster.closest_hit(bvh, ray)
    @test intersects
end

# BVH test with spheres removed - using triangle meshes instead
@testset "Test BVH with triangle meshes in a row" begin
    triangle_meshes = []

    # Create triangle meshes at different z positions
    positions = [0, 4, 8]
    vertices = [Point3f(-1, -1, 0), Point3f(1, -1, 0), Point3f(0, 1, 0)]
    for (i, z) in enumerate(positions)
        core = RayCaster.translate(Vec3f(0, 0, z))
        vs = core.(vertices)
        mesh = RayCaster.TriangleMesh(
            vs,
            UInt32[1, 2, 3],
            [RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1)],
        )
        push!(triangle_meshes, mesh)
    end

    bvh = RayCaster.BVHAccel(triangle_meshes)
    # Test that BVH can be created and has a valid bound
    bound = RayCaster.world_bound(bvh)
    @test !isnothing(bound)

    # Test intersection with the first triangle
    ray = RayCaster.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    intersects, triangle = RayCaster.closest_hit(bvh, ray)
    @test intersects
    # BVH closest_hit returns Triangle object, not SurfaceInteraction
    @test typeof(triangle) == RayCaster.Triangle
end
