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

@testset "Ray-Sphere intersection" begin
    # Sphere at the origin.
    core = RayCaster.ShapeCore(RayCaster.Transformation(), false)
    s = RayCaster.Sphere(core, 1f0, 360f0)

    r = RayCaster.Ray(o = Point3f(0, -2, 0), d = Vec3f(0, 1, 0))
    i, t, interaction = RayCaster.intersect(s, r, false)
    ip = RayCaster.intersect_p(s, r, false)
    @test i == ip
    @test t ≈ 1f0
    @test RayCaster.apply(r, t) ≈ Point3f(0, -1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f(0, -1, 0) # Object intersection.
    @test interaction.core.n ≈ RayCaster.Normal3f(0, -1, 0)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0
    # Spawn new ray from intersection.
    spawn_direction = Vec3f(0, -1, 0)
    spawned_ray = RayCaster.spawn_ray(interaction, spawn_direction)
    @test spawned_ray.o ≈ Point3f(interaction.core.p)
    @test spawned_ray.d ≈ Vec3f(spawn_direction)
    i, t, interaction = RayCaster.intersect(s, spawned_ray, false)
    @test !i

    r = RayCaster.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    i, t, interaction = RayCaster.intersect(s, r, false)
    ip = RayCaster.intersect_p(s, r, false)
    @test i == ip
    @test t ≈ 1f0
    @test RayCaster.apply(r, t) ≈ Point3f(0, 0, -1) # World intersection.
    @test interaction.core.p ≈ Point3f(0, 0, -1) # Object intersection.
    @test interaction.core.n ≈ RayCaster.Normal3f(0, 0, -1)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0

    # Test ray inside a sphere.
    r0 = RayCaster.Ray(o = Point3f(0), d = Vec3f(0, 1, 0))
    i, t, interaction = RayCaster.intersect(s, r0, false)
    @test i
    @test t ≈ 1f0
    @test RayCaster.apply(r0, t) ≈ Point3f(0f0, 1f0, 0f0)
    @test interaction.core.n ≈ RayCaster.Normal3f(0, 1, 0)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0

    # Test ray at the edge of the sphere.
    ray_at_edge = RayCaster.Ray(o = Point3f(0, -1, 0), d = Vec3f(0, -1, 0))
    i, t, interaction = RayCaster.intersect(s, ray_at_edge, false)
    @test i
    @test t ≈ 0f0
    @test RayCaster.apply(ray_at_edge, t) ≈ Point3f(0, -1, 0)
    @test interaction.core.p ≈ Point3f(0, -1, 0)
    @test interaction.core.n ≈ RayCaster.Normal3f(0, -1, 0)

    # Translated sphere.
    core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(0, 2, 0)), false)
    s = RayCaster.Sphere(core, 1f0, 360f0)
    r = RayCaster.Ray(o = Point3f(0, 0, 0), d = Vec3f(0, 1, 0))

    i, t, interaction = RayCaster.intersect(s, r, false)
    ip = RayCaster.intersect_p(s, r, false)
    @test i == ip
    @test t ≈ 1f0
    @test RayCaster.apply(r, t) ≈ Point3f(0, 1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f(0, 1, 0) # Object intersection.
    @test interaction.core.n ≈ RayCaster.Normal3f(0, -1, 0)
end

@testset "Test triangle" begin
    core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(0, 0, 2)), false)
    triangles = RayCaster.create_triangle_mesh(
        core,
        UInt32[1, 2, 3],
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(1, 1, 0)],
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
        core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(i*3, i*3, 0)), false)
        mesh = RayCaster.create_triangle_mesh(
            core,
            UInt32[1, 2, 3],
            [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(1, 1, 0)],
            [RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1)],
        )
        push!(triangle_meshes, mesh)
    end

    bvh = RayCaster.BVHAccel(triangle_meshes)
    # Test basic BVH functionality with triangle meshes
    @test !isnothing(RayCaster.world_bound(bvh))
    
    # Simple intersection test
    ray = RayCaster.Ray(o = Point3f(0.5, 0.5, -1), d = Vec3f(0, 0, 1))
    intersects, interaction = RayCaster.intersect!(bvh, ray)
    @test intersects
end

# BVH test with spheres removed - using triangle meshes instead
@testset "Test BVH with triangle meshes in a row" begin
    triangle_meshes = []

    # Create triangle meshes at different z positions
    positions = [0, 4, 8]
    for (i, z) in enumerate(positions)
        core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(0, 0, z)), false)
        mesh = RayCaster.create_triangle_mesh(
            core,
            UInt32[1, 2, 3],
            [Point3f(-1, -1, 0), Point3f(1, -1, 0), Point3f(0, 1, 0)],
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
    intersects, triangle = RayCaster.intersect!(bvh, ray)
    @test intersects
    # BVH intersect! returns Triangle object, not SurfaceInteraction
    @test typeof(triangle) == RayCaster.Triangle
end
