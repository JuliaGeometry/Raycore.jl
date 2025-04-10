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

@testset "Ray-Sphere insersection" begin
    # Sphere at the origin.
    core = RayCaster.ShapeCore(RayCaster.Transformation(), false)
    s = RayCaster.Sphere(core, 1f0, 360f0)

    r = RayCaster.Ray(o = Point3f(0, -2, 0), d = Vec3f(0, 1, 0))
    i, t, interaction = RayCaster.intersect(s, r, false)
    ip = RayCaster.intersect_p(s, r, false)
    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f(0, -1, 0) # World intersection.
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
    @test r(t) ≈ Point3f(0, 0, -1) # World intersection.
    @test interaction.core.p ≈ Point3f(0, 0, -1) # Object intersection.
    @test interaction.core.n ≈ RayCaster.Normal3f(0, 0, -1)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0

    # Test ray inside a sphere.
    r0 = RayCaster.Ray(o = Point3f(0), d = Vec3f(0, 1, 0))
    i, t, interaction = RayCaster.intersect(s, r0, false)
    @test i
    @test t ≈ 1f0
    @test r0(t) ≈ Point3f(0f0, 1f0, 0f0)
    @test interaction.core.n ≈ RayCaster.Normal3f(0, 1, 0)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0

    # Test ray at the edge of the sphere.
    ray_at_edge = RayCaster.Ray(o = Point3f(0, -1, 0), d = Vec3f(0, -1, 0))
    i, t, interaction = RayCaster.intersect(s, ray_at_edge, false)
    @test i
    @test t ≈ 0f0
    @test ray_at_edge(t) ≈ Point3f(0, -1, 0)
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
    @test r(t) ≈ Point3f(0, 1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f(0, 1, 0) # Object intesection.
    @test interaction.core.n ≈ RayCaster.Normal3f(0, -1, 0)
end

@testset "Test triangle" begin
    core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(0, 0, 2)), false)
    triangles = RayCaster.create_triangle_mesh(
        core,
        1, UInt32[1, 2, 3],
        3, [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(1, 1, 0)],
        [RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1), RayCaster.Normal3f(0, 0, -1)],
    )

    tv = RayCaster.vertices(triangles[1])
    a = norm(tv[1] - tv[2])^2 * 0.5f0
    @test RayCaster.area(triangles[1]) ≈ a

    target_wb = RayCaster.Bounds3(Point3f(0, 0, 2), Point3f(1, 1, 2))
    target_ob = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 0))
    @test RayCaster.object_bound(triangles[1]) ≈ target_ob
    @test RayCaster.world_bound(triangles[1]) ≈ target_wb

    # Test ray intersection.
    ray = RayCaster.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    intersects_p = RayCaster.intersect_p(triangles[1], ray)
    intersects, t_hit, interaction = RayCaster.intersect(triangles[1], ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 4f0
    @test ray(t_hit) ≈ interaction.core.p ≈ Point3f(0, 0, 2)
    @test interaction.uv ≈ Point2f(0)
    @test interaction.core.n ≈ RayCaster.Normal3f(0, 0, -1)
    @test interaction.core.wo ≈ -ray.d
    # Test ray intersection (lower-left corner).
    ray = RayCaster.Ray(o = Point3f(1, 0.5, 0), d = Vec3f(0, 0, 1))
    intersects_p = RayCaster.intersect_p(triangles[1], ray)
    intersects, t_hit, interaction = RayCaster.intersect(triangles[1], ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 2f0
    @test ray(t_hit) ≈ interaction.core.p ≈ Point3f(1, 0.5, 2)
    @test interaction.uv ≈ Point2f(1, 0.5)
    @test interaction.core.n ≈ RayCaster.Normal3f(0, 0, -1)
    @test interaction.core.wo ≈ -ray.d
end

@testset "BVH" begin
    primitives = RayCaster.Primitive[]
    for i in 0:3:21
        core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(i, i, 0)), false)
        sphere = RayCaster.Sphere(core, 1f0, 360f0)
        push!(primitives, RayCaster.GeometricPrimitive(sphere))
    end

    bvh = RayCaster.BVHAccel(primitives[1:4])
    bvh2 = RayCaster.BVHAccel(RayCaster.Primitive[primitives[5:end]..., bvh])
    @test RayCaster.world_bound(bvh) ≈ RayCaster.Bounds3(Point3f(-1f0), Point3f(10f0, 10f0, 1f0))
    @test RayCaster.world_bound(bvh2) ≈ RayCaster.Bounds3(Point3f(-1f0), Point3f(22f0, 22f0, 1f0))

    ray1 = RayCaster.Ray(o = Point3f(-2f0, 0f0, 0f0), d = Vec3f(1f0, 0f0, 0f0))
    ray2 = RayCaster.Ray(o = Point3f(0f0, 18f0, 0f0), d = Vec3f(1f0, 0f0, 0f0))

    intersects, interaction = RayCaster.intersect!(bvh2, ray1)
    @test intersects
    @test ray1.t_max ≈ 1f0
    @test ray1(ray1.t_max) ≈ Point3f(-1f0, 0f0, 0f0)
    @test interaction.core.p ≈ Point3f(-1f0, 0f0, 0f0)

    intersects, interaction = RayCaster.intersect!(bvh2, ray2)
    @test intersects
    @test ray2.t_max ≈ 17f0
    @test ray2(ray2.t_max) ≈ Point3f(17f0, 18f0, 0f0)
    @test interaction.core.p ≈ Point3f(17f0, 18f0, 0f0)
end

@testset "Test BVH with spheres in a single row" begin
    primitives = RayCaster.Primitive[]

    core = RayCaster.ShapeCore(RayCaster.Transformation(), false)
    sphere = RayCaster.Sphere(core, 1f0, 360f0)
    push!(primitives, RayCaster.GeometricPrimitive(sphere))

    core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(0, 0, 4)), false)
    sphere = RayCaster.Sphere(core, 2f0, 360f0)
    push!(primitives, RayCaster.GeometricPrimitive(sphere))

    core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(0, 0, 11)), false)
    sphere = RayCaster.Sphere(core, 4f0, 360f0)
    push!(primitives, RayCaster.GeometricPrimitive(sphere))

    bvh = RayCaster.BVHAccel(primitives)
    @test RayCaster.world_bound(bvh) ≈ RayCaster.Bounds3(
        Point3f(-4, -4, -1), Point3f(4, 4, 15),
    )

    ray = RayCaster.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    intersects, interaction = RayCaster.intersect!(bvh, ray)
    @test intersects
    @test ray.t_max ≈ 1f0
    @test ray(ray.t_max) ≈ interaction.core.p

    ray = RayCaster.Ray(o = Point3f(1.5, 0, -2), d = Vec3f(0, 0, 1))
    intersects, interaction = RayCaster.intersect!(bvh, ray)
    @test intersects
    @test 2f0 < ray.t_max < 6f0
    @test ray(ray.t_max) ≈ interaction.core.p

    ray = RayCaster.Ray(o = Point3f(3, 0, -2), d = Vec3f(0, 0, 1))
    intersects, interaction = RayCaster.intersect!(bvh, ray)
    @test intersects
    @test 7f0 < ray.t_max < 15f0
    @test ray(ray.t_max) ≈ interaction.core.p
end
