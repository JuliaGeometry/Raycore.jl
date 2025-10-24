using LinearAlgebra

module TestData
    using GeometryBasics
    using RayCaster
    const SVector = RayCaster.StaticArrays.SVector

    # Basic geometric types
    point3f() = Point3f(1.0f0, 2.0f0, 3.0f0)
    point2f() = Point2f(0.5f0, 0.5f0)
    vec3f() = Vec3f(0.0f0, 0.0f0, 1.0f0)
    normal3f() = RayCaster.Normal3f(0.0f0, 0.0f0, 1.0f0)

    # Bounds
    bounds2() = RayCaster.Bounds2(Point2f(0.0f0), Point2f(1.0f0))
    bounds3() = RayCaster.Bounds3(Point3f(0.0f0), Point3f(1.0f0, 1.0f0, 1.0f0))

    # Rays
    ray() = RayCaster.Ray(o=Point3f(0.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))
    ray_differentials() = RayCaster.RayDifferentials(o=Point3f(0.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))

    # Transformations
    transformation() = RayCaster.Transformation()
    transformation_translate() = RayCaster.translate(Vec3f(1.0f0, 0.0f0, 0.0f0))
    transformation_rotate() = RayCaster.rotate_x(45.0f0)
    transformation_scale() = RayCaster.scale(2.0f0, 2.0f0, 2.0f0)

    # Triangle
    function triangle()
        v1 = Point3f(0.0f0, 0.0f0, 0.0f0)
        v2 = Point3f(1.0f0, 0.0f0, 0.0f0)
        v3 = Point3f(0.0f0, 1.0f0, 0.0f0)
        n1 = RayCaster.Normal3f(0.0f0, 0.0f0, 1.0f0)
        uv1 = Point2f(0.0f0, 0.0f0)
        uv2 = Point2f(1.0f0, 0.0f0)
        uv3 = Point2f(0.0f0, 1.0f0)
        RayCaster.Triangle(
            SVector(v1, v2, v3),
            SVector(n1, n1, n1),
            SVector(Vec3f(NaN), Vec3f(NaN), Vec3f(NaN)),
            SVector(uv1, uv2, uv3),
            UInt32(1)
        )
    end

    # Triangle Mesh
    function triangle_mesh()
        vertices = [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)]
        indices = UInt32[1, 2, 3]  # 1-based indices for Julia
        normals = [RayCaster.Normal3f(0, 0, 1), RayCaster.Normal3f(0, 0, 1), RayCaster.Normal3f(0, 0, 1)]
        RayCaster.TriangleMesh(vertices, indices, normals)
    end

    # BVH
    function bvh_accel()
        mesh = Rect3f(Point3f(0), Vec3f(1))
        RayCaster.BVHAccel([mesh], 1)
    end

    # Quaternion
    quaternion() = RayCaster.Quaternion()
end

# ==================== Custom Test Macros ====================

"""
    @test_opt_alloc expr

Combined macro that tests both type stability (via @test_opt) and zero allocations.
This is equivalent to:
    @test_opt expr
    @test @allocated(expr) == 0
"""
macro test_opt_alloc(expr)
    return esc(quote
        $expr # warmup
        JET.@test_opt $expr
        @test @allocated($expr) == 0
    end)
end

# ==================== Bounds Tests ====================

@testset "Type Stability: bounds.jl" begin
    @testset "Bounds2" begin
        @test_opt_alloc RayCaster.Bounds2()

        @test_opt_alloc RayCaster.Bounds2(TestData.point2f())

        @test_opt_alloc RayCaster.Bounds2c(TestData.point2f(), Point2f(1.0f0, 1.0f0))
    end

    @testset "Bounds3" begin
        @test_opt_alloc RayCaster.Bounds3()

        @test_opt_alloc RayCaster.Bounds3(TestData.point3f())

        @test_opt_alloc RayCaster.Bounds3c(TestData.point3f(), Point3f(2.0f0, 2.0f0, 2.0f0))
    end

    @testset "Bounds operations" begin
        b1 = TestData.bounds3()
        b2 = RayCaster.Bounds3(Point3f(0.5f0), Point3f(1.5f0, 1.5f0, 1.5f0))
        p = TestData.point3f()

        @test_opt_alloc Base.:(==)(b1, b2)
        @test_opt_alloc Base.:≈(b1, b2)
        @test_opt_alloc Base.getindex(b1, 1)
        @test_opt_alloc RayCaster.is_valid(b1)
        @test_opt_alloc RayCaster.corner(b1, 1)
        @test_opt_alloc Base.union(b1, b2)
        @test_opt_alloc Base.intersect(b1, b2)
        @test_opt_alloc RayCaster.overlaps(b1, b2)
        @test_opt_alloc RayCaster.inside(b1, p)
        @test_opt_alloc RayCaster.inside_exclusive(b1, p)
        @test_opt_alloc RayCaster.expand(b1, 0.1f0)
        @test_opt_alloc RayCaster.diagonal(b1)
        @test_opt_alloc RayCaster.surface_area(b1)
        @test_opt_alloc RayCaster.volume(b1)
        @test_opt_alloc RayCaster.maximum_extent(b1)
        @test_opt_alloc RayCaster.sides(b1)
        @test_opt_alloc RayCaster.inclusive_sides(b1)
        @test_opt_alloc RayCaster.bounding_sphere(b1)
        @test_opt_alloc RayCaster.offset(b1, p)
    end

    @testset "Bounds with Ray" begin
        b = TestData.bounds3()
        r = TestData.ray()

        @test_opt_alloc RayCaster.intersect(b, r)
        @test_opt_alloc RayCaster.is_dir_negative(r.d)

        inv_dir = 1.0f0 ./ r.d
        dir_neg = RayCaster.is_dir_negative(r.d)
        @test_opt_alloc RayCaster.intersect_p(b, r, inv_dir, dir_neg)
    end

    @testset "Bounds2 iteration" begin
        b = TestData.bounds2()
        @test_opt_alloc Base.length(b)
        @test_opt_alloc Base.iterate(b)
        @test_opt_alloc Base.iterate(b, Int32(1))
    end

    @testset "Distance functions" begin
        p1 = TestData.point3f()
        p2 = Point3f(2.0f0, 3.0f0, 4.0f0)

        @test_opt_alloc RayCaster.distance(p1, p2)
        @test_opt_alloc RayCaster.distance_squared(p1, p2)
    end

    @testset "Lerp functions" begin
        b = TestData.bounds3()
        p = TestData.point3f()

        @test_opt_alloc RayCaster.lerp(0.0f0, 1.0f0, 0.5f0)
        @test_opt_alloc RayCaster.lerp(Point3f(0), Point3f(1), 0.5f0)
        @test_opt_alloc RayCaster.lerp(b, Point3f(0.5f0))
    end

    @testset "Bounds2 area" begin
        b = TestData.bounds2()
        @test_opt_alloc RayCaster.area(b)
    end
end

# ==================== Ray Tests ====================

@testset "Type Stability: ray.jl" begin
    @testset "Ray construction" begin
        @test_opt_alloc RayCaster.Ray(o=TestData.point3f(), d=TestData.vec3f())
        @test_opt_alloc RayCaster.Ray(o=TestData.point3f(), d=TestData.vec3f(), t_max=10.0f0)
        @test_opt_alloc RayCaster.Ray(o=TestData.point3f(), d=TestData.vec3f(), t_max=10.0f0, time=0.5f0)
    end

    @testset "Ray copy constructor" begin
        r = TestData.ray()
        @test_opt_alloc RayCaster.Ray(r; o=Point3f(1.0f0))
        @test_opt_alloc RayCaster.Ray(r; d=Vec3f(1.0f0, 0.0f0, 0.0f0))
        @test_opt_alloc RayCaster.Ray(r; t_max=5.0f0)
    end

    @testset "RayDifferentials construction" begin
        @test_opt_alloc RayCaster.RayDifferentials(o=TestData.point3f(), d=TestData.vec3f())
        @test_opt_alloc RayCaster.RayDifferentials(TestData.ray())
    end

    @testset "Ray operations" begin
        r = TestData.ray()
        rd = TestData.ray_differentials()

        @test_opt_alloc RayCaster.set_direction(r, Vec3f(1.0f0, 0.0f0, 0.0f0))
        @test_opt_alloc RayCaster.set_direction(rd, Vec3f(1.0f0, 0.0f0, 0.0f0))
        @test_opt_alloc RayCaster.check_direction(r)
        @test_opt_alloc RayCaster.check_direction(rd)
        @test_opt_alloc RayCaster.apply(r, 1.0f0)
        @test_opt_alloc RayCaster.increase_hit(r, 0.5f0)
        @test_opt_alloc RayCaster.increase_hit(rd, 0.5f0)
    end

    @testset "RayDifferentials operations" begin
        rd = TestData.ray_differentials()
        @test_opt_alloc RayCaster.scale_differentials(rd, 0.5f0)
    end

    @testset "Intersection helpers" begin
        t = TestData.triangle()
        r = TestData.ray()
        @test_opt_alloc RayCaster.intersect_p!(t, r)
    end
end

# ==================== Transformation Tests ====================

@testset "Type Stability: transformations.jl" begin
    @testset "Transformation construction" begin
        @test_opt_alloc RayCaster.Transformation()
        @test_opt_alloc RayCaster.Transformation(Mat4f(I))
    end

    @testset "Basic transformations" begin
        @test_opt_alloc RayCaster.translate(TestData.vec3f())
        @test_opt_alloc RayCaster.scale(2.0f0, 2.0f0, 2.0f0)
        @test_opt_alloc RayCaster.rotate_x(45.0f0)
        @test_opt_alloc RayCaster.rotate_y(45.0f0)
        @test_opt_alloc RayCaster.rotate_z(45.0f0)
        @test_opt_alloc RayCaster.rotate(45.0f0, Vec3f(0, 0, 1))
    end

    @testset "Transformation operations" begin
        t1 = TestData.transformation_translate()
        t2 = TestData.transformation_rotate()

        @test_opt_alloc RayCaster.is_identity(t1)
        @test_opt_alloc Base.transpose(t1)
        @test_opt_alloc Base.inv(t1)
        @test_opt_alloc Base.:(==)(t1, t2)
        @test_opt_alloc Base.:≈(t1, t2)
        @test_opt_alloc Base.:*(t1, t2)
    end

    @testset "Transformation application" begin
        t = TestData.transformation_translate()

        @test_opt_alloc t(TestData.point3f())
        @test_opt_alloc t(TestData.vec3f())
        @test_opt_alloc t(TestData.normal3f())
        @test_opt_alloc t(TestData.bounds3())
    end

    @testset "Advanced transformations" begin
        @test_opt_alloc RayCaster.look_at(Point3f(0, 0, 5), Point3f(0), Vec3f(0, 1, 0))
        @test_opt_alloc RayCaster.perspective(60.0f0, 0.1f0, 100.0f0)
    end

    @testset "Transformation properties" begin
        t = TestData.transformation_scale()
        @test_opt_alloc RayCaster.has_scale(t)
        @test_opt_alloc RayCaster.swaps_handedness(t)
    end

    @testset "Transformation with Ray" begin
        t = TestData.transformation_translate()
        r = TestData.ray()
        rd = TestData.ray_differentials()

        @test_opt_alloc RayCaster.apply(t, r)
        @test_opt_alloc RayCaster.apply(t, rd)
    end

    @testset "Quaternion" begin
        @test_opt_alloc RayCaster.Quaternion()
        @test_opt_alloc RayCaster.Quaternion(TestData.transformation())

        q1 = TestData.quaternion()
        q2 = RayCaster.Quaternion(Vec3f(1, 0, 0), 0.5f0)

        @test_opt_alloc Base.:+(q1, q2)
        @test_opt_alloc Base.:-(q1, q2)
        @test_opt_alloc Base.:/(q1, 2.0f0)
        @test_opt_alloc Base.:*(q1, 2.0f0)
        @test_opt_alloc LinearAlgebra.dot(q1, q2)
        @test_opt_alloc LinearAlgebra.normalize(q1)
        @test_opt_alloc RayCaster.Transformation(q1)
        @test_opt_alloc RayCaster.slerp(q1, q2, 0.5f0)
    end
end

# ==================== Math Tests ====================

@testset "Type Stability: math.jl" begin
    @testset "Sampling functions" begin
        u = TestData.point2f()

        @test_opt_alloc RayCaster.concentric_sample_disk(u)
        @test_opt_alloc RayCaster.cosine_sample_hemisphere(u)
        @test_opt_alloc RayCaster.uniform_sample_sphere(u)
        @test_opt_alloc RayCaster.uniform_sample_cone(u, 0.5f0)
        @test_opt_alloc RayCaster.uniform_sample_cone(u, 0.5f0, Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1))
    end

    @testset "PDF functions" begin
        @test_opt_alloc RayCaster.uniform_sphere_pdf()
        @test_opt_alloc RayCaster.uniform_cone_pdf(0.5f0)
    end

    @testset "Shading coordinate system" begin
        w = TestData.vec3f()

        @test_opt_alloc RayCaster.cos_θ(w)
        @test_opt_alloc RayCaster.sin_θ2(w)
        @test_opt_alloc RayCaster.sin_θ(w)
        @test_opt_alloc RayCaster.tan_θ(w)
        @test_opt_alloc RayCaster.cos_ϕ(w)
        @test_opt_alloc RayCaster.sin_ϕ(w)
    end

    @testset "Vector operations" begin
        wo = TestData.vec3f()
        n = Vec3f(0, 1, 0)

        @test_opt_alloc RayCaster.reflect(wo, n)
        @test_opt_alloc RayCaster.face_forward(n, wo)
    end

    @testset "Coordinate system" begin
        v = TestData.vec3f()
        @test_opt_alloc RayCaster.coordinate_system(v)
    end

    @testset "Spherical functions" begin
        @test_opt_alloc RayCaster.spherical_direction(0.5f0, 0.5f0, 1.0f0)
        @test_opt_alloc RayCaster.spherical_direction(0.5f0, 0.5f0, 1.0f0, Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1))

        v = TestData.vec3f()
        @test_opt_alloc RayCaster.spherical_θ(v)
        @test_opt_alloc RayCaster.spherical_ϕ(v)
    end

    @testset "Helper functions" begin
        v = TestData.vec3f()
        @test_opt_alloc RayCaster.get_orthogonal_basis(v)

        t = TestData.triangle()
        @test_opt_alloc RayCaster.random_triangle_point(t)
    end

    @testset "sum_mul" begin
        a = Point3f(0.2f0, 0.3f0, 0.5f0)
        b = RayCaster.StaticArrays.SVector(Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0))
        @test_opt_alloc RayCaster.sum_mul(a, b)
    end
end

# ==================== Surface Interaction Tests ====================

@testset "Type Stability: surface_interaction.jl" begin
    @testset "Interaction construction" begin
        @test_opt_alloc RayCaster.Interaction()
        @test_opt_alloc RayCaster.Interaction(
            TestData.point3f(), 0.0f0, TestData.vec3f(), TestData.normal3f()
        )
    end

    @testset "ShadingInteraction construction" begin
        n = TestData.normal3f()
        v = TestData.vec3f()
        @test_opt_alloc RayCaster.ShadingInteraction(n, v, v, n, n)
    end

    @testset "SurfaceInteraction construction" begin
        @test_opt_alloc RayCaster.SurfaceInteraction()

        p = TestData.point3f()
        wo = TestData.vec3f()
        uv = TestData.point2f()
        n = TestData.normal3f()
        dpdu = TestData.vec3f()

        @test_opt_alloc RayCaster.SurfaceInteraction(p, 0.0f0, wo, uv, dpdu, dpdu, n, n, false)
        @test_opt_alloc RayCaster.SurfaceInteraction(n, p, 0.0f0, wo, uv, dpdu, dpdu, n, n)
    end

    @testset "SurfaceInteraction operations" begin
        si = RayCaster.SurfaceInteraction(
            TestData.point3f(), 0.0f0, TestData.vec3f(), TestData.point2f(),
            TestData.vec3f(), TestData.vec3f(), TestData.normal3f(), TestData.normal3f(), false
        )

        @test_opt_alloc RayCaster.set_shading_geometry(si, TestData.vec3f(), TestData.vec3f(),
                                                  TestData.normal3f(), TestData.normal3f(), true)
    end

    @testset "Differentials" begin
        si = RayCaster.SurfaceInteraction(
            TestData.point3f(), 0.0f0, TestData.vec3f(), TestData.point2f(),
            TestData.vec3f(), TestData.vec3f(), TestData.normal3f(), TestData.normal3f(), false
        )
        rd = TestData.ray_differentials()

        @test_opt_alloc RayCaster.compute_differentials(si, rd)
    end

    @testset "Transformation application" begin
        t = TestData.transformation_translate()
        i = RayCaster.Interaction(TestData.point3f(), 0.0f0, TestData.vec3f(), TestData.normal3f())

        @test_opt_alloc RayCaster.apply(t, i)
    end

    @testset "Spawn ray" begin
        si = RayCaster.SurfaceInteraction(
            TestData.point3f(), 0.0f0, TestData.vec3f(), TestData.point2f(),
            TestData.vec3f(), TestData.vec3f(), TestData.normal3f(), TestData.normal3f(), false
        )
        i = RayCaster.Interaction(Point3f(1,1,1), 0.0f0, TestData.vec3f(), TestData.normal3f())

        @test_opt_alloc RayCaster.spawn_ray(si.core, i)
        @test_opt_alloc RayCaster.spawn_ray(si, i)
        @test_opt_alloc RayCaster.spawn_ray(si, TestData.vec3f())
    end
end

# ==================== Triangle Mesh Tests ====================

@testset "Type Stability: triangle_mesh.jl" begin

    @testset "Triangle construction" begin
        mesh = TestData.triangle_mesh()
        @test_opt_alloc RayCaster.Triangle(mesh, 1, UInt32(1))
    end

    @testset "Triangle operations" begin
        t = TestData.triangle()

        @test_opt_alloc RayCaster.vertices(t)
        @test_opt_alloc RayCaster.normals(t)
        @test_opt_alloc RayCaster.tangents(t)
        @test_opt_alloc RayCaster.uvs(t)
        @test_opt_alloc RayCaster.area(t)
        @test_opt_alloc RayCaster.object_bound(t)
        @test_opt_alloc RayCaster.world_bound(t)
    end

    @testset "Triangle intersection" begin
        t = TestData.triangle()
        r = TestData.ray()

        @test_opt_alloc RayCaster.intersect(t, r)
        @test_opt_alloc RayCaster.intersect_p(t, r)
        @test_opt_alloc RayCaster.intersect_triangle(t.vertices, r)
    end

    @testset "Triangle helper functions" begin
        t = TestData.triangle()
        r = TestData.ray()

        # Test _to_ray_coordinate_space
        @test_opt_alloc RayCaster._to_ray_coordinate_space(t.vertices, r)

        # Test partial_derivatives
        @test_opt_alloc RayCaster.partial_derivatives(t, t.vertices, t.uv)

        # Test normal_derivatives
        @test_opt_alloc RayCaster.normal_derivatives(t, t.uv)
    end

    @testset "Triangle utilities" begin
        t = TestData.triangle()
        @test_opt_alloc RayCaster.is_degenerate(t.vertices)
    end
end

# ==================== BVH Tests ====================

@testset "Type Stability: bvh.jl" begin
    @testset "BVHPrimitiveInfo" begin
        b = TestData.bounds3()
        @test_opt_alloc RayCaster.BVHPrimitiveInfo(UInt32(1), b)
    end

    @testset "BVHNode construction" begin
        b = TestData.bounds3()
        @test_opt RayCaster.BVHNode(UInt32(0), UInt32(1), b)
    end

    @testset "LinearBVH construction" begin
        b = TestData.bounds3()
        @test_opt_alloc RayCaster.LinearBVHLeaf(b, UInt32(0), UInt32(1))
        @test_opt_alloc RayCaster.LinearBVHInterior(b, UInt32(1), UInt8(0))
    end

    @testset "BVH operations" begin
        bvh = TestData.bvh_accel()
        r = TestData.ray()

        @test_opt RayCaster.world_bound(bvh)
        @test_opt RayCaster.closest_hit(bvh, r)
        @test_opt RayCaster.any_hit(bvh, r)
    end

    @testset "Ray grid generation" begin
        bvh = TestData.bvh_accel()
        direction = Vec3f(0, 0, 1)
        # generate_ray_grid allocates - needs optimization
        @test_opt RayCaster.generate_ray_grid(bvh, direction, 10)
    end
end

# ==================== Kernels Tests ====================

@testset "Type Stability: kernels.jl" begin
    @testset "RayHit construction" begin
        @test_opt_alloc RayCaster.RayHit(true, TestData.point3f(), UInt32(1))
    end

    @testset "Kernel functions" begin
        bvh = TestData.bvh_accel()
        direction = Vec3f(0, 0, 1)

        # threading constructs allocate and prohibit type inference
        # @test_opt RayCaster.hits_from_grid(bvh, direction; grid_size=8)
        # @test_opt RayCaster.get_illumination(bvh, direction; grid_size=8)
    end
end
