using LinearAlgebra
using Raycore.StaticArrays
# ==================== Test Data Generators ====================

# Basic geometric types
gen_point3f() = Point3f(1.0f0, 2.0f0, 3.0f0)
gen_point2f() = Point2f(0.5f0, 0.5f0)
gen_vec3f() = Vec3f(0.0f0, 0.0f0, 1.0f0)
gen_normal3f() = Raycore.Normal3f(0.0f0, 0.0f0, 1.0f0)

# Bounds
gen_bounds2() = Raycore.Bounds2(Point2f(0.0f0), Point2f(1.0f0))
gen_bounds3() = Raycore.Bounds3(Point3f(0.0f0), Point3f(1.0f0, 1.0f0, 1.0f0))

# Rays
gen_ray() = Raycore.Ray(o=Point3f(0.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))
gen_ray_differentials() = Raycore.RayDifferentials(o=Point3f(0.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))

# Transformations
gen_transformation() = Raycore.Transformation()
gen_transformation_translate() = Raycore.translate(Vec3f(1.0f0, 0.0f0, 0.0f0))
gen_transformation_rotate() = Raycore.rotate_x(45.0f0)
gen_transformation_scale() = Raycore.scale(2.0f0, 2.0f0, 2.0f0)

# Triangle
function gen_triangle()
    v1 = Point3f(0.0f0, 0.0f0, 0.0f0)
    v2 = Point3f(1.0f0, 0.0f0, 0.0f0)
    v3 = Point3f(0.0f0, 1.0f0, 0.0f0)
    n1 = Raycore.Normal3f(0.0f0, 0.0f0, 1.0f0)
    uv1 = Point2f(0.0f0, 0.0f0)
    uv2 = Point2f(1.0f0, 0.0f0)
    uv3 = Point2f(0.0f0, 1.0f0)
    Raycore.Triangle(
        SVector(v1, v2, v3),
        SVector(n1, n1, n1),
        SVector(Vec3f(NaN), Vec3f(NaN), Vec3f(NaN)),
        SVector(uv1, uv2, uv3),
        UInt32(1)  # metadata (single field replaces mesh_idx and material_idx)
    )
end

# TLAS from a simple mesh
function gen_tlas_accel()
    mesh = normal_mesh(Rect3f(Point3f(0), Vec3f(1)))
    Raycore.TLAS([mesh], (mi, ti) -> UInt32(mi))
end

# Quaternion
gen_quaternion() = Raycore.Quaternion()

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
        @test_opt_alloc Raycore.Bounds2()

        @test_opt_alloc Raycore.Bounds2(gen_point2f())

        @test_opt_alloc Raycore.Bounds2c(gen_point2f(), Point2f(1.0f0, 1.0f0))
    end

    @testset "Bounds3" begin
        @test_opt_alloc Raycore.Bounds3()

        @test_opt_alloc Raycore.Bounds3(gen_point3f())

        @test_opt_alloc Raycore.Bounds3c(gen_point3f(), Point3f(2.0f0, 2.0f0, 2.0f0))
    end

    @testset "Bounds operations" begin
        b1 = gen_bounds3()
        b2 = Raycore.Bounds3(Point3f(0.5f0), Point3f(1.5f0, 1.5f0, 1.5f0))
        p = gen_point3f()

        @test_opt_alloc Base.:(==)(b1, b2)
        @test_opt_alloc Base.:≈(b1, b2)
        @test_opt_alloc Base.getindex(b1, 1)
        @test_opt_alloc Raycore.is_valid(b1)
        @test_opt_alloc Raycore.corner(b1, 1)
        @test_opt_alloc Base.union(b1, b2)
        @test_opt_alloc Base.intersect(b1, b2)
        @test_opt_alloc Raycore.overlaps(b1, b2)
        @test_opt_alloc Raycore.inside(b1, p)
        @test_opt_alloc Raycore.inside_exclusive(b1, p)
        @test_opt_alloc Raycore.expand(b1, 0.1f0)
        @test_opt_alloc Raycore.diagonal(b1)
        @test_opt_alloc Raycore.surface_area(b1)
        @test_opt_alloc Raycore.volume(b1)
        @test_opt_alloc Raycore.maximum_extent(b1)
        @test_opt_alloc Raycore.sides(b1)
        @test_opt_alloc Raycore.inclusive_sides(b1)
        @test_opt_alloc Raycore.bounding_sphere(b1)
        @test_opt_alloc Raycore.offset(b1, p)
    end

    @testset "Bounds with Ray" begin
        b = gen_bounds3()
        r = gen_ray()

        @test_opt_alloc Raycore.intersect(b, r)
        @test_opt_alloc Raycore.is_dir_negative(r.d)

        inv_dir = 1.0f0 ./ r.d
        dir_neg = Raycore.is_dir_negative(r.d)
        @test_opt_alloc Raycore.intersect_p(b, r, inv_dir, dir_neg)
    end

    @testset "Bounds2 iteration" begin
        b = gen_bounds2()
        @test_opt_alloc Base.length(b)
        @test_opt_alloc Base.iterate(b)
        @test_opt_alloc Base.iterate(b, Int32(1))
    end

    @testset "Distance functions" begin
        p1 = gen_point3f()
        p2 = Point3f(2.0f0, 3.0f0, 4.0f0)

        @test_opt_alloc Raycore.distance(p1, p2)
        @test_opt_alloc Raycore.distance_squared(p1, p2)
    end

    @testset "Lerp functions" begin
        b = gen_bounds3()
        p = gen_point3f()

        @test_opt_alloc Raycore.lerp(0.0f0, 1.0f0, 0.5f0)
        @test_opt_alloc Raycore.lerp(Point3f(0), Point3f(1), 0.5f0)
        @test_opt_alloc Raycore.lerp(b, Point3f(0.5f0))
    end

    @testset "Bounds2 area" begin
        b = gen_bounds2()
        @test_opt_alloc Raycore.area(b)
    end
end

# ==================== Ray Tests ====================

@testset "Type Stability: ray.jl" begin
    @testset "Ray construction" begin
        @test_opt_alloc Raycore.Ray(o=gen_point3f(), d=gen_vec3f())
        @test_opt_alloc Raycore.Ray(o=gen_point3f(), d=gen_vec3f(), t_max=10.0f0)
        @test_opt_alloc Raycore.Ray(o=gen_point3f(), d=gen_vec3f(), t_max=10.0f0, time=0.5f0)
    end

    @testset "Ray copy constructor" begin
        r = gen_ray()
        @test_opt_alloc Raycore.Ray(r; o=Point3f(1.0f0))
        @test_opt_alloc Raycore.Ray(r; d=Vec3f(1.0f0, 0.0f0, 0.0f0))
        @test_opt_alloc Raycore.Ray(r; t_max=5.0f0)
    end

    @testset "RayDifferentials construction" begin
        @test_opt_alloc Raycore.RayDifferentials(o=gen_point3f(), d=gen_vec3f())
        @test_opt_alloc Raycore.RayDifferentials(gen_ray())
    end

    @testset "Ray operations" begin
        r = gen_ray()
        rd = gen_ray_differentials()

        @test_opt_alloc Raycore.set_direction(r, Vec3f(1.0f0, 0.0f0, 0.0f0))
        @test_opt_alloc Raycore.set_direction(rd, Vec3f(1.0f0, 0.0f0, 0.0f0))
        @test_opt_alloc Raycore.check_direction(r)
        @test_opt_alloc Raycore.check_direction(rd)
        @test_opt_alloc Raycore.apply(r, 1.0f0)
        @test_opt_alloc Raycore.increase_hit(r, 0.5f0)
        @test_opt_alloc Raycore.increase_hit(rd, 0.5f0)
    end

    @testset "RayDifferentials operations" begin
        rd = gen_ray_differentials()
        @test_opt_alloc Raycore.scale_differentials(rd, 0.5f0)
    end

    @testset "Intersection helpers" begin
        t = gen_triangle()
        r = gen_ray()
        @test_opt_alloc Raycore.intersect_p!(t, r)
    end
end

# ==================== Transformation Tests ====================

@testset "Type Stability: transformations.jl" begin
    @testset "Transformation construction" begin
        @test_opt_alloc Raycore.Transformation()
        @test_opt_alloc Raycore.Transformation(Mat4f(I))
    end

    @testset "Basic transformations" begin
        @test_opt_alloc Raycore.translate(gen_vec3f())
        @test_opt_alloc Raycore.scale(2.0f0, 2.0f0, 2.0f0)
        @test_opt_alloc Raycore.rotate_x(45.0f0)
        @test_opt_alloc Raycore.rotate_y(45.0f0)
        @test_opt_alloc Raycore.rotate_z(45.0f0)
        @test_opt_alloc Raycore.rotate(45.0f0, Vec3f(0, 0, 1))
    end

    @testset "Transformation operations" begin
        t1 = gen_transformation_translate()
        t2 = gen_transformation_rotate()

        @test_opt_alloc Raycore.is_identity(t1)
        @test_opt_alloc Base.transpose(t1)
        @test_opt_alloc Base.inv(t1)
        @test_opt_alloc Base.:(==)(t1, t2)
        @test_opt_alloc Base.:≈(t1, t2)
        @test_opt_alloc Base.:*(t1, t2)
    end

    @testset "Transformation application" begin
        t = gen_transformation_translate()

        @test_opt_alloc t(gen_point3f())
        @test_opt_alloc t(gen_vec3f())
        @test_opt_alloc t(gen_normal3f())
        @test_opt_alloc t(gen_bounds3())
    end

    @testset "Advanced transformations" begin
        @test_opt_alloc Raycore.look_at(Point3f(0, 0, 5), Point3f(0), Vec3f(0, 1, 0))
        @test_opt_alloc Raycore.perspective(60.0f0, 0.1f0, 100.0f0)
    end

    @testset "Transformation properties" begin
        t = gen_transformation_scale()
        @test_opt_alloc Raycore.has_scale(t)
        @test_opt_alloc Raycore.swaps_handedness(t)
    end

    @testset "Transformation with Ray" begin
        t = gen_transformation_translate()
        r = gen_ray()
        rd = gen_ray_differentials()

        @test_opt_alloc Raycore.apply(t, r)
        @test_opt_alloc Raycore.apply(t, rd)
    end

    @testset "Quaternion" begin
        @test_opt_alloc Raycore.Quaternion()
        @test_opt_alloc Raycore.Quaternion(gen_transformation())

        q1 = gen_quaternion()
        q2 = Raycore.Quaternion(Vec3f(1, 0, 0), 0.5f0)

        @test_opt_alloc Base.:+(q1, q2)
        @test_opt_alloc Base.:-(q1, q2)
        @test_opt_alloc Base.:/(q1, 2.0f0)
        @test_opt_alloc Base.:*(q1, 2.0f0)
        @test_opt_alloc LinearAlgebra.dot(q1, q2)
        @test_opt_alloc LinearAlgebra.normalize(q1)
        @test_opt_alloc Raycore.Transformation(q1)
        @test_opt_alloc Raycore.slerp(q1, q2, 0.5f0)
    end
end

# ==================== Math Tests ====================

@testset "Type Stability: math.jl" begin
    @testset "Sampling functions" begin
        u = gen_point2f()

        @test_opt_alloc Raycore.concentric_sample_disk(u)
        @test_opt_alloc Raycore.cosine_sample_hemisphere(u)
        @test_opt_alloc Raycore.uniform_sample_sphere(u)
        @test_opt_alloc Raycore.uniform_sample_cone(u, 0.5f0)
        @test_opt_alloc Raycore.uniform_sample_cone(u, 0.5f0, Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1))
    end

    @testset "PDF functions" begin
        @test_opt_alloc Raycore.uniform_sphere_pdf()
        @test_opt_alloc Raycore.uniform_cone_pdf(0.5f0)
    end

    @testset "Shading coordinate system" begin
        w = gen_vec3f()

        @test_opt_alloc Raycore.cos_θ(w)
        @test_opt_alloc Raycore.sin_θ2(w)
        @test_opt_alloc Raycore.sin_θ(w)
        @test_opt_alloc Raycore.tan_θ(w)
        @test_opt_alloc Raycore.cos_ϕ(w)
        @test_opt_alloc Raycore.sin_ϕ(w)
    end

    @testset "Vector operations" begin
        wo = gen_vec3f()
        n = Vec3f(0, 1, 0)

        @test_opt_alloc Raycore.reflect(wo, n)
        @test_opt_alloc Raycore.face_forward(n, wo)
    end

    @testset "Coordinate system" begin
        v = gen_vec3f()
        @test_opt_alloc Raycore.coordinate_system(v)
    end

    @testset "Spherical functions" begin
        @test_opt_alloc Raycore.spherical_direction(0.5f0, 0.5f0, 1.0f0)
        @test_opt_alloc Raycore.spherical_direction(0.5f0, 0.5f0, 1.0f0, Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1))

        v = gen_vec3f()
        @test_opt_alloc Raycore.spherical_θ(v)
        @test_opt_alloc Raycore.spherical_ϕ(v)
    end

    @testset "Helper functions" begin
        v = gen_vec3f()
        @test_opt_alloc Raycore.get_orthogonal_basis(v)

        t = gen_triangle()
        @test_opt_alloc Raycore.random_triangle_point(t)
    end

    @testset "sum_mul" begin
        a = Point3f(0.2f0, 0.3f0, 0.5f0)
        b = Raycore.StaticArrays.SVector(Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0))
        @test_opt_alloc Raycore.sum_mul(a, b)
    end
end

@testset "Type Stability: triangle_mesh.jl" begin
    @testset "Triangle operations" begin
        t = gen_triangle()

        @test_opt_alloc Raycore.vertices(t)
        @test_opt_alloc Raycore.normals(t)
        @test_opt_alloc Raycore.tangents(t)
        @test_opt_alloc Raycore.uvs(t)
        @test_opt_alloc Raycore.area(t)
        @test_opt_alloc Raycore.object_bound(t)
        @test_opt_alloc Raycore.world_bound(t)
    end

    @testset "Triangle intersection" begin
        t = gen_triangle()
        r = gen_ray()

        @test_opt_alloc Raycore.intersect(t, r)
        @test_opt_alloc Raycore.intersect_p(t, r)
        @test_opt_alloc Raycore.intersect_triangle(t.vertices, r)
    end

    @testset "Triangle helper functions" begin
        t = gen_triangle()
        r = gen_ray()

        # Test _to_ray_coordinate_space
        @test_opt_alloc Raycore._to_ray_coordinate_space(t.vertices, r)

        # Test partial_derivatives
        @test_opt_alloc Raycore.partial_derivatives(t, t.vertices, t.uv)

        # Test normal_derivatives
        @test_opt_alloc Raycore.normal_derivatives(t, t.uv)
    end

    @testset "Triangle utilities" begin
        t = gen_triangle()
        @test_opt_alloc Raycore.is_degenerate(t.vertices)
    end
end

# ==================== TLAS Tests ====================

# TLAS test data generators
function gen_blas()
    t = gen_triangle()
    Raycore.build_blas([t])
end

function gen_tlas()
    blas = gen_blas()
    identity = Raycore.Mat4f(LinearAlgebra.I)
    inst = Raycore.InstanceDescriptor(
        UInt32(1), UInt32(1),
        identity, identity, UInt32(0)
    )
    Raycore.build_tlas([blas], [inst])
end

@testset "Type Stability: instanced-bvh.jl" begin
    @testset "BVHNode2 construction" begin
        p = gen_point3f()
        @test_opt_alloc Raycore.BVHNode2(p, p, p, p, UInt32(0), UInt32(1), UInt32(0))
    end

    @testset "InstanceDescriptor construction" begin
        identity = Raycore.Mat4f(LinearAlgebra.I)
        @test_opt_alloc Raycore.InstanceDescriptor(
            UInt32(1), UInt32(1), identity, identity, UInt32(0)
        )
    end

    @testset "TLAS traversal" begin
        tlas = gen_tlas()
        r = gen_ray()

        @test_opt Raycore.world_bound(tlas)
        @test_opt Raycore.closest_hit(tlas, r)
        @test_opt Raycore.any_hit(tlas, r)
    end

    @testset "TLAS helper functions" begin
        node = Raycore.BVHNode2(
            gen_point3f(), gen_point3f(), gen_point3f(), gen_point3f(),
            UInt32(0), UInt32(1), UInt32(0)
        )

        @test_opt_alloc Raycore.is_leaf(node)
        @test_opt_alloc Raycore.is_interior(node)
        @test_opt_alloc Raycore.get_node_aabb(node, true)
        @test_opt_alloc Raycore.get_node_aabb(node, false)
        @test_opt_alloc Raycore.get_tlas_node_aabb(node, true)
        @test_opt_alloc Raycore.get_tlas_node_aabb(node, false)
    end

    @testset "Transform utilities" begin
        identity = Raycore.Mat4f(LinearAlgebra.I)
        p = gen_point3f()
        v = gen_vec3f()

        @test_opt_alloc Raycore.transform_point(identity, p)
        @test_opt_alloc Raycore.transform_direction(identity, v)
        @test_opt_alloc Raycore.safe_invdir(v)
    end

    @testset "Ray-AABB intersection" begin
        node = Raycore.BVHNode2(
            Point3f(0), Point3f(1), Point3f(0), Point3f(1),
            UInt32(2), UInt32(3), UInt32(0)
        )
        ray_inv_d = Vec3f(1, 1, 1)
        ray_o = Point3f(0.5f0, 0.5f0, -1.0f0)

        @test_opt_alloc Raycore.intersect_internal_node(node, ray_inv_d, ray_o, 0.0f0, 100.0f0)
    end

    @testset "Morton code" begin
        p = Point3f(0.5f0, 0.5f0, 0.5f0)
        @test_opt_alloc Raycore.morton_code_30bit(p)
        @test_opt_alloc Raycore.expand_bits(UInt32(123))
    end
end

# ==================== Kernels Tests ====================

@testset "Type Stability: kernels.jl" begin
    @testset "RayHit construction" begin
        @test_opt_alloc Raycore.RayHit(true, gen_point3f(), UInt32(1))
    end
end
