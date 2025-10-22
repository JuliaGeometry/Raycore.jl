@testset "Bounds construction" begin
    # Test Bounds2
    b2 = RayCaster.Bounds2(Point2f(1, 2), Point2f(3, 4))
    @test b2.p_min == Point2f(1, 2)
    @test b2.p_max == Point2f(3, 4)

    # Test Bounds3
    b3 = RayCaster.Bounds3(Point3f(1, 2, 3), Point3f(4, 5, 6))
    @test b3.p_min == Point3f(1, 2, 3)
    @test b3.p_max == Point3f(4, 5, 6)

    # Test default constructors (invalid configuration)
    b2_default = RayCaster.Bounds2()
    @test b2_default.p_min == Point2f(Inf32)
    @test b2_default.p_max == Point2f(-Inf32)

    b3_default = RayCaster.Bounds3()
    @test b3_default.p_min == Point3f(Inf32)
    @test b3_default.p_max == Point3f(-Inf32)

    # Test point constructors
    b2_point = RayCaster.Bounds2(Point2f(5, 6))
    @test b2_point.p_min == Point2f(5, 6)
    @test b2_point.p_max == Point2f(5, 6)

    b3_point = RayCaster.Bounds3(Point3f(7, 8, 9))
    @test b3_point.p_min == Point3f(7, 8, 9)
    @test b3_point.p_max == Point3f(7, 8, 9)

    # Test corrected constructors (swap min/max if needed)
    b2_corrected = RayCaster.Bounds2c(Point2f(3, 4), Point2f(1, 2))
    @test b2_corrected.p_min == Point2f(1, 2)
    @test b2_corrected.p_max == Point2f(3, 4)

    b3_corrected = RayCaster.Bounds3c(Point3f(4, 5, 6), Point3f(1, 2, 3))
    @test b3_corrected.p_min == Point3f(1, 2, 3)
    @test b3_corrected.p_max == Point3f(4, 5, 6)
end

@testset "Bounds comparison" begin
    b1 = RayCaster.Bounds3(Point3f(1, 2, 3), Point3f(4, 5, 6))
    b2 = RayCaster.Bounds3(Point3f(1, 2, 3), Point3f(4, 5, 6))
    b3 = RayCaster.Bounds3(Point3f(1, 2, 3), Point3f(4, 5, 7))

    @test b1 == b2
    @test b1 != b3
    @test b1 ≈ b2

    # Test approximate equality with small differences
    b4 = RayCaster.Bounds3(Point3f(1, 2, 3), Point3f(4, 5, 6.000001))
    @test b1 ≈ b4
end

@testset "Bounds getindex" begin
    b = RayCaster.Bounds3(Point3f(1, 2, 3), Point3f(4, 5, 6))
    @test b[1] == Point3f(1, 2, 3)
    @test b[2] == Point3f(4, 5, 6)
    @test all(isnan.(b[3]))  # Invalid index returns NaN
end

@testset "Bounds validity" begin
    b_valid = RayCaster.Bounds3(Point3f(1, 2, 3), Point3f(4, 5, 6))
    @test RayCaster.is_valid(b_valid)

    b_invalid = RayCaster.Bounds3()
    @test !RayCaster.is_valid(b_invalid)
end

@testset "Bounds2 iteration" begin
    b = RayCaster.Bounds2(Point2f(1f0, 3f0), Point2f(4f0, 4f0))
    targets = [
        Point2f(1f0, 3f0), Point2f(2f0, 3f0), Point2f(3f0, 3f0), Point2f(4f0, 3f0),
        Point2f(1f0, 4f0), Point2f(2f0, 4f0), Point2f(3f0, 4f0), Point2f(4f0, 4f0),
    ]
    @test length(b) == 8
    for (p, t) in zip(b, targets)
        @test p == t
    end

    b = RayCaster.Bounds2(Point2f(-1f0), Point2f(1f0))
    targets = [
        Point2f(-1f0, -1f0), Point2f(0f0, -1f0), Point2f(1f0, -1f0),
        Point2f(-1f0, 0f0), Point2f(0f0, 0f0), Point2f(1f0, 0f0),
        Point2f(-1f0, 1f0), Point2f(0f0, 1f0), Point2f(1f0, 1f0),
    ]
    @test length(b) == 9
    for (p, t) in zip(b, targets)
        @test p == t
    end
end

@testset "Bounds3 corner" begin
    b = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 1))
    @test RayCaster.corner(b, 1) == Point3f(0, 0, 0)
    @test RayCaster.corner(b, 2) == Point3f(1, 0, 0)
    @test RayCaster.corner(b, 3) == Point3f(0, 1, 0)
    @test RayCaster.corner(b, 4) == Point3f(1, 1, 0)
    @test RayCaster.corner(b, 5) == Point3f(0, 0, 1)
    @test RayCaster.corner(b, 6) == Point3f(1, 0, 1)
    @test RayCaster.corner(b, 7) == Point3f(0, 1, 1)
    @test RayCaster.corner(b, 8) == Point3f(1, 1, 1)
end

@testset "Bounds union and intersect" begin
    b1 = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(2, 2, 2))
    b2 = RayCaster.Bounds3(Point3f(1, 1, 1), Point3f(3, 3, 3))

    # Union should contain both bounds
    b_union = union(b1, b2)
    @test b_union.p_min == Point3f(0, 0, 0)
    @test b_union.p_max == Point3f(3, 3, 3)

    # Intersection should be the overlap
    b_intersect = intersect(b1, b2)
    @test b_intersect.p_min == Point3f(1, 1, 1)
    @test b_intersect.p_max == Point3f(2, 2, 2)
end

@testset "Bounds overlap and containment" begin
    b1 = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(2, 2, 2))
    b2 = RayCaster.Bounds3(Point3f(1, 1, 1), Point3f(3, 3, 3))
    b3 = RayCaster.Bounds3(Point3f(5, 5, 5), Point3f(6, 6, 6))

    @test RayCaster.overlaps(b1, b2)
    @test !RayCaster.overlaps(b1, b3)

    # Test point containment
    @test RayCaster.inside(b1, Point3f(1, 1, 1))
    @test RayCaster.inside(b1, Point3f(0, 0, 0))  # On boundary
    @test RayCaster.inside(b1, Point3f(2, 2, 2))  # On boundary
    @test !RayCaster.inside(b1, Point3f(3, 3, 3))

    # Test exclusive containment
    @test RayCaster.inside_exclusive(b1, Point3f(1, 1, 1))
    @test RayCaster.inside_exclusive(b1, Point3f(0, 0, 0))  # On min boundary (inclusive)
    @test !RayCaster.inside_exclusive(b1, Point3f(2, 2, 2))  # On max boundary (exclusive)
end

@testset "Bounds geometric properties" begin
    b = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(2, 3, 4))

    # Diagonal
    @test RayCaster.diagonal(b) == Point3f(2, 3, 4)

    # Surface area: 2*(2*3 + 2*4 + 3*4) = 2*(6 + 8 + 12) = 52
    @test RayCaster.surface_area(b) == 52f0

    # Volume: 2 * 3 * 4 = 24
    @test RayCaster.volume(b) == 24f0

    # Sides
    @test RayCaster.sides(b) == Point3f(2, 3, 4)

    # Inclusive sides
    @test RayCaster.inclusive_sides(b) == Point3f(3, 4, 5)

    # Expand
    b_expanded = RayCaster.expand(b, 1f0)
    @test b_expanded.p_min == Point3f(-1, -1, -1)
    @test b_expanded.p_max == Point3f(3, 4, 5)

    # Maximum extent (longest axis)
    @test RayCaster.maximum_extent(b) == 3  # z-axis is longest

    b2 = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(5, 2, 3))
    @test RayCaster.maximum_extent(b2) == 1  # x-axis is longest
end

@testset "Bounds2 area" begin
    b = RayCaster.Bounds2(Point2f(0, 0), Point2f(3, 4))
    @test RayCaster.area(b) == 12f0
end

@testset "Bounds lerp and offset" begin
    b = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(10, 10, 10))

    # Lerp
    p_lerped = RayCaster.lerp(b, Point3f(0.5, 0.5, 0.5))
    @test p_lerped == Point3f(-4.5, -4.5, -4.5)

    # Offset
    p = Point3f(5, 5, 5)
    offset_result = RayCaster.offset(b, p)
    @test offset_result == Point3f(0.5, 0.5, 0.5)

    # Edge case: degenerate bounds
    b_degenerate = RayCaster.Bounds3(Point3f(5, 5, 5), Point3f(5, 5, 5))
    offset_degenerate = RayCaster.offset(b_degenerate, Point3f(5, 5, 5))
    @test offset_degenerate == Point3f(0, 0, 0)
end

@testset "Bounding sphere" begin
    b = RayCaster.Bounds3(Point3f(0, 0, 0), Point3f(2, 2, 2))
    center, radius = RayCaster.bounding_sphere(b)
    @test center == Point3f(1, 1, 1)
    @test radius ≈ sqrt(3.0f0)
end

@testset "Ray-Bounds intersection" begin
    b = RayCaster.Bounds3(Point3f(1), Point3f(2))

    # Ray hitting the bounds
    r1 = RayCaster.Ray(o = Point3f(0), d = Vec3f(1))
    hit, t0, t1 = RayCaster.intersect(b, r1)
    @test hit
    @test t0 ≈ 1f0
    @test t1 ≈ 2f0

    # Ray missing the bounds
    r2 = RayCaster.Ray(o = Point3f(0), d = Vec3f(1, 0, 0))
    hit, t0, t1 = RayCaster.intersect(b, r2)
    @test !hit

    # Ray inside the bounds
    r3 = RayCaster.Ray(o = Point3f(1.5), d = Vec3f(1, 1, 0))
    hit, t0, t1 = RayCaster.intersect(b, r3)
    @test hit
    @test t0 ≈ 0f0

    # Test with precomputed inv_dir and dir_is_negative
    inv_dir = 1f0 ./ r1.d
    dir_is_negative = RayCaster.is_dir_negative(r1.d)
    @test RayCaster.intersect_p(b, r1, inv_dir, dir_is_negative)

    inv_dir2 = 1f0 ./ r2.d
    dir_is_negative2 = RayCaster.is_dir_negative(r2.d)
    @test !RayCaster.intersect_p(b, r2, inv_dir2, dir_is_negative2)
end
