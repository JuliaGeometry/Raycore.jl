using Test
using GeometryBasics
using LinearAlgebra
using RayCaster
using FileIO
using ImageCore

include("test_intersection.jl")

@testset "Test Bounds2 iteration" begin
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

@testset "Sphere bound" begin
    core = RayCaster.ShapeCore(RayCaster.translate(Vec3f(0)), false)
    s = RayCaster.Sphere(core, 1f0, -1f0, 1f0, 360f0)

    sb = RayCaster.object_bound(s)
    @test sb[1] == Point3f(-1f0)
    @test sb[2] == Point3f(1f0)
end
