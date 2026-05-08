using Test, Raycore, GeometryBasics, StaticArrays, LinearAlgebra
using KernelAbstractions; const KA = KernelAbstractions
using Adapt
using Lava

@testset "AbstractAccel — surface" begin
    backend = Lava.LavaBackend()
    tlas = Raycore.TLAS(backend)
    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1f0))
    push!(tlas, mesh, SMatrix{4,4,Float32}(I))
    Raycore.sync!(tlas)

    @test Raycore.n_instances(tlas) == 1
    @test Raycore.n_geometries(tlas) == 1
    @test Raycore.world_bound(tlas) isa Raycore.Bounds3

    # wait_for_gpu! returns `accel` so it's chainable; smoke-test the contract.
    @test_nowarn Raycore.wait_for_gpu!(tlas)
    @test Raycore.wait_for_gpu!(tlas) === tlas
end

@testset "AbstractAccel contract — Lava.HWTLAS" begin
    backend = Lava.LavaBackend()
    hwtlas = Lava.HWTLAS(backend)
    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1f0))
    push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
    Raycore.sync!(hwtlas)

    @test Raycore.n_instances(hwtlas) == 1
    @test Raycore.n_geometries(hwtlas) == 1
    @test Raycore.world_bound(hwtlas) isa Raycore.Bounds3
    @test_nowarn Raycore.wait_for_gpu!(hwtlas)
    @test Raycore.wait_for_gpu!(hwtlas) === hwtlas
end
