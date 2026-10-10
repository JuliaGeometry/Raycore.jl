using Test, Raycore, GeometryBasics, StaticArrays, LinearAlgebra
using KernelAbstractions; const KA = KernelAbstractions
using Adapt

# The software TLAS on the backend under test. The same contract for the hardware
# structure, `Mantle.HWTLAS`, is asserted in Mantle's `test/test_trace_hwtlas.jl`,
# on each of its backends.
@testset "AbstractAccel — surface" begin
    backend = test_backend()
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
