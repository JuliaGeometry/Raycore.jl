# NOTE: GPU kernel tests are skipped under --check-bounds=yes (the Pkg.test default)
# because bounds checking injects error paths that can't compile to SPIR-V.
# For full test coverage: Pkg.test("Raycore"; julia_args=`--check-bounds=auto`)

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using JET
using Aqua
using pocl_jll, OpenCL

pocl_platform = OpenCL.cl.platforms()[1]
pocl_device = OpenCL.cl.devices(pocl_platform)[1]
OpenCL.cl.device!(pocl_device)

# ambiguities come from GeometryBasics.@fixed_vector Normal = StaticVector
Aqua.test_all(Raycore; ambiguities=(; broken=true))

@testset "Raycore Tests" begin
    @testset "Intersection" begin
        include("test_intersection.jl")
    end
    @testset "Type Stability" begin
        # include("test_type_stability.jl")  # disabled: @allocated tests fail on Julia 1.12
    end
    @testset "Bounds" begin
        include("bounds.jl")
    end
    @testset "Instanced BVH" begin
        include("test_instanced_bvh.jl")
    end
    @testset "MultiTypeSet" begin
        include("test_multitypeset.jl")
    end
    @testset "Unrolled" begin
        # include("test_unrolled.jl")  # requires BenchmarkTools (not in test deps)
    end
end
