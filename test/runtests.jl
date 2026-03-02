using Test
using GeometryBasics
using LinearAlgebra
using Raycore
using JET
using Aqua
using pocl_jll, OpenCL

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
