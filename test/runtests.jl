using Test
using GeometryBasics
using LinearAlgebra
using RayCaster
using JET

@testset "RayCaster Tests" begin
    @testset "Intersection" begin
        include("test_intersection.jl")
    end
    @testset "Type Stability" begin
        include("test_type_stability.jl")
    end
    @testset "Bounds" begin
        include("bounds.jl")
    end
end
