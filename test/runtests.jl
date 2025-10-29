using Test
using GeometryBasics
using LinearAlgebra
using Raycore
using JET
using Aqua

Aqua.test_all(Raycore)

@testset "Raycore Tests" begin
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
