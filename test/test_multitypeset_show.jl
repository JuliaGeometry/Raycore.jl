# A `StaticMultiTypeSet` is an `AbstractVector` and had no `size`, so printing it
# threw `MethodError: size(::StaticMultiTypeSet)`, and so did printing anything
# holding one. Found when a failing RayMakie test tried to show a render state.

using Test
using Raycore: StaticMultiTypeSet

@testset "StaticMultiTypeSet prints" begin
    empty = StaticMultiTypeSet()
    @test size(empty) == (0,)
    @test sprint(show, empty) == "StaticMultiTypeSet(0 types, 0 elements)"

    smv = StaticMultiTypeSet(([1f0, 2f0], Int32[3, 4, 5]), ())
    @test size(smv) == (5,)
    @test sprint(show, smv) == "StaticMultiTypeSet(2 types, 5 elements)"
    @test sprint(show, MIME"text/plain"(), smv) ==
          "StaticMultiTypeSet with 2 type(s), 5 element(s)\n  2× Float32\n  3× Int32"
    # Inside a container, which is how it was met.
    @test occursin("StaticMultiTypeSet(2 types, 5 elements)", sprint(show, (smv, 1)))
end
