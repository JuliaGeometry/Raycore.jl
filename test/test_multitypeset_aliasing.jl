# ==============================================================================
# MultiTypeSet: the host mirror and the device slot may be the same object
# ==============================================================================
#
# `MultiTypeSet` keeps two views of its contents: `data_vectors[T]`, a host
# `Vector`, and `static.data[i]`, the array kernels actually read. On a GPU
# backend `Adapt.adapt` builds a separate device array, so the two are distinct
# and each has to be written. On `KA.CPU()` `adapt` is the identity and they are
# the SAME `Vector` — writing both stores every item twice.
#
# `push!` and `append!` used to grow the host mirror first and only then read
# `length(slot)` for the insertion offset. Aliased, that length already included
# the new items, so they were appended a second time at a bogus offset and the
# returned `SetKey`s pointed past the real entries.
#
# What it looked like: a pbrt scene with two emissive quads (2 triangles each)
# produced SIX `DiffuseAreaLight`s on the CPU backend instead of four, the
# second quad duplicated. The extra emitters made CPU renders ~20% brighter
# than the same scene on a GPU backend, which read as "the GPU is losing light"
# right up until the emissive triangles were counted by hand.
#
# Backend-independent by construction: the assertion is that the set holds what
# was put into it, checked on whatever backend the matrix selected.

using Test
using Raycore
using KernelAbstractions
const KA = KernelAbstractions

struct _MTSItem
    a::Int32
    b::Float32
end

@testset "MultiTypeSet does not duplicate on append" begin
    backend = test_backend()

    @testset "push! one at a time" begin
        set = Raycore.MultiTypeSet(backend)
        keys = [push!(set, _MTSItem(Int32(i), Float32(i))) for i in 1:5]
        @test length(set) == 5
        stored = Array(set.static.data[1])
        @test length(stored) == 5
        @test stored == [_MTSItem(Int32(i), Float32(i)) for i in 1:5]
        # The keys have to address the items that were actually stored.
        @test [Int(k.vec_idx) for k in keys] == collect(1:5)
    end

    @testset "append! in several chunks" begin
        set = Raycore.MultiTypeSet(backend)
        all_keys = Raycore.SetKey[]
        expected = _MTSItem[]
        for chunk in 1:3
            items = [_MTSItem(Int32(10chunk + j), Float32(j)) for j in 1:2]
            append!(expected, items)
            append!(all_keys, append!(set, items))
        end
        @test length(set) == 6
        stored = Array(set.static.data[1])
        @test length(stored) == 6
        @test stored == expected
        @test [Int(k.vec_idx) for k in all_keys] == collect(1:6)
    end

    @testset "push! and append! interleaved" begin
        set = Raycore.MultiTypeSet(backend)
        push!(set, _MTSItem(Int32(1), 1f0))
        append!(set, [_MTSItem(Int32(2), 2f0), _MTSItem(Int32(3), 3f0)])
        push!(set, _MTSItem(Int32(4), 4f0))
        @test length(set) == 4
        stored = Array(set.static.data[1])
        @test stored == [_MTSItem(Int32(i), Float32(i)) for i in 1:4]
    end
end
