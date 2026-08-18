using Test
using Raycore: MultiTypeSet, StaticMultiTypeSet, SetKey, TextureRef
using Raycore: with_index, deref, is_valid, is_invalid, n_slots, update!
using KernelAbstractions
using Adapt
using Lava

backend = Lava.LavaBackend()

# Test structs - used for both CPU and GPU tests
struct SimpleMaterial{T}
    color::T
end

struct GlassMaterial{T}
    ior::T
end

struct MaterialWith2{T, T2}
    albedo::T
    texture::T2
end

@testset "MultiTypeSet basic" begin
    dhv = MultiTypeSet(backend)
    @test isempty(dhv)

    idx1 = push!(dhv, SimpleMaterial(0.5f0))
    @test idx1.type_idx == 1
    @test idx1.vec_idx == 1
    @test !isempty(dhv)

    idx2 = push!(dhv, GlassMaterial(1.5f0))
    @test idx2.type_idx == 2
    @test idx2.vec_idx == 1

    idx3 = push!(dhv, SimpleMaterial(0.8f0))
    @test idx3.type_idx == 1
    @test idx3.vec_idx == 2

    # Static is always up-to-date
    @test n_slots(dhv.static) == 2
end

@testset "MultiTypeSet update! with invalid SetKey is a no-op" begin
    # Regression guard: `push!(set, item)` can return `SetKey()` (the (0,0)
    # invalid sentinel) for types that own no slot in the set — e.g.
    # Hikari's `NullMaterial` which pbrt-v4 uses as the "Material interface"
    # /nullptr equivalent, or a `MediumInterface` side left as `nothing`.
    # Callers that reuse that key on `update!` must get a silent no-op,
    # NOT a BoundsError. Prior to the fix, `update!` indexed
    # `dhv.data_order[0]` → `BoundsError: attempt to access 1-element
    # Vector{DataType} at index [0]` — which crashed RayMakie's mesh-swap
    # path for volumes built with `MediumInterface(NullMaterial(); inside=…)`.
    dhv = MultiTypeSet(backend)
    _ = push!(dhv, SimpleMaterial(0.5f0))   # something so data_order is non-empty
    before = (n_slots(dhv.static), length(dhv.static))
    @test update!(dhv, SetKey(), SimpleMaterial(0.9f0)) === nothing
    @test update!(dhv, SetKey(), GlassMaterial(1.7f0)) === nothing  # wrong type, still no-op
    @test (n_slots(dhv.static), length(dhv.static)) == before
end

@testset "Empty MultiTypeSet" begin
    dhv = MultiTypeSet(backend)
    @test isempty(dhv)

    smv = dhv.static
    @test isempty(smv)
    @test n_slots(smv) == 0
end

@testset "GPU kernel with MaterialWith2" begin
    dhv = MultiTypeSet(backend)
    arr1 = Float32[1 2; 3 4]
    arr2 = Float32[5, 6, 7]
    arr3 = Float32[8 9; 10 11]
    arr4 = Float32[12, 13, 14]

    idx1 = push!(dhv, MaterialWith2(arr1, arr2))
    idx2 = push!(dhv, MaterialWith2(arr3, arr4))

    # static field is already GPU-ready
    smv = dhv.static

    # Check structure
    @test smv.data[1] isa Lava.LavaArray
    @test smv.textures[1] isa Lava.LavaArray
    @test smv.textures[2] isa Lava.LavaArray

    # Kernel that accesses both texture fields via deref
    @kernel function mat2_kernel(out, smv, idxs)
        i = @index(Global)
        get_sum(mat, s) = begin
            t1 = deref(s, mat.albedo)
            t2 = deref(s, mat.texture)
            t1[1,1] + t2[1]  # First element of each texture
        end
        out[i] = with_index(get_sum, smv, idxs[i], smv)
    end

    indices = LavaArray([idx1, idx2])
    output = LavaArray(zeros(Float32, 2))

    kernel = mat2_kernel(backend)
    kernel(output, smv, indices; ndrange=2)
    KernelAbstractions.synchronize(backend)

    result = Array(output)
    @test result ≈ [arr1[1,1] + arr2[1], arr3[1,1] + arr4[1]]
end

@testset "StaticMultiTypeSet on GPU (no textures)" begin
    dhv = MultiTypeSet(backend)
    idx1 = push!(dhv, SimpleMaterial(0.5f0))
    idx2 = push!(dhv, GlassMaterial(1.5f0))
    idx3 = push!(dhv, SimpleMaterial(0.8f0))

    smv = dhv.static

    # Check that inner arrays are LavaArrays
    @test smv.data[1] isa Lava.LavaArray
    @test smv.data[2] isa Lava.LavaArray

    # Run kernel
    @kernel function simple_kernel(output, hvec, indices)
        i = @index(Global)
        get_val(m::SimpleMaterial) = m.color
        get_val(m::GlassMaterial) = m.ior
        output[i] = with_index(get_val, hvec, indices[i])
    end

    indices = LavaArray([idx1, idx2, idx3])
    output = LavaArray(zeros(Float32, 3))

    kernel = simple_kernel(backend)
    kernel(output, smv, indices; ndrange=3)
    KernelAbstractions.synchronize(backend)

    result = Array(output)
    @test result ≈ [0.5f0, 1.5f0, 0.8f0]
end

# `push!` resizes the GPU slot and writes one element per call, so registering
# anything per-face (Hikari's area lights: 261 120 of them for one tessellated
# sphere) cost that many vkAllocateMemory/vkFreeMemory pairs — ~150 s per mesh.
# `append!` resizes each slot once and fills it with a single copyto!. It has to
# agree with a push! loop exactly, including the order of the returned keys,
# because callers derive flat indices from them.
@testset "MultiTypeSet append! matches a push! loop" begin
    items = [SimpleMaterial(Float32(i)) for i in 1:64]

    pushed = MultiTypeSet(backend)
    push_keys = [push!(pushed, m) for m in items]

    appended = MultiTypeSet(backend)
    append_keys = append!(appended, items)

    @test append_keys == push_keys
    @test length(appended) == length(pushed) == 64
    @test Array(Raycore.get_static(appended).data[1]) == Array(Raycore.get_static(pushed).data[1])

    @testset "into a slot that already has items" begin
        set = MultiTypeSet(backend)
        first_key = push!(set, SimpleMaterial(1f0))
        keys = append!(set, [SimpleMaterial(Float32(i)) for i in 2:10])
        @test first_key.vec_idx == 1
        @test [k.vec_idx for k in keys] == 2:10
        @test all(k -> k.type_idx == first_key.type_idx, keys)
        @test Array(Raycore.get_static(set).data[1]) == [SimpleMaterial(Float32(i)) for i in 1:10]
    end

    @testset "mixed types keep per-item order" begin
        set = MultiTypeSet(backend)
        mixed = Any[SimpleMaterial(1f0), GlassMaterial(1.5f0), SimpleMaterial(2f0),
                    GlassMaterial(1.6f0), SimpleMaterial(3f0)]
        keys = append!(set, mixed)
        @test length(set) == 5
        # Same type => same slot, in the order they appeared.
        @test keys[1].type_idx == keys[3].type_idx == keys[5].type_idx
        @test keys[2].type_idx == keys[4].type_idx
        @test keys[1].type_idx != keys[2].type_idx
        @test [keys[i].vec_idx for i in (1, 3, 5)] == [1, 2, 3]
        @test [keys[i].vec_idx for i in (2, 4)] == [1, 2]
        # And every key round-trips to the item it was made from. `with_index`
        # indexes the GPU slot, so read the slots back to the host once.
        slots = map(Array, Raycore.get_static(set).data)
        for (i, item) in enumerate(mixed)
            @test slots[keys[i].type_idx][keys[i].vec_idx] == item
        end
    end

    @testset "empty append! is a no-op" begin
        set = MultiTypeSet(backend)
        @test isempty(append!(set, SimpleMaterial{Float32}[]))
        @test isempty(set)
    end
end
