using Test
using Raycore: MultiTypeVec, StaticMultiTypeVec, HeteroVecIndex, TextureRef
using Raycore: with_index, deref, is_valid, is_invalid, n_slots
using KernelAbstractions
using Adapt
using pocl_jll
using OpenCL

backend = OpenCL.OpenCLBackend()

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

@testset "MultiTypeVec basic" begin
    dhv = MultiTypeVec(backend)
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

@testset "Empty MultiTypeVec" begin
    dhv = MultiTypeVec(backend)
    @test isempty(dhv)

    smv = dhv.static
    @test isempty(smv)
    @test n_slots(smv) == 0
end

@testset "GPU kernel with MaterialWith2" begin
    dhv = MultiTypeVec(backend)
    arr1 = Float32[1 2; 3 4]
    arr2 = Float32[5, 6, 7]
    arr3 = Float32[8 9; 10 11]
    arr4 = Float32[12, 13, 14]

    idx1 = push!(dhv, MaterialWith2(arr1, arr2))
    idx2 = push!(dhv, MaterialWith2(arr3, arr4))

    # static field is already GPU-ready
    smv = dhv.static

    # Check structure
    @test smv.data[1] isa OpenCL.CLArray
    @test smv.textures[1] isa OpenCL.CLArray
    @test smv.textures[2] isa OpenCL.CLArray

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

    indices = OpenCL.CLArray([idx1, idx2])
    output = OpenCL.CLArray(zeros(Float32, 2))

    kernel = mat2_kernel(backend)
    kernel(output, smv, indices; ndrange=2)
    KernelAbstractions.synchronize(backend)

    result = Array(output)
    @test result ≈ [arr1[1,1] + arr2[1], arr3[1,1] + arr4[1]]
end

@testset "StaticMultiTypeVec on GPU (no textures)" begin
    dhv = MultiTypeVec(backend)
    idx1 = push!(dhv, SimpleMaterial(0.5f0))
    idx2 = push!(dhv, GlassMaterial(1.5f0))
    idx3 = push!(dhv, SimpleMaterial(0.8f0))

    smv = dhv.static

    # Check that inner arrays are CLArrays
    @test smv.data[1] isa OpenCL.CLArray
    @test smv.data[2] isa OpenCL.CLArray

    # Run kernel
    @kernel function simple_kernel(output, hvec, indices)
        i = @index(Global)
        get_val(m::SimpleMaterial) = m.color
        get_val(m::GlassMaterial) = m.ior
        output[i] = with_index(get_val, hvec, indices[i])
    end

    indices = OpenCL.CLArray([idx1, idx2, idx3])
    output = OpenCL.CLArray(zeros(Float32, 3))

    kernel = simple_kernel(backend)
    kernel(output, smv, indices; ndrange=3)
    KernelAbstractions.synchronize(backend)

    result = Array(output)
    @test result ≈ [0.5f0, 1.5f0, 0.8f0]
end
