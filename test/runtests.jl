# NOTE: GPU kernel tests are skipped under --check-bounds=yes (the Pkg.test default)
# because bounds checking injects error paths that can't compile to SPIR-V.
# For full test coverage: Pkg.test("Raycore"; julia_args=`--check-bounds=auto`)
#
# The suite runs on Mantle's default device: the GPU where there is one, lavapipe
# (Mesa's Vulkan driver on the CPU) where there is none, as on CI. To run it on
# another device, call `Mantle.defaultdevice!(...)` before including this file.
# Every suite names no device: it asks `test_backend()`. The hardware TLAS
# (`Mantle.HWTLAS`) is tested in Mantle's own suite, on each of its backends.

# Before anything makes the first device, so that lavapipe is among the devices.
using Lavapipe_jll
using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using Aqua
using KernelAbstractions
const KA = KernelAbstractions
using Mantle

const TEST_BACKEND = Mantle.defaultbackend()

"""
    test_backend()

The device the suite runs on: Mantle's default one.
"""
test_backend() = TEST_BACKEND

# `trace(tlas, args...)` (`closest_hit` or `any_hit`) of one ray, in a kernel on the
# structure's device.
@kernel function traced_kernel!(out, tlas, trace, args)
    out[1] = trace(tlas, args...)
end

"""
    traced(trace, tlas, args...)

What `trace(tlas, args...)` returns, computed in a kernel on the structure's device and
read back.
"""
function traced(trace, tlas, args...)
    backend = test_backend()
    out = KA.allocate(backend, Base.promote_op(trace, typeof(tlas), typeof.(args)...), 1)
    traced_kernel!(backend)(out, tlas, trace, args; ndrange = 1)
    KA.synchronize(backend)
    return only(Array(out))
end

# ambiguities come from GeometryBasics.@fixed_vector Normal = StaticVector
Aqua.test_all(Raycore; ambiguities=(; broken=true))

@testset "Raycore Tests" begin
    @testset "Intersection" begin
        include("test_intersection.jl")
    end
    @testset "Bounds" begin
        include("bounds.jl")
    end
    @testset "Unrolled" begin
        include("test_unrolled.jl")
    end
    @testset "Instanced BVH" begin
        include("test_instanced_bvh.jl")
    end
    @testset "BVH refit ordering" begin
        include("test_bvh_refit_ordering.jl")
    end
    @testset "MultiTypeSet aliasing" begin
        include("test_multitypeset_aliasing.jl")
    end
    @testset "StaticMultiTypeSet show" begin
        include("test_multitypeset_show.jl")
    end
    @testset "MultiTypeSet" begin
        include("test_multitypeset.jl")
        include("test_tuple_textures.jl")
    end
    @testset "Mesh Update" begin
        include("test_mesh_update.jl")
    end
    @testset "AbstractAccel contract" begin
        include("test_abstract_accel_contract.jl")
    end
    @testset "TLAS Stress" begin
        include("test_tlas_stress.jl")
    end
end
