# NOTE: GPU kernel tests are skipped under --check-bounds=yes (the Pkg.test default)
# because bounds checking injects error paths that can't compile to SPIR-V.
# For full test coverage: Pkg.test("Raycore"; julia_args=`--check-bounds=auto`)
#
# Backend selection (CI matrix), RAYCORE_TEST_BACKEND:
#   cpu        KA.CPU() (default; runs on every CI worker)
#   lava       Vulkan through Lava, Mantle's `LavaBackend`, on the default device
#   lavapipe   the same backend on Mesa's CPU driver, `Device("lavapipe")`
#   metal      Metal.jl's `MetalBackend` on an Apple GPU
#
# Every suite runs on every entry and names no backend: it asks `test_backend()`.
# The KA backend over Vulkan is Mantle's since Lava's runtime moved there, so the
# Vulkan entries load Mantle (a test dependency only; Mantle depends on Raycore).
# The metal entry needs only Metal.jl, whose backend it is. The hardware TLAS
# (`Mantle.HWTLAS`) is tested in Mantle's own suite, on each of its backends.

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using JET
using Aqua
using KernelAbstractions
const KA = KernelAbstractions

const RAYCORE_TEST_BACKEND = lowercase(get(ENV, "RAYCORE_TEST_BACKEND", "cpu"))

if RAYCORE_TEST_BACKEND in ("lava", "lavapipe")
    using Mantle
elseif RAYCORE_TEST_BACKEND == "metal"
    using Metal
end

const TEST_BACKEND = if RAYCORE_TEST_BACKEND == "cpu"
    KA.CPU()
elseif RAYCORE_TEST_BACKEND == "lava"
    Mantle.LavaBackend()
elseif RAYCORE_TEST_BACKEND == "lavapipe"
    Mantle.backend(Mantle.defaultdevice!("lavapipe"))
elseif RAYCORE_TEST_BACKEND == "metal"
    Metal.MetalBackend()
else
    error("RAYCORE_TEST_BACKEND = $(repr(RAYCORE_TEST_BACKEND)): expected cpu, lava, lavapipe or metal")
end

"""
    test_backend()

KernelAbstractions backend the current CI matrix entry asks for.
"""
test_backend() = TEST_BACKEND

# ambiguities come from GeometryBasics.@fixed_vector Normal = StaticVector
Aqua.test_all(Raycore; ambiguities=(; broken=true))

@testset "Raycore Tests" begin
    # Host-only suites: no kernel, the same on every matrix entry.
    @testset "Intersection" begin
        include("test_intersection.jl")
    end
    @testset "Bounds" begin
        include("bounds.jl")
    end
    @testset "Unrolled" begin
        include("test_unrolled.jl")
    end

    # Backend-using suites: the same on every matrix entry.
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
