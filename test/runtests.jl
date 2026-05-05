# NOTE: GPU kernel tests are skipped under --check-bounds=yes (the Pkg.test default)
# because bounds checking injects error paths that can't compile to SPIR-V.
# For full test coverage: Pkg.test("Raycore"; julia_args=`--check-bounds=auto`)
#
# Backend selection (CI matrix):
#   RAYCORE_TEST_BACKEND=cpu        — KA.CPU() (default; runs on every CI worker)
#   RAYCORE_TEST_BACKEND=lavapipe   — Lava on lavapipe ICD (apt: mesa-vulkan-drivers).
#                                     CURRENTLY BAILS OUT EARLY: lavapipe's mesa-LLVM
#                                     JIT aborts the process on `X86ISD::MGATHER`
#                                     selection for `Aligned 1` loads emitted by
#                                     Lava's SPIR-V backend in AcceleratedKernels'
#                                     shared-memory reductions. A process abort
#                                     can't be recorded as `@test_broken`, so the
#                                     suite emits placeholder broken-tests until
#                                     the alignment hint is fixed in Lava's
#                                     SPIR-V emitter.
#   RAYCORE_TEST_BACKEND=lava       — Lava on whatever Vulkan ICD is found
#                                     (developer GPUs / RADV / NVIDIA / etc.).
#                                     Full Lava-using suite runs.

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using JET
using Aqua
using KernelAbstractions
const KA = KernelAbstractions

const RAYCORE_TEST_BACKEND_NAME = lowercase(get(ENV, "RAYCORE_TEST_BACKEND", "cpu"))
const _USE_LAVA   = RAYCORE_TEST_BACKEND_NAME in ("lava", "lavapipe")
const _IS_LAVAPIPE = RAYCORE_TEST_BACKEND_NAME == "lavapipe"

if _USE_LAVA
    using Lava
end

"""
    test_backend()

KernelAbstractions backend the current CI matrix entry asks for.
`KA.CPU()` (default) or `Lava.LavaBackend()` (when env var selects lava/lavapipe).
"""
test_backend() = _USE_LAVA ? Lava.LavaBackend() : KA.CPU()

"""Whether we're running on lavapipe specifically (mesa software Vulkan).
Used to mark tests broken that hit lavapipe-specific JIT bugs."""
test_is_lavapipe() = _IS_LAVAPIPE

"""Whether the current backend has VK_KHR_ray_tracing_pipeline.
Used to gate HWTLAS tests."""
test_has_hw_rt() = _USE_LAVA && Lava.vk_context().rt_pipeline_properties !== nothing

"""
    @lavapipe_broken expr

Mark a `@test` expression as broken on lavapipe specifically. Use only on
tests whose failure on lavapipe is a Julia exception (catchable). Tests
whose failure crashes the Julia process — e.g. mesa-LLVM JIT aborts —
can't be caught by this; whole testset must be replaced with a
`@test_broken false` placeholder upstream.
"""
macro lavapipe_broken(ex)
    quote
        if test_is_lavapipe()
            @test_broken $(esc(ex))
        else
            @test $(esc(ex))
        end
    end
end

# ambiguities come from GeometryBasics.@fixed_vector Normal = StaticVector
Aqua.test_all(Raycore; ambiguities=(; broken=true))

@testset "Raycore Tests" begin
    # CPU-only suites — run on every backend matrix entry.
    @testset "Intersection" begin
        include("test_intersection.jl")
    end
    @testset "Type Stability" begin
        # include("test_type_stability.jl")  # disabled: @allocated tests fail on Julia 1.12
    end
    @testset "Bounds" begin
        include("bounds.jl")
    end
    @testset "Unrolled" begin
        # include("test_unrolled.jl")  # requires BenchmarkTools (not in test deps)
    end

    # Backend-using suites.  On lavapipe, the FIRST kernel dispatch in any
    # of these aborts the Julia process via mesa's LLVM JIT before
    # `@testset` machinery can record anything — so on lavapipe we replace
    # the whole include with a placeholder broken-test.  When the
    # underlying alignment hint in Lava's SPIR-V emitter is fixed, drop
    # these placeholders and let the testsets run normally.
    if test_is_lavapipe()
        @testset "Instanced BVH (lavapipe placeholder)" begin
            @test_broken false
        end
        @testset "MultiTypeSet (lavapipe placeholder)" begin
            @test_broken false
        end
        @testset "Mesh Update (lavapipe placeholder)" begin
            @test_broken false
        end
        @testset "AbstractAccel contract (lavapipe placeholder)" begin
            @test_broken false
        end
        @testset "TLAS Stress (lavapipe placeholder)" begin
            @test_broken false
        end
    else
        # Either KA.CPU() (cpu matrix entry) or LavaBackend on real GPU.
        @testset "Instanced BVH" begin
            include("test_instanced_bvh.jl")
        end
        if _USE_LAVA
            # Suites that hard-depend on Lava-specific types (LavaArray /
            # HWTLAS).  Don't run on the cpu matrix entry.
            @testset "MultiTypeSet" begin
                include("test_multitypeset.jl")
            end
            @testset "Mesh Update (Lava SW)" begin
                include("test_mesh_update.jl")
            end
            @testset "AbstractAccel contract" begin
                include("test_abstract_accel_contract.jl")
            end
            @testset "TLAS Stress" begin
                include("test_tlas_stress.jl")
            end
        end
    end
end
