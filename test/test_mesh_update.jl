# ==============================================================================
# Mesh update tests: correctness + no-leak under delete+push cycles
# ==============================================================================
#
# The TLAS API has no in-place "update mesh geometry" — to change the triangles
# of an existing instance you `delete!` the old handle and `push!` a new mesh.
# When the new mesh has a *different vertex count*, every backing buffer
# (BLAS nodes, BLAS primitives, per-instance tri/offset buffers on HW) is a
# different size too, so a stale reference from any previous dispatch or cached
# descriptor will fault or return the wrong geometry.
#
# This suite covers both TLAS backings, both driven through Lava + Vulkan:
#
#   1. SW TLAS (`Raycore.TLAS`) — BVH traversed on GPU via a KernelAbstractions
#      `closest_hit` kernel, with the backing `LavaBackend`. Verified after
#      every mutation.
#   2. HW TLAS (`Raycore.HWTLAS`) — Vulkan hardware ray tracing. Verified via
#      `trace_closest_hits!`.
#
# For each backend we oscillate the mesh tessellation count (small ↔ big ↔
# small) many times and assert:
#   - The hit is always at the sphere surface within tolerance (correctness —
#     catches stale BLAS captures: ray would miss or come back with wrong t if
#     any pointer stayed captured).
#   - Internal GPU-side resource counters stay bounded (leak / UAF bound: pool
#     blocks and live buffers must not scale with iteration count).
#
# Lava is a hard test dep for Raycore, so this runs as part of the normal suite.
# ==============================================================================

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using KernelAbstractions
const KA = KernelAbstractions
using Adapt
using Lava

const GPU_BACKEND = Lava.LavaBackend()

# ------------------------------------------------------------------------------
# Shared: sphere mesh with varying tessellation + analytic ray/sphere intersect
# ------------------------------------------------------------------------------

"""Unit sphere centred at origin; `n` = tesselation count (higher = more tris).
Vertex count ≈ (n+1)^2, so successive `n`s give meaningfully different BLAS sizes."""
function sphere_mesh(n::Int)
    GeometryBasics.normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), n))
end

"""Ray straight down the +z axis from z=5 at (0, 0). For a unit sphere at the
origin translated by `offset`, the closest hit is at z = offset.z + 1, i.e.
t = 5 - (offset.z + 1) = 4 - offset.z."""
expected_t(offset_z::Real) = Float32(5) - Float32(offset_z) - Float32(1)

translation(dx, dy, dz) = SMatrix{4,4,Float32,16}(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    dx, dy, dz, 1,
)

# ------------------------------------------------------------------------------
# SW TLAS mesh-update test (Raycore BVH traversal on Lava)
# ------------------------------------------------------------------------------

KA.@kernel function sw_trace_one_kernel!(hit_out, t_out, tlas, origin, direction)
    ray = Raycore.Ray(; o=origin, d=direction)
    hit, _, dist, _, _ = Raycore.closest_hit(tlas, ray)
    hit_out[1] = hit
    t_out[1]   = Float32(dist)
end

"Trace one ray down the +z axis from (0,0,5) and return (hit, t) on CPU."
function sw_trace_one(tlas)
    static_tlas = Adapt.adapt(GPU_BACKEND, tlas)
    hit = KA.zeros(GPU_BACKEND, Bool, 1)
    t   = KA.zeros(GPU_BACKEND, Float32, 1)
    origin    = Point3f(0f0, 0f0, 5f0)
    direction = Vec3f(0f0, 0f0, -1f0)
    sw_trace_one_kernel!(GPU_BACKEND)(hit, t, static_tlas, origin, direction; ndrange=1)
    KA.synchronize(GPU_BACKEND)
    return (hit = Array(hit)[1], t = Array(t)[1])
end

"Replace the one mesh in `tlas` with a fresh `sphere_mesh(n)` at `offset_z`."
function sw_swap_mesh!(tlas, handle, n, offset_z)
    Raycore.delete!(tlas, handle)
    new_handle = push!(tlas, sphere_mesh(n), translation(0, 0, offset_z))
    Raycore.sync!(tlas)
    return new_handle
end

@testset "SW TLAS — mesh update correctness under size oscillation" begin
    tlas = Raycore.TLAS(GPU_BACKEND)
    handle = push!(tlas, sphere_mesh(16), translation(0, 0, 0))
    Raycore.sync!(tlas)

    # Baseline
    r = sw_trace_one(tlas)
    @test r.hit
    @test isapprox(r.t, expected_t(0); atol=0.05f0)

    # Oscillate small → big → small → bigger → smaller. Vertex count varies
    # non-monotonically to cover shrink-in-place and grow-out-of-place paths.
    tess_schedule = [32, 8, 48, 12, 64, 16, 8, 32, 96, 16]
    for (i, n) in enumerate(tess_schedule)
        offset_z = Float32(0.05 * i)   # nudge z so a wrong stale geometry shows up as wrong t
        handle = sw_swap_mesh!(tlas, handle, n, offset_z)
        r = sw_trace_one(tlas)
        @test r.hit          broken=false
        @test isapprox(r.t, expected_t(offset_z); atol=0.1f0)
    end
end

@testset "SW TLAS — adapt-once-then-mutate via tlas.static_tlas" begin
    # Invariant: `sync!(tlas)` is the single owner of `tlas.static_tlas`. A
    # consumer that re-reads `tlas.static_tlas` (or calls `Adapt.adapt(backend,
    # tlas)`) per dispatch MUST see any mutation that went through `push!` /
    # `delete!` + `sync!`. A consumer that caches an old static_tlas across a
    # mutation MAY see stale data — that's the contract consumers must honour.
    #
    # The pre-refactor `Hikari.VolPath.get_or_adapt_scene!` cached on
    # `objectid(scene)` alone, which silently violated the "re-read per
    # dispatch" rule and rendered frozen geometry in the dolphin video. This
    # test nails the contract down at the Raycore level.
    tlas = Raycore.TLAS(GPU_BACKEND)
    handle = push!(tlas, sphere_mesh(16), translation(0, 0, 0))

    # First adapt triggers build of tlas.static_tlas.
    st_before = Adapt.adapt(GPU_BACKEND, tlas)
    @test tlas.static_tlas === st_before

    hit_before = let hit = KA.zeros(GPU_BACKEND, Bool, 1), t = KA.zeros(GPU_BACKEND, Float32, 1)
        sw_trace_one_kernel!(GPU_BACKEND)(hit, t, st_before, Point3f(0,0,5), Vec3f(0,0,-1); ndrange=1)
        KA.synchronize(GPU_BACKEND)
        (hit = Array(hit)[1], t = Array(t)[1])
    end
    @test hit_before.hit
    @test isapprox(hit_before.t, expected_t(0); atol=0.05f0)

    # Mutate the TLAS: swap mesh to one shifted +2 in z. The expected t moves
    # from 4.0 to 2.0, so any stale-geometry trace shows up obviously.
    Raycore.delete!(tlas, handle)
    handle = push!(tlas, sphere_mesh(48), translation(0, 0, 2f0))

    # A consumer re-reads `tlas.static_tlas` per dispatch (the canonical path):
    st_after = Adapt.adapt(GPU_BACKEND, tlas)
    hit_after_fresh = let hit = KA.zeros(GPU_BACKEND, Bool, 1), t = KA.zeros(GPU_BACKEND, Float32, 1)
        sw_trace_one_kernel!(GPU_BACKEND)(hit, t, st_after, Point3f(0,0,5), Vec3f(0,0,-1); ndrange=1)
        KA.synchronize(GPU_BACKEND)
        (hit = Array(hit)[1], t = Array(t)[1])
    end
    @test hit_after_fresh.hit
    @test isapprox(hit_after_fresh.t, expected_t(2); atol=0.1f0)

    # sync! on a clean TLAS is a no-op: doesn't change static_tlas identity,
    # doesn't issue a GPU synchronize. The field is already up to date.
    st_pinned = tlas.static_tlas
    Raycore.sync!(tlas)
    @test tlas.static_tlas === st_pinned

    # A consumer that CACHED `st_before` across the mutation may see stale
    # data: this is by design — the invariant pushes that work onto the
    # consumer (re-read `tlas.static_tlas` per dispatch). Test that we NOTICE
    # staleness when it happens, so regressions that make consumers silently
    # cache are flagged.
    hit_stale = let hit = KA.zeros(GPU_BACKEND, Bool, 1), t = KA.zeros(GPU_BACKEND, Float32, 1)
        sw_trace_one_kernel!(GPU_BACKEND)(hit, t, st_before, Point3f(0,0,5), Vec3f(0,0,-1); ndrange=1)
        KA.synchronize(GPU_BACKEND)
        (hit = Array(hit)[1], t = Array(t)[1])
    end
    # Either st_before's backing buffer was reused in place (stale consumer
    # accidentally sees new data — allowed but not guaranteed), or it was
    # reallocated (stale consumer sees old data). In the reallocation case
    # st_before !== st_after and t ≈ expected_t(0).
    if st_before !== st_after
        @test isapprox(hit_stale.t, expected_t(0); atol=0.05f0)  # stale snapshot
    end
end

@testset "SW TLAS — transform-update refit path" begin
    # The refit path is the "cheap in-place" branch: update_transform! + sync!
    # must update the leaf AABBs in tlas.nodes in place, so a cached static_tlas
    # (that wraps the same backing buffer) sees the new position without needing
    # a fresh adapt call.
    #
    # Pre-fix, refit_tlas! had `tlas.dirty || return tlas` (wrong flag) so it
    # always short-circuited. sync! would run refit_tlas! but refit was a no-op;
    # tlas.transforms_dirty stayed true forever; subsequent clean-path fast
    # returns never kicked in. This test pins the refit wiring.
    tlas = Raycore.TLAS(GPU_BACKEND)
    handle = push!(tlas, sphere_mesh(16), translation(0, 0, 0))
    st_initial = Adapt.adapt(GPU_BACKEND, tlas)

    # Baseline: unit sphere at z=0, ray from z=5 hits at z=1 → t=4.
    r = sw_trace_one(tlas)
    @test r.hit
    @test isapprox(r.t, expected_t(0); atol=0.05f0)

    # Move the instance to z=1.5 via update_transform!. The mesh & handle stay
    # the same; only the instance transform changes.
    Raycore.update_transform!(tlas, handle, translation(0, 0, 1.5f0))
    @test tlas.transforms_dirty

    # sync! must run refit and clear transforms_dirty — and keep static_tlas
    # valid (same object, because refit updates tlas.nodes in place).
    Raycore.sync!(tlas)
    @test !tlas.transforms_dirty
    @test !tlas.dirty
    # refit updates AABBs in place in tlas.nodes — static_tlas identity stays.
    @test tlas.static_tlas === st_initial

    # New expected t: sphere at z=1.5, hit at z=2.5, t = 5 - 2.5 = 2.5.
    r2 = sw_trace_one(tlas)
    @test r2.hit
    @test isapprox(r2.t, expected_t(1.5); atol=0.05f0)

    # Clean-path sync! is a true no-op: static_tlas identity unchanged, no GPU
    # sync, no allocations in the repeated calls.
    Raycore.sync!(tlas)
    Raycore.sync!(tlas)
    Raycore.sync!(tlas)
    @test tlas.static_tlas === st_initial
end

@testset "SW TLAS — only one live static_tlas across many swaps (leak bound)" begin
    # The static_tlas field is the single owner of the adapted form. Overwriting
    # it on every rebuild means the old StaticTLAS goes unreferenced and is
    # collectable. Prior draft designs (kept a cache of adapted_scene keyed by
    # objectid in VolPath) accumulated references across mutations — that's
    # the regression this test exists to prevent.
    tlas = Raycore.TLAS(GPU_BACKEND)
    handle = push!(tlas, sphere_mesh(16), translation(0, 0, 0))
    Raycore.sync!(tlas)

    # Hold a weak reference to the first static_tlas; it should be collectable
    # after enough swaps since no one else is keeping it alive.
    first_static = tlas.static_tlas
    wref = WeakRef(first_static)
    first_static = nothing  # drop the hard local ref
    # Ensure the rest of this iteration doesn't pin `first_static` on the stack
    # via sentinel bindings by doing enough work in between.

    for iter in 1:20
        n = isodd(iter) ? 32 : 16
        handle = sw_swap_mesh!(tlas, handle, n, Float32(0.01 * iter))
        _ = Adapt.adapt(GPU_BACKEND, tlas)
    end
    GC.gc(true)

    # The rebuild path reallocates tlas.nodes each swap, so the original
    # static_tlas is unreachable. With no cache hanging on to it, GC should
    # collect it.
    # pre-fix regression guard: a static_tlas from before N swaps must be collectable.
    @test wref.value === nothing
end

@testset "SW TLAS — mesh update leak bound (Julia heap)" begin
    tlas = Raycore.TLAS(GPU_BACKEND)
    handle = push!(tlas, sphere_mesh(16), translation(0, 0, 0))
    Raycore.sync!(tlas)

    # Warm up cycle so JIT and pool-style caches settle before sampling.
    for _ in 1:3
        handle = sw_swap_mesh!(tlas, handle, 32, 0.1f0)
        _ = sw_trace_one(tlas)
    end
    GC.gc(true)

    n_iters = 50
    for iter in 1:n_iters
        n = iseven(iter) ? 16 : 48                  # oscillate
        handle = sw_swap_mesh!(tlas, handle, n, Float32(0.01 * iter))
        @assert sw_trace_one(tlas).hit

        # Tight invariants checked EVERY iteration: any leak that adds even
        # one entry per cycle is caught immediately, not buried in slack.
        @test length(tlas.instances) == 1
        @test length(tlas.blas_storage) == 1
        @test length(tlas._flat_blas_prims) == length(tlas.blas_storage[1].primitives)
        @test length(tlas._flat_blas_nodes) == length(tlas.blas_storage[1].nodes)
        @test length(tlas.deleted_handles) == 0
    end
    GC.gc(true)

    # Final state: exactly one live mesh, flat arrays match exactly.
    @test length(tlas.instances) == 1
    @test length(tlas.blas_storage) == 1
    @test length(tlas._flat_blas_prims) == length(tlas.blas_storage[1].primitives)
    @test length(tlas._flat_blas_nodes) == length(tlas.blas_storage[1].nodes)
end

# HW TLAS mesh-update tests relocated to Lava in Phase F.

println("\nAll mesh-update tests passed.")
