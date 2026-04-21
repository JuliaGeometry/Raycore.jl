# ==============================================================================
# HW TLAS stress + correctness test (Lava backend)
# ==============================================================================
#
# Not hooked into `runtests.jl` because it requires the Lava extension
# (an optional Raycore dep that pulls in a real Vulkan device). Run it
# explicitly when exercising the HW-RT path:
#
#   julia --project=/sim/Programmieren/VulkanDev \
#         dev/Raycore/test/test_hw_tlas_lava.jl
#
# The suite exercises two failure modes the HWTLAS + RaycoreLavaExt path
# is especially prone to:
#
# 1. **Correctness** — every `push!` / `delete!` / `update_transform!` /
#    `update_transform_at!` followed by `sync!(hwtlas)` must reflect in
#    subsequent `trace_closest_hits!` results. If the Vulkan TLAS / BLAS
#    / per-instance offset buffers retain stale BDA captures from before
#    the mutation, rays will hit the OLD geometry → wrong `t` / wrong
#    primitive id. We compare each batch of hits against an analytic CPU
#    reference that knows where the triangles *should* be after the
#    transform.
#
# 2. **Leak / UAF under stress** — the rebuild cycle allocates fresh
#    `LavaTLAS`, `LavaBLAS`, `tri_gpu`, `off_gpu` objects on every dirty
#    `sync!`. The extension's `release_hw_accel_state!` + Lava's
#    finalizer + pool-reuse path must drop the old ones; otherwise
#    `GPU_LIVE_BYTES`, the pool-block count, and the `LIVE_BUFFERS` set
#    grow without bound. We hammer the TLAS with ≥500 rebuild cycles and
#    assert every counter stays under a tight ceiling relative to
#    baseline. If any GPU buffer that was freed mid-cycle was still
#    referenced by a pending dispatch, a hit result would come back with
#    nonsense values (or the device would be lost) — the correctness
#    pass under the same iterations catches that.
# ==============================================================================

using Pkg
Pkg.activate("/sim/Programmieren/VulkanDev")

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Lava
using Raycore
using KernelAbstractions
const KA = KernelAbstractions

# ------------------------------------------------------------------------------
# Scene helpers
# ------------------------------------------------------------------------------

"""Single-triangle mesh at z=0: verts (0,0,0), (1,0,0), (0,1,0)."""
function unit_triangle_mesh()
    verts = [Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0)]
    faces = [GLTriangleFace(1,2,3)]
    return GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))
end

"""Axis-aligned box mesh (GeometryBasics)."""
box_mesh(origin::Vec3f, extent::Vec3f) =
    GeometryBasics.normal_mesh(Rect3f(origin, extent))

"""Translation matrix as `Mat4f`."""
translation(dx, dy, dz) = SMatrix{4,4,Float32,16}(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    dx, dy, dz, 1,
)

"""Downward ray hits a z=0 triangle translated by `offset` iff the ray's
xy point lies inside the translated triangle; returns the analytic t."""
function analytic_hit(ray_ox, ray_oy, ray_oz, offset::NTuple{3,Float32})
    ox, oy = ray_ox - offset[1], ray_oy - offset[2]
    # Point-in-triangle test for the unit-triangle (0,0), (1,0), (0,1):
    inside = (ox >= 0f0) && (oy >= 0f0) && (ox + oy <= 1f0)
    inside ? (true, ray_oz - offset[3]) : (false, 0f0)
end

# ------------------------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------------------------

"""Build an HWTLAS on Lava with `N` instances of the unit triangle, each
placed at a distinct translation so a single vertical ray per instance
hits each one exactly once."""
function build_scene(N::Int)
    hwtlas = Raycore.HWTLAS(LavaBackend())
    mesh = unit_triangle_mesh()
    handles = Raycore.TLASHandle[]
    offsets = NTuple{3,Float32}[]
    for i in 1:N
        off = (Float32(2i), 0f0, 0f0)  # spaced 2 apart in x, z=0
        T = translation(off...)
        h = push!(hwtlas, mesh, T; instance_id=UInt32(i))
        push!(handles, h)
        push!(offsets, off)
    end
    Raycore.sync!(hwtlas)
    return hwtlas, handles, offsets
end

"""Build test rays: one ray per instance, dropping from z=+1 straight
down through the triangle centroid at x=offset.x+0.25, y=0.25."""
function make_rays(offsets::Vector{NTuple{3,Float32}})
    rays = [Raycore.RTRay(ox + 0.25f0, 0.25f0, 1f0, 0f0, 0f0, 0f0, -1f0, 1f3)
            for (ox, _, _) in offsets]
    return rays
end

"""Upload rays + run HW trace and return the hit result vector (on CPU)."""
function trace_rays_cpu(hwtlas, offsets)
    rays   = make_rays(offsets)
    n      = length(rays)
    gpu_r  = LavaArray(rays)
    gpu_h  = LavaArray(fill(Raycore.RTHitResult(0, 0, 0, 0, 0, 0, 0, 0), n))
    accel  = Lava.HardwareAccel(hwtlas)
    Lava.trace_closest_hits!(gpu_h, gpu_r, accel, n)
    return Array(gpu_h)
end

# ------------------------------------------------------------------------------
# Correctness test
# ------------------------------------------------------------------------------

"""Assert that every ray hit matches the analytic expectation for the
current set of `offsets` (one per instance). Used after each mutation."""
function assert_hits_match(hits::Vector{Raycore.RTHitResult},
                           offsets::Vector{NTuple{3,Float32}})
    @assert length(hits) == length(offsets)
    all_good = true
    for i in eachindex(offsets)
        off = offsets[i]
        (want_hit, want_t) = analytic_hit(Float32(off[1] + 0.25f0),
                                           Float32(0.25f0),
                                           Float32(1),
                                           off)
        h = hits[i]
        got_hit = h.hit != UInt32(0)
        if got_hit != want_hit
            @warn "instance $i: hit mismatch" got_hit want_hit offset=off
            all_good = false
            continue
        end
        if want_hit && !isapprox(Float32(h.t), want_t; atol=1f-3)
            @warn "instance $i: t mismatch" got_t=h.t want_t offset=off
            all_good = false
        end
    end
    return all_good
end

@testset "HW TLAS — correctness under mutation" begin
    N = 8
    hwtlas, handles, offsets = build_scene(N)

    # Baseline: every instance hits as expected.
    @test assert_hits_match(trace_rays_cpu(hwtlas, offsets), offsets)

    # Mutate every transform with random translations; confirm hits follow.
    for iter in 1:20
        for i in 1:N
            new_off = (Float32(2i + 0.1 * iter),
                       Float32(0.3 * sinpi(iter / 5)),
                       Float32(-0.05 * iter))
            offsets[i] = new_off
            @assert Raycore.update_transform!(hwtlas, handles[i],
                                              translation(new_off...))
        end
        Raycore.sync!(hwtlas)
        @test assert_hits_match(trace_rays_cpu(hwtlas, offsets), offsets)
    end

    # Delete half, re-push with new transforms; topology change path.
    kept_handles  = handles[1:2:end]
    kept_offsets  = offsets[1:2:end]
    for h in handles[2:2:end]
        Raycore.delete!(hwtlas, h)
    end
    # Re-push fresh instances: same mesh, new offsets.
    new_offsets = [(Float32(100 + 2i), Float32(1), Float32(0.2 * i))
                   for i in 1:(N÷2)]
    for off in new_offsets
        h = push!(hwtlas, unit_triangle_mesh(), translation(off...);
                  instance_id=UInt32(99))
        push!(kept_handles, h)
        push!(kept_offsets, off)
    end
    Raycore.sync!(hwtlas)
    @test assert_hits_match(trace_rays_cpu(hwtlas, kept_offsets), kept_offsets)
end

# ------------------------------------------------------------------------------
# Stress / leak test
# ------------------------------------------------------------------------------

function snapshot_state()
    gpu_bytes  = Lava.GPU_LIVE_BYTES[]
    n_buffers  = length(Lava.LIVE_BUFFERS)
    n_pool     = length(Lava.POOL_BLOCKS)
    (gpu_bytes=gpu_bytes, live_bufs=n_buffers, pool_blocks=n_pool)
end

@testset "HW TLAS — stress / leak bounds" begin
    N = 16
    hwtlas, handles, offsets = build_scene(N)

    # Warm up: do a couple of rebuild cycles + a trace, then GC. The
    # baseline we lock onto is the *post-warmup* state; the first few
    # iterations legitimately populate pools, kernel caches, SBT, etc.
    for _ in 1:4
        for i in 1:N
            offsets[i] = (Float32(2i), Float32(0.1), 0f0)
            Raycore.update_transform!(hwtlas, handles[i], translation(offsets[i]...))
        end
        Raycore.sync!(hwtlas)
        _ = trace_rays_cpu(hwtlas, offsets)
    end
    GC.gc(true); GC.gc(true)
    baseline = snapshot_state()
    @info "baseline" baseline

    # Hammer the rebuild cycle. Every iteration:
    #   * shuffle per-instance transforms (push fresh NTuple{12,Float32})
    #   * call sync! — rebuilds TLAS (and maybe BLAS offsets) on Lava
    #   * trace + verify — catches use-after-free and UAF-masked noise
    n_iters   = 500
    max_hits  = 0  # debugging aid — peak samples during the loop
    for iter in 1:n_iters
        for i in 1:N
            offsets[i] = (Float32(2i + 0.01 * iter),
                          Float32(0.2 * cospi(iter / 7)),
                          Float32(-0.05 * sinpi(iter / 9)))
            Raycore.update_transform!(hwtlas, handles[i], translation(offsets[i]...))
        end
        Raycore.sync!(hwtlas)
        hits = trace_rays_cpu(hwtlas, offsets)
        @assert assert_hits_match(hits, offsets) "iter $iter: hits diverged — UAF suspected"
        max_hits = max(max_hits, count(h -> h.hit != 0, hits))
    end
    GC.gc(true); GC.gc(true)
    final = snapshot_state()
    @info "after $n_iters iterations" final peak_hits=max_hits

    # Allow small growth (pool block fragmentation, kernel caches warming up),
    # but the delta must not scale with iteration count. These ceilings are
    # intentionally tight — anything looser stops being a real leak test.
    @testset "no unbounded GPU memory growth" begin
        @test final.gpu_bytes   <= baseline.gpu_bytes + 256 * 1024^2  # +256 MiB
        @test final.live_bufs   <= baseline.live_bufs + 16
        @test final.pool_blocks <= baseline.pool_blocks + 8
    end
end

println("\nAll HW TLAS tests passed.")
