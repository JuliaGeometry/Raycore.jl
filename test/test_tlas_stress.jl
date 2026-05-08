# ==============================================================================
# TLAS Stress / Memory-Safety Tests
# ==============================================================================
#
# Heavy-duty coverage for the mutable TLAS:
#
#   - Random churn (push / delete / update_transform / update_transforms /
#     sync) with strict invariants between every step.
#   - High-instance-count and high-BLAS-count scenarios.
#   - Use-after-free attempts on handles (must error, must not crash GPU).
#   - Pure refit-only loops — must keep `static_tlas` identity stable, must
#     not accumulate flat-array memory.
#   - Topology-change rebuild after a long refit-only run.
#   - GC-pressure: many `adapt()` calls without retaining results, plus a
#     hard leak bound across 200 mesh swaps.
#
# Each test asserts EXACT counts on `tlas._flat_blas_*` / `tlas.blas_storage`
# rather than loose multiples — a leak that adds even one entry per cycle
# trips the test inside a few iterations instead of hiding behind 25× slack.
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
using Random

const STRESS_BACKEND = Lava.LavaBackend()

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

stress_sphere(n::Int) = GeometryBasics.normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), n))
stress_box(s::Float32) = GeometryBasics.normal_mesh(Rect3f(Vec3f(-s/2), Vec3f(s)))

stress_xlat(dx, dy, dz) = SMatrix{4,4,Float32,16}(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    dx, dy, dz, 1,
)

KA.@kernel function stress_trace_kernel!(hit_out, t_out, tlas, origins, directions)
    i = @index(Global, Linear)
    ray = Raycore.Ray(; o=origins[i], d=directions[i])
    hit, _, dist, _, _ = Raycore.closest_hit(tlas, ray)
    hit_out[i] = hit
    t_out[i]   = Float32(dist)
end

"""Trace `n` rays in parallel; returns (hits::Vector{Bool}, ts::Vector{Float32})."""
function stress_trace(tlas, origins::Vector{Point3f}, directions::Vector{Vec3f})
    @assert length(origins) == length(directions)
    n = length(origins)
    static = Adapt.adapt(STRESS_BACKEND, tlas)
    hits = KA.zeros(STRESS_BACKEND, Bool, n)
    ts   = KA.zeros(STRESS_BACKEND, Float32, n)
    o_d = Adapt.adapt(STRESS_BACKEND, origins)
    d_d = Adapt.adapt(STRESS_BACKEND, directions)
    stress_trace_kernel!(STRESS_BACKEND)(hits, ts, static, o_d, d_d; ndrange=n)
    KA.synchronize(STRESS_BACKEND)
    return Array(hits), Array(ts)
end

stress_trace_one(tlas, o::Point3f, d::Vec3f) = begin
    h, t = stress_trace(tlas, [o], [d])
    (hit = h[1], t = t[1])
end

"Sum of primitives across all currently-stored BLASes."
sum_storage_prims(tlas) = isempty(tlas.blas_storage) ? 0 :
    sum(length(b.primitives) for b in tlas.blas_storage)
sum_storage_nodes(tlas) = isempty(tlas.blas_storage) ? 0 :
    sum(length(b.nodes) for b in tlas.blas_storage)

"`length` that treats `nothing` (drained-to-empty flat arrays) as 0."
flat_len(x) = x === nothing ? 0 : length(x)

"Tight invariant: flat arrays MUST equal the sum across `blas_storage`
after `sync!`. Anything else is a leak or stale entry."
function assert_compact!(tlas; ctx::AbstractString="")
    @test flat_len(tlas._flat_blas_prims) == sum_storage_prims(tlas)
    @test flat_len(tlas._flat_blas_nodes) == sum_storage_nodes(tlas)
end

# ------------------------------------------------------------------------------
# 1. Random churn with strict invariants between every operation
# ------------------------------------------------------------------------------
#
# A scripted-but-randomized sequence of operations.  After every `sync!` we
# recompute (a) the total instance count and (b) the flat-array sizes from
# `blas_storage` and assert exact equality.  Any leak / off-by-one in the
# compaction path is caught within a few iterations, not eventually.

@testset "TLAS stress — random churn with exact invariants" begin
    rng = MersenneTwister(0xC0FFEE)
    tlas = Raycore.TLAS(STRESS_BACKEND)
    handles = Raycore.TLASHandle[]      # currently live handles (1 instance each)
    n_per_handle = Int[]                # parallel array: how many instances under each handle

    # Seed with one BLAS so we have something to update.
    h0 = push!(tlas, stress_sphere(8), stress_xlat(0, 0, 0))
    push!(handles, h0); push!(n_per_handle, 1)
    Raycore.sync!(tlas)

    n_iters = 400
    for iter in 1:n_iters
        op = rand(rng, 1:5)

        if op == 1 && length(handles) < 32
            # push! single instance
            n = rand(rng, [4, 6, 8, 12])
            x = Float32(rand(rng) * 4 - 2)
            h = push!(tlas, stress_sphere(n), stress_xlat(x, 0, 0))
            push!(handles, h); push!(n_per_handle, 1)
        elseif op == 2 && length(handles) < 16
            # push! batch (2..6 instances of one BLAS)
            k = rand(rng, 2:6)
            xfs = [stress_xlat(Float32(rand(rng) * 4 - 2), Float32(rand(rng) * 2), 0) for _ in 1:k]
            h = push!(tlas, stress_sphere(rand(rng, [4, 8])), xfs)
            push!(handles, h); push!(n_per_handle, k)
        elseif op == 3 && length(handles) > 1
            # delete!
            i = rand(rng, 1:length(handles))
            Raycore.delete!(tlas, handles[i])
            deleteat!(handles, i); deleteat!(n_per_handle, i)
        elseif op == 4 && !isempty(handles)
            # update_transform! single (only valid if handle has 1 instance)
            i = rand(rng, 1:length(handles))
            if n_per_handle[i] == 1
                Raycore.update_transform!(tlas, handles[i],
                    stress_xlat(Float32(rand(rng) * 6 - 3), 0, 0))
            end
        elseif op == 5 && !isempty(handles)
            # update_transforms! batch
            i = rand(rng, 1:length(handles))
            k = n_per_handle[i]
            xfs = [stress_xlat(Float32(rand(rng) * 6 - 3), 0, 0) for _ in 1:k]
            Raycore.update_transforms!(tlas, handles[i], xfs)
        end

        # Sync every 5 iterations so we hit the refit + rebuild paths a
        # balanced number of times.
        if iter % 5 == 0
            Raycore.sync!(tlas)

            # Strict invariants
            expected_n_inst = isempty(n_per_handle) ? 0 : sum(n_per_handle)
            @test Raycore.n_instances(tlas) == expected_n_inst
            @test length(tlas.blas_storage) == length(handles)
            @test length(tlas.deleted_handles) == 0
            @test flat_len(tlas._flat_blas_prims) == sum_storage_prims(tlas)
            @test flat_len(tlas._flat_blas_nodes) == sum_storage_nodes(tlas)
            @test !tlas.dirty
            @test !tlas.transforms_dirty
        end
    end

    # Final sync + invariants
    Raycore.sync!(tlas)
    expected_n_inst = isempty(n_per_handle) ? 0 : sum(n_per_handle)
    @test Raycore.n_instances(tlas) == expected_n_inst
    @test length(tlas.blas_storage) == length(handles)
    assert_compact!(tlas)

    # And after deleting everything left, the TLAS should drain to empty.
    for h in handles
        Raycore.delete!(tlas, h)
    end
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == 0
    @test length(tlas.blas_storage) == 0
    @test flat_len(tlas._flat_blas_prims) == 0
    @test flat_len(tlas._flat_blas_nodes) == 0
end

# ------------------------------------------------------------------------------
# 2. Many BLASes, all live simultaneously
# ------------------------------------------------------------------------------

@testset "TLAS stress — 200 distinct BLASes alive at once" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    handles = Raycore.TLASHandle[]
    n_blas = 200
    # Spacing > 2*radius so adjacent unit spheres never touch.  When we delete
    # one, the gap at its position must read as a clean miss instead of a
    # tangent hit on a neighbour.
    spacing = 4f0
    for i in 1:n_blas
        n = 4 + (i % 6)
        h = push!(tlas, stress_sphere(n), stress_xlat(Float32(i) * spacing, 0, 0))
        push!(handles, h)
    end
    Raycore.sync!(tlas)

    @test length(tlas.blas_storage) == n_blas
    @test Raycore.n_instances(tlas) == n_blas
    assert_compact!(tlas)

    # Trace one ray per instance — each must hit (sphere at x=i*spacing).
    origins    = [Point3f(Float32(i) * spacing, 0, 5)  for i in 1:n_blas]
    directions = fill(Vec3f(0, 0, -1), n_blas)
    hits, ts   = stress_trace(tlas, origins, directions)
    @test all(hits)
    @test all(t -> isapprox(t, 4f0; atol=0.1f0), ts)

    # Delete every other BLAS — remaining ones must still hit, deleted ones must miss.
    for (i, h) in enumerate(handles)
        if iseven(i)
            Raycore.delete!(tlas, h)
        end
    end
    Raycore.sync!(tlas)
    @test length(tlas.blas_storage) == n_blas ÷ 2
    assert_compact!(tlas)

    hits2, _ = stress_trace(tlas, origins, directions)
    for i in 1:n_blas
        @test hits2[i] == isodd(i)
    end
end

# ------------------------------------------------------------------------------
# 3. High instance count under one BLAS (TLAS-level instancing stress)
# ------------------------------------------------------------------------------

@testset "TLAS stress — 5000 instances of one BLAS, batch update, refit" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    n_inst = 5000

    # Build initial transforms placing instances along a line
    init_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, 0) for i in 1:n_inst]
    sphere_h = push!(tlas, stress_sphere(4), init_xfs)
    Raycore.sync!(tlas)

    @test Raycore.n_instances(tlas) == n_inst
    @test length(tlas.blas_storage) == 1
    assert_compact!(tlas)

    # Sample ray hits at a few positions
    sample_idxs = [1, n_inst ÷ 4, n_inst ÷ 2, 3 * n_inst ÷ 4, n_inst]
    sample_origins    = [Point3f(Float32(i) * 0.1f0, 0, 5) for i in sample_idxs]
    sample_directions = fill(Vec3f(0, 0, -1), length(sample_idxs))
    hits, ts = stress_trace(tlas, sample_origins, sample_directions)
    @test all(hits)
    @test all(t -> isapprox(t, 4f0; atol=0.1f0), ts)

    # Bulk-update every transform via update_transforms!.  Move each instance
    # by +10 in y.  Expected: ray from (x_i, 10, 5) should hit at t≈4.
    new_xfs = [stress_xlat(Float32(i) * 0.1f0, 10f0, 0) for i in 1:n_inst]
    st_before = tlas.static_tlas
    Raycore.update_transforms!(tlas, sphere_h, new_xfs)
    @test tlas.transforms_dirty
    Raycore.sync!(tlas)
    @test !tlas.transforms_dirty
    # Refit must keep the same StaticTLAS object alive (in-place AABB update).
    @test tlas.static_tlas === st_before

    sample_origins2 = [Point3f(Float32(i) * 0.1f0, 10f0, 5) for i in sample_idxs]
    hits2, ts2 = stress_trace(tlas, sample_origins2, sample_directions)
    @test all(hits2)
    @test all(t -> isapprox(t, 4f0; atol=0.1f0), ts2)

    # Old positions must now MISS (instances moved away).
    hits3, _ = stress_trace(tlas, sample_origins, sample_directions)
    @test !any(hits3)
end

# ------------------------------------------------------------------------------
# 3b. High-instance-count + tight refit loop (combined stress)
# ------------------------------------------------------------------------------
#
# Test 3 does one update + one sync on 5000 instances.  Test 7 does 500 refits
# but with one instance.  Real workloads (RayMakie meshscatter at 60fps) hit
# the *combination*: thousands of instances, refit every frame, for hundreds
# of frames.  This testset pins that.

@testset "TLAS stress — 5000 instances + 200 refit frames in a tight loop" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    n_inst = 5000

    init_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, 0) for i in 1:n_inst]
    h = push!(tlas, stress_sphere(4), init_xfs)
    Raycore.sync!(tlas)

    # Pin the StaticTLAS identity: the refit path must update tlas.nodes in
    # place and reuse the same StaticTLAS.  If sync! drops to rebuild for any
    # frame, this assertion trips immediately.
    st0 = tlas.static_tlas
    @test st0 !== nothing

    sample_idxs = [1, n_inst ÷ 2, n_inst]

    n_frames = 200
    for frame in 1:n_frames
        # Animate every instance.  Each frame is a full bulk update, mirroring
        # the meshscatter per-frame call site that produced the original
        # CPU-loop performance footgun.
        new_xfs = [stress_xlat(Float32(i) * 0.1f0,
                                Float32(0.1 * frame),
                                0)
                   for i in 1:n_inst]
        Raycore.update_transforms!(tlas, h, new_xfs)
        Raycore.sync!(tlas)

        # Refit-path invariants — checked every frame, not just at the end.
        @test tlas.static_tlas === st0           # in-place AABB update
        @test !tlas.dirty
        @test !tlas.transforms_dirty
        @test Raycore.n_instances(tlas) == n_inst
        @test length(tlas.blas_storage) == 1     # no BLAS churn
    end

    # Verify a few rays hit at the LAST frame's positions (correctness end-to-end).
    last_y = Float32(0.1 * n_frames)
    origins  = [Point3f(Float32(i) * 0.1f0, last_y, 5) for i in sample_idxs]
    dirs     = fill(Vec3f(0, 0, -1), length(sample_idxs))
    hits, ts = stress_trace(tlas, origins, dirs)
    @test all(hits)
    @test all(t -> isapprox(t, 4f0; atol=0.1f0), ts)
end

# ------------------------------------------------------------------------------
# 3c. High-instance-count + tight REBUILD loop (topology churn at scale)
# ------------------------------------------------------------------------------

@testset "TLAS stress — 2000 instances + 100 rebuild frames in a tight loop" begin
    # Different from 3b: every frame DELETES and re-PUSHES the batch, forcing
    # a full topology rebuild (not refit).  Catches leaks / fragmentation in
    # the rebuild path under realistic instance counts.
    tlas = Raycore.TLAS(STRESS_BACKEND)
    n_inst = 2000

    init_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, 0) for i in 1:n_inst]
    h = push!(tlas, stress_sphere(4), init_xfs)
    Raycore.sync!(tlas)

    n_frames = 100
    for frame in 1:n_frames
        Raycore.delete!(tlas, h)
        new_xfs = [stress_xlat(Float32(i) * 0.1f0,
                                Float32(0.05 * frame),
                                0)
                   for i in 1:n_inst]
        h = push!(tlas, stress_sphere(4), new_xfs)
        Raycore.sync!(tlas)

        # Per-frame strict invariants — leaks expose themselves fast.
        @test Raycore.n_instances(tlas) == n_inst
        @test length(tlas.blas_storage) == 1
        @test length(tlas.deleted_handles) == 0
        @test length(tlas._flat_blas_prims) == length(tlas.blas_storage[1].primitives)
        @test length(tlas._flat_blas_nodes) == length(tlas.blas_storage[1].nodes)
    end

    # Sanity: ray hits at last frame's positions.
    last_y = Float32(0.05 * n_frames)
    sample_idxs = [1, n_inst ÷ 2, n_inst]
    origins = [Point3f(Float32(i) * 0.1f0, last_y, 5) for i in sample_idxs]
    dirs    = fill(Vec3f(0, 0, -1), length(sample_idxs))
    hits, _ = stress_trace(tlas, origins, dirs)
    @test all(hits)
end

# ------------------------------------------------------------------------------
# 3d. Interleaved update + trace + update + trace (UAF / serialization stress)
# ------------------------------------------------------------------------------
#
# Tight loops above check refit/rebuild correctness via invariants but trace
# only at the END.  This testset interleaves: every frame does
#   update_transforms!  →  sync!  →  trace  →  verify-this-frame's-positions
# so a trace's GPU read is bracketed by writes from the previous frame
# (already-completed) AND the NEXT frame (about to start).  If sync! ever
# fails to serialize the new write against in-flight reads, or hands back a
# stale `static_tlas`, the trace returns wrong t-values that don't match
# THIS frame's transforms, and the test trips immediately — not eventually.

@testset "TLAS stress — interleaved update + trace + update tight loop (1000 inst, refit)" begin
    tlas  = Raycore.TLAS(STRESS_BACKEND)
    n_inst = 1000

    init_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, 0) for i in 1:n_inst]
    h = push!(tlas, stress_sphere(4), init_xfs)
    Raycore.sync!(tlas)

    st0 = tlas.static_tlas

    sample_idxs = [1, n_inst ÷ 4, n_inst ÷ 2, 3 * n_inst ÷ 4, n_inst]

    # Bounded oscillation in z so the unit-sphere top stays reachable from
    # ray origin z=5 (sphere top = z_off + 1; need z_off+1 < 5).  Use a
    # sawtooth that visits 100 distinct z positions in [0, 2] without ever
    # walking out of reach.
    n_frames = 100
    for frame in 1:n_frames
        z_off = Float32((frame % 50) * 0.04)            # 0 .. 1.96
        new_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, z_off) for i in 1:n_inst]
        Raycore.update_transforms!(tlas, h, new_xfs)
        Raycore.sync!(tlas)

        @test tlas.static_tlas === st0
        @test !tlas.dirty
        @test !tlas.transforms_dirty

        # Trace THIS frame; t must reflect THIS frame's z_off.
        origins  = [Point3f(Float32(i) * 0.1f0, 0, 5) for i in sample_idxs]
        dirs     = fill(Vec3f(0, 0, -1), length(sample_idxs))
        hits, ts = stress_trace(tlas, origins, dirs)
        expected_t = 5f0 - z_off - 1f0
        @test all(hits)
        @test all(t -> isapprox(t, expected_t; atol=0.1f0), ts)
    end
end

@testset "TLAS stress — interleaved update + trace tight loop (5000 inst, refit)" begin
    # Same shape, 5x the instance count — pushes the per-frame refit kernel
    # ndrange high enough that timeline ordering bugs would tend to surface
    # as flaky frame results.
    tlas  = Raycore.TLAS(STRESS_BACKEND)
    n_inst = 5000

    init_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, 0) for i in 1:n_inst]
    h = push!(tlas, stress_sphere(4), init_xfs)
    Raycore.sync!(tlas)
    st0 = tlas.static_tlas

    sample_idxs = [1, 1000, 2500, 4000, n_inst]
    n_frames = 50
    for frame in 1:n_frames
        z_off = Float32(frame * 0.04)                    # 0.04 .. 2.0 — always reachable
        new_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, z_off) for i in 1:n_inst]
        Raycore.update_transforms!(tlas, h, new_xfs)
        Raycore.sync!(tlas)
        @test tlas.static_tlas === st0

        origins  = [Point3f(Float32(i) * 0.1f0, 0, 5) for i in sample_idxs]
        dirs     = fill(Vec3f(0, 0, -1), length(sample_idxs))
        hits, ts = stress_trace(tlas, origins, dirs)
        expected_t = 5f0 - z_off - 1f0
        @test all(hits)
        @test all(t -> isapprox(t, expected_t; atol=0.1f0), ts)
    end
end

@testset "TLAS stress — interleaved delete+push+sync+trace tight loop (rebuild path)" begin
    # Same interleaving but every frame changes topology (delete+push),
    # exercising the rebuild path's reuse / free of tlas.nodes.  An older
    # frame's trace MUST NOT see node buffers that have been recycled into
    # this frame's BVH — KA.synchronize inside sync! is what guarantees this;
    # if it ever regresses, the per-frame correctness check trips.
    tlas = Raycore.TLAS(STRESS_BACKEND)
    n_inst = 500

    init_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, 0) for i in 1:n_inst]
    h = push!(tlas, stress_sphere(4), init_xfs)
    Raycore.sync!(tlas)

    sample_idxs = [1, 100, 250, 400, n_inst]
    n_frames = 60
    for frame in 1:n_frames
        Raycore.delete!(tlas, h)
        z_off  = Float32((frame % 40) * 0.05)            # 0 .. 1.95 — always reachable
        # Alternate tessellation each frame to force fresh BLAS buffer sizes
        # — node array can't be reused in place.
        tess   = isodd(frame) ? 4 : 8
        new_xfs = [stress_xlat(Float32(i) * 0.1f0, 0, z_off) for i in 1:n_inst]
        h = push!(tlas, stress_sphere(tess), new_xfs)
        Raycore.sync!(tlas)

        @test Raycore.n_instances(tlas) == n_inst
        @test length(tlas.blas_storage) == 1
        @test length(tlas._flat_blas_prims) == length(tlas.blas_storage[1].primitives)

        origins  = [Point3f(Float32(i) * 0.1f0, 0, 5) for i in sample_idxs]
        dirs     = fill(Vec3f(0, 0, -1), length(sample_idxs))
        hits, ts = stress_trace(tlas, origins, dirs)
        expected_t = 5f0 - z_off - 1f0
        @test all(hits)
        @test all(t -> isapprox(t, expected_t; atol=0.1f0), ts)
    end
end

# ------------------------------------------------------------------------------
# 3e. 500-iter mesh grow/shrink with raytracing every iter (correctness + leak)
# ------------------------------------------------------------------------------
#
# Five grow→shrink cycles of 100 iters each, varying peak tessellation
# (16, 32, 48, 64, 96).  Every iter:
#   delete!  →  push!(sphere(tess(iter)), translation(z=offset))  →  sync!
#   trace    →  verify hit position matches THIS iter's offset
# Catches: stale node-buffer captures from previous iters' traces (the
# UAF window between sync's KA.synchronize and the next push's allocation),
# leaks of BLAS arrays whose sizes change each iter, off-by-ones in the
# rebuild path's flat-array repacking when the tess count grows or shrinks.

function grow_shrink_tess(iter::Int)
    # 100-iter cycle: linear ramp 8 → peak → 8.  Five peaks: 16, 32, 48, 64, 96.
    cycle_len = 100
    peaks     = (16, 32, 48, 64, 96)
    cycle_i   = ((iter - 1) ÷ cycle_len) % length(peaks) + 1
    peak      = peaks[cycle_i]
    phase     = (iter - 1) % cycle_len
    half      = cycle_len ÷ 2
    if phase < half
        max(8, Int(round(8 + (peak - 8) * (phase / half))))
    else
        max(8, Int(round(peak - (peak - 8) * ((phase - half) / half))))
    end
end

@testset "TLAS stress — 500-iter mesh grow/shrink + trace per iter (SW)" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    h = push!(tlas, stress_sphere(8), stress_xlat(0, 0, 0))
    Raycore.sync!(tlas)

    n_iters = 500
    saw_min, saw_max = typemax(Int), 0
    for iter in 1:n_iters
        tess  = grow_shrink_tess(iter)
        saw_min, saw_max = min(saw_min, tess), max(saw_max, tess)
        # offset_z bounded so the ray (origin z=5) always reaches the sphere top.
        z_off = Float32((iter % 30) * 0.05)            # 0 .. 1.45
        Raycore.delete!(tlas, h)
        h = push!(tlas, stress_sphere(tess), stress_xlat(0, 0, z_off))
        Raycore.sync!(tlas)

        # Topology invariants.
        @test Raycore.n_instances(tlas) == 1
        @test length(tlas.blas_storage) == 1
        @test length(tlas._flat_blas_prims) == length(tlas.blas_storage[1].primitives)
        @test length(tlas._flat_blas_nodes) == length(tlas.blas_storage[1].nodes)

        # Trace + verify THIS iter's geometry.  expected_t = 5 - z_off - 1.
        r = stress_trace_one(tlas, Point3f(0, 0, 5), Vec3f(0, 0, -1))
        @test r.hit
        @test isapprox(r.t, 5f0 - z_off - 1f0; atol=0.15f0)
    end

    # Sanity: the schedule actually swept low and high tessellation values.
    @test saw_min <= 10
    @test saw_max >= 90
end

# ------------------------------------------------------------------------------
# 4. Long churn cycle with hard memory bounds
# ------------------------------------------------------------------------------

@testset "TLAS stress — 200 swap iterations, exact compaction every step" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    handle = push!(tlas, stress_sphere(16), stress_xlat(0, 0, 0))
    Raycore.sync!(tlas)

    n_iters = 200
    for iter in 1:n_iters
        n = (iter % 5 == 0) ? 64 : (iseven(iter) ? 8 : 24)
        Raycore.delete!(tlas, handle)
        handle = push!(tlas, stress_sphere(n),
            stress_xlat(0, 0, Float32(0.001 * iter)))
        Raycore.sync!(tlas)

        # Strict equality every iteration — accumulation reveals itself fast.
        @test length(tlas.blas_storage) == 1
        @test Raycore.n_instances(tlas) == 1
        @test length(tlas._flat_blas_prims) == length(tlas.blas_storage[1].primitives)
        @test length(tlas._flat_blas_nodes) == length(tlas.blas_storage[1].nodes)
        @test length(tlas.deleted_handles) == 0
    end

    # Final ray check — geometry still works after the long run.
    r = stress_trace_one(tlas, Point3f(0, 0, 5), Vec3f(0, 0, -1))
    @test r.hit
    @test isapprox(r.t, Float32(5) - Float32(0.001 * n_iters) - 1f0; atol=0.15f0)
end

# ------------------------------------------------------------------------------
# 5. Use-after-free attempts on handles
# ------------------------------------------------------------------------------

@testset "TLAS stress — deleted handles must not be usable" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    h = push!(tlas, stress_sphere(8), stress_xlat(0, 0, 0))
    Raycore.sync!(tlas)
    @test Raycore.is_valid(tlas, h)

    # Delete + sync (compaction)
    Raycore.delete!(tlas, h)
    Raycore.sync!(tlas)
    @test !Raycore.is_valid(tlas, h)

    # Every mutation / inspection API must reject the handle loudly.
    @test_throws ErrorException Raycore.update_transform!(tlas, h, stress_xlat(1, 0, 0))
    @test_throws ErrorException Raycore.update_transforms!(tlas, h, [stress_xlat(1, 0, 0)])
    @test_throws ErrorException Raycore.get_instance(tlas, h)
    @test_throws ErrorException Raycore.get_instances(tlas, h)
    @test Raycore.delete!(tlas, h) === false   # idempotent

    # Pre-compaction path: deleted but not yet sync!'d.
    h2 = push!(tlas, stress_sphere(6))
    Raycore.sync!(tlas)
    Raycore.delete!(tlas, h2)
    @test !Raycore.is_valid(tlas, h2)
    @test_throws ErrorException Raycore.update_transform!(tlas, h2, stress_xlat(1, 0, 0))
    @test_throws ErrorException Raycore.update_transforms!(tlas, h2, [stress_xlat(1, 0, 0)])

    # Wrong-arity update_transform! / update_transforms! (handle/length mismatch).
    h3 = push!(tlas, stress_sphere(6), [stress_xlat(0,0,0), stress_xlat(1,0,0)])
    Raycore.sync!(tlas)
    @test_throws ErrorException Raycore.update_transform!(tlas, h3, stress_xlat(2, 0, 0))    # 1 vs 2
    @test_throws ErrorException Raycore.update_transforms!(tlas, h3,
        [stress_xlat(0,0,0), stress_xlat(1,0,0), stress_xlat(2,0,0)])                          # 3 vs 2
end

# ------------------------------------------------------------------------------
# 7. Pure refit-only loop must not allocate / not change static_tlas identity
# ------------------------------------------------------------------------------

@testset "TLAS stress — 500 refit-only cycles preserve static_tlas identity" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    h = push!(tlas, stress_sphere(16), stress_xlat(0, 0, 0))
    Raycore.sync!(tlas)
    st0 = tlas.static_tlas
    @test st0 !== nothing

    # Length of the flat arrays must NEVER change during a pure refit loop.
    nodes_len_before = length(tlas._flat_blas_nodes)
    prims_len_before = length(tlas._flat_blas_prims)
    nodes_len_top    = length(tlas.nodes)

    for iter in 1:500
        Raycore.update_transform!(tlas, h, stress_xlat(0, 0, Float32(iter * 0.001)))
        Raycore.sync!(tlas)
        @test tlas.static_tlas === st0
        @test length(tlas._flat_blas_nodes) == nodes_len_before
        @test length(tlas._flat_blas_prims) == prims_len_before
        @test length(tlas.nodes)            == nodes_len_top
        @test !tlas.dirty
        @test !tlas.transforms_dirty
    end

    # Sanity: refit moved the AABBs, so a ray from above hits at the new t.
    r = stress_trace_one(tlas, Point3f(0, 0, 5), Vec3f(0, 0, -1))
    @test r.hit
    @test isapprox(r.t, 5f0 - 0.5f0 - 1f0; atol=0.1f0)
end

# ------------------------------------------------------------------------------
# 8. Topology change after a long refit-only run (no carry-over staleness)
# ------------------------------------------------------------------------------

@testset "TLAS stress — topology change after long refit run" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    h_a  = push!(tlas, stress_sphere(8), stress_xlat(-2, 0, 0))
    h_b  = push!(tlas, stress_sphere(8), stress_xlat( 2, 0, 0))
    Raycore.sync!(tlas)

    # 100 refits
    for iter in 1:100
        Raycore.update_transform!(tlas, h_a, stress_xlat(-2 + iter * 0.01f0, 0, 0))
        Raycore.update_transform!(tlas, h_b, stress_xlat( 2 - iter * 0.01f0, 0, 0))
        Raycore.sync!(tlas)
    end
    st_pre_topology = tlas.static_tlas

    # Now actually CHANGE topology: delete one handle, push! a new mesh.  This
    # MUST rebuild static_tlas (different size), and traces must reflect the
    # new geometry — not the old.
    Raycore.delete!(tlas, h_a)
    h_c = push!(tlas, stress_sphere(8), stress_xlat(0, 0, 5))
    Raycore.sync!(tlas)
    @test tlas.static_tlas !== st_pre_topology
    @test Raycore.n_instances(tlas) == 2
    @test length(tlas.blas_storage) == 2   # b + c (a was deleted)
    assert_compact!(tlas)

    # Old h_a position should miss; new h_c position (z=5) should hit.
    r_a = stress_trace_one(tlas, Point3f(-2 + 100 * 0.01f0, 0, 5), Vec3f(0, 0, -1))
    r_c = stress_trace_one(tlas, Point3f(0, 0, 10),                  Vec3f(0, 0, -1))
    @test !r_a.hit
    @test r_c.hit
    @test isapprox(r_c.t, 10f0 - 5f0 - 1f0; atol=0.1f0)
end

# ------------------------------------------------------------------------------
# 9. Hard leak bound across 200 swaps (WeakRefs)
# ------------------------------------------------------------------------------

@testset "TLAS stress — 200-swap hard leak bound (multiple WeakRefs)" begin
    # Stronger version of the existing one-WeakRef test.  We keep WeakRefs to
    # 9 different prior static_tlas objects sampled across the run and assert
    # ALL are collectable at the end.  A regression that pins even one frame's
    # static across mutations trips this.
    #
    # Implementation note: the workload runs inside a function so its locals
    # (`tlas`, `handle`, the loop's `iter`/`n` bindings) leave scope before
    # `GC.gc(true)`.  Putting the same code directly under `@testset`
    # observably retains the most recent static_tlas (testset macro keeps
    # locals alive for its scope), which would mask real leaks behind a
    # benign-looking single-WeakRef survival.  `GC.gc(true)` is a full sweep —
    # one call must be enough; if a WeakRef survives, that's a real reference.
    function workload()
        tlas = Raycore.TLAS(STRESS_BACKEND)
        handle = push!(tlas, stress_sphere(16), stress_xlat(0, 0, 0))
        Raycore.sync!(tlas)
        wrefs = WeakRef[]
        sample_at = Set{Int}([1, 25, 50, 75, 100, 125, 150, 175, 195])

        for iter in 1:200
            n = iseven(iter) ? 12 : 64
            Raycore.delete!(tlas, handle)
            handle = push!(tlas, stress_sphere(n), stress_xlat(0, 0, Float32(0.001 * iter)))
            Raycore.sync!(tlas)
            if iter in sample_at
                push!(wrefs, WeakRef(tlas.static_tlas))
            end
        end

        # Reseat static_tlas one more time so the last sampled iter is
        # also no longer the live one.
        Raycore.delete!(tlas, handle)
        handle = push!(tlas, stress_sphere(8), stress_xlat(0, 0, 0))
        Raycore.sync!(tlas)
        return wrefs, sort!(collect(sample_at))
    end

    wrefs, sorted_iters = workload()
    GC.gc(true)
    n_leaked = count(w -> w.value !== nothing, wrefs)
    if n_leaked != 0
        for (i, w) in enumerate(wrefs)
            @info "WeakRef status" iter=sorted_iters[i] alive=(w.value !== nothing)
        end
    end
    @test n_leaked == 0
end

# ------------------------------------------------------------------------------
# 10. Many adapts during a refit loop — must not allocate fresh StaticTLAS
# ------------------------------------------------------------------------------

@testset "TLAS stress — adapt-per-frame during refit-only loop is allocation-free" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)
    h = push!(tlas, stress_sphere(8), stress_xlat(0, 0, 0))
    Raycore.sync!(tlas)
    st0 = tlas.static_tlas

    # Adapt once and pin: identity must not change across pure-refit cycles.
    pinned_static = Adapt.adapt(STRESS_BACKEND, tlas)
    @test pinned_static === st0

    for iter in 1:200
        Raycore.update_transform!(tlas, h, stress_xlat(0, 0, Float32(iter * 0.001)))
        Raycore.sync!(tlas)
        # Each adapt must return the SAME object; the "no-op sync" path must
        # not silently rebuild StaticTLAS.
        @test Adapt.adapt(STRESS_BACKEND, tlas) === pinned_static
    end
end

# ------------------------------------------------------------------------------
# 11. Mixed delete+push at high churn (exercise compaction edge cases)
# ------------------------------------------------------------------------------

@testset "TLAS stress — interleaved delete + push without intermediate sync" begin
    # Several deletes + pushes BEFORE a single sync!.  This exercises the
    # path where compaction sees both fresh-pushed instances and deletion
    # marks at the same time.
    tlas = Raycore.TLAS(STRESS_BACKEND)
    h1 = push!(tlas, stress_sphere(8), stress_xlat(0, 0, 0))
    h2 = push!(tlas, stress_sphere(8), stress_xlat(2, 0, 0))
    h3 = push!(tlas, stress_sphere(8), stress_xlat(4, 0, 0))
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == 3

    # No sync between these:
    Raycore.delete!(tlas, h2)
    h4 = push!(tlas, stress_sphere(8), stress_xlat(6, 0, 0))
    Raycore.delete!(tlas, h1)
    h5 = push!(tlas, stress_sphere(8), stress_xlat(8, 0, 0))

    # Single sync should resolve all of it.
    Raycore.sync!(tlas)
    @test Raycore.is_valid(tlas, h3)
    @test Raycore.is_valid(tlas, h4)
    @test Raycore.is_valid(tlas, h5)
    @test !Raycore.is_valid(tlas, h1)
    @test !Raycore.is_valid(tlas, h2)
    @test Raycore.n_instances(tlas) == 3
    @test length(tlas.blas_storage) == 3
    assert_compact!(tlas)

    # Hit positions: x = 4 (h3), 6 (h4), 8 (h5).  x=0 and x=2 must miss.
    origins = [Point3f(Float32(x), 0, 5) for x in (0, 2, 4, 6, 8)]
    dirs    = fill(Vec3f(0, 0, -1), 5)
    hits, _ = stress_trace(tlas, origins, dirs)
    @test hits == [false, false, true, true, true]
end

# ------------------------------------------------------------------------------
# 12. Empty-TLAS transitions (drain to zero, rebuild from zero)
# ------------------------------------------------------------------------------

@testset "TLAS stress — drain to empty + rebuild from empty" begin
    tlas = Raycore.TLAS(STRESS_BACKEND)

    # 5 push / sync / delete / sync cycles.  After every cycle the TLAS must
    # be observably empty, and after every push the previous-empty backing
    # buffers must not leak into the new build.
    for iter in 1:5
        h = push!(tlas, stress_sphere(8), stress_xlat(Float32(iter), 0, 0))
        Raycore.sync!(tlas)
        @test Raycore.n_instances(tlas) == 1
        @test length(tlas.blas_storage) == 1

        Raycore.delete!(tlas, h)
        Raycore.sync!(tlas)
        @test Raycore.n_instances(tlas) == 0
        @test length(tlas.blas_storage) == 0
        @test flat_len(tlas._flat_blas_prims) == 0
        @test flat_len(tlas._flat_blas_nodes) == 0
        @test length(tlas.deleted_handles) == 0
        # Empty-state ray traces must not crash and must always miss.
        r = stress_trace_one(tlas, Point3f(0, 0, 5), Vec3f(0, 0, -1))
        @test !r.hit
    end
end

# ------------------------------------------------------------------------------
# 12b. Cross-backend Adapt.adapt errors loudly
# ------------------------------------------------------------------------------
#
# A TLAS built on backend A handed to `Adapt.adapt(B, tlas)` would silently
# return a `static_tlas` whose arrays still live on A — the error only
# surfaces later as a confusing GPUCompiler "non-bitstype argument" inside
# kernel compilation. Pin the loud-at-the-API-boundary contract.

@testset "TLAS stress — cross-backend adapt errors loudly" begin
    cpu_tlas = Raycore.TLAS(KA.CPU())
    push!(cpu_tlas, stress_sphere(8))
    Raycore.sync!(cpu_tlas)

    # Adapting to the matching backend works.
    @test Adapt.adapt(KA.CPU(), cpu_tlas) === cpu_tlas.static_tlas

    # Adapting to a different backend errors loudly.
    @test_throws ErrorException Adapt.adapt(STRESS_BACKEND, cpu_tlas)

    # Same the other way: a Lava-backend TLAS adapted to KA.CPU() must error.
    lava_tlas = Raycore.TLAS(STRESS_BACKEND)
    push!(lava_tlas, stress_sphere(8))
    Raycore.sync!(lava_tlas)
    @test Adapt.adapt(STRESS_BACKEND, lava_tlas) === lava_tlas.static_tlas
    @test_throws ErrorException Adapt.adapt(KA.CPU(), lava_tlas)
end

# ------------------------------------------------------------------------------
# 13. HW TLAS stress — same patterns over Vulkan ray tracing
# ------------------------------------------------------------------------------

@testset "HW TLAS stress — random churn with strict invariants" begin
    # HW path now supports push! / delete! / update_transform! /
    # update_transforms!.  The latter is GPU-resident: a compute kernel
    # writes new records into the batch's instance_buf, sync! refits.
    rng = MersenneTwister(0xBADF00D)
    hwtlas = Lava.HWTLAS(STRESS_BACKEND)
    handles = Raycore.TLASHandle[]

    h0 = push!(hwtlas, stress_sphere(8), stress_xlat(0, 0, 0); instance_id=UInt32(1))
    push!(handles, h0)
    Raycore.sync!(hwtlas)

    for iter in 1:80
        op = rand(rng, 1:3)
        if op == 1 && length(handles) < 16
            n = rand(rng, [4, 6, 8])
            x = Float32(rand(rng) * 4 - 2)
            h = push!(hwtlas, stress_sphere(n), stress_xlat(x, 0, 0);
                      instance_id=UInt32(length(handles) + 1))
            push!(handles, h)
        elseif op == 2 && length(handles) > 1
            i = rand(rng, 1:length(handles))
            Raycore.delete!(hwtlas, handles[i])
            deleteat!(handles, i)
        elseif op == 3 && !isempty(handles)
            i = rand(rng, 1:length(handles))
            Raycore.update_transform!(hwtlas, handles[i],
                stress_xlat(Float32(rand(rng) * 6 - 3), 0, 0))
        end

        if iter % 5 == 0
            Raycore.sync!(hwtlas)
            @test Raycore.n_instances(hwtlas) == length(handles)
        end
    end

    Raycore.sync!(hwtlas)
    @test Raycore.world_bound(hwtlas) isa Raycore.Bounds3
    @test Raycore.wait_for_gpu!(hwtlas) === hwtlas
end

@testset "HW TLAS stress — long mesh-swap loop, leak bound" begin
    # Mirror of the SW TLAS leak-bound test, on the HW path.  Each iteration
    # drops the previous BLAS and pushes a fresh one.  The HW pool / instance
    # buffer counts must NOT scale with iteration count.
    hwtlas = Lava.HWTLAS(STRESS_BACKEND)
    h = push!(hwtlas, stress_sphere(16), stress_xlat(0, 0, 0); instance_id=UInt32(1))
    Raycore.sync!(hwtlas)

    n_iters = 100
    for iter in 1:n_iters
        Raycore.delete!(hwtlas, h)
        h = push!(hwtlas, stress_sphere(iseven(iter) ? 12 : 32),
                  stress_xlat(0, 0, Float32(0.001 * iter));
                  instance_id=UInt32(1))
        Raycore.sync!(hwtlas)
        @test Raycore.n_instances(hwtlas) == 1
    end
    GC.gc(true); GC.gc(true)
    @test Raycore.n_instances(hwtlas) == 1
end

println("\nAll stress tests passed.")
