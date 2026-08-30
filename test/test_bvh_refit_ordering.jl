# ==============================================================================
# BVH refit: the parent/child handoff must be ordered
# ==============================================================================
#
# `refit_aabbs_kernel!` is Karras' bottom-up refit. Each of a node's two
# children atomically increments the parent's arrival counter; the second
# arriver reads BOTH child nodes and stores their boxes into the parent. That
# is a release/acquire handoff — the first arriver's store to `nodes[child]`
# has to be visible to the thread that later sees the counter reach 2 — and an
# atomic RMW on the counter does not by itself supply the ordering. Raycore
# states it explicitly via `bvh_publish_fence`, which each GPU backend
# implements with its own device fence.
#
# What it looked like when the fence was missing: a BVH that is *mostly* right.
# On a 4,032-triangle mesh, 33 of 8,062 interior nodes came out with boxes too
# small (children's extents read back as zero), two consecutive builds of the
# same mesh disagreed, and a path tracer lost scattered light where rays missed
# geometry that was really there. Nothing threw.
#
# The assertions below are deliberately backend-independent. The invariant is a
# property of the tree, so it is checked on whatever backend the matrix
# selected rather than against a reference image or a reference driver.

using Test
using Raycore
using GeometryBasics
using StaticArrays
using KernelAbstractions
const KA = KernelAbstractions

# Enough triangles, scattered enough, that the tree is deep and interior nodes
# genuinely have their two subtrees completed by different threads. A handful
# of triangles will not race no matter how broken the ordering is.
function _refit_test_mesh(n::Int)
    tris = Raycore.Triangle{Nothing}[]
    state = UInt32(0x12345678)
    rnd() = (state = (1103515245 * state + 12345) % UInt32(0x80000000);
             Float32(state) / Float32(0x80000000))
    for _ in 1:n
        c = Point3f(20 * (rnd() - 0.5f0), 20 * (rnd() - 0.5f0), 20 * (rnd() - 0.5f0))
        v1 = c + Point3f(rnd(), rnd(), rnd())
        v2 = c + Point3f(rnd(), rnd(), rnd())
        v3 = c + Point3f(rnd(), rnd(), rnd())
        push!(tris, Raycore.Triangle(
            SVector(v1, v2, v3),
            SVector(Normal3f(0, 0, 1), Normal3f(0, 0, 1), Normal3f(0, 0, 1)),
            SVector(Vec3f(0), Vec3f(0), Vec3f(0)),
            SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)),
            nothing))
    end
    return tris
end

# Every interior node stores its two children's boxes. For a child that is
# itself interior, that box must be exactly the union of the child's own two
# boxes — which is what the refit computes, so it holds for a correctly
# published tree and fails for a node whose sibling read went stale.
function _unclosed_nodes(nodes, n_prims::Integer)
    bad = 0
    for i in 1:(n_prims - 1)
        node = nodes[i]
        for (child, bmin, bmax) in ((node.child0, node.aabb0_min, node.aabb0_max),
                                    (node.child1, node.aabb1_min, node.aabb1_max))
            child < n_prims || continue          # leaf child: box is the triangle
            ch = nodes[child]
            umin = min.(ch.aabb0_min, ch.aabb1_min)
            umax = max.(ch.aabb0_max, ch.aabb1_max)
            (bmin == umin && bmax == umax) || (bad += 1)
        end
    end
    return bad
end

@testset "BVH refit ordering" begin
    backend = test_backend()
    tris = _refit_test_mesh(4032)
    n = length(tris)

    ref = build_blas(tris)                         # CPU: one thread per workitem, no race
    ref_nodes = Array(ref.nodes)
    @test _unclosed_nodes(ref_nodes, n) == 0

    dev_tris = KA.allocate(backend, eltype(tris), n)
    copyto!(dev_tris, tris)
    KA.synchronize(backend)

    a = Array(build_blas(dev_tris).nodes)
    b = Array(build_blas(dev_tris).nodes)

    # The invariant, on the backend under test. This is the assertion that
    # fails when the publishing fence is missing.
    @test _unclosed_nodes(a, n) == 0

    # A race does not have to break the invariant every run, so also pin the
    # two properties it always breaks: the build agrees with the serial one,
    # and it agrees with itself.
    @test a == ref_nodes
    @test a == b
end
