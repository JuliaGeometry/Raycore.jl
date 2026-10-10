module Raycore

using GeometryBasics
using LinearAlgebra
using StaticArrays
using KernelAbstractions
import GeometryBasics as GB
using Statistics
using Adapt
using GPUArraysCore: @allowscalar

abstract type AbstractRay end
abstract type Primitive end
"""
    AbstractAccel

Mutable acceleration structure for ray/geometry intersection queries.

Concrete implementations:
- `Raycore.TLAS` — software BVH/TLAS, runs on any KernelAbstractions backend.
- `Mantle.HWTLAS` — the GPU's hardware acceleration structure, on Vulkan and Metal.

# Mutation API
- `push!(accel, mesh, transform)`: add geometry, return a `TLASHandle`.
- `delete!(accel, handle)`, `update_transform!(accel, handle, transform)`,
  `update_transforms!(accel, handle, transforms)`.

# Lifecycle
- `sync!(accel)` — sole owner of `accel.static_tlas`. Rebuilds in place
  where possible; reassigns when a buffer had to grow. No-op on a clean
  accel. Does NOT block the CPU on a GPU fence; backend-internal timeline
  tracking handles the "still in flight" case.
- `Adapt.adapt(backend, accel) === accel.static_tlas` between `sync!`s.
  Consumers re-read `accel.static_tlas` (or call `Adapt.adapt`) **per
  dispatch**. Caching the adapted form across mutations is a contract
  violation.

# Query
- `closest_hit(adapted, ray, mask = 0xff) -> (hit, tri, t, bary, instance_override)`
- `any_hit(adapted, ray, mask = 0xff) -> (hit, tri, t, bary, instance_override)` — the SAME
  shape as `closest_hit`, not a `Bool`. Only the traversal differs: it stops at
  the first accepted hit, so `t`/`tri` are *an* intersection, not the nearest.
  This said `-> Bool`, and all three implementations (software `StaticTLAS`,
  Vulkan, Metal) return the tuple — a backend written to the doc would break
  every caller, and a caller written to it gets a device-side type error.
- `mask` is the ray's cull mask: an instance is seen only when its mask
  (`instance_mask` at `push!`, or the record's `mask`) shares a bit with it. Low
  8 bits, on every implementation.
- `world_bound(accel)`, `n_instances(accel)`, `n_geometries(accel)`.

# Instances a kernel writes (hardware structures)
- `instance_buffer(accel, handle)` — the batch's [`InstanceRecord`](@ref)s, a
  device array a kernel may write; `refit!(accel)` commits what it wrote.

# Flush
- `wait_for_gpu!(accel)` — block CPU until all pending GPU work on this
  accel's queue has completed. Convenience for tear-down and benchmark
  isolation. Not part of the hot path.
"""
abstract type AbstractAccel end
abstract type AbstractAdaptedAccel end
const Maybe{T} = Union{T,Nothing}

GB.@fixed_vector Normal = StaticVector
const Normal3f = Normal{3, Float32}

const DO_ASSERTS = false
macro real_assert(expr, msg="")
    if DO_ASSERTS
        esc(:(@assert $expr $msg))
    else
        return :()
    end
end

const ENABLE_INBOUNDS = true

macro _inbounds(ex)
    if ENABLE_INBOUNDS
        esc(:(@inbounds $ex))
    else
        esc(ex)
    end
end

include("ray.jl")
include("bounds.jl")
include("transformations.jl")
include("math.jl")
include("triangle_mesh.jl")
include("instanced-bvh.jl")
include("instanced-bvh-kernels.jl")
include("bvh4.jl")
include("kernel-abstractions.jl")
include("kernels.jl")
include("collision.jl")
include("soa.jl")
include("multitypeset.jl")
include("unrolled.jl")
include("rt_transport.jl")

# Macros
export @_inbounds

# Core types
export Ray, RayDifferentials, Triangle, Bounds3, Normal3f, empty_triangle

# Instanced BVH types
export BLAS, BLASDescriptor, TLAS, InstanceDescriptor, InstanceRecord, BVHNode2, build_blas, build_tlas, INVALID_NODE
export build_triangle, is_degenerate_face

# TLAS (GPU two-level acceleration structure)
export TLASHandle, StaticTLAS, INVALID_HANDLE
export sync!, update!, n_total_instances, set_visible!

# BVH4 types (HIPRT-style 4-wide nodes)
export BVHNode4, BLAS4, TLAS4, build_blas4, closest_hit4, any_hit4

# Ray intersection functions
export AbstractAccel, AbstractAdaptedAccel
export closest_hit, any_hit, world_bound, trace_rays
export n_instances, n_geometries, wait_for_gpu!

# RT transport types (used by Mantle.HWTLAS and consumers)
export RTRay, RTHitResult

# Stubs for Lava/Makie extensions
function trace_rays end

"""
    instance_buffer(tlas, handle::TLASHandle) -> device vector of InstanceRecord

The device array holding the [`InstanceRecord`](@ref)s of the batch `handle`
names, in the order of its instances: the array the batch was pushed with when
it was pushed with one, else the one `push!` made. A kernel may write it — new
transforms, ids or masks — and [`refit!`](@ref)`(tlas)` is what makes the
structure take what was written.

The array can be longer than the batch: only its first `n_instances` records
are instances. Throws an `ArgumentError` for a handle that names no batch.
"""
function instance_buffer end

export instance_buffer

"""
    refit!(tlas) -> tlas
    refit!(blas, vertices) -> blas

Bring an acceleration structure up to date with geometry that moved, keeping
its topology.

`refit!(tlas)` takes every instance record as it is now — a kernel may have
written them through [`instance_buffer`](@ref) — and refits the structure over
them in place. A structure that cannot be refit (its batches changed since it
was built) is rebuilt instead. The adapted form a kernel traces stays the same
object across a refit, so a plan holding it sees the new instances.

`refit!(blas, vertices)` moves the vertices of a bottom-level structure built
with `allow_update = true` and refits it in place: same vertex count, same
triangles. A structure built without `allow_update`, or a different vertex
count, is an `ArgumentError`. A top-level structure stores the bounds of what it
instances, so `refit!` every TLAS that instances the BLAS afterwards.

A refit keeps the tree the structure was built with, so traversal slows as the
geometry moves away from that pose; rebuild a structure that moves far.

Not exported: `Mantle` exports a `refit!` of its own, for plans, and the two
would make the bare name ambiguous wherever both packages are `using`ed.
"""
function refit! end

public refit!

# Math utilities
export reflect

# Collision detection
export ContactPair, CollisionResult, collide_instances, collide_instances_any

# Analysis functions
export get_centroid, get_illumination, view_factors

# SoA utilities
export @get, @set, similar_soa

# GPU-safe unrolled iteration
export FastClosure, for_unrolled, map_unrolled, reduce_unrolled, sum_unrolled, getindex_unrolled

# MultiTypeSet - type-stable heterogeneous collections
export SetKey, MultiTypeSet, StaticMultiTypeSet, TextureRef
export is_invalid, is_valid, with_index, with_texture, n_slots, deref, get_static, to_tuple
export maybe_convert_field, store_texture
export free!

end
