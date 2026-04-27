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
- `Lava.HWTLAS` — hardware ray tracing via `VK_KHR_ray_tracing_pipeline`.

# Mutation API
- `push!(accel, mesh, transform)`: add geometry, return a `TLASHandle`.
- `delete!(accel, handle)`, `update_transform!(accel, handle, transform)`,
  `update_transform_at!(accel, handle, i, transform)`.

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
- `closest_hit(adapted, ray) -> (hit, tri, t, bary, instance_override)`
- `any_hit(adapted, ray) -> Bool`
- `world_bound(accel)`, `n_instances(accel)`, `n_geometries(accel)`.

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
export BLAS, BLASDescriptor, TLAS, InstanceDescriptor, BVHNode2, build_blas, build_tlas, INVALID_NODE
export build_triangle, is_degenerate_face

# TLAS (GPU two-level acceleration structure)
export TLASHandle, StaticTLAS, INVALID_HANDLE
export sync!, update!, n_total_instances

# BVH4 types (HIPRT-style 4-wide nodes)
export BVHNode4, BLAS4, TLAS4, build_blas4, closest_hit4, any_hit4

# Ray intersection functions
export AbstractAccel, AbstractAdaptedAccel
export closest_hit, any_hit, world_bound, trace_rays
export n_instances, n_geometries, wait_for_gpu!

# RT transport types (used by Lava.HWTLAS and consumers)
export RTRay, RTHitResult

# Stubs for Lava/Makie extensions
function trace_rays end
function push_instances! end
export push_instances!

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
export is_invalid, is_valid, with_index, n_slots, deref, get_static, to_tuple
export maybe_convert_field, store_texture
export free!

end
