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
include("bvh.jl")
include("instanced-bvh.jl")
include("instanced-bvh-kernels.jl")
include("bvh4.jl")
include("kernel-abstractions.jl")
include("kernels.jl")
include("ray_intersection_session.jl")
include("soa.jl")
include("heterovec.jl")
include("unrolled.jl")

# Macros
export @_inbounds

# Core types
export Ray, RayDifferentials, Triangle, TriangleMesh, AccelPrimitive, BVH, Bounds3, Normal3f

# Instanced BVH types
export BLAS, TLAS, InstanceDescriptor, BVHNode2, build_blas, build_tlas, INVALID_NODE
export Instance, n_instances, n_geometries, build_triangle, is_degenerate_face

# TLASBuilder (new MultiTypeSet-style API)
export TLASBuilder, TLASHandle, StaticTLAS, INVALID_HANDLE
export sync!, update_instance!, update!, n_total_instances

# BVH4 types (HIPRT-style 4-wide nodes)
export BVHNode4, BLAS4, TLAS4, build_blas4, closest_hit4, any_hit4

# Ray intersection functions
export closest_hit, any_hit, world_bound

# Math utilities
export reflect

# Analysis functions
export get_centroid, get_illumination, view_factors

# Ray intersection session
export RayIntersectionSession, hit_points, hit_distances, hit_count, miss_count

# SoA utilities
export @get, @set, similar_soa

# GPU-safe unrolled iteration
export FastClosure, for_unrolled, map_unrolled, reduce_unrolled, sum_unrolled, getindex_unrolled

# HeterogeneousVector for type-stable heterogeneous collections
export SetKey, MultiTypeSet, StaticMultiTypeSet, TextureRef
export is_invalid, is_valid, with_index, n_slots, deref, get_static, to_tuple
export maybe_convert_field, store_texture, rebuild_static!
export free!

end
