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
include("hw-accel.jl")

# Macros
export @_inbounds

# Core types
export Ray, RayDifferentials, Triangle, Bounds3, Normal3f

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

# Hardware RT types and stubs
export HWTLAS, HWAdaptedAccel, RTRay, RTHitResult
export supports_indirect_dispatch, indirect_ndrange
export build_hw_blas, build_hw_tlas, trace_closest_hits!, trace_closest_hits_indirect!
export batch_trace_indirect, set_custom_anyhit!, mat4_to_transform_matrix
export rt_primitive_id, rt_instance_custom_index, rt_launch_id_x, rt_global_invocation_id_x
export rt_ignore_intersection, rt_terminate_ray
export rt_payload_store!, rt_payload_load, rt_trace_ray!

# Stub for Makie extension
function trace_rays end

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
export maybe_convert_field, store_texture, rebuild_static!
export free!

end
