module Raycore

using GeometryBasics
using LinearAlgebra
using StaticArrays
using KernelAbstractions
import GeometryBasics as GB
using Statistics

abstract type AbstractRay end
abstract type AbstractShape end
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

# Core types
export Ray, RayDifferentials, Triangle, TriangleMesh, AccelPrimitive, BVH, Bounds3, Normal3f

# Instanced BVH types
export BLAS, TLAS, InstanceDescriptor, BVHNode2, build_blas, build_tlas, INVALID_NODE
export Instance, InstanceHandle, find_instances, add_instance!, remove_instance!, rebuild_tlas!
export update_transform!, update_transforms!, n_instances, n_geometries

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

end
