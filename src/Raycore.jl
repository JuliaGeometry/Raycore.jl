module Raycore

using GeometryBasics
using LinearAlgebra
using StaticArrays
using Atomix
using KernelAbstractions
import GeometryBasics as GB
using Statistics

abstract type AbstractRay end
abstract type AbstractShape end
abstract type Primitive end
abstract type Material end
const Maybe{T} = Union{T,Nothing}
const Radiance = UInt8(1)
const Importance = UInt8(2)

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
include("kernel-abstractions.jl")
include("kernels.jl")
include("ray_intersection_session.jl")

# Core types
export Ray, RayDifferentials, Triangle, TriangleMesh, BVHAccel, Bounds3, Normal3f

# Ray intersection functions
export closest_hit, any_hit, world_bound

# Math utilities
export reflect

# Analysis functions (key features)
export get_centroid, get_illumination, view_factors

# Ray intersection session
export RayIntersectionSession, hit_points, hit_distances, hit_count, miss_count

end
