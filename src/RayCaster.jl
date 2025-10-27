module RayCaster

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
include("shapes/Shape.jl")
include("bvh.jl")
include("kernel-abstractions.jl")
include("kernels.jl")
include("ray_intersection_session.jl")

export RayIntersectionSession, hit_points, hit_distances, hit_count, miss_count

end
