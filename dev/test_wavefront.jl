using Revise
using Raycore
using Raycore: to_gpu
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics, Colors, LinearAlgebra
import Makie
using Makie: RGBf
import KernelAbstractions as KA
using ImageShow
using BenchmarkTools

# Load helper functions
include("raytracing-core.jl")
include("wavefront-renderer.jl")

bvh, ctx = example_scene()
# ibvh = Raycore.InstancedBVH(geom)
begin
    img = fill(RGBf(0, 0, 0), 400, 720)
    renderer = WavefrontRenderer(img, bvh, ctx)
    @btime render!(renderer)
    nothing
end
