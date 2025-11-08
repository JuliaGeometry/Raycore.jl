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

# Load helper functions
include("raytracing-core.jl")
include("wavefront-renderer.jl")

begin
    bvh, ctx = example_scene()
    img = fill(RGBf(0, 0, 0), 1024, 2048)
    renderer = WavefrontRenderer(
        img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5),
        fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
        samples_per_pixel=4
    )
    Array(@time render!(renderer))
end
using FileIO
save("wavefront.png", map(col -> mapc(c -> clamp(c, 0f0, 1f0), col), renderer.framebuffer))

using AMDGPU
amd_renderer = to_gpu(ROCArray, renderer);
Array(@time render!(amd_renderer))

using pocl_jll, OpenCL
amd_renderer = to_gpu(CLArray, renderer);
Array(@time render!(amd_renderer))
