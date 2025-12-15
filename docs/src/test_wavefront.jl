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
renderer.framebuffer
renderer_instanced.framebuffer
begin
    img = fill(RGBf(0, 0, 0), 400, 720)
    renderer_instanced = WavefrontRenderer(
        img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5),
        fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
        samples_per_pixel=4
    )
    @btime render!(renderer_instanced)
    # on windows + ryzen 395 max
    # 381.034 ms (1200456 allocations: 90.13 MiB)

    nothing
end
using ImageShow

using FileIO
save("wavefront.png", map(col -> mapc(c -> clamp(c, 0f0, 1f0), col), renderer.framebuffer))

using AMDGPU
amd_renderer = to_gpu(ROCArray, renderer);
Array(@btime render!(amd_renderer))
# 36ms on windows + amd 8060s

using pocl_jll, OpenCL
amd_renderer = to_gpu(CLArray, renderer);
Array(@time render!(amd_renderer))
r = Raycore.Ray(o=Point3f(0, 0, 0), d=Vec3f(0, 0, 1), t_max=0.0f0)

@code_warntype any_hit(ibvh, r)

function test(bvh)
    meshes = getfield(bvh, :meshes)
end

typeof(test(ibvh))
