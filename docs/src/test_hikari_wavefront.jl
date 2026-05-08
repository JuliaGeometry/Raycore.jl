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

# Load the Hikari wavefront renderer
# Create the example scene with Hikari materials
material_scene, lights = hikari_example_scene()

# Create and render
begin
    img = fill(RGBf(0, 0, 0), 400, 720)
    renderer = HikariWavefrontRenderer(img, material_scene, lights)
    render!(renderer)
    nothing
end
img
