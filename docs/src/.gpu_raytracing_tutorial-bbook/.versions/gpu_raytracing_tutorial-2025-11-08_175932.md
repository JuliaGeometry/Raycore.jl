# GPU Ray Tracing with Raycore

In this tutorial, we'll take the ray tracer from the previous tutorial and port it to the GPU using **KernelAbstractions.jl** and a GPU backend of choice (CUDA.jl, AMDGPU.jl, OpenCL.jl, OneApi.jl, or Metal.jl). We'll explore three different kernel implementations, each with different optimization strategies, and benchmark their performance against each other.

By the end, you'll understand how to write efficient GPU kernels for ray tracing and the tradeoffs between different approaches!

## Setup

```julia (editor=true, logging=true, output=true)
using Raycore, GeometryBasics, LinearAlgebra
using Colors, ImageShow
using WGLMakie
using KernelAbstractions
using BenchmarkTools
```
To run things on the GPU with KernelAbstractions, you need to chose the correct package for your GPU and set the array type we use from there on.

```julia (editor=true, logging=false, output=true)
#using CUDA; GArray = CuArray; # For NVIDIA GPUS
using AMDGPU; GArray = ROCArray; # for AMD GPUs
#using Metal; GArray = MtlArray; # for Apple hardware
#using oneAPI; GArray = oneArray; # for intel
# OpenCL with the pocl backend should work for most CPUs and some GPUs, but might not be as fast.
# using pocl_jll, OpenCL; GArray = CLArray;
#GArray = Array # For the tutorial to run on CI we just use the CPU
```
**Ready for GPU!** We have:

  * `Raycore` for fast ray-triangle intersections
  * `KernelAbstractions` for portable GPU kernels
  * `AMDGPU` for AMD GPU support
  * `BenchmarkTools` for performance comparison

## Part 1: Scene Setup (Same as CPU Tutorial)

Let's use the exact same scene as the CPU tutorial - the Makie cat with room geometry:

```julia (editor=true, logging=false, output=true)
# Load and prepare the cat model
include("raytracing-core.jl")
bvh, ctx = example_scene()
# We have a Makie extension for plotting the scene graph
f, ax, pl = plot(bvh; axis=(; show_axis=false))
f
```
```julia (editor=true, logging=false, output=true)
cam = cameracontrols(ax.scene)
cam.eyeposition[] = [0, 1.0, -5]
cam.lookat[] = [0, 0, 2]
cam.upvector[] = [0.0, 1, 0.0]
cam.fov[] = 45.0
```
## Part 5: GPU Kernel Version 1 - Basic Naive Approach

The simplest GPU kernel - one thread per pixel:

```julia (editor=true, logging=false, output=true)
import KernelAbstractions as KA

# Basic kernel: one thread per pixel, straightforward implementation
@kernel function raytrace_kernel_v1!(
    img, @Const(bvh), @Const(ctx),
    camera_pos, focal_length, aspect, sky_color, ::Val{NSamples}
) where {NSamples}
    # Get pixel coordinates
    idx = @index(Global, Linear)
    height, width = size(img)
    # Convert linear index to 2D coordinates
    x = ((idx - 1) % width) + 1
    y = ((idx - 1) ÷ width) + 1
    if x <= width && y <= height
        # Generate camera ray and do a calculate a simple light model
        color = Vec3f(0)
        for i in 1:NSamples
            color = color .+ sample_light(bvh, ctx, width, height, camera_pos, focal_length, aspect, x, y, sky_color)
        end
        @inbounds img[y, x] = to_rgb(color ./ NSamples)
    end
end
```
The `trace_gpu` function is a universal launcher that works with any of our kernels. It handles the backend-specific setup automatically using **KernelAbstractions.jl**:

```julia (editor=true, logging=false, output=true)
function trace_gpu(kernel, img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0,0.7f0,1.0f0),
        samples_per_pixel=4,
        ndrange=length(img), tilesize=nothing
    )
    height, width = size(img)
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    # KernelAbstractions automatically detects the backend (CPU/GPU) from the array type
    backend = KA.get_backend(img)

    # Create the kernel with or without tilesize (for workgroup configuration)
    kernel! = isnothing(tilesize) ? kernel(backend) : kernel(backend, tilesize)

    kernel!(img, bvh, ctx, camera_pos, focal_length, aspect, sky_color, Val(samples_per_pixel), ndrange=ndrange)

    # Ensure GPU computation completes before returning
    KA.synchronize(backend)
    return img
end
```
**Key KernelAbstractions.jl concepts:**

  * **Backend detection**: `get_backend(array)` automatically determines if we're using CPU, AMD GPU, NVIDIA GPU, etc.
  * **Kernel compilation**: `kernel(backend)` compiles the kernel for the specific backend
  * **Workgroup configuration**: Optional `tilesize` parameter controls thread organization
  * **Thread indexing**: Inside kernels, use `@index(Global, Linear)` or `@index(Global, Cartesian)` to get thread IDs
  * **Synchronization**: `synchronize(backend)` ensures all GPU work completes before continuing

Let's test kernel v1 on the CPU (yes, they always work with normal Arrays):

```julia (editor=true, logging=false, output=true)
img = fill(RGBf(0, 0, 0), 400, 720)
bench_kernel_cpu_v1 = @benchmark trace_gpu(raytrace_kernel_v1!, img, bvh, ctx)
img
```
To run things on the GPU, we simply convert the arrays to the GPU backend array type. `to_gpu` is a helper in Raycore to convert nested structs correctly for the kernel. It's not doing anything special, besides that struct of arrays need to be converted to device arrays and for pure arrays `GPUArray(array)` is enough.

```julia (editor=true, logging=false, output=true)
using Raycore: to_gpu
img = fill(RGBf(0, 0, 0), 400, 720)
img_gpu = GArray(img);
bvh_gpu = to_gpu(GArray, bvh);
ctx_gpu = to_gpu(GArray, ctx);
bench_kernel_v1 = @benchmark trace_gpu(raytrace_kernel_v1!, img_gpu, bvh_gpu, ctx_gpu)
# bring back to GPU to display image
Array(img_gpu)
```
**First GPU render!** This is the simplest approach - one thread per pixel with no optimization.

## Part 6: Optimized Kernel - Loop Unrolling

Loop overhead is significant on GPUs! Manually unrolling the sampling loop eliminates this overhead:

```julia (editor=true, logging=false, output=true)
# Optimized kernel: Unrolled sampling loop
@kernel function raytrace_kernel_unrolled!(
        img, @Const(bvh), @Const(ctx),
        camera_pos, focal_length, aspect, sky_color, ::Val{NSamples}
    ) where {NSamples}
    idx = @index(Global, Linear)
    height, width = size(img)
    x = ((idx - 1) % width) + 1
    y = ((idx - 1) ÷ width) + 1
    if x <= width && y <= height
        # ntuple with compile-time constant for unrolling
        samples = ntuple(NSamples) do i
            sample_light(bvh, ctx, width, height,
                camera_pos, focal_length, aspect,
                x, y, sky_color
            )
        end
        color = mean(samples)
        @inbounds img[y, x] = to_rgb(color)
    end
end

bench_kernel_unrolled = @benchmark trace_gpu(raytrace_kernel_unrolled!, img_gpu, bvh_gpu, ctx_gpu)
Array(img_gpu)
```
  * This eliminates branch overhead from loop conditions
  * Reduces register pressure
  * Better instruction-level parallelism
  * **1.39x faster than baseline!**

## Part 7: Tiled Kernel with Optimized Tile Size

The tile size dramatically affects performance. Let's use the optimal size discovered through benchmarking:

```julia (editor=true, logging=false, output=true)
# Tiled kernel with optimized tile size
@kernel function raytrace_kernel_tiled!(
    img, bvh, ctx,
    camera_pos, focal_length, aspect, sky_color, ::Val{NSamples}
) where {NSamples}
    # Get tile and local coordinates
    _tile_xy = @index(Group, Cartesian)
    _local_xy = @index(Local, Cartesian)
    _groupsize = @groupsize()

    # Direct tuple unpacking is faster than Vec construction
    tile_x, tile_y = Tuple(_tile_xy)
    local_x, local_y = Tuple(_local_xy)
    group_w, group_h = _groupsize

    # Compute global pixel coordinates
    x = (tile_x - 1) * group_w + local_x
    y = (tile_y - 1) * group_h + local_y

    height, width = size(img)
    if x <= width && y <= height
        samples = ntuple(NSamples) do i
            sample_light(bvh, ctx, width, height,
                camera_pos, focal_length, aspect,
                x, y, sky_color
            )
        end
        color = mean(samples)
        @inbounds img[y, x] = to_rgb(color)
    end
end
bench_kernel_tiled_32_16 = @benchmark trace_gpu(
    $raytrace_kernel_tiled!, $img_gpu, $bvh_gpu, $ctx_gpu;
    ndrange=size($img_gpu), tilesize=(32,16))

# Benchmark two more important tile sizes for comparison
bench_kernel_tiled_32_32 = @benchmark trace_gpu(
    $raytrace_kernel_tiled!, $img_gpu, $bvh_gpu, $ctx_gpu;
    ndrange=size($img_gpu), tilesize=(32,32))

bench_kernel_tiled_8_8 = @benchmark trace_gpu(
    $raytrace_kernel_tiled!, $img_gpu, $bvh_gpu, $ctx_gpu;
    ndrange=size($img_gpu), tilesize=(8,8))

# Use optimal tile size: (32, 16) - discovered through benchmarking!
Array(img_gpu)
```
**Tile size matters!** With `(32, 16)` tiles, this kernel is **1.22x faster** than baseline. With poor tile sizes like `(8, 8)`, it can be **2.5x slower**!

## Part 8: Wavefront Path Tracing

The wavefront approach reorganizes ray tracing to minimize thread divergence by grouping similar work together. Instead of each thread handling an entire pixel's path, we separate the work into stages:

```julia (editor=true, logging=false, output=true)
include("wavefront-renderer.jl")
```
Let's benchmark the wavefront renderer on both CPU and GPU:

```julia (editor=true, logging=false, output=true)
# CPU benchmark
renderer_cpu = WavefrontRenderer(img, bvh, ctx)
bench_wavefront_cpu = @benchmark render!($renderer_cpu)

# GPU benchmark
renderer_gpu = to_gpu(GArray, renderer_cpu)
bench_wavefront_gpu = @benchmark render!($renderer_gpu)

# Display result
Array(renderer_gpu.framebuffer)
```
**Wavefront benefits:**

  * Reduces thread divergence by grouping similar work
  * Better memory access patterns
  * Scales well with scene complexity
  * Enables advanced features like path tracing

## Part 9: Comprehensive Performance Benchmarks

Now let's compare all kernels including the wavefront renderer:

```julia (editor=true, logging=false, output=true)
benchmarks = [
    bench_kernel_v1, bench_kernel_cpu_v1, bench_kernel_unrolled,
    bench_kernel_tiled_8_8, bench_kernel_tiled_32_16, bench_kernel_tiled_32_32,
    bench_wavefront_cpu, bench_wavefront_gpu
]
labels = [
    "Baseline", "Baseline (cpu)", "Unrolled",
    "Tiled\n(8×8)", "Tiled\n(32×16)", "Tiled\n(32×32)",
    "Wavefront\n(cpu)", "Wavefront\n(gpu)"
]

fig, times, speedups = plot_kernel_benchmarks(benchmarks, labels)
fig
```
### Next Steps

  * Add **adaptive sampling** (more samples only where needed)
  * Explore **shared memory** optimizations for BVH traversal
  * Implement **streaming multisampling** across frames
  * Try **persistent threads** with dynamic work distribution

Happy GPU ray tracing!

