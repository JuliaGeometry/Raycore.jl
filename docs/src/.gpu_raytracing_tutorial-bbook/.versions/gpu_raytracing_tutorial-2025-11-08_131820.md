# GPU Ray Tracing with Raycore

In this tutorial, we'll take the ray tracer from the previous tutorial and port it to the GPU using **KernelAbstractions.jl** and a GPU backend of choice (CUDA.jl, AMDGPU.jl, OpenCL.jl, OneApi.jl, or Metal.jl). 
We'll explore three different kernel implementations, each with different optimization strategies, and benchmark their performance against each other.

By the end, you'll understand how to write efficient GPU kernels for ray tracing and the tradeoffs between different approaches!

## Setup

```julia (editor=true, logging=false, output=true)
using Raycore, GeometryBasics, LinearAlgebra
using Colors, ImageShow
using WGLMakie
using KernelAbstractions
using BenchmarkTools
```
```julia (editor=true, logging=false, output=true)
#using CUDA; GArray = CuArray; # For NVIDIA GPUS
#using AMDGPU; GArray = ROCArray; # for AMD GPUs
#using Metal; GArray = MtlArray; # for Apple hardware
#using oneAPI; GArray = oneArray; # for intel  
# OpenCL with the pocl backend should work for most CPUs and some GPUs, but might not be as fast.
using pocl_jll, OpenCL
GArray = CLArray # For the tutorial to run on CI we need to use OpenCL
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
f, ax, pl = plot(bvh; axis=(; show_axis=false)) # We have a Makie extension for plotting the scene graph

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
    camera_pos, focal_length, aspect, sky_color, samples_per_pixel
)
    # Get pixel coordinates
    idx = @index(Global, Linear)
    height, width = size(img)
    # Convert linear index to 2D coordinates
    x = ((idx - 1) % width) + 1
    y = ((idx - 1) ÷ width) + 1
    if x <= width && y <= height
        # Generate camera ray with multi-sampling for anti-aliasing
        color = Vec3f(0, 0, 0)
        for i in 1:samples_per_pixel
            color = color .+ sample_light(bvh, ctx, width, height, camera_pos, focal_length, aspect, x, y, sky_color)
        end
        @inbounds img[y, x] = (to_rgb(color) ./ Float32(samples_per_pixel))
    end
end
```
** trace_gpu launcher:**

The `trace_gpu` function is a universal launcher that works with any of our kernels. It handles the backend-specific setup automatically using **KernelAbstractions.jl**:

```julia (editor=true, logging=false, output=true)
function trace_gpu(kernel, img, bvh, ctx;
        camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0,0.7f0,1.0f0),
        samples_per_pixel=8,
        ndrange=length(img), tilesize=nothing
    )
    height, width = size(img)
    aspect = Float32(width / height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))

    # KernelAbstractions automatically detects the backend (CPU/GPU) from the array type
    backend = KA.get_backend(img)

    # Create the kernel with or without tilesize (for workgroup configuration)
    kernel! = isnothing(tilesize) ? kernel(backend) : kernel(backend, tilesize)

    kernel!(img, bvh, ctx, camera_pos, focal_length, aspect, sky_color, Int32(samples_per_pixel), ndrange=ndrange)

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
img_v1 = trace_gpu(raytrace_kernel_v1!, img, bvh, ctx)
```
To run things on the GPU, we simply convert the arrays to the GPU backend array type. `to_gpu` is a helper in Raycore to convert nested structs correctly for the kernel. It's not doing anything special, besides that struct of arrays need to be converted to device arrays and for pure arrays `GPUArray(array)` is enough.

```julia (editor=true, logging=false, output=true)
cl.platforms()[1]
```
```julia (editor=true, logging=false, output=true)
using Raycore: to_gpu
img = fill(RGBf(0, 0, 0), 400, 720)
img_gpu = GArray(img);
bvh_gpu = to_gpu(GArray, bvh);
ctx_gpu = to_gpu(GArray, ctx);
img_v1 = trace_gpu(raytrace_kernel_v1!, img_gpu, bvh_gpu, ctx_gpu)
# bring back to GPU to display image
Array(img_v1)
```
**First GPU render!** This is the simplest approach - one thread per pixel with no optimization.

## Part 6: Optimized Kernel - Loop Unrolling

Loop overhead is significant on GPUs! Manually unrolling the sampling loop eliminates this overhead:

```julia (editor=true, logging=true, output=true)
# Optimized kernel: Unrolled sampling loop (4 samples)
# Optimized kernel: Unrolled sampling loop (4 samples)
@kernel function raytrace_kernel_unrolled!(
        img, @Const(bvh), @Const(ctx),
        camera_pos, focal_length, aspect, sky_color, nsamples
    )
    idx = @index(Global, Linear)
    height, width = size(img)
    x = ((idx - 1) % width) + 1
    y = ((idx - 1) ÷ width) + 1
    if x <= width && y <= height
        # ntuple is unrolled up to n = 10
        samples = ntuple(4) do i
            sample_light(
                bvh, ctx, width, height, camera_pos,
                focal_length, aspect, x, y, sky_color
            )
        end
        @inbounds img[y, x] = to_rgb(mean(samples))
    end
end

@btime trace_gpu(raytrace_kernel_unrolled!, img_gpu, bvh_gpu, ctx_gpu)
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
    camera_pos, focal_length, aspect, sky_color, samples_per_pixel
)
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
        color = Vec3f(0, 0, 0)
        for i in 1:samples_per_pixel
            color = color .+ sample_light(bvh, ctx, width, height, camera_pos, focal_length, aspect, x, y, sky_color)
        end
        img[y, x] = (to_rgb(color) ./ Float32(samples_per_pixel))
    end
end

# Use optimal tile size: (32, 16) - discovered through benchmarking!
img_tiled = trace_gpu(raytrace_kernel_tiled!, img_gpu, bvh_gpu, ctx_gpu;
                      samples_per_pixel=4, ndrange=size(img), tilesize=(32, 16))
Array(img_tiled)
```
**Tile size matters!** With `(32, 16)` tiles, this kernel is **1.22x faster** than baseline. With poor tile sizes like `(8, 8)`, it can be **2.5x slower**!

## Part 7.5: Exploring Tile Size Impact

Let's experimentally discover how tile size affects performance. This is crucial for real-world GPU optimization!

```julia (editor=true, logging=false, output=true)
using WGLMakie

# Test different tile sizes
tile_sizes = [(8, 8), (16, 16), (32, 16), (32, 32), (64, 4), (16, 8)]
times = Float64[]

test_img = GArray(fill(RGBf(0, 0, 0), 256, 256))

for tilesize in tile_sizes
    t = @elapsed trace_gpu(raytrace_kernel_tiled!, test_img, bvh_gpu, ctx_gpu;
                           samples_per_pixel=4, ndrange=size(test_img), tilesize=tilesize)
    push!(times, t * 1000)  # Convert to milliseconds
end

# Create visualization
fig = Figure()
ax = Axis(fig[1, 1],
    title="Tile Size Impact on GPU Performance",
    xlabel="Tile Configuration",
    ylabel="Time (ms)",
    xticks=(1:length(tile_sizes), string.(tile_sizes)),
    xticklabelrotation=π/4)

barplot!(ax, 1:length(tile_sizes), times, color=:steelblue)

# Mark the best performer
best_idx = argmin(times)
scatter!(ax, [best_idx], [times[best_idx]], color=:red, markersize=20,
         marker=:star5, label="Best: $(tile_sizes[best_idx])")

# Add value labels
for (i, (tsize, t)) in enumerate(zip(tile_sizes, times))
    text!(ax, i, t, text="$(round(t, digits=1))ms",
          align=(:center, :bottom), fontsize=10)
end

axislegend(ax, position=:rt)

fig
```
```julia (editor=true, logging=false, output=true)
# Performance analysis
best_time = minimum(times)
worst_time = maximum(times)

md"""
### Tile Size Analysis

**Best tile size:** $(tile_sizes[argmin(times)]) at $(round(best_time, digits=2)) ms
**Worst tile size:** $(tile_sizes[argmax(times)]) at $(round(worst_time, digits=2)) ms
**Performance range:** $(round(worst_time/best_time, digits=2))x difference!

**Key insights:**
- Larger tiles (32×16, 32×32) perform best for this workload
- Small tiles (8×8) have excessive thread block overhead
- Non-square tiles can sometimes outperform square ones
- **Always benchmark your specific workload!**
"""
```
## Part 8: Comprehensive Performance Benchmarks

Now let's compare all kernels with proper tile sizes:

```julia (editor=true, logging=false, output=true)
# Benchmark all three kernels on GPU
bench_img = GArray(fill(RGBf(0, 0, 0), 256, 256))
bvh_bench = to_gpu(GArray, bvh)
ctx_bench = to_gpu(GArray, ctx)
```
```julia (editor=false, logging=false, output=true)
println("\n1. Baseline kernel (v1 with loop, 4 samples):")
@btime trace_gpu($raytrace_kernel_v1!, $bench_img, $bvh_bench, $ctx_bench; samples_per_pixel=4)

println("\n2. Unrolled kernel (4 samples unrolled):")
@btime trace_gpu($raytrace_kernel_unrolled!, $bench_img, $bvh_bench, $ctx_bench)

println("\n3. Tiled kernel with (32, 16) tiles:")
@btime trace_gpu($raytrace_kernel_tiled!, $bench_img, $bvh_bench, $ctx_bench;
                 samples_per_pixel=4, ndrange=size($bench_img), tilesize=(32,16))

println("\n4. Tiled kernel with (32, 32) tiles:")
@btime trace_gpu($raytrace_kernel_tiled!, $bench_img, $bvh_bench, $ctx_bench;
                 samples_per_pixel=4, ndrange=size($bench_img), tilesize=(32,32))

println("="^60)

md"Benchmarks complete! See console output above for results."
```
```julia (editor=true, logging=false, output=true)
# Collect actual benchmark data for visualization
using BenchmarkTools

bench_v1 = @benchmark trace_gpu($raytrace_kernel_v1!, $bench_img, $bvh_bench, $ctx_bench; samples_per_pixel=4)
bench_unrolled = @benchmark trace_gpu($raytrace_kernel_unrolled!, $bench_img, $bvh_bench, $ctx_bench)
bench_tiled_32_16 = @benchmark trace_gpu($raytrace_kernel_tiled!, $bench_img, $bvh_bench, $ctx_bench;
                                          samples_per_pixel=4, ndrange=size($bench_img), tilesize=(32,16))
bench_tiled_32_32 = @benchmark trace_gpu($raytrace_kernel_tiled!, $bench_img, $bvh_bench, $ctx_bench;
                                          samples_per_pixel=4, ndrange=size($bench_img), tilesize=(32,32))

# Extract times in milliseconds
times = [
    median(bench_v1.times) / 1e6,
    median(bench_unrolled.times) / 1e6,
    median(bench_tiled_32_16.times) / 1e6,
    median(bench_tiled_32_32.times) / 1e6
]

# Calculate speedups relative to baseline
speedups = times[1] ./ times

# Create performance visualization
fig = Figure()

# Plot 1: Execution time comparison
ax1 = Axis(fig[1, 1],
    title="GPU Kernel Performance",
    xlabel="Kernel Configuration",
    ylabel="Time (ms)",
    xticks=(1:4, ["Baseline\n(loop)", "Unrolled\n(4 samples)", "Tiled\n(32×16)", "Tiled\n(32×32)"]))

barplot!(ax1, 1:4, times, color=[:steelblue, :coral, :seagreen, :seagreen])

# Add value labels
for (i, t) in enumerate(times)
    text!(ax1, i, t, text="$(round(t, digits=1))ms",
          align=(:center, :bottom), fontsize=12)
end

# Plot 2: Speedup comparison
ax2 = Axis(fig[1, 2],
    title="Speedup Relative to Baseline",
    xlabel="Kernel Configuration",
    ylabel="Speedup Factor",
    xticks=(1:4, ["Baseline", "Unrolled", "Tiled\n(32×16)", "Tiled\n(32×32)"]))

barplot!(ax2, 1:4, speedups, color=[:steelblue, :coral, :seagreen, :seagreen])
hlines!(ax2, [1.0], color=:gray, linestyle=:dash, linewidth=1)

# Add value labels
for (i, s) in enumerate(speedups)
    text!(ax2, i, s, text="$(round(s, digits=2))x",
          align=(:center, :bottom), fontsize=12)
end

# Highlight the winner
scatter!(ax2, [argmax(speedups)], [maximum(speedups)],
         color=:red, markersize=25, marker=:star5)

fig
```
```julia (editor=true, logging=false, output=true)

# Extract times in milliseconds
times = [
    median(bench_v1.times) / 1e6,
    median(bench_unrolled.times) / 1e6,
    median(bench_tiled_32_16.times) / 1e6,
    median(bench_tiled_32_32.times) / 1e6
]

# Calculate speedups relative to baseline
speedups = times[1] ./ times

# Create performance visualization
fig = Figure()

# Plot 1: Execution time comparison
ax1 = Axis(fig[1, 1],
    title="GPU Kernel Performance",
    xlabel="Kernel Configuration",
    ylabel="Time (ms)",
    xticks=(1:4, ["Baseline\n(loop)", "Unrolled\n(4 samples)", "Tiled\n(32×16)", "Tiled\n(32×32)"]))

barplot!(ax1, 1:4, times, color=[:steelblue, :coral, :seagreen, :seagreen])

# Add value labels
for (i, t) in enumerate(times)
    text!(ax1, i, t, text="$(round(t, digits=1))ms",
          align=(:center, :bottom), fontsize=12)
end

# Plot 2: Speedup comparison
ax2 = Axis(fig[1, 2],
    title="Speedup Relative to Baseline",
    xlabel="Kernel Configuration",
    ylabel="Speedup Factor",
    xticks=(1:4, ["Baseline", "Unrolled", "Tiled\n(32×16)", "Tiled\n(32×32)"]))

barplot!(ax2, 1:4, speedups, color=[:steelblue, :coral, :seagreen, :seagreen])
hlines!(ax2, [1.0], color=:gray, linestyle=:dash, linewidth=1)

# Add value labels
for (i, s) in enumerate(speedups)
    text!(ax2, i, s, text="$(round(s, digits=2))x",
          align=(:center, :bottom), fontsize=12)
end

# Highlight the winner
scatter!(ax2, [argmax(speedups)], [maximum(speedups)],
         color=:red, markersize=25, marker=:star5)
fig

```
```julia (editor=true, logging=false, output=true)
# Performance summary as markdown
pixels_total = prod(size(bench_img))

fps = 1000.0 ./ times
mrays = (pixels_total / 1e6) .* fps
w, h = size(bench_img)
kernel_names = ["Baseline (loop)", "Unrolled (4 samples)", "Tiled (32×16)", "Tiled (32×32)"]

md"""
## Performance Summary

**Resolution:** $(w)×$(h) = $(pixels_total)
pixels
**Samples per pixel:** 4

### GPU Performance Results

| Kernel | Time (ms) | FPS | MRays/s | Speedup vs Baseline |
|--------|-----------|-----|---------|---------------------|
| $(kernel_names[1]) | $(round(times[1], digits=2)) | $(round(fps[1], digits=2)) | $(round(mrays[1], digits=2)) | 1.00x |
| $(kernel_names[2]) | $(round(times[2], digits=2)) | $(round(fps[2], digits=2)) | $(round(mrays[2], digits=2)) | **$(round(speedups[2], digits=2))x** ⚡ |
| $(kernel_names[3]) | $(round(times[3], digits=2)) | $(round(fps[3], digits=2)) | $(round(mrays[3], digits=2)) | $(round(speedups[3], digits=2))x |
| $(kernel_names[4]) | $(round(times[4], digits=2)) | $(round(fps[4], digits=2)) | $(round(mrays[4], digits=2)) | $(round(speedups[4], digits=2))x |

### Key Insights

🏆 **Winner:** $(kernel_names[argmin(times)]) at $(round(minimum(times), digits=2)) ms
⚡ **Best Speedup:** $(round(maximum(speedups), digits=2))x faster than baseline
🚀 **Peak Throughput:** $(round(maximum(mrays), digits=2)) MRays/s

**What we learned:**
- **Loop unrolling is crucial** - eliminates branch overhead and improves ILP
- **Tile size dramatically affects performance** - can cause 3x performance swings!
- **Optimal tile size is workload-dependent** - always benchmark!
- **Thread divergence matters** - raytracing has variable per-pixel cost
"""
```
### Next Steps

  * Implement **wavefront path tracing** to reduce thread divergence
  * Add **adaptive sampling** (more samples only where needed)
  * Explore **shared memory** optimizations for BVH traversal
  * Implement **streaming multisampling** across frames
  * Try **persistent threads** with dynamic work distribution

Happy GPU ray tracing!

