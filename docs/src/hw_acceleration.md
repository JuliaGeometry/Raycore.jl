# Hardware Ray Tracing with Mantle

Modern GPUs traverse BVHs and intersect triangles in fixed-function hardware (RT cores on NVIDIA, Ray Accelerators on AMD, the ray-tracing units of Apple's M3 and later). Raycore's API is written so a kernel does not care which kind of acceleration structure it traces: [Mantle](https://github.com/SimonDanisch/Mantle.jl) implements it for the hardware structures, on Vulkan (through the Lava compiler, as a ray query) and on Metal (through its intersector).

The demo below builds the same scene twice, into a software `Raycore.TLAS` and into a hardware `Mantle.HWTLAS`, traces one camera ray per pixel through both with **the same kernel**, and checks that the depth buffers agree.

## When to pick `Raycore.TLAS` vs. `Mantle.HWTLAS`

|             Aspect |                          `Raycore.TLAS` |                                                 `Mantle.HWTLAS` |
| ------------------:| ---------------------------------------:| ---------------------------------------------------------------:|
|            Backend |       any KernelAbstractions backend |  Mantle on Vulkan with ray query, or on Metal with ray tracing  |
|                BVH |  software (LBVH over instanced BLASes) |          `VkAccelerationStructureKHR` / `MTLAccelerationStructure` |
|       Tracing code | a KA `@kernel` calling `Raycore.closest_hit(accel, ray)` |                                              the same kernel |
|           Use when | portability, no RT hardware, CPU backends |                                                 RT hardware |

Both satisfy `Raycore.AbstractAccel`: `push!`, `delete!`, `update_transform!` (one transform for every instance of a handle), `update_transforms!`, `set_visible!`, `sync!`, `n_instances`, `n_geometries`, `world_bound` and `wait_for_gpu!` mean the same on both. So do the per-instance cull masks (`push!(…; instance_mask)`, traced with `closest_hit(accel, ray, mask)`) and `Raycore.refit!(accel)`. Two differences remain: only the hardware structure hands out instance records a kernel may write (`instance_buffer(accel, handle)`, then `Raycore.refit!`), and the hit itself differs: on hardware `closest_hit` returns the triangle in world space and, as its fifth value, the instance's custom index (`instance_id`); on `Raycore.TLAS` the triangle is in its BLAS's space and the fifth value is the instance's position in `accel.instances`, whose transform moves the hit to world space. The distance `t` means the same on both.

## When hardware RT helps

The biggest gains come from scenes with many triangles, deep occlusion (many traversal steps per ray) and cheap shading. For a trivial scene the software BVH already runs at memory bandwidth; the ~48k triangles below are enough to see the hardware pull ahead and small enough for a tutorial.

`Mantle.supports_hwtlas(backend)` says whether a device has the hardware path.

## Setup

```julia
using Raycore, Mantle, GeometryBasics, LinearAlgebra, Adapt
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

backend = Mantle.LavaBackend()        # Vulkan; on a Mac: `using Metal; Metal.MetalBackend()`
Mantle.supports_hwtlas(backend)       # true on an RT-capable device
```

## Building the scene twice

Tessellated spheres on a floor. Both structures take plain `GeometryBasics.Mesh` objects through `push!`.

```julia
function build_meshes()
    floor = GeometryBasics.normal_mesh(Rect3f(Vec3f(-3, -3, -0.01), Vec3f(6, 6, 0.01)))
    centers = [Point3f(i * 0.9f0, j * 0.9f0, 0.4f0) for i in -2:2 for j in -2:2]
    spheres = [GeometryBasics.normal_mesh(Tesselation(Sphere(c, 0.3f0), 32)) for c in centers]
    return [floor; spheres]
end

meshes = build_meshes()
sw = Raycore.TLAS(backend)
hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(backend)
for (i, m) in enumerate(meshes)
    push!(sw, m)
    push!(hw, m; instance_id = UInt32(i))
end
Raycore.sync!(sw)
Raycore.sync!(hw)
```

26 meshes, 26 instances in each, 48 062 triangles. `sync!` uploads the meshes and builds the structures: GPU LBVH builds (one BLAS per mesh and a TLAS over the instances) for `Raycore.TLAS`, the driver's acceleration-structure builds for `Mantle.HWTLAS`.

## One kernel, both structures

```julia
const W, H = 256, 192
cam = Point3f(0, -3.5, 1.6)
forward = normalize(Point3f(0, 0, 0.3) - cam)
right = normalize(cross(forward, Vec3f(0, 0, 1)))
up = cross(right, forward)
focal = 1f0 / tan(deg2rad(45f0 / 2))
rays = [Raycore.Ray(o = cam, d = Vec3f(normalize(forward * focal +
                                                 right * ((2f0 * (x - 0.5f0) / W - 1f0) * Float32(W / H)) +
                                                 up * (1f0 - 2f0 * (y - 0.5f0) / H))))
        for x in 1:W, y in 1:H]

@kernel function depth!(depth, @Const(rays), accel)
    i = @index(Global, Linear)
    @inbounds begin
        hit, _, t, _, _ = Raycore.closest_hit(accel, rays[i])
        depth[i] = hit ? t : -1f0
    end
end

rays_dev = Adapt.adapt(backend, vec(rays))
depth_sw = KA.zeros(backend, Float32, W * H)
depth_hw = KA.zeros(backend, Float32, W * H)
k = depth!(backend, 64)
trace!(depth, accel) = (k(depth, rays_dev, Adapt.adapt(backend, accel); ndrange = W * H); KA.synchronize(backend))
trace!(depth_sw, sw)
trace!(depth_hw, hw)
```

`Adapt.adapt(backend, accel)` is what a kernel receives: a `StaticTLAS` for the software structure, an `AdaptedAccel` for the hardware one. Call it per dispatch — after a mutation and `sync!` it may be a different object — and never cache it across mutations.

Measured on a Radeon 8060S (RADV) and on an Apple M5:

|                           | Radeon 8060S | Apple M5 |
| -------------------------:| ------------:| --------:|
| hit-mask disagreement     | 0 of 49 152  | 0 of 49 152 |
| max depth difference      | 1.3e-5       | 1.8e-5   |
| software, one frame       | 0.22 ms      | 1.0 ms   |
| hardware, one frame       | 0.10 ms      | 0.47 ms  |

The remaining depth difference is rounding: both intersect the same triangles, the hardware in its own arithmetic. The timing ratio depends on triangle count, ray coherence and the GPU; measure your scene.

## Lifetime

`sync!` owns the adapted form: consumers re-read it, through `Adapt.adapt`, per dispatch. `sync!` does not block the CPU; buffers an old structure used stay alive until the GPU is past every submission that read them. For a CPU-side drain, before tear-down or between benchmark phases, call `Raycore.wait_for_gpu!(accel)`.

## Ray-tracing pipelines (Vulkan only)

Beyond ray queries from compute kernels, Mantle can run a Vulkan ray-tracing pipeline — raygen, closest-hit, any-hit and miss shaders written in Julia and compiled by Lava, which exposes the `lava_rt_*` intrinsics for them — through `Mantle.RayTracingPipeline` and `Mantle.trace_rays!`. Metal has no counterpart reachable from Julia kernels, so code meant to run on both should trace from a kernel as above.
