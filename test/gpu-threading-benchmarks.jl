using GeometryBasics, LinearAlgebra, RayCaster, BenchmarkTools

# using CUDA
# ArrayType = CuArray

LowSphere(radius, contact=Point3f(0)) = Sphere(contact .+ Point3f(0, 0, radius), radius)

function tmesh(prim, material)
    prim =  prim isa Sphere ? Tesselation(prim, 64) : prim
    return normal_mesh(prim)
end


begin
    material_red = nothing
    s1 = tmesh(LowSphere(0.5f0), material_red)
    s2 = tmesh(LowSphere(0.3f0, Point3f(0.5, 0.5, 0)), material_red)
    s3 = tmesh(LowSphere(0.3f0, Point3f(-0.5, 0.5, 0)), material_red)
    s4 = tmesh(LowSphere(0.4f0, Point3f(0, 1.0, 0)), material_red)

    ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, 0.01)), material_red)
    back = tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, 0.01, 10)), material_red)
    l = tmesh(Rect3f(Vec3f(-2, -5, 0), Vec3f(0.01, 10, 10)), material_red)
    r = tmesh(Rect3f(Vec3f(2, -5, 0), Vec3f(0.01, 10, 10)), material_red)
    bvh = RayCaster.BVHAccel([s1, s2, s3, s4, ground, back, l, r]);
end

# using AMDGPU
# ArrayType = ROCArray
# using CUDA
# ArrayType = CuArray

# using Metal
# ArrayType = MtlArray

preserve = []
gpu_scene = to_gpu(ArrayType, scene; preserve=preserve);
gpu_img = ArrayType(zeros(RGBf, res, res));
# launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
launch_trace_image!(gpu_img, cam, gpu_scene);
launch_trace_image!(img, cam, scene, lights)
# 76.420 ms (234 allocations: 86.05 KiB)
# 75.973 ms (234 allocations: 86.05 KiB)
Array(gpu_img)

function cu_trace_image!(img, camera, bvh, lights)
    x = threadIdx().x
    y = threadIdx().y
    if checkbounds(Bool, img, (x, y))
        @_inbounds img[x, y] = trace_pixel(camera, bvh, (x,y), lights)
    end
end

k = some_kernel(img)
ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(k, size(img), (16, 16))
blocks = length(KA.blocks(iterspace))
threads = length(KA.workitems(iterspace))

function cu_launch_trace_image!(img, camera, bvh, lights)
    CUDA.@sync @cuda threads = length(img) cu_trace_image!(img, camera, bvh, lights)
    return img
end
cu_launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
Array(gpu_img)
# 380.081 ms (913 allocations: 23.55 KiB)
# CUDA (3070 mobile)
# 238.149 ms (46 allocations: 6.22 KiB)
# Int64 -> Int32
# 65.34 m
# workgroupsize=(16,16)
# 31.022 ms (35 allocations: 5.89 KiB)

function trace_image!(img, camera, scene)
    for xy in CartesianIndices(size(img))
        @_inbounds img[xy] = RGBf(trace_pixel(camera, scene, xy).c...)
    end
    return img
end

function threads_trace_image!(img, camera, bvh)
    Threads.@threads for xy in CartesianIndices(size(img))
        @_inbounds img[xy] = trace_pixel(camera, bvh, xy)
    end
    return img
end

@btime trace_image!(img, cam, bvh)
# Single: 707.754 ms (0 allocations: 0 bytes)
# New Triangle layout  1
# 860.535 ms (0 allocations: 0 bytes)
# GPU intersection compatible
# 403.335 ms (0 allocations: 0 bytes)

@btime threads_trace_image!(img, cam, bvh)
# Start
# Multi : 73.090 ms (262266 allocations: 156.04 MiB)
# BVH inline
# Multi (static): 66.564 ms (122 allocations: 45.62 KiB)
# New Triangle layout 1
# 80.222 ms (122 allocations: 32.88 KiB)
# 42.842 ms (122 allocations: 32.88 KiB) (more inline)
# GPU intersection compatible
# 42.681 ms (122 allocations: 32.88 KiB)


using Tullio

@_inbounds function tullio_trace_image!(img, camera, bvh)
    @tullio img[x, y] = trace_pixel(camera, bvh, (x, y))
    return img
end

@btime tullio_trace_image!(img, cam, bvh)
# BVH inline + tullio
# Multi: 150.944 ms (107 allocations: 33.17 KiB)
# New Triangle layout 1
# 161.447 ms (107 allocations: 33.17 KiB)
# 117.139 ms (107 allocations: 33.17 KiB) (more inline)
# GPU intersection compatible
# 82.461 ms (109 allocations: 22.39 KiB)

@btime launch_trace_image!(img, cam, bvh)
# 71.405 ms (233 allocations: 86.05 KiB)
# 47.240 ms (233 allocations: 86.09 KiB)
# GPU intersection compatible
# 44.629 ms (233 allocations: 54.50 KiB)


##########################
##########################
##########################
# Random benchmarks
v1 = Vec3f(0.0, 0.0, 0.0)
v2 = Vec3f(1.0, 0.0, 0.0)
v3 = Vec3f(0.0, 1.0, 0.0)

ray_origin = Vec3f(0.5, 0.5, 1.0)
ray_direction = Vec3f(0.0, 0.0, -1.0)

using RayCaster: Normal3f
m = RayCaster.TriangleMesh(RayCaster.ShapeCore(), UInt32[1, 2, 3], Point3f[v1, v2, v3], [Normal3f(0.0, 0.0, 1.0), Normal3f(0.0, 0.0, 1.0), Normal3f(0.0, 0.0, 1.0)])

t = RayCaster.Triangle(m, 1)
r = RayCaster.Ray(o=Point3f(ray_origin), d=ray_direction)
RayCaster.intersect_p(t, r)
RayCaster.intersect_triangle(r.o, r.d, t.vertices...)

# function launch_trace_image_ir!(img, camera, bvh, lights)
#     backend = KA.get_backend(img)
#     kernel! = ka_trace_image!(backend)
#     open("test2.ir", "w") do io
#         @device_code_llvm io begin
#             kernel!(img, camera, bvh, lights, ndrange = size(img), workgroupsize = (16, 16))
#         end
#     end
#     AMDGPU.synchronize(; stop_hostcalls=false)
#     return img
# end

ray = RayCaster.RayDifferentials(RayCaster.Ray(o=Point3f(0.5, 0.5, 1.0), d=Vec3f(0.0, 0.0, -1.0)))
open("li.llvm", "w") do io
    code_llvm(io, RayCaster.li, typeof.((RayCaster.UniformSampler(8), 5, ray, scene, 1)))
end

open("li-wt.jl", "w") do io
    code_warntype(io, RayCaster.li, typeof.((RayCaster.UniformSampler(8), 5, ray, scene, 1)))
end

camera_sample = RayCaster.get_camera_sample(integrator.sampler, Point2f(512))
ray, ω = RayCaster.generate_ray_differential(integrator.camera, camera_sample)


ray = RayCaster.Ray(o=Point3f(0.0, 0.0, 2.0), d=Vec3f(0.0, 0.0, -1.0))
function test(results, bvh, ray)
    for i in 1:100000
        results[i] = RayCaster.any_hit(bvh, ray, PerfNTuple)
    end
    return results
end

@profview test(results, bvh, ray)
@btime RayCaster.closest_hit(bvh, ray)
results = Vector{Tuple{Bool, RayCaster.Triangle, Float32, Point3f}}(undef, 100000);
@btime test(results, bvh, ray);

@btime RayCaster.any_hit(bvh, ray)

@code_typed RayCaster.traverse_bvh(RayCaster.any_hit_callback, bvh, ray, RayCaster.MemAllocator())

sizeof(zeros(RayCaster.MVector{64,Int32}))

###
# Int32 always
# 42.000 μs (1 allocation: 624 bytes)
# Tuple instead of vector for nodes_to_visit
# 43.400 μs (1 allocation: 624 bytes)
# AFTER GPU rework
# closest_hit
# 40.500 μs (1 allocation: 368 bytes)
# intersect_p
# 11.500 μs (0 allocations: 0 bytes)

### LinearBVHLeaf as one type
# 5.247460 seconds (17.55 k allocations: 19.783 MiB, 46 lock conflicts)

struct PerfNTuple{N,T}
    data::NTuple{N,T}
end

@generated function RayCaster._setindex(r::PerfNTuple{N,T}, idx::IT, value::T) where {N,T, IT <: Integer}
    expr = Expr(:tuple)
    for i in 1:N
        idxt = IT(i)
        push!(expr.args, :(idx === $idxt ? value : r.data[$idxt]))
    end
    return :($(PerfNTuple)($expr))
end

Base.@propagate_inbounds Base.getindex(r::PerfNTuple, idx::Integer) = r.data[idx]

@generated function RayCaster._allocate(::Type{PerfNTuple}, ::Type{T}, ::Val{N}) where {T,N}
    expr = Expr(:tuple)
    for i in 1:N
        push!(expr.args, :($(T(0))))
    end
    return :($(PerfNTuple){$N, $T}($expr))
end

m = RayCaster._allocate(PerfNTuple, Int32, Val(64))
m2 = RayCaster._setindex(m, 10, Int32(42))

@btime RayCaster.any_hit(bvh, ray, PerfNTuple)
