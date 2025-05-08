using RayCaster, GeometryBasics, LinearAlgebra
using GLMakie, FileIO

function LowSphere(radius, contact=Point3f(0); ntriangles=10)
    return Tesselation(Sphere(contact .+ Point3f(0, 0, radius), radius), ntriangles)
end

begin
    ntriangles = 10
    s1 = LowSphere(0.5f0, Point3f(-0.5, 0.0, 0); ntriangles)
    s2 = LowSphere(0.3f0, Point3f(1, 0.5, 0); ntriangles)
    s3 = LowSphere(0.3f0, Point3f(-0.5, 1, 0); ntriangles)
    s4 = LowSphere(0.4f0, Point3f(0, 1.0, 0); ntriangles)
    l = 0.5
    floor = Rect3f(-l, -l, -0.01, 2l, 2l, 0.01)
    cat = load(Makie.assetpath("cat.obj"))
    bvh = RayCaster.BVHAccel([s1, s2, s3, s4, cat]);
    world_mesh = GeometryBasics.Mesh(bvh)
    f, ax, pl = Makie.mesh(world_mesh; color=:teal)
    display(f)
    viewdir = normalize(ax.scene.camera.view_direction[])
end

hitpoints, centroid = RayCaster.get_centroid(bvh, viewdir)


begin
    @time "hitpoints" hitpoints, centroid = RayCaster.get_centroid(bvh, viewdir)
    @time "illum" illum = RayCaster.get_illumination(bvh, viewdir)
    @time "viewf_matrix" viewf_matrix = RayCaster.view_factors(bvh, rays_per_triangle=1000)
    viewfacts = map(i-> Float32(sum(view(viewf_matrix, :, i))), 1:length(bvh.primitives))
    world_mesh = GeometryBasics.Mesh(bvh)
    N = length(world_mesh.faces)
    areas = map(i-> area(world_mesh.position[world_mesh.faces[i]]), 1:N)
    # View factors
    f, ax, pl = mesh(world_mesh, color=:blue)
    per_face_vf = FaceView((viewfacts), [GLTriangleFace(i) for i in 1:N])
    viewfact_mesh = GeometryBasics.mesh(world_mesh, color=per_face_vf)
    pl = Makie.mesh(
        f[1, 2],
        viewfact_mesh, colormap=[:black, :red], axis=(; show_axis=false),
        shading=false, highclip=:red, lowclip=:black
    )

    # Centroid
    cax, pl = Makie.mesh(f[2, 1], world_mesh, color=(:blue, 0.5), axis=(; show_axis=false), transparency=true)

    eyepos = cax.scene.camera.eyeposition[]
    depth = map(x-> norm(x .- eyepos), hitpoints)
    meshscatter!(cax, hitpoints, color=depth, colormap=[:gray, :black], markersize=0.01)
    meshscatter!(cax, centroid, color=:red, markersize=0.05)

    # Illum
    per_face = FaceView(100f0 .* (illum ./ areas), [GLTriangleFace(i) for i in 1:N])
    illum_mesh = GeometryBasics.mesh(world_mesh, color=per_face)

    Makie.mesh(f[2, 2], illum_mesh, colormap=[:black, :yellow], shading=false, axis=(; show_axis=false))

    Label(f[0, 1], "Scene ($(length(bvh.primitives)) triangles)", tellwidth=false, fontsize=20)
    Label(f[0, 2], "Viewfactors", tellwidth=false, fontsize=20)
    Label(f[3, 1], "Centroid", tellwidth=false, fontsize=20)
    Label(f[3, 2], "Illumination", tellwidth=false, fontsize=20)

    f
end


using KernelAbstractions, Atomix

function random_scatter_kernel!(bvh, triangle, u, v, normal)
    point = RayCaster.random_triangle_point(triangle)
    o = point .+ (normal .* 0.01f0) # Offset so it doesn't self intersect
    dir = RayCaster.random_hemisphere_uniform(normal, u, v)
    ray = RayCaster.Ray(; o=o, d=dir)
    hit, prim, _ = RayCaster.intersect!(bvh, ray)
    return hit, prim
end

import GeometryBasics as GB

@kernel function viewfact_ka_kernel!(result, bvh, primitives, rays_per_triangle)
    idx = @index(Global)
    prim_idx = ((UInt32(idx) - UInt32(1)) ÷ rays_per_triangle) + UInt32(1)
    if prim_idx <= length(primitives)
        triangle, u, v, normal = primitives[prim_idx]
        hit, prim = random_scatter_kernel!(bvh, triangle, u, v, normal)
        if hit && prim.material_idx !== triangle.material_idx
            # weigh by angle?
            Atomix.@atomic result[triangle.material_idx, prim.material_idx] += 1
        end
    end
end

function view_factors!(result, bvh, prim_info, rays_per_triangle=10000)

    backend = get_backend(result)
    workgroup = 256
    total_rays = length(bvh.primitives) * rays_per_triangle
    per_workgroup = total_rays ÷ workgroup
    final_rays = per_workgroup * workgroup
    per_triangle = final_rays ÷ length(bvh.primitives)

    kernel = viewfact_ka_kernel!(backend, 256)
    kernel(result, bvh, prim_info, UInt32(per_triangle); ndrange = final_rays)
    return result
end

result = zeros(UInt32, length(bvh.primitives), length(bvh.primitives))
using AMDGPU
prim_info = map(bvh.primitives) do triangle
    n = GB.orthogonal_vector(Vec3f, GB.Triangle(triangle.vertices...))
    normal = normalize(Vec3f(n))
    u, v = RayCaster.get_orthogonal_basis(normal)
    return triangle, u, v, normal
end
bvh_gpu = RayCaster.to_gpu(ROCArray, bvh)
result_gpu = ROCArray(result)
prim_info_gpu = ROCArray(prim_info)
@time begin
    view_factors!(result_gpu, bvh_gpu, prim_info_gpu, 10000)
    KernelAbstractions.synchronize(get_backend(result_gpu))
end;



@kernel function viewfact_ka_kernel2!(result, bvh, primitives, rays_per_triangle)
    idx = @index(Global)
    prim_idx = ((UInt32(idx) - UInt32(1)) ÷ rays_per_triangle) + UInt32(1)
    if prim_idx <= length(primitives)
        triangle, u, v, normal = primitives[prim_idx]
        hit, prim = random_scatter_kernel!(bvh, triangle, u, v, normal)
        if hit && prim.material_idx !== triangle.material_idx
            # weigh by angle?
            @inbounds result[idx] = UInt32(1)
        end
    end
end


function view_factors2!(result, bvh, prim_info, per_triangle)
    backend = get_backend(result)
    kernel = viewfact_ka_kernel2!(backend, 256)
    kernel(result, bvh, prim_info, UInt32(per_triangle); ndrange = length(result))
    return result
end


using AMDGPU
workgroup = 256
rays_per_triangle = 10000
total_rays = length(bvh.primitives) * rays_per_triangle
per_workgroup = total_rays ÷ workgroup
final_rays = per_workgroup * workgroup
per_triangle = final_rays ÷ length(bvh.primitives)
result = zeros(UInt32, final_rays)

final_rays / 10^6

prim_info = map(bvh.primitives) do triangle
    n = GB.orthogonal_vector(Vec3f, GB.Triangle(triangle.vertices...))
    normal = normalize(Vec3f(n))
    u, v = RayCaster.get_orthogonal_basis(normal)
    return triangle, u, v, normal
end

bvh_gpu = RayCaster.to_gpu(ROCArray, bvh)
result_gpu = ROCArray(result)
prim_info_gpu = ROCArray(prim_info)
@time begin
    view_factors2!(result_gpu, bvh_gpu, prim_info_gpu, per_triangle)
    KernelAbstractions.synchronize(get_backend(result_gpu))
end;

@time view_factors2!(result, bvh, prim_info, per_triangle)
@code_warntype random_scatter_kernel!(bvh, prim_info[1]...)
