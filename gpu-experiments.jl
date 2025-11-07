using Rusticl_jll, pocl_jll, OpenCL, AMDGPU, KernelAbstractions, Makie, LinearAlgebra, GeometryBasics, Raycore, StaticArrays, FileIO, Colors
import KernelAbstractions as KA
using ImageShow

# Load the cat model and rotate it to face the camera
cat_mesh = Makie.loadasset("cat.obj")
angle = deg2rad(150f0)
rotation = Makie.Quaternionf(0, sin(angle / 2), 0, cos(angle / 2))
rotated_coords = [rotation * Point3f(v) for v in coordinates(cat_mesh)]

# Get bounding box and translate cat to sit on the floor
cat_bbox = Rect3f(rotated_coords)
floor_y = -1.5f0
cat_offset = Vec3f(0, floor_y - cat_bbox.origin[2], 0)

cat_mesh = GeometryBasics.normal_mesh(
    [v + cat_offset for v in rotated_coords],
    faces(cat_mesh)
)
# Create a simple room: floor, back wall, and side wall
floor = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)))
back_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)))
left_wall = normal_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)))

# Add a couple of spheres for visual interest
sphere1 = Tesselation(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), 64)
sphere2 = Tesselation(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), 64)

# Build our BVH acceleration structure
scene_geometry = [cat_mesh, floor, back_wall, left_wall, sphere1, sphere2]
bvh = Raycore.BVH(scene_geometry)

# Compute interpolated normal at hit point
function compute_normal(triangle, bary_coords)
    v0, v1, v2 = Raycore.normals(triangle)
    u, v, w = bary_coords[1], bary_coords[2], bary_coords[3]
    return Vec3f(normalize(v0 * u + v1 * v + v2 * w))
end

# Generate camera ray for a pixel with optional jitter
function camera_ray(x, y, width, height, camera_pos, focal_length, aspect; jitter=Vec2f(0))
    ndc_x = (2.0f0 * (Float32(x) - 0.5f0 + jitter[1]) / Float32(width) - 1.0f0) * aspect
    ndc_y = 1.0f0 - 2.0f0 * (Float32(y) - 0.5f0 + jitter[2]) / Float32(height)
    direction = normalize(Vec3f(ndc_x, ndc_y, focal_length))
    return Raycore.Ray(o=camera_pos, d=direction)
end
# Convert between color representations
to_vec3f(c::RGB) = Vec3f(c.r, c.g, c.b)
to_rgb(v::Vec3f) = RGB{Float32}(v[1], v[2], v[3])


@kernel function ka_trace_image!(color_callback, bvh, camera_pos, focal_length, aspect, img, sky_color)
    _idx = @index(Global, Cartesian)
    x, y = map(x-> x % Int32, Tuple(_idx))
    height, width = size(img)
    @inbounds if checkbounds(Bool, img, _idx)
        ray = camera_ray(y, x, width, height, camera_pos, focal_length, aspect)
        hit_found, triangle, distance, bary_coords = Raycore.closest_hit(bvh, ray)
        color = if hit_found
            to_vec3f(color_callback(bvh, triangle, distance, bary_coords, ray))
        else
            to_vec3f(sky_color)
        end
        img[x, y] = color
    end
    nothing
end

function ka_trace!(color_callback, ArrayType, bvh;
        width=2048, height=1024,
        camera_pos=Point3f(0, -0.9, -2.5), fov=45.0f0,
        sky_color=RGB{Float32}(0.5f0, 0.7f0, 1.0f0),
    )
    img = ArrayType(fill(Vec3f(0), height, width))
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    aspect = Float32(width/height)
    focal_length = 1.0f0 / tan(deg2rad(fov / 2))
    kernel!(color_callback, bvh, camera_pos, focal_length, aspect, img, sky_color, ndrange=size(img), workgroupsize=(16, 16))
    KA.synchronize(backend)

    return map(x-> to_rgb(clamp.(x, Vec3f(0), Vec3f(1))), Array(img))
end

using ImageShow, Colors

depth_kernel(bvh, tri, dist, bary, ray) = RGB(1.0f0 - min(dist / 10.0f0, 1.0f0))

# # ROCArray somehow stays black!
# pres = []
# g_scene = Trace.to_gpu(ROCArray, bvh; preserve=pres);
# img = ka_trace!(depth_kernel, ROCArray, g_scene)
# all(img .== RGB(0,0,0)) # nooo :(

# # OpenCL backend works, but runs on the CPU:
# pres = []
# g_scene = Trace.to_gpu(CLArray, bvh; preserve=pres);
# img_cl = ka_trace!(depth_kernel, CLArray, g_scene)

# Save the scene
save("test-ray1.png", img_cl)
using BenchmarkTools
@btime ka_trace!(depth_kernel, Array, bvh)
