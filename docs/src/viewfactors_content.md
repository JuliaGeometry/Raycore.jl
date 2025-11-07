# View Factors Analysis

This example demonstrates Raycore's analysis capabilities for radiosity and illumination calculations.

## Scene Setup

```julia (editor=true, logging=false, output=true)
using Raycore, GeometryBasics, LinearAlgebra
using WGLMakie, FileIO

function LowSphere(radius, contact=Point3f(0); ntriangles=10)
    return Tesselation(Sphere(contact .+ Point3f(0, 0, radius), radius), ntriangles)
end

# Create scene with multiple objects
ntriangles = 10
s1 = LowSphere(0.5f0, Point3f(-0.5, 0.0, 0); ntriangles)
s2 = LowSphere(0.3f0, Point3f(1, 0.5, 0); ntriangles)
s3 = LowSphere(0.3f0, Point3f(-0.5, 1, 0); ntriangles)
s4 = LowSphere(0.4f0, Point3f(0, 1.0, 0); ntriangles)
cat = load(Makie.assetpath("cat.obj"))

# Build BVH acceleration structure
bvh = BVH([s1, s2, s3, s4, cat])
world_mesh = GeometryBasics.Mesh(bvh)

# Visualize the scene
f, ax, pl = mesh(world_mesh, color=:teal, axis=(show_axis=false,))
center!(ax.scene)
f
```
## View Factors

View factors quantify how much each surface "sees" every other surface - essential for radiosity calculations.

```julia (editor=true, logging=false, output=true)
# Calculate view factors between all faces
viewf_matrix = view_factors(bvh, rays_per_triangle=1000)

# Sum up total view factor per face
viewfacts = map(i -> Float32(sum(view(viewf_matrix, :, i))), 1:length(bvh.primitives))
N = length(world_mesh.faces)

# Visualize
per_face_vf = FaceView(viewfacts, [GLTriangleFace(i) for i in 1:N])
viewfact_mesh = GeometryBasics.mesh(world_mesh, color=per_face_vf)
mesh(viewfact_mesh, colormap=:turbo, shading=false,
     lowclip=:black, colorrange=(0f0, maximum(viewfacts)),
     axis=(show_axis=false,))
```
Higher values (warm colors) indicate faces that see more of the surrounding geometry.

## Illumination

Calculate how much each face is exposed to rays from a specific viewing direction.

```julia (editor=true, logging=false, output=true)
# Get camera view direction
viewdir = normalize(ax.scene.camera.view_direction[])

# Compute illumination
illum = get_illumination(bvh, viewdir)

# Visualize
pf = FaceView(illum, [GLTriangleFace(i) for i in 1:N])
illum_mesh = GeometryBasics.mesh(world_mesh, color=pf)
mesh(illum_mesh, colormap=[:black, :yellow], colorscale=sqrt,
     shading=false, axis=(show_axis=false,))
```
Faces directly visible from the viewing direction show higher illumination (yellow).

## Centroid Calculation

Find the average position of visible surface points from a given direction.

```julia (editor=true, logging=false, output=true)
# Calculate centroid
hitpoints, centroid = get_centroid(bvh, viewdir)

# Visualize
f, ax, pl = mesh(world_mesh, color=(:blue, 0.5), transparency=true, axis=(show_axis=false,))
eyepos = ax.scene.camera.eyeposition[]
depth = map(x -> norm(x .- eyepos), hitpoints)
meshscatter!(ax, hitpoints, color=depth, colormap=[:gray, :black], markersize=0.01)
meshscatter!(ax, [centroid], color=:red, markersize=0.05)
f
```
The red sphere marks the centroid - useful for camera placement and focus calculations.

