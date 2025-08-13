# RayCaster.jl

[![Build Status](https://github.com/JuliaGeometry/RayCaster.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/JuliaGeometry/RayCaster.jl/actions/workflows/ci.yml?query=branch%3Amaster)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliageometry.github.io/RayCaster.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliageometry.github.io/RayCaster.jl/dev/)

Performant ray intersection engine for CPU and GPU.

## Getting Started

To get started with RayCaster.jl, first add the package to your Julia environment:

```julia
using Pkg
Pkg.add("RayCaster")
```

Then you can create a basic ray intersection scene:

```julia
using RayCaster, GeometryBasics, LinearAlgebra

# Create some simple spheres
function LowSphere(radius, contact=Point3f(0); ntriangles=10)
    return Tesselation(Sphere(contact .+ Point3f(0, 0, radius), radius), ntriangles)
end

# Build a scene with multiple objects
s1 = LowSphere(0.5f0, Point3f(-0.5, 0.0, 0); ntriangles=10)
s2 = LowSphere(0.3f0, Point3f(1, 0.5, 0); ntriangles=10)

# Create BVH acceleration structure
bvh = RayCaster.BVHAccel([s1, s2])

# Perform ray-scene intersections
viewdir = normalize(Vec3f(0, 0, -1))
hitpoints, centroid = RayCaster.get_centroid(bvh, viewdir)
```

## Documentation

For detailed examples and API documentation, see the [full documentation](https://juliageometry.github.io/RayCaster.jl/).
