# ImplicitBVH.jl vs Raycore.jl — GPU Benchmark

**Date**: 2026-03-29
**GPU**: AMD RX 7900 XTX (RDNA3)
**Backend**: AMDGPU.jl (ROCArray)
**Mesh**: xyzrgb_dragon.obj (249,882 triangles) + procedural random geometry

## BVH Build

| Triangles | ImplicitBVH | Raycore | Ratio |
|-----------|-------------|---------|-------|
| 250K      | 0.98 ms     | 4.93 ms | ImplicitBVH 5.0x faster |
| 1M        | 2.25 ms     | 7.46 ms | ImplicitBVH 3.3x faster |
| 4M        | 8.41 ms     | 16.16 ms | ImplicitBVH 1.9x faster |

ImplicitBVH builds faster due to simpler construction (Morton sort + bottom-up aggregate).
Raycore does more work: topology emission, parent pointers, leaf creation, atomic refit — all separate kernel launches.

## Ray Tracing — Dragon Mesh (249K triangles)

**Important**: ImplicitBVH `traverse_rays` returns bounding volume candidates (broad-phase only).
Raycore `closest_hit` returns the actual closest triangle intersection (full narrow-phase).
These are fundamentally different operations — ImplicitBVH does less work per ray but doesn't give a usable hit result.

| Rays | ImplicitBVH (LVT) | Raycore | Speedup (Raycore) |
|------|--------------------|---------|--------------------|
| 100K | 4.60 ms            | 1.33 ms | 3.5x |
| 500K | 11.06 ms           | 3.14 ms | 3.5x |
| 1M   | 20.84 ms           | 3.00 ms | 6.9x |
| 2M   | 41.52 ms           | 6.00 ms | 6.9x |
| 4M   | 83.31 ms           | 5.91 ms | 14.1x |

## Ray Tracing — Scaling with Triangle Count (1M rays)

| Triangles | ImplicitBVH (BFS) | Raycore | Speedup (Raycore) |
|-----------|--------------------|---------|--------------------|
| 250K      | 43.89 ms           | 8.99 ms | 4.9x |
| 1M        | 217.37 ms          | 11.08 ms | 19.6x |
| 4M        | 313.0 ms           | 15.41 ms | 20.3x |

## Why Raycore Is Faster for Ray Tracing

| Factor | ImplicitBVH | Raycore |
|--------|-------------|---------|
| Output | All BV candidates (variable-size list) | Single closest hit (fixed) |
| Triangle test | None (BSphere overlap only) | Moller-Trumbore per leaf |
| Passes | Two-pass (count + write) | Single-pass |
| Early termination | No — finds all overlaps | Yes — t_max shrinks on hit |
| Node layout | Implicit tree + skip array | Inline leaves (BVH2IL) |
| Allocations | Output buffer per trace | None |

## What ImplicitBVH Does Better

- **Build speed** (2-5x faster) — fewer kernel launches, implicit indexing
- **Collision detection** — LVT (leaf-vs-tree) two-pass is designed for finding all contact pairs
- **Two-BVH collision** — native support for inter-object contact detection
- **Cache reuse** — `BVHTraversal` cache avoids re-allocation across frames
- **Mixed bounding volumes** — BSphere leaves with BBox internal nodes

## Reference: ImplicitBVH README Numbers (Nvidia A100)

From the ImplicitBVH.jl README (249,882 triangles, BSphere/BBox):
- Build: 410 us
- Contact detection: 1.14 ms
- 100K rays: 2.00 ms
