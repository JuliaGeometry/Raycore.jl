# Changelog

## 0.2.0

### Breaking changes

- The single-BVH API (`BVH`, `TriangleMesh`, `AccelPrimitive`) is replaced
  by a two-level acceleration-structure API: `BLAS` / `TLAS` (and the BVH4
  variants `BLAS4` / `TLAS4`). Build with `build_blas` / `build_tlas` /
  `build_blas4`. Multiple meshes/instances now live under one `TLAS`
  rather than being merged into a single `BVH`.
- The `RayIntersectionSession` API (`hit_points`, `hit_distances`,
  `hit_count`, `miss_count`) is removed. Use `trace_rays` /
  `closest_hit` / `any_hit` against a `TLAS` directly; the result types
  are `RTHitResult` and `RTRay`.
- Primitives now carry a generic `meta::T` field instead of separate
  material/primitive indices (this was prepared in 0.1.x via PR #9 and
  is now the only supported shape).

### Upgrading

- Replace `BVH(meshes...)` with `build_blas(...)` per geometry plus
  `build_tlas(blases, instances)`. For dynamic scenes use
  `TLASHandle` + `update!` + `sync!`.
- Replace `RayIntersectionSession` workflows with
  `trace_rays(tlas, rays)` returning `Vector{RTHitResult}`, or call
  `closest_hit(tlas, ray)` / `any_hit(tlas, ray)` for single-ray queries.
- For heterogeneous geometry sets, see the new `MultiTypeSet` /
  `StaticMultiTypeSet` API (`with_index`, `deref`, `store_texture`).

### New features

- Two-level acceleration structures (`TLAS` / `BLAS`) with instance
  transforms, dynamic `update!` / `sync!`, GPU-side `instance_buffer`,
  and lifecycle helpers (`n_instances`, `n_total_instances`,
  `n_geometries`, `wait_for_gpu!`, `free!`).
- BVH4 (4-wide) acceleration: `BVHNode4`, `BLAS4`, `TLAS4`,
  `build_blas4`, `closest_hit4`, `any_hit4`.
- `MultiTypeSet` / `StaticMultiTypeSet` for heterogeneous, type-stable
  collections; companion `SetKey` / `TextureRef` indexing types and
  `store_texture`, `maybe_convert_field` mutators.
- Collision queries: `collide_instances`, `collide_instances_any`,
  `ContactPair`, `CollisionResult`.
- Struct-of-arrays helpers: `@get`, `@set`, `similar_soa`, plus the
  unrolled-iteration utilities `for_unrolled`, `map_unrolled`,
  `reduce_unrolled`, `sum_unrolled`, `getindex_unrolled`, and
  `FastClosure`.
- Abstract accel hierarchy (`AbstractAccel`, `AbstractAdaptedAccel`)
  to host both software-BVH and hardware-RT (e.g. Vulkan) backends.
