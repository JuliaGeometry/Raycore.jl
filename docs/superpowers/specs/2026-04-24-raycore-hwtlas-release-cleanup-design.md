# Raycore HWTLAS + Lava Integration Cleanup — Design

**Date:** 2026-04-24
**Branch context:** `sd/multitype-vec` (Raycore v0.1.1); Lava v0.1.0
**Status:** design approved; ready for planning

## Goal

Make Raycore's hardware ray tracing path release-ready. Replace the HWTLAS/Lava
integration's ad-hoc memory management and backend-coupling scaffolding with a
clean division: Raycore owns the software TLAS and the abstract accel
interface; Lava owns the hardware TLAS and all HW RT orchestration. Bring
tests and docs in line with the new division.

Merging and releasing are out of scope (handled by the user). What's in scope
is the quality of the code and tests that the release will ship.

## Non-Goals

- Registering Raycore.jl in General (user will do this separately).
- Introducing a second HW backend (Metal, OptiX). The design is
  non-obstructive to that work but explicitly YAGNI today.
- Redesigning Lava's timeline/BQ/DataRef machinery. We use what's already
  there.
- Rewriting Hikari's wavefront renderer. Hikari gets a small grep-and-replace
  migration for imports, nothing more.

## Where Things Live (new division)

### Raycore (core) keeps

- `AbstractAccel`, `AbstractAdaptedAccel` — abstract interface + contract
  (see "Contract" below).
- `TLAS` (software, KA-based, backend-agnostic) — the reference implementation
  of `AbstractAccel`.
- Geometry types: `Ray`, `Triangle`, `Bounds3`, `Normal3f`, `TLASHandle`,
  `InstanceDescriptor`, `build_triangle`, `is_degenerate_face`.
- Transport types: `RTRay`, `RTHitResult` (shared by SW and HW backends; no
  backend-specific layout).
- MultiTypeSet, SoA utilities, analysis (`get_centroid`, `get_illumination`,
  `view_factors`), collision.
- The Makie extension (`RaycoreMakieExt`) — unchanged.
- New helper: `empty_triangle(::Type{T})` — returns a zero-initialized
  `Triangle` for no-hit sentinel returns.

### Lava takes over

- `HWTLAS <: Raycore.AbstractAccel` — moves from `Raycore/src/hw-accel.jl` to
  `Lava/src/raytracing/hwtlas.jl`. Typed concretely on Lava's own types
  (`LavaBLAS`, `LavaTLAS`, `LavaArray{T}`, `HardwareAccel`, `BatchQueue`). No
  more `::Any`.
- `HWAdaptedAccel{H<:HWTLAS}` — same move.
- `PrecomputedHitsAccel` — moves from the extension into Lava proper.
- `HardwareAccel` — already lives in Lava. Stays mutable (so `sync!` can reuse
  the RT pipeline + SBT across rebuilds).
- `rebuild_hw_tlas!(hwtlas)` — the concrete, Lava-private entry that
  `sync!(::HWTLAS)` uses to do the Vulkan TLAS + BLAS build.

### Deletes

- `Raycore/src/hw-accel.jl` — gone (content moves to Lava).
- `Raycore/ext/RaycoreLavaExt/` — gone entirely. No more extension.
- Raycore stubs: `build_hw_blas`, `build_hw_tlas`, `release_hw_accel_state!`,
  `supports_indirect_dispatch`, `indirect_ndrange`, `mat4_to_transform_matrix`,
  `batch_trace_indirect`, `trace_closest_hits!`,
  `trace_closest_hits_indirect!`, `set_custom_anyhit!`, and the `rt_*`
  intrinsic stubs (`rt_primitive_id`, `rt_instance_custom_index`,
  `rt_instance_id`, `rt_launch_id_x`, `rt_global_invocation_id_x`,
  `rt_ignore_intersection`, `rt_terminate_ray`, `rt_payload_store!`,
  `rt_payload_load`, `rt_trace_ray!`).
- Raycore Project.toml: remove `[weakdeps] Lava`, `[extensions]
  RaycoreLavaExt`. Keep `Lava` under `[extras]` + `test` target.
- The `_HWTLASInstances` shim and the `getproperty(::HWTLAS, :instances)`
  override — replaced by an `n_instances(accel)` contract.

## Contract for `AbstractAccel`

This is the drop-in contract that both `Raycore.TLAS` and `Lava.HWTLAS`
implement. Consumers (Hikari, RayMakie, user code) program against this, not
against either concrete type.

Mutation API:

- `push!(accel, mesh, transform::Mat4f; kwargs...) -> TLASHandle`
- `push!(accel, mesh, transforms::AbstractVector{Mat4f}; kwargs...) -> TLASHandle`
- `delete!(accel, handle::TLASHandle) -> Bool`
- `update_transform!(accel, handle, transform) -> Bool`
- `update_transform_at!(accel, handle, i::Integer, transform) -> Bool`

Lifecycle:

- `sync!(accel)` — single owner of `accel.static_tlas`. Rebuilds in place
  where possible; reassigns the field when a buffer had to grow. A no-op on a
  clean accel (no GPU synchronize, no reallocation).
- `sync!` does **not** block the CPU on a GPU fence. Dirty rebuilds drop old
  backing refs; backend-internal timeline tracking handles the "still in
  flight" case.
- `Adapt.adapt(backend, accel) === accel.static_tlas` between `sync!`s.
  Consumers re-read `accel.static_tlas` (or call `Adapt.adapt(backend,
  accel)`) **per dispatch**. They MUST NOT cache the adapted form across
  mutations.

Query:

- `closest_hit(adapted, ray) -> (hit::Bool, tri, t, bary, instance_override)`
- `any_hit(adapted, ray) -> Bool`
- `world_bound(accel) -> Bounds3`
- `n_instances(accel) -> Int`
- `n_geometries(accel) -> Int`

Optional flush (not part of the per-dispatch hot path):

- `wait_for_gpu!(accel)` — blocks the CPU until the GPU has completed all
  prior work on this accel's queue. Convenience for tear-down and benchmark
  isolation. Default implementation on `AbstractAccel` forwards to
  `KA.synchronize` on the accel's backend. `Lava.HWTLAS` overrides to wait
  on `hwtlas.bq`'s timeline specifically (which may differ from the
  backend-wide queue in a future multi-queue setup).

## Lifetime Model (the core cleanup)

### HWTLAS carries an explicit BatchQueue

```julia
mutable struct HWTLAS{Tri} <: Raycore.AbstractAccel
    backend::LavaBackend
    bq::BatchQueue                       # RT dispatch + sync! work go here

    # Geometry — typed triangle vector, no Vector{Vector{Any}}
    blas_list::Vector{LavaBLAS}
    blas_triangles::Vector{Vector{Tri}}
    blas_offsets::Vector{UInt32}

    # Instances
    instance_blas_indices::Vector{Int}
    instance_transforms::Vector{NTuple{12,Float32}}
    instance_custom_indices::Vector{UInt32}

    # Handles
    handle_to_range::Dict{TLASHandle, UnitRange{Int}}
    deleted_handles::Set{TLASHandle}
    next_handle_id::UInt32

    root_aabb::Bounds3

    # Built on sync! — typed
    hw_tlas::Union{Nothing, LavaTLAS}
    hw_accel::Union{Nothing, HardwareAccel}
    tri_gpu::Union{Nothing, LavaArray{Tri}}
    off_gpu::Union{Nothing, LavaArray{UInt32}}

    static_tlas::Union{Nothing, HWAdaptedAccel{<:HWTLAS{Tri}}}
    dirty::Bool
end
```

Constructors:

```julia
HWTLAS(backend::LavaBackend; bq::BatchQueue = backend.bq)
# Triangle type inferred on first push!; second push! with a different Tri errors.

HWTLAS{Tri}(backend::LavaBackend; bq::BatchQueue = backend.bq) where Tri
# Pin the triangle type up front — useful when arguments are empty at construction.
```

`bq = backend.bq` as the default matches Lava's existing pattern (same as
`HardwareAccel`). A later multi-queue user passes `bq=custom` at construction;
nothing else changes.

### Lava-side fix: pin closure leaves in RT dispatch

`trace_rays!(bq, pipeline, tlas, ...)` currently pins `tlas.accel/storage`,
each `blas.accel/storage`, and `pipeline`. It does **not** walk the
raygen/closest-hit closures for `LavaArray` leaves. `ka_launch!` does, via
`pin_leaves!(batch, f)`.

The fix: in `trace_rays!` and `trace_rays_indirect!`, call `pin_leaves!` on
every shader closure in `pipeline` (raygen, closest_hit, miss, any_hit when
present). Every `LavaArray` closed over — including `tri_gpu` and `off_gpu`
— lands in `batch.pinned` → `sync_access!` updates its `last_write`. From
that moment on, `unsafe_free!` on those buffers is timeline-gated by
construction.

### `sync!(hwtlas)` becomes non-blocking

- No `KA.synchronize(hwtlas.backend)` call.
- Overwrites `hw_tlas`, `hw_accel`, `tri_gpu`, `off_gpu` with new values, or
  reuses in place when the same `LavaArray{T}` fits the new size.
- Calls `Lava.unsafe_free!` directly on any old object that was replaced
  (i.e. not in-place reused). Because the old objects had their `last_write`
  populated by prior dispatches (now guaranteed by the pinning fix above),
  `unsafe_free!` defers via `bq.deferred_as_frees` / `bq.deferred_frees` —
  correct regardless of whether the GPU is currently idle.
- A newly-built object with no prior dispatches has `last_write == nothing`
  → `unsafe_free!` destroys immediately. Safe because nothing on the GPU
  references it yet.

No `release_hw_accel_state!` abstraction. `Lava.unsafe_free!` is the only
name. With HWTLAS now living in Lava, there is no backend-agnostic stub to
abstract over.

### Explicit flush

`wait_for_gpu!(hwtlas)` is a separate one-liner
(`wait_on_timeline(hwtlas.bq, current_value)`) kept for users who need a
CPU-blocking drain (benchmark tear-down, inter-test isolation). Not called
from `sync!`.

### `HardwareAccel` stays mutable

`sync!` mutates `accel.tlas`, `accel.triangle_data`, `accel.blas_offsets`,
`accel.per_instance_tri_offsets` in place to keep the RT pipeline + SBT alive
across rebuilds — one pipeline per HWTLAS, not per sync. The current code
already does this; the only rename is `build_hw_tlas` → `rebuild_hw_tlas!`
(signals "mutates in place", Lava-private).

## API Cleanups

### `_HWTLASInstances` shim → `n_instances(accel)`

Drop the `getproperty(::HWTLAS, :instances)` override that returned a
length-only fake. Both `TLAS` and `HWTLAS` expose `n_instances(accel) -> Int`
and `n_geometries(accel) -> Int` as their only public "how much geometry?"
hooks. RayMakie (the one caller of `.instances`) switches to
`n_instances(accel)` — one-line change in RayMakie.

`TLAS.instances` (the real field) stays but is treated as implementation
detail, not a public accessor.

### Type-stable no-hit from `PrecomputedHitsAccel.closest_hit`

Replace `accel.triangles[1]` as the dummy-on-miss with an explicit empty
sentinel stored on the accel:

```julia
struct PrecomputedHitsAccel{R, T, O, Tri} <: AbstractAdaptedAccel
    results::R
    triangles::T
    offsets::O
    empty::Tri        # zero-initialized; returned on miss
end
```

`empty` is populated at construction from `Raycore.empty_triangle(Tri)`.
`closest_hit` returns `accel.empty` on miss. Works with zero triangles, no
contract that `triangles[1]` be a sentinel.

Raycore's SW `closest_hit` is audited for the same pattern; if it relies on a
dummy-triangle sentinel in the same way, it adopts `empty_triangle(Tri)` as
well.

### Drop Raycore-side trampolines

Remove from Raycore (list in "Deletes" above). Hikari updates its imports:

```julia
# before
using Raycore: rt_instance_id, rt_primitive_id, rt_trace_ray!, ...
# after
using Lava: lava_rt_instance_id, lava_rt_primitive_id, lava_rt_trace_ray, ...
```

The `lava_rt_*` names stay in Lava as-is; no aliasing. The prefix makes
origin explicit.

## Tests

### Raycore test suite

- **Keep unchanged:** `test_intersection.jl`, `bounds.jl`,
  `test_instanced_bvh.jl`, `test_multitypeset.jl`, `test_type_stability.jl`,
  `test_unrolled.jl` (if re-enabled).
- **Rewrite `test_mesh_update.jl`** to keep only the SW TLAS `@testset`s.
  Drop the HW sections; they move to Lava.
- **New `test_abstract_accel_contract.jl`:** parametrized over concrete accel
  factories. Same assertions for each:
  - `push!` returns a handle; `sync!` makes it visible to `closest_hit`.
  - `delete!; sync!` removes.
  - `update_transform!; sync!` moves geometry (verified by ray hit `t`).
  - `Adapt.adapt(backend, accel) === accel.static_tlas` between `sync!`s.
  - `sync!` on a clean accel is a true no-op (identity preserved).
  - `n_instances`, `n_geometries`, `world_bound` return sensible values.

  Registered factories: `(LavaBackend()) -> Raycore.TLAS`, `(LavaBackend())
  -> Lava.HWTLAS`. Same assertions, both pass — this is the drop-in proof.
- **Drop `test_hw_tlas_lava.jl`** from Raycore (moves to Lava).

### Lava test suite

- **Move** the HW portions of `Raycore/test/test_mesh_update.jl` →
  `Lava/test/test_hwtlas_mesh_update.jl`.
- **Move** `Raycore/test/test_hw_tlas_lava.jl` → `Lava/test/test_hwtlas_stress.jl`.
- **New `test_hwtlas_uaf_safety.jl`:**
  1. Build `HWTLAS` with 1 BLAS, 1 instance.
  2. Dispatch `trace_closest_hits!` (don't wait on the GPU).
  3. `push!` a second mesh, `sync!(hwtlas)` — this drops the old `tri_gpu`
     ref via `unsafe_free!` while the step-2 dispatch is still in flight.
  4. Dispatch another `trace_closest_hits!`.
  5. `wait_for_gpu!(hwtlas)`; read back both hit buffers.
  6. Assert both traces produced correct analytic hits.
  If the closure-leaves pinning is missing or `unsafe_free!` runs ahead of
  the fence, step-2's dispatch UAFs. Result: wrong hits / device lost.
- **New `test_hwtlas_nonblocking_sync.jl`:** submit a known GPU-bound
  workload on the HWTLAS's queue, call `sync!` on a dirty topology change,
  verify `sync!` returns before the GPU-bound work would have completed
  (timing-based, with a generous tolerance). If the legacy CPU fence reappears
  during refactoring, this test catches it.
- **Existing Lava tests:** untouched.

### Test deps

- Raycore keeps `Lava` as an `[extras]` + `test` target dep; direct `using
  Lava` in tests (no extension anymore).
- Lava keeps `Raycore` via `[sources]` path dep (unchanged).

## Documentation

### Raycore

- `docs/src/index.md`: update the "GPU TLAS" bullet to note that
  `Raycore.TLAS` is software / any-backend; `Lava.HWTLAS` is the Vulkan RT
  drop-in.
- `docs/src/hw_acceleration.md`: rewrite the first third —
  - import `HWTLAS` from `Lava`, not Raycore;
  - remove `Raycore.build_hw_blas` / `Raycore.rt_*` mentions;
  - new "Lifetime" section documenting the contract: `sync!` owns
    `static_tlas`; consumers re-read per dispatch; `sync!` does not block;
    `wait_for_gpu!` for explicit flush.
- `docs/src/hw_acceleration.md`: add a short "`Raycore.TLAS` vs.
  `Lava.HWTLAS`" section — one-page table of behavior, backend support,
  performance tradeoffs.
- `docs/src/raytracing_tutorial.md`, `docs/src/gpu_raytracing.md`: verify
  each snippet still runs; regenerate example images only if output changes
  (not expected).
- `README.md`: quick-start shows `Raycore.TLAS` (works anywhere). Add a
  second quick-start with `Lava.HWTLAS` and a pointer to the HW RT tutorial.
- Docstrings:
  - `Raycore.AbstractAccel` — the contract from the "Contract" section.
  - `Raycore.TLAS` — audit for any mention of the HW path that should be
    removed (was correct already; confirm).

### Lava

- `README.md`: in the Ray Tracing section, promote `HWTLAS(backend)` as the
  high-level API (currently the README shows `HardwareAccel(scene.accel)`,
  which is the low-level object).
- `Lava.HWTLAS` docstring: full contract spec + usage + `bq` parameter +
  lifetime notes.

### Not touched

- `docs/src/bvh_hit_tests.md`, `docs/src/viewfactors.md` — SW-only,
  unchanged.
- `docs/src/gpu_raytracing_tutorial.md` — KA-based, no HW RT content.
- Benchmark images — dispatch model and perf profile are unchanged by this
  refactor.

## Migration for downstream consumers

### Hikari

- `using Raycore: HWTLAS, ...` → `using Lava: HWTLAS, ...`.
- RT shader intrinsic imports: `rt_*` → `lava_rt_*` (grep-and-replace).
- `Raycore.release_hw_accel_state!(x)` → `Lava.unsafe_free!(x)` (if called
  directly; most call sites should disappear because they relied on the
  Raycore fence).

### RayMakie

- `accel.instances` → `Raycore.n_instances(accel)` — one-line fix.

### Other users

- `Raycore.HWTLAS` → `Lava.HWTLAS`. Everything else (`push!`, `delete!`,
  `sync!`, `closest_hit`) is unchanged by the refactor.

## Open Questions (to resolve during planning/implementation)

- Exact signature of `Raycore.empty_triangle(::Type{T})` — whether it's a
  generated zero-field constructor or needs per-triangle-type specialization
  for SoA or non-POD eltypes. Resolved during the PrecomputedHitsAccel task.
- Whether the contract test's "no-op on clean accel" assertion requires
  `@allocated` checks (currently disabled on Julia 1.12). If so, gate
  with the same `VERSION < v"1.12"` guard used in `test_type_stability.jl`.
- Whether `HWTLAS{Tri}(backend)` needs to narrow `hw_accel::Union{Nothing,
  HardwareAccel}` further to preserve type stability of `closest_hit`
  through the adapted form. Spot-check with `@code_warntype` during
  implementation.

## Success Criteria

1. `Raycore/ext/RaycoreLavaExt/` no longer exists; `[weakdeps]` + `[extensions]`
   removed from `Project.toml`.
2. `Raycore.jl` has no `rt_*` stubs, no `build_hw_*`, no
   `release_hw_accel_state!`, no `_HWTLASInstances`.
3. `Lava.HWTLAS` is the sole HW RT drop-in. Concrete typing (no `::Any` on
   GPU-buffer fields).
4. `sync!(hwtlas)` does not call `KA.synchronize`. `sync!` on a clean
   topology is a true no-op.
5. Lava's `trace_rays!` pins raygen/closest_hit closure leaves into
   `batch.pinned`.
6. Raycore's `test_abstract_accel_contract.jl` passes with both
   `Raycore.TLAS` and `Lava.HWTLAS` as the accel type.
7. Lava's new `test_hwtlas_uaf_safety.jl` and
   `test_hwtlas_nonblocking_sync.jl` pass.
8. Existing Raycore + Lava test suites pass on CI with no new failures.
9. `docs/src/hw_acceleration.md` builds and its code snippets execute.
10. Hikari and RayMakie import-only migrations compile and their existing
    tests pass.
