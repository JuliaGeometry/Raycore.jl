# ============================================================================
# MultiTypeSet - Type-stable heterogeneous collections for GPU
# ============================================================================
# Provides compile-time type-stable dispatch over collections of different types.
# Used for materials, textures, media, lights, etc.

using Adapt
using Base: @propagate_inbounds
import KernelAbstractions as KA

# ============================================================================
# SetKey - Encodes type slot + vector index
# ============================================================================

"""
    SetKey

Index into a heterogeneous vector, encoding both which type slot (1-based)
and the index within that type's array.

- `type_idx`: Which tuple slot (1-based), 0 = invalid/constant sentinel
- `vec_idx`: 1-based index within the vector at that slot
"""
struct SetKey
    # UInt32 (not UInt8) is intentional: LLVM's select-scalarization pass produces broken
    # IR (`select i1` with mismatched result type) when scalarizing a `select { i8, i32 }`.
    # Using uniform UInt32 fields gives `{ i32, i32 }`, which scalarizes correctly.
    type_idx::UInt32
    vec_idx::UInt32
end

# Default constructor for invalid/placeholder index
SetKey() = SetKey(UInt32(0), UInt32(0))

# Check for invalid sentinel
is_invalid(idx::SetKey) = idx.type_idx == UInt32(0) && idx.vec_idx == UInt32(0)
is_valid(idx::SetKey) = !is_invalid(idx)

# ============================================================================
# StaticMultiTypeSet - Immutable with separate texture storage for GPU
# ============================================================================

"""
    StaticMultiTypeSet{Data, Textures}

Immutable heterogeneous collection with separate texture storage.
- `data`: Tuple of GPU vectors for materials/objects
- `textures`: Tuple of GPU vectors containing isbits device pointers
"""
struct StaticMultiTypeSet{Data<:Tuple,Textures<:Tuple} <: AbstractVector{Any}
    data::Data
    textures::Textures
end

# Empty constructor
StaticMultiTypeSet() = StaticMultiTypeSet((), ())

Base.isempty(smv::StaticMultiTypeSet) = isempty(smv.data)
Base.length(smv::StaticMultiTypeSet) = sum(length, smv.data; init=0)
n_slots(smv::StaticMultiTypeSet) = length(smv.data)

# Get the static version - identity for StaticMultiTypeSet, .static field for MultiTypeSet
get_static(smv::StaticMultiTypeSet) = smv

# Convert to a flat Tuple of all elements (preserves concrete element types)
_concat_to_tuple() = ()
_concat_to_tuple(v::AbstractVector, rest...) = (v..., _concat_to_tuple(rest...)...)
to_tuple(smv::StaticMultiTypeSet) = _concat_to_tuple(smv.data...)

# ============================================================================
# foreach_element - Type-stable iteration over all elements
# ============================================================================

"""
    foreach_element(f, smv::StaticMultiTypeSet, args...)

Execute function `f` for each element in the StaticMultiTypeSet, passing additional `args`.
The function is called as `f(element, linear_idx, args...)` where `element` has a concrete type
and `linear_idx` is the 1-based linear index across all type slots.

Uses compile-time unrolled loops for type stability.
The function `f` must not capture variables - pass all data as `args`.
"""
@inline @generated function foreach_element(
    f::F, smv::StaticMultiTypeSet{Data, Textures}, args...
) where {F, Data<:Tuple, Textures}
    N = length(Data.parameters)

    if N == 0
        return :(nothing)
    end

    # Generate unrolled loops over each type slot
    loops = Expr[]
    for i in 1:N
        push!(loops, quote
            for j in eachindex(smv.data[$i])
                linear_idx += 1
                @inbounds f(smv.data[$i][j], linear_idx, args...)
            end
        end)
    end

    quote
        linear_idx = 0
        $(loops...)
        nothing
    end
end

# ============================================================================
# mapreduce - Type-stable reduction over all elements
# ============================================================================


@inline function Base.mapreduce(
        f::F, op::Op, smv::StaticMultiTypeSet{Data,Textures}, args...; init
    ) where {F,Op,Data<:Tuple,Textures}
    _mapreduce(f, op, smv, init, args...)
end
@inline function Base.mapreduce(
    f::F, op::Op, smv::StaticMultiTypeSet{Data,Textures}, args::Vararg{Union{Base.AbstractBroadcasted,AbstractArray}}; init
) where {F,Op,Data<:Tuple,Textures}
    _mapreduce(f, op, smv, init, args...)
end

@inline @generated function _mapreduce(
     f::F, op::Op, smv::StaticMultiTypeSet{Data,Textures}, init, args...) where {F, Op, Data<:Tuple, Textures}
    N = length(Data.parameters)

    if N == 0
        return :(init)
    end

    # Generate unrolled reduction over each type slot
    reductions = Expr[]
    for i in 1:N
        push!(reductions, quote
            for j in eachindex(smv.data[$i])
                @inbounds acc = op(acc, f(smv.data[$i][j], args...))
            end
        end)
    end

    quote
        acc = init
        $(reductions...)
        acc
    end
end

# ============================================================================
# TextureRef - Typed reference to a texture
# ============================================================================

# TIdx is the 1-based type slot index, idx is the element index within that slot's vector
struct TextureRef{ReferencedArrayType, T, N, TIdx} <: AbstractArray{T, N}
    idx::Int
end

Base.size(::TextureRef{ReferencedArrayType, T, N}) where {ReferencedArrayType, T, N} = ntuple(_ -> 0, N)

# Deref for StaticMultiTypeSet - textures stored as Tuple{GPUVector{IsbitsPtr1}, GPUVector{IsbitsPtr2}, ...}
@inline function deref(smv::StaticMultiTypeSet{Data, Textures}, tref::TextureRef{ReferencedArrayType, T, N, TIdx}) where {Data, Textures, ReferencedArrayType, T, N, TIdx}
    @inbounds smv.textures[TIdx][tref.idx]
end

# Fallback: if already a concrete array, just return it (no-op for CPU paths or non-TextureRef fields)
@inline deref(::StaticMultiTypeSet, arr::AbstractArray) = arr

# Fallback for nothing context (used by convenience overloads for CPU code that doesn't use MultiTypeSet)
@inline deref(::Nothing, arr::AbstractArray) = arr

@inline function deref(smv::StaticMultiTypeSet{Data,Textures}, tref::TextureRef{ReferencedArrayType,T,N,TIdx}) where {Data<:Tuple,Textures<:Tuple,ReferencedArrayType,T,N,TIdx}
    @inbounds smv.textures[TIdx][tref.idx]
end
# ============================================================================
# Dummy kernel for argconvert (same pattern as kernel-abstractions.jl)
# ============================================================================

KA.@kernel multitypeset_dummy_kernel() = nothing

function get_isbits_ptr(backend, gpu_arr)
    kernel = multitypeset_dummy_kernel(backend)
    return KA.argconvert(kernel, gpu_arr)
end

# ============================================================================
# MultiTypeSet - Mutable, builds GPU-ready structures on push!
# ============================================================================

"""
    MultiTypeSet(backend)

Mutable heterogeneous vector that builds GPU-ready structures on each push!.
Takes a KernelAbstractions backend at construction.

# Example
```julia
backend = OpenCL.OpenCLBackend()
dhv = MultiTypeSet(backend)
texture = rand(Float32, 20, 20)
idx1 = push!(dhv, MatteMaterial(texture))
idx2 = push!(dhv, GlassMaterial(1.5f0))

# Access the GPU-ready StaticMultiTypeSet directly
gpu_smv = dhv.static  # Always up-to-date, no adapt needed
```

Push converts arrays to TextureRefs and stores texture data as GPU arrays.
The static field is rebuilt on each push to stay up-to-date.
"""
mutable struct MultiTypeSet{Backend} <: AbstractVector{Any}
    backend::Backend
    # Material storage - CPU vectors for accumulation (the authoritative data).
    data_vectors::Dict{DataType, Any}  # Type -> Vector{Type}
    data_order::Vector{DataType}
    # Texture type order.  The shader-visible table of isbits device pointers
    # lives only in `static.textures[slot]` — no parallel CPU mirror.
    texture_order::Vector{DataType}
    # Keep GPU texture arrays alive (the actual texture data).  The backend
    # handle kept here is the single owner for each texture's backing buffer.
    texture_gpu_arrays::Vector{Any}
    # Canonical GPU state.  Every mutator (`push!` / `update!` /
    # `store_texture` / `copyto_texture!`) keeps this field consistent by
    # design — surgical `resize!` + `@allowscalar setindex!` on the affected
    # slot — so there is no dirty flag and no batched rebuild step.  The
    # TLAS (`scene.accel`) has its own dirty+sync because BVH rebuilds are
    # genuinely expensive; MultiTypeSet's element-level updates are cheap
    # (one scalar GPU write) and pay no amortisation benefit from batching.
    static::StaticMultiTypeSet
end

Base.size(set::MultiTypeSet) = (length(set),)
function Base.length(set::MultiTypeSet)
    return sum(length, values(set.data_vectors); init=0)
end

function Base.show(io::IO, ::MIME"text/plain", set::MultiTypeSet)
    n_types = length(set.data_order)
    total = length(set)
    print(io, "MultiTypeSet with $n_types type(s), $total element(s)")
    for T in set.data_order
        vec = set.data_vectors[T]::Vector
        print(io, "\n  ", length(vec), "× ", T)
    end
end

Base.show(io::IO, set::MultiTypeSet) = print(io, "MultiTypeSet(", length(set.data_order), " types, ", length(set), " elements)")

function MultiTypeSet(backend)
    return MultiTypeSet(
        backend,
        Dict{DataType, Any}(),
        DataType[],
        DataType[],
        Any[],
        StaticMultiTypeSet(),
    )
end

n_slots(dhv::MultiTypeSet) = length(dhv.data_order)

# `static` is always in sync with CPU state (maintained per-mutation).
get_static(dhv::MultiTypeSet) = dhv.static

# MultiTypeSet delegates to its static version
to_tuple(mts::MultiTypeSet) = to_tuple(get_static(mts))

# `rebuild_static!` is deleted — mutators (`push!`, `update!`, `store_texture`,
# `copyto_texture!`) keep `static` consistent surgically, so there is no
# batched rebuild step.  The TLAS (`scene.accel`) has its own `dirty + sync!`
# because BVH rebuilds are expensive; MultiTypeSet operations are all O(1)
# scalar GPU writes and pay no amortisation benefit from batching.

# ============================================================================
# Texture conversion and storage
# ============================================================================

"""
    maybe_convert_field(dhv::MultiTypeSet, fval)

Convert a struct field value for GPU storage. Override this for custom types.
- AbstractArray → TextureRef (uploaded to GPU)
- Everything else → unchanged (default)

Materials should use loose type parameters so fields can be either raw values OR
TextureRef. This way constant values don't need texture indirection at all.
This function should not be overloaded outside Raycore.
"""
# Convert large arrays to TextureRef, but NOT StaticArrays (they're inline values, not textures)
maybe_convert_field(dhv::MultiTypeSet, arr::A) where A<:AbstractArray = store_texture(dhv, arr)
maybe_convert_field(::MultiTypeSet, arr::StaticArrays.StaticArray) = arr  # Keep StaticArrays inline
# Don't re-convert already converted refs
maybe_convert_field(::MultiTypeSet, ref::TextureRef) = ref
# Default: recurse into structs, pass through primitives
function maybe_convert_field(dhv::MultiTypeSet, item::T) where T
    # Recurse into struct types to convert nested arrays
    if !isbitstype(T)
        return convert_to_texturerefs(dhv, item)
    end
    # Primitives and empty structs pass through unchanged
    return item
end

# Convert arrays in a struct to TextureRefs, storing them as GPU arrays
function convert_to_texturerefs(dhv::MultiTypeSet, item::T) where T
    if !isstructtype(T) || T <: AbstractArray
        return item
    end
    fnames = fieldnames(T)
    if isempty(fnames)
        return item
    end
    new_fields = map(fnames) do fname
        fval = getfield(item, fname)
        maybe_convert_field(dhv, fval)
    end
    if all(getfield(item, fn) === nf for (fn, nf) in zip(fnames, new_fields))
        return item
    end
    BaseT = Base.typename(T).wrapper
    return BaseT(new_fields...)
end

# Store a texture as a GPU array, return a TextureRef pointing to its isbits
# device pointer.  Keeps `texture_gpu_arrays` and `static.textures[slot]`
# consistent in one call:
#  * existing texture type → `resize!` + `@allowscalar setindex!` to append one
#    isbits pointer to the matching slot (one scalar GPU write).
#  * new texture type → build a 1-element LavaArray{IsbitsPtr} and grow the
#    `static.textures` tuple by one slot (tuple shape change is unavoidable).
function store_texture(dhv::MultiTypeSet, arr::AbstractArray{T}) where T
    if !isbitstype(T)
        arr = map(x -> maybe_convert_field(dhv, x), arr)
    end
    gpu_arr = Adapt.adapt(dhv.backend, arr)
    AT = typeof(gpu_arr)
    push!(dhv.texture_gpu_arrays, gpu_arr)
    isbits_ptr = get_isbits_ptr(dhv.backend, gpu_arr)

    type_idx = findfirst(==(AT), dhv.texture_order)
    if type_idx === nothing
        # New texture type: extend `static.textures` by one slot.
        push!(dhv.texture_order, AT)
        type_idx = length(dhv.texture_order)
        new_slot = Adapt.adapt(dhv.backend, [isbits_ptr])
        dhv.static = StaticMultiTypeSet(dhv.static.data, (dhv.static.textures..., new_slot))
        vec_idx = 1
    else
        # Existing texture type: surgical one-element append into the GPU slot.
        slot = dhv.static.textures[type_idx]
        old_len = length(slot)
        resize!(slot, old_len + 1)
        @allowscalar slot[old_len + 1] = isbits_ptr
        vec_idx = old_len + 1
    end
    return TextureRef{AT, eltype(AT), ndims(AT), type_idx}(vec_idx)
end

# ============================================================================
# push! - Append item, keeping CPU + GPU state consistent in one call.
# ============================================================================
# Existing type slot → surgical `resize!` + `@allowscalar setindex!` appends
# one element to `static.data[type_idx]` (one scalar GPU write).  New type
# slot → build a 1-element LavaArray and extend the `static.data` tuple by
# one slot (tuple shape change is unavoidable when a new type appears).
function Base.push!(dhv::MultiTypeSet, item::T)::SetKey where T
    # Convert arrays in the item to TextureRefs (textures stored via `store_texture`).
    converted_item = maybe_convert_field(dhv, item)
    CT = typeof(converted_item)

    type_idx = findfirst(==(CT), dhv.data_order)
    if type_idx === nothing
        # New material type: extend `static.data` by one slot.
        dhv.data_vectors[CT] = [converted_item]
        push!(dhv.data_order, CT)
        type_idx = length(dhv.data_order)
        new_slot = Adapt.adapt(dhv.backend, [converted_item])
        dhv.static = StaticMultiTypeSet((dhv.static.data..., new_slot), dhv.static.textures)
        vec_idx = 1
    else
        # Existing material type: surgical one-element append into the GPU slot.
        push!(dhv.data_vectors[CT], converted_item)
        slot = dhv.static.data[type_idx]
        old_len = length(slot)
        resize!(slot, old_len + 1)
        @allowscalar slot[old_len + 1] = converted_item
        vec_idx = old_len + 1
    end
    return SetKey(UInt32(type_idx), UInt32(vec_idx))
end

# ============================================================================
# update! - Sync modified CPU data into existing GPU arrays
# ============================================================================

"""
    update!(dhv::MultiTypeSet, key::SetKey, new_item)

Update an existing item in the set.  The new item is walked against the
stored form via `update_item`: existing TextureRef slots are reused (the new
array data is copied into the existing GPU buffer, reallocating on size
mismatch), const-Texture fields are unwrapped to their scalar values, and
other fields fall through as plain value replacement.

There is deliberately **no** `maybe_convert_field`/`store_texture` call on
this path — that would allocate a new GPU slot per update and leak hundreds
of MB per frame for plots with per-vertex color textures.
"""
function update!(dhv::MultiTypeSet, key::SetKey, new_item)
    CT = dhv.data_order[key.type_idx]
    old_converted = dhv.data_vectors[CT][key.vec_idx]
    updated = update_item(dhv, old_converted, new_item)
    if updated !== old_converted
        # Keep CPU and GPU consistent in one go — surgical single-element write.
        dhv.data_vectors[CT][key.vec_idx] = updated
        @allowscalar dhv.static.data[key.type_idx][key.vec_idx] = updated
    end
    return nothing
end

public update_item, copyto_texture!

"""
    update_item(dhv::MultiTypeSet, old, new)

Compute the updated representation of `old` after applying `new`'s data.
Reuses existing TextureRef slots (copying `new`'s arrays into them rather
than allocating fresh GPU buffers).  Extended by backend / material packages
(e.g. Hikari) with overloads for their wrapper types — notably `Texture`
(unwraps const, routes array data to `copyto_texture!`) and `VertexColorTexture`.
"""
function update_item(dhv::MultiTypeSet, old::TextureRef{AT}, new_data::AbstractArray) where AT
    copyto_texture!(dhv, old, new_data)
    return old
end

# Already-converted TextureRef on both sides: just reuse the existing slot.
update_item(::MultiTypeSet, old::TextureRef, ::TextureRef) = old

# Nothing/Nothing: no-op.
update_item(::MultiTypeSet, ::Nothing, ::Nothing) = nothing

# Generic fallback: walk field-by-field when field names match, otherwise
# replace `old` with `new` (leaf case — isbits values, identically-typed
# structs with no nested arrays, etc.).  A type parameter mismatch (e.g.
# `Diffuse{TextureRef,Float32}` ↔ `Diffuse{VertexColorTexture{Matrix},Texture{Float32}}`)
# still recurses because the field names are identical.  The reconstructed
# struct uses the concrete type produced by the per-field recursive calls;
# when TextureRefs are kept in place and const-Textures are unwrapped, that
# matches the stored (old) type.
# Tuple / NamedTuple recursion: these can't be reconstructed via
# `T.name.wrapper(fields...)` like regular structs can (`Tuple(a,b,c)` expects
# an iterable, not varargs). Handle them explicitly.
function update_item(dhv::MultiTypeSet, old::Tuple, new::Tuple)
    length(old) == length(new) || return new
    changed = false
    new_fields = ntuple(length(old)) do i
        uf = update_item(dhv, old[i], new[i])
        uf !== old[i] && (changed = true)
        uf
    end
    changed || return old
    return new_fields
end
function update_item(dhv::MultiTypeSet, old::NamedTuple{K}, new::NamedTuple{K}) where K
    changed = false
    new_fields = ntuple(length(K)) do i
        uf = update_item(dhv, old[i], new[i])
        uf !== old[i] && (changed = true)
        uf
    end
    changed || return old
    return NamedTuple{K}(new_fields)
end

function update_item(dhv::MultiTypeSet, old, new)
    T_old = typeof(old)
    fnames = fieldnames(T_old)
    isempty(fnames) && return new  # leaf — swap in the new value
    # Only recurse if new exposes the same field names (types of individual
    # fields may still differ).
    fnames == fieldnames(typeof(new)) || return new
    changed = false
    new_fields = ntuple(length(fnames)) do i
        of = getfield(old, fnames[i])
        nf = getfield(new, fnames[i])
        uf = update_item(dhv, of, nf)
        uf !== of && (changed = true)
        uf
    end
    changed || return old
    return T_old.name.wrapper(new_fields...)
end

"""
    copyto_texture!(dhv, ref, data)

Write `data` into the GPU texture addressed by `ref`.  Same-size is a plain
`copyto!` (device pointer unchanged — no other state needs updating).  Size
mismatch goes through `Base.resize!(::LavaArray)` which is capacity-aware:
the VkBuffer is only re-allocated on genuine growth beyond current capacity,
and the old buffer is retired via the deferred-free path (`bq.deferred_frees`
gated on the batch timeline — safe w.r.t. in-flight GPU work without any
CPU-side `synchronize`).

If the device pointer actually moved (pool-alloc returned a fresh buffer),
the one affected slot of `static.textures[AT_slot]` is updated via a single
scalar `setindex!` — no re-adapt of the whole table, no per-frame leak.
"""
function copyto_texture!(dhv::MultiTypeSet, ref::TextureRef{AT}, new_data::AbstractArray) where AT
    AT_slot = findfirst(==(AT), dhv.texture_order)
    AT_slot === nothing && error("MultiTypeSet has no texture type slot for $AT (TextureRef broken?)")

    count = 0
    for arr in dhv.texture_gpu_arrays
        typeof(arr) === AT || continue
        count += 1
        count == ref.idx || continue

        if size(arr) == size(new_data)
            # Same shape: pointer cannot move — one copyto!, done.
            copyto!(arr, new_data)
        else
            # Capacity-aware grow: zero-alloc within capacity, deferred-free
            # on true growth.  Compare the device pointer before/after to see
            # whether the surrounding isbits table needs one slot refreshed.
            old_ptr = get_isbits_ptr(dhv.backend, arr)
            resize!(arr, size(new_data))
            copyto!(arr, new_data)
            new_ptr = get_isbits_ptr(dhv.backend, arr)
            if new_ptr != old_ptr
                # Pointer moved (grow past capacity).  Refresh just this one
                # slot of the shader-visible isbits buffer — no rebuild, no
                # dirty flag, no sync.  A later `get_static` that finds
                # `dirty == false` sees a tuple that's already consistent.
                @allowscalar dhv.static.textures[AT_slot][ref.idx] = new_ptr
            end
        end
        return
    end
    error("GPU array not found for TextureRef(idx=$(ref.idx))")
end

# No `update!` / `resize_and_overwrite!` hook: call sites use `Base.resize!` +
# `Base.copyto!` directly.  Lava's `Base.resize!(::LavaArray)` is capacity-aware
# (no Vulkan alloc when within capacity, deferred-free on grow) and `copyto!`
# is the standard GPUArrays upload path — that pair is the full "make dst match
# src" operation without a Raycore-owned generic.

# ============================================================================
# with_index - Type-stable dispatch
# ============================================================================

"""
    with_index(f, smv::StaticMultiTypeSet, idx::SetKey, args...)

Execute function `f` with the element at index `idx`, passing additional `args`.
The function is called as `f(element, args...)` where `element` has a concrete type.

Uses a single if-elseif-else chain for SPIR-V structured control flow compatibility.
The function `f` must not capture variables - pass all data as `args`.
"""
@inline @generated function with_index(
    f::F, smv::StaticMultiTypeSet{Data, Textures}, idx::SetKey, args...
) where {F, Data<:Tuple, Textures}
    N = length(Data.parameters)

    if N == 0
        return :(error("with_index: empty StaticMultiTypeSet"))
    end

    # Build a single if-elseif-else chain for structured control flow (SPIR-V compatible)
    # Start from the last branch and work backwards to build the chain
    result = :(@inbounds f(smv.data[1][1], args...))  # default/else case

    for i in N:-1:1
        result = Expr(:if,
            :(idx.type_idx === UInt32($i)),
            :(@inbounds f(smv.data[$i][idx.vec_idx], args...)),
            result
        )
    end

    quote
        return $result
    end
end

# ============================================================================
# Adapt.jl integration for GPU array conversion
# ============================================================================

# Adapt StaticMultiTypeSet - adapts data and texture arrays
# For MultiTypeSet.static, arrays are already GPU - this converts to isbits for kernel
function Adapt.adapt_structure(to, smv::StaticMultiTypeSet)
    adapted_data = map(smv.data) do arr
        Adapt.adapt(to, arr)
    end
    adapted_textures = map(smv.textures) do tex
        Adapt.adapt(to, tex)
    end
    return StaticMultiTypeSet(adapted_data, adapted_textures)
end

# Adapt MultiTypeSet - `static` is always consistent (surgical-per-mutation);
# just hand it to the StaticMultiTypeSet adapt method.
function Adapt.adapt_structure(to, dhv::MultiTypeSet)
    return Adapt.adapt_structure(to, dhv.static)
end

# ============================================================================
# GPU Resource Cleanup
# ============================================================================

"""
    free!(set::MultiTypeSet)

Release GPU memory held by the MultiTypeSet — the shadow-owned texture
arrays plus the static material/texture slot buffers.  Does **not**
synchronize.

**Precondition (caller's responsibility):** the GPU must be idle for
`set.backend` before this is called.  `MultiTypeSet.texture_gpu_arrays`
is a shadow-ownership site: the only references from GPU work are raw
BDAs in arg buffers, so nothing inside `free!` can prove it's safe to
finalize — the caller establishes that, typically by calling `sync!` on
the enclosing accel / scene (which synchronizes its backend) or by
returning from a `colorbuffer` that completed with `device_wait_idle`.
"""
function free!(set::MultiTypeSet)
    for arr in set.texture_gpu_arrays
        finalize(arr)
    end
    empty!(set.texture_gpu_arrays)
    # `set.static.data` / `.textures` are the canonical ownership sites for
    # the adapted material / isbits-ptr arrays.  Finalize them, then drop
    # the tuples via a fresh empty StaticMultiTypeSet.
    for arr in set.static.data
        finalize(arr)
    end
    for arr in set.static.textures
        finalize(arr)
    end
    set.static = StaticMultiTypeSet()
    return nothing
end
