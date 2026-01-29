# ============================================================================
# HeterogeneousVector - Type-stable heterogeneous collections for GPU
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
    type_idx::UInt8
    vec_idx::UInt32
end

# Default constructor for invalid/placeholder index
SetKey() = SetKey(UInt8(0), UInt32(0))

# Check for invalid sentinel
is_invalid(idx::SetKey) = idx.type_idx == UInt8(0) && idx.vec_idx == UInt32(0)
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
# Fallback for Tuple (used by legacy code paths)
get_static(t::Tuple) = t

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

# ============================================================================
# Dummy kernel for argconvert (same pattern as kernel-abstractions.jl)
# ============================================================================

KA.@kernel _heterovec_dummy_kernel() = nothing

function _get_isbits_ptr(backend, gpu_arr)
    kernel = _heterovec_dummy_kernel(backend)
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
    # Material storage - CPU vectors for accumulation
    data_vectors::Dict{DataType, Any}  # Type -> Vector{Type}
    data_order::Vector{DataType}
    # Texture isbits pointers - CPU vectors for accumulation
    texture_isbits::Dict{DataType, Any}  # OriginalArrayType -> Vector{IsbitsPtr}
    texture_order::Vector{DataType}
    # Keep GPU texture arrays alive (the actual texture data)
    texture_gpu_arrays::Vector{Any}
    # Cached static version - rebuilt on each push
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
        Dict{DataType, Any}(),
        DataType[],
        Any[],
        StaticMultiTypeSet()
    )
end

n_slots(dhv::MultiTypeSet) = length(dhv.data_order)

# Get the static version - returns .static field for MultiTypeSet
get_static(dhv::MultiTypeSet) = dhv.static

# ============================================================================
# Internal: Rebuild the static tuple - converts CPU vectors to GPU
# ============================================================================

function _rebuild_static!(dhv::MultiTypeSet)
    # Convert CPU data vectors to GPU
    data_tuple = if isempty(dhv.data_order)
        ()
    else
        Tuple(Adapt.adapt(dhv.backend, dhv.data_vectors[T]) for T in dhv.data_order)
    end
    # Convert CPU isbits pointer vectors to GPU
    tex_tuple = if isempty(dhv.texture_order)
        ()
    else
        Tuple(Adapt.adapt(dhv.backend, dhv.texture_isbits[T]) for T in dhv.texture_order)
    end
    dhv.static = StaticMultiTypeSet(data_tuple, tex_tuple)
end

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

# Store a texture as GPU array, return TextureRef pointing to isbits device pointer
function store_texture(dhv::MultiTypeSet, arr::AbstractArray{T}) where T
    # Convert to GPU array and keep alive
    if !isbitstype(T)
        arr = map(x -> maybe_convert_field(dhv, x), arr)
    end
    gpu_arr = Adapt.adapt(dhv.backend, arr)
    AT = typeof(gpu_arr)
    push!(dhv.texture_gpu_arrays, gpu_arr)

    # Get isbits device pointer
    isbits_ptr = _get_isbits_ptr(dhv.backend, gpu_arr)

    # Check if this texture type already has a slot
    type_idx = findfirst(==(AT), dhv.texture_order)

    if type_idx === nothing
        # New texture type - create CPU vector for it
        dhv.texture_isbits[AT] = [isbits_ptr]
        push!(dhv.texture_order, AT)
        type_idx = length(dhv.texture_order)
    else
        # Existing type - push to CPU vector
        push!(dhv.texture_isbits[AT], isbits_ptr)
    end
    vec_idx = length(dhv.texture_isbits[AT])
    return TextureRef{AT, eltype(AT), ndims(AT), type_idx}(vec_idx)
end

# ============================================================================
# push! - Adds item to CPU vectors, rebuilds GPU static on each push
# ============================================================================

function Base.push!(dhv::MultiTypeSet, item::T)::SetKey where T
    # Convert any arrays in the item to TextureRefs (textures stored as GPU arrays)
    # Uses maybe_convert_field which dispatches to type-specific methods or generic conversion
    converted_item = maybe_convert_field(dhv, item)
    CT = typeof(converted_item)

    # Check if this material type already has a slot
    type_idx = findfirst(==(CT), dhv.data_order)

    if type_idx === nothing
        # New material type - create CPU vector for it
        dhv.data_vectors[CT] = [converted_item]
        push!(dhv.data_order, CT)
        type_idx = length(dhv.data_order)
    else
        # Existing type - push to CPU vector
        push!(dhv.data_vectors[CT], converted_item)
    end

    vec_idx = length(dhv.data_vectors[CT])

    # Rebuild GPU static on every push
    _rebuild_static!(dhv)

    return SetKey(UInt8(type_idx), UInt32(vec_idx))
end

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
            :(idx.type_idx === UInt8($i)),
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

# Adapt MultiTypeSet - returns the already GPU-ready static field
function Adapt.adapt_structure(to, dhv::MultiTypeSet)
    return Adapt.adapt_structure(to, dhv.static)
end
