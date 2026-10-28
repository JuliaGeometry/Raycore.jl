# ============================================================================
# SetView - some slots of a StaticMultiTypeSet, under the set's own numbers
# ============================================================================

"""
    SetView{DataSlots, TextureSlots}

Some of the slots of a `StaticMultiTypeSet`: the data slots numbered `DataSlots` and the texture
slots numbered `TextureSlots` (tuples of the set's own slot numbers), so a `SetKey` and a texture's
`(slot, idx)` mean what they mean in the set. `with_index` and `with_texture` call `f(nothing,
args...)` for a slot the view does not hold.

A kernel that reads a few of a set's slots takes a view of those: its type then names those slots
only, and a slot added to the set for something else leaves it, and the kernel compiled for it,
as they were. Take one with [`setview`](@ref).
"""
struct SetView{DataSlots, TextureSlots, Data<:Tuple, Textures<:Tuple}
    data::Data
    textures::Textures
end

SetView{DS, TS}(data::Data, textures::Textures) where {DS, TS, Data<:Tuple, Textures<:Tuple} =
    SetView{DS, TS, Data, Textures}(data, textures)

"""
    setview(smv::StaticMultiTypeSet, data_slots::Tuple, texture_slots::Tuple) -> SetView

The data slots `data_slots` and the texture slots `texture_slots` of `smv`, by their numbers in it.
"""
function setview(smv::StaticMultiTypeSet, data_slots::NTuple{N, Int}, texture_slots::NTuple{M, Int}) where {N, M}
    return SetView{data_slots, texture_slots}(map(i -> smv.data[i], data_slots), map(i -> smv.textures[i], texture_slots))
end

get_static(v::SetView) = v
n_slots(v::SetView) = length(v.data)

Adapt.adapt_structure(to, v::SetView{DS, TS}) where {DS, TS} =
    SetView{DS, TS}(map(arr -> Adapt.adapt(to, arr), v.data), map(tex -> Adapt.adapt(to, tex), v.textures))

@inline @generated function with_index(f::F, v::SetView{DS}, idx::SetKey, args...) where {F, DS}
    result = :(f(nothing, args...))
    for i in length(DS):-1:1
        result = Expr(:if, :(idx.type_idx === UInt32($(DS[i]))),
                      :(@inbounds f(v.data[$i][idx.vec_idx], args...)), result)
    end
    return quote
        $result
    end
end

@inline @generated function with_texture(f::F, v::SetView{DS, TS}, slot::Integer, idx::Integer, args...) where {F, DS, TS}
    result = :(f(nothing, args...))
    for i in length(TS):-1:1
        result = Expr(:if, :(slot == $(TS[i])), :(@inbounds f(v.textures[$i][idx], args...)), result)
    end
    return quote
        $result
    end
end

@inline @generated function deref(v::SetView{DS, TS}, tref::TextureRef{A, T, N, TIdx}) where {DS, TS, A, T, N, TIdx}
    i = findfirst(==(TIdx), TS)
    i === nothing && return :(error($("deref: texture slot $TIdx is not in the view")))
    return :(@inbounds v.textures[$i][tref.idx])
end
@inline deref(::SetView, arr::AbstractArray) = arr
