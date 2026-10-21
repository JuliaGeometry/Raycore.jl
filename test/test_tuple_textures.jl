using Test, Raycore, Adapt, KernelAbstractions

@kernel function read_tuple_textures!(out, set, fields)
    i = @index(Global)
    @inbounds out[i] = fields.terms[1][2] * Raycore.deref(set, fields.terms[1][1])[i] +
        fields.named.gain * Raycore.deref(set, fields.named.density)[i]
end

struct TupleTextureRecord{T,N}
    terms::T
    named::N
    gain::Float32
end

function test_tuple_textures(backend)
    @testset "Nested tuples retain texture fields and names" begin
        set = Raycore.MultiTypeSet(backend)
        values = Float32[1, 2, 4]
        input = (terms = ((values, 2f0),), named = (density = values, gain = 3f0))
        converted = Raycore.maybe_convert_field(set, input)
        @test keys(converted) == keys(input)
        @test keys(converted.named) == keys(input.named)
        @test converted.terms[1][1] isa Raycore.TextureRef
        @test converted.named.density isa Raycore.TextureRef
        @test converted.terms[1][2] == 2f0
        @test converted.named.gain == 3f0
        output = KernelAbstractions.zeros(backend, Float32, length(values))
        read_tuple_textures!(backend, 64)(output, Adapt.adapt(backend, set.static), converted;
            ndrange = length(values))
        @test Array(output) == 5f0 .* values

        # Exercise the reflective host fallback as well as tuple conversion:
        # native volume records have nested tuples, texture slots and scalars.
        for device_source in (false,true)
            source = device_source ? Adapt.adapt(backend,copy(values)) : copy(values)
            records = Raycore.MultiTypeSet(backend)
            key = push!(records,TupleTextureRecord(((source,2f0),),(;density=source,gain=3f0),1f0))
            old = records[key]
            slots = length(records.texture_gpu_arrays)
            replacement = Float32[1,3,5,7,9]
            fresh = TupleTextureRecord(((replacement,2f0),),
                (;density=replacement,gain=4f0),2.5f0)
            Raycore.update!(records,key,fresh)
            updated = records[key]
            @test typeof(updated)===typeof(old)
            @test updated.gain==2.5f0
            @test updated.terms[1][1]===old.terms[1][1]
            @test updated.named.density===old.named.density
            @test length(records.texture_gpu_arrays)==slots
            output = KernelAbstractions.zeros(backend,Float32,length(replacement))
            read_tuple_textures!(backend,64)(output,Adapt.adapt(backend,records.static),updated;
                ndrange=length(replacement))
            @test Array(output)==6f0.*replacement
        end
    end
end

test_tuple_textures(test_backend())
