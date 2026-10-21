using Test, Raycore, Adapt, KernelAbstractions

@kernel function read_tuple_textures!(out, set, fields)
    i = @index(Global)
    @inbounds out[i] = fields.terms[1][2] * Raycore.deref(set, fields.terms[1][1])[i] +
        fields.named.gain * Raycore.deref(set, fields.named.density)[i]
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
    end
end

test_tuple_textures(test_backend())
