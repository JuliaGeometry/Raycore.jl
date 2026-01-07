using Test
using Raycore
using JET
using BenchmarkTools

# ============================================================================
# Test FastClosure capture detection
# ============================================================================

@testset "FastClosure capture detection" begin
    # Regular function - should work
    @test begin
        add(x, y) = x + y
        fc = FastClosure(add, (10,))
        fc(5) == 15
    end

    # Anonymous function without capture - should work
    @test begin
        fc = FastClosure((x, y) -> x * y, (3,))
        fc(4) == 12
    end

    # Closure WITH capture (in local scope) - should error
    # Note: captures only become fields when created in local function scope
    function make_capturing_closure()
        captured = 42
        return x -> x + captured
    end
    @test_throws ErrorException FastClosure(make_capturing_closure(), ())

    # Closure with boxed capture (reassigned after closure creation) - should error
    function make_boxed_closure()
        captured = 42
        closure = x -> x + captured
        captured = 100  # reassignment causes Core.Box
        return closure
    end
    @test_throws ErrorException FastClosure(make_boxed_closure(), ())
end

# ============================================================================
# Test for_unrolled
# ============================================================================

@testset "for_unrolled with tuple" begin
    # Basic iteration
    @test begin
        results = Int[]
        push_val!(x, arr) = push!(arr, x)
        for_unrolled(push_val!, (1, 2, 3), results)
        results == [1, 2, 3]
    end

    # Heterogeneous tuple
    @test begin
        results = Any[]
        push_val!(x, arr) = push!(arr, x)
        for_unrolled(push_val!, (1, "hello", 3.14), results)
        results == [1, "hello", 3.14]
    end

    # Empty tuple
    @test begin
        count = Ref(0)
        inc!(x, c) = c[] += 1
        for_unrolled(inc!, (), count)
        count[] == 0
    end

    # Multiple extra args
    @test begin
        results = Float64[]
        scaled_push!(x, arr, scale, offset) = push!(arr, x * scale + offset)
        for_unrolled(scaled_push!, (1, 2, 3), results, 2.0, 0.5)
        results == [2.5, 4.5, 6.5]
    end
end

@testset "for_unrolled with Val{N}" begin
    @test begin
        results = Int32[]
        push_val!(i, arr) = push!(arr, i)
        for_unrolled(push_val!, Val(5), results)
        results == Int32[1, 2, 3, 4, 5]
    end

    @test begin
        results = Int32[]
        push_val!(i, arr) = push!(arr, i)
        for_unrolled(push_val!, Val(0), results)
        isempty(results)
    end
end

# ============================================================================
# Test map_unrolled
# ============================================================================

@testset "map_unrolled" begin
    # Basic mapping
    @test begin
        double(x) = 2x
        map_unrolled(double, (1, 2, 3)) == (2, 4, 6)
    end

    # With extra args
    @test begin
        scale(x, factor) = x * factor
        map_unrolled(scale, (1, 2, 3), 10) == (10, 20, 30)
    end

    # Heterogeneous tuple - returns heterogeneous result
    @test begin
        wrap(x) = [x]
        result = map_unrolled(wrap, (1, "a", 3.0))
        result == ([1], ["a"], [3.0])
    end

    # Empty tuple
    @test map_unrolled(identity, ()) == ()

    # Type-changing map
    @test begin
        to_string(x) = string(x)
        map_unrolled(to_string, (1, 2, 3)) == ("1", "2", "3")
    end
end

# ============================================================================
# Test reduce_unrolled
# ============================================================================

@testset "reduce_unrolled" begin
    # Sum reduction
    @test begin
        add_to_acc(acc, x) = acc + x
        reduce_unrolled(add_to_acc, (1, 2, 3, 4), 0) == 10
    end

    # With extra args
    @test begin
        scaled_add(acc, x, scale) = acc + x * scale
        reduce_unrolled(scaled_add, (1, 2, 3), 0, 2) == 12  # 2 + 4 + 6
    end

    # Product reduction
    @test begin
        mul_to_acc(acc, x) = acc * x
        reduce_unrolled(mul_to_acc, (1, 2, 3, 4), 1) == 24
    end

    # Empty tuple returns init
    @test begin
        add_to_acc(acc, x) = acc + x
        reduce_unrolled(add_to_acc, (), 42) == 42
    end

    # Collecting into array
    @test begin
        collect_acc(acc, x) = push!(copy(acc), x)
        reduce_unrolled(collect_acc, (1, 2, 3), Int[]) == [1, 2, 3]
    end
end

# ============================================================================
# Test sum_unrolled
# ============================================================================

@testset "sum_unrolled" begin
    # Direct sum
    @test begin
        identity_val(x) = x
        sum_unrolled(identity_val, (1, 2, 3, 4)) == 10
    end

    # With transformation
    @test begin
        square(x) = x^2
        sum_unrolled(square, (1, 2, 3)) == 14  # 1 + 4 + 9
    end

    # With extra args
    @test begin
        scaled(x, factor) = x * factor
        sum_unrolled(scaled, (1, 2, 3), 2) == 12  # 2 + 4 + 6
    end

    # Single element
    @test begin
        identity_val(x) = x
        sum_unrolled(identity_val, (42,)) == 42
    end

    # Empty tuple returns nothing
    @test begin
        identity_val(x) = x
        sum_unrolled(identity_val, ()) === nothing
    end

    # Float accumulation
    @test begin
        identity_val(x) = x
        sum_unrolled(identity_val, (1.0, 2.0, 3.0)) ≈ 6.0
    end
end

# ============================================================================
# Test compiler limits checking
# ============================================================================

@testset "Compiler limits" begin
    # Too many args should error
    @test_throws ErrorException begin
        f(x) = x
        big_args = ntuple(i -> i, 40)  # 40 > MAX_TUPLE_LENGTH
        FastClosure(f, big_args)
    end

    # Tuple arg that's too long should error
    @test_throws ErrorException begin
        f(x, _big_tuple) = x
        big_tuple = ntuple(i -> i, 40)
        FastClosure(f, (big_tuple,))
    end
end

# ============================================================================
# Type stability and allocation tests with JET
# ============================================================================

"""
    @test_opt_alloc expr

Combined macro that tests both type stability (via @test_opt) and zero allocations.
Uses BenchmarkTools for reliable allocation measurement.
"""
macro test_opt_alloc(expr)
    return esc(quote
        JET.@test_opt $expr
        b = @benchmarkable $expr
        tune!(b)
        result = run(b)
        @test result.allocs == 0
    end)
end

# Complex nested types for testing
struct Light{T}
    intensity::T
    position::Point3f
end

struct Material
    albedo::Vec3f
    roughness::Float32
end

# Test functions that will be used with unrolled iteration
add_intensity(light::Light, scale::Float32) = light.intensity * scale
compute_contribution(acc::Float32, light::Light, factor::Float32) = acc + light.intensity * factor
transform_light(light::Light, offset::Vec3f) = Light(light.intensity, light.position + offset)

@testset "Type stability: for_unrolled" begin
    # Simple tuple
    results = Float32[]
    push_f32!(x::Float32, arr) = push!(arr, x)
    @test_opt for_unrolled(push_f32!, (1.0f0, 2.0f0, 3.0f0), results)

    # Heterogeneous tuple of lights
    lights = (
        Light(1.0f0, Point3f(0, 0, 1)),
        Light(2.0f0, Point3f(1, 0, 0)),
        Light(0.5f0, Point3f(0, 1, 0)),
    )
    results2 = Float32[]
    push_intensity!(light::Light, arr) = push!(arr, light.intensity)
    @test_opt for_unrolled(push_intensity!, lights, results2)
end

@testset "Type stability: map_unrolled" begin
    # Simple transformation
    double(x::Float32) = 2.0f0 * x
    @test_opt_alloc map_unrolled(double, (1.0f0, 2.0f0, 3.0f0))

    # With extra args
    scale_val(x::Float32, factor::Float32) = x * factor
    @test_opt_alloc map_unrolled(scale_val, (1.0f0, 2.0f0, 3.0f0), 10.0f0)

    # Complex types - lights
    lights = (
        Light(1.0f0, Point3f(0, 0, 1)),
        Light(2.0f0, Point3f(1, 0, 0)),
    )
    @test_opt_alloc map_unrolled(add_intensity, lights, 2.0f0)
end

@testset "Type stability: reduce_unrolled" begin
    # Simple sum
    add_f32(acc::Float32, x::Float32) = acc + x
    @test_opt_alloc reduce_unrolled(add_f32, (1.0f0, 2.0f0, 3.0f0), 0.0f0)

    # With extra args
    scaled_add_f32(acc::Float32, x::Float32, scale::Float32) = acc + x * scale
    @test_opt_alloc reduce_unrolled(scaled_add_f32, (1.0f0, 2.0f0, 3.0f0), 0.0f0, 2.0f0)

    # Complex types - accumulate light contributions
    lights = (
        Light(1.0f0, Point3f(0, 0, 1)),
        Light(2.0f0, Point3f(1, 0, 0)),
        Light(0.5f0, Point3f(0, 1, 0)),
    )
    @test_opt_alloc reduce_unrolled(compute_contribution, lights, 0.0f0, 1.5f0)
end

@testset "Type stability: sum_unrolled" begin
    # Simple identity sum
    identity_f32(x::Float32) = x
    @test_opt_alloc sum_unrolled(identity_f32, (1.0f0, 2.0f0, 3.0f0))

    # With transformation
    square_f32(x::Float32) = x * x
    @test_opt_alloc sum_unrolled(square_f32, (1.0f0, 2.0f0, 3.0f0))

    # With extra args
    scaled_f32(x::Float32, factor::Float32) = x * factor
    @test_opt_alloc sum_unrolled(scaled_f32, (1.0f0, 2.0f0, 3.0f0), 2.0f0)

    # Complex types - sum light intensities
    lights = (
        Light(1.0f0, Point3f(0, 0, 1)),
        Light(2.0f0, Point3f(1, 0, 0)),
    )
    get_intensity(light::Light) = light.intensity
    @test_opt_alloc sum_unrolled(get_intensity, lights)
end

@testset "Type stability: deeply nested complex types" begin
    # Tuple containing vectors and nested structs
    struct SceneObject
        transform::Mat4f
        material::Material
        bounds::Rect3f
    end

    obj1 = SceneObject(
        Mat4f(I),
        Material(Vec3f(1, 0, 0), 0.5f0),
        Rect3f(Point3f(0), Vec3f(1))
    )
    obj2 = SceneObject(
        Mat4f(I),
        Material(Vec3f(0, 1, 0), 0.3f0),
        Rect3f(Point3f(1), Vec3f(2))
    )
    obj3 = SceneObject(
        Mat4f(I),
        Material(Vec3f(0, 0, 1), 0.7f0),
        Rect3f(Point3f(-1), Vec3f(1))
    )

    objects = (obj1, obj2, obj3)

    # Extract roughness values
    get_roughness(obj::SceneObject) = obj.material.roughness
    @test_opt_alloc map_unrolled(get_roughness, objects)

    # Sum roughness
    @test_opt_alloc sum_unrolled(get_roughness, objects)

    # Reduce with complex accumulator
    struct AccumBounds
        combined::Rect3f
        count::Int32
    end

    merge_bounds(acc::AccumBounds, obj::SceneObject) = AccumBounds(
        Rect3f(
            min.(acc.combined.origin, obj.bounds.origin),
            max.(acc.combined.widths, obj.bounds.widths)
        ),
        acc.count + Int32(1)
    )

    init_acc = AccumBounds(Rect3f(Point3f(Inf32), Vec3f(0)), Int32(0))
    @test_opt reduce_unrolled(merge_bounds, objects, init_acc)
end

@testset "Type stability: tuple with array references" begin
    # Tuples containing references to arrays (common in GPU code)
    arr1 = [1.0f0, 2.0f0, 3.0f0]
    arr2 = [4.0f0, 5.0f0, 6.0f0]
    arr3 = [7.0f0, 8.0f0, 9.0f0]

    arrays = (arr1, arr2, arr3)

    # Sum first elements
    get_first(arr::Vector{Float32}) = arr[1]
    @test_opt sum_unrolled(get_first, arrays)

    # Map to lengths (this allocates because length returns Int which gets boxed in tuple)
    # get_length(arr::Vector{Float32}) = length(arr)
    # @test_opt map_unrolled(get_length, arrays)
end
