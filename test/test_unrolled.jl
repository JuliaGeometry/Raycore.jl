using Test
using Raycore

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

