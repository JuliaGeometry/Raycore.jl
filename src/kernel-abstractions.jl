import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

function some_kernel(arr)
    backend = KA.get_backend(arr)
    return some_kernel_f(backend)
end

# Get KernelAbstractions backend from an ArrayType
function _array_type_to_backend(ArrayType)
    # Create a small temporary array to get the backend
    tmp = ArrayType{Int}(undef, 1)
    return KA.get_backend(tmp)
end

# Convert array to GPU array
# The caller is responsible for keeping the returned array alive.
# Typically this is done by storing in a scene struct.
function to_gpu(ArrayType, m::AbstractArray)
    arr = ArrayType(m)
    kernel = some_kernel(arr)
    return KA.argconvert(kernel, arr)
end

# GPU conversion for BLAS (instanced BVH bottom-level)
function to_gpu(ArrayType, blas::Raycore.BLAS)
    nodes = to_gpu(ArrayType, blas.nodes)
    primitives = to_gpu(ArrayType, blas.primitives)
    return Raycore.BLAS(nodes, primitives, blas.root_aabb)
end

# GPU conversion for TLAS - use Adapt to create StaticTLAS for kernel traversal
function to_gpu(ArrayType, tlas::Raycore.TLAS)
    # Get the backend from the ArrayType
    backend = _array_type_to_backend(ArrayType)
    # Adapt returns StaticTLAS with isbits arrays for kernel traversal
    return Adapt.adapt(backend, tlas)
end

# Also support StaticTLAS (already GPU-ready, just adapt arrays)
function to_gpu(ArrayType, static_tlas::Raycore.StaticTLAS)
    backend = _array_type_to_backend(ArrayType)
    return Adapt.adapt(backend, static_tlas)
end

gpu_int(x) = Base.unsafe_trunc(Int32, x)
