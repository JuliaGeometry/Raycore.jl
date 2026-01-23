import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

function some_kernel(arr)
    backend = KA.get_backend(arr)
    return some_kernel_f(backend)
end

# Convert array to GPU array
# The caller is responsible for keeping the returned array alive.
# Typically this is done by storing in a scene struct.
function to_gpu(ArrayType, m::AbstractArray)
    arr = ArrayType(m)
    kernel = some_kernel(arr)
    return KA.argconvert(kernel, arr)
end

# GPU conversion for BVH
function to_gpu(ArrayType, bvh::Raycore.BVH)
    nodes = to_gpu(ArrayType, bvh.nodes)
    triangles = to_gpu(ArrayType, bvh.triangles)
    primitives = to_gpu(ArrayType, bvh.primitives)
    return Raycore.BVH(nodes, triangles, primitives, bvh.max_node_primitives)
end

# GPU conversion for BLAS (instanced BVH bottom-level)
function to_gpu(ArrayType, blas::Raycore.BLAS)
    nodes = to_gpu(ArrayType, blas.nodes)
    primitives = to_gpu(ArrayType, blas.primitives)
    return Raycore.BLAS(nodes, primitives, blas.root_aabb)
end

# GPU conversion for TLAS (instanced BVH top-level)
# Convert each BLAS first, then wrap the array of GPU-BLAS in a GPU array
function to_gpu(ArrayType, tlas::Raycore.TLAS)
    nodes = to_gpu(ArrayType, tlas.nodes)
    instances = to_gpu(ArrayType, tlas.instances)
    # Convert each BLAS individually, then convert the array of GPU-BLAS
    blas_gpu = to_gpu(ArrayType, to_gpu.((ArrayType,), tlas.blas_array))
    return Raycore.TLAS(nodes, instances, blas_gpu, tlas.root_aabb)
end

gpu_int(x) = Base.unsafe_trunc(Int32, x)
