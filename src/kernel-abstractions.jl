import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

global PRESERVE = []

function some_kernel(arr)
    backend = KA.get_backend(arr)
    return some_kernel_f(backend)
end

function to_gpu(ArrayType, m::AbstractArray)
    arr = ArrayType(m)
    push!(PRESERVE, arr)
    finalizer((arr) -> filter!(x-> x === arr, PRESERVE), arr)
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
function to_gpu(ArrayType, tlas::Raycore.TLAS)
    nodes = to_gpu(ArrayType, tlas.nodes)
    instances = to_gpu(ArrayType, tlas.instances)
    # Convert each BLAS in the array
    blas_gpu = [to_gpu(ArrayType, b) for b in tlas.blas_array]
    # We need a concrete array type for the BLAS array on GPU
    # Since BLAS contains GPU arrays, we keep it as a regular Vector
    # but the inner arrays are on GPU
    return Raycore.TLAS(nodes, instances, blas_gpu, tlas.root_aabb)
end

gpu_int(x) = Base.unsafe_trunc(Int32, x)
