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
    original_triangles = to_gpu(ArrayType, bvh.original_triangles)
    return Raycore.BVH(nodes, triangles, original_triangles, bvh.max_node_primitives)
end

gpu_int(x) = Base.unsafe_trunc(Int32, x)
