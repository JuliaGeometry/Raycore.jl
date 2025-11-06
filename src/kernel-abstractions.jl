import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

function some_kernel(arr)
    backend = KA.get_backend(arr)
    return some_kernel_f(backend)
end

function to_gpu(ArrayType, m::AbstractArray; preserve=[])
    arr = ArrayType(m)
    push!(preserve, arr)
    kernel = some_kernel(arr)
    return KA.argconvert(kernel, arr)
end

# GPU conversion for BVH
function to_gpu(ArrayType, bvh::Raycore.BVH; preserve=[])
    nodes = to_gpu(ArrayType, bvh.nodes; preserve=preserve)
    triangles = to_gpu(ArrayType, bvh.triangles; preserve=preserve)
    original_triangles = to_gpu(ArrayType, bvh.original_triangles; preserve=preserve)
    return Raycore.BVH(nodes, triangles, original_triangles, bvh.max_node_primitives)
end
