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

# Conversion constructor for e.g. GPU arrays
# TODO, create tree on GPU? Not sure if that will gain much though...
function to_gpu(ArrayType, bvh::RayCaster.BVHAccel; preserve=[])
    primitives = to_gpu(ArrayType, bvh.primitives; preserve=preserve)
    nodes = to_gpu(ArrayType, bvh.nodes; preserve=preserve)
    return RayCaster.BVHAccel(primitives, bvh.max_node_primitives, nodes)
end
