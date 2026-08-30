"""
Raycore on Metal: the one device intrinsic the BVH refit needs.

`Raycore.bvh_publish_fence` defaults to a no-op, which is right for `KA.CPU()`
and wrong for every GPU. This supplies Metal's.

Why it cannot live in `src/`: the fence is a Metal intrinsic and Raycore does
not depend on Metal. Why it cannot live in Metal.jl: Metal does not depend on
Raycore. An extension is the only place both names are in scope, and putting it
HERE rather than in a renderer means anyone who loads Raycore and Metal gets a
correct BVH — the alternative silently reintroduces the race for a direct
Raycore user.
"""
module RaycoreMetalExt

using Raycore, Metal

# seq_cst rather than acq_rel: MSL gained acquire/release fences only in Metal
# 4.1, and this machine reports 4.0. seq_cst is available from 3.2 and is
# strictly stronger, so it is the portable spelling across Metal versions
# rather than a preference.
#
# The device flag matters as much as the ordering — `MemoryFlagNone` would
# fence nothing, and the BVH nodes being published live in device memory.
Metal.@device_override @inline function Raycore.bvh_publish_fence()
    Metal.atomic_thread_fence(Val(Metal.MemoryFlagDevice),
                              Val(Metal.memory_order_seq_cst),
                              Val(Metal.thread_scope_device))
end

end
