module RaycoreLavaExt

using Raycore
using Lava

using Base: @propagate_inbounds
import Raycore: supports_indirect_dispatch, indirect_ndrange,
                build_hw_blas, build_hw_tlas,
                trace_closest_hits!, trace_closest_hits_indirect!,
                batch_trace_indirect, mat4_to_transform_matrix,
                set_custom_anyhit!,
                rt_primitive_id, rt_instance_custom_index,
                rt_launch_id_x, rt_global_invocation_id_x,
                rt_ignore_intersection, rt_terminate_ray,
                rt_payload_store!, rt_payload_load, rt_trace_ray!,
                closest_hit, AbstractAdaptedAccel,
                HWTLAS, HWAdaptedAccel, RTRay, RTHitResult
import Adapt
import KernelAbstractions
using StaticArrays: SVector, SMatrix

using Lava: LavaBackend, LavaArray, LavaBLAS, LavaTLAS,
            lava_global_invocation_id_x,
            lava_rt_primitive_id, lava_rt_instance_custom_index,
            lava_rt_launch_id_x,
            _lava_rt_ignore_intersection, _lava_rt_terminate_ray,
            _lava_rt_payload_store_f32_at, _lava_rt_payload_load_f32_at,
            _lava_rt_trace_ray,
            _mat4_to_vk_transform

const LavaHWTLAS = HWTLAS{LavaBackend}
const LavaHWAdapted = HWAdaptedAccel{<:LavaHWTLAS}

# ============================================================================
# Indirect dispatch (Lava supports GPU-resident ndrange)
# ============================================================================

Raycore.supports_indirect_dispatch(::LavaBackend) = true
Raycore.indirect_ndrange(size_buf::LavaArray) = size_buf

# ============================================================================
# BLAS/TLAS construction
# ============================================================================

Raycore.build_hw_blas(::LavaBackend, vertices, indices) = Lava.build_blas(vertices, indices)

function Raycore.build_hw_tlas(::LavaBackend, blas_refs, blas_triangles, blas_offsets;
                                transforms, custom_indices)
    hw_tlas = Lava.build_tlas(blas_refs; transforms, custom_indices)

    all_tris_any = reduce(vcat, blas_triangles)
    T = typeof(all_tris_any[1])
    all_tris = T[t for t in all_tris_any]
    hw_accel = Lava.HardwareAccel(hw_tlas, all_tris, blas_offsets)

    tri_gpu = LavaArray(all_tris)
    off_gpu = LavaArray(blas_offsets)

    return (hw_tlas, hw_accel, tri_gpu, off_gpu)
end

Raycore.mat4_to_transform_matrix(m::SMatrix{4,4,Float32,16}) = _mat4_to_vk_transform(m)

# ============================================================================
# Batch trace dispatch
# ============================================================================

function Raycore.trace_closest_hits!(results, rays, accel::LavaHWAdapted, n)
    Lava.trace_closest_hits!(results, rays, accel.hwtlas.hw_accel, n)
end

function Raycore.trace_closest_hits_indirect!(results, rays, accel::LavaHWAdapted, n_buf)
    Lava.trace_closest_hits_indirect!(results, rays, accel.hwtlas.hw_accel, n_buf)
end

function Raycore.batch_trace_indirect(results, rays, accel::LavaHWAdapted, n_buf)
    Lava.trace_closest_hits_indirect!(results, rays, accel.hwtlas.hw_accel, n_buf)
    return PrecomputedHitsAccel(results, accel.hwtlas.tri_gpu, accel.hwtlas.off_gpu)
end

# ============================================================================
# Custom shader pipeline
# ============================================================================

function Raycore.set_custom_anyhit!(accel::LavaHWAdapted, anyhit_fn, raygen_fn)
    hw = accel.hwtlas.hw_accel
    hw === nothing && error("HWTLAS not synced")
    Lava.set_anyhit_pipeline!(hw, anyhit_fn, raygen_fn)
end

# ============================================================================
# PrecomputedHitsAccel - wraps batch-traced results for closest_hit dispatch
# ============================================================================

struct PrecomputedHitsAccel{R, T, O} <: AbstractAdaptedAccel
    results::R
    triangles::T
    offsets::O
end

function Adapt.adapt_structure(to, p::PrecomputedHitsAccel)
    PrecomputedHitsAccel(
        Adapt.adapt(to, p.results),
        Adapt.adapt(to, p.triangles),
        Adapt.adapt(to, p.offsets),
    )
end

@propagate_inbounds function Raycore.closest_hit(accel::PrecomputedHitsAccel, ray)
    tid = lava_global_invocation_id_x() + UInt32(1)
    result = accel.results[tid]

    if result.hit == UInt32(0)
        dummy = accel.triangles[1]
        return (false, dummy, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0))
    end

    tri_idx = Int(accel.offsets[result.instance_custom_index + UInt32(1)]) +
              Int(result.primitive_id) + 1
    tri = accel.triangles[tri_idx]

    w = 1f0 - result.bary_u - result.bary_v
    bary = SVector{3,Float32}(w, result.bary_u, result.bary_v)

    return (true, tri, result.t, bary)
end

# ============================================================================
# RT shader intrinsics -> Lava intrinsics
# ============================================================================

Raycore.rt_primitive_id(::LavaHWAdapted) = lava_rt_primitive_id()
Raycore.rt_instance_custom_index(::LavaHWAdapted) = lava_rt_instance_custom_index()
Raycore.rt_launch_id_x(::LavaHWAdapted) = lava_rt_launch_id_x()
Raycore.rt_global_invocation_id_x(::LavaHWAdapted) = lava_global_invocation_id_x()
Raycore.rt_ignore_intersection(::LavaHWAdapted) = _lava_rt_ignore_intersection()
Raycore.rt_terminate_ray(::LavaHWAdapted) = _lava_rt_terminate_ray()
Raycore.rt_payload_store!(::LavaHWAdapted, val, slot) = _lava_rt_payload_store_f32_at(val, slot)
Raycore.rt_payload_load(::LavaHWAdapted, slot) = _lava_rt_payload_load_f32_at(slot)

function Raycore.rt_trace_ray!(::LavaHWAdapted, flags, mask, sbt_offset, sbt_stride, miss_idx,
                                ox, oy, oz, tmin, dx, dy, dz, tmax)
    _lava_rt_trace_ray(flags, mask, sbt_offset, sbt_stride, miss_idx,
                       ox, oy, oz, tmin, dx, dy, dz, tmax)
end

end
