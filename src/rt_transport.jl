# ============================================================================
# RT transport structs — shared by all hardware RT backends
# ============================================================================

"""
    RTRay

Ray input for hardware RT dispatch. 32 bytes, matches Vulkan/Metal layout.
"""
struct RTRay
    origin_x::Float32
    origin_y::Float32
    origin_z::Float32
    tmin::Float32
    dir_x::Float32
    dir_y::Float32
    dir_z::Float32
    tmax::Float32
end

"""
    RTHitResult

Ray hit output from hardware RT dispatch. 32 bytes.

- `instance_custom_index` — value of `gl_InstanceCustomIndexEXT` at the hit.
  Under the current semantics this carries the `InstanceDescriptor.instance_id`
  (the interface-override slot).  `0` means "inherit from triangle metadata".
- `instance_id` — value of `gl_InstanceID` at the hit (0-based instance array
  position).  Used by the caller to look up per-instance data such as the
  BLAS triangle offset.
"""
struct RTHitResult
    hit::UInt32
    t::Float32
    primitive_id::UInt32
    instance_custom_index::UInt32
    bary_u::Float32
    bary_v::Float32
    instance_id::UInt32
    _pad2::UInt32
end
