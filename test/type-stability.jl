using RayCaster, GeometryBasics, StaticArrays

code_warntype(RayCaster._to_ray_coordinate_space, (SVector{3,Point3f}, RayCaster.Ray))
code_warntype(RayCaster.partial_derivatives, (RayCaster.Triangle, SVector{3,Point3f}, SVector{3,Point2f}))
code_warntype(RayCaster.normal_derivatives, (RayCaster.Triangle, SVector{3,Point2f}))
code_warntype(RayCaster.intersect, (RayCaster.Triangle, RayCaster.Ray, Bool))
code_warntype(RayCaster.intersect_triangle, (RayCaster.Triangle, RayCaster.Ray))
code_warntype(RayCaster.intersect_triangle, (RayCaster.Triangle, RayCaster.Ray))
code_warntype(RayCaster.intersect_p, (RayCaster.Triangle, RayCaster.Ray))

##########################
##########################
##########################
# Random benchmarks
v1 = Vec3f(0.0, 0.0, 0.0)
v2 = Vec3f(1.0, 0.0, 0.0)
v3 = Vec3f(0.0, 1.0, 0.0)

ray_origin = Vec3f(0.5, 0.5, 1.0)
ray_direction = Vec3f(0.0, 0.0, -1.0)

using RayCaster: Normal3f
m = RayCaster.create_triangle_mesh(RayCaster.ShapeCore(), UInt32[1, 2, 3], Point3f[v1, v2, v3], [Normal3f(0.0, 0.0, 1.0), Normal3f(0.0, 0.0, 1.0), Normal3f(0.0, 0.0, 1.0)])

t = RayCaster.Triangle(m, 1)
r = RayCaster.Ray(o=Point3f(ray_origin), d=ray_direction)
RayCaster.intersect_p(t, r)
@code_warntype RayCaster.intersect_triangle(t.vertices, r)
