@testset "Ray-Bounds intersection" begin
    b = Raycore.Bounds3(Point3f(1), Point3f(2))
    b_neg = Raycore.Bounds3(Point3f(-2), Point3f(-1))
    r0 = Raycore.Ray(o = Point3f(0), d = Vec3f(1, 0, 0))
    r1 = Raycore.Ray(o = Point3f(0), d = Vec3f(1))
    ri = Raycore.Ray(o = Point3f(1.5), d = Vec3f(1, 1, 0))

    r, t0, t1 = Raycore.intersect(b, r1)
    @test r && t0 ≈ 1f0 && t1 ≈ 2f0
    r, t0, t1 = Raycore.intersect(b, r0)
    @test !r && t0 ≈ 0f0 && t1 ≈ 0f0
    r, t0, t1 = Raycore.intersect(b, ri)
    @test r && t0 ≈ 0f0 && t1 ≈ 0.5f0

    # Test intersection with precomputed direction reciprocal.
    inv_dir = 1f0 ./ r1.d
    dir_is_negative = Raycore.is_dir_negative(r1.d)
    @test Raycore.intersect_p(b, r1, inv_dir, dir_is_negative)
    @test !Raycore.intersect_p(b_neg, r1, inv_dir, dir_is_negative)
end

@testset "Test triangle" begin
    # Construct Triangle directly (no TriangleMesh)
    triangle = Raycore.Triangle(
        SVector(Point3f(0, 0, 2), Point3f(1, 0, 2), Point3f(1, 1, 2)),
        SVector(Raycore.Normal3f(0, 0, -1), Raycore.Normal3f(0, 0, -1), Raycore.Normal3f(0, 0, -1)),
        SVector(Vec3f(NaN), Vec3f(NaN), Vec3f(NaN)),
        SVector(Point2f(0, 0), Point2f(1, 0), Point2f(1, 1)),
        nothing
    )

    tv = Raycore.vertices(triangle)
    a = norm(tv[1] - tv[2])^2 * 0.5f0
    @test Raycore.area(triangle) ≈ a

    target_wb = Raycore.Bounds3(Point3f(0, 0, 2), Point3f(1, 1, 2))
    @test Raycore.object_bound(triangle) ≈ target_wb

    # Test ray intersection
    ray = Raycore.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    intersects_p = Raycore.intersect_p(triangle, ray)
    intersects, t_hit, bary_coords = Raycore.intersect(triangle, ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 4f0
    @test Raycore.apply(ray, t_hit) ≈ Point3f(0, 0, 2)
    @test bary_coords ≈ Point3f(1, 0, 0)

    # Test ray intersection (different point).
    ray = Raycore.Ray(o = Point3f(0.5, 0.25, 0), d = Vec3f(0, 0, 1))
    intersects_p = Raycore.intersect_p(triangle, ray)
    intersects, t_hit, bary_coords = Raycore.intersect(triangle, ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 2f0
    @test Raycore.apply(ray, t_hit) ≈ Point3f(0.5, 0.25, 2)
end

@testset "TLAS with triangle meshes" begin
    using GeometryBasics
    using LinearAlgebra

    # Create simple triangle meshes as GB.Mesh at different positions
    function make_gb_mesh(offset::Vec3f=Vec3f(0, 0, 0))
        verts = [Point3f(0, 0, 0) + offset, Point3f(1, 0, 0) + offset, Point3f(1, 1, 0) + offset]
        faces = [GLTriangleFace(1, 2, 3)]
        normals = [Raycore.Normal3f(0, 0, -1), Raycore.Normal3f(0, 0, -1), Raycore.Normal3f(0, 0, -1)]
        GeometryBasics.mesh(verts, faces; normal=normals)
    end

    meshes = [make_gb_mesh(Vec3f(i*3, i*3, 0)) for i in 0:3]

    tlas = Raycore.TLAS(meshes, (mesh_idx, tri_idx) -> UInt32(mesh_idx))
    @test !isnothing(Raycore.world_bound(tlas))

    # Simple intersection test
    ray = Raycore.Ray(o = Point3f(0.5, 0.5, -1), d = Vec3f(0, 0, 1))
    hit, tri, dist, bary, inst_id = Raycore.closest_hit(tlas, ray)
    @test hit
    @test tri isa Raycore.Triangle
end

@testset "TLAS with triangle meshes in a row" begin
    using GeometryBasics
    using LinearAlgebra

    function make_gb_mesh_at_z(z::Float32)
        verts = [Point3f(-1, -1, z), Point3f(1, -1, z), Point3f(0, 1, z)]
        faces = [GLTriangleFace(1, 2, 3)]
        normals = [Raycore.Normal3f(0, 0, -1), Raycore.Normal3f(0, 0, -1), Raycore.Normal3f(0, 0, -1)]
        GeometryBasics.mesh(verts, faces; normal=normals)
    end

    meshes = [make_gb_mesh_at_z(Float32(z)) for z in [0, 4, 8]]

    tlas = Raycore.TLAS(meshes, (mesh_idx, tri_idx) -> UInt32(mesh_idx))
    bound = Raycore.world_bound(tlas)
    @test !isnothing(bound)

    # Test intersection with the first triangle
    ray = Raycore.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    hit, tri, dist, bary, inst_id = Raycore.closest_hit(tlas, ray)
    @test hit
    @test tri isa Raycore.Triangle
end

@testset "empty_triangle" begin
    e = Raycore.empty_triangle(Raycore.Triangle{UInt32})
    @test e isa Raycore.Triangle{UInt32}
    @test all(v -> all(iszero, v), e.vertices)
    @test all(n -> all(iszero, n), e.normals)
    @test all(t -> all(iszero, t), e.tangents)
    @test all(u -> all(iszero, u), e.uv)
    @test e.metadata == zero(UInt32)

    # Works for arbitrary metadata types that have `zero(T)`.
    e2 = Raycore.empty_triangle(Raycore.Triangle{Int32})
    @test e2 isa Raycore.Triangle{Int32}
    @test e2.metadata == zero(Int32)
end

@testset "closest_hit no-hit returns empty_triangle sentinel" begin
    using GeometryBasics

    function make_unit_mesh()
        verts = [Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0)]
        faces = [GLTriangleFace(1,2,3)]
        normals = [Raycore.Normal3f(0,0,1), Raycore.Normal3f(0,0,1), Raycore.Normal3f(0,0,1)]
        GeometryBasics.mesh(verts, faces; normal=normals)
    end

    tlas = Raycore.TLAS([make_unit_mesh()], (mesh_idx, tri_idx) -> UInt32(mesh_idx))

    # Ray clearly misses: origin far away, direction pointing further away
    ray = Raycore.Ray(o = Point3f(100, 100, 100), d = Vec3f(1, 0, 0))
    hit, tri, _, _, _ = Raycore.closest_hit(tlas, ray)

    @test !hit
    @test tri isa Raycore.Triangle
    # Returned sentinel must be the zero triangle, not a storage-indexed triangle
    Tri = eltype(tlas.all_blas_prims)
    @test tri == Raycore.empty_triangle(Tri)
    @test all(v -> all(iszero, v), tri.vertices)
end
