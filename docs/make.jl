using Documenter
using Raycore
using Bonito
using BonitoBook

makedocs(;
    modules = [Raycore],
    sitename = "Raycore",
    clean = false,
    format=Documenter.HTML(;
        prettyurls=false,
        size_threshold=3000000,
        example_size_threshold=3000000
    ),
    authors = "Anton Smirnov, Simon Danisch and contributors",
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "BVH Hit Tests" => "bvh_hit_tests.md",
            "Ray Tracing Tutorial" => "raytracing_tutorial.md",
            "View Factors and More" => "viewfactors.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/JuliaGeometry/Raycore.jl",
    push_preview = true,
)
