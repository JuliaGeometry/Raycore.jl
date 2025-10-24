using Documenter
using RayCaster
using Bonito
using BonitoBook

makedocs(;
    modules = [RayCaster],
    sitename = "RayCaster",
    clean = false,
    format=Documenter.HTML(;
        prettyurls=false,
        size_threshold=3000000,
        example_size_threshold=3000000
    ),
    authors = "Anton Smirnov, Simon Danisch and contributors",
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaGeometry/RayCaster.jl",
    push_preview = true,
)
