using Documenter
using DocumenterVitepress
using RayCaster

makedocs(; sitename = "RayCaster", authors = "Anton Smirnov, Simon Danisch and contributors",
    modules = [RayCaster],
    checkdocs = :all,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaGeometry/RayCaster.jl", # this must be the full URL!
        devbranch = "master",
        devurl = "dev";
    ),
    draft = false,
    source = "src",
    build = "build",
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Shadows" => "shadows.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaGeometry/RayCaster.jl",
    target = "build", # this is where Vitepress stores its output
    branch = "gh-pages",
    devbranch = "master",
    push_preview = true,
)
