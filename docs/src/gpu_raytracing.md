# GPU Ray Tracing with Raycore

```@setup gpu_raytracing
using Bonito, BonitoBook, Raycore
book_app = App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src", "gpu_raytracing_tutorial.md"))
    BonitoBook.InlineBook(path)
end
Bonito.Page()
```

```@example gpu_raytracing
book_app # hide
```
