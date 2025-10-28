# Ray Tracing with Raycore

```@setup raytracing
using Bonito
Bonito.Page()
```

```@example raytracing
using Bonito, BonitoBook, Raycore
App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src", "raytracing_tutorial_content.md"))
    BonitoBook.InlineBook(path)
end
```
