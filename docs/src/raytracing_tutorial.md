# Ray Tracing with Raycore

```@setup raytracing
using Bonito, BonitoBook, Raycore
book_app = App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src", "raytracing_tutorial_content.md"))
    BonitoBook.InlineBook(path)
end
Bonito.Page()
```

```@example raytracing
book_app # hide
```
