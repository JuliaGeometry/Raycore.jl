# Hardware-Accelerated Ray Tracing

```@setup hw_accel
using Bonito, BonitoBook, Raycore
book_app = App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src", "hw_acceleration_content.md"))
    BonitoBook.InlineBook(path)
end
Bonito.Page()
```

```@example hw_accel
book_app # hide
```
