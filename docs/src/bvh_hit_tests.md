# BVH Hit tests

```@setup raytracing
using Bonito, BonitoBook, Raycore
book_app = App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src",  "bvh_hit_tests_content.md"))
    BonitoBook.InlineBook(path)
end
Bonito.Page()
```

```@example raytracing
book_app # hide
```
