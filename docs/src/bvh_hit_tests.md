# BVH Hit tests

```@setup raytracing
using Bonito
Bonito.Page()
```

```@example raytracing
using Bonito, BonitoBook, Raycore
App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src", "bvh_hit_tests_content.md"))
    BonitoBook.InlineBook(path)
end
```
