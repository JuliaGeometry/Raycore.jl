# View Factors and More

```@setup viewfactors_wrapper
using Bonito
Bonito.Page()
```

```@example viewfactors_wrapper
using Bonito, BonitoBook, RayCore
App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src", "viewfactors_content.md"))
    BonitoBook.InlineBook(path)
end
```
