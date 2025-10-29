# View Factors and More

```@setup viewfactors_wrapper
using Bonito, BonitoBook, Raycore
book_app = App() do
    path = normpath(joinpath(dirname(pathof(Raycore)), "..", "docs", "src", "viewfactors_content.md"))
    BonitoBook.InlineBook(path)
end
Bonito.Page()
```

```@example viewfactors_wrapper
book_app # hide
```
