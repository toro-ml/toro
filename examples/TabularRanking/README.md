# TabularRanking

Fit a FastTree ranker on a named-column `Table` and score each row. The `tf` feature equals the relevance label, so the highest-relevance document should rank first in each query group.

## Run

```bash
dotnet run --project examples/TabularRanking
```

`dotnet run` clears `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` so a Python PyTorch venv does not shadow TorchSharp-cpu.

## Output

```
Tabular ranking: 4 queries, 5 documents each
table rows = 20, feature shape = [|20L; 1L|]

query 0
  #1  relevance=4  score=...
  #2  relevance=3  score=...
  ...

query 1
  ...
```

## Concepts

- `Table.create` with `Floats` / `Ints` columns
- `Table.features` stacking 1-d columns into a `[n, f]` tensor
- `Ranker.fit` / `Ranker.predict` (ML.NET FastTree under the table API)
