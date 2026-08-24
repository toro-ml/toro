# TabularRanking

Fit FastTree and LightGBM rankers on one tensor-backed `RankingDataset`, evaluate them with NDCG, and score each row. The feature equals the relevance label, so the highest-relevance document should rank first in each query group.

## Run

```bash
dotnet run --project examples/TabularRanking
```

`dotnet run` clears `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` so a Python PyTorch venv does not shadow TorchSharp-cpu.

## Output

```
Tabular ranking: 4 queries, 5 documents each
dataset rows = 20, feature shape = [|20L; 1L|]

FastTree (NDCG@1 = ...)
query 0
  #1  relevance=4  score=...
  #2  relevance=3  score=...
  ...

query 1
  ...

LightGBM (NDCG@1 = ...)
query 0
  ...
```

## Concepts

- `RankingDataset.create` over feature, label, and query-group tensors
- Independent `Toro.ML.FastTree.Ranking` and `Toro.ML.LightGbm.Ranking` APIs
- ML.NET ranking metrics returned by each package's `evaluate` function
