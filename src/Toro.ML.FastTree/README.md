# Toro.ML.FastTree

FastTree learning-to-rank over [Toro](https://www.nuget.org/packages/Toro) tensors, backed by ML.NET.

## Installation

```bash
dotnet add package Toro.ML.FastTree
dotnet add package TorchSharp-cpu
```

## Quick example

```fsharp
open Toro.ML
open Toro.ML.FastTree

let dataset = RankingDataset.create features labels groupIds
let model = Ranking.fit (RankingConfig.create ()) dataset
let scores = Ranking.predict features model
let metrics = Ranking.evaluate dataset model
```

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
