# Toro.ML.LightGbm

LightGBM algorithms for tensors in [Toro](https://github.com/toro-ml/toro), backed by ML.NET.

The package provides regression and learning-to-rank through task-specific modules.

## Installation

```bash
dotnet add package Toro.ML.LightGbm
dotnet add package TorchSharp-cpu
```

## Regression

```fsharp
open Toro.ML
open Toro.ML.LightGbm

let dataset = RegressionDataset.create features labels
let model = Regression.fit (RegressionConfig.create ()) dataset
let values = Regression.predict features model
let metrics = Regression.evaluate dataset model
```

## Ranking

```fsharp
open Toro.ML.LightGbm

let config = RankingConfig.create ()
let model = Ranking.fit config dataset
let scores = Ranking.predict dataset.Features model
```

`NumberOfLeaves`, `MinimumExampleCountPerLeaf`, and `LearningRate` are optional in each task config because ML.NET auto-tunes them from the training data when they are `None`.

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
