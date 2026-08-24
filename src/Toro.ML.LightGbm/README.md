# Toro.ML.LightGbm

LightGBM algorithms for tensors in [Toro](https://github.com/toro-ml/toro), backed by ML.NET.

The first release provides learning-to-rank through `Toro.ML.LightGbm.Ranking`.

## Installation

```bash
dotnet add package Toro.ML.LightGbm
dotnet add package TorchSharp-cpu
```

## Quick example

```fsharp
open Toro.ML.LightGbm

let config = RankingConfig.create ()
let model = Ranking.fit config dataset
let scores = Ranking.predict dataset.Features model
```

`NumberOfLeaves`, `MinimumExampleCountPerLeaf`, and `LearningRate` are optional in the common config because ML.NET auto-tunes them from the training data when they are `None`.

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
