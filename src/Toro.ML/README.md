# Toro.ML

Shared tensor datasets and ML.NET interop for classical machine learning with [Toro](https://www.nuget.org/packages/Toro).

Algorithm implementations are distributed separately, including `Toro.ML.FastTree` and `Toro.ML.LightGbm`.

## Installation

```bash
dotnet add package Toro.ML
dotnet add package TorchSharp-cpu
```

## Datasets

```fsharp
open TorchSharp
open Toro.ML

let features = torch.zeros ([| 4L; 2L |], dtype = torch.float32)
let labels = torch.zeros ([| 4L |], dtype = torch.float32)
let groupIds = torch.tensor ([| 10L; 10L; 20L; 20L |], dtype = torch.int64)

let dataset = RankingDataset.create features labels groupIds
```

Regression uses the same feature and label layout without query groups.

```fsharp
let dataset = RegressionDataset.create features labels
```

Datasets borrow their tensors. They do not copy or dispose them.

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
