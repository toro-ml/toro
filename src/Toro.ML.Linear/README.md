# Toro.ML.Linear

Linear classical machine-learning algorithms for [Toro](https://www.nuget.org/packages/Toro) tensors, backed by the standard trainers included with ML.NET.

The first implementation provides SDCA regression. Other linear trainers can be added as sibling namespaces without creating one NuGet package per trainer.

## SDCA regression

```fsharp
open Toro.ML
open Toro.ML.Linear.Sdca

let dataset = RegressionDataset.create features labels
let model = Regression.fit (RegressionConfig.create ()) dataset
let values = Regression.predict features model
let metrics = Regression.evaluate dataset model
```

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
