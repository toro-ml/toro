# Toro.Tabular

Named-column tables and FastTree ranking over [Toro](https://www.nuget.org/packages/Toro) tensors, backed by [ML.NET](https://www.nuget.org/packages/Microsoft.ML).

**[Documentation](https://toro-ml.github.io/toro/)**

## Installation

```bash
dotnet add package Toro.Tabular
dotnet add package TorchSharp-cpu
```

## Quick Example

```fsharp
open TorchSharp
open Toro
open Toro.Tabular

let tf = torch.tensor ([| 1.0f; 0.0f |], dtype = torch.float32, device = torch.CPU)
let labels = torch.tensor ([| 1.0f; 0.0f |], dtype = torch.float32, device = torch.CPU)
let qids = torch.tensor ([| 0L; 0L |], dtype = torch.int64, device = torch.CPU)

let table =
    Table.create [
        "title_tf", Floats tf
        "label", Floats labels
        "qid", Ints qids
    ]

let ranker = Ranker.fit [ "title_tf" ] "label" "qid" table
let scores = Ranker.predict table ranker
```

## Features

- **Table** -- named `Floats` / `Ints` / `Vectors` columns with a shared row count
- **features** -- stack 1-d numeric columns into a `[n, f]` float32 tensor
- **Ranker** -- FastTree ranking; scores are detached Toro tensors

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
