module RegressionDatasetTests

open System
open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro.ML

let private features rows columns =
    torch.zeros ([| int64 rows; int64 columns |], dtype = torch.float32, device = torch.CPU)

let private labels count =
    torch.zeros ([| int64 count |], dtype = torch.float32, device = torch.CPU)

[<Fact>]
let ``create keeps borrowed regression tensors and dimensions`` () =
    let x = features 4 2
    let y = labels 4
    let dataset = RegressionDataset.create x y

    dataset.Features |> should be (sameAs x)
    dataset.Labels |> should be (sameAs y)
    dataset.RowCount |> should equal 4L
    dataset.FeatureCount |> should equal 2

[<Fact>]
let ``create rejects invalid regression data`` () =
    Assert.Throws<ArgumentException>(fun () -> RegressionDataset.create (features 0 2) (labels 0) |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () -> RegressionDataset.create (features 4 2) (labels 3) |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () ->
        RegressionDataset.create (features 4 2) (torch.zeros ([| 4L |], dtype = torch.float64))
        |> ignore)
    |> ignore
