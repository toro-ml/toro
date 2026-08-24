module LightGbmRegressionTests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro.ML

let private regressionDataset () =
    let inputs = [| for value in -20 .. 20 -> float32 value / 10.0f |]
    let targets = inputs |> Array.map (fun value -> 3.0f * value + 1.0f)

    let features =
        (torch.tensor (inputs, dtype = torch.float32, device = torch.CPU)).unsqueeze 1L

    let labels = torch.tensor (targets, dtype = torch.float32, device = torch.CPU)
    RegressionDataset.create features labels

let private config () = {
    Toro.ML.LightGbm.RegressionConfig.create () with
        NumberOfIterations = 32
        NumberOfLeaves = Some 8
        MinimumExampleCountPerLeaf = Some 1
}

[<Fact>]
let ``fit predicts one float32 regression value per row`` () =
    let dataset = regressionDataset ()
    let model = Toro.ML.LightGbm.Regression.fit (config ()) dataset
    let predictions = Toro.ML.LightGbm.Regression.predict dataset.Features model

    predictions.shape |> should equal [| dataset.RowCount |]
    predictions.dtype |> should equal torch.float32
    predictions.device_type |> should equal DeviceType.CPU

[<Fact>]
let ``fitWithOptions produces finite LightGBM regression metrics`` () =
    let dataset = regressionDataset ()

    let model =
        Toro.ML.LightGbm.Regression.fitWithOptions
            (Some 0)
            (fun options ->
                options.NumberOfIterations <- 32
                options.NumberOfLeaves <- Nullable 8
                options.MinimumExampleCountPerLeaf <- Nullable 1)
            dataset

    let metrics = Toro.ML.LightGbm.Regression.evaluate dataset model

    Double.IsFinite metrics.MeanSquaredError
    |> should equal true

    metrics.MeanSquaredError |> should be (lessThan 1.0)

[<Fact>]
let ``save and load preserve LightGBM regression predictions`` () =
    let dataset = regressionDataset ()
    let model = Toro.ML.LightGbm.Regression.fit (config ()) dataset

    let path =
        Path.Combine(Path.GetTempPath(), $"toro-lightgbm-regression-{Guid.NewGuid():N}.zip")

    try
        Toro.ML.LightGbm.Regression.save path model
        let loaded = Toro.ML.LightGbm.Regression.load path

        let expected =
            Toro.ML.LightGbm.Regression.predict dataset.Features model
            |> fun values -> values.data<float32>().ToArray()

        let actual =
            Toro.ML.LightGbm.Regression.predict dataset.Features loaded
            |> fun values -> values.data<float32>().ToArray()

        actual |> should equal expected
    finally
        if File.Exists path then
            File.Delete path

[<Fact>]
let ``LightGBM regression load rejects a FastTree regression model`` () =
    let dataset = regressionDataset ()

    let fastTreeConfig = {
        Toro.ML.FastTree.RegressionConfig.create () with
            NumberOfTrees = 16
            NumberOfLeaves = 8
            MinimumExampleCountPerLeaf = 1
    }

    let fastTreeModel = Toro.ML.FastTree.Regression.fit fastTreeConfig dataset

    let path =
        Path.Combine(Path.GetTempPath(), $"toro-fasttree-regression-{Guid.NewGuid():N}.zip")

    try
        Toro.ML.FastTree.Regression.save path fastTreeModel

        Assert.Throws<InvalidDataException>(fun () -> Toro.ML.LightGbm.Regression.load path |> ignore)
        |> ignore
    finally
        if File.Exists path then
            File.Delete path

[<Fact>]
let ``LightGBM regression predict rejects a different feature width`` () =
    let dataset = regressionDataset ()
    let model = Toro.ML.LightGbm.Regression.fit (config ()) dataset

    let wrong =
        torch.zeros ([| dataset.RowCount; 2L |], dtype = torch.float32, device = torch.CPU)

    Assert.Throws<ArgumentException>(fun () -> Toro.ML.LightGbm.Regression.predict wrong model |> ignore)
    |> ignore
