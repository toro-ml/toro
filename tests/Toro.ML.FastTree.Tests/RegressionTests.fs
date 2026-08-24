module FastTreeRegressionTests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro.ML
open Toro.ML.FastTree

let private regressionDataset () =
    let inputs = [| for value in -20 .. 20 -> float32 value / 10.0f |]
    let targets = inputs |> Array.map (fun value -> 3.0f * value + 1.0f)

    let features =
        (torch.tensor (inputs, dtype = torch.float32, device = torch.CPU)).unsqueeze 1L

    let labels = torch.tensor (targets, dtype = torch.float32, device = torch.CPU)
    RegressionDataset.create features labels

let private config () = {
    RegressionConfig.create () with
        NumberOfTrees = 32
        NumberOfLeaves = 8
        MinimumExampleCountPerLeaf = 1
}

[<Fact>]
let ``fit predicts one float32 value per row`` () =
    let dataset = regressionDataset ()
    let model = Regression.fit (config ()) dataset
    let predictions = Regression.predict dataset.Features model

    predictions.shape |> should equal [| dataset.RowCount |]
    predictions.dtype |> should equal torch.float32
    predictions.device_type |> should equal DeviceType.CPU

[<Fact>]
let ``fitWithOptions produces finite regression metrics`` () =
    let dataset = regressionDataset ()

    let model =
        Regression.fitWithOptions
            (Some 0)
            (fun options ->
                options.NumberOfTrees <- 32
                options.NumberOfLeaves <- 8
                options.MinimumExampleCountPerLeaf <- 1)
            dataset

    let metrics = Regression.evaluate dataset model

    Double.IsFinite metrics.MeanSquaredError
    |> should equal true

    metrics.MeanSquaredError |> should be (lessThan 1.0)

[<Fact>]
let ``save and load preserve regression predictions`` () =
    let dataset = regressionDataset ()
    let model = Regression.fit (config ()) dataset

    let path =
        Path.Combine(Path.GetTempPath(), $"toro-fasttree-regression-{Guid.NewGuid():N}.zip")

    try
        Regression.save path model
        let loaded = Regression.load path

        let expected =
            Regression.predict dataset.Features model
            |> fun values -> values.data<float32>().ToArray()

        let actual =
            Regression.predict dataset.Features loaded
            |> fun values -> values.data<float32>().ToArray()

        actual |> should equal expected
    finally
        if File.Exists path then
            File.Delete path

[<Fact>]
let ``regression predict rejects a different feature width`` () =
    let dataset = regressionDataset ()
    let model = Regression.fit (config ()) dataset

    let wrong =
        torch.zeros ([| dataset.RowCount; 2L |], dtype = torch.float32, device = torch.CPU)

    Assert.Throws<ArgumentException>(fun () -> Regression.predict wrong model |> ignore)
    |> ignore
