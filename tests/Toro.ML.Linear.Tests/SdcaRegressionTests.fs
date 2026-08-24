module SdcaRegressionTests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro.ML
open Toro.ML.Linear.Sdca

let private regressionDataset () =
    let inputs = [| for value in -20 .. 20 -> float32 value / 10.0f |]
    let targets = inputs |> Array.map (fun value -> 3.0f * value + 1.0f)

    let features =
        (torch.tensor (inputs, dtype = torch.float32, device = torch.CPU)).unsqueeze 1L

    let labels = torch.tensor (targets, dtype = torch.float32, device = torch.CPU)
    RegressionDataset.create features labels

let private config () = {
    RegressionConfig.create () with
        MaximumNumberOfIterations = Some 100
}

[<Fact>]
let ``SDCA fit predicts one float32 value per row`` () =
    let dataset = regressionDataset ()
    let model = Regression.fit (config ()) dataset
    let predictions = Regression.predict dataset.Features model

    predictions.shape |> should equal [| dataset.RowCount |]
    predictions.dtype |> should equal torch.float32
    predictions.device_type |> should equal DeviceType.CPU

[<Fact>]
let ``SDCA fitWithOptions produces finite regression metrics`` () =
    let dataset = regressionDataset ()

    let model =
        Regression.fitWithOptions (Some 0) (fun options -> options.MaximumNumberOfIterations <- Nullable 100) dataset

    let metrics = Regression.evaluate dataset model

    Double.IsFinite metrics.MeanSquaredError
    |> should equal true

    metrics.MeanSquaredError |> should be (lessThan 0.1)

[<Fact>]
let ``SDCA save and load preserve predictions`` () =
    let dataset = regressionDataset ()
    let model = Regression.fit (config ()) dataset

    let path =
        Path.Combine(Path.GetTempPath(), $"toro-sdca-regression-{Guid.NewGuid():N}.zip")

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
let ``SDCA predict rejects a different feature width`` () =
    let dataset = regressionDataset ()
    let model = Regression.fit (config ()) dataset

    let wrong =
        torch.zeros ([| dataset.RowCount; 2L |], dtype = torch.float32, device = torch.CPU)

    Assert.Throws<ArgumentException>(fun () -> Regression.predict wrong model |> ignore)
    |> ignore
