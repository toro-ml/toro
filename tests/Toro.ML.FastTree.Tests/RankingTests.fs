module FastTreeRankingTests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro.ML
open Toro.ML.FastTree

let private rankingDataset () =
    let labels = [|
        for _group in 0..7 do
            for relevance in 0..4 -> float32 relevance
    |]

    let groups = [|
        for group in 0L .. 7L do
            for _ in 0..4 -> group
    |]

    let features =
        (torch.tensor (labels, dtype = torch.float32, device = torch.CPU)).unsqueeze 1L

    let labels = torch.tensor (labels, dtype = torch.float32, device = torch.CPU)
    let groups = torch.tensor (groups, dtype = torch.int64, device = torch.CPU)
    RankingDataset.create features labels groups

let private config () = {
    RankingConfig.create () with
        NumberOfTrees = 8
        NumberOfLeaves = 4
        MinimumExampleCountPerLeaf = 1
}

[<Fact>]
let ``fit predicts one float32 score per row`` () =
    let dataset = rankingDataset ()
    let model = Ranking.fit (config ()) dataset
    let scores = Ranking.predict dataset.Features model

    scores.shape |> should equal [| dataset.RowCount |]
    scores.dtype |> should equal torch.float32
    scores.device_type |> should equal DeviceType.CPU

[<Fact>]
let ``fitWithOptions ranks the highest relevance first`` () =
    let dataset = rankingDataset ()

    let model =
        Ranking.fitWithOptions
            (Some 0)
            (fun options ->
                options.NumberOfTrees <- 8
                options.NumberOfLeaves <- 4
                options.MinimumExampleCountPerLeaf <- 1)
            dataset

    let values =
        Ranking.predict dataset.Features model
        |> fun scores -> scores.data<float32>().ToArray()

    for group in 0..7 do
        let offset = group * 5

        let best =
            [| 0..4 |]
            |> Array.maxBy (fun index -> values[offset + index])

        best |> should equal 4

[<Fact>]
let ``evaluate returns finite NDCG values`` () =
    let dataset = rankingDataset ()
    let model = Ranking.fit (config ()) dataset
    let metrics = Ranking.evaluate dataset model

    metrics.NormalizedDiscountedCumulativeGains.Count
    |> should be (greaterThan 0)

    metrics.NormalizedDiscountedCumulativeGains
    |> Seq.iter (fun value -> Double.IsFinite value |> should equal true)

[<Fact>]
let ``save and load preserve predictions`` () =
    let dataset = rankingDataset ()
    let model = Ranking.fit (config ()) dataset
    let path = Path.Combine(Path.GetTempPath(), $"toro-fasttree-{Guid.NewGuid():N}.zip")

    try
        Ranking.save path model
        let loaded = Ranking.load path
        loaded.FeatureCount |> should equal dataset.FeatureCount

        let expected =
            Ranking.predict dataset.Features model
            |> fun scores -> scores.data<float32>().ToArray()

        let actual =
            Ranking.predict dataset.Features loaded
            |> fun scores -> scores.data<float32>().ToArray()

        actual |> should equal expected
    finally
        if File.Exists path then
            File.Delete path

[<Fact>]
let ``predict rejects a different feature width`` () =
    let dataset = rankingDataset ()
    let model = Ranking.fit (config ()) dataset

    let wrong =
        torch.zeros ([| dataset.RowCount; 2L |], dtype = torch.float32, device = torch.CPU)

    Assert.Throws<ArgumentException>(fun () -> Ranking.predict wrong model |> ignore)
    |> ignore
