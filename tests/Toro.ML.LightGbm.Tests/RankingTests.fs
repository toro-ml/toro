module LightGbmRankingTests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro.ML

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
    Toro.ML.LightGbm.RankingConfig.create () with
        NumberOfIterations = 16
        NumberOfLeaves = Some 4
        MinimumExampleCountPerLeaf = Some 1
}

[<Fact>]
let ``fit predicts one float32 score per row`` () =
    let dataset = rankingDataset ()
    let model = Toro.ML.LightGbm.Ranking.fit (config ()) dataset
    let scores = Toro.ML.LightGbm.Ranking.predict dataset.Features model

    scores.shape |> should equal [| dataset.RowCount |]
    scores.dtype |> should equal torch.float32
    scores.device_type |> should equal DeviceType.CPU

[<Fact>]
let ``fitWithOptions ranks the highest relevance first`` () =
    let dataset = rankingDataset ()

    let model =
        Toro.ML.LightGbm.Ranking.fitWithOptions
            (Some 0)
            (fun options ->
                options.NumberOfIterations <- 16
                options.NumberOfLeaves <- Nullable 4
                options.MinimumExampleCountPerLeaf <- Nullable 1)
            dataset

    let values =
        Toro.ML.LightGbm.Ranking.predict dataset.Features model
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
    let model = Toro.ML.LightGbm.Ranking.fit (config ()) dataset
    let metrics = Toro.ML.LightGbm.Ranking.evaluate dataset model

    metrics.NormalizedDiscountedCumulativeGains.Count
    |> should be (greaterThan 0)

    metrics.NormalizedDiscountedCumulativeGains
    |> Seq.iter (fun value -> Double.IsFinite value |> should equal true)

[<Fact>]
let ``save and load preserve predictions`` () =
    let dataset = rankingDataset ()
    let model = Toro.ML.LightGbm.Ranking.fit (config ()) dataset
    let path = Path.Combine(Path.GetTempPath(), $"toro-lightgbm-{Guid.NewGuid():N}.zip")

    try
        Toro.ML.LightGbm.Ranking.save path model
        let loaded = Toro.ML.LightGbm.Ranking.load path
        loaded.FeatureCount |> should equal dataset.FeatureCount

        let expected =
            Toro.ML.LightGbm.Ranking.predict dataset.Features model
            |> fun scores -> scores.data<float32>().ToArray()

        let actual =
            Toro.ML.LightGbm.Ranking.predict dataset.Features loaded
            |> fun scores -> scores.data<float32>().ToArray()

        actual |> should equal expected
    finally
        if File.Exists path then
            File.Delete path

[<Fact>]
let ``load rejects a FastTree model`` () =
    let dataset = rankingDataset ()

    let fastTreeConfig = {
        Toro.ML.FastTree.RankingConfig.create () with
            NumberOfTrees = 8
            NumberOfLeaves = 4
            MinimumExampleCountPerLeaf = 1
    }

    let fastTreeModel = Toro.ML.FastTree.Ranking.fit fastTreeConfig dataset
    let path = Path.Combine(Path.GetTempPath(), $"toro-fasttree-{Guid.NewGuid():N}.zip")

    try
        Toro.ML.FastTree.Ranking.save path fastTreeModel

        Assert.Throws<InvalidDataException>(fun () -> Toro.ML.LightGbm.Ranking.load path |> ignore)
        |> ignore
    finally
        if File.Exists path then
            File.Delete path

[<Fact>]
let ``predict rejects a different feature width`` () =
    let dataset = rankingDataset ()
    let model = Toro.ML.LightGbm.Ranking.fit (config ()) dataset

    let wrong =
        torch.zeros ([| dataset.RowCount; 2L |], dtype = torch.float32, device = torch.CPU)

    Assert.Throws<ArgumentException>(fun () -> Toro.ML.LightGbm.Ranking.predict wrong model |> ignore)
    |> ignore
