module RankingDatasetTests

open System
open Xunit
open FsUnit.Xunit
open Microsoft.ML
open Microsoft.ML.Data
open TorchSharp
open Toro.ML
open Toro.ML.Interop

let private features rows columns =
    torch.zeros ([| int64 rows; int64 columns |], dtype = torch.float32, device = torch.CPU)

let private labels count =
    torch.zeros ([| int64 count |], dtype = torch.float32, device = torch.CPU)

let private groups (values: int64 array) =
    torch.tensor (values, dtype = torch.int64, device = torch.CPU)

[<Fact>]
let ``create keeps borrowed tensors and dimensions`` () =
    let x = features 4 2
    let y = labels 4
    let qids = groups [| 10L; 10L; 20L; 20L |]
    let dataset = RankingDataset.create x y qids

    dataset.Features |> should be (sameAs x)
    dataset.Labels |> should be (sameAs y)
    dataset.GroupIds |> should be (sameAs qids)
    dataset.RowCount |> should equal 4L
    dataset.FeatureCount |> should equal 2

[<Fact>]
let ``create rejects invalid shapes and dtypes`` () =
    let x = features 4 2
    let y = labels 4
    let qids = groups [| 0L; 0L; 1L; 1L |]

    Assert.Throws<ArgumentException>(fun () ->
        RankingDataset.create (torch.zeros ([| 4L |], dtype = torch.float32)) y qids
        |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () ->
        RankingDataset.create x (torch.zeros ([| 4L |], dtype = torch.float64)) qids
        |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () ->
        RankingDataset.create x y (torch.zeros ([| 4L |], dtype = torch.int32))
        |> ignore)
    |> ignore

[<Fact>]
let ``create rejects empty data and row mismatches`` () =
    Assert.Throws<ArgumentException>(fun () ->
        RankingDataset.create (features 0 2) (labels 0) (groups [||])
        |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () ->
        RankingDataset.create (features 4 2) (labels 3) (groups [| 0L; 0L; 1L; 1L |])
        |> ignore)
    |> ignore

[<Fact>]
let ``create rejects a group split across blocks`` () =
    Assert.Throws<ArgumentException>(fun () ->
        RankingDataset.create (features 4 2) (labels 4) (groups [| 10L; 20L; 10L; 20L |])
        |> ignore)
    |> ignore

[<Fact>]
let ``interop preserves large ids by recoding groups as keys`` () =
    let dataset =
        RankingDataset.create
            (features 4 2)
            (labels 4)
            (groups [| Int64.MaxValue; Int64.MaxValue; Int64.MinValue; Int64.MinValue |])

    let context = Microsoft.ML.MLContext(seed = 0)
    let view = RankingDataView.training context dataset
    let encoded = view.GetColumn<uint32>(Columns.GroupId) |> Seq.toArray

    encoded |> should equal [| 1u; 1u; 2u; 2u |]
