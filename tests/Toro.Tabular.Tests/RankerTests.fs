module RankerTests

open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro
open Toro.Tabular

let private floats (values: float32 array) =
    torch.tensor (values, dtype = torch.float32, device = torch.CPU)

let private ints (values: int64 array) =
    torch.tensor (values, dtype = torch.int64, device = torch.CPU)

let private rankingTable () =
    let labels = [|
        for group in 0..7 do
            for relevance in 0..4 -> float32 relevance
    |]

    let groups = [|
        for group in 0L .. 7L do
            for _ in 0..4 -> group
    |]

    Table.create [
        "tf", Floats(floats labels)
        "label", Floats(floats labels)
        "qid", Ints(ints groups)
    ]

[<Fact>]
let ``Ranker.predict returns one score per row`` () =
    let table = rankingTable ()
    let ranker = Ranker.fit [ "tf" ] "label" "qid" table
    let scores = Ranker.predict table ranker
    scores.shape |> should equal [| table.Length |]
    scores.dtype |> should equal torch.float32

[<Fact>]
let ``Ranker ranks higher labels above lower labels in a group`` () =
    let table = rankingTable ()
    let ranker = Ranker.fit [ "tf" ] "label" "qid" table
    let scores = Ranker.predict table ranker
    let scoreArr = scores.data<float32>().ToArray()

    for group in 0..7 do
        let offset = group * 5
        let best = [| 0..4 |] |> Array.maxBy (fun i -> scoreArr[offset + i])
        best |> should equal 4
