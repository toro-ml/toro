open TorchSharp
open Toro
open Toro.Tabular

// Rank documents inside each query group. The tf feature equals the relevance
// label, so FastTree should put the highest-relevance document first.

let queryCount = 4
let docsPerQuery = 5

[<EntryPoint>]
let main _argv =
    let labels = [|
        for _ in 1..queryCount do
            for relevance in 0 .. docsPerQuery - 1 -> float32 relevance
    |]

    let qids = [|
        for qid in 0L .. int64 queryCount - 1L do
            for _ in 1..docsPerQuery -> qid
    |]

    let tf = torch.tensor (labels, dtype = torch.float32, device = torch.CPU)
    let label = torch.tensor (labels, dtype = torch.float32, device = torch.CPU)
    let qid = torch.tensor (qids, dtype = torch.int64, device = torch.CPU)

    let table = Table.create [ "tf", Floats tf; "label", Floats label; "qid", Ints qid ]

    let features = Table.features [ "tf" ] table
    let ranker = Ranker.fit [ "tf" ] "label" "qid" table
    let scores = Ranker.predict table ranker
    let scoreArr = scores.data<float32>().ToArray()

    printfn "Tabular ranking: %d queries, %d documents each" queryCount docsPerQuery
    printfn "table rows = %d, feature shape = %A" table.Length features.shape
    printfn ""

    for q in 0 .. queryCount - 1 do
        let offset = q * docsPerQuery

        let ranked =
            [| 0 .. docsPerQuery - 1 |]
            |> Array.sortByDescending (fun i -> scoreArr[offset + i])

        printfn "query %d" q

        ranked
        |> Array.iteri (fun rank i -> printfn "  #%d  relevance=%d  score=%.4f" (rank + 1) i scoreArr[offset + i])

        printfn ""

    0
