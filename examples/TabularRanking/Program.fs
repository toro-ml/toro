open TorchSharp
open Toro
open Toro.ML

// Rank documents inside each query group. The feature equals the relevance
// label, so both algorithms should put the highest-relevance document first.

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

    let features =
        (torch.tensor (labels, dtype = torch.float32, device = torch.CPU)).unsqueeze 1L

    let label = torch.tensor (labels, dtype = torch.float32, device = torch.CPU)
    let qid = torch.tensor (qids, dtype = torch.int64, device = torch.CPU)
    let dataset = RankingDataset.create features label qid

    let fastTreeConfig = {
        Toro.ML.FastTree.RankingConfig.create () with
            NumberOfTrees = 16
            NumberOfLeaves = 4
            MinimumExampleCountPerLeaf = 1
    }

    let lightGbmConfig = {
        Toro.ML.LightGbm.RankingConfig.create () with
            NumberOfIterations = 16
            NumberOfLeaves = Some 4
            MinimumExampleCountPerLeaf = Some 1
    }

    let fastTree = Toro.ML.FastTree.Ranking.fit fastTreeConfig dataset
    let lightGbm = Toro.ML.LightGbm.Ranking.fit lightGbmConfig dataset

    let show (name: string) (scores: Tensor) (ndcg: seq<float>) =
        let scoreArr = scores.data<float32>().ToArray()
        printfn "%s (NDCG@1 = %.4f)" name (ndcg |> Seq.head)

        for q in 0 .. queryCount - 1 do
            let offset = q * docsPerQuery

            let ranked =
                [| 0 .. docsPerQuery - 1 |]
                |> Array.sortByDescending (fun i -> scoreArr[offset + i])

            printfn "query %d" q

            ranked
            |> Array.iteri (fun rank i -> printfn "  #%d  relevance=%d  score=%.4f" (rank + 1) i scoreArr[offset + i])

        printfn ""

    printfn "Tabular ranking: %d queries, %d documents each" queryCount docsPerQuery
    printfn "dataset rows = %d, feature shape = %A" dataset.RowCount dataset.Features.shape
    printfn ""

    show
        "FastTree"
        (Toro.ML.FastTree.Ranking.predict dataset.Features fastTree)
        (Toro.ML.FastTree.Ranking.evaluate dataset fastTree).NormalizedDiscountedCumulativeGains

    show
        "LightGBM"
        (Toro.ML.LightGbm.Ranking.predict dataset.Features lightGbm)
        (Toro.ML.LightGbm.Ranking.evaluate dataset lightGbm).NormalizedDiscountedCumulativeGains

    0
